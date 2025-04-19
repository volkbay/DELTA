#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file can be used to train or finetune the ALED & DELTA networks, on either the SLED, the MVSEC,
or the M3ED datasets, as described in our "DELTA: Dense Depth from Events and LiDAR using
Transformer's Attention" article (CVPRW 2025).
Note: due to the use of parallelism during training, this script should not be launched directly,
but should be handled by torchrun. See the README for more details.
"""

import argparse
from datetime import datetime
import json
import os
from time import sleep

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm

from dataset.loaders.preprocessed_dataset_loader import PreprocessedDataset
from losses.losses import L1MSGLoss
from models.aled import ALED
from models.delta import DELTA
from transforms.transforms import PadToMaxSize, RandomCropAlignedWithPatches
from trainer_tester.trainer import Trainer


def parse_args():
  """Args parser"""
  parser = argparse.ArgumentParser()
  parser.add_argument("config_file", help="Path to the JSON config file to use for training")
  parser.add_argument("--cp", default=None, help="Checkpoint to restart from (optional)")
  return parser.parse_args()


def display_count_parameters(model: nn.Module) -> int:
  """
  Utility function to count and display the number of parameters of a network in PyTorch.
  Thanks to https://stackoverflow.com/a/62508086
  """
  total_params = 0
  for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
      continue
    params = parameter.numel()
    print(name, ":", params)
    total_params += params
  print(f"Total Trainable Params: {total_params}")
  return total_params


def main():
  """
  Main function, used for training and validating the network.
  """

  # We start by initializing the process group, as required by torchrun
  dist.init_process_group(backend="nccl")

  # We collect the local GPU ID assigned to this process, and use it to determine the name of the
  # device to use
  gpu_id = int(os.environ["LOCAL_RANK"])
  device = f"cuda:{gpu_id}"

  # We also collect the total number of GPUs
  nb_gpus = int(os.environ["LOCAL_WORLD_SIZE"])

  # We load the config file given by the user
  args = parse_args()
  with open(args.config_file, encoding="utf-8") as cfg_file:
    config = json.load(cfg_file)

  # We also configure the Tensorboard summary writer
  # Note: it is only initialized for GPU ID 0, so that not all the processes write to it
  if args.cp is not None:
    time_prefix = os.path.split(args.cp)[-1][:15]
    start_epoch = os.path.split(args.cp)[-1][16:19]
    start_epoch = int(start_epoch) + 1
  else:
    time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_epoch = 0

  if gpu_id == 0:
    if not os.path.isdir("out/runs"):
      os.mkdir("out/runs")
    writer = SummaryWriter(os.path.join("out/runs", time_prefix))
  else:
    writer = None

  # We collect the patch size (only for attention-based models)
  if config["model"] == "DELTA":
    patch_size = config["patch_size"]
  else:
    patch_size = None

  # We setup the transforms we will perform on the training dataset, i.e. padding and random
  # cropping (if required)
  train_transforms_list = []
  if config["transforms"]["pad"]["pad_input"]:
    padded_img_size_x = config["transforms"]["pad"]["padded_image_size_x"]
    padded_img_size_y = config["transforms"]["pad"]["padded_image_size_y"]
    train_transforms_list.append(PadToMaxSize((padded_img_size_y, padded_img_size_x)))
  if config["transforms"]["crop"]["crop_input"]:
    crop_size = config["transforms"]["crop"]["crop_size"]
    train_transforms_list.append(RandomCropAlignedWithPatches(crop_size, patch_size))
  train_transforms = Compose(train_transforms_list)

  # We setup the transforms we will perform on the validation dataset, i.e. padding (if required)
  if config["transforms"]["pad"]["pad_input"]:
    val_transforms = Compose([PadToMaxSize((padded_img_size_y, padded_img_size_x))])
  else:
    val_transforms = None

  # We collect the batch_size and num_workers parameters from the config file
  batch_size_train = config["batch_size_train"]
  batch_size_train_per_gpu = batch_size_train // nb_gpus
  num_workers = config["num_workers"]

  # For the validation, we constrain the batch_size to a value of 1
  batch_size_val = 1

  # We collect whether the datasets are compressed or not
  train_is_zipped = config["dataset"]["train_is_zipped"]
  val_is_zipped = config["dataset"]["val_is_zipped"]

  # We load the rules to only keep some elements of the training/validation set (because the
  # validation set can be quite large in the case of SLED for instance, and so it might be helpful
  # to only use some of the sequences instead of all of them)
  # If the rule is not set (""), we set it to "True" to accept all sequences
  train_subset_rule = config["dataset"]["train_subset_rule"]
  val_subset_rule = config["dataset"]["val_subset_rule"]

  # We load the training dataset, create the sampler (for parallelism), create the dataloader, and
  # collect the number of sequences that were loaded
  # Note: we have to set shuffle of the dataloader to False, as the sampler handles the shuffling
  train_dataset_path = config["dataset"]["path_train"]
  train_dataset = PreprocessedDataset(train_dataset_path, train_is_zipped, train_subset_rule,
                                      train_transforms)
  train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
  train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size_train_per_gpu,
                                shuffle=False, sampler=train_sampler, num_workers=num_workers,
                                persistent_workers=num_workers>0, pin_memory=True)

  # We do the same for the validation dataset
  # Note: since the validation is only done on the first GPU, we only create the dataloader for the
  # process using this GPU
  if gpu_id == 0:
    val_dataset_path = config["dataset"]["path_val"]
    val_dataset = PreprocessedDataset(val_dataset_path, val_is_zipped, val_subset_rule,
                                      val_transforms)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size_val, shuffle=False,
                                num_workers=num_workers, persistent_workers=num_workers>0,
                                pin_memory=True)
  else:
    val_dataloader = None

  # We reset the GPU memory cache, to avoid using unneccessary memory
  # See https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113/2
  torch.cuda.set_device(gpu_id)
  torch.cuda.empty_cache()

  # We determine the number of output channels for the model
  out_channels = 2 if config["predict_af_depths"] else 1

  # We initialize the network (based on the model selected in the config file)
  # For some reason, ALED performs very badly on M3ED with 10-channel event data, so we use the
  # 4-channel format used in DELTA instead for this dataset
  # The use of DistributedDataParallel allows for the use of multiple GPUs
  # As some parameters may not used for DELTA with M3ED or MVSEC, we also need to enable
  # find_unused_parameters=True in that specific case
  if config["model"] == "ALED":
    if config["dataset"]["name"] == "M3ED":
      model = ALED(1, 4, out_channels)
    else:
      model = ALED(1, 10, out_channels)
  elif config["model"] == "DELTA":
    model = DELTA(1, 4, out_channels, 2, patch_size, 1024, 4096, 4, 128)
  else:
    raise NotImplementedError(f"Model {config['model']} is not implemented")
  model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
  model.to(device)
  if config["model"] == "DELTA" and config["dataset"]["name"] in ("M3ED", "MVSEC"):
    model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
  else:
    model = DistributedDataParallel(model, device_ids=[gpu_id])

  # We display its number of parameters (uncomment if needed)
  #if gpu_id == 0:
  #  display_count_parameters(model)

  # We set the number of epochs
  num_epochs = config["epochs"]

  # We create the lambda function for the scheduler (for going from the initial_lr to final_lr)
  # This is only used if requested in the configuration, otherwise we set the function to always
  # return 1.0 (i.e., the lr is always initial_lr)
  initial_learning_rate = config["initial_learning_rate"]
  final_learning_rate = config["final_learning_rate"]
  if final_learning_rate != initial_learning_rate:
    lr_lambda = lambda epoch: (final_learning_rate/initial_learning_rate)**(epoch/(num_epochs-1))
  else:
    lr_lambda = lambda _: 1.0

  # We initialize the loss criterion, Adam optimizer, and the scheduler
  criterion = L1MSGLoss(5)
  optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

  # If the starting epoch is not 0, we must run the scheduler the adequate number of times to make
  # sure that the learning rate is correctly set (for more details, see
  # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822/2).
  # Also, PyTorch will complain, as in normal use optimizer.step() should always be called before
  # scheduler.step(), so we also print a message to make sure that the user does not panic :)
  if start_epoch != 0:
    print("!!PyTorch is going to complain, do not worry, the warning does not apply in our case!!")
    sleep(1)
    for _ in range(start_epoch):
      scheduler.step()

  # We initialize the trainer, which is a wrapper class for the training and validation of the model
  trainer = Trainer(model, train_dataloader, val_dataloader, criterion, optimizer, writer, config)

  # If a restart checkpoint is given, we load it
  if args.cp is not None:
    trainer.load_model_checkpoint(args.cp)

  # Then, for each epoch
  for epoch in tqdm(range(start_epoch, num_epochs), "Epochs", disable=gpu_id!=0):
    # We have to configure the data sampler at the beginning of each epoch to ensure that the
    # shuffling is correctly done
    # See https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    train_sampler.set_epoch(epoch)

    # We run the training for the current epoch
    trainer.train(epoch)

    # We must place a barrier here, to make sure all GPUs are in sync before going any further
    # (especially since only the first GPU goes through the validation stage)
    dist.barrier()

    # At the end of the epoch, we run a short evaluation on the test dataset, to monitor the
    # progress of the training
    # We also don't forget to save the model
    # Both these operations are only done on the first GPU
    if gpu_id == 0:
      trainer.val(epoch)
      trainer.save_model_checkpoint(time_prefix, epoch)

    # And to update the learning rate scheduler
    scheduler.step()

    # We must place another barrier here, to make sure all GPUs are in sync before going any further
    # (especially since only the first GPU goes through the validation stage)
    dist.barrier()

  # Once finished, we destroy the process group, as required by torchrun
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
