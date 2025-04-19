#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the PyTorch code for the ALED network, as originally described in the "Learning
to Estimate Two Dense Depths from LiDAR and Event Data" article (SCIA 2023).
"""

import torch
from torch import nn, Tensor

from models.submodules.aled_submodules import ConvEncodingHead, Decoder
from models.submodules.shared_submodules import ConvGRU, ResidualBasicEncoder


class ALED(nn.Module):
  """
  The ALED network, as described in the article.
  It is composed of 2 branches, one for the projected LiDAR clouds and one for the events.
  It uses convolutions for encoding/decoding, and Convolutional Gated Recurrent Units (ConvGRUs) for
  fusion, memory, and asynchronicity purposes.
  """

  def __init__(self, lidar_channels: int, event_channels: int, out_channels: int):
    super().__init__()

    # The encoding heads for the LiDAR, RGB, and event inputs
    self.lidar_head = ConvEncodingHead(lidar_channels, 32, 5, 1, 2)
    self.event_head = ConvEncodingHead(event_channels, 32, 5, 1, 2)

    # The 3 LiDAR encoders
    self.lidar_encoder1 = ResidualBasicEncoder(32, 64, 5, 2, 2, True)
    self.lidar_encoder2 = ResidualBasicEncoder(64, 128, 5, 2, 2, True)
    self.lidar_encoder3 = ResidualBasicEncoder(128, 256, 5, 2, 2, False)

    # The 3 event encoders
    self.event_encoder1 = ResidualBasicEncoder(32, 64, 5, 2, 2, True)
    self.event_encoder2 = ResidualBasicEncoder(64, 128, 5, 2, 2, True)
    self.event_encoder3 = ResidualBasicEncoder(128, 256, 5, 2, 2, False)

    # The 4 convGRU blocks for the LiDAR
    self.conv_gru_lidar0 = ConvGRU(32, 32+32, 3)
    self.conv_gru_lidar1 = ConvGRU(64, 64+64, 3)
    self.conv_gru_lidar2 = ConvGRU(128, 128+128, 3)
    self.conv_gru_lidar3 = ConvGRU(256, 256, 3)

    # The 4 convGRU blocks for the events
    self.conv_gru_events0 = ConvGRU(32, 32+32, 3)
    self.conv_gru_events1 = ConvGRU(64, 64+64, 3)
    self.conv_gru_events2 = ConvGRU(128, 128+128, 3)
    self.conv_gru_events3 = ConvGRU(256, 256, 3)

    # The 2 residual blocks
    self.residual_block1 = ResidualBasicEncoder(256, 256, 3, 1, 1, False)
    self.residual_block2 = ResidualBasicEncoder(256, 256, 3, 1, 1, False)

    # The 3 decoders
    self.decoder1 = Decoder(256, 128, 128, 2, 5, 1, 2)
    self.decoder2 = Decoder(128, 64, 64, 2, 5, 1, 2)
    self.decoder3 = Decoder(64, 32, 32, 2, 5, 1, 2)

    # The 3 convolutions used to reduce the number of channels after concatenating the decoded state
    # and the hidden state of the corresponding convGRU module
    self.conv_concat1 = nn.Conv2d(256, 128, 1)
    self.conv_concat2 = nn.Conv2d(128, 64, 1)
    self.conv_concat3 = nn.Conv2d(64, 32, 1)

    # The final prediction layer
    self.prediction_layer = nn.Conv2d(32, out_channels, 1)


  def forward(self, lidar_input: Tensor | None, event_input: Tensor | None,
              central_mems: list[Tensor] | None) -> tuple[Tensor, list[Tensor]]:
    """
    The shape of the LiDAR/event inputs should be (B, C, H, W), where B is the number of batches, C
    the number of channels, and H and W the height and width. If at a given time, an input is not
    available, its value should be set to None. Multiple inputs can be set to None, but at least one
    input should have a value to update the memories / predict the new depth map.
    The central memories should be None if they are not yet initialized, otherwise they should be an
    array of 4 Tensors, each of shape (B, C_H, H_S, W_S) where C_H is the number of channels of the
    hidden state, and H_S and W_S are the spatial size after applying the convolutions.
    """

    # PART 1: VERIFYING IF THE GIVEN MEMORY IS INITIALIZED, OTHERWISE INITIALIZE IT
    # We initialize the central memories if necessary, as a list of Nones
    if central_mems is None:
      central_mems = [None, None, None, None]

    # PART 2: ENCODING THE LIDAR DATA AND UPDATING THE MEMORIES (IF AVAILABLE)
    if lidar_input is not None:
      # We first apply the head, to go from M layers to 32, and give the result to the top level
      # convGRU to update its state
      encoded_lidar = self.lidar_head(lidar_input)
      central_mems[0] = self.conv_gru_lidar0(encoded_lidar, central_mems[0])

      # We apply the first encoder and give it to the convGRU to update its state
      encoded_lidar = self.lidar_encoder1(encoded_lidar)
      central_mems[1] = self.conv_gru_lidar1(encoded_lidar, central_mems[1])

      # We apply the second encoder and give it to the convGRU to update its state
      encoded_lidar = self.lidar_encoder2(encoded_lidar)
      central_mems[2] = self.conv_gru_lidar2(encoded_lidar, central_mems[2])

      # We apply the third encoder and give it to the convGRU to update its state
      encoded_lidar = self.lidar_encoder3(encoded_lidar)
      central_mems[3] = self.conv_gru_lidar3(encoded_lidar, central_mems[3])

    # PART 3: ENCODING THE EVENT DATA AND UPDATING THE MEMORIES (IF AVAILABLE)
    if event_input is not None:
      # We first apply the head, to go from M layers to 32, and give the result to the top level
      # convGRU to update its state
      encoded_event = self.event_head(event_input)
      central_mems[0] = self.conv_gru_events0(encoded_event, central_mems[0])

      # We apply the first encoder and give it to the convGRU to update its state
      encoded_event = self.event_encoder1(encoded_event)
      central_mems[1] = self.conv_gru_events1(encoded_event, central_mems[1])

      # We apply the second encoder and give it to the convGRU to update its state
      encoded_event = self.event_encoder2(encoded_event)
      central_mems[2] = self.conv_gru_events2(encoded_event, central_mems[2])

      # We apply the third encoder and give it to the convGRU to update its state
      encoded_event = self.event_encoder3(encoded_event)
      central_mems[3] = self.conv_gru_events3(encoded_event, central_mems[3])

    # PART 4: DECODING THE MEMORIES TO PREDICT THE FINAL DEPTH VALUES
    # The initial input for the decoding is the fourth and last central memory, on which we apply
    # the two residual blocks
    pred = self.residual_block1(central_mems[3])
    pred = self.residual_block2(pred)

    # We decompose the third central memory in two parts: a "prediction" part and an "upsampling
    # mask" part
    central_mem_2_pred = central_mems[2][:, :128, :, :]
    central_mem_2_mask = central_mems[2][:, 128:, :, :]

    # We apply the first decoder, guided by the upsampling mask
    pred = self.decoder1(pred, central_mem_2_mask)

    # We concatenate the prediction from the third central memory, and apply the convolution to go
    # from 256 to 128 channels
    pred = torch.concat((pred, central_mem_2_pred), dim=1)
    pred = self.conv_concat1(pred)

    # We decompose the second central memory in two parts: a "prediction" part and an "upsampling
    # mask" part
    central_mem_1_pred = central_mems[1][:, :64, :, :]
    central_mem_1_mask = central_mems[1][:, 64:, :, :]

    # We apply the second decoder, guided by the upsampling mask
    pred = self.decoder2(pred, central_mem_1_mask)

    # We concatenate the prediction from the second  central memory, and apply the convolution to go
    # from 128 to 64 channels
    pred = torch.concat((pred, central_mem_1_pred), dim=1)
    pred = self.conv_concat2(pred)

    # We decompose the first central memory in two parts: a "prediction" part and an "upsampling
    # mask" part
    central_mem_0_pred = central_mems[0][:, :32, :, :]
    central_mem_0_mask = central_mems[0][:, 32:, :, :]

    # We apply the last decoder, guided by the upsampling mask
    pred = self.decoder3(pred, central_mem_0_mask)

    # We concatenate the prediction from the first central memory, and apply the convolution to go
    # from 64 to 32 channels
    pred = torch.concat((pred, central_mem_0_pred), dim=1)
    pred = self.conv_concat3(pred)

    # We finish by applying the prediction layer
    pred = self.prediction_layer(pred)

    # We return the prediction and the updated central memories
    return pred, central_mems
