#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the PyTorch code for the DELTA network, as originally described in the "DELTA:
Dense Depth from Events and LiDAR using Transformer's Attention" article (CVPRW 2025).
"""

import torch
from torch import nn, Tensor

from models.submodules.shared_submodules import MEHGRU, MultiheadAttentionPreLN, PositionalEncoder2D
from models.submodules.delta_submodules import ConvEncodingHead, ConvDecodingHead


class DELTA(nn.Module):
  """
  The DELTA network, as described in the article.
  It is composed of two branches, one for the events, the other one for the LiDAR scans, and
  uses self- and cross-attention for encoding/decoding and fusion, and GRUs for memory purposes.
  """

  def __init__(self, lidar_channels: int, event_channels: int, out_channels: int,
               sa_layers: int, patch_size: int, dimensionality: int, ffnn_dimensionality: int,
               nbr_heads: int, prop_mem_size: int):
    super(DELTA, self).__init__()

    # We save the number of self-attention layers and the patch size
    self.sa_layers = sa_layers
    self.patch_size = patch_size

    # The initial state of the propagation memory (which is learnt)
    self.initial_prop_mem = nn.Parameter(torch.empty((prop_mem_size, dimensionality)))
    nn.init.uniform_(self.initial_prop_mem)

    # The saved LiDAR data, used when there is no new LiDAR data available
    self.saved_lidar = None

    # The encoding heads for the LiDAR and event inputs
    self.lidar_head = ConvEncodingHead(patch_size, lidar_channels, dimensionality)
    self.event_head = ConvEncodingHead(patch_size, event_channels, dimensionality)

    # The 2D positional encoder
    # We consider here that the max input resolution for our network is 1284x720, but this can be
    # changed if needed
    self.pos_encoder = PositionalEncoder2D(dimensionality, (720//patch_size, 1284//patch_size))

    # The N self-attention modules for the encoded LiDAR and events data
    self.lidar_sa = nn.ModuleList([MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality) for _ in range(sa_layers)])
    self.event_sa = nn.ModuleList([MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality) for _ in range(sa_layers)])

    # The cross-attention modules to update the prop. memory and to use it to propagate the LiDAR
    self.prop_mem_update_ca = MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality)
    self.lidar_prop_mem_ca = MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality)

    # The central cross-attention between the propagated LiDAR and the events
    self.central_ca = MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality)

    # The GRU-based memory update
    self.mem_update_gru = MEHGRU(dimensionality, dimensionality)

    # The self-attention modules for the decoder
    self.decoder_sa = nn.ModuleList([MultiheadAttentionPreLN(dimensionality, nbr_heads, ffnn_dimensionality) for _ in range(sa_layers)])

    # The layer normalization modules for the skip connections
    self.skip_norm = nn.ModuleList([nn.LayerNorm(dimensionality) for _ in range(sa_layers)])

    # The final feed-forward decoder
    self.decoding_head = ConvDecodingHead(dimensionality, patch_size, out_channels)


  def forward(self, lidar_input: Tensor | None, event_input: Tensor | None, central_mem: Tensor,
              prop_mem: Tensor, crop_positions: Tensor = None) -> tuple[Tensor, Tensor, Tensor]:
    # PART 0: CHECKING IF LIDAR DATA IS AVAILABLE
    # If not, we initialize it from the saved one
    # Otherwise, we replace the saved LiDAR by the new one
    if lidar_input is None:
      lidar_input = self.saved_lidar.clone()
    else:
      self.saved_lidar = lidar_input.clone()

    # PART 1: GETTING THE POSITIONAL ENCODING
    # crop_positions is of shape (B, 2), and contains the position of the top-left patch
    # We want the positions of all patches, of shape (B, L, 2)

    # We begin by extracting the batch size, and the patched height and width
    batch_size, _, h, w = lidar_input.shape
    h_p = h // self.patch_size
    w_p = w // self.patch_size

    # If crop_positions is None (i.e., no cropping), we set it to (0, 0) for every element in the
    # batch (i.e., the top-left patch is the one at position (0, 0))
    # Otherwise, we scale it with respect to the patch size
    if crop_positions is None:
      crop_positions = torch.zeros((batch_size, 2))
    else:
      crop_positions //= self.patch_size

    # We create an empty Tensor which will hold the positions
    positions = torch.empty((batch_size, h_p, w_p, 2), dtype=torch.long).to(lidar_input.device)

    # Then, for each batch, we fill this Tensor accordingly
    for b in range(batch_size):
      pos_x = torch.arange(crop_positions[b, 0], crop_positions[b, 0]+w_p)
      pos_y = torch.arange(crop_positions[b, 1], crop_positions[b, 1]+h_p)
      positions_meshgrid = torch.meshgrid(pos_x, pos_y, indexing="xy")
      positions[b, :, :, 0] = positions_meshgrid[0]
      positions[b, :, :, 1] = positions_meshgrid[1]

    # We reshape the positions, to have a correct shape of (B, L, 2)
    positions = positions.reshape(batch_size, -1, 2)

    # And we finish by encoding them through the 2D positional encoder
    encoded_pos = self.pos_encoder(positions)

    # PART 2: VERIFYING IF THE GIVEN MEMORIES ARE INITIALIZED, OTHERWISE INITIALIZE THEM
    # We initialize the central memory if necessary, using the positional encoding
    if central_mem is None:
      central_mem = encoded_pos.clone()

    # We initialize the propagation memory if necessary
    if prop_mem is None:
      # We get the learnt initial memory, and duplicate it for every batch
      # For more details, see:
      # https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/
      # https://github.com/AlbertoSabater/EventTransformer/blob/main/models/EvT.py#L258
      prop_mem = self.initial_prop_mem.clone().unsqueeze(0).expand(batch_size, -1, -1)

    # PART 3: ENCODING THE LIDAR DATA
    # Since we'll use the output of each SA layer for skip connections, we save them in a list
    encod_lidars = []

    # We first go through the LiDAR encoding head
    encod_lidar, _ = self.lidar_head(lidar_input)

    # We add the positional embedding
    encod_lidar = encod_lidar + encoded_pos

    # We use the prop. memory to propagate the LiDAR data
    propagated_lidar = self.lidar_prop_mem_ca(encod_lidar, prop_mem)
    encod_lidars.append(propagated_lidar)

    # And we apply the self-attention
    for i in range(self.sa_layers):
      encod_lidars.append(self.lidar_sa[i](encod_lidars[-1], encod_lidars[-1]))

    # PART 4: ENCODING THE EVENTS DATA
    # Since we'll use the output of each SA layer for skip connections, we save them in a list
    encod_evts = []

    # We first go through the events encoding head
    encod_evt, skip_evt = self.event_head(event_input)

    # We add the positional embedding
    encod_evt = encod_evt + encoded_pos

    # And we use the it as the input for the self-attention encoder
    encod_evts.append(encod_evt)

    # We apply the self-attention
    for i in range(self.sa_layers):
      encod_evts.append(self.event_sa[i](encod_evts[-1], encod_evts[-1]))

    # PART 5: UPDATING THE PROPAGATION MEMORY WITH THE EVENTS FOR THE NEXT TIME
    # We update the prop. memory with the encoded events (encoded only by the head, not after SA)
    prop_mem = self.prop_mem_update_ca(prop_mem, encod_evts[0])

    # PART 6: USING THE PROPAGATED LIDAR AND THE EVENTS TO UPDATE THE CENTRAL MEMORY
    # We apply the central CA between the LiDAR and the events
    fused_lidar_evts = self.central_ca(encod_evts[-1], encod_lidars[-1])

    # We compute the new memory by using the GRU module
    central_mem = self.mem_update_gru(fused_lidar_evts, central_mem)

    # PART 7: DECODING THE MEMORY TO PREDICT THE FINAL DEPTH VALUES
    # Our initial prediction is based on this memory
    pred = central_mem.clone()

    # We apply the self-attention layers on it, and add the summed and normalized inputs through the
    # skip connections after every layer
    for i in range(self.sa_layers):
      pred = self.decoder_sa[i](pred, pred)
      fused_skip = encod_lidars[-i-2] + encod_evts[-i-2]
      fused_skip = self.skip_norm[i](fused_skip)
      pred = pred + fused_skip

    # We apply the final decoding head
    h_p = lidar_input.shape[2] // self.patch_size
    w_p = lidar_input.shape[3] // self.patch_size
    pred = self.decoding_head(pred, skip_evt, h_p, w_p)

    # We return the prediction, the central memory, and the updated propagation memory
    return pred, central_mem, prop_mem
