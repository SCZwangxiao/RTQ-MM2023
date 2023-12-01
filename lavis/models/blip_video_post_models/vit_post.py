"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip_video_models.blip_video_retrieval import BlipVideoRetrieval
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipSimilarity,
    BlipIntermediateOutput,
)
from lavis.models.med import XBertEncoder
from lavis.models.backbones.vit import VisionTransformerEncoder
from torch import nn


class VisionTransformerEncoderWithPostProcess(VisionTransformerEncoder):
    @classmethod
    def from_config(cls, cfg, from_pretrained=False):
        visual_encoder = super().from_config(cfg, from_pretrained)

        num_frames = cfg.get("num_frames")
        adapter_configs = cfg.get("adapter_configs")
        adapter_arch = adapter_configs.pop('arch')

        post_adapter = registry.get_model_class(adapter_arch)(num_frames=num_frames, 
                                                              **adapter_configs)

        visual_encoder.num_frames = num_frames
        visual_encoder.post_adapter = post_adapter
        return visual_encoder
    
    def forward_features(self, x, register_blk=-1):
        # [b t c h w]
        B, T = x.shape[:2]
        x = x.reshape((-1,) + x.shape[2:])
        # [bt c h w]
        image_embeds = super().forward(x, register_blk)
        # [bt p d]
        video_embeds, video_outputs = self.post_adapter(image_embeds)
        # [b d] [b p' d]
        return video_embeds, video_outputs

    def forward(self, x, register_blk=-1):
        return self.forward_features(x, register_blk)[0]