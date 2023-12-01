"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from lavis.common.utils import is_url
from lavis.common.registry import registry
from lavis.common.dist_utils import download_cached_file
from lavis.models.backbones.vit import interpolate_pos_embed
from lavis.models.blip_models.blip_outputs import BlipOutput
from lavis.models.blip_models.blip_vqa import BlipVQA
from lavis.models.blip_video_models.blip_video_outputs import BlipVideoIntermediateOutput
from lavis.models.blip_video_models.blip_video_qa import BlipVideoQA
from lavis.models.med import XBertEncoder, XBertLMHeadDecoder
from .vit_post import VisionTransformerEncoderWithPostProcess


@registry.register_model("blip_video_post_qa")
class BlipVideoPostQA(BlipVideoQA):

    @classmethod
    def from_config(cls, cfg=None):
        # vision encoder
        backbone_arch = cfg.get('backbone_arch', 'vit')
        if backbone_arch == 'vit':
            image_encoder = VisionTransformerEncoderWithPostProcess.from_config(cfg)
        else:
            image_encoder = registry.get_model_class(backbone_arch).from_config(cfg)

        # text encoder + multimodal encoder
        text_encoder = XBertEncoder.from_config(cfg)
        text_decoder = XBertLMHeadDecoder.from_config(cfg)

        if cfg.get("freeze_video", False):
            for param in image_encoder.parameters():
                param.requires_grad = False
        if cfg.get("freeze_text", False):
            for param in text_decoder.parameters():
                param.requires_grad = False
        if cfg.get("tune_crossattn_only", True):
            for key, param in text_decoder.named_parameters():
                if 'crossattention' not in key:
                    param.requires_grad = False
        
        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model