"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

import torch
from lavis.common.registry import registry
from lavis.models.med import XBertEncoder
from lavis.models.blip_video_models.blip_video_retrieval import BlipVideoRetrieval

from .vit_post import VisionTransformerEncoderWithPostProcess


@registry.register_model("blip_video_post_retrieval")
class BlipVideoPostRetrieval(BlipVideoRetrieval):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/blip_post/blip_base_retrieval.yaml",
    }

    @classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased'
        backbone_arch = cfg.get('backbone_arch', 'vit')
        if backbone_arch == 'vit':
            image_encoder = VisionTransformerEncoderWithPostProcess.from_config(cfg)
        else:
            image_encoder = registry.get_model_class(backbone_arch).from_config(cfg)
        
        text_encoder = XBertEncoder.from_config(cfg)

        if cfg.get("freeze_video", False):
            for param in image_encoder.parameters():
                param.requires_grad = False
        if cfg.get("freeze_text", False):
            for param in text_encoder.parameters():
                param.requires_grad = False

        embed_dim = cfg.get("embed_dim", 256)
        momentum = cfg.get("momentum", 0.995)
        alpha = cfg.get("alpha", 0.4)
        negative_all_rank = cfg.get("negative_all_rank", False)

        queue_size = cfg.get("queue_size", 0)
        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            queue_size=queue_size,
            alpha=alpha,
            embed_dim=embed_dim,
            momentum=momentum,
            negative_all_rank=negative_all_rank,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)
        model.reset_queue_ptr()

        return model