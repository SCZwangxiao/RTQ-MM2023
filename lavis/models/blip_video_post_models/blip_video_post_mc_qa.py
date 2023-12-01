"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from lavis.common.registry import registry
from lavis.models.blip_video_models.blip_video_mc_qa import BlipVideoMcQA
from lavis.models.blip_models.blip_outputs import (
    BlipIntermediateOutput,
    BlipOutputWithLogits,
)
from lavis.models.med import XBertEncoder
from lavis.models.backbones.vit import VisionTransformerEncoder
from .vit_post import VisionTransformerEncoderWithPostProcess



@registry.register_model("blip_video_post_mc_qa")
class BlipVideoPostMcQA(BlipVideoMcQA):
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

        momentum = cfg.get("momentum", 0.995)
        use_distill = cfg.get("use_distill", True)
        num_classes = cfg.get("num_classes", -1)
        alpha = cfg.get("alpha", 0.4)
        max_txt_len = cfg.get("max_txt_len", 40)

        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            use_distill=use_distill,
            alpha=alpha,
            num_classes=num_classes,
            momentum=momentum,
            max_txt_len=max_txt_len,
        )

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            msg = model.load_from_pretrained(url_or_filename=pretrain_path)

        return model