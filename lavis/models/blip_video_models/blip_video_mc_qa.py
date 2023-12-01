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
from lavis.models.blip_models.blip_outputs import BlipOutputWithLogits
from lavis.models.blip_models.blip_classification import BlipClassification
from lavis.models.blip_video_models.blip_video_outputs import BlipVideoIntermediateOutput


@registry.register_model("blip_video_mc_qa")
class BlipVideoMcQA(BlipClassification):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/blip_post/blip_base_mc_qa.yaml",
    }
    def __init__(
        self,
        image_encoder,
        text_encoder,
        num_classes,
        momentum=0.995,
        alpha=0.4,
        max_txt_len=40,
        use_distill=True,
    ):
        super().__init__(image_encoder=image_encoder,
                         text_encoder=text_encoder,
                         num_classes=1,
                         momentum=momentum,
                         alpha=alpha,
                         max_txt_len=max_txt_len,
                         use_distill=use_distill)
        self.num_answers = num_classes

    def forward(self, samples, is_train=True):
        sentences = [txt for batch_txt in samples["text_input"] for txt in batch_txt]
        sentences = self.tokenizer(
            sentences,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        # [(n b) l] [(n b) l] [(n b) l]
        samples.update({"tokenized_text": sentences})

        targets = samples["label"]

        video_embeds = self.visual_encoder.forward_features(samples["video"])[1]
        # [b p h]
        video_embeds = repeat(video_embeds, 'b p h -> (n b) p h', n=self.num_answers)
        encoder_output = self.text_encoder.forward_automask(
            samples["tokenized_text"], video_embeds
        ) 
        # [(n b) p h]

        prediction = self.cls_head(encoder_output.last_hidden_state[:, 0, :]).squeeze(1)
        prediction = rearrange(prediction, '(n b) -> b n', n=self.num_answers)
        # [b n]

        if is_train:
            if self.use_distill:
                with torch.no_grad():
                    self._momentum_update()

                    video_embeds_m = self.visual_encoder_m.forward_features(samples["video"])[1]
                    # [b p h]
                    video_embeds_m = repeat(video_embeds_m, 'b p h -> (n b) p h', n=self.num_answers)
                    encoder_output_m = self.text_encoder_m.forward_automask(
                        samples["tokenized_text"], video_embeds_m
                    )
                    # [(n b) p h]

                    prediction_m = self.cls_head_m(
                        encoder_output_m.last_hidden_state[:, 0, :]
                    ).squeeze(1)
                    prediction_m = rearrange(prediction_m, '(n b) -> b n', n=self.num_answers)
                    # [b n]

                alpha = self.alpha * self._rampup_factor(
                    epoch=samples["epoch"],
                    iters=samples["iters"],
                    num_iters_per_epoch=samples["num_iters_per_epoch"],
                )

                loss = (1 - alpha) * F.cross_entropy(
                    prediction, targets
                ) - alpha * torch.sum(
                    F.log_softmax(prediction, dim=1) * F.softmax(prediction_m, dim=1),
                    dim=1,
                ).mean()
            else:
                loss = F.cross_entropy(prediction, targets)

            # return {"loss": loss}
            return BlipOutputWithLogits(
                loss=loss,
                intermediate_output=BlipVideoIntermediateOutput(
                    video_embeds=video_embeds,
                    video_embeds_m=video_embeds_m,
                    encoder_output=encoder_output,
                    encoder_output_m=encoder_output_m,
                ),
                logits=prediction,
                logits_m=prediction_m,
            )

        else:
            return {"predictions": prediction, "targets": targets}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output
    
    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        if hasattr(self.visual_encoder, 'modify_state_dict'):
            logging.info("Modify state dict according to visual encoder {}.".format(type(self.visual_encoder)))
            state_dict = self.visual_encoder.modify_state_dict(state_dict)
        if hasattr(self.text_encoder, 'modify_state_dict'):
            state_dict = self.text_encoder.modify_state_dict(state_dict)
            logging.info("Modify state dict according to text encoder {}.".format(type(self.text_encoder)))

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        if hasattr(self.visual_encoder, 'modify_state_dict'):
            logging.info("Modify state dict according to image encoder {}.".format(type(self.visual_encoder)))
            state_dict = self.visual_encoder.modify_state_dict(state_dict)
        if hasattr(self.text_encoder, 'modify_state_dict'):
            state_dict = self.text_encoder.modify_state_dict(state_dict)
            logging.info("Modify state dict according to text encoder {}.".format(type(self.text_encoder)))

        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )
        
        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg