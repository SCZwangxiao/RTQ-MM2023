"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging

import torch

from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.backbones.vit import interpolate_pos_embed
from lavis.common.registry import registry
from lavis.models.blip_models.blip_outputs import BlipOutput
from lavis.models.blip_video_models.blip_video_outputs import BlipVideoIntermediateOutput
from lavis.models.blip_models.blip_caption import BlipCaption


class BlipVideoCaption(BlipCaption):

    def forward_encoder(self, samples):
        video_outputs = self.visual_encoder.forward_features(samples["video"])[1]
        # [B p' d]
        return video_outputs

    def forward(self, samples):
        video_outputs = self.forward_encoder(samples)
        decoder_output, decoder_targets = self.forward_decoder(samples, video_outputs)

        # return decoder_out
        return BlipOutput(
            loss=decoder_output.loss,
            loss_lm=decoder_output.loss,
            intermediate_output=BlipVideoIntermediateOutput(
                video_embeds=video_outputs,
                decoder_output=decoder_output,
                decoder_labels=decoder_targets,
            ),
        )
    
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
        if hasattr(self.text_decoder, 'modify_state_dict'):
            state_dict = self.text_decoder.modify_state_dict(state_dict)
            logging.info("Modify state dict according to text encoder {}.".format(type(self.text_decoder)))

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