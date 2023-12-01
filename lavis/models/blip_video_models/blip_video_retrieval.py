"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from copy import deepcopy

import torch
import logging
import torch.nn.functional as F

from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.backbones.vit import interpolate_pos_embed
from lavis.common.registry import registry
from lavis.models.blip_video_models import compute_sim_matrix
from lavis.models.base_model import (
    MomentumDistilationMixin,
    SharedQueueMixin,
    all_gather_with_grad,
    concat_all_gather,
)
from lavis.models.blip_models.blip_retrieval import BlipRetrieval
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipSimilarity,
)
from lavis.models.blip_video_models.blip_video_outputs import BlipVideoIntermediateOutput
from torch import nn


class BlipVideoRetrieval(BlipRetrieval):

    def forward(self, samples):
        video = samples["video"]
        caption = samples["text_input"]
        idx = samples["video_id"]

        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"],
            iters=samples["iters"],
            num_iters_per_epoch=samples["num_iters_per_epoch"],
        )

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        video_feat, video_outputs = self.visual_encoder.forward_features(video)
        video_atts = torch.ones(video_outputs.size()[:-1], dtype=torch.long).to(
            video.device
        )
        video_feat = F.normalize(self.vision_proj(video_feat), dim=-1)

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(video.device)

        text_output = self.text_encoder.forward_text(text)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # video-text Contrastive Learning
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            video_feat_m = self.visual_encoder_m(video)
            video_feat_m = F.normalize(
                self.vision_proj_m(video_feat_m), dim=-1
            )
            video_feat_m_all = torch.cat(
                [video_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )

            text_output_m = self.text_encoder_m.forward_text(text)
            text_embeds_m = text_output_m.last_hidden_state
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = video_feat_m @ text_feat_m_all / self.temp
            sim_t2i_m = text_feat_m @ video_feat_m_all / self.temp

            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        sim_i2t = video_feat @ text_feat_m_all / self.temp
        sim_t2i = text_feat @ video_feat_m_all / self.temp

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(video_feat_m, text_feat_m, idx)

        # video-text Matching
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positve video-text pair
        bs = video.size(0)
        output_pos = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=video_outputs,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )

        idxs = concat_all_gather(idx)
        if self.negative_all_rank:
            # compute sample similarity
            with torch.no_grad():
                mask = torch.eq(idx, idxs.t())

                video_feat_world = concat_all_gather(video_feat)
                text_feat_world = concat_all_gather(text_feat)

                sim_i2t = video_feat @ text_feat_world.t() / self.temp
                sim_t2i = text_feat @ video_feat_world.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            video_outputs_world = all_gather_with_grad(video_outputs)

            # select a negative video (from all ranks) for each text
            video_outputs_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                video_outputs_neg.append(video_outputs_world[neg_idx])
            video_outputs_neg = torch.stack(video_outputs_neg, dim=0)

            # select a negative text (from all ranks) for each video
            input_ids_world = concat_all_gather(encoder_input_ids)
            att_mask_world = concat_all_gather(text.attention_mask)

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])

        else:
            with torch.no_grad():
                mask = torch.eq(idx, idx.t())

                sim_i2t = video_feat @ text_feat.t() / self.temp
                sim_t2i = text_feat @ video_feat.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            # select a negative video (from same rank) for each text
            video_outputs_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                video_outputs_neg.append(video_outputs[neg_idx])
            video_outputs_neg = torch.stack(video_outputs_neg, dim=0)

            # select a negative text (from same rank) for each video
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        video_outputs_all = torch.cat([video_outputs_neg, video_outputs], dim=0)
        video_atts_all = torch.cat([video_atts, video_atts], dim=0)

        output_neg = self.text_encoder(
            text_ids_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=video_outputs_all,
            encoder_attention_mask=video_atts_all,
            return_dict=True,
        )

        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        itm_logits = self.itm_head(vl_embeddings)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(self.device)
        loss_itm = F.cross_entropy(itm_logits, itm_labels)

        return BlipOutput(
            loss=loss_itc + loss_itm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            sims=BlipSimilarity(
                sim_i2t=sim_i2t,
                sim_t2i=sim_t2i,
                sim_i2t_m=sim_i2t_m,
                sim_t2i_m=sim_t2i_m,
                sim_i2t_targets=sim_i2t_targets,
                sim_t2i_targets=sim_t2i_targets,
            ),
            intermediate_output=BlipVideoIntermediateOutput(
                video_embeds=video_outputs,
                video_embeds_m=video_feat_m,
                text_embeds=text_embeds,
                text_embeds_m=text_embeds_m,
                encoder_output=output_pos,
                encoder_output_neg=output_neg,
                itm_logits=itm_logits,
                itm_labels=itm_labels,
            ),
        )

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

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
