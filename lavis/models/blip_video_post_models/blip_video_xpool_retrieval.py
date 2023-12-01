"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import time
import logging
import datetime
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, reduce

from lavis.common.logger import MetricLogger
from lavis.common.registry import registry
from lavis.models.med import XBertEncoder
import lavis.common.dist_utils as dist_utils
from lavis.models.backbones.vit import VisionTransformerEncoder
from lavis.models.blip_video_models.blip_video_retrieval import BlipVideoRetrieval
from lavis.models.base_model import (
    all_gather_with_grad,
    concat_all_gather,
)
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipSimilarity,
)
from lavis.models.blip_video_models.blip_video_outputs import BlipVideoIntermediateOutput


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_output = model.text_encoder.forward_text(text_input)
        # [B, L, h]
        text_embed = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :])
        )
        # [B, d]
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    if hasattr(model.tokenizer, "enc_token_id"):
        text_ids[:, 0] = model.tokenizer.enc_token_id

    video_outputs = []
    video_embeds = []
    start = -1
    end = 0
    for samples in data_loader:
        image = samples["video"]
        image = image.to(model.device)
        bs = image.shape[0]

        # Get the index of current samples
        start = end
        end += bs

        image = rearrange(image, 'b t c h w -> (b t) c h w')
        video_output = model.visual_encoder.forward_features(image)
        video_output = rearrange(video_output, '(b t) p h -> b t p h', b=bs)
        video_embed = F.normalize(
            model.vision_proj(video_output[:,:,0,:].mean(1))
        )
        # [B, h]
        video_outputs.append(video_output.cpu())
        video_embeds.append(video_embed)

    video_outputs = torch.cat(video_outputs, dim=0)
    video_embeds = torch.cat(video_embeds, dim=0)

    sims_matrix = video_embeds @ text_embeds.t()

    if k_test <= 0: # Do no perform reranking
        logging.info("k_test={}<=0, skip reranking.".format(k_test))
        return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every_eval(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

        encoder_output = video_outputs[start + i].repeat(k_test, 1, 1, 1).to(model.device)
        _, encoder_output = model.xpooler(encoder_output, text_embeds[topk_idx])
        # [B, p, h]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            model.device
        )
        output = model.text_encoder(
            text_ids[topk_idx],
            attention_mask=text_atts[topk_idx],
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every_eval(sims_matrix[start:end], 50, header)
    ):

        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = video_outputs[topk_idx.cpu()].to(model.device)
        _, encoder_output = model.xpooler(encoder_output, text_embeds[start + i].repeat(k_test, 1))
        # [B, p, h]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            model.device
        )
        output = model.text_encoder(
            text_ids[start + i].repeat(k_test, 1),
            attention_mask=text_atts[start + i].repeat(k_test, 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@registry.register_model("blip_video_xpool_retrieval")
class BlipVideoXpoolRetrieval(BlipVideoRetrieval):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "msrvtt": "configs/models/blip_adapter/blip_base_retrieval.yaml",
    }

    @classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased'
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        
        text_encoder = XBertEncoder.from_config(cfg)

        if cfg.get("freeze_video", False):
            for param in image_encoder.parameters():
                param.requires_grad = False
        if cfg.get("freeze_text", False):
            for param in text_encoder.parameters():
                param.requires_grad = False

        num_frames = cfg.get("num_frames")
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
        model.num_frames = num_frames

        model.load_checkpoint_from_config(cfg)
        model.reset_queue_ptr()

        return model
    
    def xpooler(self, video_outputs, text_feat):
        # [bt 1+p h] [b h]
        values = video_outputs
        keys = F.normalize(self.vision_proj(video_outputs), dim=-1)
        sims = torch.einsum('btph,bh->btp', keys, text_feat)
        attns = F.softmax(sims / self.temp, dim=1)
        # [b t p]
        video_outputs = torch.einsum('btph,btp->bph', values, attns)
        video_feat = video_outputs[:,0,:]

        return video_feat, video_outputs
    
    def forward(self, samples):
        video = samples["video"]
        caption = samples["text_input"]
        idx = samples["video_id"]

        flatten_video = rearrange(video, 'b t c h w -> (b t) c h w')

        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"],
            iters=samples["iters"],
            num_iters_per_epoch=samples["num_iters_per_epoch"],
        )

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

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

        video_outputs = self.visual_encoder.forward_features(flatten_video)
        video_outputs = rearrange(video_outputs, '(b t) p h -> b t p h', t=self.num_frames)
        video_feat = video_outputs[:,:,0,:].mean(1)
        video_feat = F.normalize(self.vision_proj(video_feat), dim=-1)
        

        # video-text Contrastive Learning
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()

            text_output_m = self.text_encoder_m.forward_text(text)
            text_embeds_m = text_output_m.last_hidden_state
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            video_outputs_m = self.visual_encoder_m(flatten_video)
            video_outputs_m = rearrange(video_outputs_m[:,:1,:], '(b t) p h -> b t p h', t=self.num_frames)
            # [b t 1 h]
            video_feat_m, _ = self.xpooler(video_outputs_m, text_feat_m)
            video_feat_m = F.normalize(self.vision_proj_m(video_feat_m), dim=-1)
            video_feat_m_all = torch.cat(
                [video_feat_m.t(), self.image_queue.clone().detach()], dim=1
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

        _, video_outputs_pos = self.xpooler(video_outputs, text_feat)
        video_atts = torch.ones(video_outputs_pos.size()[:-1], dtype=torch.long).to(
            video.device
        )
        output_pos = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=video_outputs_pos,
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
            text_feat_world = concat_all_gather(text_feat)

            text_ids_neg = []
            text_atts_neg = []
            text_feat_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])
                text_feat_neg.append(text_feat_world[neg_idx])

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
            text_feat_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])
                text_feat_neg.append(text_feat[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        text_feat_neg = torch.stack(text_feat_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)
        text_feat_all = torch.cat([text_feat, text_feat_neg], dim=0)

        video_outputs_all = torch.cat([video_outputs_neg, video_outputs], dim=0)
        video_atts_all = torch.cat([video_atts, video_atts], dim=0)

        _, video_outputs_all = self.xpooler(video_outputs_all, text_feat_all)
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