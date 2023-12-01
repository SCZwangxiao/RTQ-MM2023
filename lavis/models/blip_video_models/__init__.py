"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import os
import time

import lavis.common.dist_utils as dist_utils
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from lavis.common.dist_utils import download_cached_file
from lavis.common.logger import MetricLogger
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.backbones.vit import interpolate_pos_embed
from transformers import BertTokenizer


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
    for samples in data_loader:
        try: # video
            image = samples["video"]
        except KeyError: # image
            image = samples["image"]

        image = image.to(model.device)
        video_embed, video_output = model.visual_encoder.forward_features(image)
        # [B, h] [B, p, h]
        video_embed = model.vision_proj(video_embed)
        video_embed = F.normalize(video_embed, dim=-1)
        # {B, d}

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

        encoder_output = video_outputs[start + i].repeat(k_test, 1, 1).to(model.device)
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
