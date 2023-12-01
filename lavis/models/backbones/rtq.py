"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Based on timm code base
 https://github.com/rwightman/pytorch-image-models/tree/master/timm
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce

from mmengine.model.weight_init import constant_init
from lavis.common.registry import registry
from lavis.models import BaseModel
from lavis.models.backbones.vit import VisionTransformerEncoder, Attention, Block, interpolate_pos_embed
from .center_clip_utils import get_deep_cluster, get_cluster_inter
from .temporal_modules.divst import DividedSpaceTimeBlock
from .temporal_modules.msgtok import MessageTokenBlock
from .temporal_modules.vip import CLIPViPBlock


def build_temporal_modeling_modules(embed_dim,
                                    num_heads,
                                    mlp_ratio,
                                    qkv_bias,
                                    qk_scale,
                                    drop_rate,
                                    attn_drop_rate,
                                    dpr,
                                    norm_layer,
                                    use_grad_checkpointing,
                                    depth,
                                    ckpt_layer,
                                    num_frames_init,
                                    num_frames_final,
                                    posterior_temp_block_start_id,
                                    prior_temporal_cfg, 
                                    posterior_temporal_cfg):
    prior_arch = prior_temporal_cfg.get('arch') if prior_temporal_cfg else 'vit'
    posterior_arch = posterior_temporal_cfg.get('arch') if posterior_temporal_cfg else 'vit'
    vit_cfg = dict(dim=embed_dim,
                   num_heads=num_heads,
                   mlp_ratio=mlp_ratio,
                   qkv_bias=qkv_bias,
                   qk_scale=qk_scale,
                   drop=drop_rate,
                   attn_drop=attn_drop_rate,
                   norm_layer=norm_layer)

    # Check validity
    if posterior_temp_block_start_id is None:
        assert prior_arch == 'vit' and posterior_arch == 'vit'
        posterior_temp_block_start_id = (depth >> 1)
    if prior_arch == 'vit' and posterior_arch == 'vit':
        temporal_embedding_block_id = -1
    elif prior_arch == 'vit':
        temporal_embedding_block_id = posterior_temp_block_start_id
    else:
        temporal_embedding_block_id = 0

    # Build prior temporal modeling modules
    if prior_arch == 'vit':
        blocks = [
            Block(drop_path=dpr[i],
                  use_grad_checkpointing=(
                      use_grad_checkpointing and i >= depth - ckpt_layer
                  ),
                  **vit_cfg)
            for i in range(0, posterior_temp_block_start_id)
        ]
    elif prior_arch == 'divided_attention':
        blocks = [
            DividedSpaceTimeBlock(layer_id=i,
                                  num_frames=num_frames_init,
                                  drop_path=dpr[i],
                                  use_grad_checkpointing=(
                                      use_grad_checkpointing and i >= depth - ckpt_layer
                                  ),
                                  **vit_cfg
            )
            for i in range(0, posterior_temp_block_start_id)
        ]
    else:
        raise NotImplementedError

    # Build posterior temporal modeling modules
    if posterior_arch == 'vit':
        blocks.extend([
            Block(drop_path=dpr[i],
                  use_grad_checkpointing=(
                      use_grad_checkpointing and i >= depth - ckpt_layer
                  ),
                  **vit_cfg)
            for i in range(posterior_temp_block_start_id, depth)
        ])
    elif posterior_arch == 'divided_attention':
        blocks.extend([
            DividedSpaceTimeBlock(layer_id=i,
                                  num_frames=num_frames_final,
                                  drop_path=dpr[i],
                                  use_grad_checkpointing=(
                                      use_grad_checkpointing and i >= depth - ckpt_layer
                                  ),
                                  **vit_cfg)
            for i in range(posterior_temp_block_start_id, depth)
        ])
    elif posterior_arch == 'video_proxy':
        num_proxies = posterior_temporal_cfg.get('num_proxies', 1)
        blocks.extend([
            CLIPViPBlock(num_proxies=num_proxies,
                         num_frames=num_frames_final,
                         drop_path=dpr[i],
                         use_grad_checkpointing=(
                             use_grad_checkpointing and i >= depth - ckpt_layer
                         ),
                         **vit_cfg)
            for i in range(posterior_temp_block_start_id, depth)
        ])
    elif posterior_arch == 'message_token':
        blocks.extend([
            MessageTokenBlock(num_frames=num_frames_final,
                              drop_path=dpr[i],
                              use_grad_checkpointing=(
                                  use_grad_checkpointing and i >= depth - ckpt_layer
                              ),
                              **vit_cfg)
            for i in range(posterior_temp_block_start_id, depth)
        ])
    else:
        raise NotImplementedError

    return nn.ModuleList(blocks), temporal_embedding_block_id


class ClusterBlock(nn.Module):
    def __init__(
        self,
        dim,
        block_id,
        refinement_cfg=None,
    ):
        super().__init__()
        
        ############################################ NEW ADDED CODE ############################################
        self.block_id = block_id
        self.tokencluster_inter = get_cluster_inter(dim, block_id, refinement_cfg)

    def forward(self, x, register_hook=False):    
        # [(b t) p h]
        if self.tokencluster_inter is not None:
            x, res_x = self.tokencluster_inter(x)
    
        if self.tokencluster_inter is not None and self.tokencluster_inter.algorithm == 'token_shift':
            x, _ = self.tokencluster_inter(x)

        return x


@registry.register_model("rtq")
class RTQEncoder(VisionTransformerEncoder):
    PRETRAINED_MODEL_CONFIG_DICT = {}

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        use_grad_checkpointing=False,
        ckpt_layer=0,
        cfg=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            representation_size,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            use_grad_checkpointing,
            ckpt_layer)
        self.depth = depth

        self.media = "video"
        self.num_frames_init = cfg['num_frames']

        # Build Refinement module
        refinement_cfg = cfg.get('refinement_cfg', None)
        if refinement_cfg:
            refinement_cfg['num_patches'] = self.num_patches
            refinement_cfg['num_frames'] = self.num_frames_init
            self.num_frames_final = refinement_cfg.get('target_frames_blocks')[-1]
        else:
            self.num_frames_final = self.num_frames_init

        self.refine_blocks = nn.ModuleList(
            [
                ClusterBlock(
                    dim=embed_dim,
                    block_id=i,
                    refinement_cfg=refinement_cfg,
                )
                for i in range(depth)
            ]
        )

        # Build Temporal modeling module
        self.posterior_temp_block_start_id = cfg.get('posterior_temp_block_start_id', None)
        prior_temporal_cfg = cfg.get('prior_temporal_cfg', None)
        posterior_temporal_cfg = cfg.get('posterior_temporal_cfg', None)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks, self.temporal_embedding_block_id = \
            build_temporal_modeling_modules(embed_dim,
                                            num_heads,
                                            mlp_ratio,
                                            qkv_bias,
                                            qk_scale,
                                            drop_rate,
                                            attn_drop_rate,
                                            dpr,
                                            norm_layer,
                                            use_grad_checkpointing,
                                            depth,
                                            ckpt_layer,
                                            self.num_frames_init,
                                            self.num_frames_final,
                                            self.posterior_temp_block_start_id,
                                            prior_temporal_cfg, 
                                            posterior_temporal_cfg)

        if self.temporal_embedding_block_id == 0:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames_init, embed_dim))
        elif self.temporal_embedding_block_id > 0:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames_final, embed_dim))

        self.apply(self._init_weights)

    def add_temporal_embedding(self, embeddings, B):
        # [(b t_*) p h]
        embeddings = rearrange(embeddings, 
                               '(b t) p h -> (b p) t h', 
                               b=B)
        embeddings = embeddings + self.temporal_embedding
        embeddings = rearrange(embeddings, 
                               '(b p) t h -> (b t) p h', 
                               b=B)
        embeddings = self.pos_drop(embeddings)
        return embeddings

    def forward_features_image(self, x, register_blk=-1):
        # [b 3 h w]
        last_hidden_state = super().forward(x, register_blk)
        # [b 1+p h]

        return last_hidden_state
        
    def forward_features_video(self, x, register_blk=-1):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B*T, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        # [(b T) p h]
        for i, blk in enumerate(self.blocks):
            x = self.refine_blocks[i](x, register_blk == i)

            if i == self.temporal_embedding_block_id:
                # [(b t_*) p h]
                x = self.add_temporal_embedding(x, B)
                # [(b t_*) p h]

            x = blk(x, register_blk == i)
        last_hidden_state = self.norm(x)

        if x.shape[0] == B:
            # [(b m+pt_o h]
            x_pooled = last_hidden_state[:,0]
        elif x.shape[0] == B * self.num_frames_final:
            # [(b t_o) p h]
            x_pooled = reduce(last_hidden_state[:,0], '(b to) h -> b h', 'mean', b=B)
            if self.temporal_embedding_block_id == -1: # No temporal modeling
                last_hidden_state = reduce(last_hidden_state,
                                           '(b to) p h -> b p h', 'mean', b=B)
            else:
                last_hidden_state = rearrange(last_hidden_state,
                                              '(b to) p h -> b (p to) h', b=B)

        return x_pooled, last_hidden_state

    def forward_features(self, x, register_blk=-1):
        if self.media == "video":
            return self.forward_features_video(x, register_blk)
        elif self.media == "image":
            return self.forward_features_image(x, register_blk)
        else:
            raise NotImplementedError

    def forward(self, x, register_blk=-1):
        if self.media == "video":
            return self.forward_features_video(x, register_blk)[0]
        elif self.media == "image":
            return self.forward_features_image(x, register_blk)
        else:
            raise NotImplementedError