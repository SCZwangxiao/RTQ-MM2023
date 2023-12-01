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
from einops import rearrange
from functools import partial

from mmengine.model.weight_init import constant_init
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from lavis.common.registry import registry
from lavis.models.base_model import BaseEncoder
from lavis.models.backbones.vit import VisionTransformerEncoder, Attention, Block, interpolate_pos_embed


class TimeSformerBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        num_frames,
        layer_id=-1,
        layers_w_divst=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_grad_checkpointing=False,
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
            use_grad_checkpointing)
        
        self.num_frames = num_frames
        if layers_w_divst:
            assert layer_id > -1

        if layers_w_divst is None or layer_id in layers_w_divst:
            self.enable_divst = True
            self.temporal_attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
            self.layer_norm0 = nn.LayerNorm(dim)
            self.temporal_fc = nn.Linear(dim, dim)
            constant_init(self.temporal_fc, val=0, bias=0)

            if use_grad_checkpointing:
                self.temporal_attn = checkpoint_wrapper(self.temporal_attn)
                self.temporal_fc = checkpoint_wrapper(self.temporal_fc)
        else:
            self.enable_divst = False
    
    def _temporal_attn(self,
                       hidden_states,
                       state_shape,
                       register_hook):
        b, p, t, h = state_shape
        init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        # [B, 1, h]
        residual = hidden_states_t = hidden_states[:, 1:, :]
        # [b, pt, h]
        if self.enable_divst:
            hidden_states_t = self.layer_norm0(hidden_states_t.reshape(b * p, t, h))
            hidden_states_t = self.temporal_attn(
                hidden_states_t, register_hook
            )
            hidden_states_t = self.drop_path(hidden_states_t.contiguous())
            hidden_states_t = self.temporal_fc(hidden_states_t)
            # [bp, t, h]
            hidden_states = residual + hidden_states_t.reshape(b, p * t, h)
            # [b, pt, h]
        else:
            hidden_states = residual
        return init_cls_token, hidden_states
    
    def _spatial_attn(self,
                      init_cls_token,
                      hidden_states,
                      state_shape,
                      register_hook):
        b, p, t, h = state_shape
        residual = torch.cat((init_cls_token, hidden_states), 1)
        # [b, 1+pt, h]
        hidden_states_s = hidden_states
        # [b, pt, h]
        cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t,
                                                           h).unsqueeze(1)
        # [bt, 1, h]
        hidden_states_s = rearrange(hidden_states_s, 'b (p t) h -> (b t) p h', p=p, t=t)
        hidden_states_s = torch.cat((cls_token, hidden_states_s), 1)
        hidden_states_s = self.norm1(hidden_states_s)
        hidden_states_s = self.attn(
            hidden_states_s, register_hook
        )
        hidden_states_s = self.drop_path(hidden_states_s.contiguous())
        # [bt, 1+p, h]
        cls_token = hidden_states_s[:, 0, :].reshape(b, t, h)
        cls_token = torch.mean(cls_token, 1, True)
        # [b, 1, h]
        hidden_states_s = rearrange(
            hidden_states_s[:, 1:, :], '(b t) p h -> b (p t) h', p=p, t=t)
        hidden_states = torch.cat((cls_token, hidden_states_s), 1)
        # [b, 1+pt, h]
        hidden_states = residual + hidden_states
        return hidden_states
    
    def _ffn(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
    def forward(self, hidden_states, register_hook=False):
        # Get tensor size
        
        # [b, 1+pt, h]
        b, pt, h = hidden_states.size()
        pt -= 1
        p, t = pt // self.num_frames, self.num_frames
        state_shape = (b, p, t, h)

        init_cls_token, hidden_states = self._temporal_attn(
            hidden_states, state_shape, register_hook
        )
        # [B, 1, h] [b, pt, h] [bp, t, t]
        
        hidden_states = self._spatial_attn(
            init_cls_token, hidden_states, state_shape, register_hook
        )
        # [b, 1+pt, h] [bt, 1+p, 1+p]

        hidden_states = self._ffn(hidden_states)
        # [b, 1+pt, h]

        return hidden_states


@registry.register_model("timesformer")
class TimeSformerEncoder(VisionTransformerEncoder):
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
        num_frames = cfg.get("num_frames")
        layers_w_divst = cfg.get("layers_w_divst", list(range(depth)))

        self.num_frames = num_frames

        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                TimeSformerBlock(
                    layer_id=i,
                    dim=embed_dim,
                    num_frames=num_frames,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_grad_checkpointing=(
                        use_grad_checkpointing and i >= depth - ckpt_layer
                    ),
                    layers_w_divst=layers_w_divst,
                )
                for i in range(depth)
            ]
        )
    
    def forward_features(self, pixel_values, register_blk=-1):
        # [bt c h w]
        B, T, C, H, W = pixel_values.shape
        pixel_values = pixel_values.reshape(B*T, C, H, W)
        # [bt c h w]

        # Get vision embeddings
        patch_embeds = self.patch_embed(pixel_values)
        # [bt p d]
        cls_tokens = self.cls_token.expand(
            B*T, -1, -1
        )
        embeddings = torch.cat((cls_tokens, patch_embeds), dim=1)
        # [bt 1+p d]
        embeddings = embeddings + self.pos_embed[:, :embeddings.size(1), :]

        cls_tokens = embeddings[:B, 0, :].unsqueeze(1)
        # [b, 1, h]
        embeddings = rearrange(embeddings[:, 1:, :], 
                               '(b t) p h -> (b p) t h', 
                               b=B)
        embeddings = embeddings + self.temporal_embedding[:,:T,:]
        embeddings = rearrange(embeddings, 
                               '(b p) t h -> b (p t) h', 
                               b=B)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # [b, 1+pt, h]
        embeddings = self.pos_drop(embeddings)

        # Transformer layers
        for i, blk in enumerate(self.blocks):
            embeddings = blk(embeddings, register_blk == i)
        last_hidden_state = self.norm(embeddings)
        # [b, 1+pt, h]
        pooled_output = last_hidden_state[:,0]
        # [b, h]

        return pooled_output, last_hidden_state
    
    def forward(self, x, register_blk=-1):
        return self.forward_features(x, register_blk)[0]