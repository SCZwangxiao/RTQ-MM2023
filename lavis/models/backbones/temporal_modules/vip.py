"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Based on timm code base
 https://github.com/rwightman/pytorch-image-models/tree/master/timm
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from lavis.common.registry import registry
from lavis.models.backbones.vit import Attention, Block


class ViPAttention(Attention):
    def forward(
        self,
        query_states: torch.Tensor,
        hidden_states: torch.Tensor,
        register_hook=False
    ):
        B, N_q, C = query_states.shape
        B, N_kv, C = hidden_states.shape

        w_q, w_k, w_v = torch.unbind(self.qkv.weight.reshape(3, C, C))
        b_q, b_k, b_v = torch.unbind(self.qkv.bias.reshape(3, C))

        q = query_states @ w_q.t() + b_q
        k = hidden_states @ w_k.t() + b_k
        v = hidden_states @ w_v.t() + b_v
        q = q.reshape(B, N_q, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, N_kv, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N_kv, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            # Will cause error: `Expected to mark a variable ready only once.`
            # attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CLIPViPBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        num_frames,
        num_proxies,
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
        self.num_proxies = num_proxies
        self.attn = ViPAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        # Will cause error: `Expected to mark a variable ready only once.`
        # if use_grad_checkpointing:
        #     self.attn = checkpoint_wrapper(self.attn)
    
    def _attn(self, hidden_states, register_hook):
        # [b, m+pt, h]
        b, pt, h = hidden_states.size()
        m = self.num_proxies
        pt -= m
        p, t = pt // self.num_frames, self.num_frames

        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        vip_tokens = hidden_states[:,:m]
        # [b, m, h]
        hidden_states_s = hidden_states[:,m:]
        # [b, pt, h]

        # attention between each frame
        hidden_states_s_q = rearrange(hidden_states_s,
                                      'b (p t) h -> (b t) p h',
                                      t=t)
        hidden_states_s_kv = torch.cat([
            repeat(vip_tokens,
                   'b m h -> (b t) m h',
                   t=t),
            hidden_states_s_q],
            dim=1)
        # [bt, m+p, h]
        hidden_states_s = self.attn(
            query_states=hidden_states_s_q,
            hidden_states=hidden_states_s_kv,
            register_hook=register_hook,
        )
        # [bt, p, h]
        hidden_states_s = rearrange(hidden_states_s,
                                    '(b t) p h -> b (p t) h',
                                    t=t)
        
        # attention between frames using video proxies
        hidden_states_vip = self.attn(
            query_states=vip_tokens,
            hidden_states=hidden_states,
            register_hook=register_hook,
        )
        # [b, m, h]

        # Concatenation
        hidden_states = torch.cat([
            hidden_states_vip,
            hidden_states_s],
            dim=1)
        # [b, m+pt, h]
        hidden_states = residual + self.drop_path(hidden_states.contiguous())

        return hidden_states
    
    def _ffn(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
    
    def forward(self, hidden_states, register_hook=False):
        # [(b t_o) p h]
        # self.num_proxies
        embeddings = rearrange(embeddings, 
                               '(b t) p h -> b p t h', 
                               t=self.num_frames)
        cls_token = embeddings[:,:1].mean(2).repeat(1, self.num_proxies, 1)
        # [b m h]
        x = rearrange(embeddings[:,1:],
                      'b p t h -> b (p t) h')
        embeddings = torch.cat([cls_token, x], dim=1)
        # [b m+pt_o h]

        # [b, m+pt, h]
        hidden_states = self._attn(hidden_states, register_hook)
        # [b, m+pt, h]

        hidden_states = self._ffn(hidden_states)
        # [b, m+pt, h]

        return hidden_states