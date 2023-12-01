import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model.weight_init import constant_init
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from lavis.models.backbones.vit import Attention, Block


class DividedSpaceTimeBlock(Block): 
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
    def forward(self, x, register_hook=False):
        # [(b t) 1+p h]
        B = x.shape[0] // self.num_frames

        cls_token = rearrange(x[:,:1], '(b t) p h -> b t p h', t=self.num_frames)
        # [b t 1 h]
        cls_token = cls_token.mean(1, keepdim=True).repeat(1, self.num_frames, 1, 1)
        cls_token = rearrange(cls_token, 'b t p h -> (b t) p h')
        # [(b t) 1 h]

        x = rearrange(x[:,1:], '(b t) p h -> (b p) t h', b=B)
        x = x + self.temporal_fc(
            self.drop_path(
                self.temporal_attn(self.layer_norm0(x), register_hook).contiguous()
            )
        )
        # [(b p) t h]
        x = rearrange(x, '(b p) t h -> (b t) p h', b=B)
        x = torch.cat([cls_token, x], dim=1)
        # [(b t) 1+p h]

        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # [(b t) 1+p h]

        return x