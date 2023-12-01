import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model.weight_init import constant_init

from lavis.models.backbones.vit import Attention, Block


class MessageTokenBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        num_frames,
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
    
        self.message_fc = nn.Linear(dim, dim)
        self.message_ln = norm_layer(dim)
        self.message_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        constant_init(self.message_fc, val=0, bias=0)
    
    def forward(self, x, register_hook=False):
        # [(b t) 1+p h]
        msg_token = rearrange(x[:,0], '(b t) h -> b t h', t=self.num_frames)
        msg_token = msg_token + self.message_fc(self.drop_path(self.message_attn(self.message_ln(msg_token))))
        # [b t h]
        msg_token = rearrange(msg_token, 'b t h -> (b t) h').unsqueeze(1)
        x = torch.cat([msg_token, x[:,1:]], dim=1)
        # [(b t) 1+p h]

        x = super().forward(x, register_hook)
        return x