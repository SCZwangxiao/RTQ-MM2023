import torch.nn as nn
from einops import rearrange, reduce

from lavis.models import BaseModel
from lavis.common.registry import registry


@registry.register_model("blip_post_mean_adapter")
class BlipPostMeanPooler(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {}
    def __init__(self, num_frames) -> None:
        super().__init__()
        self.num_frames = num_frames
    
    def forward(self, image_embeds):
        # ViT output [bt 1+p d]
        image_embeds = reduce(image_embeds,
                              '(bt) p d -> b p d',
                              'mean',
                               t=self.num_frames)
        # [b 1+p d]
        video_outputs = image_embeds
        video_embeds = image_embeds[:,0]
        return video_embeds, video_outputs


@registry.register_model("blip_post_concat_adapter")
class BlipPostConcatPooler(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {}
    def __init__(self, num_frames) -> None:
        super().__init__()
        self.num_frames = num_frames
    
    def forward(self, image_embeds):
        T = self.num_frames
        B = image_embeds.shape[0] // T
        image_embeds = image_embeds.reshape((B, T) + image_embeds.shape[1:])
        # [b t p d]
        video_embeds = image_embeds[:,:,0].mean(1)
        video_outputs = rearrange(
            image_embeds,
            'b t p h -> b (t p) h'
        )
        
        return video_embeds, video_outputs