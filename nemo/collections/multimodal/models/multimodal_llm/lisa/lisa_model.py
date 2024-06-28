import torch
import torch.functional as F

from nemo.collections.nlp.modules.common.megatron.module import Float16Module, MegatronModule
from nemo.collections.vision.modules.vit.vit_backbone import VitBackbone
from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal, scaled_init_method_normal
from nemo.collections.multimodal.modules.sam.image_encoder import SAMVisionTransformer

class FrozenSAMVisionTransformer(SAMVisionTransformer):
    """Frozen version of the SAM ViT-H."""

    def __init__(self, model_cfg, model_parallel_config, pre_process=True, post_process=True):
        super().__init__(
            model_cfg,
            model_parallel_config,
            pre_process=pre_process,
            post_process=post_process,
            skip_head=False,
        )
        self.frozen = False
        self.dtype = self.config.params_dtype
    
    def train(self, mode):
        if self.frozen:
            return self
        
        super().train(mode)
        return self
    
    def forward(self, input):
        assert self.training == False
        image_embeddings = self.backbone(input)
        if self.post_process and not self.skip_head:
            image_embeddings = self.head(image_embeddings) # TODO: correct shapes before head
        return image_embeddings
    
    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        self.frozen = True

