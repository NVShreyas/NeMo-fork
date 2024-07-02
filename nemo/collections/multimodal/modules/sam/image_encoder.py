import torch
import torch.functional as F

from nemo.collections.nlp.modules.common.megatron.module import Float16Module, MegatronModule
from nemo.collections.vision.modules.vit.vit_backbone import VitBackbone
from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal, scaled_init_method_normal
from nemo.collections.nlp.modules.common.megatron.fused_layer_norm import get_layer_norm

# adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
class SAMVisionTransformer(MegatronModule):
    """Placeholder for ViT-H from SAM.
    Temporarily placed here.
    # TODO: This isn't necessary if we add a SAM-head type post process plugin to ViTbackbone.
    """
    def __init__(self, 
                 model_cfg, 
                 model_parallel_config, 
                 pre_process=True, 
                 post_process=True, 
                 skip_head=False):
        super(SAMVisionTransformer, self).__init__()
        scaled_init_method = (
            scaled_init_method_normal(model_cfg.init_method_std, model_cfg.num_layers)
            if model_cfg.use_scaled_init_method
            else init_method_normal(model_cfg.init_method_std)
        )
        self.config = model_parallel_config
        self.hidden_size = model_cfg.hidden_size
        self.output_dim = model_cfg.output_dim
        self.global_average_pool = model_cfg.global_average_pool
        self.pre_process = pre_process
        self.post_process = post_process
        self.skip_head = skip_head
        self.frozen = False

        assert model_cfg.get("preprocess_layernorm", False), "SAM ViT must use preprocess layernorm = True."

        self.backbone = VitBackbone(model_cfg=model_cfg,
                                    model_parallel_config=model_parallel_config,
                                    init_method=init_method_normal(model_cfg.init_method_std),
                                    scaled_init_method=scaled_init_method,
                                    pre_process=pre_process,
                                    post_process=post_process,
                                    class_token=False,
                                    single_token_output=False,)
        if self.post_process:
            # TODO: make sure this nn.LayerNorm == LayerNorm2D used in LISA.
            self.norm1 = get_layer_norm(self.outout_dim,
                                    model_cfg.layernorm_epsilon,
                                    model_cfg.persist_layer_norm,
                                    sequence_parallel=model_cfg.sequence_parallel)
            self.norm2 = get_layer_norm(self.outout_dim,
                                    model_cfg.layernorm_epsilon,
                                    model_cfg.persist_layer_norm,
                                    sequence_parallel=model_cfg.sequence_parallel)
            self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.hidden_size,
                self.output_dim,
                kernel_size=1,
                bias=False,
            ),
            self.norm1,
            torch.nn.Conv2d(
                self.output_dim,
                self.output_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            self.norm2,
        )
    
    def train(self, mode):
        if self.frozen:
            return self
        
        super().train(mode)
        return self

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        self.frozen = True

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.backbone.set_input_tensor(input_tensor)
    
    def forward(self, input):
        hidden_states = self.backbone(input)

        if self.post_process and not self.skip_head:
            if self.global_average_pool:
                hidden_states = hidden_states.mean(dim=1)
            else:
                hidden_states = hidden_states[:, 0]
            hidden_states = self.head(hidden_states)
        # print("vision_head", hidden_states.shape)
        return hidden_states