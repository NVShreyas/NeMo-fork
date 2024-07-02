from functools import partial
from typing import Tuple
import torch
import torch.nn as nn
import torch.functional as F

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal, scaled_init_method_normal
from nemo.collections.multimodal.modules.sam.image_encoder import ImageEncoderViT
from nemo.collections.multimodal.modules.sam.prompt_encoder import PromptEncoder
from nemo.collections.multimodal.modules.sam.mask_decoder import MaskDecoder
from nemo.collections.multimodal.modules.sam.two_way_transformer import TwoWayTransformer
from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MCoreNevaModel, MegatronNevaModel
from nemo.collections.nlp.models.nlp_model import NLPModel

try:
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class SAM:
    """
    Frozen variant of SAM with helper functions.
    Mask decoder is trainable.
    """
    #NOTE: this should take model parallel config as well.
    def __init__(self, sam_cfg):
        self.sam_cfg = sam_cfg

        self.image_encoder = ImageEncoderViT(
            depth=sam_cfg.encoder.image_encoder_depth,
            embed_dim=sam_cfg.encoder.encoder_embed_dim,
            img_size=sam_cfg.encoder.image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=sam_cfg.encoder.encoder_num_heads,
            patch_size=sam_cfg.encoder.vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=sam_cfg.encoder.encoder_global_attn_indexes,
            window_size=sam_cfg.encoder.attention_window_size,
            out_chans=sam_cfg.encoder.prompt_embed_dim,
        )
        image_embedding_size = sam_cfg.encoder.image_size // sam_cfg.encoder.vit_patch_size
        self.prompt_encoder = PromptEncoder(embed_dim=sam_cfg.encoder.prompt_embed_dim, 
                                            image_embedding_size=(image_embedding_size, image_embedding_size), 
                                            input_image_size=(sam_cfg.encoder.image_size, sam_cfg.encoder.image_size), 
                                            mask_in_chans=16)
        self.mask_decoder = MaskDecoder(
                        num_multimask_outputs=3,
                        transformer=TwoWayTransformer(
                            depth=sam_cfg.decoder.depth,
                            embedding_dim=sam_cfg.encoder.prompt_embed_dim,
                            mlp_dim=sam_cfg.decoder.mlp_dim,
                            num_heads=sam_cfg.decoder.num_heads,
                        ),
                        transformer_dim=sam_cfg.encoder.prompt_embed_dim,
                        iou_head_depth=sam_cfg.decoder.iou_head_depth,
                        iou_head_hidden_dim=sam_cfg.decoder.iou_head_hidden_dim,
                    )
    
    def freeze(self, mm_cfg):
        if mm_cfg.vision_encoder.freeze:
            self.image_encoder.freeze()
            self.prompt_encoder.freeze()
        if mm_cfg.vision_decoder.freeze:
            self.mask_decoder.freeze()
    
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """Image to embeddings using ViT."""
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """

        masks = F.interpolate(
            masks.float(),
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks


class LisaBaseModel:
    def __init__(self, model_config):
        mm_cfg = model_config.mm_cfg
        # sam_cfg = NLPModel.restore_from(mm_cfg.sam.from_pretrained, return_config=True)
        assert "sam_extra_args" in mm_cfg
        self.sam = SAM(mm_cfg.sam_extra_args)
        if mm_cfg.sam_extra_args.from_pretrained is not None:
            # NOTE: if we want to load encoder and decoder separately then change this
            self.load_sam_weights(self.sam, mm_cfg.sam_extra_args.from_pretrained)
        self.sam.freeze(mm_cfg)

        # Projection layer
        in_dim = model_config.hidden_size
        out_dim = model_config.mm_cfg.sam_extra_args.encoder.prompt_embed_dim #TODO: what is this?
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        
    def load_sam_weights(self, sam, nemo_path: str):
        state_dict = self._load_model_weights(nemo_path)
        print(state_dict)


class MCoreLisaModel(MCoreNevaModel):
    def __init__(self, 
                 model_config,
                 media_start_id,
                 media_end_id,
                 mcore_gpt,
                 **kwargs):
        super(MCoreLisaModel, self).__init__(model_config.mm_cfg,
                                            media_start_id,
                                            media_end_id,
                                            mcore_gpt, 
                                            **kwargs)
        # self.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.model = LisaBaseModel(model_config, **kwargs)

    def forward(self, *args, **kwargs):
        # images: torch.FloatTensor,
        # images_clip: torch.FloatTensor,
        # input_ids: torch.LongTensor,
        # labels: torch.LongTensor,
        # attention_masks: torch.LongTensor,
        # offset: torch.LongTensor,
        # masks_list: List[torch.FloatTensor],
        # label_list: List[torch.Tensor],
        # resize_list: List[tuple],

        images = kwargs.get('media', None)
        input_ids = kwargs.get('input_ids', None)
        image_embeddings = self.model.sam.get_visual_embs(images)

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx # TODO
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().to(images.device),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().to(images.device), seg_token_mask],
            dim=1,
        )

        images_clip_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_clip_i = (
                images_clip[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_clip_list.append(images_clip_i)
        images_clip = torch.cat(images_clip_list, dim=0)

        neva_output = super().forward(*args, **kwargs)

        hidden_states = []
        hidden_states.append(self.model.text_hidden_fcs[0](neva_output[-1]))
        
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]

        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().to(images.device), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        # TODO: why loop if it happens for every sample in batch?
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.sam.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])
        
        # return pred_masks if inference
        # else calculate loss