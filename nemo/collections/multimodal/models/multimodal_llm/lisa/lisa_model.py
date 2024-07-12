from functools import partial
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf.dictconfig import DictConfig
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
from nemo.collections.vision.data.megatron.data_samplers import MegatronVisionPretrainingRandomSampler
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal
from nemo.collections.multimodal.modules.sam.image_encoder import ImageEncoderViT
from nemo.collections.multimodal.modules.sam.prompt_encoder import PromptEncoder
from nemo.collections.multimodal.modules.sam.mask_decoder import MaskDecoder
from nemo.collections.multimodal.modules.sam.two_way_transformer import TwoWayTransformer
from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MCoreNevaModel, MegatronNevaModel
from nemo.collections.multimodal.data.neva.conversation import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from nemo.collections.multimodal.data.lisa.utils import DEFAULT_SEG_TOKEN
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import get_specs
from nemo.utils import logging
from nemo.collections.multimodal.data.lisa.lisa_dataset import ReasonSegDataset, DataCollatorForSegmentationDataset
from nemo.collections.multimodal.parts.utils import load_nemo_model_weights
from nemo.collections.multimodal.losses.segmentation_losses import dice_loss, sigmoid_ce_loss
from nemo.collections.nlp.losses.lm_loss import language_model_loss
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, OutputType, SamplingParam
from nemo.collections.multimodal.data.neva.neva_dataset import tokenize
# from nemo.collections.multimodal.data.lisa.utils import (
#     get_mask_from_json, 
#     ANSWER_LIST, 
#     DEFAULT_IMAGE_TOKEN,
#     EXPLANATORY_QUESTION_LIST, 
#     LONG_QUESTION_LIST,
#     SHORT_QUESTION_LIST,
#     DEFAULT_IM_END_TOKEN,
#     DEFAULT_IM_START_TOKEN,
#     IGNORE_INDEX,
#     DEFAULT_IMAGE_PATCH_TOKEN
# )
from nemo.collections.multimodal.data.neva import conversation as conversation_lib
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

try:
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.custom_layers.transformer_engine import TENorm
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class SAM(nn.Module):
    """
    Frozen variant of SAM with helper functions.
    Mask decoder is trainable.
    """
    #NOTE: this should take model parallel config as well.
    def __init__(self, model_config):
        super().__init__()
        sam_cfg = model_config.mm_cfg.sam_extra_args

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
                        model_config=model_config,
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
    
    def log_params(self):
        logging.info(
            f"Initialized SAM model with {sum(p.numel() for p in self.parameters())} parameters\n"
            f"SAM image encoder: {sum(p.numel() for p in self.image_encoder.parameters())} parameters. Frozen: {self.image_encoder.frozen}.\n"
            f"SAM prompt encoder: {sum(p.numel() for p in self.prompt_encoder.parameters())} parameters. Frozen: {self.prompt_encoder.frozen}.\n"
            f"SAM mask decoder: {sum(p.numel() for p in self.mask_decoder.parameters())} parameters. Frozen: {self.mask_decoder.frozen}.\n"
        )
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: torch.Tensor,
        original_size: torch.Tensor,
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
        if len(input_size.shape) > 1:
            input_size = input_size.flatten()
        assert input_size.shape == torch.Size([2])
        input_size = input_size.int()

        masks = F.interpolate(
            masks.float(),
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        masks = masks[..., : input_size[0].item(), : input_size[1].item()]
        masks = F.interpolate(
            masks, [original_size[0].item(), original_size[1].item()], mode="bilinear", align_corners=False
        )
        return masks


class LisaBaseModel(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        mm_cfg = model_config.mm_cfg
        # sam_cfg = NLPModel.restore_from(mm_cfg.sam.from_pretrained, return_config=True)
        assert "sam_extra_args" in mm_cfg
        self.sam = SAM(model_config)

        if mm_cfg.sam_extra_args.from_pretrained != "":
            # NOTE: if we want to load encoder and decoder separately then change this
            self.load_sam_weights(self.sam, mm_cfg.sam_extra_args.from_pretrained)
        self.sam.freeze(mm_cfg)

        self.sam.log_params()

        # Projection layer
        in_dim = model_config.hidden_size
        out_dim = model_config.mm_cfg.sam_extra_args.encoder.prompt_embed_dim
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
    
    def _load_model_weights(self, nemo_path):
        """
        Shared method to load model weights from a given nemo_path.
        """
        sharded_state_dict = None
        if getattr(self, "sharded_state_dict", None) is not None:
            sharded_state_dict = self.sharded_state_dict(prefix="model.")
        state_dict, self.is_dist_ckpt = load_nemo_model_weights(nemo_path, sharded_state_dict)

        return state_dict

    def load_sam_weights(self, sam, nemo_path: str):
        state_dict = self._load_model_weights(nemo_path)
        print(state_dict)

    def sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = (), **kwargs):
        state_dict = self.state_dict(prefix='', keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(state_dict, prefix=prefix)
        return sharded_state_dict

class MCoreLisaModel(MCoreNevaModel):
    def __init__(self, 
                 model_config,
                 media_start_id,
                 media_end_id,
                 seg_token_id,
                 mcore_gpt,
                 **kwargs):
        MCoreNevaModel.__init__(self, model_config.mm_cfg,
                                        media_start_id,
                                        media_end_id,
                                        mcore_gpt,
                                        **kwargs)

        self.lisa_sam = LisaBaseModel(model_config, **kwargs)
        self.seg_token_id = seg_token_id
        
        if model_config.precision in ['bf16', 'bf16-mixed']:
            self.dtype = torch.bfloat16
        elif model_config.precision in [16, '16', '16-mixed']:
            self.dtype = torch.float16
        elif model_config.precision in [32, '32', '32-true', 'fp32']:
            self.dtype = torch.float32
        else:
            raise ValueError(f"Cannot recognize precision {model_config.precision}")

        if model_config.mm_cfg.llm.freeze:
            self.freeze_llm(model_config.mm_cfg)

        # HACK: since we need hidden_states from GPT so we need this to convert hidden states to logits for loss
        # Post-process was disabled in megatronGPT model, so we need to add these 2 layers back
        self.share_embeddings_and_output_weights = model_config.share_embeddings_and_output_weights
        self.final_layernorm = TENorm(
                config=kwargs["config"],
                hidden_size=model_config.hidden_size,
                eps=kwargs["config"].layernorm_epsilon,
            )
        self.output_layer = tensor_parallel.ColumnParallelLinear(
                model_config.hidden_size,
                kwargs["vocab_size"],
                config=kwargs["config"],
                init_method=init_method_normal(model_config.init_method_std),
                bias=False,
                skip_bias_add=False,
                gather_output=False,
                skip_weight_param_allocation=model_config.pre_process and model_config.share_embeddings_and_output_weights,
                embedding_activation_buffer=None,
                grad_output_buffer=None,
            )

    def model_forward(self, *args, **kwargs):
        images = kwargs.pop('images', None)
        input_ids = kwargs.get('input_ids', None)
        image_embeddings = self.lisa_sam.sam.get_visual_embs(images)
        if image_embeddings.dim() == 3:
            # WHY?
            image_embeddings = image_embeddings.unsqueeze(0)
        # LISA Dim: torch.Size([2, 256, 64, 64])

        seg_token_mask = input_ids[:, 1:] == self.seg_token_id
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().to(images.device),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)

        # ANMOL: Input IDs already have image tokens (set as 0) inserted
        # This could be an issue if we run inference using pretrained model.
        # seg_token_mask = torch.cat(
        #     [torch.zeros((seg_token_mask.shape[0], 255)).bool().to(images.device), seg_token_mask],
        #     dim=1,
        # )

        images_clip = kwargs.get("media", None)
        offset = kwargs.pop("offset", None)
        images_clip_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_clip_i = (
                images_clip[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1, -1, -1)
                .contiguous()
            )
            images_clip_list.append(images_clip_i)
        images_clip = torch.cat(images_clip_list, dim=0)
        kwargs["media"] = images_clip

        resizes = kwargs.pop("resize", None)
        mask_shapes = kwargs.pop("mask_shapes", None)
        gt_masks_labels = kwargs.pop("masks", None)
        neva_output = super().forward(*args, **kwargs)

        neva_output = self.final_layernorm(neva_output)
        gpt_output_layer_weight = None
        if self.share_embeddings_and_output_weights:
            gpt_output_layer_weight = self.decoder.embedding.word_embeddings.weight
        gpt_logits, _ = self.output_layer(neva_output, weight=gpt_output_layer_weight)
        #import pdb; pdb.set_trace()
        neva_output = torch.transpose(neva_output, 0, 1)
        hidden_states = []
        # make sure we are getting the hidden states from the last PP stage in MCore.
        #import pdb; pdb.set_trace()
        hidden_states.append(self.lisa_sam.text_hidden_fcs[0](neva_output))
        
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        
        # embedding corresponding to seg token -> [b*n_s, 256]
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
        # pred_embeddings a list with element shape: torch.Size([3, 256])
        multimask_output = False
        pred_masks = []
        
        # TODO: why loop if it happens for every sample in batch?
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.lisa_sam.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.lisa_sam.sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.lisa_sam.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            pred_mask = self.lisa_sam.sam.postprocess_masks(
                low_res_masks,
                input_size=resizes[i],
                original_size=mask_shapes[i],
            )
            pred_masks.append(pred_mask[:, 0])

        if gt_masks_labels != None and "labels" in kwargs:
            # last PP stage or not inference
            return self.compute_losses(pred_masks, gpt_logits, kwargs["labels"], gt_masks_labels, mask_shapes)

        pred_masks = torch.stack(pred_masks, dim=0)
        return gpt_logits.transpose(0, 1).contiguous()#, pred_masks

    def compute_losses(self, 
                       pred_masks: List[torch.Tensor], 
                       gpt_logits: torch.Tensor,
                       gpt_labels: torch.Tensor,
                       gt_masks: torch.Tensor,
                       mask_shapes: torch.Tensor,
                       lm_loss_weight=1.0,
                       dice_loss_weight=0.5,
                       bce_loss_weight=2.0):

        lm_loss = 0
        #language_model_loss(gpt_labels, gpt_logits) * lm_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        for i in range(len(pred_masks)):
            gt_mask = gt_masks[i]
            pred_mask = pred_masks[i]

            # NOTE: due to this, it is not possible to vectorize
            # this is required as some GPT responses don't contain [SEG] and loss is skipped.
            if (gt_mask == -1).all():
                continue
            gt_mask = gt_mask[gt_mask != -1]
            gt_mask = gt_mask.reshape([1, mask_shapes[i][0].item(), mask_shapes[i][1].item()])

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), f"gt_mask.shape: {gt_mask.shape}, pred_mask.shape: {pred_mask.shape}"

            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = lm_loss + mask_loss

        return loss


    def forward(self, *args, **kwargs):
        if self.dtype == torch.float32:
            return self.model_forward(*args, **kwargs)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return self.model_forward(*args, **kwargs)


class MegatronLisaModel(MegatronNevaModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        print(cfg)
    
    def model_provider_func(self, pre_process, post_process):
        llm_type = self.cfg.mm_cfg.llm.get("model_type", "nvgpt")
        media_start_id = self.tokenizer.token_to_id(DEFAULT_IM_START_TOKEN[llm_type])
        media_end_id = self.tokenizer.token_to_id(DEFAULT_IM_END_TOKEN[llm_type])
        seg_token_id = self.tokenizer.token_to_id(DEFAULT_SEG_TOKEN)

        # Needed while initializing the model to map it correctly.
        media_start_id = 32001
        media_end_id = 32002
        seg_token_id = 32003
        if not parallel_state.is_initialized():
            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

        # need to override this since MCoreGPTModel returns loss or logits if pp last stage.
        # We need the hidden states to input into the SAM decoder
        # Caveat: Calculate pplx loss in LISA
        # Possible solution: update mcore gpt to return everything
        post_process = False

        model = MCoreLisaModel(
            model_config=self.cfg,
            media_start_id=media_start_id,
            media_end_id=media_end_id,
            mcore_gpt=True,
            seg_token_id=seg_token_id,
            # need the following for Neva
            config=self.transformer_config,
            transformer_layer_spec=get_specs(self.spec_name),
            vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
            max_sequence_length=self.cfg.get('encoder_seq_length', 512),
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
            position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
            rotary_percent=self.cfg.get('rotary_percentage', 1.0),
            seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
            rotary_base=self.cfg.get('rotary_base', 10000),
        )
        logging.info(
            f"Lisa model initialized with {sum(p.numel() for p in model.parameters())} total parameters"
        )
        logging.info(
            f"Lisa model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )

        return model
    
    def setup_optimizer_param_groups(self):
        super().setup_optimizer_param_groups()

    def forward(self, images, 
                media, 
                tokens,
                position_ids,
                labels, 
                attention_mask,
                offsets, 
                masks, 
                mask_shapes,
                resizes):
        
        forward_args = {
                'input_ids': tokens,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'media': media,
                "images": images,
                "masks": masks,
                "mask_shapes": mask_shapes,
                "resize": resizes,
                "offset": offsets,
            }
        outputs = self.model(**forward_args)
        return outputs

    
    def training_step(self, dataloader_iter):
        return super().training_step(dataloader_iter)

    def validation_step(self, dataloader_iter):
        return super().validation_step(dataloader_iter)
    
    def setup(self, stage=None):
        super().setup(stage)
    
    def build_train_valid_test_datasets(self):
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        
        logging.info("Building LISA datasets - only reason seg dataset supported for now.")
        global_batch_size = self.cfg.global_batch_size
        max_train_steps = self.trainer.max_steps
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
        ]

        image_processor = self.model.module.image_processor if hasattr(self.model, "module") else self.model.image_processor

        self._train_ds = ReasonSegDataset(base_image_dir=self.cfg.data.image_folder,
                            tokenizer=self.tokenizer,
                            image_processor=image_processor,
                            template_type=self.cfg.data.conv_template,
                            patch_dim=self.cfg.mm_cfg.vision_encoder.patch_dim,
                            mm_mlp_adapter_type=self.cfg.mm_cfg.mm_mlp_adapter_type,
                            add_extra_token=1,
                            context_length=self.cfg.encoder_seq_length,
                            samples_per_epoch=train_valid_test_num_samples[0],
                            image_size=self.cfg.mm_cfg.sam_extra_args.encoder.image_size,
                            reason_seg_data="reasonseg|train")
        
        self._validation_ds = ReasonSegDataset(base_image_dir=self.cfg.data.image_folder,
                            tokenizer=self.tokenizer,
                            image_processor=image_processor,
                            template_type=self.cfg.data.conv_template,
                            patch_dim=self.cfg.mm_cfg.vision_encoder.patch_dim,
                            mm_mlp_adapter_type=self.cfg.mm_cfg.mm_mlp_adapter_type,
                            add_extra_token=1,
                            context_length=self.cfg.encoder_seq_length,
                            samples_per_epoch=train_valid_test_num_samples[1],
                            image_size=self.cfg.mm_cfg.sam_extra_args.encoder.image_size,
                            reason_seg_data="reasonseg|val",
                            explanatory=-1)
        
        return self._train_ds, self._validation_ds

    def build_pretraining_data_loader(
        self, dataset, consumed_samples, dataset_type=None, drop_last=True, pad_samples_to_global_batch_size=False
    ):
        """Buld dataloader given an input dataset."""

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            micro_batch_size = self.cfg.micro_batch_size
        else:
            micro_batch_size = self.cfg.global_batch_size // parallel_state.get_data_parallel_world_size()

        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                    global_batch_size=self.cfg.global_batch_size,
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronVisionPretrainingRandomSampler(
                    dataset=dataset,
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                    data_sharding=False,
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        collate_func = DataCollatorForSegmentationDataset(self.cfg, self.tokenizer)
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_func,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

    def get_forward_output_and_loss_func(self, validation_step=False, tuning=False):
        def loss_func(output_tensor, loss_mask):
            loss_for_ub = self.loss_func(loss_mask, output_tensor)
            if validation_step and not self.cfg.data.get('validation_drop_last', True):
                raise NotImplementedError(f"`validation_drop_last=False` is not implemented in Lisa!")
            else:
                reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                return loss_for_ub, dict(avg=reduced_loss[0].unsqueeze(0))
        
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)
            if isinstance(batch, tuple):
                batch = batch[0]
            
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                for k in batch.keys():
                    if k not in ['offsets']:
                        if self.get_attention_mask_from_fusion:
                            batch[k] = batch[k].cuda(non_blocking=True) if k not in ['attention_mask'] else None
                        else:
                            batch[k] = batch[k].cuda(non_blocking=True)
            else:
                raise NotImplementedError

            forward_args = {
                'input_ids': batch['tokens'],
                'position_ids': batch['position_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels'],
                'media': batch.get('media', None),
                "images": batch["images"],
                "masks": batch["masks"],
                "mask_shapes": batch["mask_shapes"],
                "resize": batch["resizes"],
                "offset": batch["offsets"],
            }

            output_tensor = model(**forward_args)

            return output_tensor, partial(loss_func, loss_mask=batch['loss_mask'])
        
        return fwd_output_and_loss_func
    
    def generate(
        self,
        input_prompts,
        inference_config,
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
    ) -> OutputType:
        IGNORE_INDEX = -1
        IMAGE_TOKEN_INDEX = 32000
        IMAGE_TOKEN = "<image>"
        # Updated to follow neva
        # DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
        # DEFAULT_IM_START_TOKEN = "<im_start>"
        # DEFAULT_IM_END_TOKEN = "<im_end>"
        IMAGE_PATCH_TOKEN = "<extra_id_0>" #defaultdict(lambda: "<extra_id_3>")
        IM_START_TOKEN = "<extra_id_1>" #defaultdict(lambda: "<extra_id_4>")
        IM_END_TOKEN = "<extra_id_2>" #defaultdict(lambda: "<extra_id_5>")
        media_start_id = 32001
        media_end_id = 32002
        # check whether the DDP is initialized
        if not parallel_state.is_initialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

        llm_type = self.cfg.mm_cfg.llm.get("model_type", "nvgpt")
        #llm_type = "v1"
        #seg_token_id = self.tokenizer.token_to_id("[SEG]")
        seg_token_id = self.tokenizer.token_to_id("<extra_id_3>")
        prompt = input_prompts[0]["prompt"]
        image_clip = input_prompts[0]["image_clip"]
        image = input_prompts[0]["image"]
        resize_list = input_prompts[0]['resize_list']
        original_size_list = input_prompts[0]['original_size_list']
        patch_dim=self.cfg.mm_cfg.vision_encoder.patch_dim
        mm_mlp_adapter_type=self.cfg.mm_cfg.mm_mlp_adapter_type
        context_length=self.cfg.encoder_seq_length

        height_num_patches = image_clip.shape[4] // patch_dim
        width_num_patches = image_clip.shape[5] // patch_dim

        if mm_mlp_adapter_type == 'mlp_downsample':
            if height_num_patches % 2 != 0:
                height_num_patches += 1
            if width_num_patches % 2 != 0:
                width_num_patches += 1
        
        total_num_patches = height_num_patches * width_num_patches
        template_type="v1"
        conv = conversation_lib.conv_templates[template_type].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], '')
        conversation = conv.get_prompt()
        modified_conversations = []
        replace_token = (
            IM_START_TOKEN + IMAGE_PATCH_TOKEN * total_num_patches + IM_END_TOKEN
        )
        #modified_conversations.append(custom_prompts.replace(IMAGE_TOKEN, replace_token))
        modified_conversations.append(conversation.replace(IMAGE_TOKEN, replace_token))
        print(modified_conversations)
        tokens = tokenize(texts=modified_conversations, tokenizer=self.tokenizer, context_length=context_length, add_extra_token=False)
        
        tokens[tokens == 32000] = 0  # DEFAULT_IMAGE_PATCH_TOKEN
        tokens[tokens == 32006] = 1  # <s>
        tokens[tokens == 32007] = 2  # </s>
        #tokens = F.pad(tokens, (0, 1), 'constant', 0)
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                data=tokens,
                eod_token=self.tokenizer.eos_id,
                eod_mask_loss=False,
                reset_attention_mask=False,
                reset_position_ids=False,
            )   
        tokens[tokens == -1] = 0
        exit_llm = False
        args=()
        kwargs={}
        kwargs['input_ids'] = tokens.cuda()
        kwargs['position_ids'] = position_ids.cuda()
        kwargs['attention_mask'] = attention_mask.cuda()
        kwargs['media'] = image_clip.cuda()

        share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True)
        found_seg_token = False
        while (not exit_llm):
            neva_output = super(MCoreLisaModel, self.model).forward(*args, **kwargs)
            neva_output_ln = self.model.final_layernorm(neva_output)
            gpt_output_layer_weight = None
            if share_embeddings_and_output_weights:
                gpt_output_layer_weight = self.model.decoder.embedding.word_embeddings.weight
            gpt_logits, _ = self.model.output_layer(neva_output_ln, weight=gpt_output_layer_weight)

            new_token_id = gpt_logits.squeeze()[-1][:32004].argmax()
            new_token_value = self.tokenizer.ids_to_tokens([new_token_id.item()])
            print(f"Generated Token ID: {new_token_id.item()}, token: {new_token_value[0]}")
            # if found_seg_token:
            #     print(tokens)
            #     print(f"Got the output hidden states for SEG token, breaking from the llm loop")
            #     break
            if int(new_token_id.item()) == 32003:
                print(f"Found SEG token")
                found_seg_token = True
                break

            user_input = input("Enter 0 to exit, 1 to continue: ")
            try:
                user_input = int(user_input)
            except:
                print("Incorrect input. Only 0 and 1 are allowed")
                break
            if user_input:
                tokens = F.pad(tokens, (0, 1), 'constant', new_token_id.item())
                attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                        data=tokens,
                        eod_token=self.tokenizer.eos_id,
                        eod_mask_loss=False,
                        reset_attention_mask=False,
                        reset_position_ids=False,
                    )   
                tokens[tokens == -1] = 0
                args=()
                kwargs={}
                kwargs['input_ids'] = tokens.cuda()
                kwargs['position_ids'] = position_ids.cuda()
                kwargs['attention_mask'] = attention_mask.cuda()
                kwargs['media'] = image_clip.cuda()
            else:
                break
        #return self.model_forward(*args, **kwargs)
        #while (not exit_llm):
        pred_mask = None
        if found_seg_token:
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # lisa_llava_out = torch.load("/lustre/fsw/llmservice_dev_mcore/shreyasm/multimodal/LISA/output_hidden_states_fp32.pt").cuda()
            # print((lisa_llava_out.abs() - neva_output_ln.abs()).abs().mean())
            pred_embedding = self.model.lisa_sam.text_hidden_fcs[0](neva_output_ln)
            pred_embedding = pred_embedding[-1]
            # pred_embedding = torch.load("/lustre/fsw/llmservice_dev_mcore/shreyasm/multimodal/LISA/pred_embeddings_fp32.pt").cuda()
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.lisa_sam.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embedding.unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            image_embeddings = self.model.lisa_sam.sam.get_visual_embs(image)
            
            #LISA tensors - 
            # image_embeddings = torch.load("/lustre/fsw/llmservice_dev_mcore/shreyasm/multimodal/LISA/image_embeddings_fp32.pt").cuda()
            # sparse_embeddings = torch.load("/lustre/fsw/llmservice_dev_mcore/shreyasm/multimodal/LISA/sparse_embeddings_fp32.pt").cuda()
            # dense_embeddings = torch.load("/lustre/fsw/llmservice_dev_mcore/shreyasm/multimodal/LISA/dense_embeddings_fp32.pt").cuda()
            # print(torch.allclose(torch.load("/lustre/fsw/llmservice_dev_mcore/shreyasm/multimodal/LISA/pe_layer_gaus_mat_fp32.pt"), 
            #                      self.model.lisa_sam.sam.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix))

            # dense_pe = torch.load("/lustre/fsw/llmservice_dev_mcore/shreyasm/multimodal/LISA/dense_pe_fp32.pt").cuda()
            # NOTE: this does not work because of dense_pe.
            dense_pe = self.model.lisa_sam.sam.prompt_encoder.get_dense_pe()
            low_res_masks, iou_predictions = self.model.lisa_sam.sam.mask_decoder(
                image_embeddings=image_embeddings,  
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # low_res_mask_lisa = torch.load("/lustre/fsw/llmservice_dev_mcore/shreyasm/multimodal/LISA/low_res_masks_fp32.pt").cuda()
            # print(torch.allclose(low_res_mask_lisa, low_res_masks))

            pred_mask = self.model.lisa_sam.sam.postprocess_masks(
                low_res_masks,
                input_size=resize_list,
                original_size=original_size_list[0].int(),
            )
        return pred_mask
