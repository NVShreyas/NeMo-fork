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

try:
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel

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
            masks, original_size, mode="bilinear", align_corners=False
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
        elif model_config.precision in [32, '32', '32-true']:
            self.dtype = torch.float32
        else:
            raise ValueError(f"Cannot recognize precision {model_config.precision}")

        if model_config.mm_cfg.llm.freeze:
            self.freeze_llm(model_config.mm_cfg)

        # HACK: since we need hidden_states from GPT so we need this to convert hidden states to logits for loss
        self.share_embeddings_and_output_weights = model_config.share_embeddings_and_output_weights
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
        # seg_labels = kwargs.pop("seg_labels", None)
        gt_masks_labels = kwargs.pop("masks", None)

        neva_output = super().forward(*args, **kwargs)
        # NV LISA Example
        ## seg_mask shape: [1, 384]
        ## neva_output shape: [384, 1, 4096]
        # LISA example
        ## llava output: [6, 447, 5120]
        ## seg_mask: [6, 447]
        gpt_output_layer_weight = None
        if self.share_embeddings_and_output_weights:
            gpt_output_layer_weight = self.decoder.embedding.word_embeddings.weight
        gpt_logits, _ = self.output_layer(neva_output, weight=gpt_output_layer_weight)

        neva_output = torch.transpose(neva_output, 0, 1)
        hidden_states = []
        # make sure we are getting the hidden states from the last PP stage in MCore.
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
                original_size=gt_masks_labels[i].shape[1:],
            )
            pred_masks.append(pred_mask[:, 0])

        if gt_masks_labels != None and "labels" in kwargs:
            # last PP stage or inference
            return self.compute_losses(pred_masks, gpt_logits, kwargs["labels"], gt_masks_labels)

        pred_masks = torch.stack(pred_masks, dim=0)
        return gpt_logits.transpose(0, 1).contiguous()#, pred_masks

    def compute_losses(self, 
                       pred_masks: List[torch.Tensor], 
                       gpt_logits: torch.Tensor,
                       gpt_labels: torch.Tensor,
                       seg_labels: torch.Tensor,
                       lm_loss_weight=1.0,
                       dice_loss_weight=0.5,
                       bce_loss_weight=2.0):

        lm_loss = language_model_loss(gpt_labels, gpt_logits) * lm_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        # TODO: vectorize
        for i in range(len(pred_masks)):
            seg_label = seg_labels[i]
            pred_mask = pred_masks[i]

            assert (
                seg_label.shape[0] == pred_mask.shape[0]
            ), f"seg_label.shape: {seg_label.shape}, pred_mask.shape: {pred_mask.shape}"

            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, seg_label, num_masks=seg_label.shape[0])
                * seg_label.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, seg_label, num_masks=seg_label.shape[0])
                * seg_label.shape[0]
            )
            num_masks += seg_label.shape[0]

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
        seg_token_id = self.tokenizer.token_to_id("[SEG]")

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
                seg_labels, 
                resizes):
        
        forward_args = {
                'input_ids': tokens,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'media': media,
                "images": images,
                "masks": masks,
                # "seg_labels": seg_labels,
                "resize": resizes,
                "offset": offsets,
            }
        outputs = self.model(**forward_args)
        return outputs

    
    # TODO: probably have to override cause of new losses
    def training_step(self, dataloader_iter):
        return super().training_step(dataloader_iter)

    def validation_step(self, dataloader_iter):
        return super().validation_step(dataloader_iter)
    
    def setup(self, stage=None):
        # This sets up datasets internally
        super().setup(stage)
    
    def build_train_valid_test_datasets(self):
        logging.info("Building LISA datasets - only reason seg dataset supported for now.")

        image_processor = self.model.module.image_processor if hasattr(self.model, "module") else self.model.image_processor

        self._train_ds = ReasonSegDataset(base_image_dir=self.cfg.data.image_folder,
                            tokenizer=self.tokenizer,
                            image_processor=image_processor,
                            template_type=self.cfg.data.conv_template,
                            patch_dim=self.cfg.mm_cfg.vision_encoder.patch_dim,
                            mm_mlp_adapter_type=self.cfg.mm_cfg.mm_mlp_adapter_type,
                            add_extra_token=1,
                            context_length=self.cfg.encoder_seq_length,
                            samples_per_epoch=self.cfg.micro_batch_size,
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
                            samples_per_epoch=self.cfg.micro_batch_size,
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
                # "seg_labels": batch["seg_labels"],
                "resize": batch["resizes"],
                "offset": batch["offsets"],
            }

            output_tensor = model(**forward_args)

            return output_tensor, partial(loss_func, loss_mask=batch['loss_mask'])
        
        return fwd_output_and_loss_func