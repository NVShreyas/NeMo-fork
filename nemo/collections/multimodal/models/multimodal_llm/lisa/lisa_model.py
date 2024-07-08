from functools import partial
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf.dictconfig import DictConfig
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
from nemo.collections.vision.data.megatron.data_samplers import MegatronVisionPretrainingRandomSampler
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal, scaled_init_method_normal
from nemo.collections.multimodal.modules.sam.image_encoder import ImageEncoderViT
from nemo.collections.multimodal.modules.sam.prompt_encoder import PromptEncoder
from nemo.collections.multimodal.modules.sam.mask_decoder import MaskDecoder
from nemo.collections.multimodal.modules.sam.two_way_transformer import TwoWayTransformer
from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MCoreNevaModel, MegatronNevaModel
from nemo.collections.multimodal.data.neva.conversation import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import get_specs
from nemo.utils import logging
from nemo.collections.multimodal.data.lisa.lisa_dataset import ReasonSegDataset, collate_fn
from nemo.collections.multimodal.parts.utils import load_nemo_model_weights

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


class MCoreLisaModel(MCoreGPTModel):
    def __init__(self, 
                 model_config,
                 media_start_id,
                 media_end_id,
                 seg_token_id,
                 mcore_gpt,
                 **kwargs):
        MCoreGPTModel.__init__(self, **kwargs)
        #MCoreNevaModel.__init__(self, model_config.mm_cfg,
        #super().__init__()
        # super(MCoreLisaModel, self).__init__(model_config.mm_cfg,
        #                                     media_start_id,
        #                                     media_end_id,
        #                                     mcore_gpt,
        #                                     **kwargs)
        #LisaBaseModel.__init__(self, model_config, **kwargs)
        self.lisa_sam = LisaBaseModel(model_config, **kwargs)
        self.lisa_neva = MCoreNevaModel(model_config.mm_cfg,
                                        media_start_id,
                                        media_end_id,
                                        mcore_gpt,
                                        **kwargs)
        self.seg_token_id = seg_token_id
        
        if model_config.precision in ['bf16', 'bf16-mixed']:
            self.dtype = torch.bfloat16
        elif model_config.precision in [16, '16', '16-mixed']:
            self.dtype = torch.float16
        elif model_config.precision in [32, '32', '32-true']:
            self.dtype = torch.float32
        else:
            raise ValueError(f"Cannot recognize precision {model_config.precision}")
        
        # HACK: since we need hidden_states from GPT so we need this to convert hidden states to logits for loss
        # self.output_layer = tensor_parallel.ColumnParallelLinear(
        #         self.model_config.hidden_size,
        #         vocab_size=kwargs["vocab_size"],
        #         config=kwargs["config"],
        #         init_method=self.model_config.init_method,
        #         bias=False,
        #         skip_bias_add=False,
        #         gather_output=not self.parallel_output,
        #         skip_weight_param_allocation=self.pre_process
        #         and self.share_embeddings_and_output_weights,
        #         embedding_activation_buffer=self.embedding_activation_buffer,
        #         grad_output_buffer=self.grad_output_buffer,
        #     )

    def model_forward(self, *args, **kwargs):
        images = kwargs.pop('images', None)
        input_ids = kwargs.get('input_ids', None)
        image_embeddings = self.lisa_sam.sam.get_visual_embs(images)

        seg_token_mask = input_ids[:, 1:] == self.seg_token_id
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

        images_clip = kwargs.get("media", None)
        images_clip_list = []
        offset = kwargs.pop("offset", None)
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


        resize_list = kwargs.pop("resize_list", None)
        label_list = kwargs.pop("label_list", None)
        masks_list = kwargs.pop("masks_list", None)

        kwargs["media"] = images_clip

        # This is hidden states from McoreGPT
        neva_output = self.lisa_neva(*args, **kwargs)

        hidden_states = []
        # TODO: verify [-1]
        # Llava in HF returns hiddens states per layer (https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaModel:~:text=up%20sequential%20decoding.-,hidden_states,-(tuple(torch))
        # I think [-1] returns hidden state of last layer in HF models
        # We don't have to do this. We just have to make sure we are getting the hidden states from the last PP stage in MCore.
        hidden_states.append(self.lisa_sam.text_hidden_fcs[0](neva_output[-1]))
        
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
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])
        
        # return pred_masks if inference
        # else calculate loss

    def forward(self, *args, **kwargs):
        # Inputs:
        # images: torch.FloatTensor,
        # images_clip: torch.FloatTensor,
        # input_ids: torch.LongTensor,
        # labels: torch.LongTensor,
        # attention_masks: torch.LongTensor,
        # offset: torch.LongTensor,
        # masks_list: List[torch.FloatTensor],
        # label_list: List[torch.Tensor],
        # resize_list: List[tuple],

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
        if self.cfg.mm_cfg.llm.freeze:
            # this will freeze neva(llm) and clip-vit is already frozen
            # This should also filter out params that don't have grad like SAM encoder
            super().setup_optimizer_param_groups()
        else:
            return NotImplemented

    def forward(self, images, 
                images_clip, 
                tokens,
                # position_ids,
                labels, 
                attention_mask, 
                offset, 
                masks_list, 
                label_list, 
                resize_list):
        
        forward_args = {
            'input_ids': tokens,
            # 'position_ids': position_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'media': images_clip, # since neva uses media and need clip processed
            'offset': offset,
            'masks_list': masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "images": images
        }
        output_tensor = self.model(**forward_args)
        return output_tensor

    
    # TODO: probably have to override cause of new losses
    def training_step(self, dataloader_iter):
        return super().training_step(dataloader_iter)

    # TODO: probably have to override cause of new losses
    def validation_step(self, dataloader_iter):
        return super().validation_step(dataloader_iter)
    
    def setup(self, stage=None):
        #TODO: probably override it. Not sure whether # of params logged in MegatronNevaModel.setup()
        # includes LM + Embedding (including Clip) + SAM + text to vision FC
        # This sets up datasets internally
        super().setup(stage)
    
    def build_train_valid_test_datasets(self):
        logging.info("Building LISA datasets - only reason seg dataset supported for now.")

        image_processor = self.model.module.lisa_neva.image_processor if hasattr(self.model, "module") else self.model.lisa_neva.image_processor

        self._train_ds = ReasonSegDataset(base_image_dir=self.cfg.data.image_folder,
                         tokenizer=self.tokenizer,
                         image_processor=image_processor,
                         samples_per_epoch=self.cfg.micro_batch_size,
                         image_size=self.cfg.mm_cfg.sam_extra_args.encoder.image_size,
                         reason_seg_data="reasonseg|train")
        
        self._validation_ds = ReasonSegDataset(base_image_dir=self.cfg.data.image_folder,
                            tokenizer=self.tokenizer,
                            image_processor=image_processor,
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

        collate_func = partial(
            collate_fn,
            tokenizer=self.tokenizer,
            conv_type=self.cfg.data.conv_template,
            use_mm_start_end=True,
            seq_len=self.cfg.encoder_seq_length,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_func,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

    def get_forward_output_and_loss_func(self, validation_step=False, tuning=False):
        def loss_func(output_tensor):
            return self.loss_func(output_tensor)
        
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)
            if isinstance(batch, tuple):
                batch = batch[0]
            
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                for k in batch.keys():
                    if self.get_attention_mask_from_fusion:
                        batch[k] = batch[k].cuda(non_blocking=True) if k not in ['attention_mask'] else None
                    else:
                        batch[k] = batch[k].cuda(non_blocking=True)
            else:
                raise NotImplementedError

            forward_args = {
                'input_ids': batch['tokens'],
                # 'position_ids': batch['position_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels'],
                'media': batch.get('images_clip', None),
                "images": batch["images"],
                "masks_list": batch["masks_list"],
                "label_list": batch["label_list"],
                "resize_list": batch["resize_list"],
                "offset": batch["offset"],
            }

            output_tensor = model(**forward_args)

            return output_tensor, loss_func
        
        return fwd_output_and_loss_func
    
    def loss_func(self, output_tensor):
        return 0.0