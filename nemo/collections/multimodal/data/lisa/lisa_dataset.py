from dataclasses import dataclass
import glob
import json
import os
import random

import cv2
from einops import rearrange
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
import transformers

from nemo.collections.multimodal.data.neva import conversation as conversation_lib
from nemo.collections.multimodal.data.neva.neva_dataset import tokenize
from nemo.collections.multimodal.data.common.image_transforms import ResizeLongestSide
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

from nemo.collections.multimodal.data.lisa.utils import (
    get_mask_from_json, 
    ANSWER_LIST, 
    DEFAULT_IMAGE_TOKEN,
    EXPLANATORY_QUESTION_LIST, 
    LONG_QUESTION_LIST,
    SHORT_QUESTION_LIST,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN
)    



class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    ignore_label = 255 # for segmentation mask

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        image_processor,
        template_type="llava_llama_2",
        patch_dim=14,
        mm_mlp_adapter_type="mlp_downsample",
        context_length=4096,
        add_extra_token=0,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 224,
        num_classes_per_sample: int = 1,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.patch_dim = patch_dim
        self.mm_mlp_adapter_type = mm_mlp_adapter_type
        self.context_length = context_length
        self.add_extra_token = add_extra_token
        self.template_type = template_type
        assert self.template_type == "llava_llama_2"

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = image_processor

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        reason_seg_data, self.split = reason_seg_data.split("|")
        assert self.split in {"train", "val"}

        images = []
        images_split = glob.glob(
            os.path.join(
                base_image_dir, self.split, "*.jpg"
            )
        )
        images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    base_image_dir,
                    "explanatory",
                    "train.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            print("len(self.img_to_explanation): ", len(self.img_to_explanation))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def get_train_data(self, image: np.ndarray, image_path: str, json_path: str):
        ori_size = image.shape[:2]
        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2
            else:
                choice = random.randint(0, 1)

        questions = []
        answers = []
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # add explanation if applicable
            if self.explanatory != -1 and image_name in self.img_to_explanation:
                if choice == 0:  # [SEG] token
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [SEG] token + text answer
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                    questions[-1] = (
                        DEFAULT_IMAGE_TOKEN
                        + "\n"
                        + text
                        + " {}".format(random.choice(self.explanatory_question_list))
                    )
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
                    answers.append(answer)
                else:
                    raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))

            conversations = []
            conv = conversation_lib.conv_templates[self.template_type].copy()

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1
        
        if (
            self.explanatory != -1
            and image_name in self.img_to_explanation
            and choice == 2
        ):
            # This type of conversation doesn't require GPT to output [SEG], hence mask is not needed.
            masks = torch.ones(1, *ori_size) * -1
            label = torch.ones(ori_size) * self.ignore_label
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return conversations, masks, label, questions, sampled_sents

    def get_val_data(self, image: np.ndarray, json_path: str):
        mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
        sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.conv_templates[self.template_type].copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "<extra_id_0>.")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "<extra_id_0>.")
            conversations.append(conv.get_prompt())
            i += 1
        
        masks = np.stack([mask_json], axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        return conversations, masks, labels

    def __getitem__(self, idx):
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.split == "train":
            (conversations, 
             masks, 
             seg_labels, 
             questions, 
             sampled_sents) = self.get_train_data(image, image_path, json_path)
        else:
            (conversations, 
             masks, 
             seg_labels) = self.get_val_data(image, json_path)
        
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess for sam
        image = self.transform.apply_image(image)
        resize = torch.Tensor(image.shape[:2])
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        # image = image.unsqueeze(0)
        image_clip = image_clip.unsqueeze(0)

        height_num_patches = image_clip.shape[2] // self.patch_dim
        width_num_patches = image_clip.shape[3] // self.patch_dim

        if self.mm_mlp_adapter_type == 'mlp_downsample':
            if height_num_patches % 2 != 0:
                height_num_patches += 1
            if width_num_patches % 2 != 0:
                width_num_patches += 1
        
        total_num_patches = height_num_patches * width_num_patches

        if len(conversations) != 1:
            # example: [' [INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n<image>\nIf you needed to change the volume or select a different radio station, what part of the radio would you adjust? Please respond with segmentation mask. [/INST] [SEG]. <extra_id_7>', ' [INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n<image>\nTo modify the volume or switch to a different radio station, which section of the radio would you manipulate? Please respond with segmentation mask. [/INST] It is [SEG]. <extra_id_7>', ' [INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n<image>\nIf we are looking at a close-up of a radio, which part of the radio should we adjust when selecting a different radio station or changing the volume? Please respond with segmentation mask. [/INST] Sure, it is [SEG]. <extra_id_7>']
            # masks.shape: torch.Size([3, 459, 800])
            # seg_labels.shape: torch.Size([459, 800])
            # import pdb; pdb.set_trace()
            pass
        
        modified_conversations = []
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * total_num_patches + DEFAULT_IM_END_TOKEN
        )
        for conversation in conversations:
            modified_conversations.append(conversation.replace(
                    DEFAULT_IMAGE_TOKEN, replace_token))
        # conversations = [conversation]
        tokens = tokenize(texts=modified_conversations, tokenizer=self.tokenizer, context_length=self.context_length, add_extra_token=self.add_extra_token,
        )

        # llama tricks
        # LISA Checkpoint Tokens: 
        ## tokenizer.convert_ids_to_tokens([32000, 32001, 32002])
        ## ['[SEG]', '<im_start>', '<im_end>'] 
        ## <extra_id_0>, <extra_id_1>, <extra_id_2>
        tokens[tokens == 32003] = 0  # DEFAULT_IMAGE_PATCH_TOKEN
        tokens[tokens == 32006] = 1  # <s>
        tokens[tokens == 32007] = 2  # </s>
        labels = tokens.clone().detach()

        conv = conversation_lib.conv_templates[self.template_type].copy()

        # Mask labels
        sep = "[/INST] "
        for conversation, target in zip(conversations, labels):
            rounds = conversation.split(conv.sep2)
            cur_len = 0
            for i, rou in enumerate(rounds):

                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                round_len = len(self.tokenizer.text_to_ids(rou + conv.sep2))
                instruction_len = len(self.tokenizer.text_to_ids(parts[0])) - 2
                if i > 0:
                    round_len -= 1  # Remove extra token added by sp tokenizer
                else:
                    instruction_len += 1
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX
        
        if self.add_extra_token:
            tokens = tokens[:, :-1].contiguous()
            labels = labels[:, 1:].contiguous()
        else:
            labels = torch.roll(labels, shifts=-1, dims=-1)
            labels[:, -1] = IGNORE_INDEX
        
        tokens = tokens
        labels = labels
        masks = masks.float()

        if self.split == "train":
            return dict(
                tokens=tokens,
                labels=labels,
                image=image,
                image_clip=image_clip,
                mask=masks,
                # seg_labels=seg_labels,
                resize=resize,
                conversation_len=len(conversations)
            )
        return dict(
            tokens=tokens,
            labels=labels,
            image=image,
            image_clip=image_clip,
            mask=masks,
            # seg_labels=seg_labels,
            resize=resize,
            conversation_len=len(conversations)
        )


@dataclass
class DataCollatorForSegmentationDataset(object):

    model_cfg: DictConfig
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        max_len = max(instance['tokens'].shape[1] for instance in instances)
        max_len = (max_len - 1) // 64 * 64 + 64
        mask_shapes = torch.stack([torch.Tensor([instance['mask'].shape[1], instance['mask'].shape[2]]).to(torch.int)
                                    for instance in instances], dim=0)
        max_height = mask_shapes[:, 0].max().item()
        max_width = mask_shapes[:, 1].max().item()
        offset_list = [0]
        count = 0
        for instance in instances:
            pad_len = max_len - instance['tokens'].shape[1]
            pad_h = max_height - instance['mask'].shape[1]
            pad_w = max_width - instance['mask'].shape[2]
            instance['tokens'] = F.pad(instance['tokens'], (0, pad_len), 'constant', 0)
            instance['labels'] = F.pad(instance['labels'], (0, pad_len), 'constant', -1)
            instance['mask'] = F.pad(instance["mask"], (0, pad_w, 0, pad_h), 'constant', -1)
            count += instance["conversation_len"]
            offset_list.append(count)

        batch = default_collate(instances)
        tokenizer = self.tokenizer
        model_cfg = self.model_cfg
        
        tokens = batch['tokens']
        labels = batch['labels']
        # batch_size * num seqs, max seq len
        _, _, max_len = tokens.shape
        tokens = tokens.reshape(-1, max_len)
        labels = labels.reshape(-1, max_len)
        media = batch.get('image_clip')

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                data=tokens,
                eod_token=tokenizer.eos_id,
                eod_mask_loss=model_cfg.data.get("eod_mask_loss", False),
                reset_attention_mask=False,
                reset_position_ids=False,
            )
        
        loss_mask[labels == -1] = 0.0
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0

        media = rearrange(media, "b T c h w -> b T 1 c h w")

        batch = {
            'tokens': tokens,
            'labels': labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'media': media,
            "images": batch["image"],
            "masks": batch["mask"],
            "mask_shapes": mask_shapes,
            "resizes": batch["resize"],
            "offsets": torch.LongTensor(offset_list),
        }
        return batch