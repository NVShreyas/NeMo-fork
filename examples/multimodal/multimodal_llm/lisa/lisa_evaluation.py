# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import torch
from torch.utils.data import Dataset

from nemo.collections.multimodal.parts.utils import create_neva_model_and_processor
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.core.config import hydra_runner
from nemo.utils.get_rank import is_global_rank_zero

# LISA related libraries

import cv2

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize  # type: ignore
from torchvision.transforms.functional import to_pil_image


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

try:
    import modelopt.torch.quantization as mtq

    HAVE_MODELOPT = True

except (ImportError, ModuleNotFoundError):

    HAVE_MODELOPT = False

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(
        self,
    ):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


@hydra_runner(config_path="conf", config_name="lisa_inference")
def main(cfg) -> None:
    model, image_processor, video_processor = create_neva_model_and_processor(cfg, model_type="lisa")

    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
        "end_strings": cfg.inference.end_strings,
    }

    with open(cfg.prompt_file, 'r') as f:
        lines = f.readlines()

    transform = ResizeLongestSide(1024)

    media_type_token = cfg.inference.get("media_type", "image")
    media_token = f"<{media_type_token}>"
    insert_media_token = cfg.inference.get("insert_media_token", None)
    final_prompts = []
    for line in lines:
        prompt_dict = json.loads(line)
        assert 'prompt' in prompt_dict or 'text' in prompt_dict
        if 'prompt' not in prompt_dict:
            prompt_dict['prompt'] = prompt_dict['text']
        if insert_media_token == 'left':
            prompt_dict['prompt'] = media_token + "\n" + prompt_dict['prompt']
        elif insert_media_token == 'right':
            prompt_dict['prompt'] = prompt_dict['prompt'] + media_token
        if 'image' in prompt_dict:
            prompt_dict['image_path'] = prompt_dict['image']
            image_path=os.path.join(cfg.inference.media_base_path, prompt_dict['image'])
            prompt_dict['image_clip'] = image_processor(image_path).cuda()
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]
            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]
            image = (
                preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
            )
            # image = image.bfloat16()
            prompt_dict['image'] = image
            prompt_dict['resize_list'] = torch.Tensor(resize_list).cuda()
            prompt_dict['original_size_list'] = torch.Tensor(original_size_list).cuda()
        if 'video' in prompt_dict:
            prompt_dict['video_path'] = prompt_dict['video']
            prompt_dict['video'] = video_processor(os.path.join(cfg.inference.media_base_path, prompt_dict['video']))
        
        final_prompts.append(prompt_dict)
    
        pred_mask = model.generate(
            input_prompts=final_prompts, length_params=length_params, sampling_params=sampling_params, inference_config=cfg
        )
        if torch.is_tensor(pred_mask):
            pred_mask = pred_mask.detach().cpu().numpy()[0][0]
            pred_mask = pred_mask > 0
            save_path = cfg.output_mask_image_file
            cv2.imwrite(save_path, pred_mask * 100)
            print("{} has been saved.".format(save_path))

            save_path = cfg.output_seg_image_file
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
