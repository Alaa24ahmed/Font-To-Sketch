from typing import Dict, Union, Optional, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import DonutImageProcessor, DonutProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ChannelDimension, make_list_of_images
from ocr.tokenizer import Byt5LangTokenizer
from ocr.settings import settings
import matplotlib.pyplot as plt
import numpy as np


def load_processor():
    processor = SuryaProcessor()
    processor.image_processor.train = False
    processor.image_processor.max_size = settings.RECOGNITION_IMAGE_SIZE
    processor.tokenizer.model_max_length = settings.RECOGNITION_MAX_TOKENS
    return processor


class SuryaImageProcessor(DonutImageProcessor):
    def __init__(self, *args, max_size=None, train=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = kwargs.get("patch_size", (4, 4))
        self.max_size = max_size
        self.train = train

    def visualize(self, image: torch.Tensor, title: str):
        plt.figure()
        plt.title(title)
        plt.imshow(image.permute(1, 2, 0).numpy().astype(np.uint8))
        plt.axis("off")
        plt.show()

    def resize_tensor_preserve_aspect_ratio(
        self, image: torch.Tensor, size: Tuple[int, int]
    ) -> torch.Tensor:
        # Resize image preserving aspect ratio
        target_height, target_width = size
        _, input_height, input_width = image.shape

        scale = min(target_height / input_height, target_width / input_width)
        new_height = int(input_height * scale)
        new_width = int(input_width * scale)

        resized_image = F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return resized_image

    def pad_image(
        self, image: torch.Tensor, size: Tuple[int, int], pad_value: float = 0.0
    ) -> torch.Tensor:
        target_height, target_width = size
        _, input_height, input_width = image.shape

        delta_width = target_width - input_width
        delta_height = target_height - input_height

        pad_left = delta_width // 2
        pad_right = delta_width - pad_left
        pad_top = delta_height // 2
        pad_bottom = delta_height - pad_top

        padding = (pad_left, pad_right, pad_top, pad_bottom)
        padded_image = F.pad(image, padding, value=pad_value)
        return padded_image

    def align_long_axis(
        self, image: torch.Tensor, size: Tuple[int, int]
    ) -> torch.Tensor:
        input_height, input_width = image.shape[1:]
        target_height, target_width = size

        if (target_width < target_height and input_width > input_height) or (
            target_width > target_height and input_width < input_height
        ):
            image = image.permute(0, 2, 1).flip(1)
        return image

    def preprocess(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        size: Tuple[int, int] = (196, 896),
        rescale_factor: float = 1.0,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[str] = None,
    ) -> Union[Dict[str, torch.Tensor], BatchFeature]:

        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for img in images:
            if img.dim() == 4:  # Ensure the image is in CHW format, not in batch format
                img = img.squeeze(0)

            img = self.align_long_axis(img, size)

            img = self.resize_tensor_preserve_aspect_ratio(img, size)
            img = self.pad_image(img, size, pad_value=255)

            img = img.float()  # Convert to float32 for rescaling
            img = img / rescale_factor
            if image_mean is not None and image_std is not None:
                img = (img - torch.tensor(image_mean).view(-1, 1, 1)) / torch.tensor(
                    image_std
                ).view(-1, 1, 1)

            processed_images.append(img)

        processed_images = torch.stack(processed_images)
        data = {"pixel_values": processed_images}
        return BatchFeature(data=data, tensor_type=return_tensors)


class SuryaProcessor(DonutProcessor):
    def __init__(self, image_processor=None, tokenizer=None, train=False, **kwargs):
        image_processor = SuryaImageProcessor.from_pretrained(
            settings.RECOGNITION_MODEL_CHECKPOINT
        )
        tokenizer = Byt5LangTokenizer()
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        lang = kwargs.pop("lang", None)

        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError(
                "You need to specify either an `images` or `text` input to process."
            )

        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)

        if text is not None:
            encodings = self.tokenizer(text, lang, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            inputs["langs"] = encodings["langs"]
            return inputs
