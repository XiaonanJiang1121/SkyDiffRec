"""InternVL2.5 adapter with its standard dynamic-resolution tiling."""

from PIL import Image

from .base import BaseVLMAdapter, torch_dtype


def _closest_ratio(aspect_ratio, candidates, width, height, image_size):
    best_ratio = (1, 1)
    best_difference = float("inf")
    area = width * height
    for ratio in candidates:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference:
            best_difference = difference
            best_ratio = ratio
        elif difference == best_difference and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    width, height = image.size
    aspect_ratio = width / height
    ratios = {
        (columns, rows)
        for count in range(min_num, max_num + 1)
        for columns in range(1, count + 1)
        for rows in range(1, count + 1)
        if min_num <= columns * rows <= max_num
    }
    ratio = _closest_ratio(aspect_ratio, sorted(ratios, key=lambda item: item[0] * item[1]), width, height, image_size)
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    tiles = []
    for index in range(blocks):
        left = (index % ratio[0]) * image_size
        top = (index // ratio[0]) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and blocks > 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return tiles


class InternVL25Adapter(BaseVLMAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_tiles = int(self.options.get("max_tiles", 12))
        self.transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype(self.dtype),
            device_map=self.device,
        ).eval()

    def generate(self, image_path, prompt):
        with Image.open(image_path) as image:
            tiles = dynamic_preprocess(
                image.convert("RGB"), max_num=self.max_tiles, image_size=448
            )
        pixel_values = self.torch.stack([self.transform(tile) for tile in tiles])
        pixel_values = pixel_values.to(
            device=next(self.model.parameters()).device,
            dtype=torch_dtype(self.dtype),
        )
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        with self.torch.inference_mode():
            return self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
            ).strip()
