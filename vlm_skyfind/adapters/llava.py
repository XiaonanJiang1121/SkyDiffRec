"""LLaVA-compatible adapters for OneVision and GeoChat."""

from PIL import Image

from .base import BaseVLMAdapter


class LlavaAdapter(BaseVLMAdapter):
    default_conversation_mode = "qwen_1_5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model

        self.torch = torch
        self.default_image_token = DEFAULT_IMAGE_TOKEN
        self.default_image_start_token = DEFAULT_IM_START_TOKEN
        self.default_image_end_token = DEFAULT_IM_END_TOKEN
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        model_name = get_model_name_from_path(self.model_path)
        loaded = load_pretrained_model(
            self.model_path,
            None,
            model_name,
            device_map=self.device,
        )
        self.tokenizer, self.model, self.image_processor = loaded[:3]
        self.model.eval()
        self.conversation_mode = (
            self.options.get("conversation_mode") or self.default_conversation_mode
        )

    def generate(self, image_path, prompt):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = image.size
            image_tensor = self.process_images(
                [image], self.image_processor, self.model.config
            )
        if isinstance(image_tensor, list):
            image_tensor = [tensor.to(self.model.device, dtype=self.model.dtype) for tensor in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=self.model.dtype)

        conversation = self.conv_templates[self.conversation_mode].copy()
        image_token = self.default_image_token
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = (
                self.default_image_start_token
                + self.default_image_token
                + self.default_image_end_token
            )
        conversation.append_message(
            conversation.roles[0], f"{image_token}\n{prompt}"
        )
        conversation.append_message(conversation.roles[1], None)
        rendered_prompt = conversation.get_prompt()
        input_ids = self.tokenizer_image_token(
            rendered_prompt,
            self.tokenizer,
            self.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if (
            output_ids.shape[1] > input_ids.shape[1]
            and self.torch.equal(output_ids[:, :input_ids.shape[1]], input_ids)
        ):
            generated = output_ids[:, input_ids.shape[1]:]
        else:
            generated = output_ids
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


class LlavaOneVisionAdapter(LlavaAdapter):
    default_conversation_mode = "qwen_1_5"


class GeoChatAdapter(LlavaAdapter):
    default_conversation_mode = "llava_v1"
