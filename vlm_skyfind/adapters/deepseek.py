"""DeepSeek-VL 7B adapter."""

from .base import BaseVLMAdapter, torch_dtype


class DeepSeekVLAdapter(BaseVLMAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch
        from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
        from deepseek_vl.utils.io import load_pil_images

        self.torch = torch
        self.load_pil_images = load_pil_images
        self.processor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = MultiModalityCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        ).to(dtype=torch_dtype(self.dtype), device=self.device).eval()

    def generate(self, image_path, prompt):
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
                "images": [image_path],
            },
            {"role": "Assistant", "content": ""},
        ]
        images = self.load_pil_images(conversation)
        inputs = self.processor(
            conversations=conversation,
            images=images,
            force_batchify=True,
        ).to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)
        with self.torch.inference_mode():
            output_ids = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        return self.tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True).strip()
