"""Base interface shared by all VLM families."""

from abc import ABC, abstractmethod


class BaseVLMAdapter(ABC):
    def __init__(self, model_path, device="cuda:0", dtype="bfloat16", max_new_tokens=128, **kwargs):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.options = kwargs

    @abstractmethod
    def generate(self, image_path, prompt):
        """Return the model's unmodified text response."""


def torch_dtype(name):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]
