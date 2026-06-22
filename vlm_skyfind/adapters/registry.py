"""Model adapter registry."""


_ADAPTERS = {
    "qwen2.5-vl-7b": ("vlm_skyfind.adapters.qwen", "Qwen25VLAdapter"),
    "deepseek-vl-7b": ("vlm_skyfind.adapters.deepseek", "DeepSeekVLAdapter"),
    "internvl2.5-8b": ("vlm_skyfind.adapters.internvl", "InternVL25Adapter"),
    "llava-onevision-7b": ("vlm_skyfind.adapters.llava", "LlavaOneVisionAdapter"),
    "geochat-7b": ("vlm_skyfind.adapters.geochat", "GeoChatAdapter"),
}


def model_names():
    return sorted(_ADAPTERS)


def create_adapter(name, **kwargs):
    if name not in _ADAPTERS:
        raise ValueError(f"Unknown model {name!r}; choose from {model_names()}")
    module_name, class_name = _ADAPTERS[name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)(**kwargs)
