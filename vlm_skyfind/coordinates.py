"""Model-specific output coordinate conventions."""


_MODEL_NATIVE_COORDINATES = {
    # Qwen2.5-VL emits coordinates in the original image pixel space for the
    # released RSVG-ZeroOV prompt. The processor resizes model inputs but does
    # not expose or apply a bounding-box restoration step.
    "qwen2.5-vl-7b": (
        "pixel",
        "RSVG-ZeroOV behavior and saved SkyFind responses",
    ),
    # InternVL's official RefCOCO evaluator uses [0,1000], while its saved
    # SkyFind responses also contain unambiguous fractional [0,1] boxes.
    "internvl2.5-8b": (
        "normalized_1000_or_1",
        "InternVL official [0,1000] convention with observed [0,1] fallback",
    ),
    # DeepSeek-VL 7B does not publish a grounding-coordinate contract. Accept
    # only coordinate systems that can be inferred without guessing.
    "deepseek-vl-7b": (
        "auto",
        "no official grounding convention; unambiguous responses only",
    ),
    "llava-onevision-7b": (
        "auto",
        "not yet validated on SkyFind",
    ),
    "geochat-7b": (
        "auto",
        "not yet validated on SkyFind",
    ),
}


def resolve_coordinate_mode(model_name, requested_mode):
    if requested_mode != "model_native":
        return requested_mode, "explicit command-line setting"
    try:
        return _MODEL_NATIVE_COORDINATES[model_name]
    except KeyError as exc:
        raise ValueError(f"No coordinate profile for model {model_name!r}") from exc
