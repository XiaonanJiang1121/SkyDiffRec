"""Model-specific output coordinate conventions."""


_MODEL_NATIVE_COORDINATES = {
    # Qwen2.5-VL emits resized-input pixel coordinates. Final evaluation must
    # restore them with the exact processor configuration.
    "qwen2.5-vl-7b": (
        "qwen_resized_pixel",
        "Qwen resized-input pixels restored with its processor configuration",
    ),
    # Match InternVL's released RefCOCO evaluation branch exactly.
    "internvl2.5-8b": (
        "internvl_official_mixed",
        "InternVL official sum(box) mixed [0,1]/[0,1000] rule",
    ),
    # DeepSeek-VL 7B does not publish a grounding-coordinate contract. Accept
    # only coordinate systems that can be inferred without guessing.
    "deepseek-vl-7b": (
        "uncontracted_vlm_strict",
        "no official grounding convention; strict unambiguous responses only",
    ),
    "llava-onevision-7b": (
        "uncontracted_vlm_strict",
        "no official grounding convention; strict unambiguous responses only",
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
