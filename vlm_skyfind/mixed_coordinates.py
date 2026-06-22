"""Strict, model-specific coordinate protocols for saved VLM responses."""

from .boxes import validate_box_strict


def _scale(values, width, height, denominator):
    x1, y1, x2, y2 = values
    return [
        x1 * width / denominator,
        y1 * height / denominator,
        x2 * width / denominator,
        y2 * height / denominator,
    ]


def convert_internvl_official(values, width, height):
    """Match InternVL's official RefCOCO mixed-scale evaluator.

    The released evaluator divides a generated box by 1000 when the sum of
    its four coordinates is at least 4; otherwise it treats it as [0,1]. It
    does not clamp an out-of-image prediction back into the image.
    """
    if values is None:
        return None, None
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")

    if sum(values) >= 4:
        mode = "internvl_official_normalized_1000"
        box = _scale(values, width, height, 1000.0)
    else:
        mode = "internvl_official_normalized_1"
        box = _scale(values, width, height, 1.0)
    return validate_box_strict(box), mode


def convert_uncontracted_vlm(
    values, width, height, raw_text="", ambiguous_policy="strict"
):
    """Convert only a scale that is explicit or unambiguous without GT.

    LLaVA-OneVision and DeepSeek-VL do not publish a REC coordinate contract.
    Bare values above 1 therefore remain ambiguous rather than being assigned
    the scale that happens to maximize IoU.
    """
    if values is None:
        return None, None
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")

    lowered = raw_text.lower()
    if min(values) >= 0 and max(values) <= 1:
        mode = "normalized_1"
        box = _scale(values, width, height, 1.0)
    elif "normalized" in lowered and "1000" in lowered:
        mode = "explicit_normalized_1000"
        box = _scale(values, width, height, 1000.0)
    elif "percent" in lowered or "%" in lowered:
        mode = "explicit_percent_100"
        box = _scale(values, width, height, 100.0)
    elif "pixel" in lowered:
        mode = "explicit_original_pixel"
        box = list(values)
    elif ambiguous_policy == "normalized_1000":
        mode = "sensitivity_normalized_1000"
        box = _scale(values, width, height, 1000.0)
    elif ambiguous_policy == "percent_100":
        mode = "sensitivity_percent_100"
        box = _scale(values, width, height, 100.0)
    elif ambiguous_policy == "original_pixel":
        mode = "sensitivity_original_pixel"
        box = list(values)
    elif ambiguous_policy == "strict":
        return None, "ambiguous"
    else:
        raise ValueError(f"Unknown ambiguous coordinate policy: {ambiguous_policy}")

    return validate_box_strict(box), mode
