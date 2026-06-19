"""Shared prompts used by every VLM adapter."""

RSVG_PROMPT_TEMPLATE = (
    "Locate the object referred to by '{expression}' and return its box "
    "coordinates (x1, y1, x2, y2)."
)


def build_prompt(expression, width, height, variant="pixel"):
    base = RSVG_PROMPT_TEMPLATE.format(expression=expression.strip())
    if variant == "rsvg":
        return base
    if variant == "pixel":
        return (
            f"{base} The original image size is {width} x {height} pixels. "
            "Return only one box as [x1, y1, x2, y2] in original-image "
            "pixel coordinates, with no other text."
        )
    raise ValueError(f"Unknown prompt variant: {variant}")
