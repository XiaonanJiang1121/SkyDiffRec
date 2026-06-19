"""Shared prompts used by every VLM adapter."""

RSVG_PROMPT_TEMPLATE = "Locate the object referred to by '{referring_expression}' and return its box coordinates (x1, y1, x2, y2)."

PIXEL_PROMPT_TEMPLATE = "Locate the object referred to by '{referring_expression}' and return its box coordinates (x1, y1, x2, y2) as a single list [x1, y1, x2, y2] in the original {width} x {height} SkyFind image pixel coordinate system."


def build_prompt(expression, width, height, variant="rsvg"):
    if variant == "rsvg":
        return RSVG_PROMPT_TEMPLATE.format(referring_expression=expression.strip())
    if variant == "pixel":
        return PIXEL_PROMPT_TEMPLATE.format(
            referring_expression=expression.strip(),
            width=width,
            height=height,
        )
    raise ValueError(f"Unknown prompt variant: {variant}")
