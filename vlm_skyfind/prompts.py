"""Shared prompts used by every VLM adapter."""

# Keep the two spaces after "description." to reproduce the released
# RSVG-ZeroOV implementation exactly.
RSVG_PROMPT_TEMPLATE = (
    "Locate it according to the following description.  "
    "{referring_expression} "
    "The output format should be like [x1, y1, x2, y2] without any other text."
)

PIXEL_PROMPT_TEMPLATE = (
    "Locate the object referred to by '{referring_expression}' and return only "
    "its box coordinates as [x1, y1, x2, y2] in the original {width} x "
    "{height} SkyFind image pixel coordinate system."
)


def prompt_template(variant):
    if variant == "rsvg":
        return RSVG_PROMPT_TEMPLATE
    if variant == "pixel":
        return PIXEL_PROMPT_TEMPLATE
    raise ValueError(f"Unknown prompt variant: {variant}")


def build_prompt(expression, width, height, variant="rsvg"):
    return prompt_template(variant).format(
        referring_expression=expression.strip(),
        width=width,
        height=height,
    )
