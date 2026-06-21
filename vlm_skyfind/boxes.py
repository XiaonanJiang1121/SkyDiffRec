"""Bounding-box parsing and geometry helpers."""

import json
import math
import re


_NUMBER = r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![A-Za-z0-9_])"
_FOUR_NUMBER_GROUP = re.compile(
    rf"[\[\(\{{]\s*({_NUMBER})\s*[,;]\s*({_NUMBER})\s*[,;]\s*"
    rf"({_NUMBER})\s*[,;]\s*({_NUMBER})\s*[\]\)\}}]"
)
_TWO_POINT_GROUP = re.compile(
    rf"[\[\(]\s*[\[\(]\s*({_NUMBER})\s*[,;]\s*({_NUMBER})\s*[\]\)]\s*"
    rf"[,;]\s*[\[\(]\s*({_NUMBER})\s*[,;]\s*({_NUMBER})\s*[\]\)]\s*[\]\)]"
)
_LABELED_COORDINATES = re.compile(
    rf"x1\s*[:=]\s*({_NUMBER}).*?y1\s*[:=]\s*({_NUMBER}).*?"
    rf"x2\s*[:=]\s*({_NUMBER}).*?y2\s*[:=]\s*({_NUMBER})",
    flags=re.IGNORECASE | re.DOTALL,
)


def _finite(values):
    return all(math.isfinite(value) for value in values)


def extract_four_coordinates(text):
    """Extract one xyxy candidate without assuming its coordinate system."""
    if not isinstance(text, str) or not text.strip():
        return None

    for pattern in (_FOUR_NUMBER_GROUP, _TWO_POINT_GROUP, _LABELED_COORDINATES):
        match = pattern.search(text)
        if match:
            values = [float(value) for value in match.groups()]
            return values if _finite(values) else None

    try:
        payload = json.loads(text.strip().strip("`"))
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        lowered = {str(key).lower(): value for key, value in payload.items()}
        if all(key in lowered for key in ("x1", "y1", "x2", "y2")):
            values = [float(lowered[key]) for key in ("x1", "y1", "x2", "y2")]
            return values if _finite(values) else None

    numbers = re.findall(_NUMBER, text)
    if len(numbers) == 4:
        values = [float(value) for value in numbers]
        return values if _finite(values) else None
    return None


def convert_coordinates(values, width, height, mode="pixel", raw_text=""):
    if values is None:
        return None, None
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")

    detected_mode = mode
    if mode == "auto":
        lowered = raw_text.lower()
        if min(values) >= 0 and max(values) <= 1.5:
            detected_mode = "normalized_1"
        elif "normalized" in lowered and "1000" in lowered:
            detected_mode = "normalized_1000"
        elif "pixel" in lowered:
            detected_mode = "pixel"
        elif max(values) > 1000:
            detected_mode = "pixel"
        else:
            # Values such as [200, 300, 400, 500] could be either pixels or
            # normalized-to-1000 coordinates. Do not manufacture a metric.
            return None, "ambiguous"

    x1, y1, x2, y2 = values
    if detected_mode == "normalized_1000_or_1":
        detected_mode = (
            "normalized_1"
            if min(values) >= 0 and max(values) <= 1.0
            else "normalized_1000"
        )

    if detected_mode == "normalized_1":
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif detected_mode == "normalized_1000":
        x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
        y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
    elif detected_mode != "pixel":
        raise ValueError(f"Unknown coordinate mode: {mode}")

    return [x1, y1, x2, y2], detected_mode


def sanitize_box(box, width, height):
    if box is None or not _finite(box):
        return None
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def parse_prediction(text, width, height, coordinate_mode="pixel"):
    values = extract_four_coordinates(text)
    box, detected_mode = convert_coordinates(
        values, width, height, mode=coordinate_mode, raw_text=text
    )
    box = sanitize_box(box, width, height)
    return box, detected_mode
