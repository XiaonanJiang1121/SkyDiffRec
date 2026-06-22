"""Restore Qwen2.5-VL resized-input coordinates to original-image pixels."""

import json
import math
from pathlib import Path


def _round_by_factor(value, factor):
    return round(value / factor) * factor


def _ceil_by_factor(value, factor):
    return math.ceil(value / factor) * factor


def _floor_by_factor(value, factor):
    return math.floor(value / factor) * factor


def load_preprocessor_config(path):
    """Load only the resize fields needed for coordinate restoration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    required = ("min_pixels", "max_pixels", "patch_size", "merge_size")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(
            f"Qwen preprocessor config is missing required fields: {missing}"
        )

    config = {
        "min_pixels": int(raw["min_pixels"]),
        "max_pixels": int(raw["max_pixels"]),
        "patch_size": int(raw["patch_size"]),
        "merge_size": int(raw["merge_size"]),
    }
    if any(value <= 0 for value in config.values()):
        raise ValueError("Qwen resize fields must all be positive")
    config["factor"] = config["patch_size"] * config["merge_size"]
    return config


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Match the official Qwen-VL smart-resize geometry."""
    if height <= 0 or width <= 0:
        raise ValueError("Image width and height must be positive")
    if factor <= 0 or min_pixels <= 0 or max_pixels <= 0:
        raise ValueError("Resize parameters must be positive")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio exceeds Qwen's supported limit of 200")

    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    area = resized_height * resized_width

    if area > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif area < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)

    return int(resized_height), int(resized_width)


def processed_size(width, height, config):
    """Return ``(processed_width, processed_height)`` for one image."""
    processed_height, processed_width = smart_resize(
        height=height,
        width=width,
        factor=config["factor"],
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
    )
    return processed_width, processed_height


def restore_coordinates(values, width, height, config):
    """Map an xyxy box from Qwen's resized input back to original pixels."""
    processed_width, processed_height = processed_size(width, height, config)
    x1, y1, x2, y2 = values
    restored = [
        x1 * width / processed_width,
        y1 * height / processed_height,
        x2 * width / processed_width,
        y2 * height / processed_height,
    ]
    return restored, processed_width, processed_height
