"""SkyFind annotation reader."""

import json
from pathlib import Path

from PIL import Image

from .boxes import sanitize_box


SPLIT_FILES = {"val": "Val.json", "test": "Test.json"}


class InvalidImageError(RuntimeError):
    """Raised when a SkyFind image is missing or cannot be decoded."""


def source_name(file_name):
    stem = Path(file_name).stem
    return stem.split("_", 1)[0] if "_" in stem else "unknown"


class SkyFindDataset:
    def __init__(self, data_root, split, image_dir=None, source_prefixes=None):
        if split not in SPLIT_FILES:
            raise ValueError(f"Split must be one of {sorted(SPLIT_FILES)}, got {split}")
        self.data_root = Path(data_root)
        self.image_dir = Path(image_dir) if image_dir else self.data_root / "images"
        annotation_path = self.data_root / SPLIT_FILES[split]
        with annotation_path.open("r", encoding="utf-8") as handle:
            samples = json.load(handle)
        if not isinstance(samples, list):
            raise ValueError(f"Expected a JSON list in {annotation_path}")
        allowed = set(source_prefixes or [])
        self.samples = [
            (index, sample)
            for index, sample in enumerate(samples)
            if not allowed or source_name(sample["fileName"]) in allowed
        ]
        self.split = split
        self._image_info = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        annotation_index, sample = self.samples[item]
        image_path = self.image_dir / sample["fileName"]
        cache_key = str(image_path)
        if cache_key not in self._image_info:
            try:
                with Image.open(image_path) as image:
                    image.load()
                    self._image_info[cache_key] = (*image.size, None)
            except Exception as exc:
                message = f"Cannot decode image {image_path}: {type(exc).__name__}: {exc}"
                self._image_info[cache_key] = (None, None, message)
        width, height, image_error = self._image_info[cache_key]
        if image_error is not None:
            raise InvalidImageError(image_error)

        expression = sample.get("expression", "")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"Empty expression at {self.split}:{annotation_index}")
        gt_box = sanitize_box([float(value) for value in sample["bbox"]], width, height)
        if gt_box is None:
            raise ValueError(f"Invalid ground-truth box at {self.split}:{annotation_index}")
        return {
            "sample_id": f"{self.split}:{annotation_index}",
            "annotation_index": annotation_index,
            "split": self.split,
            "file_name": sample["fileName"],
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "expression": expression.strip(),
            "gt_box": gt_box,
            "source": source_name(sample["fileName"]),
        }
