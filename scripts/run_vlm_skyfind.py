#!/usr/bin/env python3
"""Run one VLM on one SkyFind split."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "vlm_models.json"
DEFAULT_DATA_ROOT = "/root/autodl-tmp/BioLoc/data/SkyFind_data"

sys.path.insert(0, str(PROJECT_ROOT))


def normalize_thread_environment():
    """Prevent libgomp from rejecting malformed inherited thread counts."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            valid = int(value) > 0
        except ValueError:
            valid = False
        if not valid:
            os.environ[name] = "1"


normalize_thread_environment()

from vlm_skyfind.adapters import model_names
from vlm_skyfind.runner import run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=model_names())
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--prompt-variant", default="rsvg", choices=("rsvg", "pixel"))
    parser.add_argument(
        "--coordinate-mode",
        default="pixel",
        choices=("pixel", "auto", "normalized_1", "normalized_1000"),
    )
    parser.add_argument("--source-prefixes", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-tracebacks", action="store_true")
    parser.add_argument("--max-consecutive-errors", default=5, type=int)
    parser.add_argument("--attn-implementation", default="sdpa", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--internvl-max-tiles", default=12, type=int)
    parser.add_argument("--conversation-mode", default=None)
    parser.add_argument("--llava-model-name", default="llava_qwen")
    args = parser.parse_args()
    if args.model_path is None:
        config_path = Path(args.model_config)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                model_paths = json.load(handle)
            args.model_path = model_paths[args.model]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(f"Cannot resolve {args.model!r} from {config_path}: {exc}")
    if args.max_consecutive_errors <= 0:
        parser.error("max_consecutive_errors must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
