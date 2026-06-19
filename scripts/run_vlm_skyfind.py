#!/usr/bin/env python3
"""Run one VLM on one SkyFind split."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_skyfind.adapters import model_names
from vlm_skyfind.runner import run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=model_names())
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--prompt-variant", default="pixel", choices=("rsvg", "pixel"))
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
    parser.add_argument("--qwen-min-pixels", default=None, type=int)
    parser.add_argument("--qwen-max-pixels", default=None, type=int)
    parser.add_argument("--conversation-mode", default=None)
    args = parser.parse_args()
    if args.max_consecutive_errors <= 0:
        parser.error("max_consecutive_errors must be positive")
    if args.qwen_min_pixels is not None and args.qwen_min_pixels <= 0:
        parser.error("qwen_min_pixels must be positive")
    if args.qwen_max_pixels is not None and args.qwen_max_pixels <= 0:
        parser.error("qwen_max_pixels must be positive")
    if (
        args.qwen_min_pixels is not None
        and args.qwen_max_pixels is not None
        and args.qwen_min_pixels > args.qwen_max_pixels
    ):
        parser.error("qwen_min_pixels cannot exceed qwen_max_pixels")
    return args


if __name__ == "__main__":
    run(parse_args())
