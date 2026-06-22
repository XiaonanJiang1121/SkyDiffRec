#!/usr/bin/env python3
"""Reparse saved Qwen2.5-VL responses in resized-input pixel coordinates."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vlm_skyfind.boxes import (  # noqa: E402
    box_iou,
    extract_four_coordinates,
    sanitize_box,
    validate_box_strict,
)
from vlm_skyfind.metrics import summarize, write_summary  # noqa: E402
from vlm_skyfind.qwen_coordinates import (  # noqa: E402
    load_preprocessor_config,
    processed_size,
    restore_coordinates,
)


COORDINATE_MODE = "qwen_resized_pixel"
COORDINATE_BASIS = (
    "Qwen2.5-VL official processor: output coordinates refer to the resized "
    "model input and are restored to the original image before evaluation"
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preprocessor-config", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--box-policy",
        choices=("strict", "sanitize"),
        default="strict",
        help="Use strict for final reporting; sanitize is audit-only",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Override min_pixels only if the original run explicitly changed it",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Override max_pixels only if the original run explicitly changed it",
    )
    return parser.parse_args()


def _load_records(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if record.get("model") != "qwen2.5-vl-7b":
                raise ValueError(
                    f"Expected qwen2.5-vl-7b in {path}:{line_number}, got "
                    f"{record.get('model')!r}"
                )
            records.append(record)
    return records


def _reparse_record(record, config, box_policy):
    if record.get("status") in ("image_error", "inference_error"):
        return record

    width = record["width"]
    height = record["height"]
    values = extract_four_coordinates(record.get("raw_response"))
    processed_width, processed_height = processed_size(width, height, config)
    if values is None:
        box = None
    else:
        box, _, _ = restore_coordinates(values, width, height, config)
        if box_policy == "strict":
            box = validate_box_strict(box)
        else:
            box = sanitize_box(box, width, height)

    record["coordinate_mode_requested"] = "model_native"
    record["coordinate_mode_resolved"] = COORDINATE_MODE
    record["coordinate_mode_basis"] = COORDINATE_BASIS
    record["coordinate_mode"] = COORDINATE_MODE
    record["box_validation"] = (
        "strict_xyxy_no_reorder_no_clamp"
        if box_policy == "strict"
        else "sanitize_reorder_and_clamp"
    )
    record["qwen_processed_width"] = processed_width
    record["qwen_processed_height"] = processed_height
    record["pred_box"] = box
    record["iou"] = box_iou(box, record["gt_box"])
    record["status"] = "ok" if box is not None else "parse_error"
    return record


def _write_protocol(
    input_path, output_path, config_path, config, box_policy
):
    input_protocol = Path(str(input_path) + ".meta.json")
    protocol = {}
    if input_protocol.exists():
        with input_protocol.open("r", encoding="utf-8") as handle:
            protocol = json.load(handle)
    protocol.update(
        {
            "model": "qwen2.5-vl-7b",
            "coordinate_mode_requested": "model_native",
            "coordinate_mode": COORDINATE_MODE,
            "coordinate_mode_basis": COORDINATE_BASIS,
            "box_validation": (
                "strict_xyxy_no_reorder_no_clamp"
                if box_policy == "strict"
                else "sanitize_reorder_and_clamp"
            ),
            "qwen_box_policy": box_policy,
            "reparsed_from": str(input_path),
            "qwen_preprocessor_config": str(config_path),
            "qwen_resize": config,
        }
    )
    with Path(str(output_path) + ".meta.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def main():
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.preprocessor_config)
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("--output must differ from --input")

    config = load_preprocessor_config(config_path)
    if args.min_pixels is not None:
        config["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        config["max_pixels"] = args.max_pixels
    if config["min_pixels"] > config["max_pixels"]:
        raise SystemExit("min_pixels cannot exceed max_pixels")

    records = [
        _reparse_record(record, config, args.box_policy)
        for record in _load_records(input_path)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    _write_protocol(
        input_path,
        output_path,
        config_path,
        config,
        args.box_policy,
    )

    summary = summarize(records)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.summary_output:
        write_summary(summary, args.summary_output)


if __name__ == "__main__":
    main()
