#!/usr/bin/env python3
"""Compare strict and repairing Qwen box postprocessing on saved responses."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vlm_skyfind.boxes import (  # noqa: E402
    box_iou,
    extract_four_coordinates,
    sanitize_box,
    validate_box_strict,
)
from vlm_skyfind.metrics import summarize, summarize_table4, write_summary  # noqa: E402
from vlm_skyfind.qwen_coordinates import (  # noqa: E402
    load_preprocessor_config,
    restore_coordinates,
)


MODEL = "qwen2.5-vl-7b"
MODE = "qwen_resized_pixel"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--preprocessor-config", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _load(path, expected_split):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("model") != MODEL or record.get("split") != expected_split:
                raise ValueError(
                    f"Unexpected model/split in {path}:{line_number}: "
                    f"{record.get('model')!r}/{record.get('split')!r}"
                )
            records.append(record)
    return records


def _audit_split(records, config):
    counts = Counter(
        {
            "strict_valid_xyxy": 0,
            "reversed_x": 0,
            "reversed_y": 0,
            "zero_area": 0,
            "out_of_bounds": 0,
            "no_coordinates": 0,
            "image_error": 0,
            "inference_error": 0,
        }
    )
    variants = {"strict": [], "sanitize": []}

    for original in records:
        if original.get("status") in ("image_error", "inference_error"):
            counts[original["status"]] += 1
            for output in variants.values():
                output.append(dict(original))
            continue

        values = extract_four_coordinates(original.get("raw_response"))
        restored = None
        if values is None:
            counts["no_coordinates"] += 1
        else:
            restored, _, _ = restore_coordinates(
                values, original["width"], original["height"], config
            )
            x1, y1, x2, y2 = restored
            if x2 < x1:
                counts["reversed_x"] += 1
            if y2 < y1:
                counts["reversed_y"] += 1
            if x2 == x1 or y2 == y1:
                counts["zero_area"] += 1
            if (
                x1 < 0
                or y1 < 0
                or x2 > original["width"]
                or y2 > original["height"]
            ):
                counts["out_of_bounds"] += 1
            if validate_box_strict(restored) is not None:
                counts["strict_valid_xyxy"] += 1

        for policy in variants:
            record = dict(original)
            if policy == "strict":
                box = validate_box_strict(restored)
                validation = "strict_xyxy_no_reorder_no_clamp"
            else:
                box = sanitize_box(
                    restored, original["width"], original["height"]
                )
                validation = "sanitize_reorder_and_clamp"
            record["coordinate_mode_requested"] = "model_native"
            record["coordinate_mode_resolved"] = MODE
            record["coordinate_mode"] = MODE
            record["box_validation"] = validation
            record["pred_box"] = box
            record["iou"] = box_iou(box, record["gt_box"])
            record["status"] = "ok" if box is not None else "parse_error"
            variants[policy].append(record)

    return dict(counts), variants


def _compact_summary(records):
    result = summarize(records)
    return {
        key: result[key]
        for key in (
            "count",
            "parsed_count",
            "parse_rate",
            "iou_at_0.5_percent",
            "iou_at_mean_percent",
            "miou",
            "status_counts",
        )
    }


def main():
    args = _parse_args()
    config = load_preprocessor_config(args.preprocessor_config)
    audits = {}
    by_policy = {"strict": {}, "sanitize": {}}
    for split, path in (("val", args.val), ("test", args.test)):
        audits[split], variants = _audit_split(_load(path, split), config)
        for policy, records in variants.items():
            by_policy[policy][split] = records

    tables = {
        policy: summarize_table4(MODEL, records["val"], records["test"])
        for policy, records in by_policy.items()
    }
    strict_average = tables["strict"]["average"]
    sanitize_average = tables["sanitize"]["average"]
    result = {
        "model": MODEL,
        "coordinate_mode": MODE,
        "pre_policy_audit": audits,
        "strict": {
            "val": _compact_summary(by_policy["strict"]["val"]),
            "test": _compact_summary(by_policy["strict"]["test"]),
            "table4": tables["strict"],
        },
        "sanitize": {
            "val": _compact_summary(by_policy["sanitize"]["val"]),
            "test": _compact_summary(by_policy["sanitize"]["test"]),
            "table4": tables["sanitize"],
        },
        "sanitize_minus_strict_average_percent": {
            "iou_at_0.5": (
                sanitize_average["iou_at_0.5_percent"]
                - strict_average["iou_at_0.5_percent"]
            ),
            "iou_at_mean": (
                sanitize_average["iou_at_mean_percent"]
                - strict_average["iou_at_mean_percent"]
            ),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if args.output:
        write_summary(result, args.output)


if __name__ == "__main__":
    main()
