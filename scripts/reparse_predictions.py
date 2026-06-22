#!/usr/bin/env python3
"""Reparse saved raw VLM responses without running model inference again."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vlm_skyfind.boxes import box_iou, parse_prediction
from vlm_skyfind.coordinates import resolve_coordinate_mode
from vlm_skyfind.metrics import summarize, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--coordinate-mode", default="model_native")
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        parser.error("--output must differ from --input")

    mode, basis = resolve_coordinate_mode(args.model, args.coordinate_mode)
    if mode == "qwen_resized_pixel":
        parser.error(
            "Qwen resized-input coordinates require "
            "scripts/reparse_qwen_predictions.py and its preprocessor config"
        )
    if mode in ("internvl_official_mixed", "uncontracted_vlm_strict"):
        parser.error(
            "This model's final strict protocol requires "
            "scripts/reparse_mixed_coordinates.py"
        )
    records = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") not in ("image_error", "inference_error"):
                box, detected_mode = parse_prediction(
                    record.get("raw_response"),
                    record["width"],
                    record["height"],
                    coordinate_mode=mode,
                )
                record["coordinate_mode_requested"] = args.coordinate_mode
                record["coordinate_mode_resolved"] = mode
                record["coordinate_mode_basis"] = basis
                record["coordinate_mode"] = detected_mode
                record["pred_box"] = box
                record["iou"] = box_iou(box, record["gt_box"])
                record["status"] = "ok" if box is not None else "parse_error"
            destination.write(json.dumps(record, ensure_ascii=True) + "\n")
            records.append(record)

    input_protocol = Path(str(input_path) + ".meta.json")
    if input_protocol.exists():
        with input_protocol.open("r", encoding="utf-8") as handle:
            protocol = json.load(handle)
        protocol["coordinate_mode_requested"] = args.coordinate_mode
        protocol["coordinate_mode"] = mode
        protocol["coordinate_mode_basis"] = basis
        protocol["reparsed_from"] = str(input_path)
        with Path(str(output_path) + ".meta.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(protocol, handle, indent=2, ensure_ascii=True)
            handle.write("\n")

    summary = summarize(records)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.summary_output:
        write_summary(summary, args.summary_output)


if __name__ == "__main__":
    main()
