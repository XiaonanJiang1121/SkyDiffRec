#!/usr/bin/env python3
"""Reparse saved InternVL, LLaVA, or DeepSeek mixed-coordinate responses."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vlm_skyfind.boxes import box_iou, extract_four_coordinates  # noqa: E402
from vlm_skyfind.metrics import summarize, write_summary  # noqa: E402
from vlm_skyfind.mixed_coordinates import (  # noqa: E402
    convert_internvl_official,
    convert_uncontracted_vlm,
)


SUPPORTED_MODELS = {
    "internvl2.5-8b": (
        "internvl_official_mixed",
        "InternVL official RefCOCO evaluator: sum(box) >= 4 selects [0,1000], "
        "otherwise [0,1]; strict xyxy validation without clamping",
    ),
    "llava-onevision-7b": (
        "uncontracted_vlm_strict",
        "No official REC coordinate contract: accept [0,1] or an explicitly "
        "stated scale; bare values above 1 remain ambiguous",
    ),
    "deepseek-vl-7b": (
        "uncontracted_vlm_strict",
        "No official REC coordinate contract: accept [0,1] or an explicitly "
        "stated scale; bare values above 1 remain ambiguous",
    ),
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--ambiguous-policy",
        choices=("strict", "normalized_1000", "percent_100", "original_pixel"),
        default="strict",
        help="Sensitivity analysis only; strict is the primary reporting policy",
    )
    return parser.parse_args()


def _load_records(path, model):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if record.get("model") != model:
                raise ValueError(
                    f"Expected {model!r} in {path}:{line_number}, got "
                    f"{record.get('model')!r}"
                )
            records.append(record)
    return records


def _reparse_record(record, model, protocol, basis, ambiguous_policy):
    if record.get("status") in ("image_error", "inference_error"):
        return record

    response = record.get("raw_response") or ""
    values = extract_four_coordinates(response)
    if model == "internvl2.5-8b":
        box, detected_mode = convert_internvl_official(
            values, record["width"], record["height"]
        )
    else:
        box, detected_mode = convert_uncontracted_vlm(
            values,
            record["width"],
            record["height"],
            response,
            ambiguous_policy=ambiguous_policy,
        )

    record["coordinate_mode_requested"] = "model_native"
    record["coordinate_mode_resolved"] = protocol
    record["coordinate_mode_basis"] = basis
    record["coordinate_mode"] = detected_mode
    record["box_validation"] = "strict_xyxy_no_reorder_no_clamp"
    record["pred_box"] = box
    record["iou"] = box_iou(box, record["gt_box"])
    record["status"] = "ok" if box is not None else "parse_error"
    return record


def _write_protocol(
    input_path, output_path, model, protocol, basis, ambiguous_policy
):
    input_protocol = Path(str(input_path) + ".meta.json")
    metadata = {}
    if input_protocol.exists():
        with input_protocol.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    metadata.update(
        {
            "model": model,
            "coordinate_mode_requested": "model_native",
            "coordinate_mode": protocol,
            "coordinate_mode_basis": basis,
            "box_validation": "strict_xyxy_no_reorder_no_clamp",
            "ambiguous_policy": ambiguous_policy,
            "reparsed_from": str(input_path),
        }
    )
    with Path(str(output_path) + ".meta.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def main():
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("--output must differ from --input")
    if args.model == "internvl2.5-8b" and args.ambiguous_policy != "strict":
        raise SystemExit(
            "InternVL has an official mixed-scale rule; --ambiguous-policy "
            "is only available for LLaVA and DeepSeek sensitivity analysis"
        )

    protocol, basis = SUPPORTED_MODELS[args.model]
    records = [
        _reparse_record(
            record, args.model, protocol, basis, args.ambiguous_policy
        )
        for record in _load_records(input_path, args.model)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    _write_protocol(
        input_path,
        output_path,
        args.model,
        protocol,
        basis,
        args.ambiguous_policy,
    )

    summary = summarize(records)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.summary_output:
        write_summary(summary, args.summary_output)


if __name__ == "__main__":
    main()
