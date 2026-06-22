#!/usr/bin/env python3
"""Generate reproducible failure-mode diagnostics from finalized JSONL files."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vlm_skyfind.failure_modes import (  # noqa: E402
    analyze_failure_modes,
    failure_table_markdown,
)
from vlm_skyfind.metrics import (  # noqa: E402
    load_jsonl,
    validate_final_protocol,
    write_summary,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        nargs=3,
        required=True,
        metavar=("MODEL", "VAL_JSONL", "TEST_JSONL"),
        help="May be repeated once per model",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output")
    parser.add_argument(
        "--markdown-output", default=None, help="Optional compact Markdown table"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    results = {}
    for model, val_path, test_path in args.result:
        if model in results:
            raise ValueError(f"Duplicate model in --result: {model}")
        val_records = load_jsonl([val_path])
        test_records = load_jsonl([test_path])
        validate_final_protocol(model, val_records)
        validate_final_protocol(model, test_records)
        results[model] = {
            "val": analyze_failure_modes(val_records),
            "test": analyze_failure_modes(test_records),
        }

    payload = {
        "definition": {
            "scale_shape_failure_at_0.5": (
                "IoU remains below 0.5 after aligning prediction and GT centers "
                "while preserving their widths and heights"
            ),
            "center_error_over_gt_diag": (
                "Euclidean center distance divided by GT box diagonal"
            ),
        },
        "models": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    if args.output:
        write_summary(payload, args.output)
    if args.markdown_output:
        path = Path(args.markdown_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(failure_table_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
