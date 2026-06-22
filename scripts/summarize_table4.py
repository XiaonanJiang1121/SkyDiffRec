#!/usr/bin/env python3
"""Create one SkyFind Table 4 row from Val and Test predictions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_skyfind.metrics import (
    load_jsonl,
    summarize_table4,
    validate_final_protocol,
    write_summary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--val", required=True, nargs="+")
    parser.add_argument("--test", required=True, nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    val_records = load_jsonl(args.val)
    test_records = load_jsonl(args.test)
    validate_final_protocol(args.model, val_records)
    validate_final_protocol(args.model, test_records)
    summary = summarize_table4(args.model, val_records, test_records)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.output:
        write_summary(summary, args.output)


if __name__ == "__main__":
    main()
