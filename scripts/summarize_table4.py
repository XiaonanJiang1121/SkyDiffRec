#!/usr/bin/env python3
"""Create one SkyFind Table 4 row from Val and Test predictions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_skyfind.metrics import load_jsonl, summarize_table4, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--val", required=True, nargs="+")
    parser.add_argument("--test", required=True, nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summary = summarize_table4(
        args.model,
        load_jsonl(args.val),
        load_jsonl(args.test),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.output:
        write_summary(summary, args.output)


if __name__ == "__main__":
    main()
