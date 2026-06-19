#!/usr/bin/env python3
"""Merge one or more JSONL shards and report SkyFind metrics."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_skyfind.metrics import load_jsonl, summarize, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = summarize(load_jsonl(args.predictions))
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.output:
        write_summary(summary, args.output)


if __name__ == "__main__":
    main()
