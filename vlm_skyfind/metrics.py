"""Metrics and diagnostic slices for SkyFind predictions."""

import json
from collections import Counter, defaultdict
from pathlib import Path


def expression_bucket(expression):
    length = len(expression.split())
    if length <= 20:
        return "short_le_20"
    if length <= 40:
        return "medium_21_40"
    return "long_gt_40"


def target_size_bucket(record):
    box = record["gt_box"]
    image_area = record["width"] * record["height"]
    ratio = ((box[2] - box[0]) * (box[3] - box[1])) / image_area
    if ratio < 0.001:
        return "tiny_lt_0.001"
    if ratio < 0.01:
        return "small_0.001_0.01"
    return "large_ge_0.01"


def _score(records):
    evaluable = [record for record in records if record.get("status") != "image_error"]
    parsed = [record for record in evaluable if record.get("status") == "ok"]
    ious = [float(record.get("iou", 0.0)) for record in evaluable]
    parsed_ious = [float(record.get("iou", 0.0)) for record in parsed]
    count = len(evaluable)
    return {
        "count": count,
        "parsed_count": len(parsed),
        "parse_rate": len(parsed) / count if count else 0.0,
        "miou": sum(ious) / count if count else 0.0,
        "parsed_miou": sum(parsed_ious) / len(parsed) if parsed else 0.0,
        "acc_0.5": sum(iou >= 0.5 for iou in ious) / count if count else 0.0,
        "acc_0.7": sum(iou >= 0.7 for iou in ious) / count if count else 0.0,
    }


def summarize(records):
    records = list(records)
    summary = _score(records)
    summary["record_count"] = len(records)
    summary["status_counts"] = dict(Counter(record.get("status", "unknown") for record in records))
    latencies = [record["latency_seconds"] for record in records if "latency_seconds" in record]
    summary["mean_latency_seconds"] = sum(latencies) / len(latencies) if latencies else None

    groups = {
        "by_source": lambda record: record.get("source", "unknown"),
        "by_expression_length": lambda record: expression_bucket(record["expression"]),
        "by_target_size": target_size_bucket,
    }
    for name, key_fn in groups.items():
        buckets = defaultdict(list)
        for record in records:
            if record.get("status") != "image_error":
                buckets[key_fn(record)].append(record)
        summary[name] = {key: _score(value) for key, value in sorted(buckets.items())}
    return summary


def load_jsonl(paths):
    by_id = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = record["sample_id"]
                if sample_id in by_id:
                    raise ValueError(f"Duplicate sample_id {sample_id} in {path}:{line_number}")
                by_id[sample_id] = record
    return [by_id[key] for key in sorted(by_id)]


def write_summary(summary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
