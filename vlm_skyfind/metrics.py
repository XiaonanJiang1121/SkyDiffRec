"""Metrics and diagnostic slices for SkyFind predictions."""

import json
from collections import Counter, defaultdict
from pathlib import Path


SKYFIND_IOU_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)

FINAL_COORDINATE_PROTOCOLS = {
    "qwen2.5-vl-7b": "qwen_resized_pixel",
    "internvl2.5-8b": "internvl_official_mixed",
    "llava-onevision-7b": "uncontracted_vlm_strict",
    "deepseek-vl-7b": "uncontracted_vlm_strict",
}
FINAL_BOX_VALIDATION = "strict_xyxy_no_reorder_no_clamp"


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
    threshold_accuracy = {
        f"{threshold:.1f}": (
            sum(iou >= threshold for iou in ious) / count if count else 0.0
        )
        for threshold in SKYFIND_IOU_THRESHOLDS
    }
    iou_at_mean = sum(threshold_accuracy.values()) / len(SKYFIND_IOU_THRESHOLDS)
    return {
        "count": count,
        "parsed_count": len(parsed),
        "parse_rate": len(parsed) / count if count else 0.0,
        "iou_at_0.5": threshold_accuracy["0.5"],
        "iou_at_mean": iou_at_mean,
        "iou_at_0.5_percent": threshold_accuracy["0.5"] * 100.0,
        "iou_at_mean_percent": iou_at_mean * 100.0,
        "iou_threshold_accuracy": threshold_accuracy,
        # Retain raw mean IoU as a localization diagnostic. It is not the
        # IoU@mean metric reported in SkyFind Table 4.
        "miou": sum(ious) / count if count else 0.0,
        "parsed_miou": sum(parsed_ious) / len(parsed) if parsed else 0.0,
        "acc_0.5": threshold_accuracy["0.5"],
        "acc_0.7": sum(iou >= 0.7 for iou in ious) / count if count else 0.0,
    }


def summarize_table4(model, val_records, test_records):
    """Build the six-column SkyFind Table 4 result for one model."""
    val_records = list(val_records)
    test_records = list(test_records)
    for split, records in (("val", val_records), ("test", test_records)):
        if not records:
            raise ValueError(f"No {split} records supplied for Table 4")
        for record in records:
            if record.get("split") != split:
                raise ValueError(
                    f"Expected {split} record, got {record.get('split')!r} "
                    f"for {record.get('sample_id', 'unknown sample')}"
                )
            if record.get("model") != model:
                raise ValueError(
                    f"Expected model {model!r}, got {record.get('model')!r} "
                    f"for {record.get('sample_id', 'unknown sample')}"
                )

    val = _score(val_records)
    test = _score(test_records)

    def columns(score):
        return {
            "iou_at_0.5": score["iou_at_0.5"],
            "iou_at_mean": score["iou_at_mean"],
            "iou_at_0.5_percent": score["iou_at_0.5_percent"],
            "iou_at_mean_percent": score["iou_at_mean_percent"],
            "count": score["count"],
        }

    average_iou_05 = (val["iou_at_0.5"] + test["iou_at_0.5"]) / 2.0
    average_iou_mean = (val["iou_at_mean"] + test["iou_at_mean"]) / 2.0
    return {
        "model": model,
        "val": columns(val),
        "test": columns(test),
        "average": {
            "iou_at_0.5": average_iou_05,
            "iou_at_mean": average_iou_mean,
            "iou_at_0.5_percent": average_iou_05 * 100.0,
            "iou_at_mean_percent": average_iou_mean * 100.0,
        },
    }


def validate_final_protocol(model, records):
    """Reject provisional raw-run records from final Table 4 reporting."""
    expected_mode = FINAL_COORDINATE_PROTOCOLS.get(model)
    if expected_mode is None:
        return
    for record in records:
        if record.get("status") in ("image_error", "inference_error"):
            continue
        if record.get("coordinate_mode_resolved") != expected_mode:
            raise ValueError(
                f"{model} final reporting requires coordinate_mode_resolved="
                f"{expected_mode!r}; got "
                f"{record.get('coordinate_mode_resolved')!r} for "
                f"{record.get('sample_id', 'unknown sample')}"
            )
        if record.get("box_validation") != FINAL_BOX_VALIDATION:
            raise ValueError(
                f"{model} final reporting requires box_validation="
                f"{FINAL_BOX_VALIDATION!r}; got "
                f"{record.get('box_validation')!r} for "
                f"{record.get('sample_id', 'unknown sample')}"
            )


def summarize(records):
    records = list(records)
    summary = _score(records)
    summary["record_count"] = len(records)
    summary["status_counts"] = dict(Counter(record.get("status", "unknown") for record in records))
    summary["skipped_image_count"] = sum(
        record.get("status") == "image_error" for record in records
    )
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
