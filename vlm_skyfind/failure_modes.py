"""Failure-mode diagnostics for finalized SkyFind prediction records."""

import math
import re
import statistics
from collections import Counter, defaultdict

from .metrics import expression_bucket, target_size_bucket


RELATION_PATTERN = re.compile(
    r"\b(next to|near|nearby|beside|between|behind|in front of|adjacent|"
    r"left of|right of|above|below|under|over|alongside|across from|"
    r"closest|nearest|farthest|furthest|another|other|relative to|"
    r"surrounded by|ahead of|following)\b",
    re.IGNORECASE,
)
ORDINAL_PATTERN = re.compile(
    r"\b(leftmost|rightmost|topmost|bottommost|first|second|third|fourth|"
    r"fifth|sixth|seventh|eighth|ninth|tenth|last|[1-9](?:st|nd|rd|th))\b",
    re.IGNORECASE,
)


def centered_box_iou(pred_box, gt_box):
    """Maximum IoU after aligning centers while preserving both box sizes."""
    pred_width = pred_box[2] - pred_box[0]
    pred_height = pred_box[3] - pred_box[1]
    gt_width = gt_box[2] - gt_box[0]
    gt_height = gt_box[3] - gt_box[1]
    if min(pred_width, pred_height, gt_width, gt_height) <= 0:
        return 0.0
    intersection = min(pred_width, gt_width) * min(pred_height, gt_height)
    union = pred_width * pred_height + gt_width * gt_height - intersection
    return intersection / union if union > 0 else 0.0


def _bucket_accuracy(records):
    result = {}
    for name, values in sorted(records.items()):
        count = len(values)
        result[name] = {
            "count": count,
            "iou_at_0.5": (
                sum(value >= 0.5 for value in values) / count if count else 0.0
            ),
            "iou_at_0.5_percent": (
                100.0 * sum(value >= 0.5 for value in values) / count
                if count
                else 0.0
            ),
        }
    return result


def analyze_failure_modes(records):
    """Analyze one finalized split without repairing any prediction boxes."""
    records = list(records)
    evaluable = [
        record for record in records if record.get("status") != "image_error"
    ]
    parsed = [
        record
        for record in evaluable
        if record.get("status") == "ok" and record.get("pred_box") is not None
    ]

    area_ratios = []
    center_errors = []
    scale_shape_failures = 0
    for record in parsed:
        pred_box = record["pred_box"]
        gt_box = record["gt_box"]
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gt_width = gt_box[2] - gt_box[0]
        gt_height = gt_box[3] - gt_box[1]
        gt_area = gt_width * gt_height
        if gt_area <= 0:
            continue
        area_ratios.append(pred_area / gt_area)

        pred_center = (
            (pred_box[0] + pred_box[2]) / 2.0,
            (pred_box[1] + pred_box[3]) / 2.0,
        )
        gt_center = (
            (gt_box[0] + gt_box[2]) / 2.0,
            (gt_box[1] + gt_box[3]) / 2.0,
        )
        gt_diagonal = math.hypot(gt_width, gt_height)
        if gt_diagonal > 0:
            center_errors.append(
                math.hypot(
                    pred_center[0] - gt_center[0],
                    pred_center[1] - gt_center[1],
                )
                / gt_diagonal
            )
        if centered_box_iou(pred_box, gt_box) < 0.5:
            scale_shape_failures += 1

    by_size = defaultdict(list)
    by_length = defaultdict(list)
    by_relation = defaultdict(list)
    by_ordinal = defaultdict(list)
    for record in evaluable:
        iou = float(record.get("iou", 0.0))
        expression = record.get("expression", "")
        by_size[target_size_bucket(record)].append(iou)
        by_length[expression_bucket(expression)].append(iou)
        by_relation[
            "relational" if RELATION_PATTERN.search(expression) else "non_relational"
        ].append(iou)
        by_ordinal[
            "ordinal" if ORDINAL_PATTERN.search(expression) else "non_ordinal"
        ].append(iou)

    count = len(evaluable)
    parsed_count = len(parsed)
    return {
        "count": count,
        "parsed_count": parsed_count,
        "parse_rate": parsed_count / count if count else 0.0,
        "parse_rate_percent": 100.0 * parsed_count / count if count else 0.0,
        "status_counts": dict(Counter(record.get("status") for record in records)),
        "area_ratio_median": (
            statistics.median(area_ratios) if area_ratios else None
        ),
        "area_ratio_gt_10x": (
            sum(value > 10.0 for value in area_ratios) / len(area_ratios)
            if area_ratios
            else 0.0
        ),
        "area_ratio_gt_10x_percent": (
            100.0 * sum(value > 10.0 for value in area_ratios) / len(area_ratios)
            if area_ratios
            else 0.0
        ),
        "center_error_over_gt_diag_median": (
            statistics.median(center_errors) if center_errors else None
        ),
        "scale_shape_failure_at_0.5": (
            scale_shape_failures / parsed_count if parsed_count else 0.0
        ),
        "scale_shape_failure_at_0.5_percent": (
            100.0 * scale_shape_failures / parsed_count if parsed_count else 0.0
        ),
        "by_size": _bucket_accuracy(by_size),
        "by_length": _bucket_accuracy(by_length),
        "by_relation": _bucket_accuracy(by_relation),
        "by_ordinal": _bucket_accuracy(by_ordinal),
    }


def failure_table_markdown(results):
    """Render the compact Val/Test failure-mode comparison table."""
    def number(value):
        return "n/a" if value is None else f"{value:.2f}"

    lines = [
        "| Model | Area ratio median Val/Test | >10x GT Val/Test | "
        "Scale/shape failure Val/Test | Center error / GT diag Val/Test | "
        "Parse rate Val/Test |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, splits in results.items():
        val = splits["val"]
        test = splits["test"]
        lines.append(
            f"| {model} | {number(val['area_ratio_median'])} / "
            f"{number(test['area_ratio_median'])} | "
            f"{val['area_ratio_gt_10x_percent']:.2f}% / "
            f"{test['area_ratio_gt_10x_percent']:.2f}% | "
            f"{val['scale_shape_failure_at_0.5_percent']:.2f}% / "
            f"{test['scale_shape_failure_at_0.5_percent']:.2f}% | "
            f"{number(val['center_error_over_gt_diag_median'])} / "
            f"{number(test['center_error_over_gt_diag_median'])} | "
            f"{val['parse_rate_percent']:.2f}% / "
            f"{test['parse_rate_percent']:.2f}% |"
        )
    return "\n".join(lines) + "\n"
