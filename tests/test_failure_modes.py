import unittest

from vlm_skyfind.failure_modes import (
    analyze_failure_modes,
    centered_box_iou,
    failure_table_markdown,
)


def record(sample_id, status, pred_box, iou, expression="the object"):
    return {
        "sample_id": sample_id,
        "split": "val",
        "model": "model",
        "status": status,
        "pred_box": pred_box,
        "gt_box": [40, 40, 60, 60],
        "iou": iou,
        "width": 100,
        "height": 100,
        "expression": expression,
    }


class FailureModeTest(unittest.TestCase):
    def test_centered_iou_measures_size_limit(self):
        self.assertEqual(centered_box_iou([0, 0, 80, 80], [40, 40, 60, 60]), 0.0625)

    def test_analyze_failure_modes(self):
        records = [
            record("val:0", "ok", [40, 40, 60, 60], 1.0, "the first object"),
            record("val:1", "ok", [10, 10, 90, 90], 0.0625, "object next to car"),
            record("val:2", "parse_error", None, 0.0),
        ]
        result = analyze_failure_modes(records)
        self.assertEqual(result["parsed_count"], 2)
        self.assertAlmostEqual(result["parse_rate"], 2 / 3)
        self.assertEqual(result["area_ratio_median"], 8.5)
        self.assertEqual(result["area_ratio_gt_10x_percent"], 50.0)
        self.assertEqual(result["scale_shape_failure_at_0.5_percent"], 50.0)
        self.assertEqual(result["by_relation"]["relational"]["count"], 1)
        self.assertEqual(result["by_ordinal"]["ordinal"]["count"], 1)

    def test_markdown_table(self):
        split = {
            "area_ratio_median": 2.0,
            "area_ratio_gt_10x_percent": 10.0,
            "scale_shape_failure_at_0.5_percent": 20.0,
            "center_error_over_gt_diag_median": 3.0,
            "parse_rate_percent": 90.0,
        }
        table = failure_table_markdown(
            {"model": {"val": split, "test": split}}
        )
        self.assertIn("Area ratio median Val/Test", table)
        self.assertIn("| model | 2.00 / 2.00 |", table)


if __name__ == "__main__":
    unittest.main()
