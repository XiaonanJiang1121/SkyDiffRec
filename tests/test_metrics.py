import unittest

from vlm_skyfind.metrics import (
    summarize,
    summarize_table4,
    validate_final_protocol,
)


def record(sample_id, status, iou):
    return {
        "sample_id": sample_id,
        "split": sample_id.split(":", 1)[0],
        "model": "model",
        "status": status,
        "iou": iou,
        "source": "Visdrone",
        "expression": "one two three",
        "gt_box": [0, 0, 10, 10],
        "width": 100,
        "height": 100,
    }


class MetricsTest(unittest.TestCase):
    def test_parse_failures_count_as_zero(self):
        summary = summarize([
            record("val:0", "ok", 0.8),
            record("val:1", "parse_error", 0.0),
        ])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["parse_rate"], 0.5)
        self.assertEqual(summary["miou"], 0.4)
        self.assertEqual(summary["acc_0.5"], 0.5)
        self.assertEqual(summary["iou_at_0.5"], 0.5)
        self.assertEqual(summary["iou_at_mean"], 0.4)

    def test_image_errors_are_excluded(self):
        valid = record("test:0", "ok", 0.8)
        image_error = {"sample_id": "test:1", "status": "image_error"}
        summary = summarize([valid, image_error])
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["skipped_image_count"], 1)
        self.assertEqual(summary["miou"], 0.8)

    def test_skyfind_iou_at_mean_averages_five_thresholds(self):
        summary = summarize([
            record("val:0", "ok", 0.95),
            record("val:1", "ok", 0.75),
            record("val:2", "ok", 0.55),
            record("val:3", "ok", 0.45),
            record("val:4", "parse_error", 0.0),
        ])
        self.assertEqual(
            summary["iou_threshold_accuracy"],
            {"0.5": 0.6, "0.6": 0.4, "0.7": 0.4, "0.8": 0.2, "0.9": 0.2},
        )
        self.assertAlmostEqual(summary["iou_at_mean"], 0.36)
        self.assertAlmostEqual(summary["iou_at_mean_percent"], 36.0)

    def test_table4_average_is_unweighted_across_splits(self):
        val = [record("val:0", "ok", 0.95)]
        test = [
            record("test:0", "ok", 0.55),
            record("test:1", "parse_error", 0.0),
        ]
        table = summarize_table4("model", val, test)
        self.assertEqual(table["val"]["iou_at_0.5"], 1.0)
        self.assertEqual(table["test"]["iou_at_0.5"], 0.5)
        self.assertEqual(table["average"]["iou_at_0.5"], 0.75)
        self.assertAlmostEqual(table["average"]["iou_at_mean"], 0.55)

    def test_table4_rejects_wrong_split(self):
        with self.assertRaisesRegex(ValueError, "Expected test record"):
            summarize_table4(
                "model",
                [record("val:0", "ok", 0.8)],
                [record("val:1", "ok", 0.8)],
            )

    def test_final_qwen_protocol_rejects_provisional_pixels(self):
        item = record("val:0", "ok", 0.8)
        item["model"] = "qwen2.5-vl-7b"
        item["coordinate_mode_resolved"] = "pixel"
        with self.assertRaisesRegex(ValueError, "qwen_resized_pixel"):
            validate_final_protocol("qwen2.5-vl-7b", [item])

    def test_final_qwen_protocol_requires_strict_boxes(self):
        item = record("val:0", "ok", 0.8)
        item.update(
            {
                "model": "qwen2.5-vl-7b",
                "coordinate_mode_resolved": "qwen_resized_pixel",
                "box_validation": "sanitize_reorder_and_clamp",
            }
        )
        with self.assertRaisesRegex(ValueError, "strict_xyxy"):
            validate_final_protocol("qwen2.5-vl-7b", [item])

        item["box_validation"] = "strict_xyxy_no_reorder_no_clamp"
        validate_final_protocol("qwen2.5-vl-7b", [item])


if __name__ == "__main__":
    unittest.main()
