import unittest

from vlm_skyfind.metrics import summarize


def record(sample_id, status, iou):
    return {
        "sample_id": sample_id,
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


if __name__ == "__main__":
    unittest.main()
