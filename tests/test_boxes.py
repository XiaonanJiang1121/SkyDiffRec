import unittest

from vlm_skyfind.boxes import box_iou, parse_prediction


class BoxParsingTest(unittest.TestCase):
    def test_pixel_bracket_box(self):
        box, mode = parse_prediction("[10, 20, 30, 40]", 100, 80, "pixel")
        self.assertEqual(box, [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(mode, "pixel")

    def test_two_point_box(self):
        box, _ = parse_prediction("bbox: [(10, 20), (30, 40)]", 100, 80, "pixel")
        self.assertEqual(box, [10.0, 20.0, 30.0, 40.0])

    def test_normalized_box(self):
        box, mode = parse_prediction("[0.1, 0.25, 0.5, 0.75]", 200, 100, "normalized_1")
        self.assertEqual(box, [20.0, 25.0, 100.0, 75.0])
        self.assertEqual(mode, "normalized_1")

    def test_reorders_and_clamps(self):
        box, _ = parse_prediction("[120, 70, -10, 10]", 100, 80, "pixel")
        self.assertEqual(box, [0.0, 10.0, 100.0, 70.0])

    def test_iou(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)


if __name__ == "__main__":
    unittest.main()
