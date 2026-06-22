import unittest

from vlm_skyfind.mixed_coordinates import (
    convert_internvl_official,
    convert_uncontracted_vlm,
)


class MixedCoordinateProtocolTest(unittest.TestCase):
    def test_internvl_official_fractional_branch(self):
        box, mode = convert_internvl_official(
            [0.25, 0.5, 0.75, 1.0], 200, 100
        )
        self.assertEqual(mode, "internvl_official_normalized_1")
        self.assertEqual(box, [50.0, 50.0, 150.0, 100.0])

    def test_internvl_official_1000_branch_does_not_clamp(self):
        box, mode = convert_internvl_official(
            [900, 800, 1100, 1200], 200, 100
        )
        self.assertEqual(mode, "internvl_official_normalized_1000")
        self.assertEqual(box, [180.0, 80.0, 220.0, 120.0])

    def test_uncontracted_model_accepts_unambiguous_fraction(self):
        box, mode = convert_uncontracted_vlm(
            [0.1, 0.2, 0.3, 0.4], 1000, 500
        )
        self.assertEqual(mode, "normalized_1")
        self.assertEqual(box, [100.0, 100.0, 300.0, 200.0])

    def test_uncontracted_model_rejects_bare_ambiguous_values(self):
        box, mode = convert_uncontracted_vlm([10, 20, 30, 40], 1000, 500)
        self.assertIsNone(box)
        self.assertEqual(mode, "ambiguous")

    def test_uncontracted_model_accepts_explicit_percent(self):
        box, mode = convert_uncontracted_vlm(
            [10, 20, 30, 40], 1000, 500, "percent coordinates [10,20,30,40]"
        )
        self.assertEqual(mode, "explicit_percent_100")
        self.assertEqual(box, [100.0, 100.0, 300.0, 200.0])

    def test_sensitivity_policy_is_explicitly_labeled(self):
        box, mode = convert_uncontracted_vlm(
            [100, 200, 300, 400],
            1000,
            500,
            ambiguous_policy="normalized_1000",
        )
        self.assertEqual(mode, "sensitivity_normalized_1000")
        self.assertEqual(box, [100.0, 100.0, 300.0, 200.0])

    def test_reversed_box_is_not_repaired(self):
        box, mode = convert_uncontracted_vlm(
            [0.8, 0.2, 0.1, 0.4], 1000, 500
        )
        self.assertEqual(mode, "normalized_1")
        self.assertIsNone(box)


if __name__ == "__main__":
    unittest.main()
