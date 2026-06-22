import unittest

from vlm_skyfind.coordinates import resolve_coordinate_mode


class CoordinateProfileTest(unittest.TestCase):
    def test_qwen_uses_resized_input_pixels(self):
        mode, _ = resolve_coordinate_mode("qwen2.5-vl-7b", "model_native")
        self.assertEqual(mode, "qwen_resized_pixel")

    def test_internvl_uses_observed_mixed_normalized_scales(self):
        mode, _ = resolve_coordinate_mode("internvl2.5-8b", "model_native")
        self.assertEqual(mode, "internvl_official_mixed")

    def test_explicit_mode_overrides_profile(self):
        mode, basis = resolve_coordinate_mode("internvl2.5-8b", "pixel")
        self.assertEqual(mode, "pixel")
        self.assertEqual(basis, "explicit command-line setting")


if __name__ == "__main__":
    unittest.main()
