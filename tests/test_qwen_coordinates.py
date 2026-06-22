import unittest

from vlm_skyfind.qwen_coordinates import processed_size, restore_coordinates


CONFIG = {
    "min_pixels": 3136,
    "max_pixels": 12845056,
    "patch_size": 14,
    "merge_size": 2,
    "factor": 28,
}


class QwenCoordinateTest(unittest.TestCase):
    def test_processed_size_matches_official_factor_rounding(self):
        self.assertEqual(processed_size(1920, 1080, CONFIG), (1932, 1092))

    def test_restore_resized_pixels_to_original_pixels(self):
        box, processed_width, processed_height = restore_coordinates(
            [966, 546, 1932, 1092], 1920, 1080, CONFIG
        )
        self.assertEqual((processed_width, processed_height), (1932, 1092))
        self.assertEqual(box, [960.0, 540.0, 1920.0, 1080.0])

    def test_large_image_is_downscaled_before_restoration(self):
        processed_width, processed_height = processed_size(8000, 4000, CONFIG)
        self.assertLessEqual(
            processed_width * processed_height, CONFIG["max_pixels"]
        )
        box, _, _ = restore_coordinates(
            [0, 0, processed_width, processed_height], 8000, 4000, CONFIG
        )
        self.assertEqual(box, [0.0, 0.0, 8000.0, 4000.0])


if __name__ == "__main__":
    unittest.main()
