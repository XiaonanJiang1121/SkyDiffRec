import unittest

from vlm_skyfind.prompts import build_prompt


class PromptTest(unittest.TestCase):
    def test_rsvg_prompt_matches_released_code(self):
        prompt = build_prompt("the red car", 1024, 540, "rsvg")
        self.assertEqual(
            prompt,
            "Locate it according to the following description.  the red car "
            "The output format should be like [x1, y1, x2, y2] without any "
            "other text.",
        )

    def test_pixel_prompt_is_one_sentence_with_original_size(self):
        prompt = build_prompt("the red car", 1024, 540, "pixel")
        self.assertEqual(prompt.count("."), 1)
        self.assertIn("return only", prompt)
        self.assertIn("original 1024 x 540 SkyFind image pixel coordinate system", prompt)


if __name__ == "__main__":
    unittest.main()
