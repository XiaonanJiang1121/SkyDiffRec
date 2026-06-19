import unittest

from vlm_skyfind.prompts import build_prompt


class PromptTest(unittest.TestCase):
    def test_rsvg_prompt_is_exactly_one_sentence(self):
        prompt = build_prompt("the red car", 1024, 540, "rsvg")
        self.assertEqual(
            prompt,
            "Locate the object referred to by 'the red car' and return its box coordinates (x1, y1, x2, y2).",
        )

    def test_pixel_prompt_is_one_sentence_with_original_size(self):
        prompt = build_prompt("the red car", 1024, 540, "pixel")
        self.assertEqual(prompt.count("."), 1)
        self.assertIn("original 1024 x 540 SkyFind image pixel coordinate system", prompt)


if __name__ == "__main__":
    unittest.main()
