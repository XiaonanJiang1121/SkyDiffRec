import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from vlm_skyfind.runner import run


class FakeAdapter:
    def __init__(self):
        self.calls = 0

    def generate(self, _image_path, _prompt):
        self.calls += 1
        return "[10, 20, 30, 40]"


class RunnerTest(unittest.TestCase):
    def test_val_and_test_skip_bad_images_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images" / "SeaDronesSee_bad.jpg").write_bytes(b"not an image")
            Image.new("RGB", (100, 80)).save(root / "images" / "Visdrone_1.jpg")
            bad_sample = {
                "fileName": "SeaDronesSee_bad.jpg",
                "bbox": [10, 20, 30, 40],
                "expression": "the person in the water",
            }
            valid_sample = {
                "fileName": "Visdrone_1.jpg",
                "bbox": [10, 20, 30, 40],
                "expression": "the car in the center",
            }
            samples = [bad_sample, valid_sample]
            (root / "Val.json").write_text(json.dumps(samples), encoding="utf-8")
            (root / "Test.json").write_text(json.dumps(samples), encoding="utf-8")
            common_args = dict(
                model="qwen2.5-vl-7b",
                model_path=str(root / "model"),
                data_root=str(root),
                image_dir=None,
                device="cpu",
                dtype="float32",
                max_new_tokens=16,
                prompt_variant="rsvg",
                coordinate_mode="pixel",
                source_prefixes=None,
                limit=None,
                start_index=0,
                resume=True,
                save_tracebacks=False,
                max_consecutive_errors=5,
                attn_implementation="eager",
                internvl_max_tiles=1,
                conversation_mode=None,
            )
            adapter = FakeAdapter()
            with patch("vlm_skyfind.runner.create_adapter", return_value=adapter):
                for split in ("val", "test"):
                    with self.subTest(split=split):
                        output = root / f"{split}_predictions.jsonl"
                        summary_path = root / f"{split}_summary.json"
                        args = Namespace(
                            **common_args,
                            split=split,
                            output=str(output),
                            summary_output=str(summary_path),
                        )
                        run(args)
                        run(args)

                        lines = output.read_text(encoding="utf-8").strip().splitlines()
                        self.assertEqual(len(lines), 2)
                        records = [json.loads(line) for line in lines]
                        self.assertEqual(records[0]["status"], "image_error")
                        self.assertEqual(records[1]["status"], "ok")
                        self.assertEqual(records[1]["iou"], 1.0)
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        self.assertEqual(summary["miou"], 1.0)
                        self.assertEqual(summary["count"], 1)
                        self.assertEqual(summary["record_count"], 2)
                        self.assertEqual(summary["skipped_image_count"], 1)
            self.assertEqual(adapter.calls, 2)


if __name__ == "__main__":
    unittest.main()
