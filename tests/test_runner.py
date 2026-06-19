import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from vlm_skyfind.runner import run


class FakeAdapter:
    def generate(self, _image_path, _prompt):
        return "[10, 20, 30, 40]"


class RunnerTest(unittest.TestCase):
    def test_run_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            Image.new("RGB", (100, 80)).save(root / "images" / "Visdrone_1.jpg")
            sample = {
                "fileName": "Visdrone_1.jpg",
                "bbox": [10, 20, 30, 40],
                "expression": "the car in the center",
            }
            (root / "Val.json").write_text(json.dumps([sample]), encoding="utf-8")
            (root / "Test.json").write_text(json.dumps([sample]), encoding="utf-8")
            output = root / "predictions.jsonl"
            args = Namespace(
                model="qwen2.5-vl-7b",
                model_path=str(root / "model"),
                data_root=str(root),
                image_dir=None,
                split="val",
                output=str(output),
                summary_output=str(root / "summary.json"),
                device="cpu",
                dtype="float32",
                max_new_tokens=16,
                prompt_variant="pixel",
                coordinate_mode="pixel",
                source_prefixes=None,
                limit=None,
                start_index=0,
                num_shards=1,
                shard_id=0,
                resume=True,
                save_tracebacks=False,
                max_consecutive_errors=5,
                attn_implementation="eager",
                internvl_max_tiles=1,
                qwen_min_pixels=None,
                qwen_max_pixels=None,
                conversation_mode=None,
            )
            with patch("vlm_skyfind.runner.create_adapter", return_value=FakeAdapter()):
                run(args)
                run(args)

            lines = output.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["iou"], 1.0)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["miou"], 1.0)


if __name__ == "__main__":
    unittest.main()
