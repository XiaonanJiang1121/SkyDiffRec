import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_vlm_skyfind.py"
SPEC = importlib.util.spec_from_file_location("run_vlm_skyfind", SCRIPT_PATH)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class CliTest(unittest.TestCase):
    def test_server_defaults_and_model_config(self):
        argv = [
            str(SCRIPT_PATH),
            "--model",
            "qwen2.5-vl-7b",
            "--split",
            "val",
            "--output",
            "predictions/val.jsonl",
        ]
        with patch.object(sys, "argv", argv):
            args = CLI.parse_args()
        self.assertEqual(args.data_root, "/root/autodl-tmp/BioLoc/data/SkyFind_data")
        self.assertEqual(args.prompt_variant, "rsvg")
        self.assertEqual(
            args.model_path,
            "/root/autodl-tmp/VLMSkyFind/models/Qwen2.5-VL-7B-Instruct",
        )


if __name__ == "__main__":
    unittest.main()
