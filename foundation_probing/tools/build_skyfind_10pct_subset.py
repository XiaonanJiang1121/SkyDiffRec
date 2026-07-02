#!/usr/bin/env python3
"""Build the stratified 10% SkyFind subset for foundation probing."""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

SPLIT_FILES = {"val": "Val.json", "test": "Test.json"}


def source_name(file_name):
    stem = Path(file_name).stem
    return stem.split("_", 1)[0] if "_" in stem else "unknown"


def sample_grouped(records, fraction, seed):
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[source_name(record["fileName"])].append((index, record))

    selected = []
    per_source = {}
    rng = random.Random(seed)
    for source in sorted(grouped):
        items = grouped[source]
        count = max(1, int(len(items) * fraction + 0.5))
        picked = rng.sample(items, count)
        picked.sort(key=lambda item: item[0])
        selected.extend(picked)
        per_source[source] = {"total": len(items), "selected": count}

    selected.sort(key=lambda item: item[0])
    return selected, per_source


def ensure_image(src, dst, mode):
    if mode == "none":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def build_subset(args):
    skyfind_root = Path(args.skyfind_root).resolve()
    output_root = Path(args.output_root).resolve()
    annotation_dir = output_root / "annotations"
    image_dir = output_root / "images"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "fraction": args.fraction,
        "skyfind_root": str(skyfind_root),
        "image_mode": args.image_mode,
        "splits": {},
    }

    for split, file_name in SPLIT_FILES.items():
        records = json.loads((skyfind_root / file_name).read_text(encoding="utf-8"))
        selected, per_source = sample_grouped(records, args.fraction, args.seed)

        subset_records = []
        missing_images = []
        for original_index, record in selected:
            local_record = dict(record)
            local_record["original_split"] = split
            local_record["original_index"] = original_index
            local_record["source"] = source_name(record["fileName"])
            local_record["imagePath"] = str(Path("..") / "images" / record["fileName"])
            subset_records.append(local_record)

            src = skyfind_root / "images" / record["fileName"]
            dst = image_dir / record["fileName"]
            if not src.exists():
                missing_images.append(record["fileName"])
                continue
            ensure_image(src, dst, args.image_mode)

        output_name = f"{split.capitalize()}_10pct.json"
        (annotation_dir / output_name).write_text(
            json.dumps(subset_records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        manifest["splits"][split] = {
            "source_file": file_name,
            "total_records": len(records),
            "selected_records": len(subset_records),
            "unique_images": len({record["fileName"] for record in subset_records}),
            "missing_images": missing_images,
            "per_source": per_source,
        }

    (annotation_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "seed": manifest["seed"],
        "fraction": manifest["fraction"],
        "image_mode": manifest["image_mode"],
        "splits": {
            split: {
                "total_records": info["total_records"],
                "selected_records": info["selected_records"],
                "unique_images": info["unique_images"],
                "missing_image_count": len(info["missing_images"]),
                "per_source": info["per_source"],
            }
            for split, info in manifest["splits"].items()
        },
    }
    (output_root / "subset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skyfind-root",
        default="../BioLoc/data/SkyFind_data",
        help="Path to the original SkyFind_data directory.",
    )
    parser.add_argument(
        "--output-root",
        default="data/foundation_probe_10pct",
        help="Output directory under DiffusionSkyFind.",
    )
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--image-mode",
        choices=("none", "symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize selected images locally.",
    )
    return parser.parse_args()


def main():
    summary = build_subset(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

