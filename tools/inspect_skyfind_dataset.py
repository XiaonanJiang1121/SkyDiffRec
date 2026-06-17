#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, UnidentifiedImageError


SKYFIND_SPLIT_FILES = {
    "train": "Train.json",
    "val": "Val.json",
    "test": "Test.json",
}


def clamp_bbox_xyxy(bbox_xyxy, image_width, image_height):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    original_bbox = [x1, y1, x2, y2]
    x1 = min(max(x1, 0.0), float(image_width - 1))
    y1 = min(max(y1, 0.0), float(image_height - 1))
    x2 = min(max(x2, 0.0), float(image_width - 1))
    y2 = min(max(y2, 0.0), float(image_height - 1))
    clamped_bbox = [x1, y1, x2, y2]
    was_clamped = clamped_bbox != original_bbox
    is_valid = (x2 > x1) and (y2 > y1)
    return original_bbox, clamped_bbox, was_clamped, is_valid


def audit_raw_split(data_root, split):
    split_file = SKYFIND_SPLIT_FILES[split]
    annotation_path = data_root / split_file
    image_root = data_root / "images"

    with annotation_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    unique_file_names = sorted({sample["fileName"] for sample in samples})
    image_info_by_file = {}
    missing_files = set()
    corrupt_files = set()

    for file_name in unique_file_names:
        image_path = image_root / file_name
        if not image_path.is_file():
            missing_files.add(file_name)
            continue
        try:
            with Image.open(image_path) as image:
                image_info_by_file[file_name] = image.size
                image.load()
        except (UnidentifiedImageError, OSError, ValueError):
            corrupt_files.add(file_name)

    kept_samples = 0
    clamped_bbox_samples = 0
    invalid_bbox_samples = 0
    non_numeric_file_name_samples = 0
    skipped_missing_samples = 0
    skipped_corrupt_samples = 0

    for sample in samples:
        file_name = sample["fileName"]
        if file_name in missing_files:
            skipped_missing_samples += 1
            continue
        if file_name in corrupt_files:
            skipped_corrupt_samples += 1
            continue

        image_width, image_height = image_info_by_file[file_name]
        _, _, was_clamped, is_valid = clamp_bbox_xyxy(
            sample["bbox"],
            image_width=image_width,
            image_height=image_height,
        )
        if not is_valid:
            invalid_bbox_samples += 1
            continue
        if was_clamped:
            clamped_bbox_samples += 1
        if not Path(file_name).stem.isdigit():
            non_numeric_file_name_samples += 1
        kept_samples += 1

    print(
        "[raw-audit] split={split} raw_samples={raw_samples} kept_samples={kept_samples} "
        "unique_images={unique_images} validated_images={validated_images} "
        "missing_images={missing_images} corrupt_images={corrupt_images} "
        "skipped_missing_samples={skipped_missing_samples} "
        "skipped_corrupt_samples={skipped_corrupt_samples} "
        "invalid_bbox_samples={invalid_bbox_samples} "
        "clamped_bbox_samples={clamped_bbox_samples} "
        "non_numeric_file_name_samples={non_numeric_file_name_samples}".format(
            split=split,
            raw_samples=len(samples),
            kept_samples=kept_samples,
            unique_images=len(unique_file_names),
            validated_images=len(image_info_by_file),
            missing_images=len(missing_files),
            corrupt_images=len(corrupt_files),
            skipped_missing_samples=skipped_missing_samples,
            skipped_corrupt_samples=skipped_corrupt_samples,
            invalid_bbox_samples=invalid_bbox_samples,
            clamped_bbox_samples=clamped_bbox_samples,
            non_numeric_file_name_samples=non_numeric_file_name_samples,
        )
    )


def inspect_dataset_instance(repo_root, data_root, split, bert_model, max_query_len, imsize, limit, transform_split, allowed_prefixes):
    baseline_root = repo_root / "baselines" / "diffusionrec_original"
    sys.path.insert(0, str(baseline_root))

    import datasets as diffusionrec_datasets
    from datasets.data_loader import TransVGDataset

    args = SimpleNamespace(
        imsize=imsize,
        aug_blur=False,
        aug_crop=False,
        aug_scale=False,
        aug_translate=False,
    )
    transform = diffusionrec_datasets.make_transforms(args, transform_split)
    dataset = TransVGDataset(
        data_root=str(data_root),
        split_root="data",
        dataset="skyfind",
        split=split,
        transform=transform,
        max_query_len=max_query_len,
        bert_model=bert_model,
        skyfind_allowed_prefixes=allowed_prefixes,
    )

    print("[dataset-init] len={} split={} transform_split={}".format(len(dataset), split, transform_split))
    print("[dataset-init] stats={}".format(dataset.skyfind_stats))

    inspect_count = min(limit, len(dataset))
    for idx in range(inspect_count):
        item = dataset[idx]
        img, img_id, img_mask, word_id, word_selection, batch_a, word_mask, bbox = item
        meta = dataset.skyfind_meta_by_img_id[int(img_id)]
        print(
            "[sample] idx={idx} img_id={img_id} file_name={file_name} "
            "image_shape={image_shape} mask_shape={mask_shape} "
            "bbox_norm={bbox} token_len={token_len} selected_len={selected_len} "
            "token_truncated_count={token_truncated_count} "
            "bbox_was_clamped={bbox_was_clamped}".format(
                idx=idx,
                img_id=int(img_id),
                file_name=meta["file_name"],
                image_shape=tuple(img.shape),
                mask_shape=tuple(img_mask.shape),
                bbox=np_array_to_list(bbox),
                token_len=int(sum(word_mask)),
                selected_len=int(sum(word_selection)),
                token_truncated_count=batch_a.get("token_truncated_count", 0),
                bbox_was_clamped=meta["bbox_was_clamped"],
            )
        )


def np_array_to_list(array_like):
    try:
        return array_like.tolist()
    except AttributeError:
        return list(array_like)


def main():
    parser = argparse.ArgumentParser(description="Inspect SkyFind dataset adaptation for DiffusionREC.")
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=sorted(SKYFIND_SPLIT_FILES))
    parser.add_argument("--bert_model", default=None, type=str)
    parser.add_argument("--max_query_len", default=128, type=int)
    parser.add_argument("--imsize", default=640, type=int)
    parser.add_argument("--limit", default=5, type=int)
    parser.add_argument("--transform_split", default="val", choices=("train", "val"))
    parser.add_argument("--allowed_prefixes", nargs='*', default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    audit_raw_split(args.data_root, args.split)
    if args.bert_model:
        inspect_dataset_instance(
            repo_root=repo_root,
            data_root=args.data_root,
            split=args.split,
            bert_model=args.bert_model,
            max_query_len=args.max_query_len,
            imsize=args.imsize,
            limit=args.limit,
            transform_split=args.transform_split,
            allowed_prefixes=args.allowed_prefixes,
        )


if __name__ == "__main__":
    main()
