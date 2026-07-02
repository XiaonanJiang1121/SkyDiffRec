#!/usr/bin/env python3
"""Run Experiment 4: full-image versus crop-image SD attention probe.

This experiment tests whether full-image 512 processing is the bottleneck for
SkyFind targets by comparing the same full expression under three image
contexts: full image, GT-centered crop, and random wrong crop.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from run_exp1_tiny_target_retention import GUIDANCE_SCALE, GRID_RESOLUTIONS, IMAGE_SIZE, NUM_DDIM_STEPS
    from run_exp2_sd_cross_attention_smoke import (
        MAP_TYPES,
        CrossAttentionStore,
        aggregate_cross_attention,
        build_prompt_records,
        compute_heatmap_metrics,
        load_sd,
        map_type_indices,
        prepare_record,
        read_json,
        register_cross_attention_store,
        restore_attention_processors,
        run_denoising_with_attention,
        save_heatmap,
        select_smoke_records,
        summarize_values,
        token_heatmap,
        write_jsonl,
    )
    from run_exp3_sd_self_attention_structure import (
        SelfAttentionRowStore,
        build_seed_indices,
        compute_self_metrics,
        register_self_attention_store,
        run_denoising as run_self_denoising,
    )
except ImportError:
    from foundation_probing.tools.run_exp1_tiny_target_retention import (
        GUIDANCE_SCALE,
        GRID_RESOLUTIONS,
        IMAGE_SIZE,
        NUM_DDIM_STEPS,
    )
    from foundation_probing.tools.run_exp2_sd_cross_attention_smoke import (
        MAP_TYPES,
        CrossAttentionStore,
        aggregate_cross_attention,
        build_prompt_records,
        compute_heatmap_metrics,
        load_sd,
        map_type_indices,
        prepare_record,
        read_json,
        register_cross_attention_store,
        restore_attention_processors,
        run_denoising_with_attention,
        save_heatmap,
        select_smoke_records,
        summarize_values,
        token_heatmap,
        write_jsonl,
    )
    from foundation_probing.tools.run_exp3_sd_self_attention_structure import (
        SelfAttentionRowStore,
        build_seed_indices,
        compute_self_metrics,
        register_self_attention_store,
        run_denoising as run_self_denoising,
    )


CONTEXT_TYPES = ("full_image", "gt_crop", "random_wrong_crop")
PROBE_TYPES = ("cross", "self")
METRIC_KEYS = (
    "pointing_game",
    "top1_hit",
    "top5_hit",
    "top10_hit",
    "gt_attention_ratio",
    "gt_area_ratio",
    "attention_enrichment",
    "peak_center_distance",
    "entropy",
    "peakness",
)
RECON_KEYS = (
    "reconstruction_full_mse",
    "reconstruction_full_psnr",
    "reconstruction_target_mse",
    "reconstruction_target_psnr",
)


def safe_stem(record, context_type, res):
    index = record.get("original_index", "unknown")
    return f"val_{index}_{Path(record['fileName']).stem}_{context_type}_r{res}"


def clip_box(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]


def valid_box(box):
    return box[2] > box[0] and box[3] > box[1]


def expand_crop_box(box, image_width, image_height, scale, square):
    x1, y1, x2, y2 = [float(value) for value in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = max(1.0, (x2 - x1) * scale)
    height = max(1.0, (y2 - y1) * scale)
    if square:
        side = max(width, height)
        width = side
        height = side
    crop = [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]
    if crop[0] < 0:
        crop[2] -= crop[0]
        crop[0] = 0.0
    if crop[1] < 0:
        crop[3] -= crop[1]
        crop[1] = 0.0
    if crop[2] > image_width:
        delta = crop[2] - image_width
        crop[0] -= delta
        crop[2] = float(image_width)
    if crop[3] > image_height:
        delta = crop[3] - image_height
        crop[1] -= delta
        crop[3] = float(image_height)
    return clip_box(crop, image_width, image_height)


def box_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def choose_random_wrong_crop(gt_box, crop_width, crop_height, image_width, image_height, seed):
    max_x = max(0.0, float(image_width) - crop_width)
    max_y = max(0.0, float(image_height) - crop_height)
    rng = np.random.default_rng(seed)
    best_crop = None
    best_iou = float("inf")
    for _ in range(128):
        x1 = float(rng.uniform(0.0, max_x)) if max_x > 0 else 0.0
        y1 = float(rng.uniform(0.0, max_y)) if max_y > 0 else 0.0
        crop = [x1, y1, x1 + crop_width, y1 + crop_height]
        iou = box_iou(crop, gt_box)
        if iou == 0.0:
            return crop
        if iou < best_iou:
            best_iou = iou
            best_crop = crop

    corners = [
        [0.0, 0.0, crop_width, crop_height],
        [max_x, 0.0, max_x + crop_width, crop_height],
        [0.0, max_y, crop_width, max_y + crop_height],
        [max_x, max_y, max_x + crop_width, max_y + crop_height],
    ]
    return min(corners + [best_crop], key=lambda crop: box_iou(crop, gt_box))


def map_box_from_crop_to_512(box, crop_box):
    crop_w = crop_box[2] - crop_box[0]
    crop_h = crop_box[3] - crop_box[1]
    return [
        (box[0] - crop_box[0]) / crop_w * IMAGE_SIZE,
        (box[1] - crop_box[1]) / crop_h * IMAGE_SIZE,
        (box[2] - crop_box[0]) / crop_w * IMAGE_SIZE,
        (box[3] - crop_box[1]) / crop_h * IMAGE_SIZE,
    ]


def crop_to_512(image, crop_box):
    left = int(math.floor(crop_box[0]))
    top = int(math.floor(crop_box[1]))
    right = int(math.ceil(crop_box[2]))
    bottom = int(math.ceil(crop_box[3]))
    return image.crop((left, top, right, bottom)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def build_contexts(item, image_root, args):
    record = item["record"]
    image_path = image_root / record["fileName"]
    with Image.open(image_path) as image:
        image.load()
        original = image.convert("RGB")

    contexts = []
    if "full_image" in args.context_types:
        contexts.append(
            {
                "context_type": "full_image",
                "image_512": item["image_512"],
                "eval_box_512": item["bbox_512"],
                "crop_box_original": [0.0, 0.0, float(item["image_width"]), float(item["image_height"])],
                "target_present": True,
                "random_crop_iou_with_gt": None,
            }
        )

    gt_crop_box = expand_crop_box(
        item["gt_box_clipped"],
        item["image_width"],
        item["image_height"],
        args.crop_context_scale,
        args.square_crop,
    )
    gt_crop_eval_box = map_box_from_crop_to_512(item["gt_box_clipped"], gt_crop_box)
    if "gt_crop" in args.context_types:
        contexts.append(
            {
                "context_type": "gt_crop",
                "image_512": crop_to_512(original, gt_crop_box),
                "eval_box_512": gt_crop_eval_box,
                "crop_box_original": gt_crop_box,
                "target_present": True,
                "random_crop_iou_with_gt": None,
            }
        )

    if "random_wrong_crop" in args.context_types:
        crop_w = gt_crop_box[2] - gt_crop_box[0]
        crop_h = gt_crop_box[3] - gt_crop_box[1]
        seed = args.seed + int(record.get("original_index", 0)) * 917
        wrong_crop_box = choose_random_wrong_crop(
            item["gt_box_clipped"],
            crop_w,
            crop_h,
            item["image_width"],
            item["image_height"],
            seed,
        )
        contexts.append(
            {
                "context_type": "random_wrong_crop",
                "image_512": crop_to_512(original, wrong_crop_box),
                "eval_box_512": gt_crop_eval_box,
                "crop_box_original": wrong_crop_box,
                "target_present": False,
                "random_crop_iou_with_gt": box_iou(wrong_crop_box, item["gt_box_clipped"]),
            }
        )

    return contexts


def to_float01(image):
    return np.asarray(image).astype(np.float32) / 255.0


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def crop_bounds_512(box_512):
    left = max(0, min(IMAGE_SIZE, int(math.floor(box_512[0]))))
    top = max(0, min(IMAGE_SIZE, int(math.floor(box_512[1]))))
    right = max(0, min(IMAGE_SIZE, int(math.ceil(box_512[2]))))
    bottom = max(0, min(IMAGE_SIZE, int(math.ceil(box_512[3]))))
    return left, top, right, bottom


def reconstruction_metrics(original_512, reconstructed_512, eval_box_512, target_present):
    original = to_float01(original_512)
    reconstructed = to_float01(reconstructed_512)
    full_mse = float(np.mean((original - reconstructed) ** 2))
    metrics = {
        "reconstruction_full_mse": full_mse,
        "reconstruction_full_psnr": psnr_from_mse(full_mse),
        "reconstruction_target_mse": None,
        "reconstruction_target_psnr": None,
    }
    if target_present and valid_box(eval_box_512):
        left, top, right, bottom = crop_bounds_512(eval_box_512)
        if right > left and bottom > top:
            orig_crop = original[top:bottom, left:right]
            rec_crop = reconstructed[top:bottom, left:right]
            target_mse = float(np.mean((orig_crop - rec_crop) ** 2))
            metrics["reconstruction_target_mse"] = target_mse
            metrics["reconstruction_target_psnr"] = psnr_from_mse(target_mse)
    return metrics


def run_cross_probe(pipe, inverter, base_attn_processors, item, context, prompt_record, entities, args):
    import torch

    processor_names = list(base_attn_processors.keys())
    store = CrossAttentionStore()
    try:
        register_cross_attention_store(pipe, store, processor_names)
        run_denoising_with_attention(inverter, context["x_t"], context["uncond_embeddings"])
    finally:
        restore_attention_processors(pipe, base_attn_processors)
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    token_indices = map_type_indices(prompt_record)
    records = []
    record = item["record"]
    for res in args.resolutions:
        attention_map = aggregate_cross_attention(store, int(res))
        if attention_map is None:
            continue
        for map_type in MAP_TYPES:
            heatmap = token_heatmap(attention_map, token_indices[map_type])
            if heatmap is None:
                metrics = None
                heatmap_paths = None
            else:
                metrics = compute_heatmap_metrics(heatmap, context["eval_box_512"])
                heatmap_paths = None
                if args.save_heatmaps and map_type == "target_entity":
                    heatmap_paths = save_heatmap(
                        heatmap,
                        context["eval_box_512"],
                        Path(args.output_dir) / "heatmaps" / context["context_type"],
                        safe_stem(record, context["context_type"], int(res)),
                        context["image_512"],
                    )

            records.append(
                {
                    "sample_id": f"val:{record.get('original_index', record.get('index', 'unknown'))}",
                    "split": "val",
                    "original_index": record.get("original_index"),
                    "file_name": record["fileName"],
                    "source": record.get("source"),
                    "probe_type": "cross",
                    "context_type": context["context_type"],
                    "target_present": context["target_present"],
                    "random_crop_iou_with_gt": context["random_crop_iou_with_gt"],
                    "crop_context_scale": args.crop_context_scale,
                    "square_crop": args.square_crop,
                    "crop_box_original": context["crop_box_original"],
                    "image_width": item["image_width"],
                    "image_height": item["image_height"],
                    "expression": record["expression"],
                    "prompt_variant": prompt_record["prompt_variant"],
                    "prompt": prompt_record["prompt"],
                    "map_type": map_type,
                    "resolution": int(res),
                    "token_indices": token_indices[map_type],
                    "tokenizer": prompt_record["tokenizer"],
                    "entities": entities,
                    "target_entity": entities[0] if entities else None,
                    "gt_box_raw": item["gt_box_raw"],
                    "gt_box_clipped": item["gt_box_clipped"],
                    "bbox_512_full_image": item["bbox_512"],
                    "eval_box_512": context["eval_box_512"],
                    "target_width_512_full_image": item["target_width_512"],
                    "target_height_512_full_image": item["target_height_512"],
                    "target_width_512_context": context["eval_box_512"][2] - context["eval_box_512"][0],
                    "target_height_512_context": context["eval_box_512"][3] - context["eval_box_512"][1],
                    "target_size_bucket": item["target_size_bucket"],
                    "reconstruction_metrics": context["reconstruction_metrics"],
                    "metrics": metrics,
                    "heatmap_paths": heatmap_paths,
                }
            )
    return records


def run_self_probe(pipe, inverter, base_attn_processors, item, context, prompt_record, entities, args):
    import torch

    processor_names = list(base_attn_processors.keys())
    seed = args.seed + int(item["record"].get("original_index", 0)) * 1009
    seed_indices = build_seed_indices(
        context["eval_box_512"],
        args.resolutions,
        args.self_control_types,
        seed,
    )
    store = SelfAttentionRowStore(seed_indices, args.resolutions)
    try:
        register_self_attention_store(pipe, store, processor_names)
        run_self_denoising(inverter, context["x_t"], context["uncond_embeddings"])
    finally:
        restore_attention_processors(pipe, base_attn_processors)
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    record = item["record"]
    records = []
    for control_type in args.self_control_types:
        for res in args.resolutions:
            heatmap = store.aggregate(control_type, int(res))
            if heatmap is None:
                continue
            metrics = compute_self_metrics(heatmap, context["eval_box_512"])
            heatmap_paths = None
            if args.save_heatmaps:
                heatmap_paths = save_heatmap(
                    heatmap,
                    context["eval_box_512"],
                    Path(args.output_dir) / "heatmaps" / f"{context['context_type']}_self_{control_type}",
                    safe_stem(record, f"{context['context_type']}_self_{control_type}", int(res)),
                    context["image_512"],
                )
            records.append(
                {
                    "sample_id": f"val:{record.get('original_index', record.get('index', 'unknown'))}",
                    "split": "val",
                    "original_index": record.get("original_index"),
                    "file_name": record["fileName"],
                    "source": record.get("source"),
                    "probe_type": "self",
                    "context_type": context["context_type"],
                    "self_control_type": control_type,
                    "target_present": context["target_present"],
                    "random_crop_iou_with_gt": context["random_crop_iou_with_gt"],
                    "crop_context_scale": args.crop_context_scale,
                    "square_crop": args.square_crop,
                    "crop_box_original": context["crop_box_original"],
                    "image_width": item["image_width"],
                    "image_height": item["image_height"],
                    "expression": record["expression"],
                    "prompt_variant": prompt_record["prompt_variant"],
                    "prompt": prompt_record["prompt"],
                    "map_type": "self_row",
                    "resolution": int(res),
                    "seed_indices": seed_indices[control_type][int(res)],
                    "tokenizer": prompt_record["tokenizer"],
                    "entities": entities,
                    "target_entity": entities[0] if entities else None,
                    "gt_box_raw": item["gt_box_raw"],
                    "gt_box_clipped": item["gt_box_clipped"],
                    "bbox_512_full_image": item["bbox_512"],
                    "eval_box_512": context["eval_box_512"],
                    "target_width_512_full_image": item["target_width_512"],
                    "target_height_512_full_image": item["target_height_512"],
                    "target_width_512_context": context["eval_box_512"][2] - context["eval_box_512"][0],
                    "target_height_512_context": context["eval_box_512"][3] - context["eval_box_512"][1],
                    "target_size_bucket": item["target_size_bucket"],
                    "reconstruction_metrics": context["reconstruction_metrics"],
                    "metrics": metrics,
                    "heatmap_paths": heatmap_paths,
                }
            )
    return records


def run_context(pipe, inverter, base_attn_processors, item, context, prompt_record, entities, args):
    restore_attention_processors(pipe, base_attn_processors)
    image_rec, x_t, uncond_embeddings = inverter.invert(
        context["image_512"],
        prompt_record["prompt"],
        args.null_inner_steps,
        args.early_stop_epsilon,
    )
    context = dict(context)
    context["x_t"] = x_t
    context["uncond_embeddings"] = uncond_embeddings
    context["reconstruction_metrics"] = reconstruction_metrics(
        context["image_512"],
        image_rec,
        context["eval_box_512"],
        context["target_present"],
    )

    records = []
    if "cross" in args.probe_types:
        records.extend(run_cross_probe(pipe, inverter, base_attn_processors, item, context, prompt_record, entities, args))
    if "self" in args.probe_types:
        records.extend(run_self_probe(pipe, inverter, base_attn_processors, item, context, prompt_record, entities, args))
    return records


def build_summary(records, skipped, args):
    metric_records = [record for record in records if record.get("metrics")]
    grouped = defaultdict(list)
    for record in metric_records:
        key = (record["probe_type"], record["context_type"], record["map_type"], str(record["resolution"]))
        grouped[key].append(record)

    by_probe_context_map_resolution = {}
    for (probe_type, context_type, map_type, resolution), group in sorted(grouped.items()):
        by_probe_context_map_resolution.setdefault(probe_type, {}).setdefault(context_type, {}).setdefault(map_type, {})[
            resolution
        ] = {
            key: summarize_values([item["metrics"][key] for item in group])
            for key in sorted({metric for item in group for metric in item["metrics"].keys()})
        }

    recon_grouped = defaultdict(list)
    seen = set()
    for record in records:
        key = (record["sample_id"], record["context_type"])
        if key in seen:
            continue
        seen.add(key)
        recon_grouped[record["context_type"]].append(record)
    by_context_reconstruction = {
        context_type: {
            key: summarize_values([item["reconstruction_metrics"].get(key) for item in group])
            for key in RECON_KEYS
        }
        for context_type, group in sorted(recon_grouped.items())
    }

    return {
        "experiment": "exp_4_full_vs_crop_attention",
        "split": "val",
        "record_count": len(records),
        "processed_metric_records": len(metric_records),
        "skipped_count": len(skipped),
        "sample_count": len({record["sample_id"] for record in records}),
        "context_types": list(args.context_types),
        "probe_types": list(args.probe_types),
        "self_control_types": list(args.self_control_types),
        "prompt_variant": "p1_full_expression",
        "resolutions": list(args.resolutions),
        "model": args.sd_model,
        "num_ddim_steps": args.num_ddim_steps,
        "guidance_scale": args.guidance_scale,
        "null_inner_steps": args.null_inner_steps,
        "crop_context_scale": args.crop_context_scale,
        "square_crop": args.square_crop,
        "by_probe_context_map_resolution": by_probe_context_map_resolution,
        "by_context_reconstruction": by_context_reconstruction,
    }


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(args.annotation_dir) / "Val_10pct.json"
    image_root = Path(args.image_root)

    records = read_json(annotation_path)
    if args.limit is not None:
        records = records[: args.limit]

    selected, skipped = select_smoke_records(records, image_root, args.smoke_count)
    pipe, inverter = load_sd(args)
    base_attn_processors = dict(pipe.unet.attn_processors)
    tokenizer = pipe.tokenizer

    all_records = []
    for item_index, item in enumerate(selected):
        record = item["record"]
        entities, prompt_records = build_prompt_records(record, tokenizer, {"p1_full_expression"})
        prompt_record = prompt_records[0]
        contexts = build_contexts(item, image_root, args)
        for context in contexts:
            print(
                f"[{item_index + 1}/{len(selected)}] {record['fileName']} "
                f"exp4_{context['context_type']}",
                flush=True,
            )
            all_records.extend(
                run_context(pipe, inverter, base_attn_processors, item, context, prompt_record, entities, args)
            )

    write_jsonl(output_dir / "exp4_val_records.jsonl", all_records)
    write_jsonl(output_dir / "exp4_val_skipped.jsonl", skipped)
    summary = build_summary(all_records, skipped, args)
    (output_dir / "exp4_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", default="data/foundation_probe_10pct/annotations")
    parser.add_argument("--image-root", default="data/foundation_probe_10pct/images")
    parser.add_argument("--output-dir", default="results/exp_4_full_vs_crop_attention_val_full")
    parser.add_argument("--smoke-count", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sd-model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--num-ddim-steps", type=int, default=NUM_DDIM_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--null-inner-steps", type=int, default=10)
    parser.add_argument("--early-stop-epsilon", type=float, default=1e-5)
    parser.add_argument("--resolutions", nargs="+", type=int, default=list(GRID_RESOLUTIONS))
    parser.add_argument("--context-types", nargs="+", default=list(CONTEXT_TYPES), choices=CONTEXT_TYPES)
    parser.add_argument("--probe-types", nargs="+", default=["cross"], choices=PROBE_TYPES)
    parser.add_argument(
        "--self-control-types",
        nargs="+",
        default=["gt_center"],
        choices=("gt_center", "random_background", "all_gt_cells"),
    )
    parser.add_argument("--crop-context-scale", type=float, default=2.0)
    parser.add_argument("--square-crop", action="store_true", default=True)
    parser.add_argument("--rectangular-crop", action="store_false", dest="square_crop")
    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--seed", type=int, default=20260702)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
