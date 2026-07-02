#!/usr/bin/env python3
"""Run Experiment 3: SD self-attention object-structure probe.

This is an oracle diagnostic: it uses the GT center only to select one
self-attention query row and tests whether that row concentrates on the target.
The GT center is not treated as an inference-time input.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from run_exp1_tiny_target_retention import GUIDANCE_SCALE, GRID_RESOLUTIONS, NUM_DDIM_STEPS
    from run_exp2_sd_cross_attention_smoke import (
        compute_heatmap_metrics,
        grid_overlap_weights,
        load_sd,
        prepare_record,
        read_json,
        restore_attention_processors,
        save_heatmap,
        select_smoke_records,
        summarize_values,
        write_jsonl,
    )
except ImportError:
    from foundation_probing.tools.run_exp1_tiny_target_retention import (
        GUIDANCE_SCALE,
        GRID_RESOLUTIONS,
        NUM_DDIM_STEPS,
    )
    from foundation_probing.tools.run_exp2_sd_cross_attention_smoke import (
        compute_heatmap_metrics,
        grid_overlap_weights,
        load_sd,
        prepare_record,
        read_json,
        restore_attention_processors,
        save_heatmap,
        select_smoke_records,
        summarize_values,
        write_jsonl,
    )


PROMPT_POLICIES = ("full_expression", "remote_sensing", "empty")
CONTROL_TYPES = ("gt_center", "random_background", "all_gt_cells")
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
    "outside_gt_energy_ratio",
    "outside_expanded_gt_energy_ratio",
    "equal_area_iou",
    "equal_area_center_error",
    "equal_area_bbox_area_ratio",
)


def attention_place(name):
    if name.startswith("down_blocks"):
        return "down"
    if name.startswith("up_blocks"):
        return "up"
    return "mid"


def prompt_for_record(record, policy):
    if policy == "full_expression":
        return record["expression"]
    if policy == "remote_sensing":
        return "a remote sensing image"
    if policy == "empty":
        return ""
    raise ValueError(f"Unknown prompt policy: {policy}")


def finite_softmax_rows(attn, query_rows, key, attention_mask, place_in_unet):
    torch = __import__("torch")
    scores = torch.bmm(query_rows.float(), key.float().transpose(1, 2)) * float(attn.scale)
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    if not torch.isfinite(scores).all():
        raise FloatingPointError(
            f"Non-finite self-attention row scores at {place_in_unet}: "
            f"shape={tuple(scores.shape)} dtype={scores.dtype}"
        )
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probs).all():
        raise FloatingPointError(
            f"Non-finite self-attention row probabilities at {place_in_unet}: "
            f"shape={tuple(probs.shape)} dtype={probs.dtype}"
        )
    return probs


class SelfAttentionRowStore:
    def __init__(self, seed_indices_by_control, resolutions):
        self.seed_indices_by_control = seed_indices_by_control
        self.resolutions = set(int(res) for res in resolutions)
        self.rows = defaultdict(list)

    def add(self, control_type, res, row_probs):
        if row_probs.shape[-1] != res * res:
            raise ValueError(f"Self-attention row length does not match resolution {res}")
        self.rows[(control_type, res)].append(row_probs.detach().float().cpu())

    def aggregate(self, control_type, res):
        rows = self.rows.get((control_type, res), [])
        if not rows:
            return None
        vector = __import__("torch").stack(rows, dim=0).mean(dim=0).numpy().astype(np.float64)
        total = float(vector.sum())
        if total <= 0 or not math.isfinite(total):
            raise FloatingPointError(f"Invalid self-attention heatmap total for {control_type} r{res}")
        heatmap = (vector / total).reshape(res, res)
        return heatmap.astype(np.float32)


class SelfAttentionRowCaptureProcessor:
    def __init__(self, store, place_in_unet, base_processor):
        self.store = store
        self.place_in_unet = place_in_unet
        self.base_processor = base_processor

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        self.capture(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        return self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
        )

    def capture(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        if encoder_hidden_states is not None:
            return

        torch = __import__("torch")
        with torch.no_grad():
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = hidden_states.shape
            res = int(math.sqrt(sequence_length))
            if res * res != sequence_length or res not in self.store.resolutions:
                return

            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            if not torch.isfinite(query).all() or not torch.isfinite(key).all():
                raise FloatingPointError(
                    f"Non-finite self-attention q/k at {self.place_in_unet}: "
                    f"query_shape={tuple(query.shape)} key_shape={tuple(key.shape)}"
                )

            if query.shape[0] % 2 == 0:
                query = query[query.shape[0] // 2 :]
                key = key[key.shape[0] // 2 :]

            attention_mask_float = attention_mask.float() if attention_mask is not None else None
            for control_type, indices_by_res in self.store.seed_indices_by_control.items():
                seed_indices = indices_by_res.get(res)
                if not seed_indices:
                    continue
                seed_tensor = torch.as_tensor(seed_indices, device=query.device, dtype=torch.long)
                query_rows = query.index_select(1, seed_tensor)
                probs = finite_softmax_rows(
                    attn,
                    query_rows,
                    key,
                    attention_mask_float,
                    self.place_in_unet,
                )
                row = probs.mean(dim=(0, 1))
                self.store.add(control_type, res, row)


def register_self_attention_store(pipe, store, processor_names):
    processors = {
        name: SelfAttentionRowCaptureProcessor(store, attention_place(name), pipe.unet.attn_processors[name])
        for name in processor_names
    }
    pipe.unet.set_attn_processor(processors)


def run_denoising(inverter, x_t, uncond_embeddings):
    torch = inverter.torch
    _, cond_embeddings = inverter.context.chunk(2)
    latent = x_t.clone().detach()
    with torch.no_grad():
        for step_index, timestep in enumerate(inverter.model.scheduler.timesteps):
            context = torch.cat([uncond_embeddings[step_index], cond_embeddings])
            latent = inverter.get_noise_pred(latent, timestep, False, context)
    return latent


def center_seed(box_512, res):
    x = (box_512[0] + box_512[2]) / 2.0
    y = (box_512[1] + box_512[3]) / 2.0
    col = min(res - 1, max(0, int(math.floor(x / 512.0 * res))))
    row = min(res - 1, max(0, int(math.floor(y / 512.0 * res))))
    return row * res + col


def gt_cell_seeds(box_512, res):
    weights = grid_overlap_weights(box_512, res)
    indices = np.flatnonzero(weights.reshape(-1) > 0).astype(int).tolist()
    return indices or [center_seed(box_512, res)]


def random_background_seed(box_512, res, seed):
    weights = grid_overlap_weights(box_512, res).reshape(-1)
    candidates = np.flatnonzero(weights == 0)
    if candidates.size == 0:
        return center_seed(box_512, res)
    rng = np.random.default_rng(seed)
    return int(candidates[int(rng.integers(0, candidates.size))])


def build_seed_indices(box_512, resolutions, control_types, seed):
    seed_indices = {}
    for control_type in control_types:
        by_res = {}
        for res in resolutions:
            if control_type == "gt_center":
                by_res[int(res)] = [center_seed(box_512, int(res))]
            elif control_type == "random_background":
                by_res[int(res)] = [random_background_seed(box_512, int(res), seed + int(res))]
            elif control_type == "all_gt_cells":
                by_res[int(res)] = gt_cell_seeds(box_512, int(res))
            else:
                raise ValueError(f"Unknown control type: {control_type}")
        seed_indices[control_type] = by_res
    return seed_indices


def expanded_box(box_512, scale):
    x1, y1, x2, y2 = [float(value) for value in box_512]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = (x2 - x1) * scale
    height = (y2 - y1) * scale
    return [
        max(0.0, cx - width / 2.0),
        max(0.0, cy - height / 2.0),
        min(512.0, cx + width / 2.0),
        min(512.0, cy + height / 2.0),
    ]


def equal_area_metrics(heatmap, box_512):
    res = heatmap.shape[0]
    weights = grid_overlap_weights(box_512, res)
    gt_mask = weights > 0
    gt_count = int(gt_mask.sum())
    if gt_count <= 0:
        return {
            "equal_area_iou": None,
            "equal_area_center_error": None,
            "equal_area_bbox_area_ratio": None,
        }

    flat_order = np.argsort(heatmap.reshape(-1))[::-1]
    selected = np.zeros(res * res, dtype=bool)
    selected[flat_order[:gt_count]] = True
    pred_mask = selected.reshape(res, res)
    intersection = float(np.logical_and(pred_mask, gt_mask).sum())
    union = float(np.logical_or(pred_mask, gt_mask).sum())
    iou = intersection / union if union > 0 else None

    rows, cols = np.where(pred_mask)
    cell = 512.0 / float(res)
    pred_cx = float((cols.mean() + 0.5) * cell)
    pred_cy = float((rows.mean() + 0.5) * cell)
    gt_cx = float((box_512[0] + box_512[2]) / 2.0)
    gt_cy = float((box_512[1] + box_512[3]) / 2.0)
    center_error = math.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) / math.sqrt(2 * 512.0**2)

    pred_area = float((cols.max() - cols.min() + 1) * cell * (rows.max() - rows.min() + 1) * cell)
    gt_area = max(1e-6, float((box_512[2] - box_512[0]) * (box_512[3] - box_512[1])))
    return {
        "equal_area_iou": float(iou) if iou is not None else None,
        "equal_area_center_error": float(center_error),
        "equal_area_bbox_area_ratio": float(pred_area / gt_area),
    }


def compute_self_metrics(heatmap, box_512):
    metrics = compute_heatmap_metrics(heatmap, box_512)
    expanded_weights = grid_overlap_weights(expanded_box(box_512, 2.0), heatmap.shape[0]).astype(np.float64)
    metrics["outside_gt_energy_ratio"] = float(1.0 - metrics["gt_attention_ratio"])
    metrics["outside_expanded_gt_energy_ratio"] = float(1.0 - (heatmap.astype(np.float64) * expanded_weights).sum())
    metrics.update(equal_area_metrics(heatmap, box_512))
    return metrics


def safe_stem(record, control_type, res):
    index = record.get("original_index", "unknown")
    return f"val_{index}_{Path(record['fileName']).stem}_{control_type}_self{res}"


def run_item(pipe, inverter, base_attn_processors, item, args):
    import torch

    record = item["record"]
    prompt = prompt_for_record(record, args.prompt_policy)
    processor_names = list(base_attn_processors.keys())
    restore_attention_processors(pipe, base_attn_processors)
    _, x_t, uncond_embeddings = inverter.invert(
        item["image_512"],
        prompt,
        args.null_inner_steps,
        args.early_stop_epsilon,
    )

    seed = args.seed + int(record.get("original_index", 0)) * 1009
    seed_indices = build_seed_indices(item["bbox_512"], args.resolutions, args.control_types, seed)
    store = SelfAttentionRowStore(seed_indices, args.resolutions)
    try:
        register_self_attention_store(pipe, store, processor_names)
        run_denoising(inverter, x_t, uncond_embeddings)
    finally:
        restore_attention_processors(pipe, base_attn_processors)
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    records = []
    for control_type in args.control_types:
        for res in args.resolutions:
            heatmap = store.aggregate(control_type, int(res))
            if heatmap is None:
                continue
            metrics = compute_self_metrics(heatmap, item["bbox_512"])
            heatmap_paths = None
            if args.save_heatmaps:
                heatmap_paths = save_heatmap(
                    heatmap,
                    item["bbox_512"],
                    Path(args.output_dir) / "heatmaps" / control_type,
                    safe_stem(record, control_type, int(res)),
                    item["image_512"],
                )
            records.append(
                {
                    "sample_id": f"val:{record.get('original_index', record.get('index', 'unknown'))}",
                    "split": "val",
                    "original_index": record.get("original_index"),
                    "file_name": record["fileName"],
                    "source": record.get("source"),
                    "prompt_policy": args.prompt_policy,
                    "prompt": prompt,
                    "control_type": control_type,
                    "seed_indices": seed_indices[control_type][int(res)],
                    "resolution": int(res),
                    "image_width": item["image_width"],
                    "image_height": item["image_height"],
                    "expression": record["expression"],
                    "gt_box_raw": item["gt_box_raw"],
                    "gt_box_clipped": item["gt_box_clipped"],
                    "bbox_was_clipped": item["bbox_was_clipped"],
                    "bbox_512": item["bbox_512"],
                    "target_width_512": item["target_width_512"],
                    "target_height_512": item["target_height_512"],
                    "area_ratio_original_clipped": item["area_ratio_original_clipped"],
                    "target_size_bucket": item["target_size_bucket"],
                    "metrics": metrics,
                    "heatmap_paths": heatmap_paths,
                }
            )
    return records


def build_summary(records, skipped, args):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["control_type"], str(record["resolution"]))].append(record)

    by_control_resolution = {}
    for (control_type, resolution), group in sorted(grouped.items()):
        by_control_resolution.setdefault(control_type, {})[resolution] = {
            key: summarize_values([item["metrics"].get(key) for item in group])
            for key in METRIC_KEYS
        }

    return {
        "experiment": "exp_3_sd_self_attention_structure",
        "split": "val",
        "record_count": len(records),
        "skipped_count": len(skipped),
        "sample_count": len({record["sample_id"] for record in records}),
        "prompt_policy": args.prompt_policy,
        "control_types": list(args.control_types),
        "resolutions": list(args.resolutions),
        "model": args.sd_model,
        "num_ddim_steps": args.num_ddim_steps,
        "guidance_scale": args.guidance_scale,
        "null_inner_steps": args.null_inner_steps,
        "by_control_resolution": by_control_resolution,
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

    all_records = []
    for item_index, item in enumerate(selected):
        record = item["record"]
        print(f"[{item_index + 1}/{len(selected)}] {record['fileName']} exp3_self_attention", flush=True)
        all_records.extend(run_item(pipe, inverter, base_attn_processors, item, args))

    write_jsonl(output_dir / "exp3_val_records.jsonl", all_records)
    write_jsonl(output_dir / "exp3_val_skipped.jsonl", skipped)
    summary = build_summary(all_records, skipped, args)
    (output_dir / "exp3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", default="data/foundation_probe_10pct/annotations")
    parser.add_argument("--image-root", default="data/foundation_probe_10pct/images")
    parser.add_argument("--output-dir", default="results/exp_3_sd_self_attention_structure_val_full")
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
    parser.add_argument("--prompt-policy", default="full_expression", choices=PROMPT_POLICIES)
    parser.add_argument("--control-types", nargs="+", default=["gt_center", "random_background"], choices=CONTROL_TYPES)
    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--seed", type=int, default=20260702)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
