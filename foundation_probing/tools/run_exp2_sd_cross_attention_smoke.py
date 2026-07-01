#!/usr/bin/env python3
"""Run Experiment 2 Stable Diffusion cross-attention smoke probe.

The code follows the RSVG-ZeroOV probing path: resize to 512, DDIM/null-text
invert with a prompt, rerun denoising while capturing UNet cross-attention, then
evaluate target-token attention around the SkyFind GT box.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    from run_exp1_tiny_target_retention import (
        GUIDANCE_SCALE,
        GRID_RESOLUTIONS,
        IMAGE_SIZE,
        NUM_DDIM_STEPS,
        NullInversion,
        area_bucket,
        box_valid,
        clip_box_to_image,
        map_box_to_512,
        resize_rsvg_512,
        source_name,
    )
    from run_exp2_tokenizer_object_audit import build_prompt_variants, tokenizer_diagnostics
    from skyfind_entity_extraction import extract_entities
except ImportError:
    from foundation_probing.tools.run_exp1_tiny_target_retention import (
        GUIDANCE_SCALE,
        GRID_RESOLUTIONS,
        IMAGE_SIZE,
        NUM_DDIM_STEPS,
        NullInversion,
        area_bucket,
        box_valid,
        clip_box_to_image,
        map_box_to_512,
        resize_rsvg_512,
        source_name,
    )
    from foundation_probing.tools.run_exp2_tokenizer_object_audit import (
        build_prompt_variants,
        tokenizer_diagnostics,
    )
    from foundation_probing.tools.skyfind_entity_extraction import extract_entities


PROMPT_VARIANTS = (
    "p1_full_expression",
    "p2_entity_set",
    "p3_domain_entity_set",
    "p4_vlm_localization",
    "c1_wrong_object_phrase",
)

MAP_TYPES = ("target_entity", "all_entities", "all_non_special")
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


def stable_attention_probs(attn, query, key, attention_mask, store, place_in_unet):
    torch = __import__("torch")
    scores = torch.bmm(query.float(), key.float().transpose(1, 2)) * float(attn.scale)
    if attention_mask is not None:
        scores = scores + attention_mask.float()

    if not torch.isfinite(scores).all():
        raise FloatingPointError(
            f"Non-finite cross-attention scores at {place_in_unet}: "
            f"shape={tuple(scores.shape)} dtype={scores.dtype}"
        )

    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probs).all():
        raise FloatingPointError(
            f"Non-finite cross-attention probabilities at {place_in_unet}: "
            f"shape={tuple(probs.shape)} dtype={probs.dtype}"
        )
    return probs


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_values(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


class CrossAttentionStore:
    def __init__(self):
        self.maps = defaultdict(list)
        self.diagnostics = defaultdict(int)

    def add(self, attention_probs, place_in_unet, nonfinite_inputs=False):
        if attention_probs.shape[1] > 64**2:
            return
        torch = __import__("torch")
        if nonfinite_inputs:
            raise FloatingPointError(f"Non-finite cross-attention q/k inputs at {place_in_unet}")
        if not torch.isfinite(attention_probs).all():
            raise FloatingPointError(
                f"Stable cross-attention capture failed at {place_in_unet}: "
                f"shape={tuple(attention_probs.shape)} dtype={attention_probs.dtype}"
            )
        probs = attention_probs.detach()
        if probs.shape[0] % 2 == 0:
            probs = probs[probs.shape[0] // 2 :]
        self.maps[f"{place_in_unet}_cross"].append(probs.float().cpu())


class CrossAttentionCaptureProcessor:
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
        is_cross = encoder_hidden_states is not None
        if not is_cross:
            return

        with __import__("torch").no_grad():
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = encoder_hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)

            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)

            torch = __import__("torch")
            nonfinite_inputs = not torch.isfinite(query).all() or not torch.isfinite(key).all()
            if nonfinite_inputs:
                raise FloatingPointError(
                    f"Non-finite cross-attention q/k inputs at {self.place_in_unet}: "
                    f"query_shape={tuple(query.shape)} key_shape={tuple(key.shape)}"
                )
            query = query.float()
            key = key.float()

            attention_mask_float = attention_mask.float() if attention_mask is not None else None
            attention_probs = stable_attention_probs(
                attn,
                query,
                key,
                attention_mask_float,
                self.store,
                self.place_in_unet,
            )
            self.store.add(attention_probs, self.place_in_unet, nonfinite_inputs)


def attention_place(name):
    if name.startswith("down_blocks"):
        return "down"
    if name.startswith("up_blocks"):
        return "up"
    return "mid"


def restore_attention_processors(pipe, processors):
    pipe.unet.set_attn_processor(dict(processors))


def register_cross_attention_store(pipe, store, processor_names):
    processors = {
        name: CrossAttentionCaptureProcessor(store, attention_place(name), pipe.unet.attn_processors[name])
        for name in processor_names
    }
    pipe.unet.set_attn_processor(processors)


def load_sd(args):
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    device = torch.device(args.device)
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model,
        scheduler=scheduler,
        torch_dtype=getattr(torch, args.torch_dtype) if args.torch_dtype else None,
    ).to(device)
    pipe.scheduler.set_timesteps(args.num_ddim_steps)
    pipe.set_progress_bar_config(disable=True)
    inverter = NullInversion(pipe, device, args.num_ddim_steps, args.guidance_scale)
    return pipe, inverter


def run_denoising_with_attention(inverter, x_t, uncond_embeddings):
    torch = inverter.torch
    _, cond_embeddings = inverter.context.chunk(2)
    latent = x_t.clone().detach()
    with torch.no_grad():
        for step_index, timestep in enumerate(inverter.model.scheduler.timesteps):
            context = torch.cat([uncond_embeddings[step_index], cond_embeddings])
            latent = inverter.get_noise_pred(latent, timestep, False, context)
    return latent


def aggregate_cross_attention(store, res):
    tensors = []
    num_pixels = res**2
    for location in ("down", "up"):
        for item in store.maps[f"{location}_cross"]:
            if item.shape[1] == num_pixels:
                tensors.append(item.reshape(-1, res, res, item.shape[-1]))
    if not tensors:
        return None
    return np.asarray(__import__("torch").cat(tensors, dim=0).mean(0).cpu(), dtype=np.float32)


def token_heatmap(attention_map, token_indices):
    valid_indices = [index for index in token_indices if 0 <= index < attention_map.shape[-1]]
    if not valid_indices:
        return None
    heatmap = attention_map[:, :, valid_indices].mean(axis=-1).astype(np.float64)
    total = float(heatmap.sum())
    if total > 0:
        heatmap /= total
    return heatmap.astype(np.float32)


def grid_overlap_weights(box_512, res):
    x1, y1, x2, y2 = box_512
    cell = IMAGE_SIZE / float(res)
    weights = np.zeros((res, res), dtype=np.float32)
    for row in range(res):
        cy1 = row * cell
        cy2 = (row + 1) * cell
        oy = max(0.0, min(y2, cy2) - max(y1, cy1))
        if oy <= 0:
            continue
        for col in range(res):
            cx1 = col * cell
            cx2 = (col + 1) * cell
            ox = max(0.0, min(x2, cx2) - max(x1, cx1))
            if ox > 0:
                weights[row, col] = (ox * oy) / (cell * cell)
    return weights


def compute_heatmap_metrics(heatmap, box_512):
    res = heatmap.shape[0]
    heatmap = heatmap.astype(np.float64)
    total = float(heatmap.sum())
    if total > 0:
        heatmap = heatmap / total

    weights = grid_overlap_weights(box_512, res).astype(np.float64)
    gt_area_ratio = float(weights.sum() / (res * res))
    gt_attention_ratio = float((heatmap * weights).sum())
    enrichment = gt_attention_ratio / gt_area_ratio if gt_area_ratio > 0 else None

    peak_flat = int(heatmap.argmax())
    peak_row, peak_col = divmod(peak_flat, res)
    pointing = float(weights[peak_row, peak_col] > 0)

    top_hits = {}
    flat_order = np.argsort(heatmap.reshape(-1))[::-1]
    flat_weights = weights.reshape(-1)
    for pct, key in ((1, "top1_hit"), (5, "top5_hit"), (10, "top10_hit")):
        top_k = max(1, int(math.ceil(res * res * pct / 100.0)))
        top_hits[key] = float(np.any(flat_weights[flat_order[:top_k]] > 0))

    cell = IMAGE_SIZE / float(res)
    peak_x = (peak_col + 0.5) * cell
    peak_y = (peak_row + 0.5) * cell
    gt_x = (box_512[0] + box_512[2]) / 2.0
    gt_y = (box_512[1] + box_512[3]) / 2.0
    distance = math.sqrt((peak_x - gt_x) ** 2 + (peak_y - gt_y) ** 2) / math.sqrt(2 * IMAGE_SIZE**2)

    nonzero = heatmap[heatmap > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / math.log(res * res))
    peakness = float(heatmap.max() * res * res)

    return {
        "pointing_game": pointing,
        **top_hits,
        "gt_attention_ratio": gt_attention_ratio,
        "gt_area_ratio": gt_area_ratio,
        "attention_enrichment": enrichment,
        "peak_center_distance": float(distance),
        "entropy": entropy,
        "peakness": peakness,
    }


def save_heatmap(heatmap, box_512, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{stem}.npy"
    png_path = output_dir / f"{stem}.png"
    np.save(npy_path, heatmap.astype(np.float32))

    scaled = heatmap.astype(np.float64)
    if scaled.max() > 0:
        scaled = scaled / scaled.max()
    gray = Image.fromarray((scaled * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    image = ImageOps.colorize(gray.convert("L"), black=(0, 0, 0), white=(255, 96, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([float(v) for v in box_512], outline=(0, 255, 255), width=2)
    image.save(png_path)
    return {"npy": str(npy_path), "png": str(png_path)}


def tokenizer_spans(tokenizer, prompt, spans):
    return tokenizer_diagnostics(tokenizer, prompt, spans)


def map_type_indices(prompt_record):
    tokenizer_info = prompt_record["tokenizer"]
    span_indices = tokenizer_info["span_token_indices"]
    primary = span_indices.get("target_entity")
    if primary is None:
        primary = span_indices.get("wrong_object", [])
    return {
        "target_entity": primary or [],
        "all_entities": tokenizer_info.get("all_entity_token_indices", []),
        "all_non_special": tokenizer_info["non_special_token_indices"],
    }


def random_heatmap(res, seed):
    rng = np.random.default_rng(seed)
    heatmap = rng.random((res, res), dtype=np.float32)
    heatmap /= float(heatmap.sum())
    return heatmap


def safe_stem(record, prompt_variant, res):
    index = record.get("original_index", "unknown")
    return f"val_{index}_{Path(record['fileName']).stem}_{prompt_variant}_r{res}"


def prepare_record(record, image_root):
    image_path = image_root / record["fileName"]
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        image_512 = resize_rsvg_512(image)

    raw_box = [float(value) for value in record["bbox"]]
    clipped_box, was_clipped = clip_box_to_image(raw_box, width, height)
    if not box_valid(clipped_box):
        return None, {
            "file_name": record["fileName"],
            "reason": "invalid_bbox_after_clipping",
            "gt_box_raw": raw_box,
            "gt_box_clipped": clipped_box,
        }
    box_512 = map_box_to_512(clipped_box, width, height)
    area_ratio = ((clipped_box[2] - clipped_box[0]) * (clipped_box[3] - clipped_box[1])) / float(width * height)
    return {
        "record": record,
        "image_512": image_512,
        "image_width": width,
        "image_height": height,
        "gt_box_raw": raw_box,
        "gt_box_clipped": clipped_box,
        "bbox_was_clipped": was_clipped,
        "bbox_512": box_512,
        "target_width_512": box_512[2] - box_512[0],
        "target_height_512": box_512[3] - box_512[1],
        "area_ratio_original_clipped": area_ratio,
        "target_size_bucket": area_bucket(area_ratio),
    }, None


def select_smoke_records(records, image_root, smoke_count):
    prepared = []
    skipped = []
    for record in records:
        try:
            item, skip = prepare_record(record, image_root)
        except (FileNotFoundError, OSError) as exc:
            item = None
            skip = {
                "file_name": record.get("fileName"),
                "reason": "missing_or_bad_image",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if item is not None:
            prepared.append(item)
        if skip is not None:
            skipped.append(skip)

    if smoke_count is None or smoke_count >= len(prepared):
        return prepared, skipped

    groups = defaultdict(list)
    for item in prepared:
        groups[item["target_size_bucket"]].append(item)

    selected = []
    per_bucket = max(1, smoke_count // 3)
    for bucket in ("tiny", "small", "large"):
        selected.extend(groups[bucket][:per_bucket])
    if len(selected) < smoke_count:
        selected_ids = {id(item) for item in selected}
        for item in prepared:
            if id(item) not in selected_ids:
                selected.append(item)
            if len(selected) == smoke_count:
                break
    return selected[:smoke_count], skipped


def build_prompt_records(record, tokenizer, prompt_variant_filter):
    entities = extract_entities(record["expression"])
    prompt_records = []
    for variant in build_prompt_variants(record["expression"], entities):
        if variant["prompt_variant"] not in prompt_variant_filter:
            continue
        prompt_record = {
            "prompt_variant": variant["prompt_variant"],
            "prompt": variant["prompt"],
            "requires_entities": variant.get("requires_entities", False),
            "control_type": variant.get("control_type"),
            "wrong_phrase": variant.get("wrong_phrase"),
            "char_spans": variant["spans"],
            "tokenizer": tokenizer_spans(tokenizer, variant["prompt"], variant["spans"]),
        }
        prompt_records.append(prompt_record)
    return entities, prompt_records


def baseline_record(item, res, args):
    record = item["record"]
    heatmap = random_heatmap(res, args.seed + int(record.get("original_index", 0)) * 100 + res)
    metrics = compute_heatmap_metrics(heatmap, item["bbox_512"])
    heatmap_paths = None
    if args.save_heatmaps:
        heatmap_paths = save_heatmap(
            heatmap,
            item["bbox_512"],
            Path(args.output_dir) / "heatmaps" / "c0_random_baseline",
            safe_stem(record, "c0_random_baseline", res),
        )
    return {
        "sample_id": f"val:{record.get('original_index', record.get('index', 'unknown'))}",
        "split": "val",
        "original_index": record.get("original_index"),
        "file_name": record["fileName"],
        "source": record.get("source") or source_name(record["fileName"]),
        "prompt_variant": "c0_random_baseline",
        "prompt": None,
        "map_type": "random",
        "resolution": res,
        "metrics": metrics,
        "heatmap_paths": heatmap_paths,
    }


def run_prompt_variant(pipe, inverter, base_attn_processors, item, prompt_record, entities, args):
    import torch

    record = item["record"]
    processor_names = list(base_attn_processors.keys())
    restore_attention_processors(pipe, base_attn_processors)
    _, x_t, uncond_embeddings = inverter.invert(
        item["image_512"],
        prompt_record["prompt"],
        args.null_inner_steps,
        args.early_stop_epsilon,
    )
    store = CrossAttentionStore()
    try:
        register_cross_attention_store(pipe, store, processor_names)
        run_denoising_with_attention(inverter, x_t, uncond_embeddings)
    finally:
        restore_attention_processors(pipe, base_attn_processors)
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    token_indices = map_type_indices(prompt_record)
    attention_capture_diagnostics = dict(store.diagnostics)
    output_records = []
    for res in args.resolutions:
        attention_map = aggregate_cross_attention(store, res)
        if attention_map is None:
            continue
        for map_type in MAP_TYPES:
            heatmap = token_heatmap(attention_map, token_indices[map_type])
            if heatmap is None:
                metrics = None
                heatmap_paths = None
            else:
                metrics = compute_heatmap_metrics(heatmap, item["bbox_512"])
                heatmap_paths = None
                if args.save_heatmaps and map_type == "target_entity":
                    heatmap_paths = save_heatmap(
                        heatmap,
                        item["bbox_512"],
                        Path(args.output_dir) / "heatmaps" / prompt_record["prompt_variant"],
                        safe_stem(record, prompt_record["prompt_variant"], res),
                    )
            output_records.append(
                {
                    "sample_id": f"val:{record.get('original_index', record.get('index', 'unknown'))}",
                    "split": "val",
                    "original_index": record.get("original_index"),
                    "file_name": record["fileName"],
                    "source": record.get("source") or source_name(record["fileName"]),
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
                    "entities": entities,
                    "target_entity": entities[0] if entities else None,
                    "prompt_variant": prompt_record["prompt_variant"],
                    "prompt": prompt_record["prompt"],
                    "map_type": map_type,
                    "resolution": res,
                    "token_indices": token_indices[map_type],
                    "tokenizer": prompt_record["tokenizer"],
                    "attention_capture_diagnostics": attention_capture_diagnostics,
                    "metrics": metrics,
                    "heatmap_paths": heatmap_paths,
                }
            )
    return output_records


def build_summary(records, skipped, args):
    metric_records = [record for record in records if record.get("metrics")]
    grouped = defaultdict(list)
    for record in metric_records:
        key = (record["prompt_variant"], record["map_type"], str(record["resolution"]))
        grouped[key].append(record)

    by_prompt_map_resolution = {}
    for (prompt_variant, map_type, resolution), group_records in sorted(grouped.items()):
        by_prompt_map_resolution.setdefault(prompt_variant, {}).setdefault(map_type, {})[resolution] = {
            key: summarize_values([record["metrics"][key] for record in group_records])
            for key in METRIC_KEYS
        }

    attention_capture_diagnostics = defaultdict(int)
    seen_prompt_sample = set()
    for record in records:
        key = (record.get("sample_id"), record.get("prompt_variant"))
        if key in seen_prompt_sample:
            continue
        seen_prompt_sample.add(key)
        for name, value in record.get("attention_capture_diagnostics", {}).items():
            attention_capture_diagnostics[name] += int(value)

    return {
        "experiment": "exp_2_sd_cross_attention_smoke",
        "split": "val",
        "processed_metric_records": len(metric_records),
        "record_count": len(records),
        "skipped_count": len(skipped),
        "smoke_count": args.smoke_count,
        "prompt_variants": list(args.prompt_variants),
        "resolutions": list(args.resolutions),
        "model": args.sd_model,
        "num_ddim_steps": args.num_ddim_steps,
        "guidance_scale": args.guidance_scale,
        "null_inner_steps": args.null_inner_steps,
        "attention_capture_diagnostics": dict(sorted(attention_capture_diagnostics.items())),
        "by_prompt_map_resolution": by_prompt_map_resolution,
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
        entities, prompt_records = build_prompt_records(record, tokenizer, set(args.prompt_variants))
        for res in args.resolutions:
            all_records.append(baseline_record(item, res, args))
        for prompt_record in prompt_records:
            print(
                f"[{item_index + 1}/{len(selected)}] {record['fileName']} "
                f"{prompt_record['prompt_variant']}",
                flush=True,
            )
            all_records.extend(
                run_prompt_variant(
                    pipe,
                    inverter,
                    base_attn_processors,
                    item,
                    prompt_record,
                    entities,
                    args,
                )
            )

    write_jsonl(output_dir / "exp2_val_records.jsonl", all_records)
    write_jsonl(output_dir / "exp2_val_skipped.jsonl", skipped)
    summary = build_summary(all_records, skipped, args)
    (output_dir / "exp2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", default="data/foundation_probe_10pct/annotations")
    parser.add_argument("--image-root", default="data/foundation_probe_10pct/images")
    parser.add_argument("--output-dir", default="results/exp_2_sd_cross_attention_smoke")
    parser.add_argument("--smoke-count", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sd-model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--num-ddim-steps", type=int, default=NUM_DDIM_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--null-inner-steps", type=int, default=10)
    parser.add_argument("--early-stop-epsilon", type=float, default=1e-5)
    parser.add_argument("--resolutions", nargs="+", type=int, default=list(GRID_RESOLUTIONS))
    parser.add_argument("--prompt-variants", nargs="+", default=list(PROMPT_VARIANTS), choices=PROMPT_VARIANTS)
    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
