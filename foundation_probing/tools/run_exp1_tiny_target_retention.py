#!/usr/bin/env python3
"""Run Experiment 1: tiny-target retention and 512-scale audit.

The 512 resize follows RSVG-ZeroOV's Stable Diffusion path: direct square
resize to 512 x 512, without aspect-ratio preservation.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 512
GRID_RESOLUTIONS = (16, 32, 64)
NUM_DDIM_STEPS = 20
GUIDANCE_SCALE = 7.5

METRIC_KEYS = (
    "full_mse",
    "full_psnr",
    "full_ssim",
    "full_lpips",
    "target_mse",
    "target_psnr",
    "target_ssim",
    "target_lpips",
)


def source_name(file_name):
    stem = Path(file_name).stem
    return stem.split("_", 1)[0] if "_" in stem else "unknown"


def expression_bucket(expression):
    length = len(expression.split())
    if length <= 20:
        return "short"
    if length <= 40:
        return "medium"
    return "long"


def area_bucket(area_ratio):
    if area_ratio < 0.001:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    return "large"


def clip_box_to_image(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    clipped = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    was_clipped = any(abs(a - b) > 1e-6 for a, b in zip([x1, y1, x2, y2], clipped))
    return clipped, was_clipped


def box_valid(box):
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1


def map_box_to_512(box, width, height):
    sx = IMAGE_SIZE / float(width)
    sy = IMAGE_SIZE / float(height)
    x1, y1, x2, y2 = box
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def int_crop_bounds(box_512):
    x1, y1, x2, y2 = box_512
    left = max(0, min(IMAGE_SIZE, int(math.floor(x1))))
    top = max(0, min(IMAGE_SIZE, int(math.floor(y1))))
    right = max(0, min(IMAGE_SIZE, int(math.ceil(x2))))
    bottom = max(0, min(IMAGE_SIZE, int(math.ceil(y2))))
    return left, top, right, bottom


def resize_rsvg_512(image):
    return image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def metric_nulls():
    return {key: None for key in METRIC_KEYS}


def to_float01(image):
    return np.asarray(image).astype(np.float32) / 255.0


def compute_mse(a, b):
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def compute_reconstruction_metrics(original_512, reconstructed_512, box_512, lpips_model, device, args):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    orig = to_float01(original_512)
    rec = to_float01(reconstructed_512)
    left, top, right, bottom = int_crop_bounds(box_512)
    orig_crop = orig[top:bottom, left:right]
    rec_crop = rec[top:bottom, left:right]

    target_ssim = None
    min_crop_side = min(orig_crop.shape[:2])
    if min_crop_side >= 3:
        win_size = min(7, min_crop_side if min_crop_side % 2 == 1 else min_crop_side - 1)
        target_ssim = float(
            structural_similarity(orig_crop, rec_crop, channel_axis=2, data_range=1.0, win_size=win_size)
        )

    target_lpips = None
    if min_crop_side >= args.min_lpips_crop_size:
        target_lpips = compute_lpips(orig_crop, rec_crop, lpips_model, device)

    metrics = {
        "full_mse": compute_mse(orig, rec),
        "full_psnr": float(peak_signal_noise_ratio(orig, rec, data_range=1.0)),
        "full_ssim": float(structural_similarity(orig, rec, channel_axis=2, data_range=1.0)),
        "full_lpips": compute_lpips(orig, rec, lpips_model, device),
        "target_mse": compute_mse(orig_crop, rec_crop),
        "target_psnr": float(peak_signal_noise_ratio(orig_crop, rec_crop, data_range=1.0)),
        "target_ssim": target_ssim,
        "target_lpips": target_lpips,
    }
    return metrics


def compute_lpips(a, b, lpips_model, device):
    import torch

    a_tensor = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
    b_tensor = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device)
    a_tensor = a_tensor * 2.0 - 1.0
    b_tensor = b_tensor * 2.0 - 1.0
    with torch.no_grad():
        return float(lpips_model(a_tensor, b_tensor).item())


def safe_stem(record, split):
    index = record.get("original_index")
    if index is None:
        index = Path(record["fileName"]).stem
    return f"{split}_{index}_{Path(record['fileName']).stem}"


def save_reconstruction_images(original_512, reconstructed_512, box_512, record, split, args):
    vis_dir = Path(args.output_dir) / "visualizations" / split
    vis_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(record, split)

    left, top, right, bottom = int_crop_bounds(box_512)
    original_crop = original_512.crop((left, top, right, bottom))
    reconstructed_crop = reconstructed_512.crop((left, top, right, bottom))

    paths = {
        "original_512": vis_dir / f"{stem}_original_512.png",
        "reconstruction_512": vis_dir / f"{stem}_reconstruction_512.png",
        "target_original": vis_dir / f"{stem}_target_original.png",
        "target_reconstruction": vis_dir / f"{stem}_target_reconstruction.png",
    }
    original_512.save(paths["original_512"])
    reconstructed_512.save(paths["reconstruction_512"])
    original_crop.save(paths["target_original"])
    reconstructed_crop.save(paths["target_reconstruction"])
    return {key: str(path) for key, path in paths.items()}


class NullInversion:
    """Minimal RSVG-ZeroOV-style DDIM inversion and null-text optimization."""

    def __init__(self, model, device, num_ddim_steps, guidance_scale):
        import torch
        from diffusers import DDIMScheduler

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        model.scheduler = scheduler
        model.scheduler.set_timesteps(num_ddim_steps)
        self.model = model
        self.device = device
        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale
        self.context = None
        self.torch = torch

    @property
    def scheduler(self):
        return self.model.scheduler

    def prev_step(self, model_output, timestep, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        return alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = (
            min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        return alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction

    def get_noise_pred_single(self, latents, timestep, context):
        return self.model.unet(latents, timestep, encoder_hidden_states=context)["sample"]

    def get_noise_pred(self, latents, timestep, is_forward=True, context=None):
        latents_input = self.torch.cat([latents] * 2)
        context = self.context if context is None else context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(latents_input, timestep, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if is_forward:
            return self.next_step(noise_pred, timestep, latents)
        return self.prev_step(noise_pred, timestep, latents)

    def init_prompt(self, prompt):
        with self.torch.no_grad():
            uncond_input = self.model.tokenizer(
                [""],
                padding="max_length",
                max_length=self.model.tokenizer.model_max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.device))[0]
            text_input = self.model.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]
            self.context = self.torch.cat([uncond_embeddings, text_embeddings])

    def image_to_latent(self, image_512):
        with self.torch.no_grad():
            array = np.asarray(image_512.convert("RGB"))
            image = self.torch.from_numpy(array).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
            latents = self.model.vae.encode(image)["latent_dist"].mean
            return latents * 0.18215

    def latent_to_image(self, latents):
        with self.torch.no_grad():
            latents = 1 / 0.18215 * latents.detach()
            image = self.model.vae.decode(latents)["sample"]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            return Image.fromarray((image * 255).astype(np.uint8))

    def ddim_loop(self, latent):
        _, cond_embeddings = self.context.chunk(2)
        all_latents = [latent]
        latent = latent.clone().detach()
        with self.torch.no_grad():
            for i in range(self.num_ddim_steps):
                timestep = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
                noise_pred = self.get_noise_pred_single(latent, timestep, cond_embeddings)
                latent = self.next_step(noise_pred, timestep, latent)
                all_latents.append(latent)
        return all_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        import torch.nn.functional as nnf

        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = self.torch.optim.Adam([uncond_embeddings], lr=1e-2 * (1.0 - i / 100.0))
            latent_prev = latents[len(latents) - i - 2]
            timestep = self.model.scheduler.timesteps[i]
            with self.torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, timestep, cond_embeddings)
            for _ in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, timestep, uncond_embeddings)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, timestep, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss.item() < epsilon + i * 2e-5:
                    break
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with self.torch.no_grad():
                context = self.torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, timestep, False, context)
        return uncond_embeddings_list

    def invert(self, image_512, prompt, num_inner_steps, early_stop_epsilon):
        self.init_prompt(prompt)
        latent = self.image_to_latent(image_512)
        image_rec = self.latent_to_image(latent)
        ddim_latents = self.ddim_loop(latent)
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return image_rec, ddim_latents[-1], uncond_embeddings


def load_sd_components(args):
    import lpips
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
    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    inverter = NullInversion(pipe, device, args.num_ddim_steps, args.guidance_scale)
    return inverter, lpips_model, device


def analyze_record(record, split, image_root, reconstruction_context):
    image_path = image_root / record["fileName"]
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        image_512 = resize_rsvg_512(image)

    raw_box = [float(value) for value in record["bbox"]]
    clipped_box, was_clipped = clip_box_to_image(raw_box, width, height)
    if not box_valid(clipped_box):
        return None, {
            "split": split,
            "file_name": record["fileName"],
            "reason": "invalid_bbox_after_clipping",
            "gt_box_raw": raw_box,
            "gt_box_clipped": clipped_box,
        }

    box_512 = map_box_to_512(clipped_box, width, height)
    target_width = box_512[2] - box_512[0]
    target_height = box_512[3] - box_512[1]
    target_area = target_width * target_height
    clipped_area = (clipped_box[2] - clipped_box[0]) * (clipped_box[3] - clipped_box[1])
    area_ratio = clipped_area / float(width * height)

    cell_spans = {}
    for res in GRID_RESOLUTIONS:
        cell_spans[str(res)] = {
            "w": target_width * res / IMAGE_SIZE,
            "h": target_height * res / IMAGE_SIZE,
            "min": min(target_width, target_height) * res / IMAGE_SIZE,
        }

    reconstruction_metrics = metric_nulls()
    visualization_paths = None
    if reconstruction_context is not None:
        inverter, lpips_model, device, args = reconstruction_context
        image_rec, _, _ = inverter.invert(
            image_512,
            record["expression"],
            args.null_inner_steps,
            args.early_stop_epsilon,
        )
        reconstruction_metrics = compute_reconstruction_metrics(
            image_512,
            image_rec,
            box_512,
            lpips_model,
            device,
            args,
        )
        if args.save_reconstructions:
            visualization_paths = save_reconstruction_images(
                image_512,
                image_rec,
                box_512,
                record,
                split,
                args,
            )

    result = {
        "sample_id": f"{split}:{record.get('original_index', record.get('index', 'unknown'))}",
        "split": split,
        "original_index": record.get("original_index"),
        "file_name": record["fileName"],
        "source": record.get("source") or source_name(record["fileName"]),
        "image_width": width,
        "image_height": height,
        "expression": record["expression"],
        "expression_word_count": len(record["expression"].split()),
        "expression_length_bucket": expression_bucket(record["expression"]),
        "gt_box_raw": raw_box,
        "gt_box_clipped": clipped_box,
        "bbox_was_clipped": was_clipped,
        "bbox_512": box_512,
        "target_width_512": target_width,
        "target_height_512": target_height,
        "target_area_512": target_area,
        "min_side_512": min(target_width, target_height),
        "area_ratio_original_clipped": area_ratio,
        "target_size_bucket": area_bucket(area_ratio),
        "span_cells": cell_spans,
        "tiny_flags": {
            "lt_2px_at_512": min(target_width, target_height) < 2.0,
            "lt_4px_at_512": min(target_width, target_height) < 4.0,
            "lt_8px_at_512": min(target_width, target_height) < 8.0,
            "lt_1_cell_16": cell_spans["16"]["min"] < 1.0,
            "lt_1_cell_32": cell_spans["32"]["min"] < 1.0,
            "lt_1_cell_64": cell_spans["64"]["min"] < 1.0,
            "lt_2_cells_32": cell_spans["32"]["min"] < 2.0,
            "lt_2_cells_64": cell_spans["64"]["min"] < 2.0,
        },
        "reconstruction_metrics": reconstruction_metrics,
        "visualization_paths": visualization_paths,
    }
    return result, None


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_group(records):
    if not records:
        return {"count": 0}

    def values(key):
        return [float(record[key]) for record in records]

    summary = {
        "count": len(records),
        "target_width_512": summarize_values(values("target_width_512")),
        "target_height_512": summarize_values(values("target_height_512")),
        "target_area_512": summarize_values(values("target_area_512")),
        "min_side_512": summarize_values(values("min_side_512")),
        "area_ratio_original_clipped": summarize_values(values("area_ratio_original_clipped")),
        "bbox_clipped_count": sum(1 for record in records if record["bbox_was_clipped"]),
        "tiny_flag_rates": {},
    }
    for flag in records[0]["tiny_flags"]:
        summary["tiny_flag_rates"][flag] = sum(1 for record in records if record["tiny_flags"][flag]) / len(records)
    for res in GRID_RESOLUTIONS:
        key = str(res)
        summary[f"span_cells_{key}_min"] = summarize_values(
            [record["span_cells"][key]["min"] for record in records]
        )
    metric_records = [
        record for record in records if record["reconstruction_metrics"]["target_mse"] is not None
    ]
    summary["reconstruction_count"] = len(metric_records)
    if metric_records:
        summary["reconstruction_metrics"] = {}
        for key in METRIC_KEYS:
            metric_values = [
                record["reconstruction_metrics"][key]
                for record in metric_records
                if record["reconstruction_metrics"][key] is not None
            ]
            summary["reconstruction_metrics"][key] = {
                "valid_count": len(metric_values),
                "stats": summarize_values(metric_values) if metric_values else None,
            }
    return summary


def summarize_values(values):
    return {
        "mean": float(np.mean(values)),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_summary(results, skipped, args):
    summary = {
        "image_size": IMAGE_SIZE,
        "grid_resolutions": list(GRID_RESOLUTIONS),
        "bbox_policy": "clip_to_image_boundary_preserve_raw",
        "resize_policy": "direct_square_resize_512_like_rsvg_zeroov",
        "run_reconstruction": args.run_reconstruction,
        "model": args.sd_model if args.run_reconstruction else None,
        "num_ddim_steps": args.num_ddim_steps if args.run_reconstruction else None,
        "guidance_scale": args.guidance_scale if args.run_reconstruction else None,
        "min_lpips_crop_size": args.min_lpips_crop_size if args.run_reconstruction else None,
        "processed_count": len(results),
        "skipped_count": len(skipped),
        "skipped_by_reason": dict(sorted(reason_counts(skipped).items())),
        "overall": summarize_group(results),
        "by_split": {},
        "by_source": {},
        "by_target_size": {},
        "by_expression_length": {},
    }
    for key, label in (
        ("split", "by_split"),
        ("source", "by_source"),
        ("target_size_bucket", "by_target_size"),
        ("expression_length_bucket", "by_expression_length"),
    ):
        grouped = defaultdict(list)
        for record in results:
            grouped[record[key]].append(record)
        summary[label] = {
            group: summarize_group(group_records)
            for group, group_records in sorted(grouped.items())
        }
    return summary


def reason_counts(skipped):
    counts = defaultdict(int)
    for item in skipped:
        counts[item["reason"]] += 1
    return counts


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args):
    annotation_dir = Path(args.annotation_dir)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reconstruction_context = None
    if args.run_reconstruction:
        inverter, lpips_model, device = load_sd_components(args)
        reconstruction_context = (inverter, lpips_model, device, args)

    all_results = []
    all_skipped = []
    for split in args.splits:
        annotation_path = annotation_dir / f"{split.capitalize()}_10pct.json"
        records = read_json(annotation_path)
        if args.limit is not None:
            records = records[: args.limit]

        split_results = []
        split_skipped = []
        for item in records:
            try:
                result, skipped = analyze_record(item, split, image_root, reconstruction_context)
            except (FileNotFoundError, OSError) as exc:
                result = None
                skipped = {
                    "split": split,
                    "file_name": item.get("fileName"),
                    "reason": "missing_or_bad_image",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if result is not None:
                split_results.append(result)
            if skipped is not None:
                split_skipped.append(skipped)

        write_jsonl(output_dir / f"exp1_{split}_records.jsonl", split_results)
        write_jsonl(output_dir / f"exp1_{split}_skipped.jsonl", split_skipped)
        all_results.extend(split_results)
        all_skipped.extend(split_skipped)

    summary = build_summary(all_results, all_skipped, args)
    (output_dir / "exp1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir",
        default="data/foundation_probe_10pct/annotations",
        help="Directory containing Val_10pct.json and Test_10pct.json.",
    )
    parser.add_argument(
        "--image-root",
        default="data/foundation_probe_10pct/images",
        help="Directory containing selected SkyFind images.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/exp_1_sd_tiny_target_retention",
        help="Output directory for JSONL records and summary.",
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=("val", "test"))
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split smoke limit.")
    parser.add_argument("--run-reconstruction", action="store_true")
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--sd-model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--min-lpips-crop-size", type=int, default=64)
    parser.add_argument("--num-ddim-steps", type=int, default=NUM_DDIM_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--null-inner-steps", type=int, default=10)
    parser.add_argument("--early-stop-epsilon", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
