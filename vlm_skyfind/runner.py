"""Resumable single-model SkyFind inference loop."""

import json
import time
import traceback
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from .adapters import create_adapter
from .boxes import (
    box_iou,
    extract_four_coordinates,
    parse_prediction,
    validate_box_strict,
)
from .coordinates import resolve_coordinate_mode
from .data import InvalidImageError, SkyFindDataset, source_name
from .metrics import summarize, write_summary
from .mixed_coordinates import convert_internvl_official, convert_uncontracted_vlm
from .prompts import build_prompt, prompt_template
from .qwen_coordinates import load_preprocessor_config, restore_coordinates


STRICT_NATIVE_MODES = {
    "qwen_resized_pixel",
    "internvl_official_mixed",
    "uncontracted_vlm_strict",
}


def completed_sample_ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                completed.add(json.loads(line)["sample_id"])
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid resume file {path}:{line_number}") from exc
    return completed


def _append_record(handle, record):
    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    handle.flush()


def _write_or_validate_protocol(
    args,
    output_path,
    resolved_coordinate_mode,
    coordinate_mode_basis,
    qwen_config=None,
):
    protocol_path = Path(str(output_path) + ".meta.json")
    protocol = {
        "model": args.model,
        "model_path": str(Path(args.model_path).resolve()),
        "split": args.split,
        "prompt_variant": args.prompt_variant,
        "prompt_template": prompt_template(args.prompt_variant),
        "coordinate_mode_requested": args.coordinate_mode,
        "coordinate_mode": resolved_coordinate_mode,
        "coordinate_mode_basis": coordinate_mode_basis,
        "box_validation": (
            "strict_xyxy_no_reorder_no_clamp"
            if resolved_coordinate_mode in STRICT_NATIVE_MODES
            else "sanitize_reorder_and_clamp"
        ),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "internvl_max_tiles": args.internvl_max_tiles,
        "conversation_mode": args.conversation_mode,
        "data_root": str(Path(args.data_root).resolve()),
        "image_dir": str(Path(args.image_dir).resolve()) if args.image_dir else None,
        "source_prefixes": args.source_prefixes,
    }
    if args.model == "llava-onevision-7b":
        protocol["llava_model_name"] = getattr(
            args, "llava_model_name", "llava_qwen"
        )
    if qwen_config is not None:
        protocol["qwen_preprocessor_config"] = str(
            (Path(args.model_path) / "preprocessor_config.json").resolve()
        )
        protocol["qwen_resize"] = qwen_config
    if args.resume and protocol_path.exists():
        with protocol_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != protocol:
            raise ValueError(
                f"Resume protocol mismatch for {output_path}: "
                f"existing={previous}, requested={protocol}"
            )
    else:
        with protocol_path.open("w", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, ensure_ascii=True)
            handle.write("\n")


def _parse_response(
    response, width, height, resolved_coordinate_mode, qwen_config=None
):
    if resolved_coordinate_mode == "qwen_resized_pixel":
        values = extract_four_coordinates(response)
        if values is None:
            return None, resolved_coordinate_mode, {}
        restored, processed_width, processed_height = restore_coordinates(
            values, width, height, qwen_config
        )
        return (
            validate_box_strict(restored),
            resolved_coordinate_mode,
            {
                "qwen_processed_width": processed_width,
                "qwen_processed_height": processed_height,
            },
        )
    if resolved_coordinate_mode == "internvl_official_mixed":
        box, detected_mode = convert_internvl_official(
            extract_four_coordinates(response), width, height
        )
        return box, detected_mode, {}
    if resolved_coordinate_mode == "uncontracted_vlm_strict":
        box, detected_mode = convert_uncontracted_vlm(
            extract_four_coordinates(response), width, height, response
        )
        return box, detected_mode, {}

    box, detected_mode = parse_prediction(
        response, width, height, coordinate_mode=resolved_coordinate_mode
    )
    return box, detected_mode, {}


def run(args):
    dataset = SkyFindDataset(
        args.data_root,
        args.split,
        image_dir=args.image_dir,
        source_prefixes=args.source_prefixes,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_coordinate_mode, coordinate_mode_basis = resolve_coordinate_mode(
        args.model, args.coordinate_mode
    )
    qwen_config = None
    if resolved_coordinate_mode == "qwen_resized_pixel":
        qwen_config = load_preprocessor_config(
            Path(args.model_path) / "preprocessor_config.json"
        )
    _write_or_validate_protocol(
        args,
        output_path,
        resolved_coordinate_mode,
        coordinate_mode_basis,
        qwen_config=qwen_config,
    )
    done = completed_sample_ids(output_path) if args.resume else set()
    indices = list(range(len(dataset)))
    if args.start_index:
        indices = [index for index in indices if index >= args.start_index]
    if args.limit is not None:
        indices = indices[:args.limit]

    adapter = create_adapter(
        args.model,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
        max_tiles=args.internvl_max_tiles,
        conversation_mode=args.conversation_mode,
        llava_model_name=getattr(args, "llava_model_name", "llava_qwen"),
    )

    mode = "a" if args.resume else "w"
    records = []
    consecutive_inference_errors = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for index in tqdm(indices, desc=f"{args.model}:{args.split}"):
            fallback_id = f"{args.split}:{dataset.samples[index][0]}"
            if fallback_id in done:
                continue
            try:
                sample = dataset[index]
            except InvalidImageError as exc:
                annotation_index, raw_sample = dataset.samples[index]
                record = {
                    "sample_id": fallback_id,
                    "annotation_index": annotation_index,
                    "split": args.split,
                    "model": args.model,
                    "file_name": raw_sample.get("fileName"),
                    "source": source_name(raw_sample.get("fileName", "")),
                    "status": "image_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _append_record(handle, record)
                records.append(record)
                continue

            prompt = build_prompt(
                sample["expression"], sample["width"], sample["height"], args.prompt_variant
            )
            start = time.perf_counter()
            try:
                response = adapter.generate(sample["image_path"], prompt)
                latency = time.perf_counter() - start
                pred_box, detected_mode, coordinate_details = _parse_response(
                    response,
                    sample["width"],
                    sample["height"],
                    resolved_coordinate_mode,
                    qwen_config=qwen_config,
                )
                status = "ok" if pred_box is not None else "parse_error"
                iou = box_iou(pred_box, sample["gt_box"])
                record = {
                    **{key: value for key, value in sample.items() if key != "image_path"},
                    "model": args.model,
                    "model_path": args.model_path,
                    "prompt_variant": args.prompt_variant,
                    "prompt": prompt,
                    "raw_response": response,
                    "coordinate_mode_requested": args.coordinate_mode,
                    "coordinate_mode_resolved": resolved_coordinate_mode,
                    "coordinate_mode_basis": coordinate_mode_basis,
                    "coordinate_mode": detected_mode,
                    "box_validation": (
                        "strict_xyxy_no_reorder_no_clamp"
                        if resolved_coordinate_mode in STRICT_NATIVE_MODES
                        else "sanitize_reorder_and_clamp"
                    ),
                    **coordinate_details,
                    "pred_box": pred_box,
                    "iou": iou,
                    "latency_seconds": latency,
                    "status": status,
                }
            except Exception as exc:
                record = {
                    **{key: value for key, value in sample.items() if key != "image_path"},
                    "model": args.model,
                    "model_path": args.model_path,
                    "prompt_variant": args.prompt_variant,
                    "prompt": prompt,
                    "coordinate_mode_requested": args.coordinate_mode,
                    "coordinate_mode_resolved": resolved_coordinate_mode,
                    "coordinate_mode_basis": coordinate_mode_basis,
                    "pred_box": None,
                    "iou": 0.0,
                    "latency_seconds": time.perf_counter() - start,
                    "status": "inference_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if args.save_tracebacks:
                    record["traceback"] = traceback.format_exc()
            _append_record(handle, record)
            records.append(record)
            if record["status"] == "inference_error":
                consecutive_inference_errors += 1
                if consecutive_inference_errors >= args.max_consecutive_errors:
                    raise RuntimeError(
                        f"Stopped after {consecutive_inference_errors} consecutive "
                        "inference errors; inspect the JSONL records before resuming"
                    )
            else:
                consecutive_inference_errors = 0

    if args.summary_output:
        from .metrics import load_jsonl

        write_summary(summarize(load_jsonl([output_path])), args.summary_output)
    return records
