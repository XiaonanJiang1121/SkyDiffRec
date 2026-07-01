#!/usr/bin/env python3
"""Run Experiment 2 tokenizer and entity-set audit.

This script does not run Stable Diffusion. It checks the prompt/entity design
before GPU attention probing:

1. extract a LazyMCoT-inspired entity set from each SkyFind expression
2. build confirmed Experiment 2 prompt variants
3. record Stable-Diffusion CLIP tokenizer length/truncation diagnostics
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from skyfind_entity_extraction import entity_set_text, extract_entities, select_wrong_phrase


PROMPT_PREFIX = "Locate it according to the following description. "


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def percentile(values, q):
    if not values:
        return None
    values = sorted(float(value) for value in values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def summarize_values(values):
    values = [float(value) for value in values]
    if not values:
        return None
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "min": min(values),
        "max": max(values),
    }


def load_tokenizer(args):
    from transformers import CLIPTokenizerFast

    kwargs = {}
    if args.tokenizer_subfolder:
        kwargs["subfolder"] = args.tokenizer_subfolder
    kwargs["local_files_only"] = args.local_files_only
    return CLIPTokenizerFast.from_pretrained(args.sd_model, **kwargs)


def offset_token_indices(offsets, char_start, char_end):
    indices = []
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if max(start, char_start) < min(end, char_end):
            indices.append(index)
    return indices


def tokenizer_diagnostics(tokenizer, prompt, spans):
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_offsets_mapping=True,
    )
    untruncated = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special_ids = set(tokenizer.all_special_ids)
    non_special_indices = [
        index
        for index, token_id in enumerate(input_ids)
        if token_id not in special_ids and offsets[index][1] > offsets[index][0]
    ]
    span_token_indices = {
        name: offset_token_indices(offsets, span["char_start"], span["char_end"])
        for name, span in spans.items()
    }
    entity_span_names = sorted(
        [name for name in span_token_indices if name.startswith("entity_")],
        key=lambda name: int(name.split("_", 1)[1]),
    )
    all_entity_token_indices = sorted(
        {
            index
            for name in entity_span_names
            for index in span_token_indices[name]
        }
    )
    return {
        "clip_model_max_length": tokenizer.model_max_length,
        "clip_token_count_with_special": len(untruncated["input_ids"]),
        "clip_token_count_without_special": max(0, len(untruncated["input_ids"]) - 2),
        "clip_active_token_count": len(non_special_indices),
        "clip_truncated": len(untruncated["input_ids"]) > tokenizer.model_max_length,
        "tokens": tokens,
        "non_special_token_indices": non_special_indices,
        "span_token_indices": span_token_indices,
        "all_entity_token_indices": all_entity_token_indices,
    }


def build_prompt_variants(expression, entities):
    variants = []
    target_entity = entities[0] if entities else None
    entity_text = entity_set_text(entities)

    variants.append(
        {
            "prompt_variant": "p1_full_expression",
            "prompt": expression,
            "requires_entities": False,
            "spans": entity_spans_for_expression(entities, offset=0),
        }
    )
    if entities:
        variants.append(
            {
                "prompt_variant": "p2_entity_set",
                "prompt": entity_text,
                "requires_entities": True,
                "spans": entity_spans_for_entity_prompt(entities, prefix=""),
            }
        )
        prefix = "a remote sensing image containing "
        variants.append(
            {
                "prompt_variant": "p3_domain_entity_set",
                "prompt": prefix + entity_text,
                "requires_entities": True,
                "spans": entity_spans_for_entity_prompt(entities, prefix=prefix),
            }
        )

    variants.append(
        {
            "prompt_variant": "p4_vlm_localization",
            "prompt": PROMPT_PREFIX + expression,
            "requires_entities": False,
            "spans": entity_spans_for_expression(entities, offset=len(PROMPT_PREFIX)),
        }
    )
    if entities:
        wrong_phrase = select_wrong_phrase(target_entity["head"])
        variants.append(
            {
                "prompt_variant": "c1_wrong_object_phrase",
                "prompt": wrong_phrase,
                "requires_entities": True,
                "control_type": "wrong_object_phrase",
                "wrong_phrase": wrong_phrase,
                "spans": {
                    "wrong_object": {
                        "char_start": 0,
                        "char_end": len(wrong_phrase),
                    }
                },
            }
        )
    return variants


def entity_spans_for_expression(entities, offset):
    spans = {}
    if entities:
        spans["target_entity"] = {
            "char_start": entities[0]["char_start"] + offset,
            "char_end": entities[0]["char_end"] + offset,
        }
        for index, entity in enumerate(entities):
            spans[f"entity_{index}"] = {
                "char_start": entity["char_start"] + offset,
                "char_end": entity["char_end"] + offset,
            }
    return spans


def entity_spans_for_entity_prompt(entities, prefix):
    spans = {}
    cursor = len(prefix)
    for index, entity in enumerate(entities):
        phrase = entity["surface_phrase"]
        start = cursor
        end = start + len(phrase)
        spans[f"entity_{index}"] = {"char_start": start, "char_end": end}
        if index == 0:
            spans["target_entity"] = {"char_start": start, "char_end": end}
        cursor = end + 2
    if entities:
        spans["all_entities"] = {
            "char_start": len(prefix),
            "char_end": cursor - 2,
        }
    return spans


def analyze_record(record, split, tokenizer):
    expression = record["expression"]
    entities = extract_entities(expression)
    prompt_records = []
    for variant in build_prompt_variants(expression, entities):
        prompt_record = {
            "prompt_variant": variant["prompt_variant"],
            "prompt": variant["prompt"],
            "requires_entities": variant.get("requires_entities", False),
            "control_type": variant.get("control_type"),
            "wrong_phrase": variant.get("wrong_phrase"),
            "char_spans": variant["spans"],
        }
        if tokenizer is not None:
            prompt_record["tokenizer"] = tokenizer_diagnostics(
                tokenizer,
                variant["prompt"],
                variant["spans"],
            )
        prompt_records.append(prompt_record)

    return {
        "sample_id": f"{split}:{record.get('original_index', record.get('index', 'unknown'))}",
        "split": split,
        "original_index": record.get("original_index"),
        "file_name": record["fileName"],
        "source": record.get("source"),
        "expression": expression,
        "expression_word_count": len(expression.split()),
        "entities": entities,
        "target_entity": entities[0] if entities else None,
        "entity_count": len(entities),
        "entity_set_text": entity_set_text(entities),
        "prompt_records": prompt_records,
    }


def summarize(records):
    prompt_groups = defaultdict(list)
    for record in records:
        for prompt_record in record["prompt_records"]:
            prompt_groups[prompt_record["prompt_variant"]].append(prompt_record)

    prompt_summary = {}
    for variant, prompt_records in sorted(prompt_groups.items()):
        tokenizer_records = [
            prompt_record["tokenizer"]
            for prompt_record in prompt_records
            if "tokenizer" in prompt_record
        ]
        item = {
            "count": len(prompt_records),
            "requires_entities_count": sum(
                1 for prompt_record in prompt_records if prompt_record["requires_entities"]
            ),
        }
        if tokenizer_records:
            item.update(
                {
                    "clip_token_count_with_special": summarize_values(
                        [
                            tokenizer_record["clip_token_count_with_special"]
                            for tokenizer_record in tokenizer_records
                        ]
                    ),
                    "clip_active_token_count": summarize_values(
                        [
                            tokenizer_record["clip_active_token_count"]
                            for tokenizer_record in tokenizer_records
                        ]
                    ),
                    "clip_truncated_count": sum(
                        1 for tokenizer_record in tokenizer_records if tokenizer_record["clip_truncated"]
                    ),
                    "clip_truncated_rate": sum(
                        1 for tokenizer_record in tokenizer_records if tokenizer_record["clip_truncated"]
                    )
                    / len(tokenizer_records),
                    "target_entity_token_missing_count": sum(
                        1
                        for tokenizer_record in tokenizer_records
                        if "target_entity" in tokenizer_record["span_token_indices"]
                        and not tokenizer_record["span_token_indices"]["target_entity"]
                    ),
                }
            )
        prompt_summary[variant] = item

    head_counts = Counter()
    phrase_counts = Counter()
    for record in records:
        for entity in record["entities"]:
            head_counts[entity["head"]] += 1
            phrase_counts[entity["surface_phrase"]] += 1

    return {
        "record_count": len(records),
        "entity_extracted_count": sum(1 for record in records if record["entities"]),
        "entity_missing_count": sum(1 for record in records if not record["entities"]),
        "entity_missing_rate": sum(1 for record in records if not record["entities"]) / len(records)
        if records
        else None,
        "entity_count": summarize_values([record["entity_count"] for record in records]),
        "expression_word_count": summarize_values(
            [record["expression_word_count"] for record in records]
        ),
        "prompt_variants": prompt_summary,
        "top_entity_heads": head_counts.most_common(50),
        "top_entity_phrases": phrase_counts.most_common(50),
    }


def run(args):
    tokenizer = None if args.skip_tokenizer else load_tokenizer(args)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    by_split = {}
    for split in args.splits:
        records = read_json(annotation_dir / f"{split.capitalize()}_10pct.json")
        if args.limit is not None:
            records = records[: args.limit]
        analyzed = [analyze_record(record, split, tokenizer) for record in records]
        write_jsonl(output_dir / f"exp2_{split}_tokenizer_object_audit.jsonl", analyzed)
        by_split[split] = summarize(analyzed)
        all_records.extend(analyzed)

    summary = {
        "splits": args.splits,
        "tokenizer_enabled": tokenizer is not None,
        "sd_model": args.sd_model if tokenizer is not None else None,
        "tokenizer_subfolder": args.tokenizer_subfolder if tokenizer is not None else None,
        "overall": summarize(all_records),
        "by_split": by_split,
    }
    (output_dir / "exp2_tokenizer_object_audit_summary.json").write_text(
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
        "--output-dir",
        default="results/exp_2_tokenizer_object_audit",
        help="Output directory for audit JSONL and summary.",
    )
    parser.add_argument("--splits", nargs="+", default=["val"], choices=("val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sd-model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--tokenizer-subfolder", default="tokenizer")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Only run deterministic entity extraction; do not load CLIP tokenizer.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
