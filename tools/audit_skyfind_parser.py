#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

import sys


def build_split_file_map():
    return {
        "train": "Train.json",
        "train_aug": "Train_Aug.json",
        "val": "Val.json",
        "test": "Test.json",
    }


def load_parser_functions(repo_root):
    baseline_root = repo_root / "baselines" / "diffusionrec_original"
    sys.path.insert(0, str(baseline_root))
    from datasets.data_loader import prompt_convert, ralation_analysis
    return prompt_convert, ralation_analysis


def iter_split_expressions(data_root, split):
    split_file = build_split_file_map()[split]
    annotation_path = data_root / split_file
    with annotation_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    if split == "train_aug":
        for idx, sample in enumerate(samples):
            for field in ("expression_aug1", "expression_aug2"):
                expression = sample.get(field, "")
                if isinstance(expression, str) and expression.strip():
                    yield idx, sample["fileName"], field, expression.strip()
    else:
        for idx, sample in enumerate(samples):
            expression = sample["expression"]
            if isinstance(expression, str) and expression.strip():
                yield idx, sample["fileName"], "expression", expression.strip()


def percentile(sorted_values, ratio):
    if not sorted_values:
        return 0
    index = int(ratio * (len(sorted_values) - 1))
    return sorted_values[index]


def summarize_lengths(values):
    if not values:
        return {"mean": 0.0, "p95": 0, "max": 0}
    sorted_values = sorted(values)
    return {
        "mean": sum(values) / float(len(values)),
        "p95": percentile(sorted_values, 0.95),
        "max": sorted_values[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Audit SkyFind parser/tokenization behavior for DiffusionREC.")
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=sorted(build_split_file_map()))
    parser.add_argument("--bert_model", required=True, type=str)
    parser.add_argument("--max_query_len", default=128, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Optional max number of expressions to inspect.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    prompt_convert, ralation_analysis = load_parser_functions(repo_root)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)

    total = 0
    parser_failed_count = 0
    truncated_count = 0
    exact_match_count = 0
    shorter_than_token_count = 0
    longer_than_token_count = 0

    raw_expression_words = []
    raw_expression_token_lens = []
    prompted_token_lens = []
    actual_token_lens = []

    failed_examples = []
    truncated_examples = []

    for idx, file_name, field_name, expression in iter_split_expressions(args.data_root, args.split):
        if args.limit is not None and total >= args.limit:
            break

        total += 1
        prompt_text = prompt_convert(expression)
        raw_expression_words.append(len(expression.split()))
        raw_expression_token_lens.append(len(tokenizer.encode(expression, add_special_tokens=True)))
        prompted_len = len(tokenizer.encode(prompt_text, add_special_tokens=True))
        prompted_token_lens.append(prompted_len)

        encoded = tokenizer.batch_encode_plus(
            [prompt_text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_query_len,
        )
        actual_len = int(encoded["attention_mask"][0].sum().item())
        actual_token_lens.append(actual_len)
        if prompted_len > args.max_query_len:
            truncated_count += 1
            if len(truncated_examples) < 10:
                truncated_examples.append(
                    {
                        "index": idx,
                        "file_name": file_name,
                        "field": field_name,
                        "prompted_token_len": prompted_len,
                        "expression": expression[:220],
                    }
                )

        try:
            words_mask = ralation_analysis(expression)
        except Exception as exc:
            parser_failed_count += 1
            if len(failed_examples) < 10:
                failed_examples.append(
                    {
                        "index": idx,
                        "file_name": file_name,
                        "field": field_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "expression": expression[:220],
                    }
                )
            continue

        parser_len = len(words_mask)
        if parser_len == actual_len:
            exact_match_count += 1
        elif parser_len < actual_len:
            shorter_than_token_count += 1
        else:
            longer_than_token_count += 1

    raw_word_stats = summarize_lengths(raw_expression_words)
    raw_token_stats = summarize_lengths(raw_expression_token_lens)
    prompted_token_stats = summarize_lengths(prompted_token_lens)
    actual_token_stats = summarize_lengths(actual_token_lens)

    print(
        "[parser-audit] split={split} total={total} parser_failed_count={parser_failed_count} "
        "truncated_count={truncated_count} truncated_ratio={truncated_ratio:.6f} "
        "exact_match_count={exact_match_count} shorter_than_token_count={shorter_than_token_count} "
        "longer_than_token_count={longer_than_token_count}".format(
            split=args.split,
            total=total,
            parser_failed_count=parser_failed_count,
            truncated_count=truncated_count,
            truncated_ratio=(truncated_count / float(total)) if total else 0.0,
            exact_match_count=exact_match_count,
            shorter_than_token_count=shorter_than_token_count,
            longer_than_token_count=longer_than_token_count,
        )
    )
    print(
        "[parser-audit:lengths] raw_expression_words mean={:.2f} p95={} max={}".format(
            raw_word_stats["mean"], raw_word_stats["p95"], raw_word_stats["max"]
        )
    )
    print(
        "[parser-audit:lengths] raw_expression_token_len mean={:.2f} p95={} max={}".format(
            raw_token_stats["mean"], raw_token_stats["p95"], raw_token_stats["max"]
        )
    )
    print(
        "[parser-audit:lengths] prompted_token_len mean={:.2f} p95={} max={}".format(
            prompted_token_stats["mean"], prompted_token_stats["p95"], prompted_token_stats["max"]
        )
    )
    print(
        "[parser-audit:lengths] actual_token_len_after_trunc mean={:.2f} p95={} max={}".format(
            actual_token_stats["mean"], actual_token_stats["p95"], actual_token_stats["max"]
        )
    )

    if failed_examples:
        print("[parser-audit:failed_examples]")
        for example in failed_examples:
            print(json.dumps(example, ensure_ascii=False))

    if truncated_examples:
        print("[parser-audit:truncated_examples]")
        for example in truncated_examples:
            print(json.dumps(example, ensure_ascii=False))


if __name__ == "__main__":
    main()
