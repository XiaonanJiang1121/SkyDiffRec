# Experiment 2 Design: SD Cross-Attention GT Response

Date: 2026-06-26

Status: design for review. Do not implement until details are confirmed.

Server project path:

```text
/root/autodl-tmp/DiffusionSkyFind
```

## 1. Goal

Experiment 2 tests whether frozen Stable Diffusion v1.4 cross-attention
responds near the SkyFind target under several prompt variants.

This experiment does not assume diffusion is the coarse module. It asks:

```text
Can SD cross-attention provide a weak target-related spatial signal?
```

## 2. Reference Implementation

Use RSVG-ZeroOV as the first reference.

Relevant local reference files:

```text
BioLoc/reference/RSVG-ZeroOV/generate_png.py
BioLoc/reference/RSVG-ZeroOV/ptp_utils.py
```

RSVG-ZeroOV behavior to preserve:

```text
StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
NUM_DDIM_STEPS = 20
GUIDANCE_SCALE = 7.5
direct 512 x 512 square resize
AttentionStore for UNet attention capture
aggregate_attention(..., res=16/32/64, from_where=["down", "up"], is_cross=True)
```

Important difference:

```text
RSVG-ZeroOV is segmentation-oriented and later fuses with self-attention/SAM.
Experiment 2 only evaluates cross-attention response near GT.
```

## 3. Inputs

Use the same 10% probing subset as Experiment 1:

```text
data/foundation_probe_10pct/annotations/Val_10pct.json
data/foundation_probe_10pct/annotations/Test_10pct.json
data/foundation_probe_10pct/images/
```

For server setup, recreate or sync this data under:

```text
/root/autodl-tmp/DiffusionSkyFind/data/foundation_probe_10pct/
```

Bad / missing images:

```text
skip and record in exp2_*_skipped.jsonl
```

BBox boundary policy:

```text
preserve gt_box_raw
clip gt_box_clipped to image boundary
map clipped bbox to 512 for metric evaluation
record bbox_was_clipped
```

This matches Experiment 1 and avoids silently losing annotation information.

## 4. Prompt Variants

Run each sample with the following prompt variants.

### P1: Full Expression

```text
{expression}
```

Purpose:

```text
Test whether SD's CLIP text encoder responds to the original SkyFind language.
```

### P2: Object Phrase

```text
{object_phrase}
```

Object phrase extraction should be deterministic and auditable.

Initial implementation choice:

```text
Use a simple noun-phrase extractor only if we explicitly choose one before
coding. Otherwise require a precomputed object_phrase field.
```

No LLM fallback should be hidden inside the experiment script.

### P3: Remote-Sensing Object Prompt

```text
a remote sensing image of {object_phrase}
```

Purpose:

```text
Test whether a short domain prompt improves SD cross-attention compared with
raw object phrase prompting.
```

### P4: RSVG/VLM Localization Prompt

```text
Locate it according to the following description.  {expression} The output
format should be like [x1, y1, x2, y2] without any other text.
```

Purpose:

```text
Test whether the RSVG-style localization instruction changes SD attention,
even though SD is not a box-output VLM.
```

## 5. Text Length and Token Diagnostics

For each prompt variant, record:

```json
{
  "prompt_variant": "p1_full_expression",
  "prompt": "...",
  "clip_token_count_with_special": 77,
  "clip_token_count_without_special": 75,
  "clip_truncated": true,
  "tokens": ["..."]
}
```

Stable Diffusion v1.4 uses the CLIP tokenizer length limit. Long SkyFind
expressions may be truncated. This is a key diagnostic, not a nuisance field.

Prompt splitting:

```text
Do not implement RSVG-style sentence splitting in the first Experiment 2 code
unless we explicitly decide to test it as an additional variant.
```

Reason:

```text
The first result should tell us whether the direct prompt variants are useful.
Prompt splitting changes the question and should be a controlled ablation.
```

## 6. Attention Extraction

For each sample and prompt variant:

```text
1. resize image to 512 x 512 using the same function as Experiment 1
2. run SD inversion / null-text optimization as in Experiment 1
3. rerun the diffusion process with AttentionStore registered
4. aggregate cross-attention at 16 / 32 / 64
5. save prompt-level and token-level attention tensors
```

Expected raw tensor forms:

```text
cross16: [16, 16, token_count]
cross32: [32, 32, token_count]
cross64: [64, 64, token_count]
```

Phrase aggregation:

```text
For P1/P4, aggregate attention over selected target/object tokens if available.
For P2/P3, aggregate over all non-special tokens of the object phrase.
```

If token selection is not available for P1/P4, use all non-special tokens for
the first implementation and record:

```json
"token_selection_policy": "all_non_special_tokens"
```

## 7. Metrics

Compute metrics on each attention resolution separately, and optionally on a
simple average of upsampled maps.

Metrics:

```text
Pointing Game:
  attention peak inside GT_512

Top-k Hit:
  top 1%, 5%, 10% attention pixels overlap GT_512

GT Attention Ratio:
  sum attention inside GT_512 / sum attention full map

Peak Center Distance:
  distance(attention peak, GT center) / GT diagonal

Entropy / Peakness:
  diagnose diffuse attention
```

Heatmap-to-box IoU:

```text
Not primary in Experiment 2.
If implemented, report as diagnostic only and do not over-interpret it as final
REC performance.
```

## 8. Controls

Run controls on a smaller subset first because each control multiplies SD
inversion cost.

Required controls for a smoke run:

```text
C1 same image + wrong noun
C2 same image + generic prompt
C3 wrong image + same expression
```

Suggested formal control strategy:

```text
run controls on 20-50 samples first
expand only if positive attention response appears meaningful
```

Controls should write the same metric schema as positive prompts and include:

```json
"is_control": true
```

## 9. Outputs

Suggested output directory:

```text
results/exp_2_sd_cross_attention_response/
```

Files:

```text
exp2_val_records.jsonl
exp2_test_records.jsonl
exp2_val_skipped.jsonl
exp2_test_skipped.jsonl
exp2_summary.json
```

Large tensors:

```text
results/exp_2_sd_cross_attention_response/attention_tensors/
```

Tensor files should be ignored by git.

Per-record fields:

```json
{
  "sample_id": "val:0",
  "split": "val",
  "file_name": "...jpg",
  "source": "UAVDT",
  "prompt_variant": "p1_full_expression",
  "prompt": "...",
  "clip_truncated": false,
  "gt_box_raw": [x1, y1, x2, y2],
  "gt_box_clipped": [x1, y1, x2, y2],
  "bbox_512": [x1, y1, x2, y2],
  "target_width_512": 0.0,
  "target_height_512": 0.0,
  "attention_paths": {
    "cross16": "...pt",
    "cross32": "...pt",
    "cross64": "...pt"
  },
  "metrics": {
    "cross16": {},
    "cross32": {},
    "cross64": {}
  }
}
```

## 10. Implementation Boundaries

Follow the current code-writing rules:

```text
skip bad / missing images and record them
preserve raw bbox and use clipped bbox for visible-region metrics
do not silently repair prompts
do not silently replace failed object phrase extraction
do not add SAM
do not add self-attention fusion
do not add crop variants
do not add heatmap-to-box as a primary metric
```

Experiment 2 should be implemented as a new script only after this design is
confirmed:

```text
foundation_probing/tools/run_exp2_sd_cross_attention_response.py
```

## 11. Decisions Needed Before Coding

Please confirm these details before implementation:

```text
1. Object phrase source:
   a) require precomputed object_phrase field
   b) use a deterministic local noun-phrase extractor
   c) skip P2/P3 in the first implementation

2. Prompt splitting:
   a) no splitting in first implementation
   b) add RSVG-style split prompt as P5

3. Controls:
   a) smoke controls only first
   b) controls for the full 10% subset

4. Attention tensor storage:
   a) save full token-level cross16/32/64 tensors
   b) save only phrase-aggregated heatmaps plus token diagnostics
```

