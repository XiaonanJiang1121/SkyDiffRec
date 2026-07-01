# Experiment 2 Design: SD Cross-Attention GT Response

Date: 2026-07-01

Status: tokenizer/entity audit implemented. SD attention smoke implementation
is being debugged with one-sample numerical ablations before the smoke-30 run.

Server project path:

```text
/root/autodl-tmp/DiffusionSkyFind
```

## 1. Goal

Experiment 2 tests whether frozen Stable Diffusion v1.4 cross-attention
responds near the SkyFind target under several prompt policies.

This experiment does not assume diffusion is the coarse module. It asks:

```text
Can SD cross-attention provide a weak target-related spatial signal?
Which prompt policy gives the cleanest target-related response?
```

Experiment 2 must be completed before Experiment 3/4 because the SD denoising
trajectory, including self-attention, is still conditioned by prompt/context.

## 2. Reference Implementation

Use RSVG-ZeroOV as the first SD attention reference.

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

LazyMCoT reference for entity extraction:

```text
LazyMCoT first decomposes the question into canonical entities E, including
target and referring objects. The entity set is then used by both its visual
expert branch and attention branch.
```

SkyFind first implementation follows the idea but uses a deterministic
auditable noun-phrase extractor before any optional VLM/LLM entity extraction
is introduced.

## 3. Inputs

Use the Val split of the 10% foundation probing subset:

```text
data/foundation_probe_10pct/annotations/Val_10pct.json
data/foundation_probe_10pct/images/
```

All exploratory experiments after Experiment 1 are Val-only first. Test is not
used for prompt/attention design.

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

## 4. Stage A: Tokenizer And Entity Audit

Code:

```text
foundation_probing/tools/run_exp2_sd_cross_attention_smoke.py
foundation_probing/tools/run_exp2_tokenizer_object_audit.py
foundation_probing/tools/skyfind_entity_extraction.py
```

Purpose:

```text
1. extract target object and referring objects from every expression
2. build the confirmed prompt variants
3. audit SD/CLIP tokenizer length and truncation
4. record target_entity token spans and all_entities token spans
```

The audit runs on CPU and does not run Stable Diffusion.

Output:

```text
results/exp_2_tokenizer_object_audit/
```

Server command:

```bash
cd /root/autodl-tmp/DiffusionSkyFind
python foundation_probing/tools/run_exp2_tokenizer_object_audit.py \
  --splits val \
  --sd-model /root/autodl-tmp/DiffusionSkyFind/stable-diffusion-v1-4 \
  --local-files-only \
  --output-dir results/exp_2_tokenizer_object_audit
```

## 5. Entity Extraction Policy

Current annotation fields:

```text
fileName / bbox / expression / source / imagePath / original_index
```

There is no object category field, so object/entity phrases must be extracted
from the expression.

Confirmed first implementation:

```text
use deterministic SkyFind entity extraction
do not rely on a fixed positive object-category lexicon
extract noun-like phrases with syntactic boundaries and spatial/function-word filtering
extract target object + referring objects
record target_entity as the first extracted entity
record all_entities as the full extracted entity set
do not hide an LLM fallback in the experiment script
```

Optional later implementation:

```text
use a downloaded VLM/LLM to run LazyMCoT-style decomposition
save its outputs as an explicit JSON/JSONL input
audit the saved entity file before attention probing
```

## 6. Prompt Variants

### P1: Full Expression

```text
{expression}
```

Tests the original SkyFind language.

### P2: Entity Set

```text
{entity_1}, {entity_2}, ..., {entity_n}
```

Tests whether a LazyMCoT-style object/entity list gives cleaner attention than
the full expression.

### P3: Domain Entity-Set Prompt

```text
a remote sensing image containing {entity_1}, {entity_2}, ..., {entity_n}
```

Tests whether a short remote-sensing prompt improves SD cross-attention over
the raw entity list.

### P4: VLM Localization Prompt

```text
Locate it according to the following description. {expression}
```

This follows the VLM localization style without adding an output-format
instruction that is irrelevant to SD.

### C1: Wrong Object Phrase Control

```text
{wrong_object_phrase}
```

The wrong object phrase is deterministically chosen from a different coarse
category than the target entity.

## 7. Token Aggregation

Primary metric map:

```text
aggregate cross-attention over target_entity tokens
```

Diagnostic entity-set map:

```text
aggregate cross-attention over all_entities tokens
```

All-token diagnostic:

```text
aggregate cross-attention over all non-special tokens
```

If target_entity tokens are missing, record the missing span. Do not silently
replace target_entity with all non-special tokens for the primary metric.

## 8. Stage B: SD Attention Smoke

Run after the tokenizer/entity audit is reviewed.

Scale:

```text
Val stratified smoke: 30 samples
tiny / small / large balanced when possible
```

For each selected sample and prompt variant:

```text
1. resize image to 512 x 512 using Experiment 1 / RSVG resize policy
2. run SD inversion / null-text optimization with the original SD attention processors
3. rerun denoising with cross-attention capture wrappers registered
4. aggregate cross-attention at 16 / 32 / 64
5. compute metrics for target_entity, all_entities, and all_non_special maps
6. save compact heatmap PNG/NPY for smoke
```

The capture wrapper must not replace the model's attention computation. It
records the qk-softmax attention map on the side, while the UNet output is
still produced by the original diffusers attention processor.

Numerical rule:

```text
null-text optimization updates the unconditional text embedding in fp32
loss is computed in fp32
non-finite q/k, attention scores, or probabilities are treated as invalid
do not sanitize non-finite attention into a heatmap
```

Debug result:

```text
float32 + null-text optimization: finite, non-uniform heatmap
float16 + no null-text optimization: finite, non-uniform heatmap
float16 + fp16 null-text optimization: invalid; produces non-finite q/k and
near-uniform or NaN heatmaps
```

Prompt variants in smoke:

```text
P1 / P2 / P3 / P4 / C1
```

C0 random / uniform heatmap baseline is computed without running SD.

Server smoke command:

```bash
cd /root/autodl-tmp/DiffusionSkyFind
python foundation_probing/tools/run_exp2_sd_cross_attention_smoke.py \
  --smoke-count 30 \
  --output-dir results/exp_2_sd_cross_attention_smoke \
  --sd-model /root/autodl-tmp/DiffusionSkyFind/stable-diffusion-v1-4 \
  --device cuda \
  --torch-dtype float32 \
  --save-heatmaps
```

Fast fp16 no-null control:

```bash
cd /root/autodl-tmp/DiffusionSkyFind
python foundation_probing/tools/run_exp2_sd_cross_attention_smoke.py \
  --smoke-count 30 \
  --output-dir results/exp_2_sd_cross_attention_smoke_no_null \
  --sd-model /root/autodl-tmp/DiffusionSkyFind/stable-diffusion-v1-4 \
  --device cuda \
  --torch-dtype float16 \
  --null-inner-steps 0 \
  --save-heatmaps
```

## 9. Metrics

Compute metrics on `cross16`, `cross32`, and `cross64`.

Core metrics:

```text
Pointing Game:
  attention peak inside GT_512

Top-k Hit:
  top 1%, 5%, 10% attention cells overlap GT_512

GT Attention Ratio:
  sum attention inside GT_512 / sum attention full map

GT Area Ratio:
  GT grid area / full grid area

Attention Enrichment:
  GT Attention Ratio / GT Area Ratio

Peak Center Distance:
  distance(attention peak, GT center) / GT diagonal

Entropy / Peakness:
  diagnose diffuse attention
```

Heatmap-to-box IoU is diagnostic only. It is not the primary Experiment 2
metric.

## 10. Stage C: Full Val

Run only after smoke identifies the best prompt policies.

Confirmed policy:

```text
full Val uses only the best 1-2 prompt policies
full Val saves metrics by default
full Val does not save compact heatmaps unless explicitly requested
```

## 11. Outputs

Tokenizer/entity audit:

```text
results/exp_2_tokenizer_object_audit/
exp2_val_tokenizer_object_audit.jsonl
exp2_tokenizer_object_audit_summary.json
```

SD attention smoke / full Val:

```text
results/exp_2_sd_cross_attention_smoke/
exp2_val_records.jsonl
exp2_val_skipped.jsonl
exp2_summary.json
```

Optional smoke heatmaps:

```text
results/exp_2_sd_cross_attention_smoke/heatmaps/
```

Do not save full attention tensors by default.

## 12. Confirmed Decisions

```text
1. Object phrase/entity source:
   deterministic LazyMCoT-inspired SkyFind entity extractor first.
   optional VLM/LLM entity extraction must be saved as explicit input later.

2. Extracted phrases:
   extract all objects in the expression, including target and referring objects.
   record target_entity and all_entities separately.

3. VLM localization prompt:
   "Locate it according to the following description. {expression}"

4. Prompt splitting:
   run tokenizer audit first.
   do not add RSVG-style split prompt to the first SD attention smoke.

5. Scale:
   Val-only exploratory experiments.
   smoke first with 30 samples.

6. Controls:
   C0 random / uniform baseline.
   C1 wrong object phrase.

7. Storage:
   smoke saves compact heatmap PNG/NPY.
   full Val saves metrics by default.

8. Full Val:
   run only the best 1-2 prompt policies from smoke.
```
