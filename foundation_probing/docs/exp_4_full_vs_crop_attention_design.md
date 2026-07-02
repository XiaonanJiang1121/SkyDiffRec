# Experiment 4 Design: Full Image vs Crop Image

Date: 2026-07-02

Status: implemented for Val-502 probing with full-expression prompting.

Server project path:

```text
/root/autodl-tmp/DiffusionSkyFind
```

## 1. Goal

Experiment 4 tests whether SD attention failures on SkyFind are caused mainly
by the full-image 512 resize scale bottleneck or by remote-sensing / target
semantic mismatch.

Core question:

```text
If the target is enlarged by a GT-centered crop, does SD cross-attention become
more target-aligned than under full-image 512 processing?
```

This is an oracle diagnostic. The GT crop is not an inference-time method.

## 2. Prompt Policy

Use the original full SkyFind expression:

```text
{expression}
```

Reason:

```text
Experiment 2 full Val is still running. Until prompt results are finalized,
full expression preserves target, reference object, and spatial-relation
context better than object-only prompts.
```

## 3. Image Contexts

For each Val sample, compare:

```text
full_image:
  original full image resized directly to 512 x 512

gt_crop:
  GT-centered crop resized to 512 x 512

random_wrong_crop:
  same-size crop sampled away from the GT box, resized to 512 x 512
```

Default GT crop:

```text
square crop
side length = max(GT width, GT height) * 2.0
shifted back inside image boundary if needed
```

The target box is remapped into each 512 context. For `random_wrong_crop`, the
same relative target box from the GT crop is used as a pseudo evaluation box.
This is a control for prompt/location bias; `target_present=false` is recorded.

## 4. Reference Implementation

Use the same SD attention path as Experiment 2 for cross-attention and the
same row-only probe as Experiment 3 for self-attention:

```text
Stable Diffusion v1.4
DDIM inversion / null-text optimization
direct 512 x 512 processing per context
cross-attention aggregation over down/up blocks at 16 / 32 / 64
target_entity / all_entities / all_non_special token maps
self-attention GT-center row probe at 16 / 32 / 64 when --probe-types self is used
```

Code:

```text
foundation_probing/tools/run_exp4_full_vs_crop_attention.py
```

## 5. Metrics

Attention metrics reuse Experiment 2:

```text
Pointing Game
Top-k Hit at 1%, 5%, 10%
GT Attention Ratio
GT Area Ratio
Attention Enrichment
Peak Center Distance
Entropy / Peakness
```

Simple reconstruction diagnostics are also recorded per image context:

```text
reconstruction_full_mse
reconstruction_full_psnr
reconstruction_target_mse
reconstruction_target_psnr
```

These are lightweight diagnostics only. The primary Experiment 4 signal is the
change in target-entity cross-attention and GT-center self-attention between
`full_image` and `gt_crop`.

## 6. Interpretation

Positive scale-bottleneck signal:

```text
gt_crop target_entity attention enrichment > full_image target_entity
gt_crop peak distance < full_image peak distance
random_wrong_crop does not show the same gain
```

If `gt_crop` remains poor:

```text
SD v1.4 cross-attention is likely not sufficient for SkyFind target/domain
structure, even when target scale is improved by oracle cropping.
```

## 7. Outputs

Suggested output directory:

```text
results/exp_4_full_vs_crop_attention_val_full/
```

Files:

```text
exp4_val_records.jsonl
exp4_val_skipped.jsonl
exp4_summary.json
```

Optional compact heatmaps:

```text
results/exp_4_full_vs_crop_attention_val_full/heatmaps/
```

Do not save heatmaps for full Val unless manually inspecting a smoke run.

## 8. Full Val Command

```bash
cd /root/autodl-tmp/DiffusionSkyFind
python foundation_probing/tools/run_exp4_full_vs_crop_attention.py \
  --smoke-count 502 \
  --output-dir results/exp_4_full_vs_crop_attention_val_full \
  --sd-model /root/autodl-tmp/DiffusionSkyFind/stable-diffusion-v1-4 \
  --device cuda \
  --torch-dtype float16 \
  --context-types full_image gt_crop random_wrong_crop \
  --probe-types cross self \
  --self-control-types gt_center \
  --crop-context-scale 2.0 \
  --resolutions 16 32 64
```

For a visual smoke run:

```bash
python foundation_probing/tools/run_exp4_full_vs_crop_attention.py \
  --smoke-count 3 \
  --output-dir results/exp_4_full_vs_crop_attention_smoke \
  --sd-model /root/autodl-tmp/DiffusionSkyFind/stable-diffusion-v1-4 \
  --device cuda \
  --torch-dtype float16 \
  --context-types full_image gt_crop random_wrong_crop \
  --probe-types cross self \
  --save-heatmaps
```
