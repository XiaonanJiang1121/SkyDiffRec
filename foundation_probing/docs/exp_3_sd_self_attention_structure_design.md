# Experiment 3 Design: SD Self-Attention Object Structure

Date: 2026-07-02

Status: implemented for Val-502 probing with full-expression prompting.

Server project path:

```text
/root/autodl-tmp/DiffusionSkyFind
```

## 1. Goal

Experiment 3 tests whether frozen Stable Diffusion v1.4 self-attention
contains object-structure or object-extent information for SkyFind targets.

This is an oracle diagnostic, not a deployable method:

```text
Use the GT center only to probe one self-attention query row.
Do not treat GT center as available at inference time.
```

Core question:

```text
If we seed SD self-attention from the target center cell, does the attention map
stay on the same object, or does it spread into background / similar instances?
```

The current implementation uses the original full SkyFind expression. This
matches the July 2 decision to continue Experiment 3/4 while Experiment 2 full
Val is still running, using full expression as the shared prompt policy.

## 2. Reference Implementation

Use RSVG-ZeroOV as the first reference.

Relevant local files:

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
from_where = ["down", "up"]
is_cross = False for self-attention
```

Important implementation change:

```text
Do not save per-layer / per-head / per-step raw self-attention tensors.
At 64 x 64, a full self-attention map is 4096 x 4096 per head/layer/step and
is too large for full-subset probing.
```

Debug matrix policy:

```text
By default, the script only saves row-derived compact heatmaps and metrics.
For debugging, use --save-self-attention-matrices to save the averaged full
self-attention matrix for selected resolutions.

Recommended first debug setting:
  --save-self-attention-matrices --self-matrix-resolutions 64

This saves one averaged self64 matrix per sample, shaped [4096, 4096]. It is
large but useful for checking where the self-attention structure fails. It does
not save per-layer / per-head / per-step raw matrices.
```

Instead, implement a row-only self-attention probe:

```text
During each self-attention forward pass, keep only the attention row(s) for the
oracle seed cell(s), average over heads/layers/steps, and write a compact
heatmap of shape [res, res].
```

This follows the RSVG attention-store idea while avoiding unnecessary memory
pressure.

## 3. Inputs

Use the same 10% foundation probing subset:

```text
data/foundation_probe_10pct/annotations/Val_10pct.json
data/foundation_probe_10pct/annotations/Test_10pct.json
data/foundation_probe_10pct/images/
```

Initial run:

```text
Val only.
```

Supported smoke subset:

```text
set --smoke-count for a stratified tiny / small / large sample
```

Full Val:

```text
502 samples from Val_10pct.json
```

Bad / missing images:

```text
skip and record in exp3_*_skipped.jsonl
```

BBox boundary policy:

```text
preserve gt_box_raw
clip gt_box_clipped to image boundary
map clipped bbox to 512 for all metrics
record bbox_was_clipped
```

This matches Experiment 1.

## 4. Prompt Policy

Primary prompt:

```text
{expression}
```

Reason:

```text
Experiment 2 is still running and the best prompt group is not finalized.
Full expression preserves target/reference/spatial context and is the safest
default for Experiment 3/4 until Experiment 2 returns full results.
```

Optional prompt controls:

```text
generic prompt: "a remote sensing image"
empty prompt: ""
```

Do not use multiple prompt variants in the first full Val run.

## 5. Self-Attention Probe

For each sample:

```text
1. resize image to 512 x 512 using the Experiment 1 / RSVG resize policy
2. map clipped GT bbox to 512
3. compute GT center in 512 coordinates
4. run SD inversion / null-text optimization
5. rerun denoising with a row-only self-attention probe
6. aggregate self-attention heatmaps at 16 / 32 / 64 when available
```

Primary seed:

```text
center_cell
```

Mapping:

```text
cell_x = floor(gt_center_x_512 / 512 * res)
cell_y = floor(gt_center_y_512 / 512 * res)
seed_index = cell_y * res + cell_x
```

Clamp seed cell to `[0, res - 1]`.

Optional diagnostic seed:

```text
all_gt_cells
```

For larger targets, average rows from all grid cells whose cell centers fall
inside the GT bbox. If no cell center falls inside GT, use the center_cell.

The first implementation supports `center_cell`, `random_background`, and
`all_gt_cells`. The default run uses `gt_center + random_background`.

## 6. Attention Resolutions

Compute:

```text
self16
self32
self64
```

Primary analysis:

```text
self64
```

Reason:

```text
Experiment 1 showed that 16 x 16 and 32 x 32 are heavily scale-limited for
SkyFind targets. 64 x 64 is still imperfect but is the first resolution where
many Val targets occupy at least one cell.
```

Do not claim 64 x 64 is sufficient before Experiment 3 results. It is only the
most plausible resolution to test first.

## 7. Heatmap Normalization

For each row-derived heatmap:

```text
normalize heatmap so sum(heatmap) = 1
record entropy and peakness
```

If averaging multiple heads/layers/steps:

```text
average raw attention rows first, then normalize once at the end
```

Suggested step aggregation:

```text
primary: average over all DDIM steps
diagnostic: early / middle / late step averages if cheap to record
```

The first implementation can record all-step average only.

## 8. Metrics

Compute every metric per resolution.

### 8.1 Attention Mass Metrics

```text
GT Attention Ratio:
  sum heatmap inside GT grid box / sum heatmap full grid

GT Area Ratio:
  GT grid area / full grid area

Attention Enrichment:
  GT Attention Ratio / GT Area Ratio
```

Attention enrichment is important because raw GT Attention Ratio is naturally
small for tiny boxes.

### 8.2 Peak Metrics

```text
Pointing Game:
  heatmap peak lies inside GT grid box

Peak Center Distance:
  distance(heatmap peak, GT center) / GT diagonal
```

### 8.3 Top-k Metrics

```text
Top-k Hit:
  top 1%, 5%, 10% heatmap cells overlap GT grid box

Top-k IoU:
  IoU between top-k binary region and GT grid box
```

### 8.4 Equal-Area Region Metrics

Use a threshold-free region conversion:

```text
Select the top N heatmap cells, where N equals the number of grid cells covered
by the GT grid box.
```

Then compute:

```text
equal_area_iou_with_gt
equal_area_center_error
equal_area_bbox_area_ratio
```

This avoids tuning a heatmap threshold after seeing the result.

### 8.5 Leakage / Diffusion Metrics

```text
outside_gt_energy_ratio:
  1 - GT Attention Ratio

outside_expanded_gt_energy_ratio:
  attention mass outside a 2x expanded GT box

entropy:
  normalized entropy of the heatmap

peakness:
  max heatmap value / mean heatmap value
```

## 9. Controls

Controls are necessary because the seed is an oracle.

Recommended controls:

```text
C1 GT center seed: primary probe
C2 random background seed: same image, seed outside GT
C3 shifted-neighbor seed: same image, seed near but outside GT
```

Control metrics should use the same GT bbox. A useful self-attention structure
signal should show stronger GT enrichment from C1 than C2/C3.

First run:

```text
gt_center + random_background
```

C3 can be added after the first smoke if C1 looks promising.

## 10. Outputs

Suggested output directory:

```text
results/exp_3_sd_self_attention_structure_val_full/
```

Files:

```text
exp3_val_records.jsonl
exp3_val_skipped.jsonl
exp3_summary.json
```

Optional compact heatmaps:

```text
results/exp_3_sd_self_attention_structure/heatmaps/
```

Do not save full self-attention tensors.

Implementation:

```text
foundation_probing/tools/run_exp3_sd_self_attention_structure.py
```

Full Val command:

```bash
cd /root/autodl-tmp/DiffusionSkyFind
python foundation_probing/tools/run_exp3_sd_self_attention_structure.py \
  --smoke-count 502 \
  --output-dir results/exp_3_sd_self_attention_structure_val_full \
  --sd-model /root/autodl-tmp/DiffusionSkyFind/stable-diffusion-v1-4 \
  --device cuda \
  --torch-dtype float16 \
  --prompt-policy full_expression \
  --control-types gt_center random_background \
  --save-self-attention-matrices \
  --self-matrix-resolutions 64 \
  --resolutions 16 32 64
```

Per-record fields:

```json
{
  "sample_id": "val:0",
  "split": "val",
  "file_name": "...jpg",
  "source": "UAVDT",
  "prompt": "a remote sensing image",
  "seed_policy": "center_cell",
  "control_type": "gt_center",
  "gt_box_raw": [x1, y1, x2, y2],
  "gt_box_clipped": [x1, y1, x2, y2],
  "bbox_512": [x1, y1, x2, y2],
  "target_width_512": 0.0,
  "target_height_512": 0.0,
  "attention_paths": {
    "self16": "...npy",
    "self32": "...npy",
    "self64": "...npy"
  },
  "metrics": {
    "self16": {},
    "self32": {},
    "self64": {}
  }
}
```

## 11. Interpretation

Positive signal:

```text
GT-center self-attention has higher GT enrichment, higher top-k overlap, lower
center error, and lower leakage than random background seeds.
```

If positive:

```text
SD self-attention can be considered a candidate object-structure signal, but it
still requires a non-oracle seed from VLM, SD cross-attention, or another coarse
mechanism.
```

If negative:

```text
SD self-attention under full-image 512 processing is not a reliable structural
prior for SkyFind tiny targets. Experiment 4 should test whether crop-based
scale recovery changes this result.
```

## 12. Current Decisions

Current implementation choices:

```text
1. Prompt: full SkyFind expression.
2. Scale: Val 10% full set, 502 samples.
3. Seed policy: center_cell by default; all_gt_cells available as diagnostic.
4. Controls: GT center + random background seed.
5. Saved artifacts: metrics by default; compact heatmaps with --save-heatmaps;
   averaged full self-attention matrices with --save-self-attention-matrices.
```
