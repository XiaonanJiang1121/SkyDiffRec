# Experiment 1: SD Tiny-Target Retention

Date: 2026-06-25

Status: scale audit complete locally; SD reconstruction pending server setup.

## Goal

Measure whether SkyFind targets remain visible under the RSVG-ZeroOV /
Stable-Diffusion-style `512 x 512` square resize before evaluating SD
inversion, cross-attention, or self-attention.

## Code

```text
foundation_probing/tools/run_exp1_tiny_target_retention.py
```

The resize policy follows RSVG-ZeroOV's `load_512()` behavior:

```text
direct square resize to 512 x 512
no aspect-ratio preservation
```

BBox policy:

```text
preserve raw annotation bbox
clip a second bbox to image boundaries for visible-region analysis
record bbox_was_clipped
use clipped bbox for 512 mapping and target crop metrics
```

Missing or bad images are skipped and recorded in `*_skipped.jsonl`.

## Local Scale Audit Command

```bash
python foundation_probing/tools/run_exp1_tiny_target_retention.py \
  --output-dir results/exp_1_sd_tiny_target_retention
```

Input subset:

```text
data/foundation_probe_10pct/annotations/Val_10pct.json
data/foundation_probe_10pct/annotations/Test_10pct.json
```

Output:

```text
results/exp_1_sd_tiny_target_retention/exp1_val_records.jsonl
results/exp_1_sd_tiny_target_retention/exp1_test_records.jsonl
results/exp_1_sd_tiny_target_retention/exp1_val_skipped.jsonl
results/exp_1_sd_tiny_target_retention/exp1_test_skipped.jsonl
results/exp_1_sd_tiny_target_retention/exp1_summary.json
```

The result directory is intentionally ignored by git.

## Scale Audit Summary

Processed:

```text
processed_count: 2155
skipped_count: 2
skipped reason: missing_or_bad_image
```

Skipped image records:

```text
SeaDronesSee_36.jpg: OSError, image file is truncated
```

Overall 512 target scale:

```text
median min-side at 512: 10.40 px
p10 min-side at 512: 4.80 px
min min-side at 512: 2.20 px
target min-side < 4 px: 3.29%
```

Attention-grid pressure:

```text
min-side < 1 cell at 16x16: 91.28%
min-side < 1 cell at 32x32: 68.96%
min-side < 1 cell at 64x64: 36.66%
```

By split:

```text
Val:
  processed: 502
  median min-side at 512: 16.91 px
  min-side < 1 cell at 32x32: 46.61%
  min-side < 1 cell at 64x64: 20.12%

Test:
  processed: 1653
  median min-side at 512: 9.07 px
  min-side < 1 cell at 32x32: 75.74%
  min-side < 1 cell at 64x64: 41.68%
```

Interpretation:

```text
Full-image SD attention at 16x16 and 32x32 is heavily scale-limited on the
10% probing subset, especially on Test. Later SD attention failures must be
reported together with target cell-span statistics.
```

## Server Reconstruction Command

After configuring GPU dependencies and SD v1.4 on the server:

```bash
cd /root/autodl-tmp/DiffusionSkyFind
python foundation_probing/tools/run_exp1_tiny_target_retention.py \
  --splits val test \
  --output-dir results/exp_1_sd_tiny_target_retention_reconstruction \
  --run-reconstruction \
  --sd-model CompVis/stable-diffusion-v1-4 \
  --device cuda \
  --torch-dtype float16 \
  --num-ddim-steps 20 \
  --guidance-scale 7.5 \
  --null-inner-steps 10 \
  --early-stop-epsilon 1e-5 \
  --save-reconstructions
```

Recorded reconstruction metrics:

```text
full_mse / full_psnr / full_ssim / full_lpips
target_mse / target_psnr / target_ssim / target_lpips
```

Notes:

```text
target_ssim is null when the target crop is smaller than the minimum valid
SSIM window.
target_lpips is null when the target crop is smaller than --min-lpips-crop-size.
full-image PSNR / SSIM / LPIPS are always recorded during reconstruction.
```
