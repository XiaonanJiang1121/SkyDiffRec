# PaperDesign Sync

Date: 2026-06-25

This document summarizes the user-provided Notion `PaperDesign` content that
defines the accepted diffusion foundation probing plan.

## Motivation

Explore VLM, Stable Diffusion, and diffusion-style box generation for SkyFind
to provide training-free or light-training weak spatial grounding signals.

The intended problems are:

```text
UAV tiny targets
long spatial expressions
similar-instance confusion
rough VLM boxes
missing object-level spatial annotations
```

## Problem

SkyFind provides only:

```text
image + expression + target bbox
```

It does not provide:

```text
auxiliary object annotations
masks
scene graphs
reference-object labels
```

This is especially limiting because many expressions contain spatial relations
or implicit reference objects.

## VLM Findings

The completed VLM exploration used:

```text
Locate it according to the following description.  {referring expression} The
output format should be like [x1, y1, x2, y2] without any other text.
```

Data:

```text
Val: 5000 samples
Test: 16546 samples
```

Summary:

```text
Qwen2.5-VL-7B has meaningful zero-shot grounding ability and is close to or
above some SkyFind-reported results on average.

DeepSeek, InternVL2.5, and LLaVA-OneVision largely fail on tiny objects and
often output boxes that are too large or shifted.
```

Open concern:

```text
Post-processing was reviewed, but model-specific output protocols should remain
auditable because poor boxes may partly reflect coordinate or protocol issues.
```

## Diffusion Motivation

The diffusion probing asks:

```text
Can Stable Diffusion, as a frozen image-generation foundation model, provide
target-related or spatial signals for SkyFind UAV tiny targets, long
expressions, and remote-sensing imagery?
```

Specifically:

```text
Does real-image denoising / inversion preserve target structure?
Do target-related tokens produce useful cross-attention?
Does self-attention contain object extent information?
```

## Accepted Diffusion Probing List

Use 10% of Val and 10% of Test for formal probing, after smoke tests.

### Experiment 1: Tiny-Target Retention

Test whether SD v1.4 can still see SkyFind tiny targets under the RSVG-ZeroOV
default 512x512 setting.

Record:

```text
GT bbox width / height after resize to 512
target resized area
reconstruction image
full-image and target-crop reconstruction quality
```

Metrics:

```text
target_resized_width / target_resized_height / target_resized_area
target-crop MSE
PSNR / SSIM / LPIPS if practical
error split around very small targets, such as <4x4 pixels at 512
```

Keep 512x512 first because dynamic resolution can introduce OOM, inversion
instability, and VLM/DM feature-alignment issues.

### Experiment 2: Cross-Attention GT Response

Test whether SD cross-attention responds near the GT target.

Prompt variants:

```text
1. SkyFind expression only
2. extracted object phrases
3. "a remote sensing image of {object phrase}"
4. the same VLM / RSVG-style localization prompt
```

The tokenizer limit and SkyFind long-expression length are central diagnostics.

### Experiment 3: Self-Attention Object Structure

Test whether SD self-attention has object-structure ability on SkyFind.

Use GT center only as an oracle diagnostic, not as a method input:

```text
take the self-attention map for the GT-center cell
test whether it covers the same target
```

Metrics:

```text
self-attention region bbox area ratio
centered scale-shape failure
IoU with GT
non-target / background energy ratio
```

### Experiment 4: Full Image vs Crop Image

Test whether SD failure is caused by tiny-target resolution or by remote
sensing domain / semantic mismatch.

Compare:

```text
full image -> 512
GT crop -> 512
random wrong crop -> 512
```

Optional later comparison:

```text
VLM crop -> 512
diffusion / attention proposed crop -> 512
```

Interpretation:

```text
full poor + GT crop good:
  SD is scale-limited and may still be useful with a crop/seed mechanism.

GT crop poor:
  SD v1.4 is likely not well adapted to SkyFind target/domain structure.
```

## Deferred Stage

After the four foundation experiments, design the attention-derived spatial
prior:

```text
VLM cross-attention + DM self-attention
-> weak spatial prior
-> substitute for missing object-level SpatialInfo
```

This is where relation words, implicit reference objects, and similar-instance
disambiguation should be addressed.
