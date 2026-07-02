# Source Context for Diffusion Foundation Probing

Date: 2026-06-25

This note records the project context used to draft the probing protocol.

The Notion `PaperDesign` content was provided directly in this thread and has
been integrated into `experiment_protocol.md`.

## SkyFind Task Constraint

SkyFind standard samples provide:

```text
image + expression + target bbox
```

They do not provide:

```text
auxiliary object boxes
landmark boxes
target / reference phrase spans
relation pairs
scene graph annotations
```

Therefore, any spatial reasoning module must be built from weak evidence such
as image-text attention maps, generated heatmaps, pseudo regions, or continuous
image-space spatial fields.

## Existing VLM Probing Result

The completed `VLMSkyFind` study shows that direct zero-shot VLM grounding is
not uniformly solved on SkyFind.

Key facts from `VLMSkyFind/docs/full_result_review_2026-06-21.md`:

```text
Qwen2.5-VL-7B average IoU@0.5 / IoU@mean:
  41.44 / 30.39

DeepSeek-VL-7B average:
  0.48 / 0.15

InternVL2.5-8B average:
  0.81 / 0.29

LLaVA-OneVision-7B average:
  0.64 / 0.20
```

Qwen is the only strong direct-box VLM baseline, but it still drops on:

```text
tiny targets
long expressions
ordinal expressions
scale / shape precision
```

This motivates testing whether diffusion / attention-derived signals can
provide additional weak spatial evidence rather than assuming VLM boxes are the
only valid coarse signal.

## RSVG-ZeroOV Reference

The local `BioLoc/reference/RSVG-ZeroOV` code uses:

```text
Qwen2.5-VL attention for VLM image-token attention
Stable Diffusion v1.4
DDIM inversion / null-text optimization
UNet cross-attention maps at 16 / 32 / 64
UNet self-attention maps at 16 / 32 / 64
cross/self attention fusion
optional SAM refinement
```

For SkyFind, the important risk is not only domain mismatch but also target
scale. When the whole image is resized to 512, many UAV targets become too
small for the 16x16 or 32x32 attention grids.

## Spatial Prior Motivation

The intended "attention-derived spatial prior" is not an external scene graph.
It means:

```text
image + expression
-> target/reference/spatial-word attention maps
-> weak continuous spatial fields or constraints
-> candidate box scoring / filtering / evolution
```

This belongs after the foundation probing stage. The probing first asks whether
VLM and diffusion attention maps contain usable localization, structure, and
relation cues.

## Accepted Probing Experiments

The current accepted diffusion probing list is:

```text
1. SD tiny-target retention at the default 512x512 setting.
2. SD cross-attention response near GT under several prompt variants.
3. SD self-attention object-structure probing using GT center as diagnostic
   oracle only.
4. Full image vs crop image to separate resolution bottleneck from domain /
   semantic mismatch.
```

Spatial-prior construction is deferred until these basic capabilities are
measured.
