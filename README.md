# DiffusionSkyFind

Date: 2026-06-16

This folder records the diffusion-native SkyFind research direction. It is kept
separate from `BioLoc/` so that diffusion experiments do not become coupled to
the AerialVG-style BioLoc baseline.

## Research Position

SkyFind is a UAV-view referring expression localization task:

```text
image + expression -> target bbox
```

The diffusion direction should therefore focus on perception-space denoising:

```text
noisy boxes / noisy spatial fields
-> text-conditioned visual denoising
-> final referred target bbox
```

`DiffusionUavLoc` is not used as a main method reference because it solves
cross-view UAV-satellite retrieval/geolocalization rather than referring bbox
localization. It may only provide weak background inspiration about fixed-step
diffusion representations or text-free conditioning.

## Experiment Route

### Experiment 0: DiffusionREC Original on SkyFind

Goal:

```text
Run the original DiffusionREC-style method on SkyFind with only the necessary
dataset adaptation.
```

Rules:

1. Keep the DiffusionREC model, loss, training loop, and sampling behavior as
   close to the reference implementation as possible.
2. Add only the minimal SkyFind dataset branch needed to read SkyFind images,
   expressions, and boxes.
3. Do not add long-expression modules.
4. Do not add spatial reasoning modules.
5. Do not add aerial-specific priors.
6. Record metrics and failure cases as the baseline for later method design.

Primary metrics:

```text
Acc@0.5
mIoU
```

Additional diagnostics:

```text
Acc@0.7
small-object subset Acc@0.5 / Acc@0.7
expression length buckets
qualitative failure cases
training speed and memory
```

Detailed migration notes:

```text
DiffusionSkyFind/diffusionrec_skyfind_migration.md
DiffusionSkyFind/docs/diffusionrec_skyfind_dataset_adaptation_spec.md
```

### Experiment 1: Long Expression Handling

Motivation:

SkyFind expressions are longer and more spatially descriptive than typical
RefCOCO-style phrases. A direct short-phrase text encoder may truncate or dilute
important target, attribute, and relation words.

Candidate directions:

```text
hierarchical text encoding
target / attribute / relation token selection
long-context text encoder
sentence decomposition before diffusion conditioning
```

This experiment should start only after Experiment 0 produces a reliable
baseline.

### Experiment 2: Spatial Reasoning Enhancement

Motivation:

SkyFind UAV images contain small targets, repeated objects, and strong spatial
language. Existing diffusion REC methods denoise boxes but do not explicitly
model aerial spatial reasoning.

Candidate directions:

```text
coordinate-aware box denoising
expression-conditioned spatial prior
map-guided box renewal
landmark/relation-aware text supervision
Box + Spatial Prior Joint Diffusion
```

`Box + Spatial Prior Joint Diffusion` is currently remembered as a strong
future direction, not part of Experiment 0.

### Experiment 3: Few-Step / One-Step Box Diffusion

Motivation:

Multi-step diffusion may be too slow for practical localization. Once a strong
multi-step SkyFind diffusion model exists, compress it into fewer denoising
steps.

Potential references:

```text
Consistency Models
Rectified Flow / Flow Matching
Shortcut Models
Distribution Matching Distillation
```

The one-step direction should be explored after there is a trained SkyFind
diffusion teacher or at least a validated few-step baseline.

## Code Boundary

External reference code stays under:

```text
BioLoc/reference/DiffusionREC
BioLoc/reference/DiffusionDet
BioLoc/reference/RSVG-ZeroOV
```

`BioLoc/reference/DiffusionREC` is treated as an external reference only.
Experiment code lives under:

```text
DiffusionSkyFind/baselines/diffusionrec_original
```

New diffusion experiment code should live under:

```text
DiffusionSkyFind/
  configs/
  datasets/
  models/
  training/
  evaluation/
  tools/
```

Avoid writing new SkyFind diffusion code inside `BioLoc/` unless it is a small
utility explicitly shared by both projects.
