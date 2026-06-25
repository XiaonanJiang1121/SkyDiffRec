# SkyFind Diffusion Foundation Probing

Date: 2026-06-25

Status: synced with the user-provided Notion `PaperDesign` content in this
thread.

## Purpose

This folder manages the training-free diffusion foundation capability probing
for SkyFind.

The central question is:

```text
Can frozen VLM / Stable Diffusion / diffusion-style attention signals provide
weak spatial grounding evidence for SkyFind?
```

This probing stage does not pre-assign roles such as "VLM is coarse" or
"diffusion is refinement". Instead, it tests which source can provide which
kind of signal:

```text
VLM direct box / attention
SD cross-attention
SD self-attention
VLM + SD attention fusion
attention-derived spatial prior candidates
```

The later method design should be decided from the probing results.

## Current Source Sync

Accessible and reviewed:

```text
DiffusionSkyFind/foundation_probing/docs/paper_design_sync.md
VLMSkyFind/docs/full_result_review_2026-06-21.md
A_Improvement_ReasonLoc/bioloc_bio_goal.md
BioLoc/reference/RSVG-ZeroOV
```

Synced from user-provided Notion `PaperDesign` content:

```text
Problem:
  tiny UAV targets, similar instances, long spatial expressions, no auxiliary
  object boxes/masks/scene graph, implicit reference objects without labels

Goal:
  two-stage coarse-to-fine architecture, with VLM/DM roles decided by probing;
  later attention-derived spatial prior to replace missing object-level
  SpatialInfo

Accepted DM probing list:
  1. SD tiny-target retention at 512
  2. SD cross-attention GT response
  3. SD self-attention object-structure ability
  4. full image vs crop image
```

## Directory Layout

```text
foundation_probing/
  README.md
  docs/
    experiment_protocol.md
    paper_design_sync.md
    source_context.md
  configs/
    README.md
  reports/
    README.md
  results/
    README.md
```

## Execution Rule

Run experiments one group at a time. Each group should produce:

```text
1. config / command record
2. JSONL or JSON metrics
3. qualitative visualizations when attention maps are involved
4. short report section under reports/
5. go / no-go decision for the next group
```

Do not mix these probing results with unrelated or abandoned training baselines.
