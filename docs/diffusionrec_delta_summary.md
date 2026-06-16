# DiffusionREC Delta Summary for SkyFind Experiment 0

Date: 2026-06-16

This note summarizes where `SkyDiffRec` currently differs from the copied
reference `DiffusionREC` baseline.

Reference source:

```text
BioLoc/reference/DiffusionREC/diffusion_REC
```

Working baseline copy:

```text
DiffusionSkyFind/baselines/diffusionrec_original
```

## 1. Intentional Behavior Changes

### 1.1 New `skyfind` dataset branch

Files:

```text
baselines/diffusionrec_original/datasets/data_loader.py
```

Added:

```text
dataset == "skyfind"
direct JSON loading from Train/Val/Test
raw xyxy bbox reading
img_id = within-split index
file_name/raw_img_id metadata
```

### 1.2 Eager image and bbox validation

Files:

```text
baselines/diffusionrec_original/datasets/data_loader.py
tools/inspect_skyfind_dataset.py
```

Added:

```text
image existence check
image openability check
bbox clamp and validity check
split-level statistics for kept/skipped/clamped samples
```

### 1.3 Prompt token truncation alignment

Files:

```text
baselines/diffusionrec_original/datasets/data_loader.py
```

Changed:

```text
tokenizer.batch_encode_plus now uses:
  padding="max_length"
  truncation=True
  max_length=max_query_len
```

Purpose:

```text
prevent long SkyFind expressions from triggering assertion failures after
prompt wrapping
```

### 1.4 Strict parser mode

Files:

```text
baselines/diffusionrec_original/datasets/data_loader.py
tools/audit_skyfind_parser.py
docs/diffusionrec_skyfind_dataset_adaptation_spec.md
```

Changed:

```text
no parser fallback in the training path
parser failures must surface explicitly
word_selection remains parser-conditioned
```

This preserves the original DiffusionREC capability boundary more faithfully
than the earlier fallback version.

### 1.5 SkyFind argument guards

Files:

```text
baselines/diffusionrec_original/train.py
baselines/diffusionrec_original/eval.py
```

Added:

```text
SkyFind requires imsize == 640
SkyFind raises/overrides if max_query_len < 128
```

Purpose:

```text
avoid accidental non-comparable runs caused by mismatched size or too-short text
length
```

## 2. New Audit Utilities

Added:

```text
tools/inspect_skyfind_dataset.py
tools/audit_skyfind_parser.py
```

Roles:

```text
inspect_skyfind_dataset.py:
  raw image/bbox audit
  optional dataset instance inspection with tokenizer/model path

audit_skyfind_parser.py:
  prompt tokenization audit
  parser failure audit
  parser mask length vs token length audit
```

## 3. Documentation Added

Added:

```text
README.md
diffusionrec_skyfind_migration.md
docs/diffusionrec_skyfind_dataset_adaptation_spec.md
docs/diffusionrec_delta_summary.md
```

These documents define the Experiment 0 scope and keep the implementation
boundary explicit.

## 4. Repository/Tracking Changes

Added:

```text
.gitignore
```

Corrected:

```text
the old broad datasets/ ignore rule that accidentally hid baseline dataset
source directories from git tracking
```

This is a repository hygiene fix, not a model behavior change.

## 5. Non-Changes Preserved

The following are intentionally still original DiffusionREC behavior:

```text
model architecture
loss functions
sampling logic
two-path text conditioning design
strict dependence on sng_parser-derived word_selection
validation metric style
640-based internal coordinate assumption
```

## 6. Known Constraints

These are still not solved by design in Experiment 0:

```text
val metric logic still effectively assumes batch size 1
server-side BERT checkpoint is still required
server-side sng_parser environment still required
detectron2/mmcv environment compatibility still matters
```
