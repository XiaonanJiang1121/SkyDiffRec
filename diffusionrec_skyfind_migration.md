# DiffusionREC on SkyFind: Minimal Migration Plan

Date: 2026-06-16

Status: Experiment 0 design

## 1. Goal

Run a faithful DiffusionREC-style baseline on SkyFind:

```text
SkyFind image + expression
-> DiffusionREC model
-> predicted bbox
```

The first run is not intended to be our final method. It answers:

```text
How well does an existing diffusion REC design work on SkyFind before adding
SkyFind-specific long-expression or spatial-reasoning improvements?
```

## 2. Non-Goals

For this baseline, do not add:

```text
long-expression redesign
spatial prior branch
Box + Spatial Prior Joint Diffusion
RSVG-ZeroOV attention prior
AerialVG relation-aware grounding
one-step distillation
new loss terms beyond what is needed for basic compatibility
```

The purpose is to keep the baseline interpretable. If it improves or fails, we
should know the result comes from DiffusionREC itself rather than from our
later additions.

## 3. Reference Code

Reference repository:

```text
BioLoc/reference/DiffusionREC
```

Main files to inspect/adapt:

```text
BioLoc/reference/DiffusionREC/diffusion_REC/train.py
BioLoc/reference/DiffusionREC/diffusion_REC/engine.py
BioLoc/reference/DiffusionREC/diffusion_REC/datasets/__init__.py
BioLoc/reference/DiffusionREC/diffusion_REC/datasets/data_loader.py
BioLoc/reference/DiffusionREC/diffusion_REC/models/trans_vg.py
BioLoc/reference/DiffusionREC/diffusion_REC/models/language_model/bert.py
BioLoc/reference/DiffusionREC/diffusion_REC/utils/eval_utils.py
```

Recommended implementation rule:

```text
Do not heavily edit the external reference checkout.
Create a DiffusionSkyFind copy/wrapper first, then patch only what is needed.
```

## 4. SkyFind Data Format

SkyFind root:

```text
BioLoc/data/SkyFind_data
```

Files:

```text
Train.json
Train_Aug.json
Val.json
Test.json
images/
```

Native annotation item:

```json
{
  "fileName": "000000.jpg",
  "bbox": [x1, y1, x2, y2],
  "expression": "..."
}
```

For `Train_Aug.json`, the text fields are:

```text
expression_aug1
expression_aug2
```

Existing BioLoc loader behavior:

```text
train split       -> Train.json expression
train_aug split   -> two augmented expressions per image
train_merged      -> randomly sample one of original / aug1 / aug2
val split         -> Val.json expression
test split        -> Test.json expression
```

For a faithful first DiffusionREC baseline, use:

```text
train: Train.json only, or train_merged if we want to match SkyFind official-style training.
val:   Val.json
test:  Test.json
```

The cleanest first run is:

```text
train = Train.json
val   = Val.json
```

Then run a second baseline variant:

```text
train = Train.json + Train_Aug.json sampled as train_merged
val   = Val.json
```

## 5. Minimal Dataset Adaptation

DiffusionREC `TransVGDataset` currently assumes RefCOCO/ReferIt-style cached
data:

```text
split_root/dataset/*.pth
COCO-like image filename parsing
RefCOCO bbox stored as xywh
```

SkyFind needs a new `dataset == "skyfind"` branch.

### 5.1 New Dataset Option

Add `skyfind` to `SUPPORTED_DATASETS`:

```python
"skyfind": {
    "splits": ("train", "val", "test"),
}
```

If using augmented training later, expose either:

```text
split=train_merged
```

or an argument:

```text
--skyfind-use-aug
```

For Experiment 0 first run, keep `train/val/test` only.

### 5.2 Image Root

For SkyFind:

```text
self.dataset_root = args.data_root
self.im_dir = args.data_root / "images"
```

Example CLI:

```text
--dataset skyfind
--data_root /Users/jxxxxn/Desktop/3D_Point/SpatialLLM/Text2Loc/BioLoc/data/SkyFind_data
```

### 5.3 Annotation Loading

Map split to JSON:

```text
train -> Train.json
val   -> Val.json
test  -> Test.json
```

Build `self.images` as tuples compatible with the existing `pull_item` output:

```python
(img_file, None, bbox_xyxy, phrase, None)
```

where:

```text
img_file = sample["fileName"]
bbox_xyxy = sample["bbox"]
phrase = sample["expression"]
```

### 5.4 BBox Format

Important:

```text
SkyFind bbox is already xyxy.
RefCOCO branch converts xywh -> xyxy.
```

Therefore, for SkyFind, skip this logic:

```python
bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
```

The transform pipeline can still resize/pad and update bbox if it expects xyxy.

### 5.5 Image ID

DiffusionREC currently parses COCO filenames:

```python
img_id = int(img_file.split("_")[2].split(".")[0])
```

For SkyFind use:

```python
img_id = int(Path(img_file).stem)
```

Fallback:

```python
img_id = idx
```

if the filename is not numeric.

### 5.6 Text Lowercasing

DiffusionREC lowercases phrases:

```python
phrase = phrase.lower()
```

Keep this for the first baseline to avoid unnecessary changes.

## 6. Query Length Setting

SkyFind expression statistics from local data:

```text
Train words: mean 28.39, p95 44, max 115
Val words:   mean 28.51, p95 45, max 115
Test words:  mean 25.48, p95 41, max 78
```

DiffusionREC RefCOCO scripts use:

```text
max_query_len = 20
```

DiffusionREC parser default:

```text
max_query_len = 50
```

For SkyFind baseline:

```text
max_query_len = 128
```

If tokenized prompt length still overflows, move to:

```text
max_query_len = 192
```

Do not design a new long-expression module in Experiment 0. Just avoid obvious
truncation.

## 7. Training Command Shape

Initial single-GPU smoke run:

```bash
cd DiffusionSkyFind
python training/train_diffusionrec_skyfind.py \
  --dataset skyfind \
  --data_root ../BioLoc/data/SkyFind_data \
  --split_root ./data \
  --batch_size 1 \
  --num_workers 2 \
  --imsize 640 \
  --max_query_len 128 \
  --output_dir ./outputs/diffusionrec_skyfind_smoke
```

The exact script name may change depending on whether we copy the original
DiffusionREC entry or wrap it.

Full training should be decided after the smoke run reports:

```text
forward works
loss finite
validation metric works
GPU memory usage
iteration time
```

## 8. Verification Checklist

Before long training:

1. Dataset item loads an image, expression, and xyxy bbox.
2. Transform output bbox remains valid after resize/pad.
3. Tokenized SkyFind expression fits `max_query_len`.
4. One batch passes through model forward.
5. Training loss is finite for at least 20 iterations.
6. Validation computes mIoU and Acc@0.5.
7. Predicted boxes can be visualized on several validation images.

## 9. Metrics and Diagnostics

Minimum metrics:

```text
mIoU
Acc@0.5
```

Add for SkyFind analysis:

```text
Acc@0.7
small-object subset metrics
medium/large-object subset metrics
expression length buckets
```

Suggested length buckets:

```text
<= 20 words
21-40 words
41-60 words
> 60 words
```

Suggested object area buckets:

```text
small:  bbox area ratio < 0.001
medium: 0.001 <= bbox area ratio < 0.01
large:  bbox area ratio >= 0.01
```

These diagnostics are analysis only. They should not alter the Experiment 0
model.

## 10. Expected Baseline Limitations

This baseline may underperform because:

1. SkyFind expressions are much longer than RefCOCO phrases.
2. UAV targets are often tiny and visually ambiguous.
3. Spatial relation language is more important than in common REC datasets.
4. DiffusionREC does not explicitly model aerial spatial priors.
5. The reference implementation has several hardcoded assumptions inherited
   from RefCOCO/TransVG-style pipelines.

These limitations are the reason for later experiments:

```text
Experiment 1: long expression handling
Experiment 2: spatial reasoning enhancement
Experiment 3: few-step / one-step diffusion
```
