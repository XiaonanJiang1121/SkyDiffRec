# DiffusionREC Original on SkyFind: Dataset Adaptation Development Spec

Date: 2026-06-16

Status: Ready for implementation review

Owner: SkyDiffRec Experiment 0

## 1. Executive Summary

This document specifies the minimum code changes required to run the original
DiffusionREC baseline on SkyFind.

Experiment 0 must answer one narrow question:

```text
How does the existing DiffusionREC design perform on SkyFind when we only adapt
the dataset input contract?
```

Therefore, the implementation must not introduce:

```text
new long-expression architecture
new spatial reasoning module
new aerial prior
new loss term
new sampling algorithm
new image resolution setting
```

The baseline code lives in:

```text
DiffusionSkyFind/baselines/diffusionrec_original
```

The reference copy was copied from:

```text
BioLoc/reference/DiffusionREC/diffusion_REC
```

excluding generated files such as `__pycache__`, `.pyc`, `.DS_Store`, and
`outputs/`.

## 2. Version Control Boundary

Use the new GitHub repository for the diffusion project:

```text
git@github.com:XiaonanJiang1121/SkyDiffRec.git
```

Recommended local repository boundary:

```text
DiffusionSkyFind/
```

Do not push the entire `Text2Loc/` workspace to `SkyDiffRec`, because that root
contains ReasonLoc, BioLoc, rebuttal files, reference repositories, and many
unrelated changes. `SkyDiffRec` should track only the diffusion SkyFind project.

## 3. Source Code Facts Confirmed From DiffusionREC

### 3.1 Dataset Return Contract

`TransVGDataset.__getitem__` currently returns this tuple:

```python
(
    img,              # Tensor, [3, imsize, imsize]
    img_id,           # int-compatible
    img_mask,         # [imsize, imsize], 0 valid / 1 padded
    word_id,          # token ids, length max_query_len
    word_selection,   # selected word mask, length max_query_len
    batch_a,          # tokenizer output dict for prompt sentence
    word_mask,        # token attention mask, length max_query_len
    bbox,             # normalized cxcywh, divided by imsize
)
```

`utils.misc.collate_fn` and `collate_fn_val` hard-code this tuple shape.

Implication:

```text
Any metadata such as file_name/raw_img_id/orig_size can pass through transforms,
but it is currently dropped before collation.
```

For training and validation metrics this is acceptable. For prediction export
or visualization, we must add a separate metadata path.

### 3.2 Pre-Transform Input Contract

The transform pipeline expects:

```python
{
    "img": PIL.Image,
    "box": torch.FloatTensor([x1, y1, x2, y2]),  # raw pixel xyxy
    "text": str,
    "img_id": int,
}
```

SkyFind must provide raw pixel `xyxy` boxes. It must not convert boxes to
`xywh` before transform.

### 3.3 Transform Contract

For validation/test, `datasets/__init__.py` uses:

```python
T.RandomResize([imsize])
T.ToTensor()
T.NormalizeAndPad(size=imsize)
```

`RandomResize([imsize])` keeps aspect ratio and resizes by the long side:

```text
scale = imsize / max(H_raw, W_raw)
```

`NormalizeAndPad` then center-pads the resized image into `imsize x imsize` when
`aug_translate=False`.

It converts bbox from transformed raw `xyxy` to normalized `cxcywh`:

```python
box = xyxy2xywh(box)
box = box / torch.tensor([W_out, H_out, W_out, H_out])
```

with:

```text
W_out = H_out = imsize
```

### 3.4 Metric Coordinate System

DiffusionREC validation computes IoU in normalized transformed coordinates:

```python
pred cxcywh normalized -> xyxy normalized
gt cxcywh normalized   -> xyxy normalized
IoU
```

Because validation/test use the same aspect-ratio-preserving scale and padding
for both prediction and GT, IoU in transformed coordinates is equivalent to IoU
in original image coordinates.

However, any exported prediction file must recover raw original coordinates:

```text
pred_bbox_xyxy_raw
gt_bbox_xyxy_raw
```

### 3.5 Fixed Image Size

The model contains a hardcoded 640 coordinate assumption:

```python
images_whwh = torch.ones((bs, 4)).to(device) * 640
```

Therefore Experiment 0 must use:

```text
--imsize 640
```

Changing `imsize` is not a dataset adaptation. It is a later controlled ablation.

### 3.6 Query Length Dependencies

`max_query_len` is used in at least three places:

```text
1. Dataset token padding length.
2. Vision-language positional embedding size:
   num_total = visual_tokens + max_query_len + 1
3. BERT text inputs through `batch_a` / `text_info_data`.
```

The current tokenizer call does not truncate:

```python
batch_a = tokenizer.batch_encode_plus([prompt_tokens_a], return_tensors="pt", padding=True)
```

and then asserts:

```python
assert len(input_ids) == seq_length
```

For SkyFind, this can fail when token length exceeds `max_query_len`.

## 4. SkyFind Data Facts Confirmed Locally

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

Native annotation:

```json
{
  "fileName": "...jpg",
  "bbox": [x1, y1, x2, y2],
  "expression": "..."
}
```

`Train_Aug.json` contains:

```text
expression_aug1
expression_aug2
```

Experiment 0 result priority:

```text
E0-A primary result:
  Train.json -> Val.json

E0-B supplementary result:
  Train.json + Train_Aug.json -> Val.json
```

`E0-A` is the cleanest original DiffusionREC-on-SkyFind baseline. `E0-B` is
still considered dataset adaptation, but it changes the language distribution
through augmented expressions and should therefore be reported as a separate
supplementary baseline.

Local data audit:

```text
Train.json:
  samples: 331364
  non-numeric filenames: 268675
  corrupt images: 0
  invalid raw boxes: 0
  out-of-bounds boxes: 198

Train_Aug.json:
  samples: 331364
  non-numeric filenames: 268675
  corrupt images: 0
  invalid raw boxes: 0
  out-of-bounds boxes: 198

Val.json:
  samples: 5000
  non-numeric filenames: 4054
  corrupt images: 0
  invalid raw boxes: 0
  out-of-bounds boxes: 3

Test.json:
  samples: 16546
  non-numeric filenames: 16546
  corrupt image entries: 4
  invalid raw boxes: 0
  out-of-bounds boxes: 1
```

Important consequence:

```text
img_id cannot be parsed with int(Path(fileName).stem).
```

Use a contiguous integer index for DiffusionREC compatibility and store the raw
file identity separately.

## 5. Required Implementation

### 5.1 Add SkyFind Dataset Branch

File:

```text
DiffusionSkyFind/baselines/diffusionrec_original/datasets/data_loader.py
```

Add:

```python
"skyfind": {
    "splits": ("train", "val", "test"),
}
```

to `SUPPORTED_DATASETS`.

For `dataset == "skyfind"`:

```text
self.dataset_root = self.data_root
self.im_dir = os.path.join(self.dataset_root, "images")
```

Bypass the original `.pth` cache requirement:

```text
Do not require split_root/skyfind/*.pth.
Do not call torch.load(imgset_path) for SkyFind.
```

Load JSON directly:

```text
train -> Train.json
val   -> Val.json
test  -> Test.json
```

### 5.2 SkyFind Sample Record

Internal `self.images` entries should be dictionaries or tuples with these
fields:

```python
{
    "img_id": idx,
    "file_name": sample["fileName"],
    "raw_img_id": Path(sample["fileName"]).stem,
    "bbox_xyxy_raw": [x1, y1, x2, y2],
    "expression": sample["expression"],
    "orig_size": (H, W),
}
```

For maximum compatibility with the old `pull_item`, either:

1. make `pull_item` branch on `self.dataset == "skyfind"`, or
2. store tuples and branch only for image id / bbox format.

Preferred:

```python
if self.dataset == "skyfind":
    return self.pull_skyfind_item(idx)
```

### 5.3 ID Mapping Contract

DiffusionREC-facing id:

```python
img_id = idx
```

`img_id` is a within-split index:

```text
train: 0 ... len(train)-1
val:   0 ... len(val)-1
test:  0 ... len(test)-1
```

Do not create a global cross-split `img_id` mapping. `self.skyfind_meta_by_img_id`
belongs to the current dataset instance, because train/val/test may all contain
`img_id == 0`.

SkyFind identity metadata:

```python
file_name = sample["fileName"]
raw_img_id = Path(file_name).stem
```

For training/validation tuple compatibility, only `img_id` is returned.

For export/visualization, use a side table:

```python
self.skyfind_meta_by_img_id[img_id] = {
    "file_name": file_name,
    "raw_img_id": raw_img_id,
    "expression": expression,
    "gt_bbox_xyxy_raw_original": original_bbox,
    "gt_bbox_xyxy_raw_clamped": clamped_bbox,
    "bbox_was_clamped": bbox_was_clamped,
    "orig_size": (H, W),
    "resize_info": resize_info,
}
```

### 5.4 BBox Validation and Clamp

SkyFind boxes are raw pixel `xyxy`.

Implementation:

```python
x1, y1, x2, y2 = map(float, sample["bbox"])
x1 = min(max(x1, 0.0), W - 1.0)
y1 = min(max(y1, 0.0), H - 1.0)
x2 = min(max(x2, 0.0), W - 1.0)
y2 = min(max(y2, 0.0), H - 1.0)
```

After clamping:

```python
if x2 <= x1 or y2 <= y1:
    skip sample and count it as invalid
```

Do not terminate full training for rare bad annotations. Log counts during
dataset initialization.

Training and evaluation use the clamped bbox. Metadata should keep both the
original annotation and the clamped version:

```python
gt_bbox_xyxy_raw_original
gt_bbox_xyxy_raw_clamped
bbox_was_clamped
```

This keeps the baseline robust while preserving enough information to audit
unusual IoU cases later.

### 5.5 Corrupt Image Handling

Train and val currently have no corrupt images. Test contains repeated corrupt
entries for:

```text
SeaDronesSee_3863.jpg
```

Dataset initialization should verify image openability and skip unreadable
samples with a logged count.

For Experiment 0 training:

```text
Train.json and Val.json are sufficient.
```

The corrupt test image matters when test-set evaluation/export is added.

### 5.6 Query Length Handling

Experiment 0 setting:

```text
--max_query_len 128
```

Required compatibility fix:

```python
batch_a = tokenizer.batch_encode_plus(
    [prompt_tokens_a],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=seq_length,
)
```

Then:

```python
input_ids = list(np.array(batch_a["input_ids"][0]))
input_mask = list(np.array(batch_a["attention_mask"][0]))
```

`words_mask` must also be clipped/padded to `seq_length`:

```python
words_mask = words_mask[:seq_length]
while len(words_mask) < seq_length:
    words_mask.append(0)
```

Why `batch_a` must be truncated too:

```text
The model uses `batch_a` through `text_info_data` in BERT. Truncating only
word_id would still allow over-length `batch_a` to reach the text encoder.
```

This is a compatibility fix, not a long-expression method.

Tokenization diagnostics should distinguish the raw expression and the prompted
text actually sent to BERT:

```text
raw_expression_words
raw_expression_token_len
prompted_token_len
truncated_by_batch_a
```

The prompted token length is the important limit for DiffusionREC because the
model encodes the prompt-wrapped sentence, not only the raw SkyFind expression.

### 5.7 Relation Parser Strict Mode

`ralation_analysis()` uses `sng_parser`. Long SkyFind expressions may trigger
parser failures.

Experiment 0 must preserve the original DiffusionREC capability boundary:

```text
No parser fallback is allowed in the training path.
```

Mask semantics:

```text
1 = selected / valid token
0 = ignored / padding token
```

Implementation rule:

```python
words_mask = ralation_analysis(example.text_a)
```

must succeed. The returned mask is then deterministically clipped or padded to
match the tokenizer-side `actual_text_len`, and finally padded to `seq_length`
with zeros on padding positions.

If `sng_parser.parse(expression)` fails, Experiment 0 should fail in parser
audit or dataset construction rather than silently changing model behavior.

### 5.8 Transform Metadata for Export

For validation metrics, raw inverse coordinates are not strictly required
because IoU is invariant under the deterministic aspect-ratio-preserving scale
and pad.

For prediction export, store:

```python
resize_info = {
    "scale": scale,
    "pad_x": left,
    "pad_y": top,
    "resized_w": resized_w,
    "resized_h": resized_h,
    "imsize": imsize,
}
```

Raw inverse:

```python
x_raw = (x_640 - pad_x) / scale
y_raw = (y_640 - pad_y) / scale
```

Clamp recovered raw coordinates to image bounds before JSON export.

Implementation note:

```text
This metadata does not have to be returned in the training tuple. It can be
stored in `self.skyfind_meta_by_img_id` or returned only by a dedicated export
dataset path.
```

## 6. Training and Evaluation Settings

### 6.1 Smoke Run

Goal:

```text
Verify data loading, tokenization, forward, loss, backward, and validation.
```

Use:

```text
--dataset skyfind
--data_root ../../../BioLoc/data/SkyFind_data
--imsize 640
--max_query_len 128
--batch_size 1
--num_workers 2
```

Important correction:

```text
Turning off --aug_crop, --aug_scale, and --aug_translate does not disable all
training augmentation. The original train transform still includes ColorJitter
and RandomHorizontalFlip.
```

For a pure data-contract smoke test, use either:

```text
1. A dataset inspection script that uses the val transform.
2. A temporary debug flag that forces train split to use val transform.
```

For faithful DiffusionREC training, keep the original train augmentation unless
we explicitly define a no-augmentation ablation.

Use two smoke stages:

```text
Smoke-0 data-contract:
  val transform, no augmentation, inspect 20 samples

Smoke-1 training-contract:
  original train transform, batch_size=1, 20 iterations
```

Smoke-0 validates coordinates and tokenization. Smoke-1 validates the actual
training path.

### 6.2 Main Baseline

First baseline:

```text
E0-A:
train = Train.json
val   = Val.json
```

Fixed settings:

```text
--imsize 640
--max_query_len 128
```

Augmentation:

```text
Use the original DiffusionREC train transform unless it causes instability.
```

Second baseline after first success:

```text
E0-B:
train = Train.json + Train_Aug.json sampled as official-style train_merged
val   = Val.json
```

This second run is still dataset adaptation, not a method innovation.

## 7. Metrics

Minimum original-style metrics:

```text
mIoU
Acc@0.5
```

SkyFind diagnostics:

```text
Acc@0.7
small/medium/large object buckets
expression length buckets
token truncation ratio
```

Important limitation:

```text
The current DiffusionREC validation path effectively assumes val batch size 1.
Keep validation batch size at 1 until metric aggregation is cleaned up.
```

## 8. Prediction Export Contract

When export is implemented, write:

```json
{
  "img_id": 0,
  "raw_img_id": "Visdrone_1136",
  "file_name": "Visdrone_1136.jpg",
  "expression": "...",
  "gt_bbox_xyxy_raw": [x1, y1, x2, y2],
  "pred_bbox_xyxy_raw": [x1, y1, x2, y2],
  "iou": 0.0,
  "score": null,
  "resize_info": {
    "scale": 0.333333,
    "pad_x": 0,
    "pad_y": 140,
    "imsize": 640
  }
}
```

Prediction export is not required for the first 20-iteration smoke run, but it
is required before reporting final Experiment 0 results.

## 9. Implementation Checklist

1. Add `skyfind` to `SUPPORTED_DATASETS`.
2. Add direct JSON loading for `Train.json`, `Val.json`, and `Test.json`.
3. Bypass `.pth` split cache for SkyFind.
4. Use `img_id = idx`.
5. Preserve `file_name` and `raw_img_id` in side metadata.
6. Load image from `data_root/images/fileName`.
7. Validate and clamp raw `xyxy` bbox.
8. Skip unreadable images and invalid clamped boxes with logged counts.
9. Keep raw `xyxy` into transform.
10. Keep transform output as normalized `cxcywh`.
11. Set `max_query_len=128`.
12. Add tokenizer truncation/padding to `seq_length` for `batch_a`.
13. Clip/pad `words_mask` to `seq_length`.
14. Enforce strict `sng_parser` alignment without fallback.
15. Keep `imsize=640`.
16. Keep val batch size 1.
17. Add a dataset smoke script or command.
18. Add a one-batch forward smoke command.
19. Add 20-iteration smoke train command.
20. Add prediction export before final reporting.

## 10. Acceptance Criteria

The implementation is acceptable when:

```text
Dataset:
  SkyFind train/val samples load without crashing.
  Initialization logs skipped/clamped/corrupt counts.
  Returned model target is normalized cxcywh in [0, 1].

Tokenization:
  No assertion failure for long SkyFind expressions.
  token_truncated_count is reported.

Model:
  One batch forward works.
  20 training iterations produce finite loss.
  Validation over a small subset returns mIoU and Acc@0.5.

Experiment integrity:
  No new model module, loss, spatial prior, or long-expression architecture is
  introduced.
```

## 10.1 First Commit Scope

The first implementation commit should contain:

```text
1. docs + README
2. skyfind dataset branch
3. tokenizer truncation fix
4. strict parser alignment
5. dataset inspection smoke script
```

Suggested commit title:

```text
Add SkyFind dataset adaptation spec and loader for DiffusionREC baseline
```

## 11. Deferred Work

Do not implement these in Experiment 0:

```text
long-expression hierarchical encoder
target/relation/landmark phrase decomposition
spatial prior map
Box + Spatial Prior Joint Diffusion
one-step or few-step distillation
image resolution ablation
augmentation redesign
```

These belong to later experiments after the original DiffusionREC baseline is
measured.
