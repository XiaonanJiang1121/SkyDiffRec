# Full VLM Result Review (updated 2026-06-22)

This review independently reloads the full JSONL records and recomputes the
SkyFind Table 4 metrics. Percentages below are `IoU@0.5 / IoU@mean`.

## Completion and integrity

| Model | Val | Test | Status |
| --- | ---: | ---: | --- |
| Qwen2.5-VL-7B | 5,000/5,000 | 16,546/16,546 | complete |
| DeepSeek-VL-7B | 5,000/5,000 | 16,546/16,546 | complete |
| InternVL2.5-8B | 5,000/5,000 | 16,546/16,546 | complete |
| LLaVA-OneVision-7B | 5,000/5,000 | 16,546/16,546 | complete |

The completed Test files contain 34 `image_error` records. They are retained
for auditability but correctly excluded from the 16,512-sample metric
denominator. Val contains no image errors.

## SkyFind Table 4 results

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">SkyFind Val</th>
      <th colspan="2">SkyFind Test</th>
      <th colspan="2">Average</th>
    </tr>
    <tr>
      <th>IoU@0.5</th>
      <th>IoU@mean</th>
      <th>IoU@0.5</th>
      <th>IoU@mean</th>
      <th>IoU@0.5</th>
      <th>IoU@mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-VL-7B</td>
      <td>37.48</td><td>29.38</td>
      <td>45.40</td><td>31.40</td>
      <td><strong>41.44</strong></td><td><strong>30.39</strong></td>
    </tr>
    <tr>
      <td>DeepSeek-VL-7B</td>
      <td>0.68</td><td>0.21</td>
      <td>0.28</td><td>0.09</td>
      <td><strong>0.48</strong></td><td><strong>0.15</strong></td>
    </tr>
    <tr>
      <td>InternVL2.5-8B</td>
      <td>1.02</td><td>0.40</td>
      <td>0.60</td><td>0.19</td>
      <td><strong>0.81</strong></td><td><strong>0.29</strong></td>
    </tr>
    <tr>
      <td>LLaVA-OneVision-7B</td>
      <td>1.18</td><td>0.36</td>
      <td>0.10</td><td>0.03</td>
      <td><strong>0.64</strong></td><td><strong>0.20</strong></td>
    </tr>
  </tbody>
</table>

Qwen has a 98.24% Val parse rate and 98.55% Test parse rate. Its raw mIoU is
33.73% on Val and 40.15% on Test. The final parser reconstructs the official
Qwen smart-resize dimensions (`patch_size * merge_size = 28`) and maps each
generated resized-input box back to original SkyFind pixels before IoU. The
calculation uses only the saved raw responses, original image dimensions, and
the official processor configuration; no model inference or image loading is
performed. It then applies the same strict no-reorder/no-clamp validation as the
other models. The old 34.88 / 21.50 average incorrectly treated generated
resized-input coordinates as original pixels and is superseded.

### Qwen strict-versus-sanitize audit

Before box policy is applied, Val contains 3 reversed-x boxes, 38 out-of-bound
boxes, and 85 responses without four coordinates. Test contains 48 reversed-x
boxes, 78 out-of-bound boxes, 192 responses without four coordinates, and the
34 excluded image errors. Neither split contains a reversed-y or zero-area
box. These categories can overlap. Strict validation retains valid out-of-image
`xyxy` boxes but rejects reversed or zero-area boxes.

The former repairing policy gives an Average of 41.49 / 30.44; strict gives
41.44 / 30.39. Sanitizing therefore changes `IoU@0.5` by only +0.050 points and
`IoU@mean` by +0.058 points. The Qwen conclusion is stable, but strict is used
as the primary row for cross-model fairness.

DeepSeek has an even higher parse rate (98.86% Val and 99.85% Test), so its low
IoU is not caused by the parser. Its median predicted box area is about 5.85
times the ground-truth area on Val, versus 1.11 times for Qwen. The model mostly
returns valid `[0,1]` coordinates but localizes overly broad regions, which
explains the low overlap.

## Failure-mode diagnostics

The following results were regenerated on 2026-06-22 with
`scripts/analyze_box_failure_modes.py` from the final strict Val/Test JSONL
files. The script first validates each model's final coordinate protocol, so
provisional coordinate profiles and repaired boxes cannot enter this table.
Test percentages exclude the same 34 unreadable images used by Table 4.

`Area ratio` is predicted-box area divided by GT-box area. `Center error` is
the Euclidean distance between box centers divided by the GT-box diagonal.
For `scale/shape failure`, the predicted box is translated to the GT center
without changing its width or height; a centered IoU below 0.5 is counted as a
failure. Area, center, and scale/shape statistics use strict valid predictions;
parse rate uses the full evaluable split denominator.

| Model | Area ratio median Val/Test | >10x GT Val/Test | Scale/shape failure Val/Test | Center error / GT diagonal Val/Test | Parse rate Val/Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 1.11 / 1.14 | 2.85% / 3.63% | 26.49% / 29.40% | 0.85 / 0.25 | 98.24% / 98.55% |
| InternVL2.5-8B | 3.11 / 8.15 | 21.93% / 43.43% | 84.88% / 91.30% | 3.94 / 3.93 | 96.30% / 98.61% |
| LLaVA-OneVision-7B | 11.94 / 31.32 | 55.07% / 81.28% | 92.60% / 99.01% | 2.57 / 4.20 | 86.54% / 98.29% |
| DeepSeek-VL-7B | 5.85 / 8.13 | 34.11% / 42.41% | 92.01% / 93.44% | 3.11 / 3.47 | 98.86% / 99.85% |

The grouped `IoU@0.5` diagnostics are below. Every cell is `Val / Test` in
percent. Target-size buckets use GT area divided by image area: large is at
least 1%, small is 0.1%-1%, and tiny is below 0.1%. Expression lengths are
short (at most 20 words), medium (21-40), and long (over 40). Relation and
ordinal subsets are deterministic keyword diagnostics implemented in the
script, not official SkyFind annotations.

| Model | Large | Small | Tiny | Short | Medium | Long | Non-relational | Relational | Non-ordinal | Ordinal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 61.83 / 84.81 | 41.24 / 68.12 | 21.79 / 32.75 | 51.53 / 56.68 | 35.50 / 41.75 | 26.71 / 27.02 | 38.70 / 49.89 | 36.76 / 43.27 | 45.49 / 47.19 | 28.39 / 37.91 |
| InternVL2.5-8B | 2.84 / 7.59 | 1.21 / 1.20 | 0.00 / 0.05 | 1.47 / 0.91 | 1.02 / 0.48 | 0.21 / 0.33 | 0.97 / 0.81 | 1.05 / 0.50 | 1.54 / 0.65 | 0.43 / 0.38 |
| LLaVA-OneVision-7B | 9.31 / 4.30 | 0.00 / 0.00 | 0.00 / 0.00 | 2.04 / 0.14 | 0.99 / 0.08 | 1.04 / 0.11 | 1.68 / 0.21 | 0.89 / 0.05 | 1.50 / 0.11 | 0.81 / 0.09 |
| DeepSeek-VL-7B | 5.05 / 9.62 | 0.07 / 0.15 | 0.00 / 0.00 | 1.47 / 0.26 | 0.52 / 0.31 | 0.41 / 0.00 | 0.97 / 0.45 | 0.51 / 0.20 | 0.98 / 0.29 | 0.34 / 0.25 |

These results sharpen the model-level diagnosis. Qwen's predicted area is
usually close to GT, but accuracy falls monotonically for smaller targets and
longer expressions, and ordinal expressions are substantially harder than
non-ordinal ones. InternVL combines large center error with frequent
scale/shape mismatch. LLaVA's dominant failure is severe over-sizing,
especially on Test. DeepSeek parses nearly every response, yet more than 92%
of its valid boxes still fail the centered scale/shape test, confirming that
its low score is a localization-quality failure rather than a parsing failure.

## Mixed-coordinate protocol audit

The final mixed-coordinate pass follows two reproducibility rules used by
published grounding systems:

1. InternVL's official RefCOCO evaluator selects `[0,1000]` when
   `sum(box) >= 4`, otherwise `[0,1]`, then scales to original image dimensions.
   It does not choose a scale using ground truth and does not clamp an
   out-of-image prediction back into the image.
2. Kosmos-2's ICLR 2024 RefCOCO evaluator decodes only its declared location
   representation. When no valid box is decoded, it inserts a zero-area box so
   that the sample remains in the denominator. This supports treating an
   undeclared or ambiguous coordinate scale as IoU zero rather than guessing.

These rules are implemented by `scripts/reparse_mixed_coordinates.py`. Boxes
must already be valid `xyxy`; the audit does not reorder reversed corners or
clip hallucinated coordinates to the image boundary.

Primary references are the [InternVL CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_InternVL_Scaling_up_Vision_Foundation_Models_and_Aligning_for_Generic_CVPR_2024_paper.html),
its [official RefCOCO evaluator](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/eval/refcoco/evaluate_grounding.py),
and the [official Kosmos-2 RefCOCO evaluator](https://github.com/microsoft/unilm/blob/master/kosmos-2/evaluation/refcoco/refexp_evaluate.py).

## InternVL result

InternVL Val contains 4,815 strict valid-box predictions, 60 parse failures,
and 125 inference errors. All 125 inference errors are CUDA out-of-memory
failures with roughly 2.4 GiB reserved but unallocated, consistent with allocator
fragmentation under dynamic high-resolution tiling. Per the fixed evaluation
decision, they remain in the 5,000-sample denominator as IoU zero and are not
rerun. Therefore the reported row is final for this experiment rather than
provisional.

The official mixed parser assigns 551 Val responses to `[0,1]` and 4,324 to
`[0,1000]`; 125 are inference errors. Test assigns 5,451 responses to `[0,1]`
and 11,061 to `[0,1000]`, with the 34 common bad images excluded. This includes
557 Val and 48 Test boxes containing values above 1000: the official evaluator
still applies the `[0,1000]` branch, leaving them out of image instead of
guessing original-pixel coordinates or clipping them. Strict valid-box parse
rates are 96.30% Val and 98.61% Test. The metrics remain 1.02 / 0.396 and
0.600 / 0.188, confirming that the conclusion is not driven by coordinate
repair.

## LLaVA-OneVision result

LLaVA completed both splits without inference errors after enabling expandable
CUDA segments and releasing unused cache between samples. Val parses 86.72% of
responses under the earlier repairing parser. Under strict no-reorder/no-clamp
validation, Val parses 86.54% and Test parses 98.29%; the Test file also
contains the same 34 bad image records excluded for every model. Therefore its
low Test accuracy is not an environment or parser failure.

The principal error is box scale. Median predicted area is 11.9 times the
ground-truth area on Val and 31.2 times on Test. `IoU@0.5` is zero for both
small and tiny target buckets; large targets reach 9.31% on Val and 4.30% on
Test. Performance also drops sharply from Val to the maritime Test domain,
including 0% `IoU@0.5` on MOBDrone.

Most outputs use unambiguous `[0,1]` coordinates and are restored to original
pixels correctly. The remaining 665 Val and 129 Test boxes use bare values
above 1 that could be percentages, normalized-to-1000 coordinates, processor
pixels, or original pixels. LLaVA-OneVision publishes no REC coordinate
contract, so they remain `ambiguous` and count as IoU zero. A further 8 Val and
154 Test `[0,1]` outputs are reversed or zero-area and are rejected rather than
repaired.

As a sensitivity check, every ambiguous LLaVA box was alternatively interpreted
under one fixed scale at a time: `[0,100]`, `[0,1000]`, or original pixels. All
three alternatives produce exactly the same Table 4 values as the strict row;
none of these boxes reaches a reported IoU threshold. Thus the LLaVA result is
invariant to this unresolved subset without using a per-sample oracle.

## DeepSeek result

DeepSeek has 57 ambiguous Val outputs. Test has 17 ambiguous boxes and 8
responses with no parseable coordinates; all remain in their split denominator
as IoU zero. Strict parse rates are 98.86% Val and 99.85% Test. As with LLaVA,
fixed sensitivity runs that interpret every ambiguous box as `[0,100]`,
`[0,1000]`, or original pixels produce exactly the same reported metrics. The
0.48 / 0.15 average is therefore not an artifact of choosing the strict parser.

GeoChat was removed from the experiment scope and has no reported result.
