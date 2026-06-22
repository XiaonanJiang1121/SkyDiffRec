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
      <td>37.52</td><td>29.45</td>
      <td>45.46</td><td>31.44</td>
      <td><strong>41.49</strong></td><td><strong>30.44</strong></td>
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

Qwen has a 98.28% Val parse rate and 98.82% Test parse rate. Its raw mIoU is
33.78% on Val and 40.18% on Test. The final parser reconstructs the official
Qwen smart-resize dimensions (`patch_size * merge_size = 28`) and maps each
generated resized-input box back to original SkyFind pixels before IoU. The
calculation uses only the saved raw responses, original image dimensions, and
the official processor configuration; no model inference or image loading is
performed. The old 34.88 / 21.50 average incorrectly treated generated
resized-input coordinates as original pixels and is superseded.

DeepSeek has an even higher parse rate (98.86% Val and 99.85% Test), so its low
IoU is not caused by the parser. Its median predicted box area is about 5.85
times the ground-truth area on Val, versus 1.11 times for Qwen. The model mostly
returns valid `[0,1]` coordinates but localizes overly broad regions, which
explains the low overlap.

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
