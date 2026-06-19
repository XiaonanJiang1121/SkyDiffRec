# Smoke Result Review: 2026-06-20

This review covers the strict RSVG v2 five-sample Val runs for Qwen2.5-VL-7B
and DeepSeek-VL-7B.

## Metric Correction

The previous `miou` field was the arithmetic mean of per-sample box IoUs. It is
useful diagnostically but is not the `IoU@mean` metric in SkyFind Table 4.
SkyFind averages threshold accuracies at IoU 0.5, 0.6, 0.7, 0.8, and 0.9.

| Model | Parse rate | Raw mean IoU | Table 4 IoU@0.5 | Table 4 IoU@mean |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 100% | 14.7031% | 0.00% | 0.00% |
| DeepSeek-VL-7B | 100% | 0.2717% | 0.00% | 0.00% |

Only five examples were evaluated, four of which have target area below 0.1%
of the image. These values validate the pipeline but are not performance
estimates comparable to the 5,000-sample Val results in Table 4.

## DeepSeek Output Protocol

All five new responses are valid output-only lists and all use unambiguous
normalized `[0,1]` `xyxy` coordinates. Multiplying x by original width and y by
original height is internally consistent. Alternative interpretations were
checked against the saved outputs:

- treating the tuple as `yxyx`: raw mean IoU 0
- flipping x, y, or both axes: raw mean IoU 0
- undoing DeepSeek's square-padding geometry: raw mean IoU 0.0292%
- current original-content normalization: raw mean IoU 0.2717%

The padding alternative is not supported by output behavior. For example, on
`Visdrone_4174.jpg`, direct normalization places the predicted center at y=845,
close to the target center y=854; interpreting it in padded-square coordinates
would move it to y=952.

## Why DeepSeek Is Low

The low score is primarily localization quality rather than parsing or an axis
conversion bug:

- Two predictions have centers relatively near the target, but their areas are
  28.1x and 15.2x the ground-truth areas, producing almost no IoU.
- Two predictions select clearly wrong regions.
- DeepSeek emits only two decimal places. A 0.01 step equals 19.2 pixels in a
  1920-wide image, while several SkyFind targets are only 9-60 pixels wide.
- The official DeepSeek-VL-7B release defines no bounding-box grounding output
  protocol; this is a general chat VLM rather than a grounding-tuned model.

The strict RSVG run should therefore remain unchanged as the primary baseline.
A separate `pixel` prompt run can test whether explicit dimensions and integer
coordinates reduce DeepSeek's quantization and oversized-box problem, but that
result must be reported as a prompt ablation rather than the RSVG baseline.
