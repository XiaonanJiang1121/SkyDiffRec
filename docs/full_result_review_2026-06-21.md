# Full VLM Result Review (2026-06-21)

This review independently reloads the full JSONL records and recomputes the
SkyFind Table 4 metrics. Percentages below are `IoU@0.5 / IoU@mean`.

## Completion and integrity

| Model | Val | Test | Status |
| --- | ---: | ---: | --- |
| Qwen2.5-VL-7B | 5,000/5,000 | 16,546/16,546 | complete |
| DeepSeek-VL-7B | 5,000/5,000 | 16,546/16,546 | complete |
| InternVL2.5-8B | 5,000/5,000 | 16,546/16,546 | complete |

The completed Test files contain 34 `image_error` records. They are retained
for auditability but correctly excluded from the 16,512-sample metric
denominator. Val contains no image errors.

## SkyFind Table 4 results

| Model | Val | Test | Average |
| --- | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 32.74 / 20.38 | 37.02 / 22.62 | **34.88 / 21.50** |
| DeepSeek-VL-7B | 0.68 / 0.21 | 0.28 / 0.09 | **0.48 / 0.15** |
| InternVL2.5-8B | 1.02 / 0.40 | 0.60 / 0.19 | **0.81 / 0.29** |

Qwen has a 98.28% Val parse rate and 98.83% Test parse rate. Its raw mIoU is
28.62% on Val and 33.79% on Test. The result is a valid and strong direct-box
zero-shot baseline under the strict RSVG prompt.

DeepSeek has an even higher parse rate (98.86% Val and 99.85% Test), so its low
IoU is not caused by the parser. Its median predicted box area is about 5.85
times the ground-truth area on Val, versus 1.11 times for Qwen. The model mostly
returns valid `[0,1]` coordinates but localizes overly broad regions, which
explains the low overlap.

## InternVL caveat

InternVL Val contains 4,262 parsed predictions, 613 parse failures, and 125
inference errors. All 125 inference errors are CUDA out-of-memory failures with
roughly 2.4 GiB reserved but unallocated, consistent with allocator
fragmentation under dynamic high-resolution tiling. These are runtime failures,
not valid model predictions, so the current 1.02 / 0.40 Val row is provisional.
The failed samples should be retried before reporting the final InternVL
baseline. Test completed without model inference errors and has a 98.64% parse
rate.

InternVL emits mixed coordinate scales: 552 Val and 5,451 Test responses contain
four coordinates entirely within `[0,1]`, while most other valid responses use
the official `[0,1000]` convention. The protocol-corrected figures above
deterministically interpret the former as fractional coordinates and the latter as
normalized-to-1000 coordinates. This changes Test from 0.575 / 0.180 to
0.600 / 0.188 but does not alter the main conclusion. Coordinates above 1000
and zero-area boxes remain parse failures rather than fabricated predictions.
