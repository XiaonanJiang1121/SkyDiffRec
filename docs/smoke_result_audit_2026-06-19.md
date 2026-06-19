# Smoke Result Audit: 2026-06-19

This audit covers the saved Qwen2.5-VL, DeepSeek-VL, and InternVL2.5 five-sample
Val outputs. LLaVA-OneVision and GeoChat are intentionally deferred.

## Qwen2.5-VL-7B

All five responses were parsed. The saved summary reports parse rate `1.0` and
mIoU `0.16149`. These numbers are useful only as a pipeline diagnostic because
the run used the earlier prompt without the RSVG output-only constraint.

The raw responses establish that Qwen's output in this protocol is original
pixel space, not `[0,1000]`: the 1920 x 1080 sample returned
`[1536, 327, 1598, 360]`. The RSVG-ZeroOV Qwen path decodes the generated text
directly; `process_vision_info` performs model input preprocessing but no
bounding-box restoration call is present.

## DeepSeek-VL-7B

The saved summary reports two parsed responses out of five, but neither old
metric is valid:

1. One refusal repeated `(x1, y1, x2, y2)`. The old parser extracted the digits
   in those variable names as `[1, 1, 2, 2]`.
2. One answer returned `(0.25, 0.65, 0.31, 0.71)`. The old run declared pixel
   mode, so normalized coordinates were evaluated as sub-pixel values.

The parser now excludes digits attached to coordinate labels. DeepSeek uses
`auto` because its official release does not define a visual-grounding
coordinate convention. Unambiguous `[0,1]` answers are scaled by the original
image width and height; ambiguous integer boxes are not guessed.

## InternVL2.5-8B

All five samples failed before generation with:

```text
AttributeError: 'InternLM2ForCausalLM' object has no attribute 'generate'
```

This is a Transformers runtime mismatch, not a missing bounding-box capability.
InternVL's official RefCOCO evaluation calls `model.chat`, parses a four-number
box, divides it by 1000, and rescales it to the original image size. The official
InternVL requirements pin `transformers==4.37.2`; newer Transformers releases
no longer provide the same inherited `generate` behavior to this remote model
class.

The fix is an isolated `.venvs/internvl` runtime. No compatibility monkeypatch
or downgrade is applied to the working Qwen/DeepSeek environment.

## Rerun Rule

All three models must use new output paths after this protocol change. Old JSONL
files remain diagnostic artifacts and must not be resumed into strict RSVG
results.
