# VLMSkyFind

`VLMSkyFind` evaluates the zero-shot referring-object localization ability of
five 7B/8B vision-language models on the SkyFind Val and Test splits:

1. GeoChat-7B
2. LLaVA-OneVision-7B
3. Qwen2.5-VL-7B-Instruct
4. DeepSeek-VL-7B-Chat
5. InternVL2.5-8B

The evaluation reproduces the released RSVG-ZeroOV prompt, runs one model at a
time on a single GPU, and records raw responses, parsed boxes, IoU, latency, and
failure status in resumable JSONL files. Images enter each model at their
original SkyFind dimensions before model-native preprocessing. Missing or
corrupt images in either Val or Test are logged and skipped without interrupting
the split.

See [`docs/vlm_skyfind_evaluation.md`](docs/vlm_skyfind_evaluation.md) for the
evaluation protocol and server commands. The initial output audit is recorded in
[`docs/smoke_result_audit_2026-06-19.md`](docs/smoke_result_audit_2026-06-19.md).
The strict-prompt result and Table 4 metric review is in
[`docs/smoke_result_review_2026-06-20.md`](docs/smoke_result_review_2026-06-20.md).
The first complete Qwen/DeepSeek result audit is in
[`docs/full_result_review_2026-06-21.md`](docs/full_result_review_2026-06-21.md).
