# VLMSkyFind

`VLMSkyFind` evaluates the zero-shot referring-object localization ability of
five 7B/8B vision-language models on the SkyFind Val and Test splits:

1. GeoChat-7B
2. LLaVA-OneVision-7B
3. Qwen2.5-VL-7B-Instruct
4. DeepSeek-VL-7B-Chat
5. InternVL2.5-8B

The evaluation uses the unified one-sentence RSVG prompt, runs one model at a
time on a single GPU, and records raw responses, parsed boxes, IoU, latency, and
failure status in resumable JSONL files. Images enter each model at their
original SkyFind dimensions before model-native preprocessing. Missing or
corrupt images in either Val or Test are logged and skipped without interrupting
the split.

See [`docs/vlm_skyfind_evaluation.md`](docs/vlm_skyfind_evaluation.md) for the
evaluation protocol and server commands. See [`SmokeTest.md`](SmokeTest.md) for
the five model smoke commands, isolated LLaVA/GeoChat environments, and resolved
server errors.
