# VLM-SkyFind Zero-Shot Evaluation

## 1. Goal

Measure the direct zero-shot grounding ability of five 7B/8B VLMs on SkyFind
before designing later coarse-to-fine methods:

1. GeoChat-7B
2. LLaVA-OneVision-7B
3. Qwen2.5-VL-7B-Instruct
4. DeepSeek-VL-7B-Chat
5. InternVL2.5-8B

The primary protocol uses one prompt for every model. Its first sentence is the
RSVG-ZeroOV localization prompt:

```text
Locate the object referred to by '{referring expression}' and return its box coordinates (x1, y1, x2, y2).
```

The default `pixel` variant appends the original image dimensions and asks for
one bracketed box in original-image pixels. This removes an otherwise serious
ambiguity between pixel, `[0, 1]`, and `[0, 1000]` coordinates. The exact prompt
without the suffix remains available as `--prompt-variant rsvg`.

## 2. What Is Recorded

Every sample is appended immediately to JSONL, so an interrupted multi-hour run
can resume safely. A record contains the expression, image metadata, ground
truth, full prompt, raw model response, parsed prediction, IoU, latency, and
failure status. Failed parses receive IoU 0. Corrupt or missing source images are
reported as `image_error`, skipped without calling the model, and excluded from
model-quality denominators. `skipped_image_count` reports their total.

Each JSONL also has a `.meta.json` protocol file. Resume refuses to mix models,
splits, prompts, or coordinate conventions in one output.
The runner stops after five consecutive inference exceptions by default, which
prevents an adapter/environment mismatch from wasting an entire split.

The summary reports:

- mIoU, Acc@0.5, and Acc@0.7 over all evaluable samples
- response parse rate and parsed-only mIoU
- metrics by source dataset
- metrics by expression length: `<=20`, `21-40`, and `>40` words
- metrics by target area ratio: `<0.001`, `0.001-0.01`, and `>=0.01`

Val measures in-domain zero-shot behavior. Test is especially important because
its SeaDronesSee/MOBDrone composition probes maritime cross-domain transfer and
contains substantially smaller targets.

## 3. Environment

From the project root:

```bash
pip install -r requirements-vlm.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Qwen2.5-VL and InternVL use Transformers plus their remote model code.
DeepSeek-VL additionally requires the official `deepseek_vl` Python package.
LLaVA-OneVision requires the official LLaVA-NeXT package, imported as `llava`.
GeoChat also uses a LLaVA-compatible runtime; install its official repository
when its model code differs from the installed LLaVA-NeXT version.

These upstream projects can require different Transformers versions. If one
combined environment cannot load all five models, use one environment per model
family. The JSONL format and evaluator are environment-independent.

## 4. Smoke Test First

Set the server paths. `DATA_ROOT` must contain `Val.json`, `Test.json`, and
`images/`.

```bash
export PROJECT=/root/autodl-tmp/VLMSkyFind
export DATA_ROOT=/root/autodl-tmp/VLMSkyFind/data/SkyFind_data
cd "$PROJECT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 python scripts/run_vlm_skyfind.py \
  --model qwen2.5-vl-7b \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --data-root "$DATA_ROOT" \
  --split val \
  --limit 20 \
  --output predictions/qwen2.5-vl-7b_val_smoke.jsonl \
  --summary-output predictions/qwen2.5-vl-7b_val_smoke_summary.json
```

Inspect raw answers and parse failures before launching a full split:

```bash
sed -n '1,5p' predictions/qwen2.5-vl-7b_val_smoke.jsonl
```

Do not silently switch coordinate conventions after seeing ground truth. If a
model consistently follows a documented normalized convention, run it with the
corresponding explicit `--coordinate-mode` and record that protocol change.

SkyFind Test contains many high-resolution maritime images. Qwen's native
dynamic-resolution processor is left unchanged by default. If a 24 GB GPU runs
out of memory, set an explicit limit such as `--qwen-max-pixels 1003520`, repeat
the pilot, and keep that setting fixed for both Val and Test. InternVL similarly
records `--internvl-max-tiles` (default 12) in the protocol metadata.

## 5. Full Val/Test Runs

Replace the model name and path using `configs/vlm_models.json`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_vlm_skyfind.py \
  --model internvl2.5-8b \
  --model-path models/InternVL2_5-8B \
  --data-root "$DATA_ROOT" \
  --split val \
  --output predictions/internvl2.5-8b_val.jsonl \
  --summary-output predictions/internvl2.5-8b_val_summary.json
```

The same command with `--split test` evaluates Test. Resume is enabled by
default. Use `--no-resume` only when intentionally replacing an output file.

## 6. Single-GPU Execution

The benchmark is intentionally serial: one process loads one model on the
single RTX 4090 and evaluates one split. Keep `CUDA_VISIBLE_DEVICES=0`; do not
launch two model processes concurrently. After one model finishes, release the
process before starting the next model so its GPU memory is returned.

## 7. Recommended Experimental Order

1. Run 20-sample smoke tests for all five adapters and inspect raw responses.
2. Run a fixed 300-sample Val pilot for all models to estimate speed and format
   compliance.
3. Complete full Val for all five models.
4. Complete Test only after each model's protocol is frozen.
5. Compare direct box prediction with a later candidate-reranking experiment;
   direct VLM localization alone does not isolate semantic reasoning from
   small-object spatial precision.
