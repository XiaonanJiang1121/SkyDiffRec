# VLM-SkyFind Zero-Shot Evaluation

## 1. Goal

Measure the direct zero-shot grounding ability of five 7B/8B VLMs on SkyFind
before designing later coarse-to-fine methods:

1. GeoChat-7B
2. LLaVA-OneVision-7B
3. Qwen2.5-VL-7B-Instruct
4. DeepSeek-VL-7B-Chat
5. InternVL2.5-8B

The primary protocol reproduces the exact text assembled by the released
RSVG-ZeroOV `llmattn.py`. The two spaces after `description.` are intentional:

```text
Locate it according to the following description.  {referring expression} The output format should be like [x1, y1, x2, y2] without any other text.
```

The sentence often quoted in the paper description, `Locate the object referred
to by ...`, is not the prompt executed by the released RSVG-ZeroOV code. The
output-only sentence is essential for direct-box evaluation, especially for
general chat models.

The default is `--prompt-variant rsvg`. An optional `pixel` protocol remains
available as another complete one-sentence prompt:

```text
Locate the object referred to by '{referring expression}' and return only its box coordinates as [x1, y1, x2, y2] in the original {width} x {height} SkyFind image pixel coordinate system.
```

Do not mix the two prompt variants in one result file.

## 2. What Is Recorded

Every sample is appended immediately to JSONL, so an interrupted multi-hour run
can resume safely. A record contains the expression, image metadata, ground
truth, full prompt, raw model response, parsed prediction, IoU, latency, and
failure status. Failed parses receive IoU 0. For both Val and Test, corrupt or
missing source images are reported as `image_error`, skipped without calling the
model, and excluded from model-quality denominators. `skipped_image_count`
reports their total. Each unique image is fully decoded once before inference;
its original size or failure state is cached for repeated expressions.

Each JSONL also has a `.meta.json` protocol file. It stores the complete prompt
template, requested and resolved coordinate modes, and the reason for the
model-native choice. Resume refuses to mix models, splits, prompt text, or
coordinate conventions in one output.
The runner stops after five consecutive inference exceptions by default, which
prevents an adapter/environment mismatch from wasting an entire split.

The primary summary metrics match SkyFind Table 4:

- `iou_at_0.5`: fraction of samples whose box IoU is at least 0.5
- `iou_at_mean`: equal-weight mean of IoU-threshold accuracies at 0.5, 0.6,
  0.7, 0.8, and 0.9
- percentage versions of both values for direct comparison with the paper

The paper describes `IoU@mean` as an average over the five thresholds and does
not publish unequal weights, so this implementation assigns each threshold
weight 1/5. `miou` remains in the JSON only as the conventional raw mean of
per-sample IoUs; it must not be reported as SkyFind `IoU@mean`.

The summary also reports:

- response parse rate and parsed-only raw mIoU
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Qwen2.5-VL and InternVL use Transformers plus their remote model code.
DeepSeek-VL additionally requires the official `deepseek_vl` Python package.
LLaVA-OneVision requires the official LLaVA-NeXT package, imported as `llava`.
GeoChat requires its own official package, imported as `geochat`; it must not
reuse the LLaVA-NeXT adapter. Install these pinned official source runtimes as
needed:

```bash
bash scripts/setup_model_runtimes.sh deepseek
bash scripts/setup_model_runtimes.sh internvl
bash scripts/setup_model_runtimes.sh llava
bash scripts/setup_model_runtimes.sh geochat
```

DeepSeek is installed in the active environment. InternVL uses
`.venvs/internvl`, which reuses the active environment's PyTorch/CUDA packages
but installs the official `transformers==4.37.2` and `tokenizers==0.15.1`
separately. Run it with `.venvs/internvl/bin/python`. This resolves the missing
`InternLM2ForCausalLM.generate` method without modifying model code or changing
the working Qwen environment.

LLaVA-OneVision and GeoChat must not share that environment: their official
code imports Transformers APIs
from different generations. The setup script creates `vlm-llava` with the exact
Transformers commit pinned by LLaVA-NeXT (4.40.0.dev0) and `vlm-geochat` with
the official `transformers==4.31.0`. Their official PyTorch versions are also
installed independently. This leaves the working `model` environment for
Qwen, InternVL, and DeepSeek unchanged. Project code, datasets, and model
weights remain shared and are not copied into the Conda environments.

If pip reports that it cannot download `setuptools>=61.0` while installing an
editable official runtime, first pull the latest branch and rerun the setup
command. The cloned source directory is reused. Do not manually install LLaVA
or GeoChat into the working `model` environment. Verify their isolated
environments only if the retry still fails:

```bash
conda run -n vlm-llava python -c "import setuptools; print(setuptools.__version__)"
conda run -n vlm-geochat python -c "import setuptools; print(setuptools.__version__)"
```

Run LLaVA and GeoChat through their dedicated environments:

```bash
conda run --no-capture-output -n vlm-llava python scripts/run_vlm_skyfind.py --help
conda run --no-capture-output -n vlm-geochat python scripts/run_vlm_skyfind.py --help
```

Both keep their vision towers in FP16, so their evaluation commands must include
`--dtype float16`. The JSONL format and evaluator are environment-independent.

## 4. Smoke Test First

The repository, `models/`, and `configs/` are siblings under `PROJECT`. The
SkyFind annotations and images remain under the BioLoc data directory:

```bash
export PROJECT=/root/autodl-tmp/VLMSkyFind
export DATA_ROOT=/root/autodl-tmp/BioLoc/data/SkyFind_data
cd "$PROJECT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python scripts/run_vlm_skyfind.py \
  --model qwen2.5-vl-7b \
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

All five copy-paste smoke commands and the resolved server errors are kept in
[`../SmokeTest.md`](../SmokeTest.md). LLaVA-OneVision explicitly uses
`--llava-model-name llava_qwen --dtype float16`; deriving the architecture from
the local directory name `llava-onevision-7b` would select the wrong upstream
model class.

`--model-path` is optional: by default the runner resolves the selected model
from `configs/vlm_models.json`. `--data-root` also defaults to the BioLoc path
shown above. Both options can still be overridden explicitly.

`--coordinate-mode model_native` is the default. It resolves as follows:

| Model | Resolved mode | Basis |
| --- | --- | --- |
| Qwen2.5-VL-7B | `pixel` | Saved SkyFind responses include coordinates above 1000 for 1920-wide images; RSVG applies no box restoration function |
| InternVL2.5-8B | `normalized_1000` | Official RefCOCO evaluator divides by 1000, then multiplies x by original width and y by original height |
| DeepSeek-VL-7B | `auto` | Official DeepSeek-VL release defines no grounding-coordinate contract |
| LLaVA/GeoChat | `auto` | Deferred until their runtimes are validated |

For a normalized InternVL prediction, conversion is performed before IoU:

```text
x_pixel = x_normalized * image_width / 1000
y_pixel = y_normalized * image_height / 1000
```

DeepSeek `auto` accepts `[0,1]` coordinates and responses that explicitly state
their scale. A four-number list that could mean either pixels or `[0,1000]` is
marked `ambiguous` and receives no fabricated box. Do not silently switch a
coordinate convention after seeing ground truth.

No shared resize is applied by the dataset or runner. Each adapter receives the
original SkyFind image file and original width/height. Qwen, InternVL,
LLaVA/GeoChat, and DeepSeek then perform only their required model-native image
preprocessing. Predictions are parsed, clamped, and evaluated in the original
SkyFind coordinate system.

## 5. Full Val/Test Runs

Model paths are resolved through `configs/vlm_models.json`. Run InternVL through
its isolated runtime:

```bash
.venvs/internvl/bin/python scripts/run_vlm_skyfind.py \
  --model internvl2.5-8b \
  --data-root "$DATA_ROOT" \
  --split val \
  --output predictions/internvl2.5-8b_val.jsonl \
  --summary-output predictions/internvl2.5-8b_val_summary.json
```

The same command with `--split test` evaluates Test. Resume is enabled by
default. Use `--no-resume` only when intentionally replacing an output file.

After both splits finish, create the six-column Table 4 row. `Average` is the
simple arithmetic mean of the Val and Test metrics, matching the paper; it is
not pooled by the unequal split sizes.

```bash
python scripts/summarize_table4.py \
  --model qwen2.5-vl-7b \
  --val predictions/qwen2.5-vl-7b_val.jsonl \
  --test predictions/qwen2.5-vl-7b_test.jsonl \
  --output predictions/qwen2.5-vl-7b_table4.json
```

## 6. Single-GPU Execution

The benchmark is intentionally serial: one process loads one model on the
single RTX 4090 and evaluates one split. Keep `CUDA_VISIBLE_DEVICES=0`; do not
launch two model processes concurrently. After one model finishes, release the
process before starting the next model so its GPU memory is returned.

## 7. Recommended Experimental Order

1. Run 20-sample smoke tests for Qwen, DeepSeek, and InternVL and inspect raw responses.
2. Run a fixed 300-sample Val pilot for these models to estimate speed and format
   compliance.
3. Complete full Val, then Test only after each model's protocol is frozen.
4. Return to LLaVA-OneVision and GeoChat as a separate runtime task.
5. Compare direct box prediction with a later candidate-reranking experiment;
   direct VLM localization alone does not isolate semantic reasoning from
   small-object spatial precision.
