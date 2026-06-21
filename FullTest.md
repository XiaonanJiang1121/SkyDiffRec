# SkyFind VLM 全集测试命令

以下命令使用单张 RTX 4090，严格采用 RSVG prompt 和各模型的原生坐标协议。Val 与 Test 请顺序运行，不要同时启动两个模型进程。

## 运行前准备

```bash
cd /root/autodl-tmp/VLMSkyFind
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

所有全集命令均不包含 `--limit`。断点续跑默认开启；任务中断后直接重新执行同一条命令，不要添加 `--no-resume`。Val 和 Test 中的坏图会记录为 `image_error`，不调用模型，也不计入指标。

## Qwen2.5-VL-7B 全集测试

### Val 全集

```bash
python scripts/run_vlm_skyfind.py \
  --model qwen2.5-vl-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --output predictions/qwen2.5-vl-7b_val_rsvg_full.jsonl \
  --summary-output predictions/qwen2.5-vl-7b_val_rsvg_full_summary.json
```

### Test 全集

```bash
python scripts/run_vlm_skyfind.py \
  --model qwen2.5-vl-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split test \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --output predictions/qwen2.5-vl-7b_test_rsvg_full.jsonl \
  --summary-output predictions/qwen2.5-vl-7b_test_rsvg_full_summary.json
```

## DeepSeek-VL-7B 全集测试

首次运行前：

```bash
bash scripts/setup_model_runtimes.sh deepseek
```

### Val 全集

```bash
python scripts/run_vlm_skyfind.py \
  --model deepseek-vl-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --output predictions/deepseek-vl-7b_val_rsvg_full.jsonl \
  --summary-output predictions/deepseek-vl-7b_val_rsvg_full_summary.json
```

### Test 全集

```bash
python scripts/run_vlm_skyfind.py \
  --model deepseek-vl-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split test \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --output predictions/deepseek-vl-7b_test_rsvg_full.jsonl \
  --summary-output predictions/deepseek-vl-7b_test_rsvg_full_summary.json
```

## InternVL2.5-8B 全集测试

首次运行前：

```bash
bash scripts/setup_model_runtimes.sh internvl
.venvs/internvl/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

版本检查应显示 `transformers 4.37.2`。InternVL 必须使用 `.venvs/internvl/bin/python`，不能使用当前 Conda 环境中的 `python`。

### Val 全集

```bash
.venvs/internvl/bin/python scripts/run_vlm_skyfind.py \
  --model internvl2.5-8b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --output predictions/internvl2.5-8b_val_rsvg_full.jsonl \
  --summary-output predictions/internvl2.5-8b_val_rsvg_full_summary.json
```

### Test 全集

```bash
.venvs/internvl/bin/python scripts/run_vlm_skyfind.py \
  --model internvl2.5-8b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split test \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --output predictions/internvl2.5-8b_test_rsvg_full.jsonl \
  --summary-output predictions/internvl2.5-8b_test_rsvg_full_summary.json
```

已经使用旧版固定 `[0,1000]` 解析完成 InternVL 推理时，不需要重新调用模型。
使用保存的 raw response 生成混合归一化协议结果：

```bash
python scripts/reparse_predictions.py \
  --model internvl2.5-8b \
  --input predictions/internvl2.5-8b_val_rsvg_full.jsonl \
  --output predictions/internvl2.5-8b_val_rsvg_full_reparsed.jsonl \
  --summary-output predictions/internvl2.5-8b_val_rsvg_full_reparsed_summary.json

python scripts/reparse_predictions.py \
  --model internvl2.5-8b \
  --input predictions/internvl2.5-8b_test_rsvg_full.jsonl \
  --output predictions/internvl2.5-8b_test_rsvg_full_reparsed.jsonl \
  --summary-output predictions/internvl2.5-8b_test_rsvg_full_reparsed_summary.json
```

## GeoChat-7B 全集测试

首次运行前：

```bash
bash scripts/setup_model_runtimes.sh geochat
.venvs/geochat/bin/python -c "import torch, transformers, geochat; print(torch.__version__, transformers.__version__)"
```

### Val 全集

```bash
.venvs/geochat/bin/python scripts/run_vlm_skyfind.py \
  --model geochat-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype float16 \
  --max-new-tokens 128 \
  --output predictions/geochat-7b_val_rsvg_full.jsonl \
  --summary-output predictions/geochat-7b_val_rsvg_full_summary.json
```

### Test 全集

```bash
.venvs/geochat/bin/python scripts/run_vlm_skyfind.py \
  --model geochat-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split test \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype float16 \
  --max-new-tokens 128 \
  --output predictions/geochat-7b_test_rsvg_full.jsonl \
  --summary-output predictions/geochat-7b_test_rsvg_full_summary.json
```

## LLaVA-OneVision-7B 全集测试

首次运行前：

```bash
bash scripts/setup_model_runtimes.sh llava
.venvs/llava/bin/python -c "import torch, transformers, llava; print(torch.__version__, transformers.__version__)"
```

运行脚本会默认启用 PyTorch 可扩展显存段，并在每个样本前释放未使用缓存。
这保留 FP16 权重与原始图像预处理，不需要降低图像分辨率或量化模型。

### Val 全集

```bash
.venvs/llava/bin/python scripts/run_vlm_skyfind.py \
  --model llava-onevision-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype float16 \
  --max-new-tokens 128 \
  --llava-model-name llava_qwen \
  --output predictions/llava-onevision-7b_val_rsvg_full.jsonl \
  --summary-output predictions/llava-onevision-7b_val_rsvg_full_summary.json
```

### Test 全集

```bash
.venvs/llava/bin/python scripts/run_vlm_skyfind.py \
  --model llava-onevision-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split test \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype float16 \
  --max-new-tokens 128 \
  --llava-model-name llava_qwen \
  --output predictions/llava-onevision-7b_test_rsvg_full.jsonl \
  --summary-output predictions/llava-onevision-7b_test_rsvg_full_summary.json
```

## 生成 SkyFind Table 4 指标

每个模型的 Val 和 Test 均完成后执行。下面以 Qwen 为例；其余模型替换模型名和文件名前缀即可。

```bash
python scripts/summarize_table4.py \
  --model qwen2.5-vl-7b \
  --val predictions/qwen2.5-vl-7b_val_rsvg_full.jsonl \
  --test predictions/qwen2.5-vl-7b_test_rsvg_full.jsonl \
  --output predictions/qwen2.5-vl-7b_rsvg_full_table4.json
```
