cd /root/autodl-tmp/VLMSkyFind
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 首次运行 DeepSeek/InternVL 前安装对应运行时。InternVL 使用独立轻量
# venv，保持当前 Qwen/DeepSeek 可用环境不变。
bash scripts/setup_model_runtimes.sh deepseek
bash scripts/setup_model_runtimes.sh internvl

# LLaVA/GeoChat 当前暂停，不影响以下三个模型的实验。

# Qwen
python scripts/run_vlm_skyfind.py \
  --model qwen2.5-vl-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --no-resume \
  --limit 5 \
  --output predictions/qwen2.5-vl-7b_val_rsvg_v2_smoke.jsonl \
  --summary-output predictions/qwen2.5-vl-7b_val_rsvg_v2_smoke_summary.json

# 检查结果：
python -m json.tool predictions/qwen2.5-vl-7b_val_rsvg_v2_smoke_summary.json
sed -n '1,5p' predictions/qwen2.5-vl-7b_val_rsvg_v2_smoke.jsonl

# 专门验证 Test 坏图跳过及后续样本继续运行：
python scripts/run_vlm_skyfind.py \
  --model qwen2.5-vl-7b \
  --split test \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --no-resume \
  --start-index 4990 \
  --limit 4 \
  --output predictions/qwen2.5-vl-7b_test_bad_image_smoke.jsonl \
  --summary-output predictions/qwen2.5-vl-7b_test_bad_image_smoke_summary.json

这里预计出现 1 条 image_error，另外 3 条正常进入模型：
python -m json.tool predictions/qwen2.5-vl-7b_test_bad_image_smoke_summary.json

# deepseek-vl-7b
python scripts/run_vlm_skyfind.py \
  --model deepseek-vl-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --no-resume \
  --limit 5 \
  --output predictions/deepseek-vl-7b_val_rsvg_v2_smoke.jsonl \
  --summary-output predictions/deepseek-vl-7b_val_rsvg_v2_smoke_summary.json

    #报错
    libgomp: Invalid value for environment variable OMP_NUM_THREADS
    Traceback (most recent call last):
      File "/root/autodl-tmp/VLMSkyFind/scripts/run_vlm_skyfind.py", line 62, in <module>
        run(parse_args())
      File "/root/autodl-tmp/VLMSkyFind/vlm_skyfind/runner.py", line 90, in run
        adapter = create_adapter(
      File "/root/autodl-tmp/VLMSkyFind/vlm_skyfind/adapters/registry.py", line 22, in create_adapter
        return getattr(module, class_name)(**kwargs)
      File "/root/autodl-tmp/VLMSkyFind/vlm_skyfind/adapters/deepseek.py", line 10, in __init__
        from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
    ModuleNotFoundError: No module named 'deepseek_vl'

    # 结论：修复前未安装 DeepSeek-VL 官方源码包。执行上面的
    # setup_model_runtimes.sh deepseek 后重新运行本命令。

# internvl2.5-8b：必须使用官方版本隔离环境，不修改当前 model 环境。
.venvs/internvl/bin/python scripts/run_vlm_skyfind.py \
  --model internvl2.5-8b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --no-resume \
  --limit 5 \
  --output predictions/internvl2.5-8b_val_rsvg_v2_smoke.jsonl \
  --summary-output predictions/internvl2.5-8b_val_rsvg_v2_smoke_summary.json


# llava-onevision-7b
conda run --no-capture-output -n vlm-llava python scripts/run_vlm_skyfind.py \
  --model llava-onevision-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype float16 \
  --llava-model-name llava_qwen \
  --no-resume \
  --limit 5 \
  --output predictions/llava-onevision-7b_val_rsvg_v2_smoke.jsonl \
  --summary-output predictions/llava-onevision-7b_val_rsvg_v2_smoke_summary.json

    # 报错
    libgomp: Invalid value for environment variable OMP_NUM_THREADS
    Traceback (most recent call last):
      File "/root/autodl-tmp/VLMSkyFind/scripts/run_vlm_skyfind.py", line 62, in <module>
        run(parse_args())
      File "/root/autodl-tmp/VLMSkyFind/vlm_skyfind/runner.py", line 90, in run
        adapter = create_adapter(
      File "/root/autodl-tmp/VLMSkyFind/vlm_skyfind/adapters/registry.py", line 22, in create_adapter
        return getattr(module, class_name)(**kwargs)
      File "/root/autodl-tmp/VLMSkyFind/vlm_skyfind/adapters/llava.py", line 14, in __init__
        from llava.constants import (
    ModuleNotFoundError: No module named 'llava'

    # 结论：修复前未安装 LLaVA-NeXT 官方源码包。适配器现在还会显式使用
    # llava_qwen 架构名、FP16 和 sdpa，避免本地目录名导致模型架构误判、
    # 视觉塔 dtype 不一致或强制依赖 flash-attn。

# geochat-7b
conda run --no-capture-output -n vlm-geochat python scripts/run_vlm_skyfind.py \
  --model geochat-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode model_native \
  --dtype float16 \
  --no-resume \
  --limit 5 \
  --output predictions/geochat-7b_val_rsvg_v2_smoke.jsonl \
  --summary-output predictions/geochat-7b_val_rsvg_v2_smoke_summary.json

    # 报错
    libgomp: Invalid value for environment variable OMP_NUM_THREADS
    Traceback (most recent call last):
      File "/root/autodl-tmp/VLMSkyFind/scripts/run_vlm_skyfind.py", line 62, in <module>
        run(parse_args())
      File "/root/autodl-tmp/VLMSkyFind/vlm_skyfind/runner.py", line 90, in run
        adapter = create_adapter(
      File "/root/autodl-tmp/VLMSkyFind/vlm_skyfind/adapters/registry.py", line 22, in create_adapter
        return getattr(module, class_name)(**kwargs)
      File "/root/autodl-tmp/VLMSkyFind/vlm_skyfind/adapters/llava.py", line 14, in __init__
        from llava.constants import (
    ModuleNotFoundError: No module named 'llava'

    # 结论：旧适配器错误地把 GeoChat 当作 LLaVA-NeXT。现在改用 GeoChat
    # 官方 geochat 包及其图像预处理；官方 loader 固定 FP16，因此命令显式指定
    # --dtype float16。

## 2026-06-19 旧协议结果检查

- 这些结果使用旧 prompt，只能用于诊断，不得续跑或作为最终 baseline。
- Qwen Val smoke：5/5 正常解析，parse rate 1.0，mIoU 0.16149；原始
  response 证明其输出为原图 pixel 坐标。
- Qwen Test 坏图 smoke：4 条记录中 1 条 `image_error`、3 条进入模型，坏图未计入指标。
- DeepSeek 的旧 `parsed_count=2` 无效：一条误解析 `(x1,y1,x2,y2)` 中的
  数字，另一条 `[0,1]` 坐标被错误当成 pixel。两处均已修复。
- InternVL 五条记录均为 `InternLM2ForCausalLM.generate` 运行时错误；现改用
  官方 `transformers==4.37.2` 的独立 `.venvs/internvl`，等待服务器重跑。
- 上述 DeepSeek/LLaVA/GeoChat traceback 是修复前记录，保留用于追踪问题来源。
- LLaVA 的 `apply_chunking_to_forward` 与 GeoChat 的 BLOOM `_expand_mask`
  报错说明它们不能共享当前新版 Transformers；两个模型现改用各自的官方
  依赖环境，不修改上游代码，也不降级现有 `model` 环境。
