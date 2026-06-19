cd /root/autodl-tmp/VLMSkyFind
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 首次运行 DeepSeek/LLaVA/GeoChat 前，分别安装其官方源码运行时。
# DeepSeek 使用当前 model 环境；LLaVA 和 GeoChat 分别使用 vlm-llava 与
# vlm-geochat，现有 model 环境中的 Qwen/InternVL/DeepSeek 不会被修改。
bash scripts/setup_model_runtimes.sh deepseek
bash scripts/setup_model_runtimes.sh llava
bash scripts/setup_model_runtimes.sh geochat

# Qwen
python scripts/run_vlm_skyfind.py \
  --model qwen2.5-vl-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode pixel \
  --limit 5 \
  --output predictions/qwen2.5-vl-7b_val_smoke.jsonl \
  --summary-output predictions/qwen2.5-vl-7b_val_smoke_summary.json

# 检查结果：
python -m json.tool predictions/qwen2.5-vl-7b_val_smoke_summary.json
sed -n '1,5p' predictions/qwen2.5-vl-7b_val_smoke.jsonl

# 专门验证 Test 坏图跳过及后续样本继续运行：
python scripts/run_vlm_skyfind.py \
  --model qwen2.5-vl-7b \
  --split test \
  --prompt-variant rsvg \
  --coordinate-mode pixel \
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
  --coordinate-mode pixel \
  --no-resume \
  --limit 5 \
  --output predictions/deepseek-vl-7b_val_smoke.jsonl \
  --summary-output predictions/deepseek-vl-7b_val_smoke_summary.json

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

# internvl2.5-8b
python scripts/run_vlm_skyfind.py \
  --model internvl2.5-8b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode pixel \
  --limit 5 \
  --output predictions/internvl2.5-8b_val_smoke.jsonl \
  --summary-output predictions/internvl2.5-8b_val_smoke_summary.json


# llava-onevision-7b
conda run --no-capture-output -n vlm-llava python scripts/run_vlm_skyfind.py \
  --model llava-onevision-7b \
  --data-root /root/autodl-tmp/BioLoc/data/SkyFind_data \
  --split val \
  --prompt-variant rsvg \
  --coordinate-mode pixel \
  --dtype float16 \
  --llava-model-name llava_qwen \
  --no-resume \
  --limit 5 \
  --output predictions/llava-onevision-7b_val_smoke.jsonl \
  --summary-output predictions/llava-onevision-7b_val_smoke_summary.json

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
  --coordinate-mode pixel \
  --dtype float16 \
  --no-resume \
  --limit 5 \
  --output predictions/geochat-7b_val_smoke.jsonl \
  --summary-output predictions/geochat-7b_val_smoke_summary.json

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

## 2026-06-19 本地结果检查

- Qwen Val smoke：5/5 正常解析，parse rate 1.0，mIoU 0.16149。
- Qwen Test 坏图 smoke：4 条记录中 1 条 `image_error`、3 条进入模型，坏图未计入指标。
- Qwen 返回的是原图 pixel 坐标；越界坐标按评测协议裁剪到图像边界，行为正常。
- InternVL 命令下没有记录报错，也没有对应 predictions，当前不能视为已通过，需要在服务器重跑。
- 上述 DeepSeek/LLaVA/GeoChat traceback 是修复前记录，保留用于追踪问题来源。
- LLaVA 的 `apply_chunking_to_forward` 与 GeoChat 的 BLOOM `_expand_mask`
  报错说明它们不能共享当前新版 Transformers；两个模型现改用各自的官方
  依赖环境，不修改上游代码，也不降级现有 `model` 环境。
