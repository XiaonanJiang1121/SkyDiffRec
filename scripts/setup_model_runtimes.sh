#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_ROOT="${PROJECT_ROOT}/third_party"
VENV_ROOT="${PROJECT_ROOT}/.venvs"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="${1:-all}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
PACKAGE_INDEX_URL="${VLM_PYPI_INDEX_URL:-https://pypi.org/simple}"

mkdir -p "${THIRD_PARTY_ROOT}" "${VENV_ROOT}"

ensure_source() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local destination="${THIRD_PARTY_ROOT}/${name}"

  if [[ ! -d "${destination}/.git" ]]; then
    mkdir -p "${destination}"
    git -C "${destination}" init
    git -C "${destination}" remote add origin "${url}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
}

install_active_runtime() {
  local name="$1"
  local destination="${THIRD_PARTY_ROOT}/${name}"

  # Reuse the active Conda environment's setuptools instead of downloading a
  # temporary build toolchain from a possibly incomplete mirror.
  "${PYTHON_BIN}" -m pip install \
    --no-build-isolation \
    --no-deps \
    -e "${destination}"
}

ensure_conda_env() {
  local env_name="$1"

  if [[ -z "${CONDA_BIN}" ]]; then
    echo "conda is required to isolate the LLaVA and GeoChat runtimes" >&2
    exit 1
  fi
  if "${CONDA_BIN}" run -n "${env_name}" python -c "pass" >/dev/null 2>&1; then
    echo "Reusing Conda environment: ${env_name}"
  else
    "${CONDA_BIN}" create -y -n "${env_name}" python=3.10 pip setuptools wheel
  fi
}

run_in_env() {
  local env_name="$1"
  shift
  "${CONDA_BIN}" run --no-capture-output -n "${env_name}" "$@"
}

install_env_editable() {
  local env_name="$1"
  local source_name="$2"

  run_in_env "${env_name}" python -m pip install \
    --no-build-isolation \
    --no-deps \
    -e "${THIRD_PARTY_ROOT}/${source_name}"
}

setup_deepseek() {
  "${PYTHON_BIN}" -m pip install attrdict
  ensure_source \
    DeepSeek-VL \
    https://github.com/deepseek-ai/DeepSeek-VL.git \
    681bffb4519856ad27cc17531aacde31ddf6f1a7
  install_active_runtime DeepSeek-VL
}

setup_internvl() {
  local venv="${VENV_ROOT}/internvl"

  if [[ ! -x "${venv}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${venv}"
  fi
  "${venv}/bin/python" -m pip install \
    --index-url "${PACKAGE_INDEX_URL}" \
    transformers==4.37.2 \
    tokenizers==0.15.1 \
    sentencepiece==0.1.99
  "${venv}/bin/python" -c \
    'import torch, transformers; assert transformers.__version__ == "4.37.2"; print(torch.__version__, transformers.__version__)'
}

setup_llava() {
  local env_name="vlm-llava"

  ensure_source \
    LLaVA-NeXT \
    https://github.com/LLaVA-VL/LLaVA-NeXT.git \
    bce12e479bc4dfee2b9c50c88137b01ff51bd483
  ensure_source \
    Transformers-LLaVA \
    https://github.com/huggingface/transformers.git \
    1c39974a4c4036fd641bc1191cc32799f85715a4
  ensure_conda_env "${env_name}"
  run_in_env "${env_name}" python -m pip install \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
  run_in_env "${env_name}" python -m pip install \
    --index-url "${PACKAGE_INDEX_URL}" \
    accelerate==0.29.3 \
    einops==0.6.1 \
    einops-exts==0.0.4 \
    filelock \
    ftfy \
    huggingface-hub==0.22.2 \
    httpx==0.24.0 \
    numpy==1.26.4 \
    'packaging>=20.0' \
    'Pillow>=10.0.1' \
    pyyaml \
    regex \
    requests \
    safetensors==0.4.3 \
    sentencepiece==0.1.99 \
    shortuuid \
    tokenizers==0.15.2 \
    tqdm
  install_env_editable "${env_name}" Transformers-LLaVA
  install_env_editable "${env_name}" LLaVA-NeXT
  run_in_env "${env_name}" python -c \
    'import torch, transformers, llava; assert torch.__version__.startswith("2.1.2"); assert transformers.__version__ == "4.40.0.dev0"; print(torch.__version__, transformers.__version__)'
}

setup_geochat() {
  local env_name="vlm-geochat"

  ensure_source \
    GeoChat \
    https://github.com/mbzuai-oryx/GeoChat.git \
    4850920e005a849bd224d0ce35aa9db031fa5155
  ensure_conda_env "${env_name}"
  run_in_env "${env_name}" python -m pip install \
    torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
  run_in_env "${env_name}" python -m pip install \
    --index-url "${PACKAGE_INDEX_URL}" \
    accelerate==0.21.0 \
    einops==0.6.1 \
    einops-exts==0.0.4 \
    numpy==1.26.4 \
    Pillow \
    sentencepiece==0.1.99 \
    shortuuid \
    timm==0.6.13 \
    tokenizers==0.13.3 \
    tqdm \
    transformers==4.31.0
  install_env_editable "${env_name}" GeoChat
  run_in_env "${env_name}" python -c \
    'import torch, transformers, geochat; assert torch.__version__.startswith("2.0.1"); assert transformers.__version__ == "4.31.0"; print(torch.__version__, transformers.__version__)'
}

case "${TARGET}" in
  all)
    setup_deepseek
    setup_internvl
    setup_llava
    setup_geochat
    ;;
  deepseek) setup_deepseek ;;
  internvl) setup_internvl ;;
  llava) setup_llava ;;
  geochat) setup_geochat ;;
  *)
    echo "Usage: $0 [all|deepseek|internvl|llava|geochat]" >&2
    exit 2
    ;;
esac

echo "Installed ${TARGET} runtime; InternVL/LLaVA/GeoChat use isolated environments."
