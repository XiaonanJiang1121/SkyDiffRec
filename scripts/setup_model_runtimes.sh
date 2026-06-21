#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_ROOT="${PROJECT_ROOT}/third_party"
VENV_ROOT="${PROJECT_ROOT}/.venvs"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="${1:-all}"
PACKAGE_INDEX_URL="${VLM_PYPI_INDEX_URL:-https://pypi.org/simple}"

mkdir -p "${THIRD_PARTY_ROOT}" "${VENV_ROOT}"

log_step() {
  printf '\n[%s] %s\n' "$1" "$2"
}

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

ensure_shared_torch_venv() {
  local venv="$1"

  if [[ ! -x "${venv}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${venv}"
  fi
}

install_venv_editable() {
  local venv="$1"
  local source_name="$2"

  "${venv}/bin/python" -m pip install \
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

  log_step "InternVL 1/3" "Creating or reusing ${venv}"
  if [[ ! -x "${venv}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${venv}"
  fi
  log_step "InternVL 2/3" "Installing the official pinned text runtime"
  "${venv}/bin/python" -m pip install \
    --index-url "${PACKAGE_INDEX_URL}" \
    --no-deps \
    transformers==4.37.2 \
    tokenizers==0.15.1 \
    sentencepiece==0.1.99
  log_step "InternVL 3/3" "Verifying the isolated interpreter"
  "${venv}/bin/python" -c \
    'import sys, torch, tokenizers, transformers; assert transformers.__version__ == "4.37.2"; assert tokenizers.__version__ == "0.15.1"; print(sys.executable); print("torch", torch.__version__, "transformers", transformers.__version__, "tokenizers", tokenizers.__version__)'
  echo "InternVL is a project venv, so it is intentionally absent from 'conda env list'."
}

setup_llava() {
  local venv="${VENV_ROOT}/llava"

  log_step "LLaVA 1/5" "Fetching the pinned LLaVA-NeXT source"
  ensure_source \
    LLaVA-NeXT \
    https://github.com/LLaVA-VL/LLaVA-NeXT.git \
    bce12e479bc4dfee2b9c50c88137b01ff51bd483
  log_step "LLaVA 2/5" "Fetching the pinned Transformers source"
  ensure_source \
    Transformers-LLaVA \
    https://github.com/huggingface/transformers.git \
    1c39974a4c4036fd641bc1191cc32799f85715a4
  log_step "LLaVA 3/5" "Creating or reusing ${venv} with the active PyTorch/CUDA runtime"
  ensure_shared_torch_venv "${venv}"
  log_step "LLaVA 4/5" "Installing only the pinned text runtime and official source packages"
  "${venv}/bin/python" -m pip install \
    --index-url "${PACKAGE_INDEX_URL}" \
    --no-deps \
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
  install_venv_editable "${venv}" Transformers-LLaVA
  install_venv_editable "${venv}" LLaVA-NeXT
  log_step "LLaVA 5/5" "Verifying shared PyTorch and isolated Transformers/LLaVA imports"
  "${venv}/bin/python" -c \
    'import sys, torch, transformers, llava; assert transformers.__version__ == "4.40.0.dev0"; print(sys.executable); print("torch", torch.__version__, "transformers", transformers.__version__)'
}

setup_geochat() {
  local venv="${VENV_ROOT}/geochat"

  log_step "GeoChat 1/4" "Fetching the pinned GeoChat source"
  ensure_source \
    GeoChat \
    https://github.com/mbzuai-oryx/GeoChat.git \
    4850920e005a849bd224d0ce35aa9db031fa5155
  log_step "GeoChat 2/4" "Creating or reusing ${venv} with the active PyTorch/CUDA runtime"
  ensure_shared_torch_venv "${venv}"
  log_step "GeoChat 3/4" "Installing only pinned text dependencies and the official source package"
  "${venv}/bin/python" -m pip install \
    --index-url "${PACKAGE_INDEX_URL}" \
    --no-deps \
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
  install_venv_editable "${venv}" GeoChat
  log_step "GeoChat 4/4" "Verifying shared PyTorch and isolated Transformers/GeoChat imports"
  "${venv}/bin/python" -c \
    'import sys, torch, transformers, geochat; assert transformers.__version__ == "4.31.0"; print(sys.executable); print("torch", torch.__version__, "transformers", transformers.__version__)'
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

echo "Installed ${TARGET} runtime; InternVL/LLaVA/GeoChat use lightweight venvs that share the active PyTorch/CUDA installation."
