#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_ROOT="${PROJECT_ROOT}/third_party"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="${1:-all}"

mkdir -p "${THIRD_PARTY_ROOT}"

install_source_runtime() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local destination="${THIRD_PARTY_ROOT}/${name}"

  if [[ ! -d "${destination}/.git" ]]; then
    git clone "${url}" "${destination}"
  fi
  git -C "${destination}" fetch origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
  "${PYTHON_BIN}" -m pip install --no-deps -e "${destination}"
}

setup_deepseek() {
  "${PYTHON_BIN}" -m pip install attrdict
  install_source_runtime \
    DeepSeek-VL \
    https://github.com/deepseek-ai/DeepSeek-VL.git \
    681bffb4519856ad27cc17531aacde31ddf6f1a7
}

setup_llava() {
  "${PYTHON_BIN}" -m pip install einops-exts ftfy shortuuid
  install_source_runtime \
    LLaVA-NeXT \
    https://github.com/LLaVA-VL/LLaVA-NeXT.git \
    bce12e479bc4dfee2b9c50c88137b01ff51bd483
}

setup_geochat() {
  "${PYTHON_BIN}" -m pip install einops-exts shortuuid
  install_source_runtime \
    GeoChat \
    https://github.com/mbzuai-oryx/GeoChat.git \
    4850920e005a849bd224d0ce35aa9db031fa5155
}

case "${TARGET}" in
  all)
    setup_deepseek
    setup_llava
    setup_geochat
    ;;
  deepseek) setup_deepseek ;;
  llava) setup_llava ;;
  geochat) setup_geochat ;;
  *)
    echo "Usage: $0 [all|deepseek|llava|geochat]" >&2
    exit 2
    ;;
esac

echo "Installed ${TARGET} model runtime without changing torch or transformers."
