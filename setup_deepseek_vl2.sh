#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/yhlee/miniconda3}"
ENV_DIR="${ENV_DIR:-${ROOT_DIR}/.conda_envs/deepseek_vl2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "conda.sh not found under ${CONDA_BASE}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -d "${ENV_DIR}" ]]; then
  conda create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_DIR}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade -r "${ROOT_DIR}/requirements_deepseek_vl2.txt"

echo ""
echo "DeepSeek-VL2 environment is ready."
echo "Python: ${ENV_DIR}/bin/python"
echo "Activate with: conda activate ${ENV_DIR}"
echo "Test with: ${ENV_DIR}/bin/python ${ROOT_DIR}/108_ocr_only_patient_local_qwen.py --model_id deepseek-ai/deepseek-vl2 --patient_name Patient_10"
