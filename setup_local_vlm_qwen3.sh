#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv_local_vlm}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# CUDA 12.8 wheels are appropriate for modern NVIDIA Linux drivers (e.g. Blackwell).
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -m pip install --upgrade -r requirements_local_vlm_qwen3.txt

echo ""
echo "Environment is ready."
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Run dry-run: python 106_paper_to_cdm_SA_live_local_vlm.py --dry_run"
