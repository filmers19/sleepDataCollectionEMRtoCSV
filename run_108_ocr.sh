#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="Qwen/Qwen3-VL-30B-A3B-Instruct"
ARGS=("$@")

for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--model_id" ]] && (( i + 1 < ${#ARGS[@]} )); then
    MODEL_ID="${ARGS[$((i + 1))]}"
  fi
done

MODEL_ID_L="$(printf '%s' "${MODEL_ID}" | tr '[:upper:]' '[:lower:]')"
if [[ "${MODEL_ID_L}" == *"deepseek-vl2"* ]]; then
  PYTHON_BIN="${ROOT_DIR}/.conda_envs/deepseek_vl2/bin/python"
elif [[ "${MODEL_ID_L}" == *"paddleocr-vl"* ]]; then
  PYTHON_BIN="${ROOT_DIR}/.conda_envs/paddleocr_vl/bin/python"
else
  PYTHON_BIN="${ROOT_DIR}/.venv_local_vlm/bin/python"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Using python: ${PYTHON_BIN}"
exec "${PYTHON_BIN}" "${ROOT_DIR}/108_ocr_only_patient_local_qwen.py" "${ARGS[@]}"
