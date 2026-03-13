#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <image_path> [output_root]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_PATH="$(realpath "$1")"
if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Image not found: ${IMAGE_PATH}" >&2
  exit 1
fi

IMAGE_NAME="$(basename "${IMAGE_PATH}")"
IMAGE_STEM="${IMAGE_NAME%.*}"
PATIENT_NAME="compare_${IMAGE_STEM}"
INPUT_ROOT="${ROOT_DIR}/tmp_model_compare_auto"
PATIENT_DIR="${INPUT_ROOT}/${PATIENT_NAME}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${2:-${ROOT_DIR}/out_compare_${IMAGE_STEM}_${STAMP}}"

mkdir -p "${PATIENT_DIR}" "${OUT_ROOT}"
ln -sfn "${IMAGE_PATH}" "${PATIENT_DIR}/${IMAGE_NAME}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
DEEPSEEK_MODEL_ID="${DEEPSEEK_MODEL_ID:-deepseek-ai/deepseek-vl2}"
TROCR_MODEL_ID="${TROCR_MODEL_ID:-ddobokki/ko-trocr}"
PADDLEOCR_MODEL_ID="${PADDLEOCR_MODEL_ID:-PaddlePaddle/PaddleOCR-VL-1.5}"

"${ROOT_DIR}/run_108_ocr.sh" \
  --input_root "${INPUT_ROOT}" \
  --patient_name "${PATIENT_NAME}" \
  --output_dir "${OUT_ROOT}/qwen" \
  --model_id "${QWEN_MODEL_ID}" \
  --preload_model \
  --concurrency 1 \
  --max_inflight 1

"${ROOT_DIR}/run_108_ocr.sh" \
  --input_root "${INPUT_ROOT}" \
  --patient_name "${PATIENT_NAME}" \
  --output_dir "${OUT_ROOT}/deepseek" \
  --model_id "${DEEPSEEK_MODEL_ID}" \
  --preload_model \
  --concurrency 1 \
  --max_inflight 1 \
  --max_new_tokens 1024

"${ROOT_DIR}/run_108_ocr.sh" \
  --input_root "${INPUT_ROOT}" \
  --patient_name "${PATIENT_NAME}" \
  --output_dir "${OUT_ROOT}/trocr" \
  --model_id "${TROCR_MODEL_ID}" \
  --preload_model \
  --concurrency 1 \
  --max_inflight 1 \
  --trocr_layout_mode auto \
  --trocr_batch_size 8 \
  --trocr_max_new_tokens 128

"${ROOT_DIR}/run_108_ocr.sh" \
  --input_root "${INPUT_ROOT}" \
  --patient_name "${PATIENT_NAME}" \
  --output_dir "${OUT_ROOT}/paddleocr_vl" \
  --model_id "${PADDLEOCR_MODEL_ID}" \
  --preload_model \
  --concurrency 1 \
  --max_inflight 1 \
  --max_new_tokens 1024

echo ""
echo "Comparison outputs:"
echo "  qwen:     ${OUT_ROOT}/qwen/ocr_pages/${IMAGE_STEM}.txt"
echo "  deepseek: ${OUT_ROOT}/deepseek/ocr_pages/${IMAGE_STEM}.txt"
echo "  trocr:    ${OUT_ROOT}/trocr/ocr_pages/${IMAGE_STEM}.txt"
echo "  paddle:   ${OUT_ROOT}/paddleocr_vl/ocr_pages/${IMAGE_STEM}.txt"
