#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv_local_vlm"
LOG_DIR="$ROOT/logs"
DATA_DIR="$ROOT/data/qwen35_digit_ft/aihub_handwriting_digits_20k"
OUT_DIR="$ROOT/outputs/qwen35_digit_lora_aihub_digits"
LOG_FILE="$LOG_DIR/qwen35_digit_lora_aihub.log"
TRAIN_MANIFEST="$DATA_DIR/train/train.jsonl"
VAL_MANIFEST="$DATA_DIR/val/val.jsonl"

mkdir -p "$LOG_DIR" "$ROOT/outputs" "$ROOT/data/qwen35_digit_ft"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing venv python: $VENV/bin/python" >&2
  exit 1
fi

if [[ " $* " != *" --train_manifest "* ]] && [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VAL_MANIFEST" ]]; then
  echo "Preparing AI-Hub handwriting digit dataset under $DATA_DIR"
  "$VENV/bin/python" "$ROOT/prepare_aihub_handwriting_digits.py" \
    --output_dir "$DATA_DIR" \
    --train_limit 20000 \
    --val_limit 2000
fi

{
  echo "[$(date '+%F %T')] Starting Qwen digit LoRA training on AI-Hub handwriting digits"
  "$VENV/bin/python" "$ROOT/train_qwen_digit_lora.py" \
    --model_id Qwen/Qwen3.5-35B-A3B \
    --train_manifest "$TRAIN_MANIFEST" \
    --val_manifest "$VAL_MANIFEST" \
    --output_dir "$OUT_DIR" \
    --logging_dir "$ROOT/logs/tensorboard/qwen35_digit_lora_aihub" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_train_samples 20000 \
    --max_val_samples 2000 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --no-gradient_checkpointing \
    --dataloader_num_workers 4 \
    --trust_remote_code \
    "$@"
} 2>&1 | tee "$LOG_FILE"
