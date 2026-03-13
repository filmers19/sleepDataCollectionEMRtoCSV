#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$ROOT/logs/qwen35_digit_lora.log"

tmux new-session -d -s qwenft "bash -lc 'tail -f \"$LOG_FILE\"'"
tmux split-window -t qwenft:0 -v "watch -n 2 nvidia-smi"
tmux split-window -t qwenft:0 -h "watch -n 5 'ls -lah \"$ROOT/outputs/qwen35_digit_lora\"'"
tmux select-layout -t qwenft:0 tiled
tmux attach -t qwenft
