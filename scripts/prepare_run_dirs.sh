#!/usr/bin/env bash
set -euo pipefail

export AINA_RUN_ROOT="${AINA_RUN_ROOT:-/home/data/aina-runs}"

mkdir -p \
  "$AINA_RUN_ROOT/data/raw" \
  "$AINA_RUN_ROOT/data/clean" \
  "$AINA_RUN_ROOT/tokenizers/aina-nano-1m" \
  "$AINA_RUN_ROOT/tokenizers/aina-nano-10m" \
  "$AINA_RUN_ROOT/checkpoints" \
  "$AINA_RUN_ROOT/models" \
  "$AINA_RUN_ROOT/gguf" \
  "$AINA_RUN_ROOT/reports" \
  "$AINA_RUN_ROOT/logs"

echo "AINA_RUN_ROOT=$AINA_RUN_ROOT"
find "$AINA_RUN_ROOT" -maxdepth 2 -type d | sort
