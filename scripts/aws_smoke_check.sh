#!/usr/bin/env bash
set -euo pipefail

export AINA_RUN_ROOT="${AINA_RUN_ROOT:-/home/data/aina-runs}"

./scripts/prepare_run_dirs.sh
python scripts/check_device.py
python data/clean_dataset.py --raw-dir "$AINA_RUN_ROOT/data/raw" --output-dir "$AINA_RUN_ROOT/data/clean"
python tokenizer/train_tokenizer.py --train-file "$AINA_RUN_ROOT/data/clean/train.jsonl" --output-dir "$AINA_RUN_ROOT/tokenizers/aina-nano-1m" --vocab-size 1024
python tokenizer/test_tokenizer.py --tokenizer-dir "$AINA_RUN_ROOT/tokenizers/aina-nano-1m"
python train/pretrain.py --config train/configs/aina_nano_1m.yaml
python eval/generate.py --model-dir "$AINA_RUN_ROOT/checkpoints/aina-nano-1m/pretrain" --prompt "API adalah" --max-new-tokens 80
python export/export_hf.py --checkpoint-dir "$AINA_RUN_ROOT/checkpoints/aina-nano-1m/pretrain" --output-dir "$AINA_RUN_ROOT/models/aina-nano-1m"

echo "Smoke test succeeded. You may run train/configs/aina_nano_10m.yaml next."
