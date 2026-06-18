#!/usr/bin/env bash
set -euo pipefail

python scripts/check_device.py
python data/clean_dataset.py
python tokenizer/train_tokenizer.py --vocab-size 8000
python tokenizer/test_tokenizer.py
python train/pretrain.py --config train/configs/aina_nano_1m.yaml
python eval/generate.py --model-dir checkpoints/aina-nano-1m/pretrain --prompt "API adalah" --max-new-tokens 80
python export/export_hf.py --checkpoint-dir checkpoints/aina-nano-1m/pretrain --output-dir models/aina-nano-1m

echo "Smoke test succeeded. You may run train/configs/aina_nano_10m.yaml next."
