#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE=${1:-10m}
MODEL_NAME=${2:-aina-nano-0.01b-base}
export AINA_RUN_ROOT="${AINA_RUN_ROOT:-/home/data/aina-runs}"

if [[ "$MODEL_SIZE" == "1m" ]]; then
  MODEL_CONFIG="model/config_1m.json"
  TRAIN_CONFIG="train/configs/aina_nano_1m.yaml"
  CHECKPOINT_DIR="$AINA_RUN_ROOT/checkpoints/aina-nano-1m/pretrain"
  TOKENIZER_DIR="$AINA_RUN_ROOT/tokenizers/aina-nano-1m"
  DEFAULT_OUT="$AINA_RUN_ROOT/models/aina-nano-1m-base"
elif [[ "$MODEL_SIZE" == "10m" ]]; then
  MODEL_CONFIG="model/config_10m.json"
  TRAIN_CONFIG="train/configs/aina_nano_10m.yaml"
  CHECKPOINT_DIR="$AINA_RUN_ROOT/checkpoints/aina-nano-10m/pretrain"
  TOKENIZER_DIR="$AINA_RUN_ROOT/tokenizers/aina-nano-10m"
  DEFAULT_OUT="$AINA_RUN_ROOT/models/aina-nano-10m-base"
else
  echo "MODEL_SIZE must be 1m or 10m" >&2
  exit 1
fi

OUT_DIR=${3:-$DEFAULT_OUT}
REPORT_DIR="$AINA_RUN_ROOT/reports/${MODEL_NAME}"
mkdir -p "$REPORT_DIR"

python scripts/count_params.py \
  --config "$MODEL_CONFIG" \
  --output "$REPORT_DIR/params.json"

python scripts/count_dataset_tokens.py \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --input "$AINA_RUN_ROOT/data/clean/train.jsonl" \
  --output "$REPORT_DIR/train_tokens.json"

python eval/eval_prompts.py \
  --model-dir "$CHECKPOINT_DIR" \
  --prompts eval/prompts.txt \
  --output "$REPORT_DIR/eval_prompts.jsonl" \
  --max-new-tokens 120

python scripts/write_release_manifest.py \
  --model-name "$MODEL_NAME" \
  --model-config "$MODEL_CONFIG" \
  --train-config "$TRAIN_CONFIG" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --dataset-report "$REPORT_DIR/train_tokens.json" \
  --param-report "$REPORT_DIR/params.json" \
  --eval-report "$REPORT_DIR/eval_prompts.jsonl" \
  --output "$REPORT_DIR/manifest.json"

python export/export_hf.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-dir "$OUT_DIR" \
  --model-name "$MODEL_NAME" \
  --metadata "$REPORT_DIR/manifest.json"

echo "Release HF artifact ready: $OUT_DIR"
echo "Release reports ready: $REPORT_DIR"
echo "Optional GGUF: ./export/convert_gguf.sh $OUT_DIR $AINA_RUN_ROOT/gguf/${MODEL_NAME}.gguf"
