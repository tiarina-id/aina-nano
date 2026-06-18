#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-../llama.cpp}
MODEL_DIR=${1:-models/aina-nano-1m}
OUTFILE=${2:-models/aina-nano-1m.gguf}

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" --outfile "$OUTFILE"
