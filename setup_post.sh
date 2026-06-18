#!/bin/bash
set -e

# Jalankan SETELAH reboot (driver GPU sudah aktif).
# Prasyarat: setup.sh sudah dijalankan (repo sudah ter-clone) dan sudo reboot sudah dilakukan.

REPO_DIR="${REPO_DIR:-$HOME/aina-nano}"
AINA_RUN_ROOT="${AINA_RUN_ROOT:-/home/data/aina-runs}"

echo "=== Verifikasi driver GPU ==="
nvidia-smi

echo "=== Siapkan folder hasil training ==="
echo "AINA_RUN_ROOT=$AINA_RUN_ROOT"

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

echo "=== Masuk ke repo ==="
cd "$REPO_DIR"

echo "=== Buat virtualenv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "=== Install dependencies ==="
pip install -r requirements.txt

echo "=== Install PyTorch GPU ==="
pip install torch

echo "=== Cek device ==="
python scripts/check_device.py

if python scripts/check_device.py | grep -q "cuda_available: True"; then
  echo ""
  echo "GPU siap."
  echo ""
  echo "Upload dataset ke:"
  echo "  $AINA_RUN_ROOT/data/raw/"
  echo ""
  echo "Sebelum training:"
  echo "  cd $REPO_DIR"
  echo "  source .venv/bin/activate"
  echo "  export AINA_RUN_ROOT=$AINA_RUN_ROOT"
  echo ""
  echo "Smoke test:"
  echo "  ./scripts/aws_smoke_check.sh"
else
  echo ""
  echo "ERROR: cuda_available: False — JANGAN lanjut training."
  echo "Cek ulang driver (nvidia-smi) dan instalasi torch."
  exit 1
fi
