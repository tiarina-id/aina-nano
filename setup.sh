#!/bin/bash
set -e

DISK="/dev/nvme1n1"
MOUNT_POINT="/data"
RUN_DATA_LINK="/home/data"
MODEL_DIR="/data/models"

# Repo (pakai HTTPS biar jalan tanpa setup SSH di server baru).
# Kalau SSH GitHub sudah diset, ganti ke: git@github.com:tiarina-id/aina-nano.git
REPO_URL="https://github.com/tiarina-id/aina-nano.git"
REPO_DIR="$HOME/aina-nano"

echo "=== Update system ==="
sudo apt update
sudo apt upgrade -y

echo "=== Install pciutils (required for GPU detection) ==="
sudo apt install -y pciutils

echo "=== Install NVIDIA driver ==="
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

GPU_ARCH=$(lspci -n | grep "10de:" | head -1 | awk '{print $3}' | cut -d: -f2 | cut -c1-2)
echo "GPU PCI prefix detected: $GPU_ARCH"

if [[ "$GPU_ARCH" =~ ^1[5-9a-f]$ ]]; then
  echo "Legacy GPU (Pascal/Volta) — installing proprietary driver"
  sudo ubuntu-drivers install
else
  echo "Modern GPU (Turing+) — installing open kernel module"
  sudo apt install -y nvidia-open
fi

rm -f cuda-keyring_1.1-1_all.deb

echo "=== Install packages ==="
sudo apt install -y \
  curl \
  wget \
  git \
  jq \
  nginx \
  ufw \
  nvtop \
  fastfetch \
  certbot \
  python3-certbot-nginx \
  python3-venv \
  python3-pip \
  tmux \
  htop \
  build-essential

echo "=== Mount data disk ==="
sudo mkdir -p $MOUNT_POINT

UUID=$(sudo blkid -s UUID -o value $DISK)

grep -q "$UUID" /etc/fstab || \
echo "UUID=$UUID $MOUNT_POINT ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

sudo systemctl daemon-reload
sudo mount -a
sudo chown -R "$USER:$USER" $MOUNT_POINT

echo "=== Link /home/data ke $MOUNT_POINT ==="
if [ ! -e "$RUN_DATA_LINK" ]; then
  sudo ln -s "$MOUNT_POINT" "$RUN_DATA_LINK"
fi

df -h

echo "=== Install Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Configure Ollama (models di $MODEL_DIR) ==="
sudo mkdir -p "$MODEL_DIR"
sudo chown -R ollama:ollama "$MODEL_DIR"

sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<EOF
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_MODELS=$MODEL_DIR"
EOF

sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl restart ollama

echo "=== Clone repo ==="
if [ -d "$REPO_DIR/.git" ]; then
  echo "Repo sudah ada di $REPO_DIR — skip clone"
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

echo ""
echo "IMPORTANT: Reboot sekarang agar driver GPU aktif:"
echo "  sudo reboot"
echo ""
echo "Setelah reboot, verifikasi GPU lalu jalankan setup tahap 2:"
echo "  nvidia-smi"
echo "  cd $REPO_DIR"
echo "  ./setup_post.sh"
