#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# CPU PyTorch for the sandbox (don’t pull CUDA wheels)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Your deps (don’t repin torch)
pip install -r builder/requirements.txt --no-deps

# Optional: ffmpeg for audio tooling
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y ffmpeg
fi