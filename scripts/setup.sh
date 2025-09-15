#!/usr/bin/env bash
set -euo pipefail

SETUP_VERSION=3
echo "== Codex setup v$SETUP_VERSION =="

python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# CPU PyTorch for the sandbox (no CUDA wheels)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Your deps (don’t repin torch)
pip install -r builder/requirements.txt --no-deps

# Optional: ffmpeg for audio tooling
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y ffmpeg
fi

echo "$SETUP_VERSION" > .codex_setup_version