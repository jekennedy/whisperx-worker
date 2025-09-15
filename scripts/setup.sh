#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r builder/requirements.txt --no-deps
# Optional: pre-download models if you want faster runs and HF_TOKEN is provided
if [[ -n "${HF_TOKEN:-}" && -x builder/download_models.sh ]]; then
  ./builder/download_models.sh
fi