# install uv user-space Python manager
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# fresh venv on Python 3.11 (wheels exist for faster-whisper)
uv python install 3.11
uv venv -p 3.11 .venv
source .venv/bin/activate

pip install -U pip wheel
# CPU torch for the sandbox
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# your project deps (no torch re-pin)
pip install -r builder/requirements.txt --no-deps

# NLTK tokenizers for alignment
python - <<'PY'
import nltk
for pkg in ("punkt","punkt_tab"):
    try: nltk.data.find(f"tokenizers/{pkg}")
    except LookupError: nltk.download(pkg, quiet=True)
PY

# optional: ffmpeg if available
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y ffmpeg
fi