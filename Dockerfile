# syntax=docker/dockerfile:1.6

# GPU base with CUDA 12.4; matches RunPod image family
FROM runpod/base:0.6.2-cuda12.4.1

SHELL ["/bin/bash", "-c"]
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# ---- System deps ----
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Torch stack (CUDA 12.4 channel) ----
RUN python3 -m pip install -U pip wheel && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision torchaudio

# ---- Python deps (leave Torch to the base step above) ----
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps -r /app/requirements.txt

# ---- App code ----
# src/ should include rp_handler.py and predict.py (your script)
COPY src/ /app/src/

# ---- Bundle local VAD model (optional, speeds cold starts) ----
# Keep it both in a known path and in torch cache for libs that look there.
# Place your file at repo path: models/vad/whisperx-vad-segmentation.bin
COPY models/vad/whisperx-vad-segmentation.bin /app/models/vad/whisperx-vad-segmentation.bin
RUN mkdir -p /root/.cache/torch && \
    cp /app/models/vad/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin
ENV VAD_MODEL_PATH=/app/models/vad/whisperx-vad-segmentation.bin

# ---- Pre-download WhisperX model weights (optional but recommended) ----
# Provide a lightweight builder script at builder/download_models.sh that:
# - honors $HF_TOKEN if set (for gated repos),
# - downloads faster-whisper-large-v3 into $MODEL_DIR/$WHISPERX_MODEL_NAME
#   so it matches predict.py's: whisper_arch = "./models/faster-whisper-large-v3"
# If you don't have this script, skip this block; WhisperX will download at runtime.
COPY builder/download_models.sh /app/builder/download_models.sh
RUN chmod +x /app/builder/download_models.sh
ENV MODEL_DIR=/app/models \
    WHISPERX_MODEL_NAME=faster-whisper-large-v3
# If you pass a secret at build time: DOCKER_BUILDKIT=1 docker build --secret id=hf_token,src=./hf_token.txt ...
RUN --mount=type=secret,id=hf_token \
    bash -lc 'export HF_TOKEN="$( [ -f /run/secrets/hf_token ] && cat /run/secrets/hf_token || true )"; /app/builder/download_models.sh'

# Ensure the path used in predict.py exists even if the script skipped
RUN mkdir -p /app/models/faster-whisper-large-v3
ENV WHISPERX_MODEL_DIR=/app/models/faster-whisper-large-v3

# ---- Entrypoint ----
# RunPod imports the handler; keep module import simple.
CMD ["python3", "-c", "import runpod, src.rp_handler"]