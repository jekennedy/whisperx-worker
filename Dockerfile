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
COPY builder/requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps -r /app/requirements.txt

# ---- App code ----
# src/ should include rp_handler.py and predict.py (your script)
COPY src/ /app/src/

# ---- Bundle local models (optional, speeds cold starts) ----
# Copy whatever is present under repo ./models into image ./models.
# If a VAD binary is present at ./models/whisperx-vad-segmentation.bin or ./models/vad/..., place it under ./models/vad.
COPY models/ /app/models/
RUN mkdir -p /app/models/vad /root/.cache/torch && \
    if [ -f /app/models/whisperx-vad-segmentation.bin ]; then \
      mv /app/models/whisperx-vad-segmentation.bin /app/models/vad/whisperx-vad-segmentation.bin; \
    fi && \
    if [ -f /app/models/vad/whisperx-vad-segmentation.bin ]; then \
      cp /app/models/vad/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin; \
    fi
ENV VAD_MODEL_PATH=/app/models/vad/whisperx-vad-segmentation.bin

# ---- Pre-download WhisperX model weights (optional but recommended) ----
# Provide a lightweight builder script at builder/download_models.sh that:
# - honors $HF_TOKEN if set (for gated repos),
# - downloads faster-whisper-large-v3 into $MODEL_DIR/$WHISPERX_MODEL_NAME
#   so it matches predict.py's: whisper_arch = "./models/faster-whisper-large-v3"
# If you don't have this script, skip this block; WhisperX will download at runtime.
COPY builder/download_models.sh /app/builder/download_models.sh
RUN chmod +x /app/builder/download_models.sh
ARG PRELOAD_MODELS=0
ENV MODEL_DIR=/app/models \
    WHISPERX_MODEL_NAME=faster-whisper-large-v3
# Enable optional preloading with BuildKit cache mounts; off by default.
# Pass HF token at build with: --secret id=hf_token,src=./hf_token.txt
RUN --mount=type=secret,id=hf_token \
    --mount=type=cache,target=/cache/models \
    --mount=type=cache,target=/cache/hf \
    bash -lc '\
      if [[ "$PRELOAD_MODELS" == "1" ]]; then \
        export HF_TOKEN="$( [ -f /run/secrets/hf_token ] && cat /run/secrets/hf_token || true )"; \
        export HF_HOME=/cache/hf; \
        /app/builder/download_models.sh; \
      else \
        echo "Skipping model pre-download (PRELOAD_MODELS=0)"; \
      fi'

# Ensure the path used in predict.py exists even if the script skipped
RUN mkdir -p /app/models/faster-whisper-large-v3
ENV WHISPERX_MODEL_DIR=/app/models/faster-whisper-large-v3

# ---- Entrypoint ----
# RunPod imports the handler; keep module import simple.
CMD ["python3", "-c", "import runpod, src.rp_handler"]
