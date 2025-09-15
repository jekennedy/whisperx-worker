import os, pathlib, sys

for p in [os.getenv("HF_HOME"),
          os.getenv("HUGGINGFACE_HUB_CACHE"),
          os.getenv("TRANSFORMERS_CACHE"),
          os.getenv("TORCH_HOME")]:
    if p:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# Optional: sanity log
print("[INIT] Caches:",
      "HF_HOME=", os.getenv("HF_HOME"),
      "TRANSFORMERS_CACHE=", os.getenv("TRANSFORMERS_CACHE"),
      "TORCH_HOME=", os.getenv("TORCH_HOME"),
      flush=True)

# hand off to RunPod serverless loader (imports rp_handler)
import runpod, src.rp_handler  # noqa: F401