import os, pathlib, sys
import runpod
from . import rp_handler

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

# Start serverless handler (clean separation from rp_handler import)
runpod.serverless.start({"handler": rp_handler.run})
