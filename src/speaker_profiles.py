# speaker_profiles.py
import os
import tempfile
import requests
import numpy as np
import torch
import librosa
from scipy.spatial.distance import cdist
from pyannote.audio import Inference

# Device selection
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face token (needed for gated pyannote models)
_HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None

# Pyannote speaker embedding model (512-D)
_EMBED = Inference(
    "pyannote/embedding",
    device=_DEVICE,
    use_auth_token=_HF_TOKEN
)

# In-memory cache: name -> 512-D normalized vector
_CACHE: dict[str, np.ndarray] = {}

def _l2(x: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    n = np.linalg.norm(x)
    return x if n == 0 else (x / n)

def load_embeddings(profiles):
    """
    Load and cache 512-D speaker embeddings for given profiles.

    profiles: [{"name": "alice", "url": "https://.../alice.wav"}, ...]

    returns: {"alice": np.ndarray(shape=(512,)), ...}
    """
    out = {}
    for p in profiles:
        name = p["name"]
        url  = p["url"]
        if name in _CACHE:
            out[name] = _CACHE[name]
            continue

        # Download to a temp WAV/MP3 and compute embedding
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            tmp.write(resp.content)
            tmp.flush()

            wav, _sr = librosa.load(tmp.name, sr=16_000, mono=True)
            with torch.no_grad():
                emb = _EMBED(torch.tensor(wav).unsqueeze(0)).cpu().numpy().reshape(-1)  # 512-D
            emb = _l2(emb)
            _CACHE[name] = emb
            out[name] = emb
    return out

def relabel(diarize_df, transcription, embeds, threshold=0.75):
    """
    Replace diarization labels in 'transcription' with closest profile name.

    diarize_df:    pd.DataFrame from a DiarizationPipeline (unused here but kept for API parity)
    transcription: dict with a 'segments' list (WhisperX output); each segment may have:
                   - 'speaker': diarization label like 'SPEAKER_00'
                   - 'words': list of words with optional 'speaker' and 'embedding'
    embeds:        dict of known embeddings {"alice": vec512, ...}
    threshold:     cosine similarity threshold (0..1)

    returns: updated transcription
    """
    if not embeds:
        return transcription

    names = list(embeds.keys())
    vecstack = np.stack(list(embeds.values()))  # (N, 512)

    segs = transcription.get("segments") or []
    for seg in segs:
        dia_spk = seg.get("speaker")
        if not dia_spk:
            continue

        # Gather embeddings from words that belong to the diarized speaker
        word_vecs = [
            w.get("embedding")
            for w in seg.get("words", [])
            if w.get("speaker") == dia_spk and w.get("embedding") is not None
        ]
        if not word_vecs:
            continue

        centroid = np.mean(word_vecs, axis=0, keepdims=True)  # (1, 512)
        sim = 1.0 - cdist(centroid, vecstack, metric="cosine")  # similarity = 1 - distance
        best_idx = int(sim.argmax())
        best_sim = float(sim[0, best_idx])
        if best_sim >= threshold:
            real = names[best_idx]
            seg["speaker"] = real
            seg["similarity"] = best_sim
            for w in seg.get("words", []):
                # overwrite diarization label with verified identity
                if w.get("speaker") == dia_spk:
                    w["speaker"] = real
    return transcription