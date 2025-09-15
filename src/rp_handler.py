# src/rp_handler.py

from dotenv import load_dotenv, find_dotenv
import os
import sys
import json
import shutil
import logging
from urllib.parse import urlparse

import boto3
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup

from src.rp_schema import INPUT_VALIDATIONS
from src.predict import Predictor, Output

# Optional HF login for diarization or gated models
from huggingface_hub import login, whoami
import torch
import numpy as np

# Speaker helpers will be imported lazily inside functions to avoid heavy deps during module import

# -----------------------------------------------------------------------------
# Env and logging
# -----------------------------------------------------------------------------
load_dotenv(find_dotenv())

# If a bundled VAD model is present, advertise it to downstream code
VAD_MODEL_PATH = os.getenv("VAD_MODEL_PATH")
if VAD_MODEL_PATH and os.path.isfile(VAD_MODEL_PATH):
    os.environ["VAD_MODEL_PATH"] = VAD_MODEL_PATH

logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if VAD_MODEL_PATH and os.path.isfile(VAD_MODEL_PATH):
    logger.info(f"Using bundled VAD model: {VAD_MODEL_PATH}")


# -----------------------------------------------------------------------------
# Hugging Face auth (optional)
# -----------------------------------------------------------------------------
raw_token = os.environ.get("HF_TOKEN", "").strip()
hf_token = raw_token
if hf_token and not hf_token.startswith("hf_"):
    logger.warning("HF_TOKEN does not start with 'hf_'; leaving as-is but authentication may fail.")

if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face authenticated as: {user.get('name') or user.get('email')}")
    except Exception:
        logger.error("Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.info("No HF_TOKEN provided. Diarization with gated models may fail if required.")

# -----------------------------------------------------------------------------
# S3 / R2 client from environment for artifact uploads
# -----------------------------------------------------------------------------
STORAGE_ENDPOINT = os.environ.get("STORAGE_ENDPOINT")
STORAGE_BUCKET   = os.environ.get("STORAGE_BUCKET")
STORAGE_ACCESS   = os.environ.get("STORAGE_ACCESS_KEY")
STORAGE_SECRET   = os.environ.get("STORAGE_SECRET_KEY")
PREFIX_TRANSCRIPTS     = os.environ.get("PREFIX_TRANSCRIPTS", "transcripts/")

_s3 = None
if all([STORAGE_ENDPOINT, STORAGE_BUCKET, STORAGE_ACCESS, STORAGE_SECRET]):
    try:
        _s3 = boto3.client(
            "s3",
            endpoint_url=STORAGE_ENDPOINT,
            aws_access_key_id=STORAGE_ACCESS,
            aws_secret_access_key=STORAGE_SECRET,
        )
        logger.info("Initialized S3/R2 client for artifact uploads.")
    except Exception:
        logger.error("Failed to initialize S3/R2 client. Will return inline results only.", exc_info=True)
        _s3 = None
else:
    logger.info("S3/R2 env not fully set. Will return inline results only.")

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
MODEL = Predictor()
MODEL.setup()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def cleanup_job_files(job_id, jobs_directory="/jobs"):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception:
            logger.error(f"Error removing job directory {job_path}", exc_info=True)
    else:
        logger.debug(f"Job directory not found: {job_path}")

def _stem_from(path_or_url: str) -> str:
    try:
        p = urlparse(path_or_url)
        base = os.path.basename(p.path)
    except Exception:
        base = os.path.basename(path_or_url)
    return os.path.splitext(base)[0]

def _put_text(key: str, text: str, content_type: str):
    _s3.put_object(
        Bucket=STORAGE_BUCKET,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType=content_type,
    )

def _put_json(key: str, obj):
    _s3.put_object(
        Bucket=STORAGE_BUCKET,
        Key=key,
        Body=json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

# -----------------------------------------------------------------------------
# Main handler
# -----------------------------------------------------------------------------
def run(job):
    job_id    = job["id"]
    job_input = job["input"]

    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    # 1) Download primary audio
    try:
        audio_file_path = download_files_from_urls(job_id, [job_input["audio_file"]])[0]
        logger.debug(f"Audio downloaded -> {audio_file_path}")
    except Exception as e:
        logger.error("Audio download failed", exc_info=True)
        return {"error": f"audio download: {e}"}

    # 2) Speaker profiles optional
    speaker_profiles = job_input.get("speaker_samples", [])
    embeddings = {}
    if speaker_profiles:
        # Lazy import heavy modules only if needed
        from src.speaker_processing import (
            load_known_speakers_from_samples,
            identify_speakers_on_segments,
            relabel_speakers_by_avg_similarity,
        )
        try:
            embeddings = load_known_speakers_from_samples(
                speaker_profiles,
                huggingface_access_token=job_input.get("huggingface_access_token") or hf_token
            )
            logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        except Exception as e:
            logger.error("Enrollment failed", exc_info=True)

    # 3) WhisperX / diarization
    predict_input = {
        "audio_file": audio_file_path,
        "language": job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt": job_input.get("initial_prompt"),
        "batch_size": job_input.get("batch_size", 64),
        "temperature": job_input.get("temperature", 0),
        "vad_onset": job_input.get("vad_onset", 0.50),
        "vad_offset": job_input.get("vad_offset", 0.363),
        "align_output": job_input.get("align_output", False),
        "diarization": job_input.get("diarization", False),
        "huggingface_access_token": job_input.get("huggingface_access_token") or hf_token,
        "min_speakers": job_input.get("min_speakers"),
        "max_speakers": job_input.get("max_speakers"),
        "debug": job_input.get("debug", False),
    }

    try:
        result: Output = MODEL.predict(**predict_input)
    except Exception as e:
        logger.error("WhisperX prediction failed", exc_info=True)
        return {"error": f"prediction: {e}"}

    output_dict = {
        "segments": result.segments,
        "detected_language": getattr(result, "detected_language", None),
    }

    # 4) Optional speaker identification on segments
    if embeddings:
        # Import here as well to keep top-level import light
        from src.speaker_processing import identify_speakers_on_segments, relabel_speakers_by_avg_similarity
        try:
            segments_with_speakers = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=0.1,
            )
            segments_with_final_labels = relabel_speakers_by_avg_similarity(segments_with_speakers)
            output_dict["segments"] = segments_with_final_labels
            logger.info("Speaker identification completed.")
        except Exception:
            logger.error("Speaker identification failed", exc_info=True)

    # 5) Upload artifacts to R2 when configured, else inline minimal results
    artifacts = {}
    stem = _stem_from(job_input.get("audio_file") or audio_file_path)

    if _s3:
        try:
            seg_key = f"{PREFIX_TRANSCRIPTS}{stem}.segments.json"
            _put_json(seg_key, output_dict["segments"])
            artifacts["segments_key"] = seg_key
        except Exception:
            logger.error("Failed uploading segments JSON", exc_info=True)

        srt_text = getattr(result, "srt", None)
        if isinstance(srt_text, str) and srt_text.strip():
            try:
                srt_key = f"{PREFIX_TRANSCRIPTS}{stem}.srt"
                _put_text(srt_key, srt_text, "application/x-subrip")
                artifacts["srt_key"] = srt_key
            except Exception:
                logger.error("Failed uploading SRT", exc_info=True)

        vtt_text = getattr(result, "vtt", None)
        if isinstance(vtt_text, str) and vtt_text.strip():
            try:
                vtt_key = f"{PREFIX_TRANSCRIPTS}{stem}.vtt"
                _put_text(vtt_key, vtt_text, "text/vtt")
                artifacts["vtt_key"] = vtt_key
            except Exception:
                logger.error("Failed uploading VTT", exc_info=True)

    small_output = {
        "detected_language": output_dict.get("detected_language"),
    }
    if artifacts:
        small_output.update(artifacts)
    else:
        # Fallback when S3 is not configured
        segs = output_dict.get("segments")
        if isinstance(segs, list):
            small_output["segments"] = segs[:50]  # small slice to avoid large responses
        else:
            small_output["segments"] = segs

    # 6) Cleanup and return through RunPod helper
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception:
        logger.warning("Cleanup issue", exc_info=True)

    return {"output": small_output}

runpod.serverless.start({"handler": run})
