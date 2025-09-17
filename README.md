[![RunPod](https://api.runpod.io/badge/kodxana/whisperx-worker)](https://www.runpod.io/console/hub/kodxana/whisperx-worker)

# WhisperX Worker for RunPod

A serverless worker that provides high-quality speech transcription with timestamp alignment and speaker diarization using WhisperX on the RunPod platform.

## Features

- Automatic speech transcription with WhisperX
- Automatic language detection
- Word-level timestamp alignment
- Speaker diarization (optional)
- Highly parallelized batch processing
- Voice activity detection with configurable parameters
- RunPod serverless compatibility

## Input Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio_file` | string | Yes | N/A | URL to the audio file for transcription |
| `language` | string | No | `null` | ISO code of the language spoken in the audio (e.g., 'en', 'fr'). If not specified, automatic detection will be performed |
| `language_detection_min_prob` | float | No | `0` | Minimum probability threshold for language detection |
| `language_detection_max_tries` | int | No | `5` | Maximum number of attempts for language detection |
| `initial_prompt` | string | No | `null` | Optional text to provide as a prompt for the first transcription window |
| `batch_size` | int | No | `64` | Batch size for parallelized input audio transcription |
| `temperature` | float | No | `0` | Temperature to use for sampling (higher = more random) |
| `vad_onset` | float | No | `0.500` | Voice Activity Detection onset threshold |
| `vad_offset` | float | No | `0.363` | Voice Activity Detection offset threshold |
| `align_output` | bool | No | `false` | Whether to align Whisper output for accurate word-level timestamps |
| `diarization` | bool | No | `false` | Whether to assign speaker ID labels to segments |
| `huggingface_access_token` | string | No* | `null` | HuggingFace token for diarization model access (*Required if diarization is enabled) |
| `min_speakers` | int | No | `null` | Minimum number of speakers (only applicable if diarization is enabled) |
| `max_speakers` | int | No | `null` | Maximum number of speakers (only applicable if diarization is enabled) |
| `debug` | bool | No | `false` | Whether to print compute/inference times and memory usage information |
| `speaker_samples` | list | No | `[]` | List of speaker sample objects for speaker diarization |

## Usage Examples

### Basic Transcription

```json
{
  "input": {
    "audio_file": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"
  }
}
```

### Transcription with Language Detection and Alignment

```json
{
  "input": {
    "audio_file": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "align_output": true,
    "batch_size": 32,
    "debug": true
  }
}
```

### Full Configuration with Diarization

```json
{
  "input": {
    "audio_file": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "language": "en",
    "batch_size": 32,
    "temperature": 0.2,
    "align_output": true,
    "diarization": true,
    "huggingface_access_token": "YOUR_HUGGINGFACE_TOKEN",
    "min_speakers": 2,
    "max_speakers": 5,
    "debug": true
  }
}
```
### Full Configuration with Speaker Verification. There is no limit to the number of voice you can upload,  but precision maybe be reduced over a certain threshold
```json
  "input": {
    "audio_file": "https://example.com/audio/sample.mp3",
    "language": "en",
    "batch_size": 32,
    "temperature": 0.2,
    "align_output": true,
    "diarization": true,
    "huggingface_access_token": "YOUR_HUGGINGFACE_TOKEN",
    "min_speakers": 2,
    "max_speakers": 5,
    "debug": true,
    "speaker_verification": true,
    "speaker_samples": [
      {
        "name": "Speaker1",
        "url": "https://example.com/speaker1.wav"
      },
      {
        "name": "Speaker2",
        "url": "https://example.com/speaker2.wav"
      },
      {
        "name": "Speaker3",
        "url": "https://example.com/speaker3.wav"
      }
      ...
    ]
  }
}

### Speaker Verification Tips (presigned URLs and naming)

- URLs: Provide HTTP(S) URLs for `speaker_samples`. If your samples live in S3/R2, presign them first and pass the HTTPS link. Example presign (RunPod S3 API):

```bash
aws --endpoint-url https://s3api-<region>.runpod.io \
    s3 presign s3://<bucket>/speaker-samples/swami-1.m4a --expires-in 86400
```

- Naming: If you omit `name`, the worker derives it from the sample URL in this order:
  - Query parameter `?name=swami`
  - Fragment `#swami`
  - Filename stem (e.g., `swami-1`)

- Example with derived names in URLs:

```json
{
  "input": {
    "audio_file": "https://…/minute.m4a",
    "diarization": true,
    "speaker_verification": true,
    "speaker_samples": [
      { "url": "https://…/swami-sample-1.m4a?name=swami" },
      { "url": "https://…/swami-sample-2.m4a#swami" }
    ]
  }
}
```

The worker enrolls each sample and relabels diarized segments to the most likely enrolled speaker name.
## Output Format

The service returns a JSON object structured as follows:

### Without Diarization

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Transcribed text segment 1",
      "words": [
        {"word": "Transcribed", "start": 0.1, "end": 0.7},
        {"word": "text", "start": 0.8, "end": 1.2},
        {"word": "segment", "start": 1.3, "end": 1.9},
        {"word": "1", "start": 2.0, "end": 2.4}
      ]
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "Transcribed text segment 2",
      "words": [
        {"word": "Transcribed", "start": 2.6, "end": 3.2},
        {"word": "text", "start": 3.3, "end": 3.7},
        {"word": "segment", "start": 3.8, "end": 4.4},
        {"word": "2", "start": 4.5, "end": 4.9}
      ]
    }
  ],
  "detected_language": "en",
  "language_probability": 0.997
}
```

### With Diarization

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Transcribed text segment 1",
      "words": [
        {"word": "Transcribed", "start": 0.1, "end": 0.7, "speaker": "SPEAKER_01"},
        {"word": "text", "start": 0.8, "end": 1.2, "speaker": "SPEAKER_01"},
        {"word": "segment", "start": 1.3, "end": 1.9, "speaker": "SPEAKER_01"},
        {"word": "1", "start": 2.0, "end": 2.4, "speaker": "SPEAKER_01"}
      ],
      "speaker": "SPEAKER_01"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "Transcribed text segment 2",
      "words": [
        {"word": "Transcribed", "start": 2.6, "end": 3.2, "speaker": "SPEAKER_02"},
        {"word": "text", "start": 3.3, "end": 3.7, "speaker": "SPEAKER_02"},
        {"word": "segment", "start": 3.8, "end": 4.4, "speaker": "SPEAKER_02"},
        {"word": "2", "start": 4.5, "end": 4.9, "speaker": "SPEAKER_02"}
      ],
      "speaker": "SPEAKER_02"
    }
  ],
  "detected_language": "en",
  "language_probability": 0.997,
  "speakers": {
    "SPEAKER_01": {"name": "Speaker 1", "time": 2.5},
    "SPEAKER_02": {"name": "Speaker 2", "time": 2.5}
  }
}
```

## Performance Considerations

- **GPU Memory**: Adjust `batch_size` based on available GPU memory for optimal performance
- **Processing Time**: Enabling diarization and alignment will increase processing time
- **File Size**: Large audio files may require more processing time and resources
- **Language Detection**: For shorter audio clips, language detection may be less accurate

## Troubleshooting

### Common Issues

1. **"Model was trained with pyannote.audio 0.0.1, yours is X.X.X"**
   - This is a warning only and shouldn't affect functionality in most cases
   - If issues persist, consider downgrading pyannote.audio

2. **Diarization failures**
   - Ensure you're providing a valid HuggingFace access token
   - Try specifying reasonable min/max speaker values

## Development and Deployment

### Building Your Own Image

```bash
docker build -t your-username/whisperx-worker:your-tag .
```

### Build Options (models and caching)

- Optional preloading (default off):
  - The Dockerfile supports baking model weights into the image, but this is disabled by default for fast builds and smaller images.
  - To enable preloading and speed up cold starts, pass a build arg and (optionally) an HF token secret.

- Fast build (no preloading):
  - `DOCKER_BUILDKIT=1 docker build -t your/worker:fast .`

- Preload models (uses BuildKit cache, repeat builds are faster):
  - `DOCKER_BUILDKIT=1 docker build \`
    `  --build-arg PRELOAD_MODELS=1 \`
    `  --secret id=hf_token,src=./hf_token.txt \`
    `  -t your/worker:preloaded .`

- Notes:
  - BuildKit cache mounts keep downloads out of final layers while reusing them across builds.
  - If you do not preload, the first runtime call will download models automatically.

### Runtime Model Selection

- You can select a smaller model at runtime for quick validation without rebuilding:
  - Set env `WHISPERX_MODEL=small` (or `medium`, `large-v3`, etc.).
  - If unset, the worker uses any preloaded local path (`WHISPERX_MODEL_DIR`, default `/app/models/faster-whisper-large-v3`).

### Runtime Env Vars

- Model selection
  - `WHISPERX_MODEL` or `WHISPERX_MODEL_NAME`: model name (e.g., `small`, `large-v3`) or an absolute path.
  - `WHISPERX_MODEL_DIR`: local directory to use if it contains `model.bin` (default `/app/models/faster-whisper-large-v3`).
    - If the directory is missing `model.bin`, the worker falls back to a model name to auto‑download.
- Device and precision
  - `WHISPERX_DEVICE`: `auto` (default), `cpu`, or `cuda`.
  - The worker selects `compute_type=float16` on GPU and `int8` on CPU automatically.
- Hugging Face caches (recommended with a RunPod Network Volume)
  - `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME` → point these to your attached volume, e.g. `/runpod-volume/hf` and `/runpod-volume/torch`.
  - First run downloads models to the volume; later runs reuse the cache for faster cold starts.

- Scratch and downloads (avoid filling root filesystem)
  - `JOBS_DIR` — where input audio and job files are stored (e.g., `/runpod-volume/jobs`).
  - `TMPDIR` — where temporary files are written (e.g., `/runpod-volume/tmp`).

- Logging and diagnostics
  - `DEBUG=1` — switch console logs to DEBUG and default job-level `debug` to true if not provided in input.
  - `CONSOLE_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR|CRITICAL` — explicit console verbosity.
  - `PRINT_MOUNTS=1` — print `df -h` at startup to show mounts and free space.

### Diarization Notes

- Set `HF_TOKEN` and accept terms for gated pyannote models to enable diarization.
  - Gated repos: `pyannote/embedding`, `pyannote/speaker-diarization@2.1`.
- If diarization models cannot be loaded (no token/terms), the worker logs a message and skips diarization; transcription still succeeds.
- Speaker verification (SpeechBrain ECAPA) is lazy‑loaded on first use.

### Local Testing

- Build and run the included local harness on CPU with a small model:
  - `DOCKER_BUILDKIT=1 docker build -t whisperx-worker:local .`
  - `docker run --rm -it \`
    `  -v "$PWD/tests:/app/tests" \`
    `  -v "$PWD/samples:/app/samples" \`
    `  -e WHISPERX_MODEL=small \`
    `  whisperx-worker:local \`
    `  python3 /app/tests/test_worker_local.py`

### Bundled VAD Model (optional)

- To speed up VAD initialization, you may include the VAD file in the repo:
  - Place at `models/whisperx-vad-segmentation.bin` or `models/vad/whisperx-vad-segmentation.bin`.
  - The Dockerfile normalizes it to `/app/models/vad/whisperx-vad-segmentation.bin` and sets `VAD_MODEL_PATH` accordingly.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project utilizes code from [WhisperX](https://github.com/m-bain/whisperX), licensed under the BSD-2-Clause license
- Special thanks to the RunPod team for the serverless platform
