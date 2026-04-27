# Project Context

## What this is

A local, offline-capable CLI tool that diarizes and transcribes an audio file.
No cloud APIs. Runs entirely on-device using HuggingFace models.

**Usage:**
```bash
python main.py path/to/audio.mp3
```

---

## File structure

```
transcriber/
├── main.py          # CLI entry point: arg parsing, WAV conversion, file output
├── diarize.py       # Speaker diarization via pyannote.audio
├── transcribe.py    # Speech-to-text via Whisper (HuggingFace Transformers)
├── requirements.txt
├── .env.example
├── README.md
└── context/
    └── context.md   # This file
```

---

## How the pipeline works

1. `main.py` takes an audio file path as a CLI argument
2. If the file isn't already WAV, it's converted to a temp WAV via pydub/ffmpeg (cleaned up after)
3. `diarize()` runs pyannote speaker diarization → list of `{start_time, end_time, speaker}` segments
4. `transcribe()` loads Whisper, iterates each segment, slices the audio, runs inference, and prints progress live as each segment completes
5. `main.py` writes two output files next to the source audio:
   - `{stem}.json` — full results array
   - `{stem}.md` — formatted Markdown transcript with bold speaker labels and UTC generation timestamp

---

## Models

| Model | Purpose | Size |
|---|---|---|
| `pyannote/speaker-diarization-community-1` | Speaker diarization | ~300 MB |
| `openai/whisper-large-v3-turbo` | Speech-to-text | ~1.6 GB |

Both are cached by HuggingFace in `~/.cache/huggingface/` after first download.

---

## Key implementation details

### Device selection
Priority order: **CUDA → MPS (Apple Silicon) → CPU** — both `diarize.py` and `transcribe.py` follow this.

- `diarize.py`: checks `torch.cuda.is_available()` then `torch.backends.mps.is_available()`
- `transcribe.py`: same check; dtype is `float16` on CUDA, `float32` on MPS and CPU (float16 has incomplete op support on MPS); SDPA attention enabled on CUDA only

### `from_pretrained()` parameter note
- The `dtype=` parameter is used (not `torch_dtype=` — that name is deprecated in newer transformers versions)
- `torch_dtype=` is still passed to the `pipeline()` call (this is a different parameter on a different object and has not been renamed)

### Offline mode
- Controlled by `HF_HUB_OFFLINE=1` in `.env`
- `transcribe.py`: passes `local_files_only=True` to both `AutoModelForSpeechSeq2Seq.from_pretrained()` and `AutoProcessor.from_pretrained()` when offline
- `diarize.py`: pyannote doesn't support `local_files_only` directly — `HF_HUB_OFFLINE=1` env var handles it at the HF Hub level; token is set to `None` in offline mode
- `TRANSFORMERS_OFFLINE=1` is also checked as an alternative
- **Important:** `HF_HUB_OFFLINE=1` will crash if the model has never been downloaded. Must run once with `HF_HUB_OFFLINE=0` (and network access) to populate the cache, then offline mode works forever after.

### HF token handling
- `diarize.py` checks `HF_TOKEN` then `HUGGING_FACE_HUB_TOKEN` env vars
- Token is skipped entirely when `HF_HUB_OFFLINE=1`
- Whisper model requires no token

### `.env` loading
- `load_dotenv()` is called at module level in both `main.py` and `diarize.py`
- Points to `.env` in the project root (relative to `__file__`)

### Progress output (`transcribe.py`)
Printed live as each segment is processed:
```
  [1/38] 00:00.480 → 00:03.820  SPEAKER_00
    Hey, how's it going?
```
- `format_time()` is a public function in `transcribe.py`, imported by `main.py`

### Output files
- Written to the same directory as the input audio file, named after its stem
- JSON: array of `{"segmentInfo": {start_time, end_time, speaker}, "text": str}`
- Markdown: title, UTC generation timestamp, one line per non-empty segment with bold speaker label

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace auth token (needed to download pyannote model) |
| `HUGGING_FACE_HUB_TOKEN` | — | Alternate token key (same purpose) |
| `HF_HUB_OFFLINE` | `0` | Set to `1` to block all network calls |
| `TRANSFORMERS_OFFLINE` | `0` | Alternate offline flag (same effect) |
| `HF_HOME` | `~/.cache/huggingface` | Override model cache location |

---

## Dependencies (requirements.txt)

```
accelerate
pandas
pydub
pyannote-audio
python-dotenv
torch
transformers
```

ffmpeg must also be installed system-wide for audio conversion.

---

## Pending / ideas discussed

*(nothing currently open)*

---

## Git / deployment

- Remote: `git@github.com:imimim-username/transcriber.git`
- Branch: `main`
- SSH key for pushing: `/workspace/extra/github-keys/github_deploy`
- Push command: `GIT_SSH_COMMAND="ssh -i /workspace/extra/github-keys/github_deploy -o StrictHostKeyChecking=no" git push origin main`
