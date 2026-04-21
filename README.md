# transcriber

A local, offline-capable pipeline that diarizes and transcribes an audio file.
No cloud APIs. No subscriptions. Everything runs on your machine.

---

## What it does

1. **Diarization** — identifies *who* is speaking and *when*, using [pyannote.audio](https://github.com/pyannote/pyannote-audio).
2. **Transcription** — converts each speaker's audio segment to text using [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) via HuggingFace Transformers.

Output looks like:

```
[00:00.480 → 00:03.820] SPEAKER_00: Hey, how's it going?
[00:04.100 → 00:07.650] SPEAKER_01: Pretty good, just finished the report.
[00:08.000 → 00:12.340] SPEAKER_00: Nice. Did you send it over yet?
```

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and on your `PATH` (for audio conversion)
- A CUDA-capable GPU is strongly recommended for reasonable speed.
  CPU inference works but is significantly slower, especially for Whisper.

---

## Installation

```bash
# Clone and enter the repo
git clone <repo-url>
cd transcriber

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Setup

### 1. Copy the example env file

```bash
cp .env.example .env
```

### 2. Set your HuggingFace token (first run only)

The diarization model (`pyannote/speaker-diarization-community-1`) is hosted on
HuggingFace. On the first run, it is downloaded automatically. If the model page
requires you to accept terms, you'll need to:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Visit the [model page](https://huggingface.co/pyannote/speaker-diarization-community-1) and accept the conditions
3. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token
4. Add it to your `.env`:

```
HF_TOKEN=hf_yourtoken
```

The Whisper model downloads automatically with no account needed.

---

## Usage

```bash
python main.py path/to/audio.mp3
```

Any format ffmpeg supports works: `.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`, etc.

On first run, the models are downloaded from HuggingFace (~3 GB total) and
cached locally. Subsequent runs load from cache and are much faster.

---

## Offline mode

Once the models have been downloaded at least once, you can prevent the script
from making any network calls by setting in your `.env`:

```
HF_HUB_OFFLINE=1
```

In offline mode:
- Both models are loaded directly from your local HuggingFace cache
- No HF token is required
- If a model isn't cached yet, the script will fail with a clear error

To find where models are cached:

```bash
python -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)"
```

To move the cache to a different disk (useful if your home partition is small),
add to `.env`:

```
HF_HOME=/mnt/data/huggingface
```

---

## How it works

### `main.py`

The entry point. Parses the CLI argument, converts the input file to WAV if
necessary (using pydub/ffmpeg), calls `diarize()` then `transcribe()`, and
prints the results.

### `diarize.py`

Uses `pyannote.audio`'s `Pipeline` to perform speaker diarization. The pipeline:

- Detects segments of continuous speech
- Assigns a speaker label (e.g. `SPEAKER_00`, `SPEAKER_01`) to each segment
- Returns a list of `{"start_time", "end_time", "speaker"}` dicts sorted by time

Runs on GPU automatically if CUDA is available, otherwise CPU.

### `transcribe.py`

Uses HuggingFace Transformers to run `openai/whisper-large-v3-turbo`. For each
diarized segment it:

- Slices that portion of audio from the original file
- Writes it to a temporary WAV file
- Runs Whisper inference on it
- Returns the text

Uses `float16` on GPU and `float32` on CPU. SDPA attention is enabled
automatically on CUDA for better performance.

---

## Models

| Model | Size | Purpose |
|---|---|---|
| `pyannote/speaker-diarization-community-1` | ~300 MB | Speaker diarization |
| `openai/whisper-large-v3-turbo` | ~1.6 GB | Speech-to-text |

Both are cached by HuggingFace in `~/.cache/huggingface/` after the first download.

---

## Troubleshooting

**`FileNotFoundError: Audio file not found`**
Double-check the path you passed. Relative paths are resolved from your current
working directory, not from the script location.

**`OSError: ffmpeg not found`**
Install ffmpeg: `sudo apt install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS).

**`OSError: …local_files_only=True…` / model not found in cache**
You set `HF_HUB_OFFLINE=1` but the model hasn't been downloaded yet. Either run
once with `HF_HUB_OFFLINE=0` (the default) to download, or remove the setting.

**CUDA out of memory**
Whisper large-v3-turbo needs ~6 GB VRAM. If you have less, consider using a
smaller model by editing the `model_id` default in `transcribe.py` — for example
`openai/whisper-base` (~150 MB, works fine on CPU).

**Very slow on CPU**
Expected. Whisper large-v3-turbo on CPU can take several minutes per minute of
audio. For CPU-only machines, switch to `openai/whisper-base` in `transcribe.py`.
