# Project Context

## What this is

A local, offline-capable CLI tool that diarizes and transcribes audio.
No cloud APIs. Runs entirely on-device using HuggingFace models.

Two modes:

**Single audio file** — diarize → transcribe per-segment:
```bash
python main.py path/to/audio.mp3
```

**Multi-track zip** — one audio file per speaker, no diarization:
```bash
python main.py path/to/recording.zip
```

---

## File structure

```
transcriber/
├── main.py              # CLI entry point: dispatches audio vs. zip mode
├── diarize.py           # Speaker diarization via pyannote.audio
├── transcribe.py        # Speech-to-text via Whisper (HuggingFace Transformers)
├── transcribe_zip.py    # Multi-track zip pipeline (no diarization)
├── requirements.txt
├── .env.example
├── README.md
├── context/
│   └── context.md       # This file
└── tests/
    ├── __init__.py
    ├── conftest.py          # ML dependency stubs (torch, transformers, pyannote, …)
    ├── test_diarize.py
    ├── test_main.py
    ├── test_transcribe.py
    └── test_transcribe_zip.py
```

---

## How the pipeline works

### Single audio file mode

1. `main.py` takes an audio file path as a CLI argument
2. If the file isn't already WAV, it's converted to a temp WAV via pydub/ffmpeg (cleaned up after)
3. `diarize()` runs pyannote speaker diarization → list of `{start_time, end_time, speaker}` segments
4. `transcribe()` loads Whisper, iterates each segment, slices the audio, runs inference, and prints progress live as each segment completes
5. `main.py` writes two output files next to the source audio:
   - `{stem}.json` — full results array
   - `{stem}.md` — formatted Markdown transcript with bold speaker labels and UTC generation timestamp

### Zip mode (multi-track per-speaker recording)

1. `main.py` detects a `.zip` extension and delegates to `transcribe_zip.process_zip()`
2. Zip is extracted to a `tempfile.mkdtemp()` directory (cleaned up in `finally`)
3. `info.txt` is parsed for `Start time: YYYY-MM-DDThh:mm:ss.sssZ` → date used for output filenames
4. Audio files are discovered by scanning for `[number]-[speaker].[ext]` filenames; sorted by number
5. Whisper is loaded once (`openai/whisper-large-v3-turbo`)
6. Each track is transcribed with `pipe(str(audio_path), return_timestamps=True)` — produces chunk-level timestamps; no diarization needed (speaker identity from filename)
7. All segments from all tracks are merged and sorted chronologically by `start_time`
8. Output written next to the zip:
   - `meeting-YYYY-MM-DD.json`
   - `meeting-YYYY-MM-DD.md`

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

### `transcribe_zip.py` implementation details

- `process_zip(zip_path, *, model_id="openai/whisper-large-v3-turbo") -> tuple[Path, Path]` — public API
- `parse_info_txt(info_path)` — regex `r"Start time:\s+(\d{4}-\d{2}-\d{2})T(\S+)"`, falls back to today's date
- `_speaker_from_filename(name)` — regex `r"^\d+-(.+)$"` on stem: `"1-alice.aac"` → `"alice"`
- `_load_model(model_id)` — same CUDA → MPS → CPU device detection and dtype logic as `transcribe.py`
- `_transcribe_track(audio_path, speaker, pipe)` — calls `pipe(str(audio_path), return_timestamps=True)`, iterates `result["chunks"]`; handles `None` end timestamps with `end = start` fallback
- tqdm progress bar over tracks (`unit="track"`); `tqdm.write()` for per-segment text output
- `_AUDIO_EXTENSIONS` frozenset: `.mp3 .mp4 .m4a .aac .ogg .flac .wav .wma .opus .webm`
- Imports `format_time` from `transcribe`

### Progress output

**`transcribe.py`** (single audio file mode) — printed live per diarized segment:
```
  [1/38] 00:00.480 → 00:03.820  SPEAKER_00
    Hey, how's it going?
```

**`transcribe_zip.py`** (zip mode) — printed live per track and per chunk:
```
── [1/2] alice (1-alice.aac) ──
  Running Whisper inference…
  [1/14] 00:00.000 → 00:02.400  alice
    Hello there.
  [2/14] 00:02.500 → 00:05.100  alice
    How's it going?
```
- Outer tqdm bar tracks overall track progress (`desc="Tracks"`)
- `tqdm.write()` used for all per-segment/per-chunk output to avoid clobbering the bar
- `format_time()` is a public function in `transcribe.py`, imported by both `main.py` and `transcribe_zip.py`

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
pytest
python-dotenv
torch
tqdm
transformers
```

ffmpeg must also be installed system-wide for audio conversion.

## Running the tests

```bash
pytest tests/ -v
```

No GPU, no models, no ffmpeg needed. `tests/conftest.py` stubs all heavy ML
imports (`torch`, `transformers`, `pyannote`, `pydub`, `tqdm`, `dotenv`) into
`sys.modules` before the production code is imported, so the suite runs in any
plain Python environment with only `pytest` installed.

89 tests total:
- `test_transcribe.py` — `format_time`, `_offline`, `transcribe()`
- `test_diarize.py` — token resolution, annotation parsing, `diarize()`
- `test_transcribe_zip.py` — `parse_info_txt`, speaker extraction, writers, `_transcribe_track`, progress output (spying on `tqdm.write`), `process_zip()`
- `test_main.py` — `_to_wav`, output writers, `main()` dispatch and cleanup

Progress output tests use `patch.object(transcribe_zip.tqdm, "write")` to spy on `tqdm.write` calls and assert correct format and ordering (e.g. "Running Whisper inference…" printed before `pipe()` is invoked).

---

## Pending / ideas discussed

*(nothing currently open)*

---

## Recent changes

### 2026-04-27
- Added `transcribe_zip.py` — full multi-track zip pipeline (no diarization, speaker from filename, chronological merge)
- Updated `main.py` to detect `.zip` extension and dispatch to `process_zip()`
- Updated `README.md` with zip mode usage, zip layout, `info.txt` format, and output filename docs
- Added full test suite: 83 tests across 4 files; `conftest.py` stubs all ML deps so tests run with only pytest installed
- Added pytest to `requirements.txt`; updated all documentation
- Added progress output to `transcribe_zip.py`: "Running Whisper inference…" before each `pipe()` call, `[i/total_chunks]` chunk counter per segment line, `[idx/total_tracks]` counter in track section headers, tqdm bar desc changed to "Tracks"
- Added 6 progress-output tests (89 total) using `patch.object(transcribe_zip.tqdm, "write")` to spy on tqdm calls
- Added Mermaid flowchart to README.md showing both pipeline modes; replaced prose "How it works" section with a concise module reference table

---

## Git / deployment

- Remote: `git@github.com:imimim-username/transcriber.git`
- Branch: `main`
- SSH key for pushing: `/workspace/extra/github-keys/github_deploy`
- Push command: `GIT_SSH_COMMAND="ssh -i /workspace/extra/github-keys/github_deploy -o StrictHostKeyChecking=no" git push origin main`
