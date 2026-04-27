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
├── model_utils.py       # Shared: offline() + load_whisper() used by both transcription modules
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
    ├── test_model_utils.py
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
4. Audio files are discovered by recursively scanning (`rglob`) for `[number]-[speaker].[ext]` filenames anywhere in the zip; sorted by number
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
| `openai/whisper-large-v3-turbo` | Speech-to-text (CUDA / MPS) | ~1.6 GB |
| `Systran/faster-whisper-large-v3-turbo` | Speech-to-text (CPU, int8) | ~800 MB |

All cached by HuggingFace in `~/.cache/huggingface/` after first download. The correct Whisper variant is selected automatically.

---

## Key implementation details

### Shared model utilities (`model_utils.py`)
`model_utils.py` centralises the duplicated logic that previously existed in both `transcribe.py` and `transcribe_zip.py`:
- `offline() -> bool` — checks `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` env vars
- `load_whisper(model_id: str) -> Any` — device detection, then either HF pipeline (CUDA/MPS) or `FasterWhisperAdapter` (CPU)
- `FasterWhisperAdapter` — wraps `faster_whisper.WhisperModel` with the HuggingFace pipeline call signature so callers need no conditional logic

Both `transcribe.py` and `transcribe_zip.py` import from `model_utils` and contain no duplicate model-loading code.

`transcribe()` accepts an optional `pipe=` parameter — if a pre-loaded pipeline is passed in, `load_whisper()` is skipped entirely. This allows the caller to load the model once and reuse it across multiple calls.

### Device selection
Priority order: **CUDA → MPS (Apple Silicon) → CPU** — `model_utils.py` and `diarize.py` both follow this.

- `model_utils.py` (Whisper):
  - CUDA/MPS → HuggingFace `transformers` pipeline; dtype `float16` on CUDA, `float32` on MPS; SDPA attention on CUDA only
  - CPU → `FasterWhisperAdapter` wrapping `faster_whisper.WhisperModel`; `compute_type="int8"`; VAD filter enabled; typically 4–8× faster than HF pipeline on CPU
  - Model-ID mapping: `openai/whisper-*` IDs are translated to faster-whisper size strings (e.g. `"large-v3-turbo"`); unknown IDs pass through as-is
- `diarize.py`: same CUDA → MPS → CPU check; always uses pyannote (no faster-whisper equivalent)

### `from_pretrained()` and `pipeline()` parameter note
- `dtype=` is used everywhere (not `torch_dtype=`) — `torch_dtype` is deprecated in newer transformers versions for both `from_pretrained()` and `pipeline()`

### Offline mode
- Controlled by `HF_HUB_OFFLINE=1` in `.env`
- `model_utils.py`: on CUDA/MPS, passes `local_files_only=True` to both `AutoModelForSpeechSeq2Seq.from_pretrained()` and `AutoProcessor.from_pretrained()` when offline; on CPU, passes `local_files_only=True` to `WhisperModel()`
- `diarize.py`: pyannote doesn't support `local_files_only` directly — `HF_HUB_OFFLINE=1` env var handles it at the HF Hub level; token is set to `None` in offline mode
- `TRANSFORMERS_OFFLINE=1` is also checked as an alternative
- **Important:** `HF_HUB_OFFLINE=1` will crash if the model has never been downloaded. Must run once with `HF_HUB_OFFLINE=0` (and network access) to populate the cache, then offline mode works forever after.

### Zip path-traversal protection
`transcribe_zip._safe_extractall(zf, dest)` replaces `zf.extractall()`. It iterates over all members, resolves each target path, and calls `target.relative_to(dest_resolved)`. Any member whose resolved path escapes the destination directory raises `RuntimeError` before extraction begins.

### `format_time` — hours support
`format_time(seconds)` uses integer millisecond arithmetic to avoid float rounding errors. Output:
- Under 1 hour: `MM:SS.mmm` (e.g. `59:45.678`)
- 1 hour or more: `H:MM:SS.mmm` (e.g. `1:23:45.678`)

### HF token handling
- `diarize.py` checks `HF_TOKEN` then `HUGGING_FACE_HUB_TOKEN` env vars
- Token is skipped entirely when `HF_HUB_OFFLINE=1` **or** `TRANSFORMERS_OFFLINE=1` (consistent with `model_utils.offline()`)
- Whisper model requires no token

### `.env` loading
- `load_dotenv()` is called at module level in both `main.py` and `diarize.py`
- Points to `.env` in the project root (relative to `__file__`)

### `transcribe_zip.py` implementation details

- `process_zip(zip_path, *, model_id="openai/whisper-large-v3-turbo") -> tuple[Path, Path]` — public API
- `_safe_extractall(zf, dest)` — iterates all zip members, resolves each target path, rejects any that escape `dest` with `RuntimeError`; replaces bare `zf.extractall()`
- `parse_info_txt(info_path)` — regex `r"Start time:\s+(\d{4}-\d{2}-\d{2})T(\S+)"`, falls back to today's date
- `_speaker_from_filename(name)` — regex `r"^\d+-(.+)$"` on stem: `"1-alice.aac"` → `"alice"`
- Uses `load_whisper(model_id)` from `model_utils` (no local device/dtype logic)
- Audio file discovery uses `tmp_dir.rglob("*")` — scans recursively, so audio files anywhere in the zip (nested subdirectories included) are found; sorted by leading number
- `_transcribe_track(audio_path, speaker, pipe)` — calls `pipe(str(audio_path), return_timestamps=True)`, iterates `result["chunks"]`; handles `None` end timestamps with `end = start` fallback
- tqdm progress bar over tracks (`unit="track"`); `tqdm.write()` for per-segment text output
- `_AUDIO_EXTENSIONS` frozenset: `.mp3 .mp4 .m4a .aac .ogg .flac .wav .wma .opus .webm`
- Imports `format_time` from `transcribe`

### Progress output

**`transcribe.py`** (single audio file mode) — printed live per diarized segment via `tqdm.write()`; a tqdm bar tracks overall count separately:
```
  00:00.480 → 00:03.820  SPEAKER_00
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

All packages are pinned with `>=` minimum version constraints:

```
accelerate>=0.27.0
pandas>=2.0.0
pydub>=0.25.1
pyannote-audio>=3.1.0
pytest>=7.0.0
python-dotenv>=1.0.0
torch>=2.1.0
tqdm>=4.66.0
transformers>=4.39.0
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

121 tests total:
- `test_model_utils.py` — `offline()`, `FasterWhisperAdapter` (chunks/text output, VAD, language), `load_whisper()` HF path (CUDA/MPS device, SDPA, `local_files_only`, generation_config), CPU path (returns `FasterWhisperAdapter`, model-ID mapping, int8 compute type, `local_files_only`)
- `test_transcribe.py` — `format_time` (including hours and two-hour cases), `transcribe()`, `pipe=` injection bypasses `load_whisper`
- `test_diarize.py` — token resolution, annotation parsing, `diarize()`
- `test_transcribe_zip.py` — `parse_info_txt`, speaker extraction, writers, `_transcribe_track`, `_safe_extractall` (safe extraction + path-traversal rejection + nested paths), recursive subdirectory discovery, progress output (spying on `tqdm.write`), `process_zip()`
- `test_main.py` — `_to_wav`, output writers, `main()` dispatch and cleanup

Progress output tests use `patch.object(transcribe_zip.tqdm, "write")` to spy on `tqdm.write` calls and assert correct format and ordering (e.g. "Running Whisper inference…" printed before `pipe()` is invoked).

Patch targets after `model_utils` refactor:
- `"transcribe.load_whisper"` — patches `load_whisper` in transcribe's namespace
- `"transcribe_zip.load_whisper"` — patches `load_whisper` in transcribe_zip's namespace
- No stub for `model_utils` itself in `conftest.py` — it imports from already-stubbed `torch`/`transformers`

---

## Pending / ideas discussed

*(nothing currently open)*

---

## Recent changes

### 2026-04-27 (faster-whisper CPU path)
- Added `FasterWhisperAdapter` class to `model_utils.py` — wraps `faster_whisper.WhisperModel` with the HuggingFace pipeline call signature; supports both `return_timestamps=True` (chunks) and `return_timestamps=False` (flat text); VAD filtering always enabled
- `load_whisper()` now branches on device: CUDA/MPS → HuggingFace pipeline (unchanged); CPU → `FasterWhisperAdapter` with `compute_type="int8"` (4–8× faster, built-in VAD)
- `_HF_TO_FW` dict maps `openai/whisper-*` HuggingFace model IDs to faster-whisper size strings; unknown IDs pass through as-is
- Added `faster-whisper>=1.0.0` to `requirements.txt`
- `tests/conftest.py`: added `faster_whisper` stub (`WhisperModel` as MagicMock)
- `tests/test_model_utils.py`: refactored into `TestLoadWhisperHF` (CUDA/MPS, forces `cuda=True`) and `TestLoadWhisperCPU` (CPU, faster-whisper path); added `TestFasterWhisperAdapter` (5 tests); 121 tests total, all passing
- Added spinner (`_spinning_inference`) during Whisper inference in `transcribe_zip.py` — background thread writes rotating `|/-\` to stderr every 100 ms; cleared on completion

### 2026-04-27 (post-feedback fixes)
- Fixed `pipeline()` call in `model_utils.py`: changed `torch_dtype=torch_dtype` → `dtype=torch_dtype`; `torch_dtype=` is deprecated in newer transformers for both `from_pretrained()` and `pipeline()`
- Fixed `diarize._resolve_hf_token()` to check both `HF_HUB_OFFLINE` **and** `TRANSFORMERS_OFFLINE` — was only checking `HF_HUB_OFFLINE`; added `test_returns_none_when_transformers_offline` test
- Switched zip audio file discovery from `iterdir()` to `rglob("*")` — now scans recursively so audio files in subdirectories within the zip are found; added `test_audio_files_in_subdirectory_discovered` test
- 107 tests total across 5 files; all passing

### 2026-04-27 (feedback batch)
- Extracted `model_utils.py` — shared `offline()` and `load_whisper()` functions; eliminates duplication between `transcribe.py` and `transcribe_zip.py`
- `transcribe()` now accepts an optional `pipe=` parameter; pass a pre-loaded pipeline to skip model loading on repeated calls
- Added `_safe_extractall()` to `transcribe_zip.py` — rejects zip members with path-traversal (e.g. `../`) attacks via `Path.resolve()` + `relative_to()` check
- Fixed `format_time()` to display `H:MM:SS.mmm` for audio ≥ 1 hour; uses integer millisecond arithmetic to avoid float rounding errors
- Pinned all `requirements.txt` entries with `>=` version constraints
- Added `tests/test_model_utils.py` (13 new tests): `offline()` and `load_whisper()` coverage
- Updated `tests/test_transcribe.py`: removed `_offline`/`TestOffline`, updated `_patch_pipeline` to use `patch("transcribe.load_whisper")`, added `test_injected_pipe_used_directly`, fixed and expanded `format_time` tests (hours, two-hour cases)
- Updated `tests/test_transcribe_zip.py`: changed `_load_model` patch targets to `load_whisper`; added `TestSafeExtractall` (3 tests)
- 104 tests total across 5 files; all passing
- Added `generate_kwargs={"language": "en", "task": "transcribe"}` to `load_whisper()` pipeline call — suppresses multilingual language-detection warnings and forces English transcription in both modes

### 2026-04-27 (initial)
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
