"""Multi-track zip transcription: one audio file per speaker, merged into one transcript.

Expected zip layout::

    info.txt          — recording metadata (see below)
    1-alice.aac       — audio track for speaker "alice"
    2-bob.ogg         — audio track for speaker "bob"
    …

Audio files are named ``[number]-[speaker].[ext]``.  The number is cosmetic and
only used for sort order; the speaker label comes from the filename stem.

``info.txt`` is expected to contain a line like::

    Start time:  2026-04-27T11:51:12.426Z

That date is used to name the output files::

    meeting-2026-04-27.json
    meeting-2026-04-27.md

Both output files are written next to the zip file.
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
import sys
import tempfile
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from model_utils import load_whisper
from transcribe import format_time


# File extensions treated as audio tracks inside the zip.
_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".mp4", ".m4a", ".aac", ".ogg", ".flac", ".wav", ".wma", ".opus", ".webm"}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract all zip members to *dest*, rejecting path-traversal entries.

    Raises :class:`RuntimeError` if any member's resolved path would fall
    outside *dest* (e.g. a ``../`` attack).
    """
    dest_resolved = dest.resolve()
    for member in zf.infolist():
        target = (dest_resolved / member.filename).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError:
            raise RuntimeError(
                f"Refusing to extract unsafe zip member: {member.filename!r}"
            )
        zf.extract(member, dest)


def parse_info_txt(info_path: Path) -> dict[str, str]:
    """Parse ``info.txt`` and return a dict with at least ``date_str`` (YYYY-MM-DD).

    Keys returned:
      ``date_str``         — e.g. ``"2026-04-27"``
      ``start_time_raw``   — the raw ISO string, e.g. ``"2026-04-27T11:51:12.426Z"``
    """
    try:
        content = info_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        content = ""

    result: dict[str, str] = {}

    m = re.search(r"Start time:\s+(\d{4}-\d{2}-\d{2})T(\S+)", content)
    if m:
        result["date_str"] = m.group(1)
        result["start_time_raw"] = f"{m.group(1)}T{m.group(2)}"
    else:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result["date_str"] = today
        result["start_time_raw"] = today

    return result


def _speaker_from_filename(name: str) -> str:
    """Extract speaker label from ``'1-alice.aac'`` → ``'alice'``.

    Falls back to the full stem if the pattern doesn't match.
    """
    stem = Path(name).stem           # '1-alice'
    m = re.match(r"^\d+-(.+)$", stem)
    return m.group(1) if m else stem


def _sort_key(entry: tuple[int, str, Path]) -> int:
    return entry[0]


def _spinning_inference(pipe: Any, audio_path: Path) -> dict[str, Any]:
    """Call ``pipe()`` while showing a spinner on stderr.

    Whisper inference on a long track can take several minutes with no
    visible output.  The spinner reassures the user that the process has
    not stalled.  A background thread updates the spinner character every
    100 ms while the (blocking) ``pipe()`` call runs in the foreground.
    The spinner line is cleared before returning so subsequent output is
    not affected.
    """
    frames = itertools.cycle("|/-\\")
    stop = threading.Event()

    def _spin() -> None:
        while not stop.is_set():
            sys.stderr.write(f"\r  Running Whisper inference… {next(frames)} ")
            sys.stderr.flush()
            stop.wait(0.1)
        # Erase the spinner line so it doesn't linger in the terminal.
        sys.stderr.write("\r" + " " * 50 + "\r")
        sys.stderr.flush()

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        return pipe(str(audio_path), return_timestamps=True)
    finally:
        stop.set()
        thread.join()


# ---------------------------------------------------------------------------
# Per-track transcription
# ---------------------------------------------------------------------------

def _transcribe_track(
    audio_path: Path,
    speaker: str,
    pipe: Any,
) -> list[dict[str, Any]]:
    """Transcribe a full audio track with chunk-level timestamps.

    Returns a list of segment dicts::

        {"segmentInfo": {"start_time": float, "end_time": float, "speaker": str},
         "text": str}

    Empty chunks are skipped.
    """
    result = _spinning_inference(pipe, audio_path)
    chunks: list[dict[str, Any]] = result.get("chunks") or []
    total_chunks = len(chunks)

    segments: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, 1):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        ts = chunk.get("timestamp") or (None, None)
        start: float = float(ts[0]) if ts[0] is not None else 0.0
        # Whisper occasionally returns None for the end of the last chunk.
        end: float = float(ts[1]) if ts[1] is not None else start

        tqdm.write(f"  [{i}/{total_chunks}] {format_time(start)} → {format_time(end)}  {speaker}")
        tqdm.write(f"    {text}")

        segments.append(
            {
                "segmentInfo": {
                    "start_time": start,
                    "end_time": end,
                    "speaker": speaker,
                },
                "text": text,
            }
        )

    return segments


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_json(path: Path, segments: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(segments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_md(
    path: Path,
    segments: list[dict[str, Any]],
    source_name: str,
    start_time_raw: str,
) -> None:
    lines = [
        f"# Transcript: {source_name}",
        "",
        f"*Start time: {start_time_raw}*",
        "",
        "---",
        "",
    ]
    for seg in segments:
        info = seg["segmentInfo"]
        start_fmt = format_time(info["start_time"])
        end_fmt = format_time(info["end_time"])
        speaker = info["speaker"]
        text = seg["text"]
        lines.append(f"[{start_fmt} → {end_fmt}] **{speaker}:** {text}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_zip(
    zip_path: str | Path,
    *,
    model_id: str = "openai/whisper-large-v3-turbo",
) -> tuple[Path, Path]:
    """Process a zip of per-speaker audio tracks and produce a merged transcript.

    Extracts the zip to a temporary directory (cleaned up on exit), transcribes
    each audio track, merges all segments sorted by start time, and writes::

        meeting-YYYY-MM-DD.json
        meeting-YYYY-MM-DD.md

    Both files are written next to ``zip_path``.  Returns ``(json_path, md_path)``.

    The zip is extracted with path-traversal protection: any member whose
    resolved path falls outside the temporary directory raises :class:`RuntimeError`.
    """
    zip_path = Path(zip_path).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="transcriber_zip_"))
    try:
        # --- Extract (with path-traversal protection) ---
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extractall(zf, tmp_dir)

        # --- Parse info.txt ---
        info_file = tmp_dir / "info.txt"
        info = parse_info_txt(info_file)
        date_str = info["date_str"]
        start_time_raw = info["start_time_raw"]

        # --- Discover audio files (recursive) ---
        audio_entries: list[tuple[int, str, Path]] = []
        for f in tmp_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in _AUDIO_EXTENSIONS:
                m = re.match(r"^(\d+)-", f.name)
                num = int(m.group(1)) if m else 0
                speaker = _speaker_from_filename(f.name)
                audio_entries.append((num, speaker, f))

        audio_entries.sort(key=_sort_key)

        if not audio_entries:
            raise RuntimeError(
                f"No audio files found in {zip_path.name}. "
                f"Expected files named like '1-speaker.aac'."
            )

        # --- Load model once ---
        print(f"Found {len(audio_entries)} track(s). Loading Whisper model…")
        pipe = load_whisper(model_id)

        # --- Transcribe each track ---
        all_segments: list[dict[str, Any]] = []

        total_tracks = len(audio_entries)
        with tqdm(
            total=total_tracks,
            unit="track",
            desc="Tracks",
            ncols=80,
        ) as bar:
            for idx, (_num, speaker, audio_path) in enumerate(audio_entries, 1):
                tqdm.write(f"\n── [{idx}/{total_tracks}] {speaker} ({audio_path.name}) ──")
                segments = _transcribe_track(audio_path, speaker, pipe)
                all_segments.extend(segments)
                bar.update(1)

        # --- Merge: sort all segments chronologically ---
        all_segments.sort(key=lambda s: s["segmentInfo"]["start_time"])

        # --- Write outputs ---
        out_stem = f"meeting-{date_str}"
        out_dir = zip_path.parent
        json_path = out_dir / f"{out_stem}.json"
        md_path = out_dir / f"{out_stem}.md"

        _write_json(json_path, all_segments)
        _write_md(md_path, all_segments, zip_path.name, start_time_raw)

        print(f"\nDone. Writing output files…")
        print(f"  {json_path}")
        print(f"  {md_path}")

        return json_path, md_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
