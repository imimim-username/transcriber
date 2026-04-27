"""Entry point: diarize-and-transcribe an audio file, or transcribe a multi-track zip.

Usage — single audio file:
    python main.py path/to/audio.mp3

    Supported formats: anything pydub/ffmpeg can read (mp3, m4a, wav, flac, ogg, …).
    The file is converted to WAV internally; the original is untouched.
    Output files are written next to the input:
        audiofile.json  — full results as JSON
        audiofile.md    — human-readable Markdown transcript

Usage — multi-track zip:
    python main.py path/to/recording.zip

    The zip must contain one audio file per speaker named ``[number]-[speaker].ext``
    (e.g. ``1-alice.aac``, ``2-bob.ogg``) and an optional ``info.txt`` with metadata.
    Diarization is skipped; each track is transcribed independently, then all
    segments are merged chronologically.
    Output files are written next to the zip:
        meeting-YYYY-MM-DD.json
        meeting-YYYY-MM-DD.md

Set HF_HUB_OFFLINE=1 in .env (or the environment) to prevent all network calls
after the models have been downloaded once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Must be set before PyTorch (or any OpenMP-linked library) is imported.
# On macOS with Homebrew/conda, duplicate OpenMP libraries cause a crash;
# this env var tells OpenMP to tolerate the conflict.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
from pydub import AudioSegment

from diarize import diarize
from transcribe import format_time, transcribe

load_dotenv(Path(__file__).resolve().parent / ".env")


def _to_wav(audio_path: Path) -> tuple[Path, bool]:
    """Return a WAV version of *audio_path*.

    If the file is already a WAV, return it as-is (``created=False``).
    Otherwise convert it to a temporary file (``created=True``);
    the caller is responsible for deleting it.
    """
    if audio_path.suffix.lower() == ".wav":
        return audio_path, False

    audio = AudioSegment.from_file(str(audio_path))
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio.export(tmp, format="wav")
    return Path(tmp), True


def _write_json(results: list[dict], out_path: Path) -> None:
    """Write *results* as a JSON array to *out_path*."""
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _write_markdown(
    results: list[dict],
    source_name: str,
    out_path: Path,
) -> None:
    """Write a human-readable Markdown transcript to *out_path*."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Transcript: {source_name}",
        "",
        f"*Generated: {now}*",
        "",
        "---",
        "",
    ]

    for item in results:
        seg = item["segmentInfo"]
        text = item["text"]
        if not text:
            continue
        start = format_time(seg["start_time"])
        end = format_time(seg["end_time"])
        speaker = seg["speaker"]
        lines.append(f"[{start} → {end}] **{speaker}:** {text}")
        lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diarize and transcribe an audio file, "
            "or transcribe a multi-track zip of per-speaker audio files."
        ),
    )
    parser.add_argument(
        "input_file",
        help=(
            "Path to an audio file (mp3, m4a, wav, flac, …) "
            "or a zip file containing per-speaker audio tracks."
        ),
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_file).resolve()
    if not input_path.is_file():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # --- Zip mode: multi-track per-speaker recording ---
    if input_path.suffix.lower() == ".zip":
        from transcribe_zip import process_zip
        print(f"Processing zip: {input_path.name}")
        process_zip(input_path)
        return

    # --- Single audio file mode (original behaviour) ---
    audio_path = input_path
    stem = audio_path.stem
    out_dir = audio_path.parent
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    print(f"Processing: {audio_path.name}")

    wav_path, wav_created = _to_wav(audio_path)
    try:
        print("Running diarization…")
        segments = diarize(str(wav_path))

        if not segments:
            print("No speech segments found.")
            return

        print(f"Transcribing {len(segments)} segment(s)…\n")
        results = transcribe(str(wav_path), segments)
    finally:
        if wav_created and wav_path.is_file():
            wav_path.unlink()

    print(f"\nDone. Writing output files…")
    _write_json(results, json_path)
    _write_markdown(results, audio_path.name, md_path)
    print(f"  {json_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
