"""Entry point: diarize and transcribe an audio file.

Usage:
    python main.py <audio_file>

Supported formats: anything pydub/ffmpeg can read (mp3, m4a, wav, flac, ogg, …).
The file is converted to WAV internally before processing; the original is untouched.

Set HF_HUB_OFFLINE=1 in .env (or the environment) to prevent all network calls
after the models have been downloaded once.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from pydub import AudioSegment

from diarize import diarize
from transcribe import transcribe

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


def _format_time(seconds: float) -> str:
    """Format *seconds* as ``MM:SS.mmm``."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Diarize and transcribe an audio file using local models.",
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to process (mp3, m4a, wav, flac, …).",
    )
    args = parser.parse_args(argv)

    audio_path = Path(args.audio_file).resolve()
    if not audio_path.is_file():
        print(f"error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

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

    for item in results:
        seg = item["segmentInfo"]
        start = _format_time(seg["start_time"])
        end = _format_time(seg["end_time"])
        speaker = seg["speaker"]
        text = item["text"]
        if text:
            print(f"[{start} → {end}] {speaker}: {text}")


if __name__ == "__main__":
    main()
