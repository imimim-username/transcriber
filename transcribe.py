"""Per-segment transcription from diarization boundaries."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from pydub import AudioSegment
from tqdm import tqdm

from model_utils import load_whisper


def format_time(seconds: float) -> str:
    """Format *seconds* as ``MM:SS.mmm``, or ``H:MM:SS.mmm`` for >= 1 hour.

    Uses integer millisecond arithmetic to avoid floating-point rounding errors.
    """
    total_ms = round(float(seconds) * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    if h:
        return f"{h}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def transcribe(
    audio_path: str | Path,
    diarized_segments: list[dict[str, Any]],
    *,
    model_id: str = "openai/whisper-large-v3-turbo",
    pipe: Any = None,
) -> list[dict[str, Any]]:
    """Transcribe ``audio_path`` for each diarized segment.

    ``diarized_segments`` is the list returned by :func:`diarize.diarize`, each
    item ``{"start_time": float, "end_time": float, "speaker": str}`` (seconds).

    Returns a list of ``{"segmentInfo": segment_dict, "text": str}``.

    Pass a pre-loaded pipeline via ``pipe`` to skip model loading (useful when
    calling :func:`transcribe` multiple times in the same process).  If ``pipe``
    is ``None`` (the default), :func:`model_utils.load_whisper` is called once
    internally.

    Set ``HF_HUB_OFFLINE=1`` or ``TRANSFORMERS_OFFLINE=1`` in the environment
    (or in ``.env``) to prevent any network access — models must be cached first.
    ``model_id`` may also be an absolute path to a local directory containing the
    model files.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio = AudioSegment.from_file(str(path))

    if pipe is None:
        pipe = load_whisper(model_id)

    transcribed: list[dict[str, Any]] = []
    total = len(diarized_segments)

    with tqdm(total=total, unit="seg", desc="Transcribing", ncols=80) as bar:
        for segment in diarized_segments:
            start_ms = int(float(segment["start_time"]) * 1000)
            end_ms = int(float(segment["end_time"]) * 1000)
            if end_ms <= start_ms:
                bar.update(1)
                continue

            start_fmt = format_time(segment["start_time"])
            end_fmt = format_time(segment["end_time"])
            speaker = segment["speaker"]
            tqdm.write(f"  {start_fmt} → {end_fmt}  {speaker}")

            cropped = audio[start_ms:end_ms]

            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                cropped.export(temp_path, format="wav")
                result = pipe(temp_path)
                text = result.get("text", "").strip()
                transcribed.append({"segmentInfo": segment, "text": text})
                if text:
                    tqdm.write(f"    {text}")
            except (ValueError, OSError, RuntimeError):
                # Unusable snippet (e.g. near-silent clip) or I/O / model runtime error
                transcribed.append({"segmentInfo": segment, "text": ""})
            finally:
                if os.path.isfile(temp_path):
                    os.unlink(temp_path)

            bar.update(1)

    return transcribed
