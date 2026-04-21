"""Per-segment transcription from diarization boundaries."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def format_time(seconds: float) -> str:
    """Format *seconds* as ``MM:SS.mmm``."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def _offline() -> bool:
    """Return True when the HuggingFace offline env vars are set."""
    return (
        os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    )


def transcribe(
    audio_path: str | Path,
    diarized_segments: list[dict[str, Any]],
    *,
    model_id: str = "openai/whisper-large-v3-turbo",
) -> list[dict[str, Any]]:
    """Transcribe ``audio_path`` for each diarized segment.

    ``diarized_segments`` is the list returned by :func:`diarize.diarize`, each
    item ``{"start_time": float, "end_time": float, "speaker": str}`` (seconds).

    Returns a list of ``{"segmentInfo": segment_dict, "text": str}``.

    Set ``HF_HUB_OFFLINE=1`` or ``TRANSFORMERS_OFFLINE=1`` in the environment
    (or in ``.env``) to prevent any network access — models must be cached first.
    ``model_id`` may also be an absolute path to a local directory containing the
    model files.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio = AudioSegment.from_file(str(path))

    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # float16 has incomplete op support on MPS
    else:
        device = "cpu"
        torch_dtype = torch.float32
    local_files_only = _offline()

    model_kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "local_files_only": local_files_only,
    }
    if device.startswith("cuda"):
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **model_kwargs)
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

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
