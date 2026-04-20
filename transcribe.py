"""Per-segment transcription from diarization boundaries."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio = AudioSegment.from_file(str(path))

    # Match https://huggingface.co/openai/whisper-large-v3-turbo usage
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
    }
    if device.startswith("cuda"):
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **model_kwargs)
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    transcribed: list[dict[str, Any]] = []

    for segment in diarized_segments:
        start_ms = int(float(segment["start_time"]) * 1000)
        end_ms = int(float(segment["end_time"]) * 1000)
        if end_ms <= start_ms:
            continue

        cropped = audio[start_ms:end_ms]

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            cropped.export(temp_path, format="wav")
            result = pipe(temp_path)
            transcribed.append(
                {
                    "segmentInfo": segment,
                    "text": result.get("text", "").strip(),
                }
            )
        except (ValueError, OSError, RuntimeError):
            # Unusable snippet (e.g. near-silent clip) or I/O / model runtime error
            transcribed.append(
                {
                    "segmentInfo": segment,
                    "text": "",
                }
            )
        finally:
            if os.path.isfile(temp_path):
                os.unlink(temp_path)

    return transcribed
