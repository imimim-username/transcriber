"""Speaker diarization using pyannote.audio."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation


def _resolve_hf_token() -> str | bool | None:
    """Token for Hugging Face Hub (gated models). None uses the hub's default lookup."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _annotation_from_pipeline_output(result: Any) -> Annotation:
    """Handle both DiarizeOutput (default) and legacy Annotation returns."""
    annotation = getattr(result, "speaker_diarization", None)
    if annotation is not None:
        return annotation
    if isinstance(result, Annotation):
        return result
    raise TypeError(f"Unexpected pipeline output type: {type(result)!r}")


def diarize(
    audio_path: str | Path,
    *,
    model_id: str = "pyannote/speaker-diarization-community-1",
) -> list[dict[str, Any]]:
    """Run speaker diarization on ``audio_path``.

    Returns a list of segments, each
    ``{"start_time": float, "end_time": float, "speaker": str}`` (seconds).

    The diarization model is gated on Hugging Face: accept the conditions on the
    model card, then set ``HF_TOKEN`` in a project ``.env`` file (loaded
    automatically), or export ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``, or use
    ``huggingface-cli login``.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    pipeline = Pipeline.from_pretrained(model_id, token=_resolve_hf_token())
    pipeline.to(_inference_device())

    with ProgressHook() as hook:
        raw = pipeline(str(path), hook=hook)

    annotation = _annotation_from_pipeline_output(raw)
    segments = [
        {
            "start_time": float(segment.start),
            "end_time": float(segment.end),
            "speaker": str(label),
        }
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: s["start_time"])
    return segments
