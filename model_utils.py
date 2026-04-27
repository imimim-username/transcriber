"""Shared utilities for Whisper model loading.

Both :mod:`transcribe` and :mod:`transcribe_zip` use the same device-detection,
dtype-selection, and model-loading logic.  This module centralises it so there
is a single source of truth.

On CUDA and MPS the HuggingFace ``transformers`` pipeline is used.  On CPU,
:class:`FasterWhisperAdapter` wraps ``faster_whisper.WhisperModel`` for
significantly better performance via CTranslate2 int8 quantisation and
built-in VAD filtering.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# ---------------------------------------------------------------------------
# Model-ID mapping
# ---------------------------------------------------------------------------

# Maps openai/whisper-* HuggingFace model IDs to faster-whisper size strings.
# Unknown IDs are passed through as-is (e.g. a Systran repo ID or a local
# directory path already compatible with faster-whisper).
_HF_TO_FW: dict[str, str] = {
    "openai/whisper-tiny":           "tiny",
    "openai/whisper-base":           "base",
    "openai/whisper-small":          "small",
    "openai/whisper-medium":         "medium",
    "openai/whisper-large-v2":       "large-v2",
    "openai/whisper-large-v3":       "large-v3",
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
}


# ---------------------------------------------------------------------------
# CPU adapter
# ---------------------------------------------------------------------------

class FasterWhisperAdapter:
    """Wrap ``faster_whisper.WhisperModel`` with the HuggingFace pipeline interface.

    Callers use ``pipe(audio_path, return_timestamps=True)`` regardless of
    whether the HuggingFace pipeline or faster-whisper is running underneath.

    ``return_timestamps=True`` (used by :mod:`transcribe_zip`):
        Returns ``{"chunks": [{"text": str, "timestamp": (start, end)}, …]}``.

    ``return_timestamps=False`` (default, used by :mod:`transcribe`):
        Returns ``{"text": str}`` — all segment texts joined with spaces.

    VAD filtering is always enabled so silent regions are skipped before
    reaching Whisper, which can dramatically cut CPU time for recordings
    with long muted stretches.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(
        self,
        audio: str,
        return_timestamps: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            task="transcribe",
            vad_filter=True,
        )
        # Materialise the generator — faster-whisper is lazy by default.
        chunks = [
            {"text": seg.text, "timestamp": (seg.start, seg.end)}
            for seg in segments
        ]
        if return_timestamps:
            return {"chunks": chunks}
        return {"text": " ".join(c["text"].strip() for c in chunks)}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def offline() -> bool:
    """Return ``True`` when any HuggingFace offline env var is set."""
    return (
        os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    )


def load_whisper(model_id: str) -> Any:
    """Load a Whisper model and return a callable pipeline object.

    **Device selection:** CUDA → MPS (Apple Silicon) → CPU.

    - **CUDA / MPS** — returns a HuggingFace ``transformers`` pipeline.
      Uses ``float16`` on CUDA and ``float32`` on MPS (float16 has incomplete
      op support on MPS).  SDPA attention is enabled on CUDA only.
    - **CPU** — returns :class:`FasterWhisperAdapter` wrapping a
      ``faster_whisper.WhisperModel`` with ``compute_type="int8"``.  This is
      typically 4–8× faster than the HuggingFace pipeline on CPU.

    ``local_files_only`` is set automatically from :func:`offline`.
    """
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # float16 has incomplete op support on MPS
    else:
        device = "cpu"
        torch_dtype = None  # unused on the CPU/faster-whisper path

    local_files_only = offline()

    # ── CPU path: faster-whisper with CTranslate2 int8 ───────────────────────
    if device == "cpu":
        from faster_whisper import WhisperModel  # lazy — not needed on GPU
        fw_model_id = _HF_TO_FW.get(model_id, model_id)
        model = WhisperModel(
            fw_model_id,
            device="cpu",
            compute_type="int8",
            local_files_only=local_files_only,
        )
        return FasterWhisperAdapter(model)

    # ── CUDA / MPS path: HuggingFace transformers pipeline ───────────────────
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

    # Set language/task on generation_config so Whisper's internal .generate()
    # handles them natively — avoids the duplicate-logits-processor warning.
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=device,
    )
