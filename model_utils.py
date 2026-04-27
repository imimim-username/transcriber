"""Shared utilities for Whisper model loading.

Both :mod:`transcribe` and :mod:`transcribe_zip` use the same device-detection,
dtype-selection, and model-loading logic.  This module centralises it so there
is a single source of truth.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def offline() -> bool:
    """Return ``True`` when any HuggingFace offline env var is set."""
    return (
        os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    )


def load_whisper(model_id: str) -> Any:
    """Load a Whisper model and return a HuggingFace *pipeline* object.

    Device priority: **CUDA → MPS (Apple Silicon) → CPU**.

    - ``float16`` is used on CUDA; ``float32`` on MPS and CPU (float16 has
      incomplete op support on MPS).
    - SDPA attention is enabled on CUDA only.
    - ``local_files_only`` is set automatically based on :func:`offline`.
    """
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # float16 has incomplete op support on MPS
    else:
        device = "cpu"
        torch_dtype = torch.float32

    local_files_only = offline()

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
    # handles them natively.  Passing them via generate_kwargs= on the pipeline
    # causes a duplicate-logits-processor warning because .generate() also
    # creates the same processors internally when it sees language/task.
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
