"""Tests for transcribe.py — format_time and transcribe()."""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# format_time
# ---------------------------------------------------------------------------

from transcribe import format_time


class TestFormatTime:
    def test_zero(self):
        assert format_time(0.0) == "00:00.000"

    def test_sub_minute(self):
        assert format_time(3.82) == "00:03.820"

    def test_exactly_one_minute(self):
        assert format_time(60.0) == "01:00.000"

    def test_multiple_minutes(self):
        assert format_time(125.5) == "02:05.500"

    def test_millisecond_precision(self):
        assert format_time(0.001) == "00:00.001"

    def test_large_value_sub_hour(self):
        # 59 minutes 45 seconds 678 ms — under 1 hour → MM:SS.mmm
        assert format_time(59 * 60 + 45.678) == "59:45.678"

    def test_exactly_one_hour(self):
        assert format_time(3600.0) == "1:00:00.000"

    def test_hours_display(self):
        # 1 hour, 23 minutes, 45.678 seconds → H:MM:SS.mmm
        assert format_time(3600 + 23 * 60 + 45.678) == "1:23:45.678"

    def test_fractional_seconds(self):
        assert format_time(9.999) == "00:09.999"

    def test_two_hours(self):
        assert format_time(2 * 3600 + 5 * 60 + 3.001) == "2:05:03.001"


# ---------------------------------------------------------------------------
# transcribe() — heavy model is mocked
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration_ms: int = 1000, sample_rate: int = 16000) -> None:
    """Write a minimal silent WAV file."""
    n_frames = int(sample_rate * duration_ms / 1000)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _mock_pipe_factory(text: str = "hello") -> MagicMock:
    """Return a mock pipeline callable that always returns *text*."""
    mock = MagicMock()
    mock.return_value = {"text": text}
    return mock


def _make_segments(**kwargs: Any) -> list[dict]:
    defaults = [
        {"start_time": 0.0, "end_time": 0.5, "speaker": "SPEAKER_00"},
        {"start_time": 0.5, "end_time": 1.0, "speaker": "SPEAKER_01"},
    ]
    return kwargs.get("segments", defaults)


@pytest.fixture()
def wav_file(tmp_path: Path) -> Path:
    p = tmp_path / "audio.wav"
    _make_wav(p, duration_ms=2000)
    return p


class TestTranscribe:
    """Tests for transcribe() with mocked Whisper model."""

    def _patch_pipeline(self, mock_text: str = "hi"):
        """Return a (patch context-manager, mock_pipe) pair.

        Patches ``transcribe.load_whisper`` so the model is never actually loaded.
        """
        mock_pipe = _mock_pipe_factory(mock_text)
        ctx = patch("transcribe.load_whisper", return_value=mock_pipe)
        return ctx, mock_pipe

    def test_missing_file_raises(self):
        from transcribe import transcribe
        with pytest.raises(FileNotFoundError):
            transcribe("/nonexistent/audio.wav", [])

    def test_empty_segments_returns_empty(self, wav_file: Path):
        from transcribe import transcribe
        ctx, mock_pipe = self._patch_pipeline()
        with ctx:
            result = transcribe(str(wav_file), [])
        assert result == []
        mock_pipe.assert_not_called()

    def test_returns_one_result_per_segment(self, wav_file: Path):
        from transcribe import transcribe
        segments = _make_segments()
        ctx, mock_pipe = self._patch_pipeline("hey")
        with ctx:
            result = transcribe(str(wav_file), segments)
        assert len(result) == len(segments)

    def test_result_structure(self, wav_file: Path):
        from transcribe import transcribe
        seg = {"start_time": 0.0, "end_time": 0.5, "speaker": "SPEAKER_00"}
        ctx, _ = self._patch_pipeline("hello world")
        with ctx:
            result = transcribe(str(wav_file), [seg])
        assert result[0]["segmentInfo"] is seg
        assert result[0]["text"] == "hello world"

    def test_zero_length_segment_skipped(self, wav_file: Path):
        from transcribe import transcribe
        seg = {"start_time": 0.5, "end_time": 0.5, "speaker": "SPEAKER_00"}
        ctx, mock_pipe = self._patch_pipeline()
        with ctx:
            result = transcribe(str(wav_file), [seg])
        # Zero-length segments are skipped (not added to results)
        assert result == []
        mock_pipe.assert_not_called()

    def test_pipeline_error_yields_empty_text(self, wav_file: Path):
        from transcribe import transcribe

        error_pipe = MagicMock(side_effect=RuntimeError("inference exploded"))
        seg = {"start_time": 0.0, "end_time": 0.5, "speaker": "SPEAKER_00"}
        with patch("transcribe.load_whisper", return_value=error_pipe):
            result = transcribe(str(wav_file), [seg])
        assert result[0]["text"] == ""

    def test_text_is_stripped(self, wav_file: Path):
        from transcribe import transcribe
        seg = {"start_time": 0.0, "end_time": 0.5, "speaker": "SPEAKER_00"}
        ctx, _ = self._patch_pipeline("  padded  ")
        with ctx:
            result = transcribe(str(wav_file), [seg])
        assert result[0]["text"] == "padded"

    def test_injected_pipe_used_directly(self, wav_file: Path):
        """When pipe= is supplied, load_whisper must NOT be called."""
        from transcribe import transcribe

        pre_loaded_pipe = _mock_pipe_factory("pre-loaded result")
        seg = {"start_time": 0.0, "end_time": 0.5, "speaker": "SPEAKER_00"}

        with patch("transcribe.load_whisper") as mock_load:
            result = transcribe(str(wav_file), [seg], pipe=pre_loaded_pipe)

        mock_load.assert_not_called()
        assert result[0]["text"] == "pre-loaded result"
