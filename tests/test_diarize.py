"""Tests for diarize.py — token resolution, device selection, annotation parsing, diarize()."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# _resolve_hf_token
# ---------------------------------------------------------------------------

from diarize import _resolve_hf_token


class TestResolveHfToken:
    def test_returns_none_in_offline_mode(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        assert _resolve_hf_token() is None

    def test_prefers_hf_token(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setenv("HF_TOKEN", "hf_primary")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_secondary")
        assert _resolve_hf_token() == "hf_primary"

    def test_falls_back_to_hub_token(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_secondary")
        assert _resolve_hf_token() == "hf_secondary"

    def test_returns_none_when_no_token_set(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        assert _resolve_hf_token() is None


# ---------------------------------------------------------------------------
# _annotation_from_pipeline_output
# ---------------------------------------------------------------------------

from diarize import _annotation_from_pipeline_output
from pyannote.core import Annotation


class TestAnnotationFromPipelineOutput:
    def test_accepts_annotation_directly(self):
        ann = Annotation()
        assert _annotation_from_pipeline_output(ann) is ann

    def test_extracts_speaker_diarization_attribute(self):
        ann = Annotation()
        result_obj = MagicMock()
        result_obj.speaker_diarization = ann
        assert _annotation_from_pipeline_output(result_obj) is ann

    def test_raises_on_unknown_type(self):
        with pytest.raises(TypeError, match="Unexpected pipeline output"):
            _annotation_from_pipeline_output({"not": "an annotation"})


# ---------------------------------------------------------------------------
# diarize() — pyannote Pipeline is mocked
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration_ms: int = 1000, sample_rate: int = 16000) -> None:
    n_frames = int(sample_rate * duration_ms / 1000)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _fake_annotation(segments: list[tuple[float, float, str]]) -> Annotation:
    """Build a pyannote Annotation from (start, end, speaker) tuples."""
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for start, end, speaker in segments:
        ann[Segment(start, end)] = speaker
    return ann


@pytest.fixture()
def wav_file(tmp_path: Path) -> Path:
    p = tmp_path / "audio.wav"
    _make_wav(p, duration_ms=3000)
    return p


class TestDiarize:
    def _patch_pipeline(self, annotation: Annotation):
        """Patch Pipeline.from_pretrained to return a mock that yields annotation."""
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = annotation  # direct Annotation return
        mock_pipeline_cls = MagicMock(return_value=mock_pipeline_instance)
        return patch("diarize.Pipeline.from_pretrained", mock_pipeline_cls)

    def test_missing_file_raises(self):
        from diarize import diarize
        with pytest.raises(FileNotFoundError):
            diarize("/nonexistent.wav")

    def test_empty_annotation_returns_empty_list(self, wav_file: Path):
        from diarize import diarize
        ann = _fake_annotation([])
        with self._patch_pipeline(ann):
            result = diarize(str(wav_file))
        assert result == []

    def test_segments_sorted_by_start(self, wav_file: Path):
        from diarize import diarize
        ann = _fake_annotation([
            (1.0, 1.5, "SPEAKER_01"),
            (0.0, 0.8, "SPEAKER_00"),
        ])
        with self._patch_pipeline(ann):
            result = diarize(str(wav_file))
        starts = [s["start_time"] for s in result]
        assert starts == sorted(starts)

    def test_result_structure(self, wav_file: Path):
        from diarize import diarize
        ann = _fake_annotation([(0.0, 1.0, "SPEAKER_00")])
        with self._patch_pipeline(ann):
            result = diarize(str(wav_file))
        assert len(result) == 1
        seg = result[0]
        assert seg["start_time"] == pytest.approx(0.0)
        assert seg["end_time"] == pytest.approx(1.0)
        assert seg["speaker"] == "SPEAKER_00"

    def test_multiple_speakers(self, wav_file: Path):
        from diarize import diarize
        ann = _fake_annotation([
            (0.0, 0.5, "SPEAKER_00"),
            (0.6, 1.2, "SPEAKER_01"),
            (1.3, 2.0, "SPEAKER_00"),
        ])
        with self._patch_pipeline(ann):
            result = diarize(str(wav_file))
        speakers = {s["speaker"] for s in result}
        assert speakers == {"SPEAKER_00", "SPEAKER_01"}
        assert len(result) == 3
