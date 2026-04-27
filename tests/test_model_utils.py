"""Tests for model_utils.py — offline(), FasterWhisperAdapter, and load_whisper()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# offline()
# ---------------------------------------------------------------------------

from model_utils import offline


class TestOffline:
    def test_default_is_online(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        assert offline() is False

    def test_hf_hub_offline_flag(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        assert offline() is True

    def test_transformers_offline_flag(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        assert offline() is True

    def test_zero_value_is_online(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
        assert offline() is False


# ---------------------------------------------------------------------------
# FasterWhisperAdapter — unit tests for the CPU-path adapter
# ---------------------------------------------------------------------------

import model_utils
from model_utils import FasterWhisperAdapter


def _make_fw_segment(text: str, start: float, end: float) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    return seg


class TestFasterWhisperAdapter:
    def test_returns_chunks_when_return_timestamps_true(self):
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (
            [_make_fw_segment(" hello", 0.0, 1.0)],
            MagicMock(),
        )
        adapter = FasterWhisperAdapter(model_mock)
        result = adapter("audio.wav", return_timestamps=True)
        assert result == {"chunks": [{"text": " hello", "timestamp": (0.0, 1.0)}]}

    def test_returns_text_when_return_timestamps_false(self):
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (
            [
                _make_fw_segment(" hello", 0.0, 1.0),
                _make_fw_segment(" world", 1.0, 2.0),
            ],
            MagicMock(),
        )
        adapter = FasterWhisperAdapter(model_mock)
        result = adapter("audio.wav", return_timestamps=False)
        assert result == {"text": "hello world"}

    def test_default_is_no_timestamps(self):
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (
            [_make_fw_segment(" hi", 0.0, 0.5)],
            MagicMock(),
        )
        adapter = FasterWhisperAdapter(model_mock)
        result = adapter("audio.wav")
        assert "text" in result
        assert "chunks" not in result

    def test_vad_filter_always_enabled(self):
        model_mock = MagicMock()
        model_mock.transcribe.return_value = ([], MagicMock())
        adapter = FasterWhisperAdapter(model_mock)
        adapter("audio.wav")
        kwargs = model_mock.transcribe.call_args.kwargs
        assert kwargs.get("vad_filter") is True

    def test_english_language_forced(self):
        model_mock = MagicMock()
        model_mock.transcribe.return_value = ([], MagicMock())
        adapter = FasterWhisperAdapter(model_mock)
        adapter("audio.wav")
        kwargs = model_mock.transcribe.call_args.kwargs
        assert kwargs.get("language") == "en"
        assert kwargs.get("task") == "transcribe"


# ---------------------------------------------------------------------------
# load_whisper() — HuggingFace path (CUDA / MPS)
# ---------------------------------------------------------------------------

class TestLoadWhisperHF:
    """Tests for the HuggingFace pipeline path.  Force cuda=True to ensure the
    HF branch is taken regardless of the host machine's hardware."""

    def _cuda_context(self, *, monkeypatch=None):
        """Return a tuple of patches that put us on the CUDA path."""
        return (
            patch.object(model_utils.torch.cuda, "is_available", return_value=True),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
        )

    def test_cuda_device_selected(self):
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=True),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
            assert model_utils.pipeline.call_args.kwargs["device"] == "cuda:0"

    def test_mps_device_selected(self):
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=True),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
            assert model_utils.pipeline.call_args.kwargs["device"] == "mps"

    def test_from_pretrained_called_with_model_id(self):
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=True),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock
            model_utils.load_whisper("my-model-id")
            assert mock_model_cls.from_pretrained.call_args.args[0] == "my-model-id"

    def test_local_files_only_set_when_offline(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=True),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
            kwargs = mock_model_cls.from_pretrained.call_args.kwargs
            assert kwargs.get("local_files_only") is True

    def test_local_files_only_false_when_online(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=True),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
            kwargs = mock_model_cls.from_pretrained.call_args.kwargs
            assert kwargs.get("local_files_only") is False

    def test_cuda_uses_sdpa_attention(self):
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=True),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
            kwargs = mock_model_cls.from_pretrained.call_args.kwargs
            assert kwargs.get("attn_implementation") == "sdpa"

    def test_mps_does_not_use_sdpa(self):
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=True),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
            kwargs = mock_model_cls.from_pretrained.call_args.kwargs
            assert "attn_implementation" not in kwargs

    def test_returns_pipeline(self):
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=True),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock
            result = model_utils.load_whisper("openai/whisper-large-v3-turbo")
            assert result is pipe_mock

    def test_english_language_forced_via_generation_config(self):
        """language/task must be set on model.generation_config, not in pipeline kwargs."""
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=True),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
            assert model_mock.generation_config.language == "en"
            assert model_mock.generation_config.task == "transcribe"
            assert "generate_kwargs" not in model_utils.pipeline.call_args.kwargs


# ---------------------------------------------------------------------------
# load_whisper() — faster-whisper CPU path
# ---------------------------------------------------------------------------

import faster_whisper as _fw_mod  # the stub registered by conftest.py


class TestLoadWhisperCPU:
    """Tests for the faster-whisper path (no CUDA, no MPS)."""

    def _cpu_patches(self):
        return (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
        )

    def test_cpu_returns_faster_whisper_adapter(self):
        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch("faster_whisper.WhisperModel"),
        ):
            result = model_utils.load_whisper("openai/whisper-large-v3-turbo")
        assert isinstance(result, FasterWhisperAdapter)

    def test_cpu_does_not_call_hf_pipeline(self):
        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch("faster_whisper.WhisperModel"),
            patch.object(model_utils, "pipeline") as mock_pipeline,
        ):
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
        mock_pipeline.assert_not_called()

    def test_cpu_maps_known_hf_model_id(self):
        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch("faster_whisper.WhisperModel") as mock_wm,
        ):
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
        assert mock_wm.call_args.args[0] == "large-v3-turbo"

    def test_cpu_unknown_model_id_passes_through(self):
        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch("faster_whisper.WhisperModel") as mock_wm,
        ):
            model_utils.load_whisper("Systran/faster-whisper-large-v3-turbo")
        assert mock_wm.call_args.args[0] == "Systran/faster-whisper-large-v3-turbo"

    def test_cpu_uses_int8_compute_type(self):
        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch("faster_whisper.WhisperModel") as mock_wm,
        ):
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
        assert mock_wm.call_args.kwargs.get("compute_type") == "int8"

    def test_cpu_local_files_only_when_offline(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch("faster_whisper.WhisperModel") as mock_wm,
        ):
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
        assert mock_wm.call_args.kwargs.get("local_files_only") is True

    def test_cpu_local_files_only_false_when_online(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch("faster_whisper.WhisperModel") as mock_wm,
        ):
            model_utils.load_whisper("openai/whisper-large-v3-turbo")
        assert mock_wm.call_args.kwargs.get("local_files_only") is False
