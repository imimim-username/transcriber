"""Tests for model_utils.py — offline() and load_whisper()."""

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
# load_whisper() — device selection and model loading calls
# ---------------------------------------------------------------------------

import model_utils


class TestLoadWhisper:
    """Verify device selection and that the right HF calls are made."""

    def _patch_all(self, *, cuda: bool = False, mps: bool = False):
        """Return a context-manager that patches torch availability + HF APIs."""
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        return (
            patch.object(model_utils.torch.cuda, "is_available", return_value=cuda),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=mps),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq"),
            patch.object(model_utils, "AutoProcessor"),
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
            model_mock,
            processor_mock,
            pipe_mock,
        )

    def test_cpu_device_selected(self):
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock

            result = model_utils.load_whisper("openai/whisper-large-v3-turbo")

            # pipeline called with cpu device
            call_kwargs = model_utils.pipeline.call_args
            assert call_kwargs.kwargs.get("device") == "cpu" or (
                len(call_kwargs.args) == 0 and call_kwargs.kwargs["device"] == "cpu"
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

            # pipeline called with cuda:0
            call_kwargs = model_utils.pipeline.call_args.kwargs
            assert call_kwargs["device"] == "cuda:0"

    def test_from_pretrained_called_with_model_id(self):
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock

            model_utils.load_whisper("my-model-id")

            mock_model_cls.from_pretrained.assert_called_once()
            call_args = mock_model_cls.from_pretrained.call_args
            assert call_args.args[0] == "my-model-id"

    def test_local_files_only_set_when_offline(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock

            model_utils.load_whisper("openai/whisper-large-v3-turbo")

            call_kwargs = mock_model_cls.from_pretrained.call_args.kwargs
            assert call_kwargs.get("local_files_only") is True

    def test_local_files_only_false_when_online(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock

            model_utils.load_whisper("openai/whisper-large-v3-turbo")

            call_kwargs = mock_model_cls.from_pretrained.call_args.kwargs
            assert call_kwargs.get("local_files_only") is False

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

            call_kwargs = mock_model_cls.from_pretrained.call_args.kwargs
            assert call_kwargs.get("attn_implementation") == "sdpa"

    def test_cpu_does_not_use_sdpa(self):
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock

            model_utils.load_whisper("openai/whisper-large-v3-turbo")

            call_kwargs = mock_model_cls.from_pretrained.call_args.kwargs
            assert "attn_implementation" not in call_kwargs

    def test_returns_pipeline(self):
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock

            result = model_utils.load_whisper("openai/whisper-large-v3-turbo")

            assert result is pipe_mock

    def test_english_language_forced(self):
        """language='en' and task='transcribe' must be set on model.generation_config.

        We set them there (not via pipeline generate_kwargs=) to avoid the
        duplicate-logits-processor warning that transformers emits when both the
        pipeline and Whisper's internal .generate() try to create the same
        SuppressTokens processors.
        """
        model_mock = MagicMock()
        processor_mock = MagicMock()
        pipe_mock = MagicMock()

        with (
            patch.object(model_utils.torch.cuda, "is_available", return_value=False),
            patch.object(model_utils.torch.backends.mps, "is_available", return_value=False),
            patch.object(model_utils, "AutoModelForSpeechSeq2Seq") as mock_model_cls,
            patch.object(model_utils, "AutoProcessor") as mock_proc_cls,
            patch.object(model_utils, "pipeline", return_value=pipe_mock),
        ):
            mock_model_cls.from_pretrained.return_value = model_mock
            mock_proc_cls.from_pretrained.return_value = processor_mock

            model_utils.load_whisper("openai/whisper-large-v3-turbo")

            # language/task go on generation_config, NOT in pipeline generate_kwargs
            assert model_mock.generation_config.language == "en"
            assert model_mock.generation_config.task == "transcribe"
            pipeline_kwargs = model_utils.pipeline.call_args.kwargs
            assert "generate_kwargs" not in pipeline_kwargs
