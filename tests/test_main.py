"""Tests for main.py — _to_wav, _write_json, _write_markdown, and main() dispatch."""

from __future__ import annotations

import json
import wave
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration_ms: int = 500) -> None:
    n_frames = int(16000 * duration_ms / 1000)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * n_frames)


# ---------------------------------------------------------------------------
# _to_wav
# ---------------------------------------------------------------------------

from main import _to_wav


class TestToWav:
    def test_wav_returned_as_is(self, tmp_path: Path):
        wav = tmp_path / "audio.wav"
        _make_wav(wav)
        result_path, created = _to_wav(wav)
        assert result_path == wav
        assert created is False

    def test_non_wav_converted(self, tmp_path: Path):
        """Conversion requires pydub/ffmpeg; we mock AudioSegment.from_file."""
        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"fake mp3 data")

        mock_audio = MagicMock()
        mock_audio.export = MagicMock()

        with patch("main.AudioSegment.from_file", return_value=mock_audio):
            result_path, created = _to_wav(mp3)

        assert created is True
        assert result_path != mp3
        assert result_path.suffix == ".wav"
        mock_audio.export.assert_called_once()

    def test_wav_not_deleted_by_caller_flag(self, tmp_path: Path):
        wav = tmp_path / "audio.wav"
        _make_wav(wav)
        _, created = _to_wav(wav)
        assert created is False  # caller must NOT delete it


# ---------------------------------------------------------------------------
# _write_json
# ---------------------------------------------------------------------------

from main import _write_json


class TestWriteJson:
    def test_writes_valid_json(self, tmp_path: Path):
        results = [
            {"segmentInfo": {"start_time": 0.0, "end_time": 1.0, "speaker": "A"}, "text": "hi"},
        ]
        out = tmp_path / "out.json"
        _write_json(results, out)
        loaded = json.loads(out.read_text())
        assert loaded == results

    def test_empty_results(self, tmp_path: Path):
        out = tmp_path / "out.json"
        _write_json([], out)
        assert json.loads(out.read_text()) == []

    def test_unicode_preserved(self, tmp_path: Path):
        results = [{"segmentInfo": {}, "text": "héllo 🐔"}]
        out = tmp_path / "out.json"
        _write_json(results, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded[0]["text"] == "héllo 🐔"


# ---------------------------------------------------------------------------
# _write_markdown
# ---------------------------------------------------------------------------

from main import _write_markdown


class TestWriteMarkdown:
    def _results(self) -> list[dict]:
        return [
            {"segmentInfo": {"start_time": 0.0, "end_time": 2.0, "speaker": "SPEAKER_00"}, "text": "Hello."},
            {"segmentInfo": {"start_time": 2.5, "end_time": 4.0, "speaker": "SPEAKER_01"}, "text": "Hi."},
            {"segmentInfo": {"start_time": 5.0, "end_time": 6.0, "speaker": "SPEAKER_00"}, "text": ""},
        ]

    def test_contains_title(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_markdown(self._results(), "audio.mp3", out)
        assert "# Transcript: audio.mp3" in out.read_text()

    def test_contains_generated_timestamp(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_markdown(self._results(), "audio.mp3", out)
        # Should contain "Generated:" and "UTC"
        content = out.read_text()
        assert "Generated" in content
        assert "UTC" in content

    def test_speaker_labels_bold(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_markdown(self._results(), "audio.mp3", out)
        content = out.read_text()
        # Format is **SPEAKER_XX:** (colon inside the bold span)
        assert "**SPEAKER_00:**" in content
        assert "**SPEAKER_01:**" in content

    def test_empty_text_segments_omitted(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_markdown(self._results(), "audio.mp3", out)
        content = out.read_text()
        # The third segment is empty — speaker_00 appears once, not twice
        assert content.count("**SPEAKER_00:**") == 1

    def test_text_content_present(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_markdown(self._results(), "audio.mp3", out)
        content = out.read_text()
        assert "Hello." in content
        assert "Hi." in content


# ---------------------------------------------------------------------------
# main() — dispatch logic
# ---------------------------------------------------------------------------

class TestMainDispatch:
    """Test that main() dispatches correctly between audio and zip modes."""

    def test_nonexistent_file_exits_with_error(self, capsys, tmp_path):
        from main import main
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path / "nope.wav")])
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_zip_mode_calls_process_zip(self, tmp_path: Path):
        from main import main
        # Build a minimal valid zip (content irrelevant — process_zip is mocked)
        zp = tmp_path / "recording.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("info.txt", "Start time: 2026-01-01T00:00:00Z\n")

        mock_process = MagicMock(return_value=(tmp_path / "a.json", tmp_path / "a.md"))
        # process_zip is imported locally inside main(), so patch at source module
        with patch("transcribe_zip.process_zip", mock_process):
            main([str(zp)])

        mock_process.assert_called_once_with(zp)

    def test_audio_mode_calls_diarize_and_transcribe(self, tmp_path: Path):
        from main import main
        wav = tmp_path / "audio.wav"
        _make_wav(wav)

        fake_segments = [{"start_time": 0.0, "end_time": 0.5, "speaker": "SPEAKER_00"}]
        fake_results = [{"segmentInfo": fake_segments[0], "text": "hello"}]

        with (
            patch("main.diarize", return_value=fake_segments) as mock_diarize,
            patch("main.transcribe", return_value=fake_results) as mock_transcribe,
        ):
            main([str(wav)])

        mock_diarize.assert_called_once()
        mock_transcribe.assert_called_once()

    def test_audio_mode_writes_output_files(self, tmp_path: Path):
        from main import main
        wav = tmp_path / "audio.wav"
        _make_wav(wav)

        fake_segments = [{"start_time": 0.0, "end_time": 0.5, "speaker": "SPEAKER_00"}]
        fake_results = [{"segmentInfo": fake_segments[0], "text": "hello"}]

        with (
            patch("main.diarize", return_value=fake_segments),
            patch("main.transcribe", return_value=fake_results),
        ):
            main([str(wav)])

        assert (tmp_path / "audio.json").exists()
        assert (tmp_path / "audio.md").exists()

    def test_audio_mode_no_segments_exits_early(self, tmp_path: Path, capsys):
        from main import main
        wav = tmp_path / "audio.wav"
        _make_wav(wav)

        with patch("main.diarize", return_value=[]):
            main([str(wav)])

        captured = capsys.readouterr()
        assert "No speech segments" in captured.out
        # No output files should be created
        assert not (tmp_path / "audio.json").exists()

    def test_wav_temp_file_cleaned_up_on_success(self, tmp_path: Path):
        """A converted temp WAV should be deleted after transcription."""
        from main import main
        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"fake")

        fake_wav = tmp_path / "fake_temp.wav"
        _make_wav(fake_wav)

        mock_audio = MagicMock()
        mock_audio.export = MagicMock()

        fake_segments = [{"start_time": 0.0, "end_time": 0.5, "speaker": "SPEAKER_00"}]
        fake_results = [{"segmentInfo": fake_segments[0], "text": "hi"}]

        # We need to capture what temp wav path _to_wav returns
        created_paths = []

        original_to_wav = __import__("main")._to_wav

        def fake_to_wav(path):
            _make_wav(fake_wav)
            created_paths.append(fake_wav)
            return fake_wav, True

        with (
            patch("main._to_wav", side_effect=fake_to_wav),
            patch("main.diarize", return_value=fake_segments),
            patch("main.transcribe", return_value=fake_results),
        ):
            main([str(mp3)])

        assert not fake_wav.exists(), "Temp WAV should be deleted after transcription"

    def test_wav_temp_file_cleaned_up_on_diarize_error(self, tmp_path: Path):
        """Temp WAV must be cleaned up even if diarize() raises."""
        from main import main
        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"fake")
        fake_wav = tmp_path / "fake_temp2.wav"
        _make_wav(fake_wav)

        def fake_to_wav(path):
            _make_wav(fake_wav)
            return fake_wav, True

        with (
            patch("main._to_wav", side_effect=fake_to_wav),
            patch("main.diarize", side_effect=RuntimeError("boom")),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                main([str(mp3)])

        assert not fake_wav.exists(), "Temp WAV should be deleted even on error"
