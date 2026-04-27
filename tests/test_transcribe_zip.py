"""Tests for transcribe_zip.py — pure helpers, I/O writers, and process_zip()."""

from __future__ import annotations

import json
import wave
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# parse_info_txt
# ---------------------------------------------------------------------------

from transcribe_zip import parse_info_txt


class TestParseInfoTxt:
    def test_valid_start_time(self, tmp_path: Path):
        info = tmp_path / "info.txt"
        info.write_text("Start time:  2026-04-27T11:51:12.426Z\n")
        result = parse_info_txt(info)
        assert result["date_str"] == "2026-04-27"
        assert result["start_time_raw"] == "2026-04-27T11:51:12.426Z"

    def test_extra_whitespace(self, tmp_path: Path):
        info = tmp_path / "info.txt"
        info.write_text("Start time:    2026-01-01T00:00:00Z\n")
        result = parse_info_txt(info)
        assert result["date_str"] == "2026-01-01"

    def test_missing_file_falls_back_to_today(self, tmp_path: Path):
        from datetime import datetime, timezone
        info = tmp_path / "missing.txt"
        result = parse_info_txt(info)  # file does not exist
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert result["date_str"] == today

    def test_no_start_time_line_falls_back_to_today(self, tmp_path: Path):
        from datetime import datetime, timezone
        info = tmp_path / "info.txt"
        info.write_text("Room: conf-a\nParticipants: 3\n")
        result = parse_info_txt(info)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert result["date_str"] == today

    def test_other_metadata_ignored(self, tmp_path: Path):
        info = tmp_path / "info.txt"
        info.write_text(
            "Room: board-room\n"
            "Start time:  2025-12-31T23:59:59.000Z\n"
            "Duration: 3600\n"
        )
        result = parse_info_txt(info)
        assert result["date_str"] == "2025-12-31"


# ---------------------------------------------------------------------------
# _speaker_from_filename
# ---------------------------------------------------------------------------

from transcribe_zip import _speaker_from_filename


class TestSpeakerFromFilename:
    def test_standard_format(self):
        assert _speaker_from_filename("1-alice.aac") == "alice"

    def test_multichar_number(self):
        assert _speaker_from_filename("12-bob.ogg") == "bob"

    def test_speaker_with_spaces(self):
        # Filenames don't normally have spaces but the function shouldn't crash
        assert _speaker_from_filename("3-john doe.wav") == "john doe"

    def test_no_number_prefix_returns_stem(self):
        assert _speaker_from_filename("carol.mp3") == "carol"

    def test_speaker_name_preserved_case(self):
        assert _speaker_from_filename("2-Alice.m4a") == "Alice"

    def test_hyphen_in_speaker_name(self):
        # "1-mary-jane.aac" — regex matches first number group, rest is speaker
        assert _speaker_from_filename("1-mary-jane.aac") == "mary-jane"


# ---------------------------------------------------------------------------
# _write_json
# ---------------------------------------------------------------------------

from transcribe_zip import _write_json


class TestWriteJson:
    def test_creates_valid_json_file(self, tmp_path: Path):
        segs = [
            {"segmentInfo": {"start_time": 0.0, "end_time": 1.0, "speaker": "alice"}, "text": "hello"},
        ]
        out = tmp_path / "out.json"
        _write_json(out, segs)
        loaded = json.loads(out.read_text())
        assert loaded == segs

    def test_empty_list(self, tmp_path: Path):
        out = tmp_path / "out.json"
        _write_json(out, [])
        assert json.loads(out.read_text()) == []

    def test_utf8_text(self, tmp_path: Path):
        segs = [{"segmentInfo": {"start_time": 0.0, "end_time": 1.0, "speaker": "s"}, "text": "héllo wörld 🐔"}]
        out = tmp_path / "out.json"
        _write_json(out, segs)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded[0]["text"] == "héllo wörld 🐔"


# ---------------------------------------------------------------------------
# _write_md
# ---------------------------------------------------------------------------

from transcribe_zip import _write_md


class TestWriteMd:
    def _segs(self) -> list[dict]:
        return [
            {"segmentInfo": {"start_time": 0.0, "end_time": 2.5, "speaker": "alice"}, "text": "Hello there."},
            {"segmentInfo": {"start_time": 3.0, "end_time": 5.0, "speaker": "bob"}, "text": "Hi back."},
        ]

    def test_contains_title(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_md(out, self._segs(), "recording.zip", "2026-04-27T11:00:00Z")
        content = out.read_text()
        assert "# Transcript: recording.zip" in content

    def test_contains_start_time(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_md(out, self._segs(), "recording.zip", "2026-04-27T11:00:00Z")
        content = out.read_text()
        assert "2026-04-27T11:00:00Z" in content

    def test_contains_speaker_labels(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_md(out, self._segs(), "r.zip", "2026-04-27T00:00:00Z")
        content = out.read_text()
        # Format is **speaker:** (colon inside the bold span)
        assert "**alice:**" in content
        assert "**bob:**" in content

    def test_contains_text(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_md(out, self._segs(), "r.zip", "2026-04-27T00:00:00Z")
        content = out.read_text()
        assert "Hello there." in content
        assert "Hi back." in content

    def test_timestamp_format(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_md(out, self._segs(), "r.zip", "2026-04-27T00:00:00Z")
        content = out.read_text()
        # Start of first segment: 0.0 → 00:00.000
        assert "00:00.000" in content
        assert "00:02.500" in content

    def test_empty_segments(self, tmp_path: Path):
        out = tmp_path / "out.md"
        _write_md(out, [], "r.zip", "2026-04-27T00:00:00Z")
        content = out.read_text()
        assert "# Transcript:" in content  # header still written


# ---------------------------------------------------------------------------
# _transcribe_track
# ---------------------------------------------------------------------------

from transcribe_zip import _transcribe_track


class TestTranscribeTrack:
    def _make_wav(self, path: Path) -> None:
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)

    def _make_pipe(self, chunks: list[dict]) -> MagicMock:
        mock = MagicMock()
        mock.return_value = {"chunks": chunks}
        return mock

    def test_basic_segments_returned(self, tmp_path: Path):
        audio = tmp_path / "track.wav"
        self._make_wav(audio)
        pipe = self._make_pipe([
            {"text": "hello", "timestamp": (0.0, 1.5)},
            {"text": "world", "timestamp": (2.0, 3.0)},
        ])
        result = _transcribe_track(audio, "alice", pipe)
        assert len(result) == 2
        assert result[0]["text"] == "hello"
        assert result[0]["segmentInfo"]["speaker"] == "alice"
        assert result[1]["text"] == "world"

    def test_empty_text_chunks_skipped(self, tmp_path: Path):
        audio = tmp_path / "track.wav"
        self._make_wav(audio)
        pipe = self._make_pipe([
            {"text": "   ", "timestamp": (0.0, 1.0)},
            {"text": "real text", "timestamp": (1.0, 2.0)},
        ])
        result = _transcribe_track(audio, "bob", pipe)
        assert len(result) == 1
        assert result[0]["text"] == "real text"

    def test_none_end_timestamp_falls_back_to_start(self, tmp_path: Path):
        audio = tmp_path / "track.wav"
        self._make_wav(audio)
        pipe = self._make_pipe([
            {"text": "final chunk", "timestamp": (5.0, None)},
        ])
        result = _transcribe_track(audio, "carol", pipe)
        assert result[0]["segmentInfo"]["end_time"] == pytest.approx(5.0)

    def test_no_chunks_returns_empty(self, tmp_path: Path):
        audio = tmp_path / "track.wav"
        self._make_wav(audio)
        pipe = self._make_pipe([])
        result = _transcribe_track(audio, "dave", pipe)
        assert result == []

    def test_segment_info_times(self, tmp_path: Path):
        audio = tmp_path / "track.wav"
        self._make_wav(audio)
        pipe = self._make_pipe([{"text": "hi", "timestamp": (1.23, 4.56)}])
        result = _transcribe_track(audio, "eve", pipe)
        info = result[0]["segmentInfo"]
        assert info["start_time"] == pytest.approx(1.23)
        assert info["end_time"] == pytest.approx(4.56)


# ---------------------------------------------------------------------------
# Progress output — _transcribe_track and process_zip
# ---------------------------------------------------------------------------

class TestProgressOutput:
    """Verify that progress text is emitted in the expected format."""

    def _make_wav(self, path: Path) -> None:
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)

    def test_running_inference_message_emitted(self, tmp_path: Path):
        import transcribe_zip
        audio = tmp_path / "track.wav"
        self._make_wav(audio)
        pipe = MagicMock(return_value={"chunks": []})

        written: list[str] = []
        with patch.object(transcribe_zip.tqdm, "write", side_effect=written.append):
            _transcribe_track(audio, "alice", pipe)

        assert any("Running Whisper inference" in m for m in written)

    def test_inference_message_printed_before_pipe_called(self, tmp_path: Path):
        """'Running Whisper inference…' must appear before the pipe is invoked."""
        import transcribe_zip
        audio = tmp_path / "track.wav"
        self._make_wav(audio)

        call_order: list[str] = []

        def fake_write(msg: str) -> None:
            if "Running Whisper" in msg:
                call_order.append("write")

        def fake_pipe(path: str, **kwargs: Any) -> dict:
            call_order.append("pipe")
            return {"chunks": []}

        with patch.object(transcribe_zip.tqdm, "write", side_effect=fake_write):
            _transcribe_track(audio, "alice", fake_pipe)

        assert call_order == ["write", "pipe"]

    def test_chunk_counter_format(self, tmp_path: Path):
        """Segment lines include [i/total] chunk counter."""
        import transcribe_zip
        audio = tmp_path / "track.wav"
        self._make_wav(audio)
        pipe = MagicMock(return_value={"chunks": [
            {"text": "first",  "timestamp": (0.0, 1.0)},
            {"text": "second", "timestamp": (1.0, 2.0)},
        ]})

        written: list[str] = []
        with patch.object(transcribe_zip.tqdm, "write", side_effect=written.append):
            _transcribe_track(audio, "alice", pipe)

        segment_lines = [m for m in written if "→" in m]
        assert any("[1/2]" in m for m in segment_lines)
        assert any("[2/2]" in m for m in segment_lines)

    def test_chunk_counter_includes_speaker(self, tmp_path: Path):
        import transcribe_zip
        audio = tmp_path / "track.wav"
        self._make_wav(audio)
        pipe = MagicMock(return_value={"chunks": [
            {"text": "hello", "timestamp": (0.0, 1.0)},
        ]})

        written: list[str] = []
        with patch.object(transcribe_zip.tqdm, "write", side_effect=written.append):
            _transcribe_track(audio, "gorby", pipe)

        segment_lines = [m for m in written if "→" in m]
        assert any("gorby" in m for m in segment_lines)

    def test_process_zip_track_header_has_counter(self, tmp_path: Path):
        """Track section headers include [idx/total] counter."""
        import transcribe_zip
        from transcribe_zip import process_zip

        wav = _make_wav_bytes()
        zp = _make_zip(tmp_path, [
            ("1-alice.wav", wav),
            ("2-bob.wav",   wav),
        ])

        written: list[str] = []
        with patch("transcribe_zip._load_model", return_value=_make_mock_pipe()):
            with patch.object(transcribe_zip.tqdm, "write", side_effect=written.append):
                process_zip(zp)

        headers = [m for m in written if "──" in m]
        assert any("[1/2]" in h for h in headers)
        assert any("[2/2]" in h for h in headers)

    def test_process_zip_track_header_includes_speaker_and_filename(self, tmp_path: Path):
        import transcribe_zip
        from transcribe_zip import process_zip

        wav = _make_wav_bytes()
        zp = _make_zip(tmp_path, [("1-alice.wav", wav)])

        written: list[str] = []
        with patch("transcribe_zip._load_model", return_value=_make_mock_pipe()):
            with patch.object(transcribe_zip.tqdm, "write", side_effect=written.append):
                process_zip(zp)

        headers = [m for m in written if "──" in m]
        assert any("alice" in h for h in headers)
        assert any("1-alice.wav" in h for h in headers)


# ---------------------------------------------------------------------------
# process_zip() — full integration with mocked model
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_ms: int = 500, sample_rate: int = 16000) -> bytes:
    """Return raw bytes of a minimal silent WAV."""
    import io
    n_frames = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def _make_zip(
    tmp_path: Path,
    tracks: list[tuple[str, bytes]],
    info_content: str | None = "Start time:  2026-04-27T11:51:12.426Z\n",
) -> Path:
    """Create a zip file with the given tracks (name, bytes) and optional info.txt."""
    zp = tmp_path / "recording.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        if info_content is not None:
            zf.writestr("info.txt", info_content)
        for name, data in tracks:
            zf.writestr(name, data)
    return zp


def _make_mock_pipe(chunks: list[dict] | None = None) -> MagicMock:
    if chunks is None:
        chunks = [{"text": "test transcription", "timestamp": (0.0, 1.0)}]
    mock = MagicMock()
    mock.return_value = {"chunks": chunks}
    return mock


class TestProcessZip:
    def test_missing_zip_raises(self, tmp_path: Path):
        from transcribe_zip import process_zip
        with pytest.raises(FileNotFoundError):
            process_zip(tmp_path / "nope.zip")

    def test_no_audio_files_raises(self, tmp_path: Path):
        from transcribe_zip import process_zip
        zp = _make_zip(tmp_path, [], info_content="Start time: 2026-04-27T00:00:00Z\n")
        with patch("transcribe_zip._load_model", return_value=_make_mock_pipe()):
            with pytest.raises(RuntimeError, match="No audio files found"):
                process_zip(zp)

    def test_produces_json_and_md(self, tmp_path: Path):
        from transcribe_zip import process_zip
        wav = _make_wav_bytes()
        zp = _make_zip(tmp_path, [("1-alice.wav", wav)])
        mock_pipe = _make_mock_pipe()

        with patch("transcribe_zip._load_model", return_value=mock_pipe):
            json_path, md_path = process_zip(zp)

        assert json_path.exists()
        assert md_path.exists()

    def test_output_filenames_use_date_from_info(self, tmp_path: Path):
        from transcribe_zip import process_zip
        wav = _make_wav_bytes()
        zp = _make_zip(tmp_path, [("1-alice.wav", wav)],
                       info_content="Start time:  2026-07-04T12:00:00Z\n")
        with patch("transcribe_zip._load_model", return_value=_make_mock_pipe()):
            json_path, md_path = process_zip(zp)

        assert json_path.name == "meeting-2026-07-04.json"
        assert md_path.name == "meeting-2026-07-04.md"

    def test_output_written_next_to_zip(self, tmp_path: Path):
        from transcribe_zip import process_zip
        wav = _make_wav_bytes()
        zp = _make_zip(tmp_path, [("1-alice.wav", wav)])
        with patch("transcribe_zip._load_model", return_value=_make_mock_pipe()):
            json_path, md_path = process_zip(zp)

        assert json_path.parent == tmp_path
        assert md_path.parent == tmp_path

    def test_json_contains_segments(self, tmp_path: Path):
        from transcribe_zip import process_zip
        wav = _make_wav_bytes()
        zp = _make_zip(tmp_path, [("1-alice.wav", wav)])
        chunks = [{"text": "hello zip", "timestamp": (0.0, 1.5)}]
        with patch("transcribe_zip._load_model", return_value=_make_mock_pipe(chunks)):
            json_path, _ = process_zip(zp)

        data = json.loads(json_path.read_text())
        assert len(data) == 1
        assert data[0]["text"] == "hello zip"
        assert data[0]["segmentInfo"]["speaker"] == "alice"

    def test_segments_merged_chronologically(self, tmp_path: Path):
        from transcribe_zip import process_zip
        wav = _make_wav_bytes()
        zp = _make_zip(tmp_path, [
            ("1-alice.wav", wav),
            ("2-bob.wav", wav),
        ])

        # alice has a late chunk, bob has an early chunk → should interleave
        alice_chunks = [{"text": "I said this second", "timestamp": (2.0, 3.0)}]
        bob_chunks = [{"text": "I said this first", "timestamp": (0.5, 1.5)}]
        call_count = [0]

        def pipe_factory_side_effect(audio_path_str, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"chunks": alice_chunks}
            return {"chunks": bob_chunks}

        mock_pipe = MagicMock(side_effect=pipe_factory_side_effect)

        with patch("transcribe_zip._load_model", return_value=mock_pipe):
            json_path, _ = process_zip(zp)

        data = json.loads(json_path.read_text())
        starts = [s["segmentInfo"]["start_time"] for s in data]
        assert starts == sorted(starts), "Segments should be sorted by start_time"
        assert data[0]["segmentInfo"]["speaker"] == "bob"
        assert data[1]["segmentInfo"]["speaker"] == "alice"

    def test_speaker_name_extracted_from_filename(self, tmp_path: Path):
        from transcribe_zip import process_zip
        wav = _make_wav_bytes()
        zp = _make_zip(tmp_path, [("3-gorby.wav", wav)])
        chunks = [{"text": "hi", "timestamp": (0.0, 1.0)}]
        with patch("transcribe_zip._load_model", return_value=_make_mock_pipe(chunks)):
            json_path, _ = process_zip(zp)

        data = json.loads(json_path.read_text())
        assert data[0]["segmentInfo"]["speaker"] == "gorby"

    def test_temp_dir_cleaned_up(self, tmp_path: Path):
        """After process_zip, no temp_transcriber_zip_ directories should remain."""
        import tempfile
        from transcribe_zip import process_zip

        existing_tmps = set(Path(tempfile.gettempdir()).glob("transcriber_zip_*"))
        wav = _make_wav_bytes()
        zp = _make_zip(tmp_path, [("1-alice.wav", wav)])
        with patch("transcribe_zip._load_model", return_value=_make_mock_pipe()):
            process_zip(zp)

        remaining = set(Path(tempfile.gettempdir()).glob("transcriber_zip_*")) - existing_tmps
        assert remaining == set(), f"Temp dirs not cleaned up: {remaining}"

    def test_missing_info_txt_falls_back_to_today(self, tmp_path: Path):
        from datetime import datetime, timezone
        from transcribe_zip import process_zip

        wav = _make_wav_bytes()
        zp = _make_zip(tmp_path, [("1-alice.wav", wav)], info_content=None)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with patch("transcribe_zip._load_model", return_value=_make_mock_pipe()):
            json_path, _ = process_zip(zp)

        assert json_path.name == f"meeting-{today}.json"
