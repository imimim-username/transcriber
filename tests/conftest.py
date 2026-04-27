"""Stub out heavy ML dependencies so the test suite runs without GPU/models.

torch, transformers, pyannote, pydub, tqdm, and dotenv are all injected as
lightweight MagicMock modules before the production code imports them.  Each
test that exercises code touching those libraries patches the relevant objects
at the call site with unittest.mock.patch.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _stub(name: str, **attrs) -> types.ModuleType:
    """Create and register a minimal stub module."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _ensure_not_already_imported(*names: str) -> None:
    for name in names:
        if name in sys.modules:
            del sys.modules[name]


# Order matters — remove any real imports that snuck in before we stub.
_ensure_not_already_imported(
    "torch", "torch.backends", "torch.backends.mps",
    "faster_whisper",
    "pydub", "tqdm", "dotenv",
    "transformers",
    "pyannote", "pyannote.audio", "pyannote.audio.pipelines",
    "pyannote.audio.pipelines.utils", "pyannote.audio.pipelines.utils.hook",
    "pyannote.core",
    "transcribe", "diarize", "transcribe_zip", "main",
)

# ── torch ──────────────────────────────────────────────────────────────────
class _FakeDtype:
    def __init__(self, name: str):
        self._name = name
    def __repr__(self):
        return f"torch.{self._name}"

torch_mod = _stub("torch")
torch_mod.float16 = _FakeDtype("float16")
torch_mod.float32 = _FakeDtype("float32")
torch_mod.device = lambda x: x  # torch.device("cpu") → "cpu"
torch_mod.cuda = MagicMock()
torch_mod.cuda.is_available = MagicMock(return_value=False)

backends_mod = _stub("torch.backends")
mps_mod = _stub("torch.backends.mps")
mps_mod.is_available = MagicMock(return_value=False)
backends_mod.mps = mps_mod
torch_mod.backends = backends_mod

# ── pydub ──────────────────────────────────────────────────────────────────
pydub_mod = _stub("pydub")
pydub_mod.AudioSegment = MagicMock()

# ── tqdm ───────────────────────────────────────────────────────────────────
tqdm_mod = _stub("tqdm")

class _FakeTqdm:
    def __init__(self, *a, **kw):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def update(self, n=1):
        pass
    @staticmethod
    def write(msg):
        pass

tqdm_mod.tqdm = _FakeTqdm

# ── dotenv ─────────────────────────────────────────────────────────────────
dotenv_mod = _stub("dotenv")
dotenv_mod.load_dotenv = MagicMock()

# ── faster_whisper ─────────────────────────────────────────────────────────
faster_whisper_mod = _stub("faster_whisper")
faster_whisper_mod.WhisperModel = MagicMock()

# ── transformers ───────────────────────────────────────────────────────────
transformers_mod = _stub("transformers")
transformers_mod.AutoModelForSpeechSeq2Seq = MagicMock()
transformers_mod.AutoProcessor = MagicMock()
transformers_mod.pipeline = MagicMock()

# ── pyannote ───────────────────────────────────────────────────────────────
pyannote_mod = _stub("pyannote")

# pyannote.core — we need *real* Annotation/Segment for diarize tests
# Provide minimal pure-Python implementations.

class Segment:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end
    def __repr__(self):
        return f"Segment({self.start}, {self.end})"
    def __hash__(self):
        return hash((self.start, self.end))
    def __eq__(self, other):
        return isinstance(other, Segment) and self.start == other.start and self.end == other.end


class Annotation:
    def __init__(self):
        self._tracks: list[tuple[Segment, str]] = []

    def __setitem__(self, segment: Segment, label: str) -> None:
        self._tracks.append((segment, label))

    def itertracks(self, yield_label: bool = False):
        for seg, label in self._tracks:
            if yield_label:
                yield seg, None, label
            else:
                yield seg, None


core_mod = _stub("pyannote.core")
core_mod.Annotation = Annotation
core_mod.Segment = Segment
pyannote_mod.core = core_mod

audio_mod = _stub("pyannote.audio")
audio_mod.Pipeline = MagicMock()
pyannote_mod.audio = audio_mod

pipelines_mod = _stub("pyannote.audio.pipelines")
utils_mod = _stub("pyannote.audio.pipelines.utils")
hook_mod = _stub("pyannote.audio.pipelines.utils.hook")

class _FakeProgressHook:
    def __enter__(self): return self
    def __exit__(self, *a): pass

hook_mod.ProgressHook = _FakeProgressHook
audio_mod.pipelines = pipelines_mod
audio_mod.pipelines.utils = utils_mod
audio_mod.pipelines.utils.hook = hook_mod
