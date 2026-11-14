"""Speech recognition backends. Both expose transcribe(pcm_bytes) -> str.

transcribe() is blocking and CPU-heavy, so the server always calls it through
asyncio.to_thread(): it runs on a worker thread while the tick loop keeps going.
"""
from __future__ import annotations

import os
import time

import numpy as np

from engine.audio import FRAME_BYTES, FRAME_MS


class WhisperASR:
    """faster-whisper (CTranslate2): Whisper re-implemented for fast CPU/GPU inference."""

    def __init__(self, model_size: str = "base.en", device: str = "cpu",
                 compute_type: str = "int8") -> None:
        from faster_whisper import WhisperModel

        # num_workers=2 lets a partial and a final transcription run at the same time,
        # so the final never waits behind a partial that is still running.
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                  num_workers=2)
        self.name = f"whisper-{model_size}-{device}-{compute_type}"

    def transcribe(self, pcm: bytes) -> str:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,                       # greedy decoding: fastest
            vad_filter=False,                  # we already cut the turn with our own VAD
            without_timestamps=True,
            condition_on_previous_text=False,  # each call stands alone
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def warmup(self) -> float:
        """The first call is slow (memory allocation). Do it before anyone talks."""
        t0 = time.perf_counter()
        self.transcribe(bytes(FRAME_BYTES * 50))  # 1 s of silence
        return (time.perf_counter() - t0) * 1000


class LengthASR:
    """No model: 'transcribes' audio as its length. Used for tests and as a fallback."""

    name = "length-only"

    def transcribe(self, pcm: bytes) -> str:
        ms = len(pcm) // FRAME_BYTES * FRAME_MS
        return f"[{ms} ms of speech]"

    def warmup(self) -> float:
        return 0.0


def load_asr():
    """Configure with DUPLEX_ASR_MODEL (tiny.en, base.en, small.en...) or DUPLEX_ASR=none."""
    if os.environ.get("DUPLEX_ASR", "whisper") == "whisper":
        try:
            asr = WhisperASR(os.environ.get("DUPLEX_ASR_MODEL", "base.en"))
            print(f"[asr] loaded {asr.name}, warmup {asr.warmup():.0f} ms")
            return asr
        except ImportError as err:
            print(f"[asr] faster-whisper unavailable ({err}); using length-only stand-in")
    return LengthASR()
