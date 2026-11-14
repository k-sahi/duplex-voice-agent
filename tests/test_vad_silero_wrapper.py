"""Tests the Silero wrapper's framing with a fake model (no torch needed)."""
import numpy as np

from engine.audio import FRAME_BYTES, SILENCE
from server.vad import SILERO_WINDOW, SileroVAD
from tests.test_vad_energy import LOUD


def test_feeds_model_exact_512_sample_windows():
    seen = []
    vad = SileroVAD(prob_fn=lambda w: seen.append(w) or 0.0)
    for _ in range(8):                      # 8 x 320 = 2560 samples = 5 windows
        vad.process(SILENCE)
    assert len(seen) == 5
    assert all(w.shape == (SILERO_WINDOW,) and w.dtype == np.float32 for w in seen)


def test_loud_audio_with_high_probability_starts_and_ends():
    loudness = lambda w: 0.95 if np.abs(w).mean() > 0.01 else 0.02
    vad = SileroVAD(prob_fn=loudness)
    events = [r for f in [LOUD] * 15 + [SILENCE] * 30 if (r := vad.process(f))]
    assert events == ["start", "end"]
    assert vad.end_frames == 20 and len(LOUD) == FRAME_BYTES
