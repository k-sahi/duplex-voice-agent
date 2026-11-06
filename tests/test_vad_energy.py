import math
import struct

from engine.audio import FRAME_SAMPLES, SAMPLE_RATE, SILENCE
from engine.vad_energy import EnergyVAD


def tone_frame(amp: int = 4000, freq: float = 220.0) -> bytes:
    return struct.pack(f"<{FRAME_SAMPLES}h", *(
        int(amp * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(FRAME_SAMPLES)))


LOUD = tone_frame()


def feed(vad, frames):
    return [r for r in (vad.process(f) for f in frames) if r]


def test_silence_never_starts():
    assert feed(EnergyVAD(), [SILENCE] * 100) == []


def test_speech_starts_then_ends_after_500ms_quiet():
    vad = EnergyVAD()
    assert feed(vad, [SILENCE] * 20 + [LOUD] * 15) == ["start"]
    assert feed(vad, [SILENCE] * 24) == []      # 480 ms: still the same turn
    assert feed(vad, [SILENCE]) == ["end"]       # 500 ms


def test_short_click_is_ignored():
    assert feed(EnergyVAD(), [SILENCE] * 20 + [LOUD] * 3 + [SILENCE] * 20) == []


def test_short_pause_does_not_end_turn():
    vad = EnergyVAD()
    events = feed(vad, [LOUD] * 15 + [SILENCE] * 10 + [LOUD] * 10)
    assert events == ["start"] and vad.speaking
