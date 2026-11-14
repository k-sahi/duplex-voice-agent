"""Voice activity detection backends. Both expose process(frame) -> 'start' | 'end' | None.

- SileroVAD: small neural model, robust to noise, typing and breathing.
- EnergyVAD (engine/vad_energy.py): loudness only, no dependencies, used as fallback.
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np

from engine.audio import SAMPLE_RATE
from engine.speech_gate import SpeechGate
from engine.vad_energy import EnergyVAD

SILERO_WINDOW = 512  # Silero v5 needs exactly 512 samples (32 ms) per call at 16 kHz


def load_silero_prob_fn() -> Callable[[np.ndarray], float]:
    """Load the model once per connection (it keeps internal state between calls)."""
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)  # tiny model; extra threads only add overhead
    model = load_silero_vad()

    def prob(window: np.ndarray) -> float:
        with torch.no_grad():
            return float(model(torch.from_numpy(window), SAMPLE_RATE).item())

    return prob


class SileroVAD:
    def __init__(self, prob_fn: Callable[[np.ndarray], float] | None = None,
                 gate: SpeechGate | None = None) -> None:
        self._prob = prob_fn or load_silero_prob_fn()
        self.gate = gate or SpeechGate()
        self._pending = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0

    @property
    def speaking(self) -> bool:
        return self.gate.speaking

    @property
    def end_frames(self) -> int:
        return self.gate.end_frames

    def process(self, frame: bytes) -> str | None:
        # Our frames are 320 samples, Silero wants 512: collect, then run on full windows.
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, samples])
        while len(self._pending) >= SILERO_WINDOW:
            window = np.ascontiguousarray(self._pending[:SILERO_WINDOW])
            self._pending = self._pending[SILERO_WINDOW:]
            self.last_prob = self._prob(window)
        return self.gate.update(self.last_prob)


def make_vad():
    """Silero if installed (or DUPLEX_VAD=energy to force the simple one)."""
    if os.environ.get("DUPLEX_VAD", "silero") == "silero":
        try:
            return SileroVAD()
        except ImportError as err:
            print(f"[vad] Silero unavailable ({err}); falling back to energy VAD")
    return EnergyVAD()
