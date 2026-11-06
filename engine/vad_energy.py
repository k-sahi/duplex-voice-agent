"""Simple energy-based voice activity detector (placeholder until Silero VAD in Step 5).

Decides per 20 ms frame whether someone is speaking, with two safeguards:
- start needs most of the last 200 ms to be loud (a click or cough is ignored)
- end needs 500 ms of quiet (short pauses between words don't end the turn)
"""
from __future__ import annotations

import math
from array import array
from collections import deque


def frame_rms(frame: bytes) -> float:
    samples = array("h", frame)  # int16, little-endian on all common machines
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class EnergyVAD:
    def __init__(self, start_frames: int = 10, start_ratio: float = 0.8,
                 end_frames: int = 25, min_rms: float = 300.0, ratio: float = 3.0) -> None:
        self.start_frames = start_frames  # look-back window for starting (10 x 20 ms = 200 ms)
        self.start_ratio = start_ratio    # fraction of that window that must be loud
        self.end_frames = end_frames      # quiet frames needed to end (25 x 20 ms = 500 ms)
        self.min_rms = min_rms            # never treat anything quieter than this as speech
        self.ratio = ratio                # speech must be this many times above the noise floor
        self.noise_floor = 100.0
        self.speaking = False
        self._history: deque[bool] = deque(maxlen=start_frames)
        self._quiet = 0

    def threshold(self) -> float:
        return max(self.min_rms, self.noise_floor * self.ratio)

    def process(self, frame: bytes) -> str | None:
        """Feed one frame. Returns 'start', 'end' or None."""
        rms = frame_rms(frame)
        loud = rms > self.threshold()

        if not self.speaking:
            if not loud:  # learn the room's background level only from quiet frames
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
            self._history.append(loud)
            if (len(self._history) == self.start_frames
                    and sum(self._history) >= self.start_ratio * self.start_frames):
                self.speaking = True
                self._quiet = 0
                return "start"
            return None

        self._quiet = 0 if loud else self._quiet + 1
        if self._quiet >= self.end_frames:
            self.speaking = False
            self._history.clear()
            return "end"
        return None
