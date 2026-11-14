"""Turns per-frame speech probabilities into clean 'start' / 'end' decisions.

A neural VAD (Silero) gives a probability from 0 to 1 that each slice of audio
is speech. Raw probabilities flicker, so this gate adds:
- hysteresis: start above 0.5, end only below 0.35 (in between = no change)
- a start window: 100 ms of confident speech, so a click doesn't start a turn
- an end window: 400 ms of quiet, so a pause between words doesn't end it

Pure logic, no model: easy to test and to port.
"""
from __future__ import annotations


class SpeechGate:
    def __init__(self, start_threshold: float = 0.5, end_threshold: float = 0.35,
                 start_frames: int = 5, end_frames: int = 20) -> None:
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.start_frames = start_frames  # 5 x 20 ms = 100 ms
        self.end_frames = end_frames      # 20 x 20 ms = 400 ms
        self.speaking = False
        self._run = 0  # consecutive frames pointing toward a state change

    def update(self, prob: float) -> str | None:
        """Feed the speech probability for one 20 ms frame."""
        if not self.speaking:
            self._run = self._run + 1 if prob >= self.start_threshold else 0
            if self._run >= self.start_frames:
                self.speaking, self._run = True, 0
                return "start"
            return None

        self._run = self._run + 1 if prob < self.end_threshold else 0
        if self._run >= self.end_frames:
            self.speaking, self._run = False, 0
            return "end"
        return None
