"""Audio framing: everything in the engine moves in fixed 20 ms frames."""
from __future__ import annotations

from collections import deque

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples
FRAME_BYTES = FRAME_SAMPLES * 2                 # int16 mono -> 640 bytes
SILENCE = bytes(FRAME_BYTES)


class OutputBuffer:
    """Agent audio waiting to go out, released one frame per tick.

    TTS pushes chunks of any size; the tick loop pulls exactly one frame.
    flush() is what makes barge-in instant: queued speech is simply dropped.
    """

    def __init__(self) -> None:
        self._frames: deque[bytes] = deque()
        self._partial = b""
        self.done = False  # True once the current utterance has been fully pushed

    def push(self, pcm: bytes) -> None:
        data = self._partial + pcm
        n = len(data) // FRAME_BYTES
        for i in range(n):
            self._frames.append(data[i * FRAME_BYTES:(i + 1) * FRAME_BYTES])
        self._partial = data[n * FRAME_BYTES:]

    def mark_done(self) -> None:
        """End of utterance: pad the leftover partial frame with silence."""
        if self._partial:
            self._frames.append(self._partial.ljust(FRAME_BYTES, b"\x00"))
            self._partial = b""
        self.done = True

    def next_frame(self) -> tuple[bytes, bool]:
        """Return (frame, is_agent_audio). Never blocks; silence if empty."""
        if self._frames:
            return self._frames.popleft(), True
        return SILENCE, False

    def flush(self) -> int:
        dropped = len(self._frames)
        self._frames.clear()
        self._partial = b""
        self.done = False
        return dropped

    def __len__(self) -> int:
        return len(self._frames)
