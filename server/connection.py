"""One browser connection: mic frames in, engine frames + JSON events out.

Step 4 has no AI yet. A stand-in "echo agent" replies by playing your own
words back, which is enough to exercise the whole real-time path:
VAD -> state machine -> agent task -> output buffer -> tick -> WebSocket,
including barge-in (talk over the echo and it stops).
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from itertools import islice

from engine.audio import FRAME_BYTES, FRAME_MS, SILENCE
from engine.session import Session
from engine.state_machine import (
    CancelAgentJobs, FinalizeASR, FlushPlayback, StartASR, StartResponse,
    State, TranscriptFinal, UserSpeechEnded, UserSpeechStarted,
)
from engine.tick import TickLoop
from engine.vad_energy import EnergyVAD

MIN_UTTERANCE_MS = 400   # shorter than this counts as noise
FAKE_THINK_S = 0.3       # stand-in for LLM + TTS time-to-first-chunk


class Connection:
    def __init__(self, ws) -> None:
        self.ws = ws  # anything with async receive(), send_bytes(), send_text()
        self.out: asyncio.Queue = asyncio.Queue()
        self.session = Session(self._send_audio, self._on_action)
        self.vad = EnergyVAD()
        self.ticks = TickLoop(self._tick)
        self.preroll: deque[bytes] = deque(maxlen=15)  # last 300 ms, so word onsets aren't cut
        self.recording: list[bytes] | None = None
        self.utterance = b""
        self.agent_task: asyncio.Task | None = None
        self.prev_state = State.IDLE
        self.speech_end_ms: int | None = None
        self.t0 = time.perf_counter()

    # ---------- helpers ----------
    def now_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

    def _event(self, kind: str, **data) -> None:
        self.out.put_nowait({"type": kind, "t_ms": self.now_ms(), **data})

    def _send_audio(self, frame: bytes) -> None:
        # The tick produces a frame every 20 ms. Silence is not sent over the
        # network; the browser plays silence on its own when nothing arrives.
        if frame is not SILENCE:
            self.out.put_nowait(frame)

    # ---------- engine callbacks ----------
    def _tick(self, n: int) -> None:
        self.session.tick(n)
        state = self.session.state
        if state != self.prev_state:
            extra = {}
            if state == State.SPEAKING and self.speech_end_ms is not None:
                extra["ttfo_ms"] = self.now_ms() - self.speech_end_ms
            self._event("state", state=state.name, **extra)
            self.prev_state = state

    def _on_action(self, action) -> None:
        match action:
            case StartASR():
                self.recording = list(self.preroll)
            case FinalizeASR():
                frames = self.recording or []
                self.recording = None
                frames = frames[:-self.vad.end_frames]  # drop the trailing 500 ms of silence
                self.utterance = b"".join(frames)
                ms = len(frames) * FRAME_MS
                text = f"[{ms} ms of speech]" if ms >= MIN_UTTERANCE_MS else ""
                self._event("transcript", text=text or "(too short, ignored)")
                self.session.post(TranscriptFinal(self.now_ms(), text))
            case StartResponse():
                self.agent_task = asyncio.create_task(self._echo_agent(self.utterance))
            case CancelAgentJobs(reason=reason):
                if self.agent_task and not self.agent_task.done():
                    self.agent_task.cancel()
                self._event("cancel", reason=reason)
            case FlushPlayback():
                # Session already emptied its own buffer; now empty the browser's.
                self.out.put_nowait({"type": "flush", "t_ms": self.now_ms()})

    # ---------- the stand-in agent ----------
    async def _echo_agent(self, audio: bytes) -> None:
        await asyncio.sleep(FAKE_THINK_S)
        chunk = FRAME_BYTES * 10  # 200 ms per chunk
        for i in range(0, len(audio), chunk):
            self.session.output.push(audio[i:i + chunk])
            await asyncio.sleep(0.1)  # faster than real time, like a real TTS
        self.session.output.mark_done()

    # ---------- network ----------
    def on_mic_frame(self, frame: bytes) -> None:
        if len(frame) != FRAME_BYTES:
            return
        self.preroll.append(frame)
        if self.recording is not None:
            self.recording.append(frame)
        result = self.vad.process(frame)
        if result == "start":
            self._event("vad", speaking=True)
            self.session.post(UserSpeechStarted(self.now_ms()))
        elif result == "end":
            self.speech_end_ms = self.now_ms()
            self._event("vad", speaking=False)
            self.session.post(UserSpeechEnded(self.speech_end_ms))

    async def _sender(self) -> None:
        while True:
            item = await self.out.get()
            if isinstance(item, bytes):
                await self.ws.send_bytes(item)
            else:
                await self.ws.send_text(json.dumps(item))

    async def _stats(self) -> None:
        while True:
            await asyncio.sleep(2)
            late = list(islice(reversed(self.ticks.stats.lateness_ms), 250))
            late.sort()
            p99 = late[int(0.99 * (len(late) - 1))] if late else 0.0
            self._event("stats", tick_late_p99_ms=round(p99, 2),
                        overruns=self.ticks.stats.overruns, skipped=self.ticks.stats.skipped)

    async def run(self) -> None:
        tasks = [asyncio.create_task(t) for t in (self._sender(), self.ticks.run(), self._stats())]
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    self.on_mic_frame(msg["bytes"])
        finally:
            self.ticks.stop()
            if self.agent_task:
                self.agent_task.cancel()
            for t in tasks:
                t.cancel()
            print("session ended:", self.ticks.stats.summary())
