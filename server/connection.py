"""One browser connection: mic frames in, engine frames + JSON events out.

Step 5: real VAD (Silero) and real speech recognition (faster-whisper).
- While you talk, a partial transcript is refreshed every 0.5 s.
- When you stop, a final transcript is made and handed to the state machine.
The agent is still a stand-in that plays your own words back; Step 6 adds an
LLM and text-to-speech.
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
from server.asr import LengthASR

MIN_UTTERANCE_MS = 400   # shorter than this counts as noise
PARTIAL_EVERY_S = 0.5    # how often to refresh the live transcript
FAKE_THINK_S = 0.3       # stand-in for LLM + TTS time-to-first-chunk


class Connection:
    def __init__(self, ws, vad=None, asr=None) -> None:
        self.ws = ws  # anything with async receive(), send_bytes(), send_text()
        self.vad = vad or EnergyVAD()
        self.asr = asr or LengthASR()
        self.out: asyncio.Queue = asyncio.Queue()
        self.session = Session(self._send_audio, self._on_action)
        self.ticks = TickLoop(self._tick)
        self.preroll: deque[bytes] = deque(maxlen=15)  # last 300 ms, so word onsets aren't cut
        self.recording: list[bytes] | None = None
        self.turn = 0                     # increases every user turn; stale results are dropped
        self.turn_frames: list[bytes] = []
        self.utterance = b""
        self.partial_task: asyncio.Task | None = None
        self.final_task: asyncio.Task | None = None
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
        # Silence is not sent over the network; the browser plays silence by itself.
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
                resuming = self.final_task is not None and not self.final_task.done()
                self.turn += 1  # any result still running for the old turn id is now stale
                # Resumed talking before the transcript came back: keep the earlier words
                # so the final transcript covers the whole sentence, not just its second half.
                earlier = self.turn_frames if resuming else []
                self.recording = earlier + list(self.preroll)
                self.partial_task = asyncio.create_task(self._partials(self.turn))
            case FinalizeASR():
                frames = (self.recording or [])[:-self.vad.end_frames]  # drop trailing silence
                self.recording = None
                self.turn_frames = frames
                if self.partial_task:
                    self.partial_task.cancel()
                self.final_task = asyncio.create_task(self._final(self.turn, frames))
            case StartResponse():
                self.agent_task = asyncio.create_task(self._echo_agent(self.utterance))
            case CancelAgentJobs(reason=reason):
                if self.agent_task and not self.agent_task.done():
                    self.agent_task.cancel()
                self._event("cancel", reason=reason)
            case FlushPlayback():
                # Session already emptied its own buffer; now empty the browser's.
                self.out.put_nowait({"type": "flush", "t_ms": self.now_ms()})

    # ---------- speech recognition ----------
    async def _partials(self, turn: int) -> None:
        """Re-transcribe the growing turn every 0.5 s so the user sees words appear."""
        while True:
            await asyncio.sleep(PARTIAL_EVERY_S)
            if self.recording is None or turn != self.turn:
                return
            pcm = b"".join(self.recording)
            t0 = time.perf_counter()
            text = await asyncio.to_thread(self.asr.transcribe, pcm)
            if turn == self.turn and self.recording is not None:  # still the same, unfinished turn
                self._event("partial", text=text, asr_ms=round((time.perf_counter() - t0) * 1000))

    async def _final(self, turn: int, frames: list[bytes]) -> None:
        ms = len(frames) * FRAME_MS
        pcm = b"".join(frames)
        text, asr_ms = "", 0
        if ms >= MIN_UTTERANCE_MS:
            t0 = time.perf_counter()
            text = await asyncio.to_thread(self.asr.transcribe, pcm)
            asr_ms = round((time.perf_counter() - t0) * 1000)
        if turn != self.turn:  # the user started a new turn meanwhile: drop this result
            return
        self.utterance = pcm
        self._event("transcript", text=text or "(no speech, ignored)", audio_ms=ms, asr_ms=asr_ms)
        self.session.post(TranscriptFinal(self.now_ms(), text))

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
            late = sorted(islice(reversed(self.ticks.stats.lateness_ms), 250))
            p99 = late[int(0.99 * (len(late) - 1))] if late else 0.0
            self._event("stats", tick_late_p99_ms=round(p99, 2),
                        overruns=self.ticks.stats.overruns, skipped=self.ticks.stats.skipped)

    async def run(self) -> None:
        self._event("hello", vad=type(self.vad).__name__, asr=getattr(self.asr, "name", "?"))
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
            for t in (self.agent_task, self.partial_task, self.final_task, *tasks):
                if t:
                    t.cancel()
            print("session ended:", self.ticks.stats.summary())
