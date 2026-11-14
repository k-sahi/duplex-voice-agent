"""End-to-end test of the server path with a fake WebSocket and fake ASR (no browser, no models)."""
import asyncio
import json
import time

from engine.audio import SILENCE
from server.connection import Connection
from tests.test_vad_energy import LOUD


class FakeWS:
    def __init__(self, frames, gap_s=0.02):
        self.frames, self.gap = frames, gap_s
        self.audio_out, self.events = [], []

    async def receive(self):
        if not self.frames:
            await asyncio.sleep(0.8)  # let the agent finish before disconnecting
            return {"type": "websocket.disconnect"}
        await asyncio.sleep(self.gap)  # real time: one frame per 20 ms
        return {"type": "websocket.receive", "bytes": self.frames.pop(0)}

    async def send_bytes(self, b):
        self.audio_out.append(b)

    async def send_text(self, t):
        self.events.append(json.loads(t))


class SlowASR:
    """Pretends to be Whisper: blocks its thread for 150 ms, returns fixed text."""
    name = "fake"

    def transcribe(self, pcm):
        time.sleep(0.15)
        return "hello there"


def states(ws):
    return [e["state"] for e in ws.events if e["type"] == "state"]


def of_type(ws, kind):
    return [e for e in ws.events if e["type"] == kind]


def test_echo_turn_plays_audio_back():
    ws = FakeWS([SILENCE] * 10 + [LOUD] * 50 + [SILENCE] * 50)
    asyncio.run(Connection(ws).run())
    assert states(ws)[:3] == ["LISTENING", "THINKING", "SPEAKING"]
    assert len(ws.audio_out) > 30
    assert any("ttfo_ms" in e for e in ws.events)


def test_partials_while_talking_and_final_text_from_asr():
    ws = FakeWS([LOUD] * 75 + [SILENCE] * 50)       # 1.5 s of speech
    conn = Connection(ws, asr=SlowASR())
    asyncio.run(conn.run())
    assert len(of_type(ws, "partial")) >= 1
    final = of_type(ws, "transcript")[0]
    assert final["text"] == "hello there" and final["asr_ms"] >= 150
    assert conn.ticks.stats.overruns == 0             # slow ASR ran off the tick thread


def test_barge_in_flushes_browser_and_cancels_agent():
    ws = FakeWS([LOUD] * 50 + [SILENCE] * 50 + [LOUD] * 30 + [SILENCE] * 40)
    asyncio.run(Connection(ws).run())
    kinds = [e["type"] for e in ws.events]
    assert "flush" in kinds and "cancel" in kinds
    s = states(ws)
    assert s[s.index("SPEAKING") + 1] == "LISTENING"   # interrupted mid-speech


class RecordingASR(SlowASR):
    def __init__(self):
        self.lengths = []

    def transcribe(self, pcm):
        self.lengths.append(len(pcm))
        time.sleep(0.4)  # slow, so the user resumes before the first final returns
        return "one long sentence"


def test_pause_then_continue_is_one_turn_with_all_the_words():
    # 0.8 s speech, 0.44 s pause (turn ends), 0.8 s more speech before ASR returns
    ws = FakeWS([LOUD] * 40 + [SILENCE] * 22 + [LOUD] * 40 + [SILENCE] * 60)
    asr = RecordingASR()
    asyncio.run(Connection(ws, asr=asr).run())
    finals = of_type(ws, "transcript")
    assert len(finals) == 1                       # the stale first transcript was dropped
    assert states(ws).count("THINKING") == 1
    assert finals[0]["audio_ms"] >= 1400          # covers both halves, not just the second
