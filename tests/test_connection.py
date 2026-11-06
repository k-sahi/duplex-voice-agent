"""End-to-end test of the server path with a fake WebSocket (no browser needed)."""
import asyncio
import json

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


def states(ws):
    return [e["state"] for e in ws.events if e["type"] == "state"]


def test_echo_turn_plays_audio_back():
    # 1 s of speech, then 1 s of silence
    ws = FakeWS([SILENCE] * 10 + [LOUD] * 50 + [SILENCE] * 50)
    asyncio.run(Connection(ws).run())
    assert states(ws)[:3] == ["LISTENING", "THINKING", "SPEAKING"]
    assert len(ws.audio_out) > 30                  # the echo came back
    assert any("ttfo_ms" in e for e in ws.events)


def test_barge_in_flushes_browser_and_cancels_agent():
    # speak 1 s, pause 1 s (turn ends, echo starts playing), then talk over it
    ws = FakeWS([LOUD] * 50 + [SILENCE] * 50 + [LOUD] * 30 + [SILENCE] * 40)
    asyncio.run(Connection(ws).run())
    kinds = [e["type"] for e in ws.events]
    assert "flush" in kinds and "cancel" in kinds
    s = states(ws)
    i = s.index("SPEAKING")
    assert s[i + 1] == "LISTENING"   # interrupted mid-speech, not finished
