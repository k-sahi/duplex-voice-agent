"""Scripted conversation with a fake TTS, including a barge-in.

Run:  python -m bench.tick_demo
"""
import asyncio
import json
import math
import struct

from engine.audio import SAMPLE_RATE, SILENCE
from engine.session import Session
from engine.state_machine import (
    CancelAgentJobs, StartResponse, TranscriptFinal,
    UserSpeechEnded, UserSpeechStarted,
)
from engine.tick import TickLoop


def tone(ms: int, freq: float = 220.0) -> bytes:
    n = SAMPLE_RATE * ms // 1000
    return struct.pack(f"<{n}h", *(int(8000 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
                                   for i in range(n)))


async def main() -> None:
    counts = {"speech_frames": 0, "silence_frames": 0}
    agent_task: asyncio.Task | None = None

    def send_audio(frame: bytes) -> None:
        counts["silence_frames" if frame == SILENCE else "speech_frames"] += 1

    async def fake_tts(text: str) -> None:
        pushed = 0
        try:
            await asyncio.sleep(0.3)              # pretend LLM + TTS first-chunk latency
            for _ in range(20):                   # 20 x 100 ms = 2 s of "speech"
                session.output.push(tone(100))
                pushed += 1
                await asyncio.sleep(0.05)         # TTS runs faster than real time
            session.output.mark_done()
        except asyncio.CancelledError:
            print(f"  [fake_tts] cancelled after {pushed} chunks for {text!r}")
            raise

    def on_action(action) -> None:
        nonlocal agent_task
        if isinstance(action, StartResponse):
            agent_task = asyncio.create_task(fake_tts(action.text))
        elif isinstance(action, CancelAgentJobs) and agent_task:
            agent_task.cancel()

    session = Session(send_audio, on_action)
    loop = TickLoop(session.tick, period_ms=20)

    async def user_script() -> None:
        await asyncio.sleep(0.2); session.post(UserSpeechStarted(200))
        await asyncio.sleep(0.8); session.post(UserSpeechEnded(1000))
        await asyncio.sleep(0.1); session.post(TranscriptFinal(1100, "tell me a story"))
        await asyncio.sleep(1.2); session.post(UserSpeechStarted(2300))   # barge-in
        await asyncio.sleep(0.6); session.post(UserSpeechEnded(2900))
        await asyncio.sleep(0.1); session.post(TranscriptFinal(3000, "actually, never mind"))

    await asyncio.gather(loop.run(duration_s=6.0), user_script())
    if agent_task and not agent_task.done():
        agent_task.cancel()

    print("\nTimeline (engine clock):")
    for t_ms, msg in session.log:
        print(f"  {t_ms:5d} ms  {msg}")
    print("\nFrames sent:", counts)
    print("Tick stats:", json.dumps(loop.stats.summary(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
