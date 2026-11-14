"""FastAPI app: serves the browser page and the /ws audio WebSocket.

Run:  uvicorn server.main:app
Open: http://localhost:8000

Settings (environment variables):
  DUPLEX_ASR_MODEL  tiny.en | base.en (default) | small.en
  DUPLEX_ASR        whisper (default) | none
  DUPLEX_VAD        silero (default) | energy
  DUPLEX_MAX_SESSIONS  how many people can talk at once (default 3)
"""
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from server.asr import load_asr
from server.connection import Connection
from server.vad import make_vad

if sys.platform == "win32":
    # Windows wakes sleeping code on a ~15.6 ms clock by default, which makes
    # a 20 ms tick fire up to a whole tick late. Ask for 1 ms resolution.
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)

MAX_SESSIONS = int(os.environ.get("DUPLEX_MAX_SESSIONS", "3"))
active_sessions = 0

CLIENT = Path(__file__).resolve().parent.parent / "client" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the speech model once for all connections (first run downloads it).
    app.state.asr = await asyncio.to_thread(load_asr)
    yield


app = FastAPI(title="duplex-engine", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(CLIENT)


@app.websocket("/ws")
async def ws_audio(ws: WebSocket) -> None:
    global active_sessions
    await ws.accept()
    # The public demo runs on a small shared CPU: a few people at a time keeps
    # latency honest for each of them.
    if active_sessions >= MAX_SESSIONS:
        await ws.send_text(json.dumps({"type": "busy", "reason": f"{MAX_SESSIONS} people are already talking to it."}))
        await ws.close(code=1013)  # 1013 = try again later
        return
    active_sessions += 1
    try:
        vad = await asyncio.to_thread(make_vad)  # one per connection: Silero keeps state
        await Connection(ws, vad=vad, asr=ws.app.state.asr).run()
    finally:
        active_sessions -= 1
