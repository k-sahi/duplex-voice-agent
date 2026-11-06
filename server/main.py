"""FastAPI app: serves the browser page and the /ws audio WebSocket.

Run:  uvicorn server.main:app --reload
Open: http://localhost:8000
"""
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from server.connection import Connection

CLIENT = Path(__file__).resolve().parent.parent / "client" / "index.html"

app = FastAPI(title="duplex-engine")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(CLIENT)


@app.websocket("/ws")
async def ws_audio(ws: WebSocket) -> None:
    await ws.accept()
    await Connection(ws).run()
