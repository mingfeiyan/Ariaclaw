import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="AriaClaw Dashboard")

# Shared state set by the coordinator
_state = {
    "aria": "disconnected",
    "gemini": "disconnected",
    "openclaw": "not_configured",
}
_ws_clients: list[WebSocket] = []
_get_latest_jpeg = None  # callable -> bytes | None
_main_loop = None  # asyncio event loop from the main thread
_on_start_session = None  # async callable (runs on main loop)
_on_stop_session = None  # async callable (runs on main loop)

STATIC_DIR = Path(__file__).parent / "static"


def configure(
    main_loop,  # I6: accept main event loop reference
    get_latest_jpeg=None,
    on_start_session=None,
    on_stop_session=None,
):
    """Set callbacks from the coordinator."""
    global _get_latest_jpeg, _on_start_session, _on_stop_session, _main_loop
    _main_loop = main_loop
    _get_latest_jpeg = get_latest_jpeg
    _on_start_session = on_start_session
    _on_stop_session = on_stop_session


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/video")
async def video_feed():
    """MJPEG stream from Aria RGB camera."""

    async def generate():
        while True:
            jpeg = _get_latest_jpeg() if _get_latest_jpeg else None
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.2)  # ~5 fps

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    logger.info("Dashboard client connected (%d total)", len(_ws_clients))

    # Send current state
    await ws.send_json({"type": "status", **_state})

    try:
        while True:
            data = await ws.receive_json()
            cmd = data.get("command")
            # I6: schedule on the main event loop, not uvicorn's loop
            if cmd == "start" and _on_start_session and _main_loop:
                asyncio.run_coroutine_threadsafe(_on_start_session(), _main_loop)
            elif cmd == "stop" and _on_stop_session and _main_loop:
                asyncio.run_coroutine_threadsafe(_on_stop_session(), _main_loop)
    except WebSocketDisconnect:
        pass
    finally:
        # C2: safe remove — client may already have been removed by broadcast()
        try:
            _ws_clients.remove(ws)
        except ValueError:
            pass
        logger.info("Dashboard client disconnected (%d total)", len(_ws_clients))


async def broadcast(message: dict):
    """Send a message to all connected dashboard clients."""
    if not _ws_clients:
        return
    text = json.dumps(message)
    disconnected = []
    for ws in list(_ws_clients):  # C3: iterate over a copy
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        try:
            _ws_clients.remove(ws)
        except ValueError:
            pass


async def update_status(key: str, value: str):
    """Update a connection status and broadcast to dashboard."""
    _state[key] = value
    await broadcast({"type": "status", **_state})


async def send_transcript(role: str, text: str):
    """Send a transcription line to the dashboard."""
    await broadcast({"type": "transcript", "role": role, "text": text})


async def send_tool_event(event_type: str, data: str):
    """Send a tool call/result event to the dashboard."""
    await broadcast({"type": event_type, "data": data})
