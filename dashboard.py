import asyncio
import json
import logging
import threading
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
_state_lock = threading.Lock()
_ws_clients: list[WebSocket] = []
_broadcast_queue: asyncio.Queue = None  # created on startup, lives on uvicorn's loop
_get_latest_jpeg = None  # callable -> bytes | None
_main_loop = None  # asyncio event loop from the main thread
_on_start_session = None  # async callable (runs on main loop)
_on_stop_session = None  # async callable (runs on main loop)
_uvicorn_loop = None  # uvicorn's event loop, set on startup

STATIC_DIR = Path(__file__).parent / "static"


def configure(
    main_loop,
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


@app.on_event("startup")
async def _startup():
    """Initialize broadcast queue and drainer on uvicorn's event loop."""
    global _broadcast_queue, _uvicorn_loop
    _uvicorn_loop = asyncio.get_running_loop()
    _broadcast_queue = asyncio.Queue(maxsize=256)
    asyncio.create_task(_broadcast_drainer())


async def _broadcast_drainer():
    """Single task on uvicorn's loop that drains the queue and sends to all clients."""
    while True:
        msg = await _broadcast_queue.get()
        if not _ws_clients:
            continue
        text = json.dumps(msg)
        disconnected = []
        for ws in list(_ws_clients):
            try:
                await ws.send_text(text)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            try:
                _ws_clients.remove(ws)
            except ValueError:
                pass


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/video")
async def video_feed():
    """MJPEG stream from Aria RGB camera."""

    async def generate():
        try:
            while True:
                jpeg = _get_latest_jpeg() if _get_latest_jpeg else None
                if jpeg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                await asyncio.sleep(0.2)  # ~5 fps
        except asyncio.CancelledError:
            return

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
    with _state_lock:
        state_snapshot = dict(_state)
    await ws.send_json({"type": "status", **state_snapshot})

    try:
        while True:
            data = await ws.receive_json()
            cmd = data.get("command")
            if cmd == "start" and _on_start_session and _main_loop:
                asyncio.run_coroutine_threadsafe(_on_start_session(), _main_loop)
            elif cmd == "stop" and _on_stop_session and _main_loop:
                asyncio.run_coroutine_threadsafe(_on_stop_session(), _main_loop)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            _ws_clients.remove(ws)
        except ValueError:
            pass
        logger.info("Dashboard client disconnected (%d total)", len(_ws_clients))


def _enqueue_broadcast(message: dict):
    """Thread-safe: enqueue a message for broadcast on uvicorn's event loop."""
    if _uvicorn_loop is None or _broadcast_queue is None:
        return
    try:
        _uvicorn_loop.call_soon_threadsafe(_broadcast_queue.put_nowait, message)
    except (RuntimeError, asyncio.QueueFull):
        pass  # loop closed or queue full — drop message


def update_status(key: str, value: str):
    """Update a connection status and broadcast to dashboard."""
    with _state_lock:
        _state[key] = value
        snapshot = dict(_state)
    _enqueue_broadcast({"type": "status", **snapshot})


def send_transcript(role: str, text: str):
    """Send a transcription line to the dashboard."""
    _enqueue_broadcast({"type": "transcript", "role": role, "text": text})


def send_tool_event(event_type: str, data: str):
    """Send a tool call/result event to the dashboard."""
    _enqueue_broadcast({"type": event_type, "data": data})


def send_context_event(event_type: str, content: dict, timestamp: str):
    """Send a context event to the dashboard."""
    _enqueue_broadcast({
        "type": "context_event",
        "event_type": event_type,
        "timestamp": timestamp,
        "content": content,
    })


def send_heart_rate(bpm: int):
    """Send current heart rate to the dashboard."""
    _enqueue_broadcast({"type": "heart_rate", "bpm": bpm})
