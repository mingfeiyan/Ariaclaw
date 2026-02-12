# AriaClaw Design

A Python desktop app that streams video from Aria Gen 2 glasses and audio from the Aria mic array to Google Gemini Live API, with a local web dashboard and OpenClaw integration for agentic actions.

## Architecture

```
Aria Gen 2 Glasses (USB/WiFi)
    ├── RGB Camera ──→ NumPy frames ──→ JPEG encode ──→ Gemini WebSocket
    └── 8-Mic Array ──→ PCM audio ──→ resample to 16kHz mono ──→ Gemini WebSocket
                                                                        ↕
Mac Speakers ←── PCM 24kHz ←── Gemini Live API ←── tool calls ──→ OpenClaw HTTP
                                                                   (localhost:18789)

CLI starts the app → FastAPI server on localhost:8080
    Browser dashboard:
    ├── Live video preview (MJPEG stream from Aria RGB)
    ├── Connection status (Aria / Gemini / OpenClaw)
    ├── Scrolling transcription (input + output, via WebSocket)
    └── Start/Stop session button
```

## Components

### 1. Aria Streaming Module (`aria_stream.py`)

Connects to Aria Gen 2 glasses and provides video frames + audio data via callbacks.

**Video pipeline:**
- Register `rgb_callback` via `StreamReceiver`
- Receives `ImageData` → `to_numpy_array()` → raw frame
- Throttle to ~1 fps
- JPEG encode at 50% quality → base64 for Gemini
- Also push raw frames to the web dashboard's MJPEG endpoint

**Audio pipeline:**
- Register `audio_callback` via `StreamReceiver`
- 8-mic array → mix down to mono
- Resample to 16kHz using `scipy.signal.resample`
- Convert to Int16 PCM
- Accumulate into ~100ms chunks (3200 bytes)
- Forward to Gemini WebSocket

**Connection management:**
- `DeviceClient` connects via USB or WiFi
- Use `mp_streaming_demo` profile initially
- States: `disconnected → connecting → streaming → error`
- Graceful shutdown: `device.stop_streaming()` + `stream_receiver.stop_server()`

**Config:**
```python
STREAMING_INTERFACE = "USB_NCM"  # or WIFI
STREAMING_PROFILE = "mp_streaming_demo"
STREAMING_PORT = 6768
VIDEO_FPS_TARGET = 1.0
JPEG_QUALITY = 50
AUDIO_TARGET_RATE = 16000
AUDIO_CHUNK_MS = 100
```

### 2. Gemini Live Service (`gemini_service.py`)

WebSocket client to Gemini's bidirectional streaming API.

**Connection:**
- URL: `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=<API_KEY>`
- Model: `gemini-2.5-flash-native-audio-preview-12-2025`
- Uses Python `websockets` library with `asyncio`

**Setup message:**
- `responseModalities: ["AUDIO"]`
- `thinkingBudget: 0`
- System instruction adapted for Aria glasses
- `execute` tool declaration for OpenClaw
- Input/output audio transcription enabled
- Activity detection: high start sensitivity, low end sensitivity, 500ms silence

**Sending:**
- `send_video_frame(jpeg_base64)` → `realtimeInput.video`
- `send_audio(pcm_base64)` → `realtimeInput.audio` (mime: `audio/pcm;rate=16000`)
- `send_tool_response(call_id, result)` → `toolResponse.functionResponses`

**Receiving (async loop):**
- `setupComplete` → mark ready
- `serverContent.modelTurn.parts[].inlineData` → PCM audio → playback
- `serverContent.inputTranscription` → user text → dashboard
- `serverContent.outputTranscription` → model text → dashboard
- `toolCall` → OpenClaw bridge
- `goAway` → reconnect

**States:** `disconnected → connecting → setting_up → ready → error`

### 3. Audio Playback (`audio_playback.py`)

Plays Gemini's audio responses through Mac speakers.

- Receives PCM Int16 at 24kHz from Gemini
- Uses `pyaudio` output stream on default Mac audio device
- Streams chunks as they arrive (low latency)
- Tracks `is_speaking` state

No echo cancellation needed — mic on glasses, speaker on Mac.

### 4. OpenClaw Bridge (`openclaw_bridge.py`)

HTTP client to local OpenClaw gateway.

- Receives tool calls from Gemini (name: `execute`, param: `task`)
- POST to `http://localhost:18789/v1/chat/completions`
- Headers: `Authorization: Bearer <token>`, `x-openclaw-session-key: <key>`
- Body: `{"model": "openclaw", "messages": [...], "stream": false}`
- Maintains conversation history (up to 20 messages)
- Returns result text back to Gemini via `send_tool_response()`
- Session key format: `agent:main:aria:<ISO8601-timestamp>`
- States: `not_configured → checking → connected → unreachable`

### 5. Web Dashboard (`dashboard.py` + `static/index.html`)

FastAPI server on `localhost:8080`.

**Endpoints:**

| Route | Type | Purpose |
|---|---|---|
| `GET /` | HTML | Dashboard page |
| `GET /video` | MJPEG | Live Aria RGB feed (~5 fps) |
| `WS /ws` | WebSocket | Real-time status + transcription |

**WebSocket messages:**
```json
{"type": "status", "aria": "streaming", "gemini": "ready", "openclaw": "connected"}
{"type": "transcript", "role": "user", "text": "What am I looking at?"}
{"type": "transcript", "role": "model", "text": "I can see a coffee mug on your desk."}
{"type": "tool_call", "task": "search for coffee mug prices"}
{"type": "tool_result", "result": "Found 3 results..."}
```

**Dashboard layout:**
```
┌──────────────────────────────────────────┐
│  AriaClaw            ● Aria  ● Gemini  ● OC │
├────────────────────┬─────────────────────┤
│                    │ Transcription        │
│  Live Video Feed   │                     │
│  (MJPEG stream)    │ You: What's this?   │
│                    │ AI: That's a coffee  │
│                    │     mug on your...   │
├────────────────────┴─────────────────────┤
│  [ Start Session ]  [ Stop Session ]     │
└──────────────────────────────────────────┘
```

### 6. App Coordinator (`ariaclaw.py` + `config.py`)

**`config.py`:**
```python
# Aria
ARIA_STREAMING_INTERFACE = "USB_NCM"
ARIA_STREAMING_PROFILE = "mp_streaming_demo"
ARIA_STREAMING_PORT = 6768

# Gemini
GEMINI_API_KEY = "..."  # or env var
GEMINI_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
VIDEO_FRAME_INTERVAL = 1.0
JPEG_QUALITY = 50
AUDIO_CHUNK_MS = 100

# OpenClaw
OPENCLAW_HOST = "http://localhost"
OPENCLAW_PORT = 18789
OPENCLAW_TOKEN = "..."  # or env var

# Dashboard
DASHBOARD_PORT = 8080
```

**Startup sequence:**
1. Load config (env var overrides for secrets)
2. Start FastAPI dashboard (uvicorn, background thread)
3. Connect to Aria glasses via DeviceClient
4. Check OpenClaw reachability
5. Wait for "Start Session" from dashboard
6. Open Gemini WebSocket, send setup
7. Start Aria streaming → callbacks flow to Gemini
8. Receive loop: Gemini → audio playback + transcription
9. Tool calls → OpenClaw → responses back to Gemini
10. "Stop Session" → tear down in reverse

## Project Structure

```
ariaclaw/
├── ariaclaw.py              # Entry point & coordinator
├── config.py                # All configuration
├── aria_stream.py           # Aria Gen 2 connection & streaming
├── gemini_service.py        # Gemini Live WebSocket client
├── audio_playback.py        # Mac speaker output via pyaudio
├── openclaw_bridge.py       # OpenClaw HTTP bridge
├── dashboard.py             # FastAPI server
├── static/
│   └── index.html           # Dashboard UI
├── requirements.txt         # Dependencies
└── README.md                # Setup & usage
```

## Dependencies

```
projectaria-client-sdk
fastapi
uvicorn
websockets
numpy
scipy
pyaudio
```

## Key Decisions

- **Aria mics for input, Mac speakers for output** — hands-free capture with Gemini's natural voice
- **Web dashboard over native GUI** — no PyQt dependency, viewable from any browser
- **1:1 port of Gemini/OpenClaw protocols** — proven logic from VisionClaw
- **~800-1000 lines estimated** — significantly leaner than VisionClaw's 3,430 lines
