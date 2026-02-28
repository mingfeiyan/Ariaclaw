# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

AriaClaw streams video and audio from Meta Aria Gen 2 smart glasses to the Google Gemini Live API for real-time voice+vision AI conversations, with OpenClaw integration for agentic actions (messaging, web search, smart home, etc.). A FastAPI dashboard at localhost:8080 provides live video, transcription, and session controls.

## Commands

```bash
# Activate the virtualenv (Python 3.11 required by Aria SDK)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m ariaclaw

# Run all tests (40 tests, ~1.5s)
pytest tests/

# Run a single test file
pytest tests/test_memory_writer.py

# Run a specific test
pytest tests/test_memory_writer.py::TestMemoryWriter::test_write_speech_self_event

# Filter tests by name pattern
pytest tests/ -k "scene"
```

No linter or formatter is configured. No Makefile or pyproject.toml.

## Architecture

### Coordinator Pattern

`AriaClaw` (ariaclaw.py) is the central coordinator. Components have **no direct references** to each other — all wiring happens via typed callback slots (`on_video_frame`, `on_audio_chunk`, `on_state_changed`, etc.) that AriaClaw connects in `_start_session()`.

### Components

| Component | File | Role |
|---|---|---|
| AriaClaw | `ariaclaw.py` | Coordinator, lifecycle, callback wiring |
| AriaStream | `aria_stream.py` | Device connection, video/audio/PPG/GPS streaming, VAD |
| GeminiService | `gemini_service.py` | Bidirectional WebSocket to Gemini Live API |
| AudioPlayback | `audio_playback.py` | Speaker output via PyAudio (dedicated thread + queue) |
| OpenClawBridge | `openclaw_bridge.py` | HTTP client to OpenClaw gateway (aiohttp) |
| Dashboard | `dashboard.py` | FastAPI server: REST, WebSocket, MJPEG stream |

### Context Pipeline (always-on, independent of Gemini session)

```
AriaStream sensors → ContextPipeline → sub-processors → ContextEvent → MemoryWriter → daily .md log
```

Sub-processors in `context/`: `audio_processor.py` (Whisper transcription), `scene_processor.py` (pHash keyframes), `spatial_processor.py` (GPS/orientation/HR events), `ppg_processor.py` (heart rate from raw PPG), `ocr_processor.py` (Apple Vision OCR, macOS only). `daily_summarizer.py` compresses daily logs into long-term MEMORY.md.

Output goes to `~/.openclaw/workspace/memory/YYYY-MM-DD.md` (daily logs) and `~/.openclaw/workspace/context/YYYY-MM-DD/` (keyframe images).

### Threading Model

- **Aria SDK callbacks** fire on SDK threads → use `asyncio.run_coroutine_threadsafe()` to schedule on the main asyncio loop
- **Dashboard/uvicorn** runs its own event loop in a daemon thread; cross-loop broadcast uses `call_soon_threadsafe()`
- **CPU-heavy work** (Whisper, pHash, disk I/O) runs in dedicated daemon threads with `queue.Queue`

### Key Patterns

- **State machines with enum + callbacks**: `AriaConnectionState`, `GeminiConnectionState`, `OpenClawConnectionState` — each component tracks state and notifies via `on_state_changed`
- **VAD suppression**: When Gemini audio plays back, `_vad_suppressed = True` prevents the AI's voice (picked up by Aria mics) from triggering self-interruption
- **Auto-reconnect**: Gemini reconnects with exponential backoff (2s → 60s, resets on success)
- **Single Gemini tool**: All agentic actions go through one function declaration `execute({task: string})` → OpenClawBridge → OpenClaw's OpenAI-compatible API with session-scoped conversation history (max 10 turns)

## Configuration

`config.py` is gitignored (copy from `config.example.py`). All values use `os.getenv(KEY, default)` — environment variables always win. `GEMINI_API_KEY` is the only required key.

## Tests

Tests cover only the `context/` package (the rest requires hardware/network). Tests use class-based pytest style with `setup_method`/`teardown_method` and `tempfile.mkdtemp()` for filesystem isolation. Aria SDK and Whisper gracefully degrade when unavailable (try/except import pattern with `*_AVAILABLE` flags).
