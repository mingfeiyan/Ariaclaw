# AriaClaw

A real-time AI assistant that streams video and audio from [Meta Aria Gen 2](https://www.projectaria.com/) smart glasses to the [Google Gemini Live API](https://ai.google.dev/gemini-api/docs/live), with [OpenClaw](https://openclaw.com/) integration for agentic actions.

Speak naturally, see through your glasses, and let AI take actions on your behalf — send messages, search the web, control smart home devices, and more.

## Architecture

```
Aria Gen 2 Glasses (USB / WiFi)
    ├── RGB Camera ──→ JPEG (768px, 1 fps) ──→ Gemini WebSocket
    └── 8-Mic Array ──→ mono 16kHz Int16 PCM ──→ Gemini WebSocket
                                                        ↕
Mac Speakers ←── PCM 24kHz ←── Gemini Live API ──→ tool calls ──→ OpenClaw
                                                                   (localhost:18789)

Dashboard: http://localhost:8080
    ├── Live video preview (MJPEG from Aria RGB camera)
    ├── Connection status indicators (Aria / Gemini / OpenClaw)
    ├── Scrolling transcription (your speech + AI responses)
    └── Start / Stop session controls
```

## Features

- **Real-time voice conversation** through Aria's mic array with Gemini's native audio model
- **Live video understanding** — Gemini sees what you see through the Aria RGB camera
- **Manual voice activity detection (VAD)** with automatic suppression during AI speech to prevent self-interruption
- **Agentic actions via OpenClaw** — send messages (iMessage, Telegram, WhatsApp), search the web, manage tasks, control smart home, and more
- **Web dashboard** with live video feed, transcription, and status monitoring
- **Audio playback** of Gemini's voice responses through Mac speakers

## Prerequisites

- **Meta Aria Gen 2 glasses** connected via USB-NCM or WiFi
- **Python 3.11** with [Project Aria Client SDK](https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/)
- **Google Gemini API key** with access to the Live API
- **macOS** (for PyAudio speaker output)
- **OpenClaw** (optional) — local gateway for agentic actions

## Setup

1. **Create a Python environment** (the Aria SDK requires Python 3.11):

   ```bash
   python3.11 -m venv ariaclaw_env
   source ariaclaw_env/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure credentials** — copy the example config and fill in your keys:

   ```bash
   cp config.example.py config.py
   ```

   Edit `config.py` and set:
   - `GEMINI_API_KEY` — your Google Gemini API key
   - `OPENCLAW_TOKEN` — your OpenClaw gateway token (if using OpenClaw)

   Alternatively, use environment variables:

   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   export OPENCLAW_TOKEN="your-openclaw-token"
   ```

4. **Connect Aria glasses** via USB and ensure they are discoverable by the Aria SDK.

## Usage

```bash
python -m ariaclaw
```

Then open **http://localhost:8080** in your browser and click **Start Session**.

### Environment Variables

All settings in `config.py` can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `GEMINI_MODEL` | `models/gemini-2.5-flash-native-audio-preview-12-2025` | Gemini model ID |
| `OPENCLAW_TOKEN` | (optional) | OpenClaw gateway token |
| `OPENCLAW_HOST` | `http://localhost` | OpenClaw host |
| `OPENCLAW_PORT` | `18789` | OpenClaw port |
| `ARIA_STREAMING_INTERFACE` | `USB_NCM` | `USB_NCM` or `WIFI` |
| `VIDEO_FRAME_INTERVAL` | `1.0` | Seconds between video frames sent to Gemini |
| `JPEG_QUALITY` | `50` | JPEG quality for Gemini frames (1-100) |
| `DASHBOARD_PORT` | `8080` | Dashboard web server port |

### SSL Certificate Issues

If you see `SSL: CERTIFICATE_VERIFY_FAILED` errors, set the SSL cert path:

```bash
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

## Project Structure

```
ariaclaw/
├── ariaclaw.py           # Entry point and coordinator
├── config.py             # Configuration (gitignored)
├── config.example.py     # Example config with placeholders
├── aria_stream.py        # Aria Gen 2 connection, video/audio streaming, VAD
├── gemini_service.py     # Gemini Live API WebSocket client
├── audio_playback.py     # Mac speaker output via PyAudio
├── openclaw_bridge.py    # OpenClaw HTTP bridge for agentic actions
├── dashboard.py          # FastAPI web dashboard server
├── static/
│   └── index.html        # Dashboard UI
└── requirements.txt      # Python dependencies
```

## How It Works

1. **Aria streams** 8-channel audio (16kHz per channel) and RGB video over USB
2. **Audio pipeline** picks a single mic channel, converts to Int16 PCM, and accumulates into 100ms chunks
3. **Video pipeline** encodes frames as JPEG (resized to 768px max), sent to Gemini at ~1 fps
4. **Manual VAD** detects speech start/end using RMS energy thresholds, sending `activityStart`/`activityEnd` signals to Gemini (automatic activity detection is disabled as it doesn't work reliably with the Aria audio pipeline)
5. **VAD suppression** prevents the AI's own audio output (picked up by Aria mics) from triggering a self-interruption
6. **Gemini responds** with streaming audio (24kHz PCM) played through Mac speakers, plus text transcriptions shown on the dashboard
7. **Tool calls** are routed to OpenClaw, which executes actions and returns results to Gemini

## License

MIT
