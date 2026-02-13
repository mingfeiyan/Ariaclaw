# AriaClaw Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AriaClaw (Python)                            │
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │              │    │                  │    │                    │  │
│  │  AriaStream  │───▶│  AriaClaw Core   │◀──▶│  GeminiLiveService │  │
│  │              │    │  (Coordinator)   │    │                    │  │
│  └──────────────┘    └──┬─────┬─────┬───┘    └────────────────────┘  │
│                         │     │     │                                │
│              ┌──────────┘     │     └────────────┐                   │
│              │                │                  │                   │
│              ▼                ▼                  ▼                   │
│  ┌──────────────────┐ ┌────────────┐ ┌──────────────────┐           │
│  │                  │ │            │ │                  │           │
│  │  OpenClawBridge  │ │ Dashboard  │ │  AudioPlayback   │           │
│  │  (HTTP Client)   │ │ (FastAPI)  │ │  (PyAudio)       │           │
│  │                  │ │            │ │                  │           │
│  └──────────────────┘ └────────────┘ └──────────────────┘           │
└──────────────────────────────────────────────────────────────────────┘

AriaStream ──▶ Core:  video frames, audio chunks, VAD signals
Core ◀──▶ Gemini:     sends audio/video/tool responses, receives
                      audio responses, transcriptions, tool calls
Core ──▶ OpenClaw:    delegates tool calls (e.g. "send a message")
Core ──▶ Dashboard:   status updates, transcriptions, tool events
Core ──▶ AudioPlay:   Gemini's voice audio → Mac speakers
```

## Data Flow

```
                          INPUTS (continuous)
                          ==================

┌───────────────┐         ┌──────────────┐
│  Aria Gen 2   │         │  AriaStream  │
│  Glasses      │         │              │         ┌───────────────────────┐
│               │         │              │         │  Gemini Live API      │
│  ┌─────────┐  │  USB    │  ┌────────┐  │  JPEG   │  (Bidirectional WS)   │
│  │RGB Cam  │──┼────────▶│  │Video   │──┼────────▶│                       │
│  │1440x1440│  │         │  │Pipeline│  │ 768px   │  ┌───────────────────┐ │
│  └─────────┘  │         │  └────────┘  │ 1 fps   │  │  Multimodal       │ │
│               │         │              │         │  │  Context           │ │
│  ┌─────────┐  │  USB    │  ┌────────┐  │ 16kHz   │  │                   │ │
│  │8-Mic    │──┼────────▶│  │Audio   │──┼────────▶│  │  Accumulates:     │ │
│  │Array    │  │         │  │Pipeline│  │ Int16   │  │  - Video frames   │ │
│  │16kHz/ch │  │         │  └────────┘  │ PCM     │  │  - Audio stream   │ │
│  └─────────┘  │         │              │         │  │  - System prompt  │ │
│               │         │  ┌────────┐  │ activity│  │  - Tool results   │ │
│               │         │  │Manual  │──┼────────▶│  │                   │ │
│               │         │  │VAD     │  │ start/  │  │  On activityEnd,  │ │
│               │         │  └────────┘  │ end     │  │  generates unified│ │
└───────────────┘         └──────────────┘         │  │  response using   │ │
                                                   │  │  ALL context      │ │
                                                   │  └─────────┬─────────┘ │
                          OUTPUTS (on response)    │            │           │
                          ======================== └────────────┼───────────┘
                                                                │
                          ┌──────────────┐                      │
                          │              │◀─────────────────────┘
                          │  AriaClaw    │
                          │  Core        │  3 output channels:
                          │              │
                          └──┬─────┬──┬──┘
                             │     │  │
              ┌──────────────┘     │  └──────────────┐
              │                    │                  │
              ▼                    ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Audio Response  │  │  Transcriptions  │  │  Tool Calls      │
│                  │  │                  │  │                  │
│  24kHz PCM       │  │  Input: what     │  │  execute({task}) │
│  → AudioPlayback │  │  user said       │  │  → OpenClawBridge│
│  → Mac Speakers  │  │                  │  │  → OpenClaw      │
│                  │  │  Output: what    │  │    Gateway       │
│  Gemini's voice  │  │  AI said         │  │                  │
│  speaking the    │  │                  │  │  Results fed back│
│  response        │  │  → Dashboard     │  │  into Gemini's   │
│                  │  │    transcript    │  │  context as       │
│                  │  │    panel         │  │  toolResponse     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

## How Gemini Uses Video

```
Video frames (1 fps) are a ONE-WAY input — no separate "vision response" comes back.
Gemini silently absorbs frames into its multimodal context.

                    Continuous (1 fps)
  Aria RGB Camera ──────────────────────▶ ┌─────────────────────────────┐
                                          │  Gemini Multimodal Context  │
                    Continuous (100ms)     │                             │
  Aria Microphone ──────────────────────▶ │  Video frames accumulate    │
                                          │  alongside audio input.     │
                    On speech              │                             │
  VAD Signals     ──────────────────────▶ │  When user speaks, Gemini   │
                                          │  reasons over BOTH what it  │
                                          │  hears AND what it sees.    │
                                          │                             │
                                          │  "What am I looking at?"    │
                                          │  → uses recent frames to    │
                                          │    describe the scene       │
                                          │                             │
                                          │  "Send this to my husband"  │
                                          │  → sees the image + hears   │
                                          │    the request → tool call  │
                                          └──────────────┬──────────────┘
                                                         │
                              ┌───────────────────────────┼────────────────┐
                              │                           │                │
                              ▼                           ▼                ▼
                    Audio Response (24kHz)     Transcription Text     Tool Calls
                    → Mac Speakers            → Dashboard            → OpenClaw
```

All responses (whether informed by vision or not) come back through the same
audio + transcription channels. There is no separate "image description" output.

## Audio Pipeline Detail

```
Aria 8-Mic Array (int64, 16kHz/channel, interleaved)
         │
         ▼
┌─────────────────────┐
│ De-interleave       │  2560 samples ÷ 8 channels = 320 samples/ch
│ Pick Channel 0      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Normalize           │  int64 → float64, divide by 2^25
│ (avoid clipping)    │  raw [-12M, +9M] → float [-0.36, +0.28]
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Clip [-1.0, 1.0]    │
│ Convert to Int16    │  float × 32767 → Int16 PCM
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Accumulate 100ms    │  320 samples × ~10 callbacks = 3200 bytes
│ chunks              │
└─────────┬───────────┘
          │
          ├──────────────────────────────────▶ Gemini (base64 encoded)
          │
          ▼
┌─────────────────────┐
│ VAD (RMS energy)    │
│                     │
│ Speech threshold:   │  RMS > 0.02 for 3 consecutive chunks
│   → activityStart   │
│                     │
│ Silence threshold:  │  RMS < 0.008 for 15 consecutive chunks
│   → activityEnd     │
│                     │
│ Suppressed while    │  Prevents AI audio from triggering
│ AI is speaking      │  self-interruption
└─────────────────────┘
```

## Gemini Session Lifecycle

```
Dashboard: [Start Session]
         │
         ▼
┌──────────────────┐     WebSocket      ┌──────────────────┐
│  GeminiService   │────────────────────▶│  Gemini API      │
│                  │                     │                  │
│  State Machine:  │     Setup Msg       │  Setup:          │
│                  │────────────────────▶│  - Model config  │
│  DISCONNECTED    │     (JSON)          │  - System prompt │
│       │          │                     │  - Tool decl.    │
│       ▼          │                     │  - Manual VAD    │
│  CONNECTING      │     setupComplete   │  - Transcription │
│       │          │◀────────────────────│                  │
│       ▼          │                     │                  │
│  SETTING_UP      │     Audio/Video     │                  │
│       │          │────────────────────▶│  Processing...   │
│       ▼          │     (streaming)     │                  │
│  READY ◀─────────┼─── auto-reconnect  │                  │
│       │          │     on disconnect   │  Response:       │
│       ▼          │                     │  - Audio (24kHz) │
│  DISCONNECTED    │◀────────────────────│  - Transcription │
│                  │     Audio + Text    │  - Tool calls    │
└──────────────────┘                     └──────────────────┘
```

## Tool Call Flow

```
User: "Send a message to my husband"
         │
         ▼
┌──────────────────┐     "Sure, sending    ┌──────────────────┐
│  Gemini          │      that now."       │  Mac Speakers    │
│                  │──────────────────────▶│  (voice response)│
│  Generates tool  │                       └──────────────────┘
│  call:           │
│  execute({task:  │
│   "send msg..."})│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    POST /v1/chat/     ┌──────────────────┐
│  OpenClawBridge  │    completions        │  OpenClaw        │
│                  │──────────────────────▶│  Gateway         │
│  Maintains       │    + session key      │                  │
│  conversation    │    + auth token       │  Routes to       │
│  history         │                       │  plugin:         │
│                  │◀──────────────────────│  - Telegram      │
│                  │    Result text        │  - iMessage      │
└────────┬─────────┘                       │  - Web search    │
         │                                 │  - etc.          │
         ▼                                 └──────────────────┘
┌──────────────────┐
│  Gemini          │
│                  │     "Message sent!"
│  toolResponse    │──────────────────────▶ Mac Speakers
│  → continues     │
│  conversation    │──────────────────────▶ Dashboard
└──────────────────┘                        (transcript)
```
