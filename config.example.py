import os

# Aria Gen 2
ARIA_STREAMING_INTERFACE = os.getenv("ARIA_STREAMING_INTERFACE", "USB_NCM")
ARIA_STREAMING_PROFILE = os.getenv("ARIA_STREAMING_PROFILE", "mp_streaming_demo")
ARIA_STREAMING_PORT = int(os.getenv("ARIA_STREAMING_PORT", "6768"))

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL", "models/gemini-2.5-flash-native-audio-preview-12-2025"
)
GEMINI_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
VIDEO_FRAME_INTERVAL = float(os.getenv("VIDEO_FRAME_INTERVAL", "1.0"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "50"))
AUDIO_INPUT_SAMPLE_RATE = 16000
AUDIO_OUTPUT_SAMPLE_RATE = 24000
AUDIO_CHUNK_MS = 100

# OpenClaw
OPENCLAW_HOST = os.getenv("OPENCLAW_HOST", "http://localhost")
OPENCLAW_PORT = int(os.getenv("OPENCLAW_PORT", "18789"))
OPENCLAW_TOKEN = os.getenv("OPENCLAW_TOKEN", "YOUR_OPENCLAW_TOKEN")

# Dashboard
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))

# System instruction for Gemini
SYSTEM_INSTRUCTION = """You are an AI assistant for someone wearing Aria smart glasses. \
You can see through their camera and have a voice conversation. \
Keep responses concise and natural.

CRITICAL: You have NO memory, NO storage, and NO ability to take actions on your own.

You have exactly ONE tool: execute. \
ALWAYS use execute when the user asks you to:
- Send a message to someone
- Search or look up anything
- Add, create, or modify anything
- Research, analyze, or draft anything
- Control or interact with apps, devices, or services
- Remember or store any information for later

IMPORTANT: Before calling execute, ALWAYS speak a brief acknowledgment first.
"""

# Context Persistence
CONTEXT_ENABLED = os.getenv("CONTEXT_ENABLED", "true").lower() == "true"
CONTEXT_OUTPUT_DIR = os.getenv("CONTEXT_OUTPUT_DIR", "")  # empty = auto-detect OpenClaw workspace
OPENCLAW_WORKSPACE = os.getenv("OPENCLAW_WORKSPACE", os.path.expanduser("~/.openclaw/workspace"))

# Whisper transcription
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # tiny, base, small, medium
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

# Scene capture
SCENE_CHANGE_THRESHOLD = int(os.getenv("SCENE_CHANGE_THRESHOLD", "10"))  # pHash hamming distance
SCENE_IDLE_INTERVAL = float(os.getenv("SCENE_IDLE_INTERVAL", "10.0"))  # seconds between frames when stable
SCENE_ACTIVE_INTERVAL = float(os.getenv("SCENE_ACTIVE_INTERVAL", "1.0"))  # seconds between frames on change
SCENE_BURST_FPS = float(os.getenv("SCENE_BURST_FPS", "3.0"))  # fps during burst capture
SCENE_BURST_DURATION = float(os.getenv("SCENE_BURST_DURATION", "5.0"))  # seconds of burst

# Scene description (Gemini Flash batch)
SCENE_DESCRIPTION_BATCH_INTERVAL = float(os.getenv("SCENE_DESCRIPTION_BATCH_INTERVAL", "300.0"))  # 5 min

# PPG / Heart rate
PPG_SAMPLE_INTERVAL = float(os.getenv("PPG_SAMPLE_INTERVAL", "30.0"))  # seconds
PPG_ANOMALY_THRESHOLD = float(os.getenv("PPG_ANOMALY_THRESHOLD", "2.0"))  # z-score

# Spatial
LOCATION_CHANGE_METERS = float(os.getenv("LOCATION_CHANGE_METERS", "3.0"))
GAZE_FOCUS_MIN_SECONDS = float(os.getenv("GAZE_FOCUS_MIN_SECONDS", "2.0"))
ACTIVITY_ORIENTATION_THRESHOLD = float(os.getenv("ACTIVITY_ORIENTATION_THRESHOLD", "30.0"))  # degrees

# Memory management
MEMORY_FULL_RETENTION_DAYS = int(os.getenv("MEMORY_FULL_RETENTION_DAYS", "7"))
MEMORY_PRUNED_RETENTION_DAYS = int(os.getenv("MEMORY_PRUNED_RETENTION_DAYS", "30"))
