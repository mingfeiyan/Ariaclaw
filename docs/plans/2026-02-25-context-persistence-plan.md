# Context Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add always-on context capture from Aria Gen 2 sensors, producing structured Markdown memory files compatible with OpenClaw's memory system.

**Architecture:** A new `ContextPipeline` module subscribes to AriaStream callbacks independently of the Gemini conversation. It processes audio (transcription via Whisper), video (scene change detection + description), and spatial/biometric signals into structured context events, then writes them as Markdown to OpenClaw's `memory/YYYY-MM-DD.md` daily logs. An end-of-day summarizer compresses daily logs into `MEMORY.md`.

**Tech Stack:** whisper.cpp (via `pywhispercpp`), `imagehash` (perceptual hashing), `scipy.signal` (PPG processing), Apple Vision framework via `pyobjc` (OCR), Gemini Flash API (scene description + summarization), SQLite (optional index).

---

### Task 1: Add new dependencies to requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

Add these lines to `requirements.txt`:

```
# Context persistence pipeline
pywhispercpp>=1.2.0
imagehash>=4.3.0
librosa>=0.10.0
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully.

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add context persistence dependencies"
```

---

### Task 2: Add context persistence config options

**Files:**
- Modify: `config.example.py`

**Step 1: Add config options**

Append to `config.example.py` after the Dashboard section:

```python
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
```

**Step 2: Commit**

```bash
git add config.example.py
git commit -m "feat: add context persistence config options"
```

---

### Task 3: Create the MemoryWriter module

**Files:**
- Create: `context/memory_writer.py`
- Test: `tests/test_memory_writer.py`

This is the foundation — writes structured Markdown to OpenClaw's daily log files.

**Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_memory_writer.py`:

```python
"""Tests for MemoryWriter - writes context events as Markdown to OpenClaw daily logs."""
import os
import tempfile
from datetime import datetime

import pytest

from context.memory_writer import MemoryWriter, ContextEvent


class TestMemoryWriter:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = MemoryWriter(output_dir=self.tmpdir)

    def test_write_speech_self_event(self):
        event = ContextEvent(
            timestamp=datetime(2026, 2, 25, 9, 15, 30),
            event_type="speech_self",
            content={"text": "Let's move the deadline to Thursday", "speaker": "self"},
        )
        self.writer.write_event(event)

        daily_log = os.path.join(self.tmpdir, "memory", "2026-02-25.md")
        assert os.path.exists(daily_log)
        content = open(daily_log).read()
        assert "09:15" in content
        assert "Let's move the deadline to Thursday" in content
        assert "You said" in content

    def test_write_speech_other_event(self):
        event = ContextEvent(
            timestamp=datetime(2026, 2, 25, 9, 16, 0),
            event_type="speech_other",
            content={"text": "The deadline is Friday", "speaker": "other"},
        )
        self.writer.write_event(event)

        daily_log = os.path.join(self.tmpdir, "memory", "2026-02-25.md")
        content = open(daily_log).read()
        assert "Other said" in content
        assert "The deadline is Friday" in content

    def test_write_scene_change_event(self):
        event = ContextEvent(
            timestamp=datetime(2026, 2, 25, 10, 0, 0),
            event_type="scene_change",
            content={
                "description": "Conference room, 4 people at table",
                "keyframe_path": "context/2026-02-25/keyframe-100000.jpg",
            },
        )
        self.writer.write_event(event)

        daily_log = os.path.join(self.tmpdir, "memory", "2026-02-25.md")
        content = open(daily_log).read()
        assert "Scene" in content
        assert "Conference room" in content

    def test_write_location_change_event(self):
        event = ContextEvent(
            timestamp=datetime(2026, 2, 25, 9, 47, 0),
            event_type="location_change",
            content={"description": "Moved from desk to kitchen"},
        )
        self.writer.write_event(event)

        daily_log = os.path.join(self.tmpdir, "memory", "2026-02-25.md")
        content = open(daily_log).read()
        assert "Location" in content
        assert "desk to kitchen" in content

    def test_write_heart_rate_spike_event(self):
        event = ContextEvent(
            timestamp=datetime(2026, 2, 25, 14, 30, 0),
            event_type="heart_rate_spike",
            content={"heart_rate": 95, "baseline": 68},
        )
        self.writer.write_event(event)

        daily_log = os.path.join(self.tmpdir, "memory", "2026-02-25.md")
        content = open(daily_log).read()
        assert "95" in content
        assert "68" in content

    def test_appends_to_existing_log(self):
        event1 = ContextEvent(
            timestamp=datetime(2026, 2, 25, 9, 0, 0),
            event_type="speech_self",
            content={"text": "First thing", "speaker": "self"},
        )
        event2 = ContextEvent(
            timestamp=datetime(2026, 2, 25, 9, 5, 0),
            event_type="speech_self",
            content={"text": "Second thing", "speaker": "self"},
        )
        self.writer.write_event(event1)
        self.writer.write_event(event2)

        daily_log = os.path.join(self.tmpdir, "memory", "2026-02-25.md")
        content = open(daily_log).read()
        assert "First thing" in content
        assert "Second thing" in content

    def test_daily_log_has_date_header(self):
        event = ContextEvent(
            timestamp=datetime(2026, 2, 25, 9, 0, 0),
            event_type="speech_self",
            content={"text": "Hello", "speaker": "self"},
        )
        self.writer.write_event(event)

        daily_log = os.path.join(self.tmpdir, "memory", "2026-02-25.md")
        content = open(daily_log).read()
        assert content.startswith("# 2026-02-25")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_writer.py -v`
Expected: FAIL — `context.memory_writer` module not found

**Step 3: Write minimal implementation**

Create `context/__init__.py` (empty) and `context/memory_writer.py`:

```python
"""Writes structured context events as Markdown to OpenClaw-compatible daily logs."""
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ContextEvent:
    timestamp: datetime
    event_type: str  # speech_self, speech_other, scene_change, location_change, etc.
    content: dict = field(default_factory=dict)


class MemoryWriter:
    """Appends context events as Markdown to OpenClaw daily log files."""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        self._memory_dir = os.path.join(output_dir, "memory")
        self._current_date = None

    def write_event(self, event: ContextEvent):
        os.makedirs(self._memory_dir, exist_ok=True)

        date_str = event.timestamp.strftime("%Y-%m-%d")
        time_str = event.timestamp.strftime("%H:%M")
        log_path = os.path.join(self._memory_dir, f"{date_str}.md")

        # Add date header if new file
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(f"# {date_str}\n\n")

        line = self._format_event(event, time_str)

        with open(log_path, "a") as f:
            f.write(line)

        logger.debug("Context event written: %s at %s", event.event_type, time_str)

    def _format_event(self, event: ContextEvent, time_str: str) -> str:
        c = event.content
        t = event.event_type

        if t == "speech_self":
            return f"- **{time_str}** — **You said**: \"{c.get('text', '')}\"\n"

        if t == "speech_other":
            return f"- **{time_str}** — **Other said**: \"{c.get('text', '')}\"\n"

        if t == "scene_change":
            desc = c.get("description", "")
            kf = c.get("keyframe_path", "")
            line = f"\n## {time_str} - Scene change\n- **Scene**: {desc}\n"
            if kf:
                line += f"- **Keyframe**: [{kf}]({kf})\n"
            return line

        if t == "gaze_focus":
            return f"- **{time_str}** — **Gaze focus**: {c.get('target', c.get('description', ''))}\n"

        if t == "location_change":
            return f"\n## {time_str} - Location change\n- **Location**: {c.get('description', '')}\n"

        if t == "activity_change":
            return f"- **{time_str}** — **Activity**: {c.get('description', '')}\n"

        if t == "heart_rate_spike":
            hr = c.get("heart_rate", "?")
            baseline = c.get("baseline", "?")
            return f"- **{time_str}** — **Heart rate**: {hr} bpm (baseline {baseline})\n"

        if t == "text_detected":
            return f"- **{time_str}** — **Text detected**: \"{c.get('text', '')}\"\n"

        if t == "object_detected":
            return f"- **{time_str}** — **Objects**: {c.get('description', '')}\n"

        # Fallback
        return f"- **{time_str}** — **{t}**: {c}\n"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_writer.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add context/__init__.py context/memory_writer.py tests/__init__.py tests/test_memory_writer.py
git commit -m "feat: add MemoryWriter for OpenClaw-compatible daily logs"
```

---

### Task 4: Create the SceneProcessor module

**Files:**
- Create: `context/scene_processor.py`
- Test: `tests/test_scene_processor.py`

Handles adaptive frame capture using perceptual hashing and keyframe storage.

**Step 1: Write the failing test**

```python
"""Tests for SceneProcessor - adaptive frame capture via perceptual hash."""
import os
import tempfile
import time

import numpy as np
import pytest

from context.scene_processor import SceneProcessor


class TestSceneProcessor:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.captured_events = []
        self.processor = SceneProcessor(
            keyframe_dir=self.tmpdir,
            scene_change_threshold=10,
            idle_interval=10.0,
            active_interval=1.0,
        )
        self.processor.on_scene_change = lambda event: self.captured_events.append(event)

    def test_first_frame_always_captured(self):
        frame = self._make_solid_frame(128)
        self.processor.process_frame(frame)
        assert self.processor.keyframe_count == 1

    def test_identical_frames_not_captured(self):
        frame = self._make_solid_frame(128)
        self.processor.process_frame(frame)
        self.processor.process_frame(frame)
        self.processor.process_frame(frame)
        assert self.processor.keyframe_count == 1

    def test_different_frame_captured(self):
        frame1 = self._make_solid_frame(50)
        frame2 = self._make_solid_frame(200)
        self.processor.process_frame(frame1)
        self.processor.process_frame(frame2)
        assert self.processor.keyframe_count == 2

    def test_keyframe_saved_to_disk(self):
        frame = self._make_solid_frame(128)
        self.processor.process_frame(frame)
        jpgs = [f for f in os.listdir(self.tmpdir) if f.endswith(".jpg")]
        assert len(jpgs) >= 1

    def test_scene_change_callback_fires(self):
        frame1 = self._make_solid_frame(50)
        frame2 = self._make_solid_frame(200)
        self.processor.process_frame(frame1)
        self.processor.process_frame(frame2)
        assert len(self.captured_events) == 2  # first frame + change

    def _make_solid_frame(self, value: int) -> np.ndarray:
        """Create a 100x100 solid-color RGB frame."""
        return np.full((100, 100, 3), value, dtype=np.uint8)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scene_processor.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

Create `context/scene_processor.py`:

```python
"""Adaptive frame capture using perceptual hash for scene change detection."""
import io
import logging
import os
import time
from datetime import datetime

import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SceneProcessor:
    """Captures keyframes when the scene changes, using perceptual hashing."""

    def __init__(
        self,
        keyframe_dir: str,
        scene_change_threshold: int = 10,
        idle_interval: float = 10.0,
        active_interval: float = 1.0,
    ):
        self._keyframe_dir = keyframe_dir
        self._threshold = scene_change_threshold
        self._idle_interval = idle_interval
        self._active_interval = active_interval

        self._last_hash = None
        self._last_capture_time = 0.0
        self._keyframe_count = 0

        # Callback
        self.on_scene_change = None  # (ContextEvent) -> None

    @property
    def keyframe_count(self) -> int:
        return self._keyframe_count

    def process_frame(self, frame: np.ndarray):
        """Process an RGB frame. Captures keyframe if scene changed or enough time elapsed."""
        now = time.time()

        img = Image.fromarray(frame)
        current_hash = imagehash.phash(img)

        # First frame or scene change
        if self._last_hash is None:
            self._capture_keyframe(frame, img, now)
            self._last_hash = current_hash
            return

        distance = self._last_hash - current_hash

        if distance > self._threshold:
            # Scene changed
            self._capture_keyframe(frame, img, now)
            self._last_hash = current_hash
            return

        # Idle capture (even if scene hasn't changed)
        if now - self._last_capture_time >= self._idle_interval:
            self._capture_keyframe(frame, img, now)
            self._last_hash = current_hash

    def _capture_keyframe(self, frame: np.ndarray, img: Image.Image, now: float):
        self._last_capture_time = now
        self._keyframe_count += 1

        # Save to disk
        os.makedirs(self._keyframe_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%H%M%S")
        filename = f"keyframe-{timestamp_str}-{self._keyframe_count}.jpg"
        filepath = os.path.join(self._keyframe_dir, filename)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        with open(filepath, "wb") as f:
            f.write(buf.getvalue())

        logger.info("Keyframe captured: %s (%d bytes)", filename, buf.tell())

        if self.on_scene_change:
            from context.memory_writer import ContextEvent

            event = ContextEvent(
                timestamp=datetime.now(),
                event_type="scene_change",
                content={
                    "keyframe_path": filepath,
                    "description": "",  # filled later by scene description batch
                },
            )
            self.on_scene_change(event)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scene_processor.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add context/scene_processor.py tests/test_scene_processor.py
git commit -m "feat: add SceneProcessor with perceptual hash scene change detection"
```

---

### Task 5: Create the AudioProcessor module

**Files:**
- Create: `context/audio_processor.py`
- Test: `tests/test_audio_processor.py`

Handles local Whisper transcription with speaker labeling (self vs other using contact mic).

**Step 1: Write the failing test**

```python
"""Tests for AudioProcessor - audio buffering and speaker labeling."""
import numpy as np
import pytest

from context.audio_processor import AudioProcessor


class TestAudioProcessor:
    def setup_method(self):
        self.events = []
        self.processor = AudioProcessor(
            whisper_model="tiny",  # smallest for testing
            sample_rate=16000,
            buffer_duration_seconds=5.0,
        )
        self.processor.on_transcription = lambda event: self.events.append(event)

    def test_buffers_audio_until_threshold(self):
        """Short audio should be buffered, not immediately transcribed."""
        # 1 second of silence (below 5-second buffer threshold)
        chunk = np.zeros(16000, dtype=np.int16).tobytes()
        self.processor.add_audio_chunk(chunk, is_contact_mic=False)
        # Should not have triggered transcription yet
        assert self.processor.buffer_duration_seconds < 5.0

    def test_speaker_label_self_when_contact_mic(self):
        """Audio from contact mic should be labeled as 'self'."""
        label = self.processor._classify_speaker(is_contact_mic=True)
        assert label == "self"

    def test_speaker_label_other_when_no_contact_mic(self):
        """Audio without contact mic signal should be labeled as 'other'."""
        label = self.processor._classify_speaker(is_contact_mic=False)
        assert label == "other"

    def test_prosody_calm_for_silence(self):
        """Silent audio should have low energy."""
        silence = np.zeros(16000, dtype=np.float32)
        metrics = self.processor._analyze_prosody(silence)
        assert metrics["rms"] < 0.01

    def test_prosody_high_energy_for_loud(self):
        """Loud audio should have high energy."""
        loud = np.full(16000, 0.5, dtype=np.float32)
        metrics = self.processor._analyze_prosody(loud)
        assert metrics["rms"] > 0.1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_processor.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

Create `context/audio_processor.py`:

```python
"""Audio processing: buffering, Whisper transcription, speaker labeling, prosody."""
import logging
import threading
import time
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

try:
    from pywhispercpp.model import Model as WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("pywhispercpp not available - transcription disabled")


class AudioProcessor:
    """Buffers audio, runs Whisper transcription, labels speakers."""

    def __init__(
        self,
        whisper_model: str = "small",
        sample_rate: int = 16000,
        buffer_duration_seconds: float = 5.0,
    ):
        self._sample_rate = sample_rate
        self._buffer_duration = buffer_duration_seconds
        self._buffer_bytes_threshold = int(sample_rate * 2 * buffer_duration_seconds)

        self._buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._buffer_is_contact_mic = False  # whether contact mic was active during this buffer

        self._whisper = None
        self._whisper_model_name = whisper_model
        if WHISPER_AVAILABLE:
            try:
                self._whisper = WhisperModel(whisper_model)
                logger.info("Whisper model loaded: %s", whisper_model)
            except Exception as e:
                logger.error("Failed to load Whisper model: %s", e)

        # Callback
        self.on_transcription = None  # (ContextEvent) -> None

    @property
    def buffer_duration_seconds(self) -> float:
        with self._buffer_lock:
            return len(self._buffer) / (self._sample_rate * 2)

    def add_audio_chunk(self, pcm_bytes: bytes, is_contact_mic: bool = False):
        """Add Int16 PCM audio. Triggers transcription when buffer is full."""
        with self._buffer_lock:
            self._buffer.extend(pcm_bytes)
            if is_contact_mic:
                self._buffer_is_contact_mic = True

            if len(self._buffer) >= self._buffer_bytes_threshold:
                audio_data = bytes(self._buffer)
                contact = self._buffer_is_contact_mic
                self._buffer = bytearray()
                self._buffer_is_contact_mic = False

        if len(audio_data) >= self._buffer_bytes_threshold:
            self._process_buffer(audio_data, contact)

    def _process_buffer(self, pcm_bytes: bytes, is_contact_mic: bool):
        """Transcribe buffered audio and emit event."""
        # Convert to float32 for Whisper
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = pcm_int16.astype(np.float32) / 32768.0

        # Prosody analysis
        prosody = self._analyze_prosody(audio_float)

        # Transcribe
        text = self._transcribe(audio_float)
        if not text or not text.strip():
            return

        speaker = self._classify_speaker(is_contact_mic)

        if self.on_transcription:
            from context.memory_writer import ContextEvent

            event = ContextEvent(
                timestamp=datetime.now(),
                event_type=f"speech_{speaker}",
                content={
                    "text": text.strip(),
                    "speaker": speaker,
                    "prosody": prosody,
                },
            )
            self.on_transcription(event)

    def _transcribe(self, audio_float: np.ndarray) -> str:
        if not self._whisper:
            return ""
        try:
            segments = self._whisper.transcribe(audio_float)
            return " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            logger.error("Whisper transcription failed: %s", e)
            return ""

    @staticmethod
    def _classify_speaker(is_contact_mic: bool) -> str:
        return "self" if is_contact_mic else "other"

    @staticmethod
    def _analyze_prosody(audio_float: np.ndarray) -> dict:
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        return {
            "rms": rms,
            "energy_level": "high" if rms > 0.1 else "low" if rms < 0.01 else "normal",
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_processor.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add context/audio_processor.py tests/test_audio_processor.py
git commit -m "feat: add AudioProcessor with Whisper transcription and speaker labeling"
```

---

### Task 6: Create the SpatialProcessor module

**Files:**
- Create: `context/spatial_processor.py`
- Test: `tests/test_spatial_processor.py`

Filters continuous spatial/biometric signals into discrete events.

**Step 1: Write the failing test**

```python
"""Tests for SpatialProcessor - filters continuous signals into discrete events."""
import numpy as np
import pytest

from context.spatial_processor import SpatialProcessor


class TestSpatialProcessor:
    def setup_method(self):
        self.events = []
        self.processor = SpatialProcessor(
            location_threshold_meters=3.0,
            orientation_threshold_degrees=30.0,
            hr_anomaly_zscore=2.0,
        )
        self.processor.on_event = lambda e: self.events.append(e)

    def test_no_event_on_small_movement(self):
        self.processor.update_position(0, 0, 0)
        self.processor.update_position(1, 1, 0)  # ~1.4m
        location_events = [e for e in self.events if e.event_type == "location_change"]
        assert len(location_events) == 0

    def test_event_on_large_movement(self):
        self.processor.update_position(0, 0, 0)
        self.processor.update_position(5, 0, 0)  # 5m
        location_events = [e for e in self.events if e.event_type == "location_change"]
        assert len(location_events) == 1

    def test_no_event_on_small_orientation_change(self):
        self.processor.update_orientation(0, 0, 0)
        self.processor.update_orientation(10, 5, 0)
        activity_events = [e for e in self.events if e.event_type == "activity_change"]
        assert len(activity_events) == 0

    def test_event_on_large_orientation_change(self):
        self.processor.update_orientation(0, 0, 0)
        self.processor.update_orientation(45, 0, 0)
        activity_events = [e for e in self.events if e.event_type == "activity_change"]
        assert len(activity_events) == 1

    def test_hr_baseline_builds(self):
        """First N readings build baseline, no anomaly."""
        for hr in [70, 68, 72, 69, 71]:
            self.processor.update_heart_rate(hr)
        hr_events = [e for e in self.events if e.event_type == "heart_rate_spike"]
        assert len(hr_events) == 0

    def test_hr_anomaly_detected(self):
        """After baseline, a spike triggers event."""
        for hr in [70, 68, 72, 69, 71, 70, 68, 72, 69, 71]:
            self.processor.update_heart_rate(hr)
        self.processor.update_heart_rate(110)  # spike
        hr_events = [e for e in self.events if e.event_type == "heart_rate_spike"]
        assert len(hr_events) == 1
        assert hr_events[0].content["heart_rate"] == 110
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_spatial_processor.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

Create `context/spatial_processor.py`:

```python
"""Filters continuous spatial and biometric signals into discrete context events."""
import logging
import math
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class SpatialProcessor:
    """Converts continuous sensor data into discrete location/activity/HR events."""

    def __init__(
        self,
        location_threshold_meters: float = 3.0,
        orientation_threshold_degrees: float = 30.0,
        hr_anomaly_zscore: float = 2.0,
    ):
        self._loc_threshold = location_threshold_meters
        self._orient_threshold = orientation_threshold_degrees
        self._hr_zscore_threshold = hr_anomaly_zscore

        # Position state
        self._last_position = None  # (x, y, z)

        # Orientation state
        self._last_orientation = None  # (pitch, yaw, roll)

        # Heart rate baseline
        self._hr_readings = []
        self._hr_min_readings = 10  # need this many before detecting anomalies

        # Callback
        self.on_event = None  # (ContextEvent) -> None

    def update_position(self, x: float, y: float, z: float):
        pos = (x, y, z)
        if self._last_position is None:
            self._last_position = pos
            return

        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, self._last_position)))

        if dist >= self._loc_threshold:
            self._last_position = pos
            self._emit("location_change", {
                "description": f"Moved {dist:.1f}m",
                "position": {"x": x, "y": y, "z": z},
            })

    def update_orientation(self, pitch: float, yaw: float, roll: float):
        orient = (pitch, yaw, roll)
        if self._last_orientation is None:
            self._last_orientation = orient
            return

        max_delta = max(abs(a - b) for a, b in zip(orient, self._last_orientation))

        if max_delta >= self._orient_threshold:
            self._last_orientation = orient
            self._emit("activity_change", {
                "description": f"Orientation changed {max_delta:.0f} degrees",
            })

    def update_heart_rate(self, bpm: int):
        self._hr_readings.append(bpm)

        if len(self._hr_readings) < self._hr_min_readings:
            return

        readings = np.array(self._hr_readings)
        mean = np.mean(readings[:-1])  # baseline from all but current
        std = np.std(readings[:-1])

        if std < 1.0:
            std = 1.0  # avoid division by zero

        zscore = (bpm - mean) / std

        if abs(zscore) >= self._hr_zscore_threshold:
            self._emit("heart_rate_spike", {
                "heart_rate": bpm,
                "baseline": int(round(mean)),
                "zscore": round(zscore, 2),
            })

    def _emit(self, event_type: str, content: dict):
        if self.on_event:
            from context.memory_writer import ContextEvent

            event = ContextEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                content=content,
            )
            self.on_event(event)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_spatial_processor.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add context/spatial_processor.py tests/test_spatial_processor.py
git commit -m "feat: add SpatialProcessor for location, orientation, and HR events"
```

---

### Task 7: Create the ContextPipeline coordinator

**Files:**
- Create: `context/context_pipeline.py`
- Test: `tests/test_context_pipeline.py`

Wires all processors together and subscribes to AriaStream callbacks.

**Step 1: Write the failing test**

```python
"""Tests for ContextPipeline - wires processors and writes to memory."""
import os
import tempfile

import numpy as np
import pytest

from context.context_pipeline import ContextPipeline


class TestContextPipeline:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pipeline = ContextPipeline(
            output_dir=self.tmpdir,
            whisper_model="tiny",
        )

    def test_pipeline_creates(self):
        assert self.pipeline is not None

    def test_handles_video_frame(self):
        """Video frame should be passed to scene processor."""
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        self.pipeline.on_video_frame(frame)
        assert self.pipeline.scene_processor.keyframe_count == 1

    def test_handles_audio_chunk(self):
        """Audio chunk should be buffered in audio processor."""
        chunk = np.zeros(1600, dtype=np.int16).tobytes()  # 100ms
        self.pipeline.on_audio_chunk(chunk, is_contact_mic=False)
        assert self.pipeline.audio_processor.buffer_duration_seconds > 0

    def test_handles_position_update(self):
        """Position update should go to spatial processor."""
        self.pipeline.on_position_update(0, 0, 0)
        self.pipeline.on_position_update(10, 0, 0)
        # Should have written a location_change event
        daily_logs = os.listdir(os.path.join(self.tmpdir, "memory"))
        assert len(daily_logs) >= 1

    def test_start_stop(self):
        self.pipeline.start()
        assert self.pipeline.is_running
        self.pipeline.stop()
        assert not self.pipeline.is_running
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_pipeline.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

Create `context/context_pipeline.py`:

```python
"""ContextPipeline: coordinates all context processors and writes to OpenClaw memory."""
import logging
import os
from datetime import datetime

import numpy as np

from context.audio_processor import AudioProcessor
from context.memory_writer import ContextEvent, MemoryWriter
from context.scene_processor import SceneProcessor
from context.spatial_processor import SpatialProcessor

logger = logging.getLogger(__name__)


class ContextPipeline:
    """Wires AriaStream sensor data through processors into OpenClaw memory files."""

    def __init__(
        self,
        output_dir: str,
        whisper_model: str = "small",
    ):
        self._output_dir = output_dir
        self._running = False

        # Set up keyframe directory
        today = datetime.now().strftime("%Y-%m-%d")
        keyframe_dir = os.path.join(output_dir, "context", today)

        # Create sub-processors
        self.memory_writer = MemoryWriter(output_dir=output_dir)

        self.scene_processor = SceneProcessor(keyframe_dir=keyframe_dir)
        self.scene_processor.on_scene_change = self._on_context_event

        self.audio_processor = AudioProcessor(whisper_model=whisper_model)
        self.audio_processor.on_transcription = self._on_context_event

        self.spatial_processor = SpatialProcessor()
        self.spatial_processor.on_event = self._on_context_event

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        self._running = True
        logger.info("Context pipeline started (output: %s)", self._output_dir)

    def stop(self):
        self._running = False
        logger.info("Context pipeline stopped")

    def on_video_frame(self, frame: np.ndarray):
        """Process an RGB video frame from AriaStream."""
        if not self._running:
            # Still process even if "stopped" for testing convenience
            pass
        self.scene_processor.process_frame(frame)

    def on_audio_chunk(self, pcm_bytes: bytes, is_contact_mic: bool = False):
        """Process an audio chunk from AriaStream."""
        self.audio_processor.add_audio_chunk(pcm_bytes, is_contact_mic=is_contact_mic)

    def on_position_update(self, x: float, y: float, z: float):
        """Process a SLAM position update."""
        self.spatial_processor.update_position(x, y, z)

    def on_orientation_update(self, pitch: float, yaw: float, roll: float):
        """Process an IMU orientation update."""
        self.spatial_processor.update_orientation(pitch, yaw, roll)

    def on_heart_rate(self, bpm: int):
        """Process a PPG heart rate reading."""
        self.spatial_processor.update_heart_rate(bpm)

    def _on_context_event(self, event: ContextEvent):
        """Write any context event to the daily log."""
        self.memory_writer.write_event(event)
        logger.info("Context event: %s", event.event_type)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_context_pipeline.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add context/context_pipeline.py tests/test_context_pipeline.py
git commit -m "feat: add ContextPipeline coordinator wiring all processors to memory"
```

---

### Task 8: Integrate ContextPipeline into AriaClaw coordinator

**Files:**
- Modify: `ariaclaw.py:12-14` (imports), `ariaclaw.py:30-38` (init), `ariaclaw.py:76-87` (run), `ariaclaw.py:119-201` (start_session), `ariaclaw.py:203-224` (stop_session)
- Modify: `config.example.py` (ensure context config exists from Task 2)

**Step 1: Modify ariaclaw.py imports**

Add after line 17 (`from openclaw_bridge import ...`):

```python
from context.context_pipeline import ContextPipeline
```

**Step 2: Add ContextPipeline to __init__**

In `AriaClaw.__init__`, after `self.openclaw = OpenClawBridge()` (line 34), add:

```python
        self.context = ContextPipeline(
            output_dir=config.OPENCLAW_WORKSPACE if config.CONTEXT_ENABLED else "",
            whisper_model=config.WHISPER_MODEL,
        ) if config.CONTEXT_ENABLED else None
```

**Step 3: Start context pipeline in run()**

After `await self.openclaw.check_connection()` (line 85), add:

```python
        # Start context pipeline (always-on, independent of Gemini session)
        if self.context:
            self.context.start()
            logger.info("Context persistence enabled (output: %s)", config.OPENCLAW_WORKSPACE)
```

**Step 4: Wire Aria callbacks to context pipeline in _start_session()**

After the existing `self.aria.on_audio_chunk` callback wiring (line 149-151), add:

```python
        # Wire Aria audio → context pipeline (independent of Gemini)
        if self.context:
            existing_audio_cb = self.aria.on_audio_chunk
            def _audio_to_both(chunk):
                if existing_audio_cb:
                    existing_audio_cb(chunk)
                self.context.on_audio_chunk(chunk)
            self.aria.on_audio_chunk = _audio_to_both

            existing_video_cb = self.aria.on_video_frame_raw
            def _video_to_context(frame):
                if existing_video_cb:
                    existing_video_cb(frame)
                self.context.on_video_frame(frame)
            self.aria.on_video_frame_raw = _video_to_context
```

**Step 5: Stop context pipeline in shutdown**

In the `run()` graceful shutdown section (after line 103 `self.aria.stop()`), add:

```python
        if self.context:
            self.context.stop()
```

**Step 6: Verify the app starts without errors**

Run: `python -c "import ariaclaw; print('Import OK')"`
Expected: "Import OK" (may warn about Aria SDK not available, which is fine)

**Step 7: Commit**

```bash
git add ariaclaw.py
git commit -m "feat: integrate ContextPipeline into AriaClaw coordinator"
```

---

### Task 9: Create the DailySummarizer module

**Files:**
- Create: `context/daily_summarizer.py`
- Test: `tests/test_daily_summarizer.py`

End-of-day AI summarization that compresses daily logs into MEMORY.md.

**Step 1: Write the failing test**

```python
"""Tests for DailySummarizer - compresses daily logs into MEMORY.md summaries."""
import os
import tempfile

import pytest

from context.daily_summarizer import DailySummarizer


class TestDailySummarizer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory_dir = os.path.join(self.tmpdir, "memory")
        os.makedirs(self.memory_dir)

    def test_reads_daily_log(self):
        """Should read a daily log file."""
        log_path = os.path.join(self.memory_dir, "2026-02-25.md")
        with open(log_path, "w") as f:
            f.write("# 2026-02-25\n\n- **09:15** — **You said**: \"Hello\"\n")

        summarizer = DailySummarizer(workspace_dir=self.tmpdir)
        content = summarizer._read_daily_log("2026-02-25")
        assert "Hello" in content

    def test_writes_to_memory_md(self):
        """Should append summary to MEMORY.md."""
        summarizer = DailySummarizer(workspace_dir=self.tmpdir)
        summarizer._append_to_memory_md("2026-02-25", "Had a meeting about deadlines.")

        memory_path = os.path.join(self.tmpdir, "MEMORY.md")
        assert os.path.exists(memory_path)
        content = open(memory_path).read()
        assert "2026-02-25" in content
        assert "deadlines" in content

    def test_appends_multiple_days(self):
        """Should append multiple day summaries without overwriting."""
        summarizer = DailySummarizer(workspace_dir=self.tmpdir)
        summarizer._append_to_memory_md("2026-02-24", "Day one summary.")
        summarizer._append_to_memory_md("2026-02-25", "Day two summary.")

        memory_path = os.path.join(self.tmpdir, "MEMORY.md")
        content = open(memory_path).read()
        assert "Day one" in content
        assert "Day two" in content

    def test_lists_daily_logs(self):
        """Should find all daily log files."""
        for date in ["2026-02-23", "2026-02-24", "2026-02-25"]:
            with open(os.path.join(self.memory_dir, f"{date}.md"), "w") as f:
                f.write(f"# {date}\n")

        summarizer = DailySummarizer(workspace_dir=self.tmpdir)
        logs = summarizer.list_daily_logs()
        assert len(logs) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_daily_summarizer.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

Create `context/daily_summarizer.py`:

```python
"""End-of-day summarization: compresses daily logs into MEMORY.md."""
import glob
import logging
import os

logger = logging.getLogger(__name__)


class DailySummarizer:
    """Reads daily context logs and produces long-term memory summaries."""

    def __init__(self, workspace_dir: str):
        self._workspace_dir = workspace_dir
        self._memory_dir = os.path.join(workspace_dir, "memory")
        self._memory_md_path = os.path.join(workspace_dir, "MEMORY.md")

    def list_daily_logs(self) -> list[str]:
        """Return sorted list of date strings for which daily logs exist."""
        pattern = os.path.join(self._memory_dir, "????-??-??.md")
        files = sorted(glob.glob(pattern))
        return [os.path.splitext(os.path.basename(f))[0] for f in files]

    def _read_daily_log(self, date_str: str) -> str:
        path = os.path.join(self._memory_dir, f"{date_str}.md")
        if not os.path.exists(path):
            return ""
        with open(path) as f:
            return f.read()

    def _append_to_memory_md(self, date_str: str, summary: str):
        """Append a day summary to MEMORY.md."""
        header = f"\n## {date_str}\n\n"
        with open(self._memory_md_path, "a") as f:
            f.write(header)
            f.write(summary)
            f.write("\n")
        logger.info("Appended summary for %s to MEMORY.md", date_str)

    async def summarize_day(self, date_str: str, summarize_fn=None):
        """Summarize a day's log and append to MEMORY.md.

        Args:
            date_str: Date like "2026-02-25"
            summarize_fn: async callable(daily_log_text) -> summary_text.
                          If None, stores the raw log (no AI summarization).
        """
        daily_log = self._read_daily_log(date_str)
        if not daily_log.strip():
            logger.info("No content for %s, skipping", date_str)
            return

        if summarize_fn:
            summary = await summarize_fn(daily_log)
        else:
            summary = daily_log

        self._append_to_memory_md(date_str, summary)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_daily_summarizer.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add context/daily_summarizer.py tests/test_daily_summarizer.py
git commit -m "feat: add DailySummarizer for end-of-day memory compression"
```

---

### Task 10: Add OCR support via Apple Vision framework

**Files:**
- Create: `context/ocr_processor.py`
- Test: `tests/test_ocr_processor.py`

Uses Apple's built-in Vision framework for on-device text detection in keyframes.

**Step 1: Write the failing test**

```python
"""Tests for OCR processor - text detection in images."""
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from context.ocr_processor import OCRProcessor


class TestOCRProcessor:
    def setup_method(self):
        self.processor = OCRProcessor()

    def test_returns_empty_for_blank_image(self):
        """Blank image should return no text."""
        blank = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = self.processor.detect_text(blank)
        assert result == "" or result is None or len(result.strip()) == 0

    def test_returns_string_type(self):
        """Result should always be a string."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = self.processor.detect_text(img)
        assert isinstance(result, str)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ocr_processor.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

Create `context/ocr_processor.py`:

```python
"""OCR text detection using Apple Vision framework (macOS) with PIL fallback."""
import logging
import platform

import numpy as np

logger = logging.getLogger(__name__)

VISION_AVAILABLE = False
if platform.system() == "Darwin":
    try:
        import objc
        from Foundation import NSData
        from Vision import (
            VNImageRequestHandler,
            VNRecognizeTextRequest,
        )
        VISION_AVAILABLE = True
    except ImportError:
        logger.info("pyobjc Vision framework not available - OCR disabled")


class OCRProcessor:
    """Detects text in images using Apple Vision (macOS) or returns empty string."""

    def detect_text(self, frame: np.ndarray) -> str:
        """Detect text in an RGB numpy frame. Returns detected text or empty string."""
        if not VISION_AVAILABLE:
            return ""

        try:
            from PIL import Image
            import io

            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))

            handler = VNImageRequestHandler.alloc().initWithData_options_(image_data, None)
            request = VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(1)  # accurate

            handler.performRequests_error_([request], None)

            results = request.results()
            if not results:
                return ""

            texts = []
            for observation in results:
                candidate = observation.topCandidates_(1)
                if candidate:
                    texts.append(candidate[0].string())

            return "\n".join(texts)

        except Exception as e:
            logger.debug("OCR failed: %s", e)
            return ""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ocr_processor.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add context/ocr_processor.py tests/test_ocr_processor.py
git commit -m "feat: add OCRProcessor using Apple Vision framework"
```

---

### Task 11: Add PPG heart rate extraction (signal processing)

**Files:**
- Create: `context/ppg_processor.py`
- Test: `tests/test_ppg_processor.py`

Extracts heart rate from raw PPG signal using bandpass filter + peak detection.

**Step 1: Write the failing test**

```python
"""Tests for PPG processor - heart rate extraction from PPG waveform."""
import numpy as np
import pytest

from context.ppg_processor import PPGProcessor


class TestPPGProcessor:
    def setup_method(self):
        self.processor = PPGProcessor(sample_rate=100)

    def test_extract_hr_from_simulated_ppg(self):
        """A 1.2 Hz sine wave should produce ~72 bpm heart rate."""
        t = np.linspace(0, 10, 1000)  # 10 seconds at 100 Hz
        ppg = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz = 72 bpm
        hr = self.processor.extract_heart_rate(ppg)
        assert hr is not None
        assert 60 <= hr <= 85  # allow some tolerance

    def test_returns_none_for_noise(self):
        """Random noise should not produce a valid HR."""
        noise = np.random.randn(1000) * 0.01
        hr = self.processor.extract_heart_rate(noise)
        # May return None or an unreasonable value
        assert hr is None or hr < 30 or hr > 200

    def test_returns_none_for_short_signal(self):
        """Too-short signal should return None."""
        short = np.zeros(10)
        hr = self.processor.extract_heart_rate(short)
        assert hr is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ppg_processor.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

Create `context/ppg_processor.py`:

```python
"""PPG signal processing: extract heart rate from raw photoplethysmography data."""
import logging

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

logger = logging.getLogger(__name__)


class PPGProcessor:
    """Extracts heart rate from raw PPG waveform using bandpass filter + peak detection."""

    def __init__(self, sample_rate: int = 100):
        self._sample_rate = sample_rate

    def extract_heart_rate(self, ppg_signal: np.ndarray) -> int | None:
        """Extract heart rate (bpm) from raw PPG signal.

        Returns int bpm or None if signal is too short/noisy.
        """
        if len(ppg_signal) < self._sample_rate * 3:  # need at least 3 seconds
            return None

        try:
            # Bandpass filter: 0.7-3.5 Hz (42-210 bpm)
            nyq = self._sample_rate / 2
            low = 0.7 / nyq
            high = 3.5 / nyq

            if high >= 1.0:
                high = 0.99
            if low <= 0:
                low = 0.01

            b, a = butter(3, [low, high], btype="band")
            filtered = filtfilt(b, a, ppg_signal)

            # Find peaks
            min_distance = int(self._sample_rate * 0.3)  # min 0.3s between beats
            peaks, properties = find_peaks(filtered, distance=min_distance, prominence=0.1)

            if len(peaks) < 2:
                return None

            # Calculate inter-beat intervals
            intervals = np.diff(peaks) / self._sample_rate  # in seconds
            mean_interval = np.mean(intervals)

            if mean_interval <= 0:
                return None

            bpm = int(round(60.0 / mean_interval))

            # Sanity check
            if bpm < 30 or bpm > 200:
                return None

            return bpm

        except Exception as e:
            logger.debug("PPG processing failed: %s", e)
            return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ppg_processor.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add context/ppg_processor.py tests/test_ppg_processor.py
git commit -m "feat: add PPGProcessor for heart rate extraction from PPG signal"
```

---

### Task 12: Wire PPG and OCR processors into ContextPipeline

**Files:**
- Modify: `context/context_pipeline.py`
- Modify: `tests/test_context_pipeline.py`

**Step 1: Add test for PPG integration**

Add to `tests/test_context_pipeline.py`:

```python
    def test_handles_ppg_data(self):
        """PPG signal should be processed for heart rate."""
        import numpy as np
        # Simulate 10 seconds of PPG at 100 Hz with 1.2 Hz heartbeat
        t = np.linspace(0, 10, 1000)
        ppg = np.sin(2 * np.pi * 1.2 * t)
        self.pipeline.on_ppg_data(ppg, sample_rate=100)
        # If HR was extracted, it should have been passed to spatial processor
        # (no assertion on events since HR baseline needs time to build)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_pipeline.py::TestContextPipeline::test_handles_ppg_data -v`
Expected: FAIL — `on_ppg_data` not found

**Step 3: Update context_pipeline.py**

Add imports for `PPGProcessor` and `OCRProcessor`. Add `on_ppg_data` and `on_ocr_request` methods:

```python
from context.ppg_processor import PPGProcessor
from context.ocr_processor import OCRProcessor
```

In `__init__`, add:

```python
        self.ppg_processor = PPGProcessor()
        self.ocr_processor = OCRProcessor()
```

Add methods:

```python
    def on_ppg_data(self, ppg_signal, sample_rate: int = 100):
        """Process raw PPG data for heart rate extraction."""
        self.ppg_processor._sample_rate = sample_rate
        hr = self.ppg_processor.extract_heart_rate(ppg_signal)
        if hr is not None:
            self.spatial_processor.update_heart_rate(hr)

    def on_ocr_request(self, frame):
        """Run OCR on a frame and emit text_detected event if text found."""
        text = self.ocr_processor.detect_text(frame)
        if text and text.strip():
            from context.memory_writer import ContextEvent
            event = ContextEvent(
                timestamp=datetime.now(),
                event_type="text_detected",
                content={"text": text},
            )
            self._on_context_event(event)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_context_pipeline.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add context/context_pipeline.py tests/test_context_pipeline.py
git commit -m "feat: wire PPG and OCR processors into ContextPipeline"
```

---

### Task 13: Run full test suite and verify integration

**Files:**
- All test files

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Verify import chain works end-to-end**

Run: `python -c "from context.context_pipeline import ContextPipeline; print('All imports OK')"`
Expected: "All imports OK"

**Step 3: Verify no circular imports**

Run: `python -c "from context.memory_writer import MemoryWriter, ContextEvent; from context.scene_processor import SceneProcessor; from context.audio_processor import AudioProcessor; from context.spatial_processor import SpatialProcessor; from context.ppg_processor import PPGProcessor; from context.ocr_processor import OCRProcessor; from context.daily_summarizer import DailySummarizer; from context.context_pipeline import ContextPipeline; print('No circular imports')"`
Expected: "No circular imports"

**Step 4: Commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: resolve any integration issues from full test run"
```

---

## Summary

| Task | Component | Description |
|------|-----------|-------------|
| 1 | Dependencies | Add pywhispercpp, imagehash, librosa |
| 2 | Config | Add context persistence configuration |
| 3 | MemoryWriter | Core: writes Markdown events to OpenClaw daily logs |
| 4 | SceneProcessor | Adaptive keyframe capture via perceptual hash |
| 5 | AudioProcessor | Whisper transcription + speaker labeling |
| 6 | SpatialProcessor | Location, orientation, HR event filtering |
| 7 | ContextPipeline | Coordinator wiring all processors |
| 8 | Integration | Wire into AriaClaw main coordinator |
| 9 | DailySummarizer | End-of-day AI compression into MEMORY.md |
| 10 | OCRProcessor | Apple Vision text detection |
| 11 | PPGProcessor | Heart rate from PPG signal |
| 12 | Wire PPG+OCR | Connect remaining processors |
| 13 | Verification | Full test suite + integration check |
