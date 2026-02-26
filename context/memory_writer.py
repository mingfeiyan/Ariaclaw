"""Writes structured context events as Markdown to OpenClaw-compatible daily logs."""
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ContextEvent:
    timestamp: datetime
    event_type: str
    content: dict = field(default_factory=dict)


class MemoryWriter:
    """Appends context events as Markdown to OpenClaw daily log files."""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        self._memory_dir = os.path.join(output_dir, "memory")
        self._lock = threading.Lock()

    def write_event(self, event: ContextEvent):
        os.makedirs(self._memory_dir, exist_ok=True)
        date_str = event.timestamp.strftime("%Y-%m-%d")
        time_str = event.timestamp.strftime("%H:%M")
        log_path = os.path.join(self._memory_dir, f"{date_str}.md")
        line = self._format_event(event, time_str)
        with self._lock:
            if not os.path.exists(log_path):
                with open(log_path, "w") as f:
                    f.write(f"# {date_str}\n\n")
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
        return f"- **{time_str}** — **{t}**: {c}\n"
