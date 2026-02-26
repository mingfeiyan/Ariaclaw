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
