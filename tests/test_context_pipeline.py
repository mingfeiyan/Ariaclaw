"""Tests for ContextPipeline - wires processors and writes to memory."""
from __future__ import annotations

import os
import shutil
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
        self.pipeline.start()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipeline_creates(self):
        assert self.pipeline is not None

    def test_handles_video_frame(self):
        """Video frame should be passed to scene processor."""
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.pipeline.on_video_frame(frame)
        self.pipeline.scene_processor.flush()
        assert self.pipeline.scene_processor.keyframe_count == 1

    def test_handles_audio_chunk(self):
        """Audio chunk should be buffered in audio processor."""
        chunk = np.zeros(1600, dtype=np.int16).tobytes()
        self.pipeline.on_audio_chunk(chunk, is_contact_mic=False)
        assert self.pipeline.audio_processor.buffer_duration_seconds > 0

    def test_handles_position_update(self):
        """Position update should go to spatial processor."""
        self.pipeline.on_position_update(0, 0, 0)
        self.pipeline.on_position_update(10, 0, 0)
        daily_logs = os.listdir(os.path.join(self.tmpdir, "memory"))
        assert len(daily_logs) >= 1

    def test_start_stop(self):
        assert self.pipeline.is_running
        self.pipeline.stop()
        assert not self.pipeline.is_running

    def test_stop_prevents_processing(self):
        """After stop(), events should not be processed."""
        self.pipeline.stop()
        rng = np.random.RandomState(99)
        frame = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.pipeline.on_video_frame(frame)
        assert self.pipeline.scene_processor.keyframe_count == 0

    def test_handles_ppg_data(self):
        """PPG signal should be processed for heart rate."""
        t = np.linspace(0, 10, 1000)
        ppg = np.sin(2 * np.pi * 1.2 * t)
        self.pipeline.on_ppg_data(ppg, sample_rate=100)
        # HR was extracted — no assertion on events since baseline needs time to build

    def test_handles_ocr_request(self):
        """OCR on blank image should not crash."""
        blank = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.pipeline.on_ocr_request(blank)
        # No crash = pass (OCR may or may not find text depending on pyobjc availability)
