"""Tests for SceneProcessor - adaptive frame capture via perceptual hash."""
import os
import shutil
import tempfile

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

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_first_frame_always_captured(self):
        frame = self._make_solid_frame(128)
        self.processor.process_frame(frame)
        self.processor.flush()
        assert self.processor.keyframe_count == 1

    def test_identical_frames_not_captured(self):
        frame = self._make_solid_frame(128)
        self.processor.process_frame(frame)
        self.processor.flush()
        self.processor.process_frame(frame)
        self.processor.flush()
        self.processor.process_frame(frame)
        self.processor.flush()
        assert self.processor.keyframe_count == 1

    def test_different_frame_captured(self):
        frame1 = self._make_solid_frame(50)
        frame2 = self._make_solid_frame(200)
        self.processor.process_frame(frame1)
        self.processor.flush()
        self.processor.process_frame(frame2)
        self.processor.flush()
        assert self.processor.keyframe_count == 2

    def test_keyframe_saved_to_disk(self):
        frame = self._make_solid_frame(128)
        self.processor.process_frame(frame)
        self.processor.flush()
        jpgs = [f for f in os.listdir(self.tmpdir) if f.endswith(".jpg")]
        assert len(jpgs) >= 1

    def test_scene_change_callback_fires(self):
        frame1 = self._make_solid_frame(50)
        frame2 = self._make_solid_frame(200)
        self.processor.process_frame(frame1)
        self.processor.flush()
        self.processor.process_frame(frame2)
        self.processor.flush()
        assert len(self.captured_events) == 2  # first frame + change

    def _make_solid_frame(self, value: int) -> np.ndarray:
        """Create a 100x100 RGB frame with a distinct pattern based on value.

        Uses value as a seed to produce structurally different images so that
        perceptual hashing (phash) reports a large distance between them.
        """
        rng = np.random.RandomState(value)
        return rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
