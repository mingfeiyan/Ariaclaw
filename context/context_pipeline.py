"""ContextPipeline: coordinates all context processors and writes to OpenClaw memory."""
from __future__ import annotations

import logging
import os
from datetime import datetime

import numpy as np

from context.audio_processor import AudioProcessor
from context.memory_writer import ContextEvent, MemoryWriter
from context.ocr_processor import OCRProcessor
from context.ppg_processor import PPGProcessor
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

        self.ppg_processor = PPGProcessor()
        self.ocr_processor = OCRProcessor()

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

    def on_ppg_data(self, ppg_signal: np.ndarray, sample_rate: int = 100):
        """Process raw PPG data for heart rate extraction."""
        self.ppg_processor._sample_rate = sample_rate
        hr = self.ppg_processor.extract_heart_rate(ppg_signal)
        if hr is not None:
            self.spatial_processor.update_heart_rate(hr)

    def on_ocr_request(self, frame: np.ndarray):
        """Run OCR on a frame and emit text_detected event if text found."""
        text = self.ocr_processor.detect_text(frame)
        if text and text.strip():
            event = ContextEvent(
                timestamp=datetime.now(),
                event_type="text_detected",
                content={"text": text},
            )
            self._on_context_event(event)

    def _on_context_event(self, event: ContextEvent):
        """Write any context event to the daily log."""
        self.memory_writer.write_event(event)
        logger.info("Context event: %s", event.event_type)
