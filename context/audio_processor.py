"""Audio processing: buffering, Whisper transcription, speaker labeling, prosody."""
import logging
import queue
import threading
import time
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

try:
    from pywhispercpp.model import Model as WhisperModel
    WHISPER_AVAILABLE = True
except (ImportError, TypeError):
    # TypeError: pywhispercpp uses union type syntax (bool | TextIO) requiring Python 3.10+
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
        self._buffer_is_contact_mic = False

        self._whisper = None
        self._whisper_model_name = whisper_model
        if WHISPER_AVAILABLE:
            try:
                self._whisper = WhisperModel(whisper_model)
                logger.info("Whisper model loaded: %s", whisper_model)
            except Exception as e:
                logger.error("Failed to load Whisper model: %s", e)

        # Worker thread for transcription (avoids blocking SDK callback thread)
        self._work_queue = queue.Queue(maxsize=4)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.on_transcription = None

    @property
    def buffer_duration_seconds(self) -> float:
        with self._buffer_lock:
            return len(self._buffer) / (self._sample_rate * 2)

    _SENTINEL = None  # signals worker thread to exit

    def add_audio_chunk(self, pcm_bytes: bytes, is_contact_mic: bool = False):
        """Add Int16 PCM audio. Enqueues transcription when buffer is full."""
        with self._buffer_lock:
            self._buffer.extend(pcm_bytes)
            if is_contact_mic:
                self._buffer_is_contact_mic = True

            if len(self._buffer) >= self._buffer_bytes_threshold:
                audio_data = bytes(self._buffer)
                contact = self._buffer_is_contact_mic
                self._buffer = bytearray()
                self._buffer_is_contact_mic = False
            else:
                return

        # Enqueue for worker thread; drop if queue is full to avoid backpressure
        try:
            self._work_queue.put_nowait((audio_data, contact))
        except queue.Full:
            logger.warning("Transcription queue full, dropping audio buffer")

    def stop(self):
        """Flush pending work and stop the worker thread."""
        self._work_queue.put(self._SENTINEL)
        self._worker.join(timeout=5)

    def _worker_loop(self):
        """Background thread that processes audio buffers for transcription."""
        while True:
            item = self._work_queue.get()
            if item is self._SENTINEL:
                break
            audio_data, contact = item
            try:
                self._process_buffer(audio_data, contact)
            except Exception as e:
                logger.error("Audio processing error: %s", e)

    def _process_buffer(self, pcm_bytes: bytes, is_contact_mic: bool):
        """Transcribe buffered audio and emit event."""
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = pcm_int16.astype(np.float32) / 32768.0

        prosody = self._analyze_prosody(audio_float)
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
