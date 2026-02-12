import logging
import queue
import threading

import pyaudio

import config

logger = logging.getLogger(__name__)


class AudioPlayback:
    """Plays Gemini's PCM Int16 24kHz audio responses through Mac speakers.

    Uses a dedicated playback thread with a queue so the caller (asyncio thread)
    is never blocked by pyaudio's blocking write().
    """

    def __init__(self):
        self._pa = None
        self._stream = None
        self.is_speaking = False
        self._queue = queue.Queue()
        self._thread = None
        self._running = False

    def start(self):
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=config.AUDIO_OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=1024,
        )
        self._running = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Audio playback started: %dHz mono Int16",
            config.AUDIO_OUTPUT_SAMPLE_RATE,
        )

    def play(self, pcm_data: bytes):
        """Queue a chunk of PCM Int16 audio for playback. Non-blocking."""
        if self._running:
            self._queue.put(pcm_data)

    def stop_playback(self):
        """Interrupt current playback (e.g., user spoke over AI)."""
        # Drain the queue to discard buffered audio
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        # Stop and restart the stream to flush pyaudio's internal buffer
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.start_stream()
            except Exception as e:
                logger.error("Audio interrupt error: %s", e)
        self.is_speaking = False

    def stop(self):
        """Shut down the audio system."""
        self._running = False
        self.is_speaking = False
        # Unblock the playback thread
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
        logger.info("Audio playback stopped")

    def _playback_loop(self):
        """Dedicated thread that drains the queue and writes to pyaudio."""
        while self._running:
            try:
                data = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self.is_speaking:
                    self.is_speaking = False
                continue
            if data is None:
                break
            self.is_speaking = True
            try:
                self._stream.write(data)
            except Exception as e:
                logger.error("Audio playback error: %s — reopening stream", e)
                self._reopen_stream()
        self.is_speaking = False

    def _reopen_stream(self):
        """Reopen the audio stream after a PortAudio error."""
        try:
            if self._stream:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    pass
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=config.AUDIO_OUTPUT_SAMPLE_RATE,
                output=True,
                frames_per_buffer=1024,
            )
            logger.info("Audio stream reopened")
        except Exception as e:
            logger.error("Failed to reopen audio stream: %s", e)
