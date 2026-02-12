import base64
import io
import logging
import threading
import time
from enum import Enum

import numpy as np
from PIL import Image
from scipy.signal import decimate

import config

logger = logging.getLogger(__name__)

try:
    import aria.sdk_gen2 as sdk_gen2
    import aria.stream_receiver as receiver
    from projectaria_tools.core.sensor_data import ImageData, ImageDataRecord

    ARIA_SDK_AVAILABLE = True
except ImportError:
    ARIA_SDK_AVAILABLE = False
    logger.warning("Aria SDK not available - aria_stream will use mock mode")


class AriaConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    ERROR = "error"


class AriaStream:
    """Connects to Aria Gen 2 glasses and streams video + audio."""

    def __init__(self):
        self.connection_state = AriaConnectionState.DISCONNECTED
        self._device_client = None
        self._device = None
        self._stream_receiver = None

        # Callbacks
        self.on_video_frame = None  # (jpeg_base64: str) -> None
        self.on_video_frame_raw = None  # (numpy_array) -> None  (for dashboard preview)
        self.on_audio_chunk = None  # (pcm_bytes: bytes) -> None
        self.on_activity_start = None  # () -> None  (speech detected)
        self.on_activity_end = None  # () -> None  (silence detected)
        self.on_state_changed = None  # (AriaConnectionState) -> None

        # Video throttling
        self._last_frame_time = 0.0

        # Audio accumulator
        self._audio_lock = threading.Lock()
        self._audio_buffer = bytearray()
        self._min_audio_bytes = int(
            config.AUDIO_INPUT_SAMPLE_RATE * 2 * (config.AUDIO_CHUNK_MS / 1000)
        )  # 3200 bytes for 100ms @ 16kHz mono Int16

        # VAD (voice activity detection) state
        self._vad_active = False  # whether we think speech is happening
        self._vad_silence_count = 0  # consecutive quiet chunks
        self._vad_speech_start_count = 0  # consecutive loud chunks to confirm speech
        self._vad_speech_threshold = 0.02  # RMS threshold for speech start
        self._vad_silence_threshold = 0.008  # RMS threshold for silence
        self._vad_speech_chunks_needed = 3  # ~150ms of loud audio to confirm speech start
        self._vad_silence_chunks_needed = 15  # ~750ms of silence to end speech
        self._vad_warmup = 20  # ignore first N audio callbacks (mic init transient)
        self._vad_suppressed = False  # suppress VAD while AI is speaking

        # Latest JPEG frame for MJPEG streaming
        self._latest_jpeg = None
        self._jpeg_lock = threading.Lock()

    def _set_state(self, state):
        self.connection_state = state
        if self.on_state_changed:
            self.on_state_changed(state)

    @property
    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg

    def connect_and_start(self):
        """Connect to Aria device and start streaming. Blocks until streaming starts."""
        if not ARIA_SDK_AVAILABLE:
            raise RuntimeError(
                "Aria SDK not available. Install projectaria-client-sdk."
            )

        self._set_state(AriaConnectionState.CONNECTING)

        try:
            # Connect to device
            self._device_client = sdk_gen2.DeviceClient()
            device_config = sdk_gen2.DeviceClientConfig()
            self._device_client.set_client_config(device_config)
            self._device = self._device_client.connect()
            logger.info("Connected to Aria device")

            # Configure streaming
            streaming_config = sdk_gen2.HttpStreamingConfig()
            streaming_config.profile_name = config.ARIA_STREAMING_PROFILE
            if config.ARIA_STREAMING_INTERFACE == "USB_NCM":
                streaming_config.streaming_interface = (
                    sdk_gen2.StreamingInterface.USB_NCM
                )
            else:
                streaming_config.streaming_interface = (
                    sdk_gen2.StreamingInterface.WIFI
                )
            self._device.set_streaming_config(streaming_config)

            # Set up stream receiver
            self._stream_receiver = receiver.StreamReceiver(
                enable_image_decoding=True,
                enable_raw_stream=False,
            )

            server_config = sdk_gen2.HttpServerConfig()
            server_config.address = "0.0.0.0"
            server_config.port = config.ARIA_STREAMING_PORT
            self._stream_receiver.set_server_config(server_config)

            # Minimize buffering for real-time
            self._stream_receiver.set_rgb_queue_size(2)
            self._stream_receiver.set_slam_queue_size(2)

            # Register callbacks
            self._stream_receiver.register_rgb_callback(self._on_rgb_frame)
            self._stream_receiver.register_audio_callback(self._on_audio_data)

            # Start
            self._device.start_streaming()
            self._stream_receiver.start_server()

            self._set_state(AriaConnectionState.STREAMING)
            logger.info("Aria streaming started")

        except Exception as e:
            logger.error("Aria connection failed: %s", e)
            self._set_state(AriaConnectionState.ERROR)
            raise

    def stop(self):
        """Stop streaming and disconnect."""
        try:
            if self._device:
                self._device.stop_streaming()
            if self._stream_receiver:
                self._stream_receiver.stop_server()
        except Exception as e:
            logger.error("Aria stop error: %s", e)
        finally:
            self._device = None
            self._device_client = None
            self._stream_receiver = None
            self._set_state(AriaConnectionState.DISCONNECTED)
            logger.info("Aria streaming stopped")

    def _on_rgb_frame(self, image_data: "ImageData", image_record: "ImageDataRecord"):
        """Called by Aria SDK for each RGB camera frame."""
        now = time.time()

        try:
            img_array = image_data.to_numpy_array()
        except Exception:
            return

        if self.on_video_frame_raw:
            self.on_video_frame_raw(img_array)

        # Encode to JPEG for MJPEG preview
        jpeg_bytes = _numpy_to_jpeg(img_array, quality=70)
        if jpeg_bytes:
            with self._jpeg_lock:
                self._latest_jpeg = jpeg_bytes

        # Throttle for Gemini (1 fps)
        if now - self._last_frame_time < config.VIDEO_FRAME_INTERVAL:
            return
        self._last_frame_time = now

        # Encode at lower quality and smaller size for Gemini
        gemini_jpeg = _numpy_to_jpeg(img_array, quality=config.JPEG_QUALITY, max_dim=768)

        if gemini_jpeg and self.on_video_frame:
            b64 = base64.b64encode(gemini_jpeg).decode("ascii")
            logger.info("Sending video frame to Gemini: %d bytes JPEG", len(gemini_jpeg))
            self.on_video_frame(b64)

    _audio_log_count = 0

    def _on_audio_data(self, audio_data, audio_record, num_channels: int):
        """Called by Aria SDK for each audio chunk from the 8-mic array."""
        self._audio_log_count += 1
        try:
            raw = np.array(audio_data.data)  # flat array of interleaved samples
        except Exception as e:
            logger.error("Audio data conversion failed: %s", e)
            return

        if self._audio_log_count <= 3:
            logger.info(
                "Audio callback #%d: %d samples, %d channels, dtype=%s, "
                "raw min=%s, raw max=%s, raw absmax=%s",
                self._audio_log_count, len(raw), num_channels, raw.dtype,
                raw.min(), raw.max(), np.abs(raw).max(),
            )

        # Normalize integer audio to float [-1.0, 1.0]
        # Aria Gen 2 sends int64 with values that are multiples of 2^16.
        # Use 2^25 to avoid clipping while keeping speech audible.
        if np.issubdtype(raw.dtype, np.integer):
            raw = raw.astype(np.float64) / (1 << 25)

        # De-interleave and pick first channel (averaging all 8 causes
        # phase cancellation that destroys speech intelligibility)
        if num_channels > 1 and len(raw) >= num_channels:
            raw = raw[: len(raw) - len(raw) % num_channels]
            multichannel = raw.reshape(-1, num_channels)
            mono = multichannel[:, 0]  # use first mic only
        else:
            mono = raw.flatten()

        # Audio is already ~16kHz per channel (80320 samples / 5s = 16064 Hz).
        # No resampling needed — matches config.AUDIO_INPUT_SAMPLE_RATE (16kHz).

        # Convert to Int16 PCM
        mono = np.clip(mono, -1.0, 1.0)
        pcm_int16 = (mono * 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        rms = np.sqrt(np.mean(mono ** 2))

        if self._audio_log_count <= 5:
            logger.info("Audio mono: %d samples, rms=%.6f, min=%.4f, max=%.4f",
                        len(mono), rms, mono.min(), mono.max())

        # Voice activity detection (manual VAD for Gemini)
        # Skip VAD while AI is speaking (prevents AI audio from triggering self-interrupt)
        if self._vad_suppressed:
            self._vad_speech_start_count = 0
            self._vad_silence_count = 0
            pass
        elif self._audio_log_count <= self._vad_warmup:
            pass
        elif not self._vad_active:
            if rms > self._vad_speech_threshold:
                self._vad_speech_start_count += 1
                if self._vad_speech_start_count >= self._vad_speech_chunks_needed:
                    self._vad_active = True
                    self._vad_silence_count = 0
                    self._vad_speech_start_count = 0
                    logger.info("VAD: speech START (rms=%.4f)", rms)
                    if self.on_activity_start:
                        self.on_activity_start()
            else:
                self._vad_speech_start_count = 0
        else:
            if rms < self._vad_silence_threshold:
                self._vad_silence_count += 1
                if self._vad_silence_count >= self._vad_silence_chunks_needed:
                    self._vad_active = False
                    logger.info("VAD: speech END (silence for %d chunks)", self._vad_silence_count)
                    if self.on_activity_end:
                        self.on_activity_end()
            else:
                self._vad_silence_count = 0

        # Accumulate into ~100ms chunks
        with self._audio_lock:
            self._audio_buffer.extend(pcm_bytes)
            if len(self._audio_buffer) >= self._min_audio_bytes:
                chunk = bytes(self._audio_buffer)
                self._audio_buffer = bytearray()
                if self._audio_log_count <= 10:
                    logger.info("Audio chunk ready: %d bytes, callback=%s",
                                len(chunk), self.on_audio_chunk is not None)
                if self.on_audio_chunk:
                    self.on_audio_chunk(chunk)


def _numpy_to_jpeg(img_array: np.ndarray, quality: int = 50, max_dim: int = 0) -> bytes | None:
    """Convert a numpy image array to JPEG bytes, optionally resizing."""
    try:
        if img_array.ndim == 2:
            img = Image.fromarray(img_array, mode="L")
        else:
            img = Image.fromarray(img_array)
        if max_dim > 0 and max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except Exception as e:
        logger.error("JPEG encode error: %s", e)
        return None
