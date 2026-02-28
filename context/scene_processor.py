"""Adaptive frame capture using perceptual hash for scene change detection."""
import io
import logging
import os
import queue
import threading
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

        # Worker thread for pHash + disk I/O (avoids blocking SDK callback thread)
        self._work_queue = queue.Queue(maxsize=2)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # Callback
        self.on_scene_change = None

    @property
    def keyframe_count(self) -> int:
        return self._keyframe_count

    _SENTINEL = None  # signals worker thread to exit

    def process_frame(self, frame: np.ndarray):
        """Enqueue an RGB frame for processing on the worker thread."""
        try:
            self._work_queue.put_nowait(frame)
        except queue.Full:
            pass  # drop frame to avoid backpressure on SDK thread

    def flush(self):
        """Block until all queued frames have been processed."""
        self._work_queue.join()

    def stop(self):
        """Flush pending work and stop the worker thread."""
        self._work_queue.join()
        self._work_queue.put(self._SENTINEL)
        self._worker.join(timeout=2)

    def _worker_loop(self):
        """Background thread that runs pHash comparison and keyframe capture."""
        while True:
            frame = self._work_queue.get()
            if frame is self._SENTINEL:
                self._work_queue.task_done()
                break
            try:
                self._process_frame(frame)
            except Exception as e:
                logger.error("Scene processing error: %s", e)
            finally:
                self._work_queue.task_done()

    def _process_frame(self, frame: np.ndarray):
        """Process an RGB frame. Captures keyframe if scene changed or enough time elapsed."""
        now = time.time()

        img = Image.fromarray(frame)
        current_hash = imagehash.phash(img)

        if self._last_hash is None:
            self._capture_keyframe(frame, img, now)
            self._last_hash = current_hash
            return

        distance = self._last_hash - current_hash

        if distance > self._threshold:
            self._capture_keyframe(frame, img, now, distance=distance)
            self._last_hash = current_hash
            return

        if now - self._last_capture_time >= self._idle_interval:
            self._capture_keyframe(frame, img, now, distance=distance)
            self._last_hash = current_hash

    def _capture_keyframe(self, frame: np.ndarray, img: Image.Image, now: float, distance: int = 0):
        self._last_capture_time = now
        self._keyframe_count += 1

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
            desc = f"Scene changed (distance: {distance})" if distance > 0 else "Initial capture"
            event = ContextEvent(
                timestamp=datetime.now(),
                event_type="scene_change",
                content={
                    "keyframe_path": filepath,
                    "keyframe": filename,
                    "description": desc,
                },
            )
            self.on_scene_change(event)
