"""Filters continuous spatial and biometric signals into discrete context events."""
import collections
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

        self._last_position = None
        self._last_orientation = None
        self._hr_readings = collections.deque(maxlen=100)
        self._hr_min_readings = 10

        self.on_event = None

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
        readings = np.array(list(self._hr_readings))
        mean = np.mean(readings[:-1])
        std = np.std(readings[:-1])
        if std < 1.0:
            std = 1.0
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
