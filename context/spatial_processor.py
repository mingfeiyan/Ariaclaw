"""Filters continuous spatial and biometric signals into discrete context events."""
import collections
import concurrent.futures
import logging
import math
from datetime import datetime

import numpy as np

import config

logger = logging.getLogger(__name__)

try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    logger.warning("geopy not available - GPS reverse geocoding disabled")


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

        # GPS state
        self._last_gps = None  # (lat, lon)
        self._last_geocode_result = None  # cached address string
        self._gps_move_threshold = config.GPS_MOVE_THRESHOLD
        self._geocoder = None
        if GEOPY_AVAILABLE:
            self._geocoder = Nominatim(user_agent="ariaclaw", timeout=5)
        self._geocode_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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

    def update_gps_position(self, lat: float, lon: float, alt: float, accuracy: float):
        """Process a GPS coordinate update. Emits location_change when moved > threshold."""
        if self._last_gps is not None:
            dist = self._haversine(self._last_gps[0], self._last_gps[1], lat, lon)
            if dist < self._gps_move_threshold:
                return
        self._last_gps = (lat, lon)

        # Emit once: use cached geocode result if available, otherwise raw coords.
        # Geocode runs in background and updates cache for the *next* event.
        description = self._last_geocode_result or f"{lat:.5f}, {lon:.5f}"
        self._emit("location_change", {
            "description": description,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "accuracy": accuracy,
        })

        # Update geocode cache asynchronously for future events
        if self._geocoder:
            self._geocode_executor.submit(self._reverse_geocode, lat, lon)

    def _reverse_geocode(self, lat: float, lon: float):
        """Run reverse geocoding in a background thread. Updates cache for next event."""
        try:
            location = self._geocoder.reverse(
                (lat, lon), exactly_one=True, language="en",
            )
            if location:
                self._last_geocode_result = location.address
        except Exception as e:
            logger.warning("Reverse geocoding failed: %s", e)

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Return distance in meters between two GPS coordinates."""
        R = 6_371_000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _emit(self, event_type: str, content: dict):
        if self.on_event:
            from context.memory_writer import ContextEvent
            event = ContextEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                content=content,
            )
            self.on_event(event)
