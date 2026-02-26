"""PPG signal processing: extract heart rate from raw photoplethysmography data."""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

logger = logging.getLogger(__name__)


class PPGProcessor:
    """Extracts heart rate from raw PPG waveform using bandpass filter + peak detection."""

    def __init__(self, sample_rate: int = 100):
        self._sample_rate = sample_rate

    def extract_heart_rate(self, ppg_signal: np.ndarray) -> int | None:
        """Extract heart rate (bpm) from raw PPG signal. Returns int bpm or None."""
        if len(ppg_signal) < self._sample_rate * 3:
            return None

        try:
            nyq = self._sample_rate / 2
            low = 0.7 / nyq
            high = 3.5 / nyq

            if high >= 1.0:
                high = 0.99
            if low <= 0:
                low = 0.01

            b, a = butter(3, [low, high], btype="band")
            filtered = filtfilt(b, a, ppg_signal)

            min_distance = int(self._sample_rate * 0.3)
            peaks, properties = find_peaks(filtered, distance=min_distance, prominence=0.1)

            if len(peaks) < 2:
                return None

            intervals = np.diff(peaks) / self._sample_rate
            mean_interval = np.mean(intervals)

            if mean_interval <= 0:
                return None

            bpm = int(round(60.0 / mean_interval))

            if bpm < 30 or bpm > 200:
                return None

            return bpm

        except Exception as e:
            logger.debug("PPG processing failed: %s", e)
            return None
