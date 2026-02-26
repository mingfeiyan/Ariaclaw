"""Tests for PPG processor - heart rate extraction from PPG waveform."""
import numpy as np
import pytest

from context.ppg_processor import PPGProcessor


class TestPPGProcessor:
    def setup_method(self):
        self.processor = PPGProcessor(sample_rate=100)

    def test_extract_hr_from_simulated_ppg(self):
        """A 1.2 Hz sine wave should produce ~72 bpm heart rate."""
        t = np.linspace(0, 10, 1000)  # 10 seconds at 100 Hz
        ppg = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz = 72 bpm
        hr = self.processor.extract_heart_rate(ppg)
        assert hr is not None
        assert 60 <= hr <= 85  # allow some tolerance

    def test_returns_none_for_noise(self):
        """Random noise should not produce a valid HR."""
        noise = np.random.randn(1000) * 0.01
        hr = self.processor.extract_heart_rate(noise)
        assert hr is None or hr < 30 or hr > 200

    def test_returns_none_for_short_signal(self):
        """Too-short signal should return None."""
        short = np.zeros(10)
        hr = self.processor.extract_heart_rate(short)
        assert hr is None
