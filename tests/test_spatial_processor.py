"""Tests for SpatialProcessor - filters continuous signals into discrete events."""
import numpy as np
import pytest

from context.spatial_processor import SpatialProcessor


class TestSpatialProcessor:
    def setup_method(self):
        self.events = []
        self.processor = SpatialProcessor(
            location_threshold_meters=3.0,
            orientation_threshold_degrees=30.0,
            hr_anomaly_zscore=2.0,
        )
        self.processor.on_event = lambda e: self.events.append(e)

    def test_no_event_on_small_movement(self):
        self.processor.update_position(0, 0, 0)
        self.processor.update_position(1, 1, 0)
        location_events = [e for e in self.events if e.event_type == "location_change"]
        assert len(location_events) == 0

    def test_event_on_large_movement(self):
        self.processor.update_position(0, 0, 0)
        self.processor.update_position(5, 0, 0)
        location_events = [e for e in self.events if e.event_type == "location_change"]
        assert len(location_events) == 1

    def test_no_event_on_small_orientation_change(self):
        self.processor.update_orientation(0, 0, 0)
        self.processor.update_orientation(10, 5, 0)
        activity_events = [e for e in self.events if e.event_type == "activity_change"]
        assert len(activity_events) == 0

    def test_event_on_large_orientation_change(self):
        self.processor.update_orientation(0, 0, 0)
        self.processor.update_orientation(45, 0, 0)
        activity_events = [e for e in self.events if e.event_type == "activity_change"]
        assert len(activity_events) == 1

    def test_hr_baseline_builds(self):
        for hr in [70, 68, 72, 69, 71]:
            self.processor.update_heart_rate(hr)
        hr_events = [e for e in self.events if e.event_type == "heart_rate_spike"]
        assert len(hr_events) == 0

    def test_hr_anomaly_detected(self):
        for hr in [70, 68, 72, 69, 71, 70, 68, 72, 69, 71]:
            self.processor.update_heart_rate(hr)
        self.processor.update_heart_rate(110)
        hr_events = [e for e in self.events if e.event_type == "heart_rate_spike"]
        assert len(hr_events) == 1
        assert hr_events[0].content["heart_rate"] == 110
