"""Tests for AudioProcessor - audio buffering and speaker labeling."""
import numpy as np
import pytest

from context.audio_processor import AudioProcessor


class TestAudioProcessor:
    def setup_method(self):
        self.events = []
        self.processor = AudioProcessor(
            whisper_model="tiny",
            sample_rate=16000,
            buffer_duration_seconds=5.0,
        )
        self.processor.on_transcription = lambda event: self.events.append(event)

    def test_buffers_audio_until_threshold(self):
        """Short audio should be buffered, not immediately transcribed."""
        chunk = np.zeros(16000, dtype=np.int16).tobytes()
        self.processor.add_audio_chunk(chunk, is_contact_mic=False)
        assert self.processor.buffer_duration_seconds < 5.0

    def test_speaker_label_self_when_contact_mic(self):
        """Audio from contact mic should be labeled as 'self'."""
        label = self.processor._classify_speaker(is_contact_mic=True)
        assert label == "self"

    def test_speaker_label_other_when_no_contact_mic(self):
        """Audio without contact mic signal should be labeled as 'other'."""
        label = self.processor._classify_speaker(is_contact_mic=False)
        assert label == "other"

    def test_prosody_calm_for_silence(self):
        """Silent audio should have low energy."""
        silence = np.zeros(16000, dtype=np.float32)
        metrics = self.processor._analyze_prosody(silence)
        assert metrics["rms"] < 0.01

    def test_prosody_high_energy_for_loud(self):
        """Loud audio should have high energy."""
        loud = np.full(16000, 0.5, dtype=np.float32)
        metrics = self.processor._analyze_prosody(loud)
        assert metrics["rms"] > 0.1
