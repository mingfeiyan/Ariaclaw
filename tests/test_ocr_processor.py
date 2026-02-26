"""Tests for OCR processor - text detection in images."""
import numpy as np
import pytest

from context.ocr_processor import OCRProcessor


class TestOCRProcessor:
    def setup_method(self):
        self.processor = OCRProcessor()

    def test_returns_empty_for_blank_image(self):
        """Blank image should return no text."""
        blank = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = self.processor.detect_text(blank)
        assert result == "" or result is None or len(result.strip()) == 0

    def test_returns_string_type(self):
        """Result should always be a string."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = self.processor.detect_text(img)
        assert isinstance(result, str)
