"""OCR text detection using Apple Vision framework (macOS) with PIL fallback."""
import logging
import platform

import numpy as np

logger = logging.getLogger(__name__)

VISION_AVAILABLE = False
if platform.system() == "Darwin":
    try:
        import objc
        from Foundation import NSData
        from Vision import (
            VNImageRequestHandler,
            VNRecognizeTextRequest,
        )
        VISION_AVAILABLE = True
    except ImportError:
        logger.info("pyobjc Vision framework not available - OCR disabled")


class OCRProcessor:
    """Detects text in images using Apple Vision (macOS) or returns empty string."""

    def detect_text(self, frame: np.ndarray) -> str:
        """Detect text in an RGB numpy frame. Returns detected text or empty string."""
        if not VISION_AVAILABLE:
            return ""

        try:
            from PIL import Image
            import io

            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))

            handler = VNImageRequestHandler.alloc().initWithData_options_(image_data, None)
            request = VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(1)  # accurate

            success, error = handler.performRequests_error_([request], None)
            if not success:
                logger.debug("Vision OCR request failed: %s", error)
                return ""

            results = request.results()
            if not results:
                return ""

            texts = []
            for observation in results:
                candidate = observation.topCandidates_(1)
                if candidate:
                    texts.append(candidate[0].string())

            return "\n".join(texts)

        except Exception as e:
            logger.debug("OCR failed: %s", e)
            return ""
