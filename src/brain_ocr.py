 # %%
"""
Basic OCR utilities for image files (PNG/JPG).

`ocr_image(path)` attempts OCR via Apple's Vision framework first
and falls back to `pytesseract` if Vision isn't available or yields no text.
Both backends are optional; you receive `None` on error or if no backend is installed.

macOS Vision (recommended):
    uv pip install pyobjc-framework-Vision

Tesseract fallback:
    brew install tesseract
    uv pip install pytesseract Pillow
"""
from pathlib import Path
from typing import Optional

# Optional imports
try:
    from Vision import VNRecognizeTextRequest, VNImageRequestHandler, VNRequestTextRecognitionLevelAccurate
    VISION_AVAILABLE = True
except Exception:
    VISION_AVAILABLE = False

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


def _ocr_with_vision(image_path: Path) -> Optional[str]:
    if not VISION_AVAILABLE:
        return None
    try:
        # Synchronous request; read results from the request afterwards
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        from Foundation import NSURL
        url = NSURL.fileURLWithPath_(str(image_path))
        handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        handler.performRequests_error_([request], None)

        texts = []
        for obs in request.results() or []:
            try:
                cands = obs.topCandidates_(1)
                if cands and len(cands) > 0:
                    # PyObjC bridges Objective-C properties as callables sometimes
                    txt = cands[0].string() if hasattr(cands[0], "string") else str(cands[0])
                    texts.append(str(txt))
            except Exception:
                continue
        return "\n".join(texts) if texts else ""
    except Exception:
        return None

def _ocr_with_tesseract(image_path: Path) -> Optional[str]:
    if not TESSERACT_AVAILABLE:
        return None
    try:
        img = Image.open(str(image_path))
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return None

def ocr_image(path: Path) -> Optional[str]:
    """
    Run OCR on a single image file and return extracted text.
    Returns None if no backend is available or on error.
    """
    p = Path(path)
    if not p.exists():
        return None

    # Try Vision first on macOS, then fall back to tesseract
    # Try Vision first; if it yields nothing, try tesseract
    text = _ocr_with_vision(p)
    if not text:
        text = _ocr_with_tesseract(p)
    return text

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/brain_ocr.py <image_path>")
        sys.exit(1)
    out = ocr_image(sys.argv[1])
    print(out or "")
