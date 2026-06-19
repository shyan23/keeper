from app.services.handwriting import looks_like_handwriting
from app.agent.providers import PrescriptionVision


def test_empty_output_is_handwriting():
    # Bangla handwriting through the English model collapses to ~nothing.
    assert looks_like_handwriting("", 95.0)
    assert looks_like_handwriting("Rx", 90.0)


def test_low_confidence_is_handwriting():
    assert looks_like_handwriting("tab amoxicillin 500mg bd", 30.0)


def test_printed_report_is_not_handwriting():
    assert not looks_like_handwriting(
        "Complete Blood Count Haemoglobin 13.5 g/dL within reference range", 88.0)


class _Tess:
    def __init__(self, text, conf):
        self._r = (text, conf)
    def ocr_with_confidence(self, data, mime):
        return self._r


class _Gemini:
    def ocr_image(self, data, mime):
        return "GEMINI-TRANSCRIPT"


def test_prescription_escalates_to_gemini():
    v = PrescriptionVision(_Tess("scrawl", 20.0), _Gemini())
    assert v.ocr_image(b"x", "image/jpeg") == "GEMINI-TRANSCRIPT"


def test_printed_stays_on_tesseract():
    v = PrescriptionVision(_Tess("clean printed lab report text here", 85.0), _Gemini())
    assert v.ocr_image(b"x", "image/jpeg") == "clean printed lab report text here"


def test_no_gemini_configured_keeps_tesseract():
    v = PrescriptionVision(_Tess("scrawl", 20.0), None)
    assert v.ocr_image(b"x", "image/jpeg") == "scrawl"


def test_gemini_failure_falls_back():
    class _Boom:
        def ocr_image(self, data, mime):
            raise RuntimeError("quota")
    v = PrescriptionVision(_Tess("scrawl", 20.0), _Boom())
    assert v.ocr_image(b"x", "image/jpeg") == "scrawl"
