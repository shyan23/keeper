from app.services.extraction import extract_text


class _FakeVision:
    def ocr_image(self, data, mime):
        return "OCR: Patient Jane Doe"


class _RecordingVision:
    """Captures the (data, mime) it is OCR'd with, to prove PDFs are rasterized."""
    def __init__(self):
        self.calls = []

    def ocr_image(self, data, mime):
        self.calls.append((data, mime))
        return "OCR page"


def test_image_routes_to_vision():
    text = extract_text(b"\x89PNG fake", mime_type="image/png", vision=_FakeVision())
    assert text == "OCR: Patient Jane Doe"


def test_plain_text_passthrough():
    text = extract_text(b"hello report", mime_type="text/plain", vision=_FakeVision())
    assert text == "hello report"


def test_text_pdf_uses_pypdf(tmp_path):
    from pypdf import PdfWriter
    import io
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    # blank PDF -> empty/whitespace extracted text -> falls back to vision
    out = extract_text(buf.getvalue(), mime_type="application/pdf", vision=_FakeVision())
    assert out == "OCR: Patient Jane Doe"


def test_scanned_pdf_is_rasterized_to_image_before_ocr():
    """A no-text-layer PDF must reach vision as JPEG image bytes, never raw PDF
    bytes (vision models reject those with 'invalid image data')."""
    from pypdf import PdfWriter
    import io
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    vision = _RecordingVision()
    extract_text(buf.getvalue(), mime_type="application/pdf", vision=vision)
    assert vision.calls, "vision was never called"
    for data, mime in vision.calls:
        assert mime == "image/jpeg"
        assert data[:3] == b"\xff\xd8\xff"  # real JPEG (SOI) marker
