from app.services.extraction import extract_text


class _FakeVision:
    def ocr_image(self, data, mime):
        return "OCR: Patient Jane Doe"


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
