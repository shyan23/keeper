from __future__ import annotations

import io


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts).strip()


def extract_text(data: bytes, *, mime_type: str, vision) -> str:
    """Return document text. Text PDFs via pypdf; images and scanned PDFs via Groq vision.

    `vision` is a VisionLLM (injected).
    """
    if mime_type == "text/plain":
        return data.decode("utf-8", errors="replace")
    if mime_type == "application/pdf":
        text = _pdf_text(data)
        if text:
            return text
        # scanned PDF (no text layer) -> OCR first page bytes as image fallback
        return vision.ocr_image(data, "application/pdf")
    if mime_type.startswith("image/"):
        return vision.ocr_image(data, mime_type)
    raise ValueError(f"unsupported mime_type: {mime_type}")
