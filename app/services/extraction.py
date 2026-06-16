from __future__ import annotations

import io

from app.cache import get_or_set, make_key

# Cap rasterized pages so a huge scan can't stall OCR. 150 DPI keeps document
# text legible; JPEG keeps each page ~10x smaller than PNG (a 150-DPI scan is
# ~2.3 MB as PNG but ~230 KB as JPEG), staying under vision-API image limits.
_MAX_OCR_PAGES = 20
_OCR_DPI = 150
_JPEG_QUALITY = 75


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts).strip()


def _pdf_to_images(data: bytes) -> list[bytes]:
    """Rasterize PDF pages to JPEG bytes. Vision models need real images, not PDF bytes."""
    import fitz  # PyMuPDF

    pages: list[bytes] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc[:_MAX_OCR_PAGES]:
            pix = page.get_pixmap(dpi=_OCR_DPI)
            pages.append(pix.tobytes("jpeg", jpg_quality=_JPEG_QUALITY))
    return pages


def extract_text(data: bytes, *, mime_type: str, vision, progress=None) -> str:
    """Return document text. Text PDFs via pypdf; images and scanned PDFs via OCR.

    Scanned PDFs (no text layer) are rasterized to JPEG per page first, then OCR'd.
    `progress(msg)` is an optional callback fired per stage / per page so the UI can
    show live status instead of one long blocking step. `vision` is a VisionLLM.
    """
    if mime_type == "text/plain":
        return data.decode("utf-8", errors="replace")

    # OCR is the slow, blocking step; cache by content+type so a re-upload of the
    # same document is instant instead of re-running OCR page by page.
    key = make_key("ocr", mime_type, data)
    return get_or_set(key, lambda: _ocr(data, mime_type, vision, progress))


def _emit(progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


def _ocr(data: bytes, mime_type: str, vision, progress=None) -> str:
    if mime_type == "application/pdf":
        _emit(progress, "Checking for a text layer…")
        text = _pdf_text(data)
        if text:
            return text
        # scanned PDF: rasterize each page and OCR it as a real JPEG image
        _emit(progress, "Scanned PDF — rasterizing pages…")
        pages = _pdf_to_images(data)
        total = len(pages)
        out = []
        for i, img in enumerate(pages, 1):
            _emit(progress, f"OCR page {i}/{total}…")
            out.append(vision.ocr_image(img, "image/jpeg"))
        return "\n\n".join(p for p in out if p).strip()
    if mime_type.startswith("image/"):
        _emit(progress, "OCR image…")
        return vision.ocr_image(data, mime_type)
    raise ValueError(f"unsupported mime_type: {mime_type}")
