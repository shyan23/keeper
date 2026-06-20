from __future__ import annotations
from pypdf import PdfReader,PdfWriter,PdfMerger
import io
import fitz
from app.cache import get_or_set, make_key

# Cap rasterized pages so a huge scan can't stall OCR. 150 DPI keeps document
# text legible; JPEG keeps each page ~10x smaller than PNG (a 150-DPI scan is
# ~2.3 MB as PNG but ~230 KB as JPEG), staying under vision-API image limits.
_MAX_OCR_PAGES = 20
_OCR_DPI = 150
_JPEG_QUALITY = 75


# Pages are joined by a form-feed sentinel so callers can recover page
# boundaries (used to split a multi-report PDF into separate documents).
_PAGE_SEP = "\f"


def _pdf_text(data: bytes) -> str:
    
    reader = PdfReader(io.BytesIO(data))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return _PAGE_SEP.join(parts).strip()


def slice_pdf(data: bytes, pages: list[int]) -> bytes:
    """Return a new PDF containing only `pages` (0-based) from `data`. Lets each
    detected report be saved as its own file so its card opens just that report,
    not the whole multi-report bundle. Out-of-range/empty -> original bytes."""
    reader = PdfReader(io.BytesIO(data))
    n = len(reader.pages)
    keep = [i for i in pages if 0 <= i < n]
    if not keep or len(keep) == n:
        return data
    writer = PdfWriter()
    for i in keep:
        writer.add_page(reader.pages[i])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _pdf_to_images(data: bytes) -> list[bytes]:
    """Rasterize PDF pages to JPEG bytes. Vision models need real images, not PDF bytes."""
    pages: list[bytes] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc[:_MAX_OCR_PAGES]:
            pix = page.get_pixmap(dpi=_OCR_DPI)
            pages.append(pix.tobytes("jpeg", jpg_quality=_JPEG_QUALITY))
    return pages


def extract_text(data: bytes, *, mime_type: str, vision, progress=None, config=None) -> str:
    """Return document text. Text PDFs via pypdf; images and scanned PDFs via OCR.

    Scanned PDFs (no text layer) are rasterized to JPEG per page first, then OCR'd.
    `progress(msg)` is an optional callback fired per stage / per page so the UI can
    show live status instead of one long blocking step. `vision` is a VisionLLM.
    `config` is the LangGraph run config; forwarded to vision.ocr_image so Gemini
    Vision calls appear as nested spans in Langfuse.
    """
    return _extract_raw(data, mime_type=mime_type, vision=vision, progress=progress,
                        config=config).replace(_PAGE_SEP, "\n\n").strip()


def extract_pages(data: bytes, *, mime_type: str, vision, progress=None, config=None) -> list[str]:
    """Same content as extract_text, but split per page (reuses the OCR cache, so
    no re-OCR). Lets ingestion segment a multi-report PDF by page."""
    raw = _extract_raw(data, mime_type=mime_type, vision=vision, progress=progress,
                       config=config)
    return [p.strip() for p in raw.split(_PAGE_SEP) if p.strip()]


def _extract_raw(data: bytes, *, mime_type: str, vision, progress=None, config=None) -> str:
    if mime_type == "text/plain":
        return data.decode("utf-8", errors="replace")
    # OCR is the slow, blocking step; cache by content+type so a re-upload of the
    # same document is instant instead of re-running OCR page by page. Key bumped
    # to ocr2 since the stored form now uses the page separator.
    # ocr4: bumped from ocr3 to evict entries cached while Gemini was returning 404
    # (handwriting pages fell back to garbage Tesseract text and got cached).
    key = make_key("ocr4", mime_type, data)
    # Reset per-doc so a Gemini failure on ANY page keeps the whole-doc OCR out of
    # the cache (it's one entry); a later healthy run then replaces it.
    if hasattr(vision, "gemini_failed"):
        vision.gemini_failed = False
    return get_or_set(
        key, lambda: _ocr(data, mime_type, vision, progress, config),
        should_cache=lambda: not getattr(vision, "gemini_failed", False))


def _emit(progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


def _ocr(data: bytes, mime_type: str, vision, progress=None, config=None) -> str:
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
            out.append(vision.ocr_image(img, "image/jpeg", config=config))
        return _PAGE_SEP.join(out).strip()
    if mime_type.startswith("image/"):
        _emit(progress, "OCR image…")
        return vision.ocr_image(data, mime_type, config=config)
    raise ValueError(f"unsupported mime_type: {mime_type}")
