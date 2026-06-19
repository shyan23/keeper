from __future__ import annotations

# Handwriting detector — no external API. Tesseract is tuned for printed text, so a
# handwritten prescription (Bangla or English) comes back unreliable in one of two
# tell-tale ways, and we escalate ONLY those to the paid Gemini OCR:
#
#   English handwriting -> Tesseract still emits words but at low mean confidence.
#   Bangla handwriting   -> run through the English model it decodes to almost
#                           nothing, so the output collapses to near-empty.
#
# Printed docs (Bangla or English, typed) clear the confidence benchmark and stay
# on Tesseract — no paid call.

# ponytail: two tuned knobs (conf_floor, min_chars). The physical input — phone
# scans, ink bleed, model version — drifts, so these are meant to be re-calibrated
# on real prescriptions, not treated as constants.
CONF_FLOOR = 55.0   # printed text clears Tesseract's ~70 benchmark; handwriting sits well below
MIN_CHARS = 12      # below this, Tesseract effectively gave up (Bangla-handwriting case)


def looks_like_handwriting(text: str, mean_conf: float,
                           conf_floor: float = CONF_FLOOR,
                           min_chars: int = MIN_CHARS) -> bool:
    """True if Tesseract output looks like a handwritten prescription.

    `mean_conf` is Tesseract's mean per-word confidence (0-100). Either signal
    fires: near-empty output, or confidence below the printed-text floor."""
    if len(text.strip()) < min_chars:
        return True
    return mean_conf < conf_floor


if __name__ == "__main__":
    # ponytail self-check: the smallest thing that breaks if the logic flips.
    assert looks_like_handwriting("", 95.0)                       # empty -> Bangla handwriting
    assert looks_like_handwriting("Rx amox", 30.0)               # low conf -> English handwriting
    assert not looks_like_handwriting(
        "Complete Blood Count: Haemoglobin 13.5 g/dL within range", 88.0)  # printed lab report
    assert not looks_like_handwriting("a" * 50, 70.0)            # long + at floor -> printed
    print("ok")
