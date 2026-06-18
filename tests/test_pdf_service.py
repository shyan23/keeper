import fitz   # pymupdf

from app.services import pdf


def _data():
    return {
        "patient_name": "Jane Doe", "age": 55, "gender": "F",
        "timeframe_label": "2021-2023",
        "diseases": [{"name": "Hyperlipidemia", "date": "2022-05-01"}],
        "symptoms": [],
        "tests": [{"test": "LDL", "value": "130", "unit": "mg/dL",
                   "reference_range": "0-100", "date": "2022-05-01",
                   "doc_type": "lipid profile", "source": "LDL 130"}],
        "timeline": [{"original_name": "lipid.pdf", "type": "lipid profile",
                      "report_date": "2022-05-01"}],
    }


def _tiny_pdf_bytes() -> bytes:
    d = fitz.open()
    d.new_page()
    return d.tobytes()


def test_build_report_returns_valid_pdf():
    out = pdf.build_report(_data(), charts=[], attachments=[])
    doc = fitz.open("pdf", out)
    assert doc.page_count >= 1
    text = "".join(p.get_text() for p in doc)
    assert "Jane Doe" in text
    assert "LDL" in text


def test_build_report_appends_pdf_attachment(tmp_path):
    src = tmp_path / "orig.pdf"
    src.write_bytes(_tiny_pdf_bytes())
    body_only = fitz.open("pdf", pdf.build_report(_data(), [], [])).page_count
    out = pdf.build_report(_data(), charts=[],
                           attachments=[{"name": "orig.pdf", "date": "2022-05-01",
                                         "file_path": str(src), "type": "lipid profile"}])
    doc = fitz.open("pdf", out)
    assert doc.page_count >= body_only + 2


def test_build_report_chart_adds_page():
    from app.services import charts
    chart_png = charts.render_metric_chart({
        "label": "LDL", "unit": "mg/dL", "ref_low": 0.0, "ref_high": 100.0,
        "points": [{"date": "2021-01-01", "value": 90.0},
                   {"date": "2022-01-01", "value": 130.0}]})
    body_only = fitz.open("pdf", pdf.build_report(_data(), [], [])).page_count
    out = pdf.build_report(_data(), charts=[("LDL over time", chart_png)], attachments=[])
    assert fitz.open("pdf", out).page_count >= body_only + 1
