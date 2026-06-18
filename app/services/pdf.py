"""Assemble the final report PDF with pymupdf (fitz): reflowable HTML body via
Story, matplotlib chart pages, then appended original source documents with an
appendix index. Returns PDF bytes. No new heavy PDF dependency."""
from __future__ import annotations

import html as _html
import io
import os

import fitz

_A4 = fitz.paper_rect("a4")
_MARGIN = (40, 40, -40, -40)


def _esc(v) -> str:
    return _html.escape(str(v)) if v is not None else ""


def _section(title: str, inner: str) -> str:
    return f"<h2>{_esc(title)}</h2>{inner}" if inner else ""


def _report_html(data: dict) -> str:
    name = _esc(data.get("patient_name"))
    age = data.get("age")
    gender = _esc(data.get("gender"))
    tf = _esc(data.get("timeframe_label") or "All records")

    cover = (f"<div style='text-align:center'>"
             f"<h1 style='font-size:26pt'>Medical Report</h1>"
             f"<h1 style='font-size:20pt;color:#444'>{name}</h1>"
             f"<p>Timeframe: {tf}</p></div>")

    info_rows = "".join(
        f"<tr><td><b>{k}</b></td><td>{_esc(v)}</td></tr>"
        for k, v in [("Name", data.get("patient_name")),
                     ("Age", age if age is not None else "-"),
                     ("Gender", gender or "-")])
    info = _section("Patient Information", f"<table>{info_rows}</table>")

    def _named(items):
        if not items:
            return ""
        return "<ul>" + "".join(
            f"<li>{_esc(i['name'])}"
            f"{(' - ' + _esc(i.get('date'))) if i.get('date') else ''}</li>"
            for i in items) + "</ul>"

    diseases = _section("Disease Summary", _named(data.get("diseases")))
    symptoms = _section("Symptoms Summary", _named(data.get("symptoms")))

    tests = data.get("tests") or []
    if tests:
        head = ("<tr><th>Test</th><th>Value</th><th>Unit</th><th>Reference</th>"
                "<th>Date</th><th>Source</th></tr>")
        body = "".join(
            f"<tr><td>{_esc(t.get('test'))}</td><td>{_esc(t.get('value'))}</td>"
            f"<td>{_esc(t.get('unit'))}</td><td>{_esc(t.get('reference_range'))}</td>"
            f"<td>{_esc(t.get('date'))}</td><td>{_esc(t.get('doc_type'))}</td></tr>"
            for t in tests)
        tests_html = _section("Medical Test Results", f"<table>{head}{body}</table>")
    else:
        tests_html = ""

    tl = data.get("timeline") or []
    timeline = _section("Timeline of Findings", "<ul>" + "".join(
        f"<li>{_esc(d.get('report_date') or d.get('date'))} - "
        f"{_esc(d.get('original_name'))} ({_esc(d.get('type'))})</li>"
        for d in tl) + "</ul>") if tl else ""

    css = ("<style>body{font-family:sans-serif;font-size:11pt;color:#222}"
           "h2{border-bottom:1px solid #ccc;padding-bottom:3px;margin-top:18px}"
           "table{border-collapse:collapse;width:100%}"
           "td,th{border:1px solid #ddd;padding:4px;text-align:left;font-size:10pt}"
           "</style>")
    return (f"<html><head>{css}</head><body>{cover}{info}{diseases}{symptoms}"
            f"{tests_html}{timeline}</body></html>")


def _render_body(html: str) -> bytes:
    buf = io.BytesIO()
    writer = fitz.DocumentWriter(buf)
    story = fitz.Story(html=html)
    where = _A4 + _MARGIN
    more = 1
    while more:
        dev = writer.begin_page(_A4)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    return buf.getvalue()


def _appendix_index(doc: fitz.Document, attachments: list[dict]) -> None:
    page = doc.new_page(width=_A4.width, height=_A4.height)
    y = 60
    page.insert_text((40, y), "Attached Original Documents", fontsize=15)
    y += 28
    for i, att in enumerate(attachments, 1):
        line = f"A{i}. {att.get('name')}  -  {att.get('date') or 'undated'}  -  {att.get('type') or ''}"
        page.insert_text((40, y), line[:110], fontsize=10)
        y += 18
        if y > _A4.height - 50:
            page = doc.new_page(width=_A4.width, height=_A4.height)
            y = 50


def build_report(data: dict, charts: list[tuple[str, bytes]],
                 attachments: list[dict]) -> bytes:
    """data = gather() output (+ timeframe_label); charts = [(title, png_bytes)];
    attachments = [{name, date, file_path, type}]. Returns assembled PDF bytes."""
    doc = fitz.open("pdf", _render_body(_report_html(data)))

    for title, png in charts:
        page = doc.new_page(width=_A4.width, height=_A4.height)
        page.insert_text((40, 45), title, fontsize=13)
        page.insert_image(fitz.Rect(40, 70, _A4.width - 40, 420), stream=png,
                          keep_proportion=True)

    if attachments:
        _appendix_index(doc, attachments)
        for att in attachments:
            path = att.get("file_path")
            if not path or not os.path.exists(path):
                continue
            if path.lower().endswith(".pdf"):
                with fitz.open(path) as src:
                    doc.insert_pdf(src)
            else:
                page = doc.new_page(width=_A4.width, height=_A4.height)
                page.insert_image(fitz.Rect(30, 30, _A4.width - 30, _A4.height - 30),
                                  filename=path, keep_proportion=True)
    return doc.tobytes()
