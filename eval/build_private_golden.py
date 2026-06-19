"""Build a PRIVATE golden set from hand-annotated real reports in Med_documents/.

Pairs each PDF's real per-page OCR (what the extractor actually sees in prod) with
an `expect` block normalized from its hand annotation JSON. Pages are stored as a
list so the harness can run the prod split-into-reports path before extracting; a
bundle of 3 reports is graded the way prod handles it, not crammed into one call.
The annotations are
heterogeneous (patient_info vs patient_name; results as dict vs list; parameter vs
test_name), so a generic recursive walk pulls the gradeable fields out of any shape.

Output contains REAL PHI -> written to eval/golden_set.private.yaml, which is
gitignored. Run: `python eval/build_private_golden.py`, then
`GOLDEN_SET=eval/golden_set.private.yaml make eval`.

Only scalars, lab tests (list-form, numeric-gradeable) and medications are emitted.
Narrative findings (xray impressions) and free-text symptoms are skipped: the scorer
matches entity/test names exactly (normalized), so verbose phrases can't be graded fairly.
"""
from __future__ import annotations

import glob
import json
import os
import re

import yaml

from app.services.extraction import extract_pages
from app.agent.providers import TesseractVision

MED_DIR = "Med_documents"
OUT = "eval/golden_set.private.yaml"

_FORMS = re.compile(r"^\s*(tab|cap|syp|inj|tablet|capsule|syrup|injection)\.?\s+", re.I)
_DOSE = re.compile(r"\s+\d+(\.\d+)?\s*(mg|ml|mcg|gm|g|iu|%)\b.*$", re.I)


def clean_med(s: str) -> str:
    """'Tab. Rostreil 135 mg' -> 'Rostreil' (brand name the model is likely to emit)."""
    s = _FORMS.sub("", str(s)).strip()
    s = _DOSE.sub("", s).strip()
    return s


def collect(node, acc: dict) -> None:
    """Recursively pull tests / medications out of any annotation shape."""
    if isinstance(node, dict):
        name = node.get("test_name") or node.get("parameter")
        if name and node.get("result") is not None:
            acc["tests"].append({"name": str(name), "value": str(node["result"])})
        med = node.get("medication")
        if isinstance(med, str):
            acc["meds"].append(clean_med(med))
        for v in node.values():
            collect(v, acc)
    elif isinstance(node, list):
        for x in node:
            collect(x, acc)


def find_first(node, key: str):
    """First value for `key` anywhere in the tree (age/gender live at varying depths)."""
    if isinstance(node, dict):
        if node.get(key) is not None and not isinstance(node[key], (dict, list)):
            return node[key]
        for v in node.values():
            r = find_first(v, key)
            if r is not None:
                return r
    elif isinstance(node, list):
        for x in node:
            r = find_first(x, key)
            if r is not None:
                return r
    return None


def find_doctor(node):
    if isinstance(node, dict):
        doc = node.get("doctor")
        if isinstance(doc, dict) and doc.get("name"):
            return doc["name"]
        for v in node.values():
            r = find_doctor(v)
            if r:
                return r
    elif isinstance(node, list):
        for x in node:
            r = find_doctor(x)
            if r:
                return r
    return None


def patient_name(d: dict):
    pi = d.get("patient_info") or {}
    return d.get("patient_name") or pi.get("name")


def dedup_tests(tests: list[dict]) -> list[dict]:
    seen, out = set(), []
    for t in tests:
        k = t["name"].strip().lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def build_case(pdf_path: str, ann: dict, vision) -> dict | None:
    pages = extract_pages(open(pdf_path, "rb").read(),
                          mime_type="application/pdf", vision=vision)
    acc = {"tests": [], "meds": []}
    collect(ann, acc)

    expect: dict = {}
    name = patient_name(ann)
    if name:
        expect["patient_name"] = name
    age = find_first(ann, "age")
    if age is not None:
        expect["patient_age"] = age
    gender = find_first(ann, "gender")
    if gender:
        expect["patient_gender"] = gender
    doc = find_doctor(ann)
    if doc:
        expect["doctor"] = doc
    tests = dedup_tests(acc["tests"])
    if tests:
        expect["tests"] = tests
    meds = sorted({m for m in acc["meds"] if m})
    if meds:
        expect["medications"] = [{"name": m} for m in meds]

    if len(expect) < 2:  # nothing meaningful to grade
        return None
    cid = os.path.splitext(os.path.basename(pdf_path))[0].lower().replace(" ", "-")
    return {"id": cid, "pages": pages, "expect": expect}


def main() -> None:
    vision = TesseractVision()
    cases = []
    for jpath in sorted(glob.glob(f"{MED_DIR}/*.json")):
        ann = json.load(open(jpath))
        pdf = os.path.join(MED_DIR, ann.get("file_name", ""))
        if not os.path.exists(pdf):
            print(f"skip {jpath}: pdf {pdf} missing")
            continue
        case = build_case(pdf, ann, vision)
        if case:
            cases.append(case)
            print(f"{case['id']}: {len(case['expect'].get('tests', []))} tests, "
                  f"{len(case['expect'].get('medications', []))} meds")
    with open(OUT, "w") as f:
        yaml.safe_dump({"extraction": cases}, f, sort_keys=False, allow_unicode=True)
    print(f"\nwrote {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
