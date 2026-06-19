"""Run the deterministic scoring suites and produce a scorecard.

Two suites:
  - extraction: runs the PROD path — split the bundle into reports, extract each,
                union the results -> deterministic field scoring. A multi-report PDF
                in prod becomes N documents; cramming it into one structured() call
                (one date, one flat test list) understated extraction. Mirroring prod
                makes the model the only thing that moves the numbers.
                Needs a free chat model (Groq free tier / local Ollama). No database.
  - retrieval:  seed synthetic chunks (real local embedder) -> search_chunks -> recall@k.
                Opt-in (--retrieval). Refuses to touch the production DB: requires
                TEST_DATABASE_URL distinct from DATABASE_URL, exactly like the test suite.

The model GENERATES; `scorers.py` SCORES. No LLM-as-judge.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from app.eval import scorers

# Default to the public synthetic set; override with GOLDEN_SET to point at a
# private, gitignored golden file built over real (PHI) reports.
GOLDEN = Path(os.environ.get("GOLDEN_SET")
              or Path(__file__).resolve().parents[2] / "eval" / "golden_set.yaml")
RESULTS = Path(__file__).resolve().parents[2] / "eval" / "last_run.json"


# ---- model construction (chat only; no embedder probe) ----

def build_chat():
    """The free chat client used by the agent, without probing the embedder.
    Groq only — Ollama fallback is not routed here (model not pulled; a dead
    fallback just turns a Groq hiccup into noisy log spam, not a usable result)."""
    from app.config import get_settings
    from app.agent.llm import GroqChat
    from app.agent.providers import FallbackChat

    s = get_settings()
    chats: list[Any] = []
    if s.ai_provider in ("groq", "fallback") and s.groq_api_key:
        chats.append(GroqChat())
    return FallbackChat(chats)


def load_golden(path: Path = GOLDEN) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---- extraction suite ----

_LIST_FIELDS = ("diseases", "symptoms", "medications", "tests")
_SCALAR_FIELDS = ("patient_name", "patient_age", "patient_gender",
                  "doc_type", "doc_date", "doctor")


def _merge_extractions(parts: list[dict]) -> dict:
    """Union N per-report extractions into one ExtractionResult-shaped dict.
    Prod keeps reports as separate documents; the annotation grades the whole file,
    so list fields are concatenated and scalars take the first non-empty value
    (patient/age/gender repeat across a bundle; doc_date/doctor vary -> first wins)."""
    merged: dict = {f: [] for f in _LIST_FIELDS}
    for p in parts:
        for f in _LIST_FIELDS:
            merged[f].extend(p.get(f) or [])
        for f in _SCALAR_FIELDS:
            if not merged.get(f) and p.get(f):
                merged[f] = p[f]
    return merged


def segment_and_extract(case: dict, chat) -> dict:
    """The prod ingest path, minus the database: split the scan into reports
    (split_reports), extract each (_extract_one), union the results. Single-report
    cases (`text` or one-element `pages`) make no split call and behave as before."""
    from types import SimpleNamespace
    from app.services.segment import split_reports
    from app.agent.nodes.ingest import _extract_one

    pages = case.get("pages") or [case.get("text", "")]
    deps = SimpleNamespace(chat=chat, ner=None)
    parts = [_extract_one(deps, seg["text"]) for seg in split_reports(chat, pages)]
    return _merge_extractions(parts)


def run_extraction(cases: list[dict], chat) -> dict:
    per_case = []
    agg = {"scalar_correct": 0, "scalar_total": 0,
           "test_name_matched": 0, "test_value_matched": 0, "test_expected": 0,
           "entity_matched": 0, "entity_expected": 0, "errors": 0}
    for c in cases:
        cid = c.get("id", "?")
        try:
            pred = segment_and_extract(c, chat)
            m = scorers.score_extraction(pred, c["expect"])
        except Exception as exc:  # noqa: BLE001 - a model failure is a case failure, not a crash
            agg["errors"] += 1
            per_case.append({"id": cid, "error": str(exc)[:200]})
            continue
        agg["scalar_correct"] += m["scalars"]["correct"]
        agg["scalar_total"] += m["scalars"]["total"]
        agg["test_name_matched"] += m["tests"]["name_matched"]
        agg["test_value_matched"] += m["tests"]["value_matched"]
        agg["test_expected"] += m["tests"]["expected"]
        for ent in m["entities"].values():
            agg["entity_matched"] += ent["matched"]
            agg["entity_expected"] += ent["expected"]
        per_case.append({"id": cid, **m})

    return {"cases": per_case, "totals": agg, "metrics": _extraction_metrics(agg)}


def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def _extraction_metrics(a: dict) -> dict:
    return {
        "scalar_accuracy": _pct(a["scalar_correct"], a["scalar_total"]),
        "test_name_recall": _pct(a["test_name_matched"], a["test_expected"]),
        "test_value_recall": _pct(a["test_value_matched"], a["test_expected"]),
        "entity_recall": _pct(a["entity_matched"], a["entity_expected"]),
        "errors": a["errors"],
    }


# ---- retrieval suite (opt-in) ----

def _eval_session_factory():
    """Session factory bound to TEST_DATABASE_URL, with the same safety guard as
    tests/conftest.py: refuse if unset or equal to DATABASE_URL. Returns None to skip."""
    from app.config import get_settings
    s = get_settings()
    test_db = (s.test_database_url or "").strip()
    prod_db = (s.database_url or "").strip()
    if not test_db or test_db == prod_db:
        return None
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import Base
    import app.models  # noqa: F401 - register tables
    engine = create_engine(test_db)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def run_retrieval(cases: list[dict]) -> dict:
    from app.agent.providers import OllamaEmbedder, build_embedder
    from app.services.retrieval import search_chunks
    from app.config import get_settings

    sf = _eval_session_factory()
    if sf is None:
        return {"skipped": "TEST_DATABASE_URL unset or equal to DATABASE_URL"}
    try:
        embedder = build_embedder([OllamaEmbedder()])
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"no embedder available: {str(exc)[:120]}"}

    k = get_settings().rag_top_k
    per_case = []
    recall_hits = 0
    for c in cases:
        pid = _seed_case(sf, embedder, c)
        try:
            with sf() as sess:
                hits = search_chunks(sess, patient_id=pid, query=c["question"],
                                     embedder=embedder, k=k)
            ok = scorers.recall_at_k(hits, c["expect"]["retrieved_doc"])
            recall_hits += int(ok)
            per_case.append({"id": c.get("id", "?"), "recall_at_k": ok,
                             "retrieved": [h.get("original_name") for h in hits]})
        finally:
            _cleanup_patient(sf, pid)

    return {"cases": per_case,
            "metrics": {"recall_at_k": _pct(recall_hits, len(cases)), "k": k,
                        "n": len(cases)}}


def _seed_case(sf, embedder, case: dict) -> int:
    """Create a synthetic patient + documents + embedded chunks; return patient_id."""
    from datetime import date
    from app.models import Patient, Document, Chunk
    with sf() as s:
        p = Patient(name=f"__eval__ {case.get('id', 'x')}")
        s.add(p)
        s.flush()
        for d in case["documents"]:
            rd = d.get("report_date")
            doc = Document(patient_id=p.id, doc_type=d.get("doc_type"),
                           original_name=d.get("original_name"),
                           report_date=date.fromisoformat(rd) if rd else None,
                           status="ready")
            s.add(doc)
            s.flush()
            texts = d["chunks"]
            vectors = embedder.embed_documents(texts)
            for i, (t, v) in enumerate(zip(texts, vectors)):
                s.add(Chunk(document_id=doc.id, patient_id=p.id, ord=i, text=t, embedding=v))
        s.commit()
        return p.id


def _cleanup_patient(sf, patient_id: int) -> None:
    from app.models import Patient
    with sf() as s:
        p = s.get(Patient, patient_id)
        if p:
            s.delete(p)  # cascade -> documents -> chunks
            s.commit()


# ---- orchestration ----

def run(*, with_retrieval: bool = False, golden: Path = GOLDEN) -> dict:
    data = load_golden(golden)
    out: dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    ext_cases = data.get("extraction", [])
    if ext_cases:
        out["extraction"] = run_extraction(ext_cases, build_chat())

    if with_retrieval:
        out["retrieval"] = run_retrieval(data.get("retrieval", []))

    RESULTS.write_text(json.dumps(out, indent=2))
    return out


def format_scorecard(out: dict) -> str:
    lines = [f"# Eval scorecard — {out.get('ts', '')}", ""]
    ext = out.get("extraction")
    if ext:
        m = ext["metrics"]
        n = len(ext["cases"])
        lines += [f"## Extraction  ({n} cases, {m['errors']} errored)",
                  f"  scalar field accuracy : {m['scalar_accuracy']}%",
                  f"  test name recall      : {m['test_name_recall']}%",
                  f"  test value recall     : {m['test_value_recall']}%",
                  f"  entity recall         : {m['entity_recall']}%", ""]
    ret = out.get("retrieval")
    if ret:
        if ret.get("skipped"):
            lines += [f"## Retrieval  SKIPPED — {ret['skipped']}", ""]
        else:
            m = ret["metrics"]
            lines += [f"## Retrieval  ({m['n']} cases, k={m['k']})",
                      f"  recall@k : {m['recall_at_k']}%", ""]
    return "\n".join(lines)
