# Evaluation harness

A **deterministic, zero-cost** quality harness for the agent. The model under test
*generates*; code *scores*. There is **no LLM-as-judge** — every metric is an exact,
numeric, or set-overlap comparison — so a run is reproducible, free, and unbiased.

## Why this design

| Decision | Why |
|---|---|
| **Deterministic scoring (no judge model)** | No API cost, no rate limits, no self-preference bias, byte-stable numbers you can publish and diff across changes. |
| **Synthetic, de-identified golden set** | Safe to commit. Real PHI stays out of git — point the harness at a private golden file for your own corpus. |
| **Hand-rolled (~2 files)** | Transparent, no heavy eval deps, calls the *real* agent code paths (`_EXTRACT_PROMPT`, `search_chunks`). |

## What it measures

**Extraction** (`text → chat.structured(ExtractionResult) → score`) — no DB:
- `scalar_accuracy` — patient name/age/gender, doc type/date, doctor (normalized exact match; honorifics & date formats normalized)
- `test_name_recall` / `test_value_recall` — lab tests found by name, and value correct (numeric-tolerant)
- `entity_recall` — diseases / symptoms / medications recalled

**Retrieval** (`seed chunks → search_chunks → score`) — opt-in, needs DB + embedder:
- `recall@k` — did a chunk from the expected document appear in the top-k?

## Run

```bash
# Extraction only (free chat model, no database):
make eval                 # or: python -m app.eval

# Add retrieval recall@k (needs a separate test DB + Ollama embedder):
export TEST_DATABASE_URL=postgresql+psycopg://.../ammajan_test   # MUST differ from DATABASE_URL
make eval-retrieval       # or: python -m app.eval --retrieval
```

Results print as a scorecard and write to `eval/last_run.json`.

> **Safety:** the retrieval suite seeds and deletes rows. It refuses to run unless
> `TEST_DATABASE_URL` is set and **distinct** from `DATABASE_URL` — the same guard the
> test suite uses. It only ever touches synthetic `__eval__ …` patients and cleans them up.

## Extending with your own data

Edit `eval/golden_set.yaml`. Each `expect` block lists only the fields you want graded
(omit a field to ignore it). For real reports, keep PHI out of the repo: copy the file
elsewhere and run `python -c "from app.eval.harness import run; run(golden=Path('…'))"`.

## Scorer correctness

The scorers are pure functions, unit-tested in `tests/eval/test_scorers.py` (no DB, no
model). Those tests are the guarantee that the judge itself is right.

## Roadmap fit

This is **Horizon 0** of `.claude/plans/2026-06-18-feature-improvement-roadmap.md` — the
single biggest credibility lever for "best open-source": it turns "trust me" into a
published number, and it's how you prove the upcoming **constrained-decoding** and
**MedGemma / OpenMed NER** upgrades actually improved extraction (re-run, compare the
scorecard).
