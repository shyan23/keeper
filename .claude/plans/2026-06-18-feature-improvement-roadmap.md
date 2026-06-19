# Roadmap — Best Open-Source, Fully-Local Agentic Medical Record System

**Date:** 2026-06-18 (rev 2)
**Author:** product planning pass (PM-skills: trio ideation + ICE; curriculum gap-audit)
**Status:** discovery / proposal — nothing committed
**Goal:** make this *the* open-source, self-hostable, **zero-cost / fully-local** agentic
medical-document-intelligence system.
**Hard constraint:** **no paid APIs.** Everything must run free on a local machine
(Groq free tier is acceptable as a free hosted path; no per-token billing).

---

## 0. Positioning — what "best open-source" means here

This is not a SaaS and not a clone of a cloud health app. The wedge is the thing
those can't offer:

> **A fully-local, zero-cost, agentic medical record system you self-host, where the
> data never leaves your machine.**

Cloud health trackers win on polish but lose on privacy, cost, and control. An
open-source project wins by being the opposite: **private by construction, free
forever, reproducible, hackable.** Every decision below is scored against that, plus
the existing north-star — *"find any fact about anyone's health in seconds, and trust
it."*

### Who it's for (in priority order)
1. **The self-hoster** — wants their family's PHI on their own disk, not a vendor's.
2. **The contributor / learner** — the agent stack is a portfolio-grade reference
   implementation of the AI-agents curriculum (see §2).
3. **The clinic-in-a-box** — a small practice in a low-connectivity setting that
   can't pay per-seat SaaS and needs offline operation.

### Outcomes we optimize
| Outcome | Signal |
|---|---|
| **Trust** | Right citations, no fabricated values, HITL catches bad extractions |
| **Privacy** | PHI never leaves the host; encrypted at rest; no telemetry phones home |
| **Zero-cost** | Runs to completion with no paid key set |
| **Recall speed** | Question → cited answer in < 10s, on a laptop |
| **Adoptability** | A stranger can `git clone` → run → ingest a report in < 15 min |

---

## 1. Strengths to protect (don't regress)

Grounded in `README.md` / `app/agent/*`:

- Corrective-RAG (HyDE → retrieve → rerank → grade → CRAG) **patient-scoped**.
- **Three HITL gates** (ingest, edit, low-confidence answer) — the strongest part.
- **Dedup-before-OCR** (SHA-256), escalating **CPU-only** Tesseract, Redis cache.
- **Graceful degradation** everywhere; **provider DI** (`ChatLLM`/`VisionLLM`/`Embedder`
  Protocols) so models are swappable without touching nodes.
- **Local-first** already: Ollama chat + embeddings, Tesseract OCR, runs offline.

---

## 2. Coverage audit vs the AI-Agents curriculum (`docs/ai-agents*`)

The roadmap.sh "AI Agents" map and the 12-stage syllabus are the industry competency
standard. Grading the repo against them shows exactly where "advanced agentic" is real
vs missing. **The gaps are the build list.**

| # | Stage | Status | Evidence / gap |
|---|---|---|---|
| 1 | Python + Async / FastAPI | ✅ **Done** | FastAPI + SSE, async streaming |
| 2 | LLM fundamentals (routing, tokens, latency, failure modes) | 🟡 Partial | Fallback chain exists; **no token/latency/cost telemetry**, no per-node model routing |
| 3 | Tool calling + **structured outputs** | 🟡 Partial | Groq `json_object` + manual parse (truncated arrays once); **not constrained decoding**; no dynamic tool discovery |
| 4 | Memory + state | 🟡 Partial | RAG vector recall ✅; but `MemorySaver` is **in-process**, no durable/cross-session long-term memory |
| 5 | Single-agent workflows (ReAct, plan-execute, self-reflection) | 🟡 Partial | Graph is plan-execute; CRAG = self-reflection; no explicit ReAct/iteration caps surfaced |
| 6 | Multi-agent orchestration (supervisor, handoffs) | ❌ **Missing** | One monolithic graph; no subagents/supervisor |
| 7 | Human-in-the-loop | ✅ **Done** | 3 interrupt gates, resume logic, edit verification |
| 8 | **Evaluation + QA** (eval harness, LLM-judge, regression) | ❌ **Missing** | Tests cover plumbing, **not answer quality**; no golden set |
| 9 | **Observability + tracing** (traces, cost, latency, alerting) | ❌ **Missing** | Per-node logs only; no trace timeline, no cost view |
| 10 | **Security + guardrails** (prompt-injection, PII redaction, sandbox, compliance) | ❌ **Missing** | No auth; **PHI in plaintext** on disk + pooled DB; no redaction; raw OCR logged |
| 11 | Production deployment (scale, CI/CD, canary) | 🟡 Partial | 120 tests ✅; single-worker only, no scale story |
| 12 | Open-source + portfolio (ship, docs, demos) | 🟡 Partial | README excellent; **no LICENSE, no demo, no published eval numbers, no sample dataset** |

**Read of the matrix:** HITL (7) and foundations (1) are best-in-class. The credibility
gaps for "advanced agentic + best OSS" are **3, 8, 9, 10** — and all four have **free,
local** implementations. That's the opportunity.

---

## 3. Idea generation — three lenses (all $0 / local)

### 3a. PM lens (value, trust, OSS adoption)
1. **One-command demo + sample dataset** — synthetic de-identified reports + `make demo`
   so a stranger sees value in minutes (Stage 12).
2. **Local RAG eval harness with published numbers** — faithfulness + citation scores in
   the README; turns "trust me" into a metric (Stage 8).
3. **Privacy-by-construction story** — at-rest encryption + no-egress guarantee +
   PII-redaction, documented as a headline feature (Stage 10).
4. **Trend / timeline view** per test — the core "tracker" capability still missing.
5. **MCP server wrapper** — expose ingest/query/records as MCP tools so *any* agent
   client (Claude Desktop, Continue, etc.) can drive it. Big "advanced agentic" signal,
   pure local.

### 3b. Designer lens (UX, capture, delight)
1. **Mobile-first capture PWA** — camera → confirm card → done.
2. **Source-span highlighting** in the HITL card (show which OCR words justified each value).
3. **Conversational follow-up memory** — "and her cholesterol?" keeps patient context.
4. **Capture-time quality coach** — detect blur/crop before OCR, ask for a retake.
5. **At-a-glance family dashboard** — open meds, last visit, recent abnormal results.

### 3c. Engineer lens (free tech leverage, the curriculum gaps)
1. **Constrained/structured decoding** — Ollama `format=<json-schema>` / Outlines / GBNF
   grammars → *guaranteed* valid extraction, kills the manual-parse hack (Stage 3). **$0.**
2. **Per-node free-model routing** — small local model for routing/rerank, the free Groq
   70B (or a bigger local model) for extraction/answer. One change to `build_deps`,
   exploits DI already in place (Stage 2). **$0.**
3. **Self-hosted tracing + cost/latency** — Langfuse OSS or OpenInference + Phoenix,
   local only (Stage 9). **$0.**
4. **Local eval harness** — Ragas / DeepEval with a *local* judge model; golden Q→fact
   set; CI gate (Stage 8). **$0.**
5. **Guardrails** — Microsoft Presidio for PII redaction + a prompt-injection filter on
   OCR'd text before it hits the LLM (Stage 10). **$0.**
6. **Durable checkpointer + at-rest encryption** — Postgres-backed saver replaces
   `MemorySaver`; encrypt `STORAGE_DIR` (Stage 4 + 10).
7. **Hybrid retrieval** — Postgres FTS/BM25 + pgvector fusion for better recall (Stage 5).

---

## 3d. Medical-specialized free model stack (NEW)

General LLMs are mediocre at medical NER and can't read imaging. **Domain models beat
them and most are truly OSS** — and all run free/local. Concrete role → model:

| Agent role | Today | Upgrade (free/local) | License | Why |
|---|---|---|---|---|
| OCR (printed lab text) | Tesseract (CPU) | **keep** Tesseract | Apache-2.0 | fast, deterministic, ~0 RAM |
| Imaging / radiology understanding | — (none) | **MedGemma 4B** multimodal | Gemma / HAI-DEF terms* | reads X-ray / USG / CT, narrative findings |
| Entity extraction (disease/symptom/med/test) | one LLM `structured()` call | **OpenMed biomedical NER** (380+ models) | **Apache-2.0** | higher precision; returns char spans → exact `source_span` |
| Value/unit structuring, routing, RAG answer | Groq-free-70B / Ollama | keep (free) | — | LLM for reasoning, not entity spotting |

> *MedGemma license caveat:* open weights, free to run, **but** the Gemma /
> Health-AI-Developer-Foundations terms are field-restricted — **not OSI/Apache**. For a
> purist-OSS build, gate MedGemma behind a config flag and keep **Tesseract-only as the
> default OSS path**; document it in LICENSE/NOTICE. OpenMed is clean Apache-2.0, so the
> NER upgrade has no licensing asterisk.

**Slots into the existing architecture with no rewrite:**
- **MedGemma** → the `VisionLLM` Protocol + `FallbackVision` already exist. Add a
  `MedGemmaVision` provider; route imaging docs → MedGemma, printed text → Tesseract;
  MedGemma unavailable → Tesseract. This finally fills the "no vision-model escalation"
  gap the README calls out.
- **OpenMed NER** → make entity extraction a **hybrid**: a deterministic NER pre-pass
  tags diseases/symptoms/meds/tests **with character offsets**, then the LLM only
  structures values/units/dates. Raises precision *and* hands back the exact `source_span`
  the schema already wants. NER is CPU-friendly and near-deterministic.
- **The eval harness (§Horizon 0, already built) proves both:** `entity_recall` and
  `test_value_recall` are exactly the metrics MedGemma+OpenMed move. Re-run `make eval`,
  diff the scorecard — that's how the upgrade earns its place instead of being assumed.

## 4. Prioritization (ICE, $0 lens)

Impact = trust × OSS-credibility (1–10); Confidence 1–10; Ease 1–10 (higher = easier).
Score = I×C×E. Reach ≈ constant (personal/self-host), so omitted.

| # | Idea (curriculum stage) | I | C | E | ICE | Tier |
|---|---|---|---|---|---|---|
| 1 | Constrained/structured decoding (S3) | 9 | 9 | 8 | **648** | Now |
| 2 | Per-node free-model routing (S2) | 8 | 9 | 9 | **648** | Now |
| 3 | Local RAG eval harness + published numbers (S8) | 9 | 8 | 7 | **504** | Now |
| 4 | PII redaction + injection guard (S10) | 9 | 8 | 6 | **432** | Now |
| 5 | One-command demo + synthetic dataset (S12) | 8 | 9 | 6 | **432** | Now |
| 6 | At-rest encryption + LICENSE + no-egress doc (S10/12) | 9 | 9 | 5 | **405** | Next |
| 7 | Self-hosted tracing/cost (Langfuse OSS) (S9) | 7 | 8 | 6 | **336** | Next |
| 8 | Durable checkpointer (Postgres saver) (S4) | 8 | 8 | 7 | **448** | Next |
| 9 | MCP server wrapper (S6-ish, agentic) | 8 | 7 | 6 | **336** | Next |
| 10 | Trend / timeline view | 9 | 8 | 7 | **504** | Next |
| 11 | Hybrid retrieval (BM25 + vector) (S5) | 7 | 7 | 6 | **294** | Later |
| 12 | Encrypted backup + restore | 9 | 9 | 7 | **567** | Next |
| 13 | Multi-agent supervisor split (S6) | 6 | 6 | 4 | **144** | Later |
| 14 | Mobile capture PWA | 9 | 7 | 5 | **315** | Later |
| 15 | OpenMed NER hybrid entity extraction (S3) | 9 | 8 | 6 | **432** | Now |
| 16 | MedGemma 4B vision for imaging studies (S3) | 8 | 7 | 5 | **280** | Next |

> Scores are deliberate estimates to force ranking, not measurements.

### The cheapest 10X under the $0 constraint

Paid-Claude is off the table, so the leverage moves to **making the free models reliable
and provable**, not smarter-by-spend:

- **#1 Constrained decoding + #2 free-model routing** is the keystone pair. The agent's
  quality ceiling is one shared `deps.chat.complete()`. Today it's a single mid model
  with a fragile JSON hack. Routing the *right free model per node* (small for routing,
  free-70B/bigger-local for extraction+answer) **and** forcing schema-valid output via
  constrained decoding lifts routing, extraction, segmentation, HyDE, rerank, grade, and
  answer **simultaneously** — for ~a day of work, zero dollars, using the DI that already
  exists. This replaces the old "pay for Claude" plan with a free equivalent that also
  fixes the truncation bug class.
- **#3 eval harness** is what turns the above from a vibe into a number you publish — the
  single biggest credibility lever for "best open-source."

Together: better answers (1+2) you can *prove* (3), at no cost. That's the 10X.

---

## 5. Outcome-focused roadmap

### Horizon 0 — Free quality + proof (the keystone)
> Outcome: *answers get sharply better and I can prove it, paying nothing.*

- **Constrained/structured decoding** for all extraction (Ollama `format` schema / Outlines
  / llama.cpp GBNF). Delete the manual `json_object` parse path.
- **Per-node free-model routing** in `build_deps` — cheap local model for routing/rerank,
  strongest *free* model for extraction/answer. No node changes (DI).
- **Local eval harness** — golden set + deterministic (code-only) scoring; extraction
  field accuracy + retrieval recall@k; CI gate; **publish the numbers in README.**
  *(Built — `app/eval/`, `eval/golden_set.yaml`, `make eval`.)*
- **OpenMed NER hybrid extraction** — Apache-2.0 biomedical NER pre-pass for
  diseases/symptoms/meds/tests (with char offsets → exact `source_span`); LLM only
  structures values/units. Lifts `entity_recall`; the harness measures the gain.

### Horizon 1 — Privacy & trust as headline features
> Outcome: *PHI is safe by construction, and that's a documented selling point.*

- **PII redaction (Presidio) + prompt-injection guard** on OCR text before the LLM.
- **At-rest encryption** of `STORAGE_DIR`; stop logging raw OCR at info; no-egress audit.
- **Encrypted backup + tested restore** (`pg_dump` + files, age/gpg) — durability runbook.
- **LICENSE + SECURITY.md + threat model** — table-stakes for a serious OSS health tool.

### Horizon 2 — Durability, insight, observability
> Outcome: *nothing is lost on restart; I see trends; I can debug a run.*

- **Durable checkpointer** (Postgres/Redis saver) replaces in-process `MemorySaver`.
- **Trend / timeline view** per test; **typed numeric results** to back it.
- **Self-hosted tracing** (Langfuse OSS / Phoenix) — per-node latency/tokens/model, local.

### Horizon 3 — Advanced agentic + adoption
> Outcome: *interoperable, capture-anywhere, easy to adopt.*

- **MCP server wrapper** — ingest/query/records as MCP tools for any agent client.
- **One-command demo + synthetic de-identified dataset** + recorded demo (Stage 12).
- **MedGemma 4B vision** — `MedGemmaVision` provider in the `VisionLLM` chain for
  radiology/imaging understanding (X-ray/USG/CT narrative); Tesseract stays the default
  OSS fallback (license caveat in §3d).
- **Hybrid retrieval** (BM25 + vector); **mobile capture PWA**.
- (Optional) **Multi-agent supervisor** split — only if a real workflow needs it; don't
  add orchestration for its own sake.

---

## 6. Cross-cutting (free) technical improvements
- **Idempotent, resumable ingest** — `persist_reports` safe to re-run after a mid-write crash.
- **Boot-time config validation** — fail fast if `DATABASE_URL` == prod in tests, Tesseract
  missing, or Ollama models not pulled. Turn README footguns into startup assertions.
- **Re-embed migration path** — document how to switch the (pinned) embedding model without
  corrupting the shared 768-dim space.
- **No-telemetry guarantee** — assert nothing phones home; part of the privacy story.

---

## 7. Explicitly deferred
- **Paid LLM APIs** — excluded by constraint. The free-model routing + constrained decoding
  path replaces it; revisit only if the user lifts the no-cost rule.
- **Cloud hosting / horizontal scale** — durable checkpointer *enables* it; don't pay the
  complexity tax for a self-host tool.
- **Multi-user accounts / doctor-sharing** — large auth/privacy surface; defer until sharing
  is a real, requested need.
- **Production k8s / vLLM / canary (Stage 11 deep end)** — overkill for self-host; a clean
  single-container compose + the durable checkpointer is enough.

---

## 8. Top assumptions to validate first
1. **A free model + constrained decoding closes most of the quality gap** — A/B the current
   path vs schema-constrained extraction on 20 real reports; measure field accuracy.
2. **The local judge agrees with you** — hand-grade 10 answers, check the eval harness
   correlates before trusting it as a CI gate.
3. **Redaction doesn't break extraction** — verify Presidio redaction of names/IDs still
   leaves enough context for correct test-result extraction (redact at storage/log layer,
   not before extraction, if it conflicts).
4. **Self-hosters actually want MCP** — cheap to gauge: does driving it from an MCP client
   feel better than the dashboard for capture/query?
5. **Restart-loss is real pain** — has an in-flight ingest ever been lost to a restart? If
   never, the durable checkpointer drops in urgency.

---

## 9. Suggested first move
Ship **Horizon 0** as one slice: **constrained decoding + per-node free-model routing +
eval harness**, behind the existing TDD workflow. It's the smallest change that makes the
free models reliably *good* and lets you put real quality numbers in the README — the
single highest-leverage step toward "best open-source," at zero cost. Privacy hardening
(Horizon 1) is the natural follow-on that makes those good answers *safe to trust*.

> Next step if approved: turn Horizon 0 into a spec + plan (`.claude/specs/` +
> `.claude/plans/`) following the existing TDD workflow.
