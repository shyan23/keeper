# PRD: Longitudinal Medical Relationship Graph & Intelligent Health Timeline

**Document version:** 1.0  
**Date:** 2026-06-21  
**Branch:** `feat/graph`

---

## 1. Summary

Patients today carry years of disconnected medical documents — lab results, prescriptions, and diagnoses that exist in isolation. This feature transforms that fragmented data into a living, relational health graph that automatically links diseases to treatments, prescriptions to ordered tests, and biomarker trends across time. The result is a single view where anyone can understand a full medical history in under 2 minutes.

---

## 2. Contacts

| Name | Role | Notes |
|------|------|-------|
| shyannafis@gmail.com | Product Owner / Engineer | Core decision-maker |
| Dr. Julian Vane (persona) | Clinical End-User | Oncology lead; validates clinical accuracy |
| Patient / Caregiver | Primary Consumer | Needs to read, not interpret |

---

## 3. Background

### Context

MedAgentic already extracts structured entities (diseases, medications, test results) from uploaded medical documents using OCR + LLM pipelines. These entities are stored in a relational database but have never been **connected to each other**. Each document is an island.

### Why now?

- The agentic chatbot (feat/agentic_chatbot) is live and can answer questions about individual records. The logical next step is enabling *cross-document* reasoning.
- The extraction pipeline (Gemini 2.5 + Tesseract + constrained decoding) now produces high-quality structured data reliable enough to build graph edges on.
- Users with multiple uploaded documents (some have 14+ documents, 5+ reports) have no way to see the big picture.

### Why this matters

A patient with diabetes doesn't think "I have 12 PDFs." They think "I've had diabetes for 3 years, I take Metformin, and my HbA1c keeps going up." That narrative is currently invisible in the product. This feature makes it visible.

---

## 4. Objective

### Goal

Let any user understand their full medical journey — across doctors, years, and document types — in a single visual session.

### Why it matters

- **For patients/caregivers:** Reduces medical anxiety. Makes it easy to answer "why was this test ordered?" and "is my condition getting better or worse?"
- **For clinicians:** Saves time in appointments. Reduces duplicate test ordering. Surfaces dangerous trends proactively.
- **For MedAgentic:** Becomes the layer above raw document storage — a medical intelligence product, not just a file viewer.

### Key Results (SMART)

| # | Metric | Target | Timeframe |
|---|--------|--------|-----------|
| KR1 | Time to understand full medical history | < 2 minutes | At launch |
| KR2 | Duplicate tests detected and surfaced | ≥ 80% recall on test data | At launch |
| KR3 | Dangerous trend alerts shown when 3+ consecutive out-of-range values exist | 100% recall | At launch |
| KR4 | Graph correctly links prescription → test within ±14 days | ≥ 75% precision | At launch |
| KR5 | Zero new npm dependencies | 0 added | At launch |

---

## 5. Market Segments

### Primary: Chronically ill patients / their caregivers

People managing long-term conditions (diabetes, thyroid, hypertension, cancer) who visit multiple doctors and accumulate years of reports.

**Jobs to be done:**
- "I need to explain my history to a new doctor in 5 minutes"
- "I need to know if my blood sugar trend is actually getting worse"
- "I don't remember why I had that MRI last year"

**Constraints:**
- Not medical professionals — language must be plain
- Often stressed, not patient with complex UIs
- Using mobile as often as desktop

### Secondary: Clinicians reviewing a patient's uploaded records

Doctors who use MedAgentic to review a patient's self-reported history before or during an appointment.

**Jobs to be done:**
- "Show me what tests were ordered alongside each prescription"
- "Are there any dangerous trends in the last 6 months?"

---

## 6. Value Propositions

### Jobs addressed

| Job | Current pain | What we give them |
|-----|-------------|-------------------|
| Understand history across years | Must read 12+ PDFs manually | One graph with all connections |
| Know if condition is worsening | Must compare individual numbers | Trend status: improving / stable / critical |
| See why a test was ordered | No link between prescription and lab | Temporal edge: Prescription → Test (±14d) |
| Catch redundant tests | No visibility into test history | Duplicate test alerts |
| Brief a new doctor | Manual summary from memory | Shareable graph view |

### Gains

- Confidence: "I understand my health"
- Time saved: Hours → 2 minutes
- Safety: Dangerous trends caught before the next appointment

### Pains avoided

- Medical confusion from disconnected documents
- Repeating unnecessary tests (cost + radiation exposure)
- Missing prescribed follow-ups

### Differentiation

No existing consumer health tool automatically infers *relationships* between documents from different doctors without requiring manual input or integration with a hospital EHR system.

---

## 7. Solution

### 7.1 UX / User Flows

#### Entry point

A new **"Graph"** tab appears in the right panel (alongside Agentic AI / Knowledge / Activity).

#### Views inside the Graph tab

```
┌─────────────────────────────────────────────┐
│  [ Graph ] [ Timeline ] [ Alerts ]          │  ← sub-tabs
├─────────────────────────────────────────────┤
│                                             │
│   ●─────────●─────────●                    │  ← SVG graph
│  Disease  Medication  Test                  │
│                                             │
│  Node colors:                               │
│    Green = normal   Yellow = watch          │
│    Red = critical   Blue = disease cluster  │
│                                             │
│  Edge label: "ordered (0.85)"              │
│  Click node → expand details card          │
└─────────────────────────────────────────────┘
```

**Graph view:** Force-directed SVG graph. Nodes grouped loosely by disease cluster. Click any node to see details (date, value, source document). Edges labeled with relationship type + confidence score.

**Timeline view:** Horizontal chronological strip. Each document becomes an event dot. Clicking an event highlights connected nodes in the graph.

**Alerts view:** Cards listing detected issues:
- 🔴 `HbA1c increasing above normal for 3 consecutive sessions`
- 🟡 `MRI repeated after only 5 days`
- 🟡 `Blood sugar out of range in last 2 results`

#### Color coding (nodes)

| Color | Meaning |
|-------|---------|
| Green | Test value within reference range |
| Yellow | Test value outside range, or confidence < 0.6 |
| Red | Critical: >20% outside range, or dangerous trend |
| Blue | Disease node |
| Purple | Medication node |
| Gray | Unknown / no value |

#### Confidence display on edges

Each edge shows: `relationship-type (confidence)` e.g. `ordered (0.85)` for a high-confidence link and `monitored_by (0.45)` for a weak inferred link.

---

### 7.2 Key Features

#### Feature 1: Graph construction from existing DB data

**What:** Builds a graph (nodes + edges) from entities already stored in `DocumentEntity`, `Disease`, `Medication`, `MedicalTest`, `TestResult`, and `Document`.

**No new tables required.** Uses:
- `DocumentEntity.entity_type ∈ {disease, medication, test_result}`
- `Document.report_date` for temporal inference
- `Document.doc_type / classification` to identify prescription vs. lab docs
- `Document.raw_ocr_text` to detect self-referrals

**Node types:**
| Type | Source model | Display |
|------|-------------|---------|
| `disease` | Disease | Disease name, first seen date |
| `medication` | Medication | Drug name, date |
| `test` | MedicalTest + TestResult | Test name, latest value, status |

---

#### Feature 2: Relationship inference engine

**Rule 1 (High confidence — co-occurrence):**  
If a disease and medication appear in the same document → `disease → treated_by → medication` (confidence: 0.85).

**Rule 2 (High confidence — co-occurrence):**  
If a disease and test appear in the same document → `disease → monitored_by → test` (confidence: 0.80).

**Rule 3 (High confidence — temporal proximity):**  
If a prescription document has a medication, and a lab document has a test result within ±14 calendar days → `medication → ordered → test` (confidence: 0.85).

**Rule 4 (Reduced confidence — extended window):**  
Same as Rule 3 but >14 days and ≤60 days → confidence: 0.45. Still surfaces as a "Weak" edge.

**Rule 5 (Self-referral bypass):**  
If the lab document's OCR text contains "self-refer", "routine", "follow-up", or "screening" → skip temporal linking (test was not prescription-driven).

**Confidence labels:**
| Score | Label |
|-------|-------|
| ≥ 0.80 | Confirmed |
| 0.60–0.79 | Likely |
| < 0.60 | Weak |

---

#### Feature 3: Smart Health Alerts

| Alert type | Trigger | Severity |
|-----------|---------|---------|
| `duplicate_test` | Same test within 30 days | Warning |
| `dangerous_trend` | 3+ consecutive sessions with test moving out of range | Critical |
| `repeated_abnormal` | Last 2 results for same test are out of reference range | Warning |

---

#### Feature 4: Biomarker trend labeling on nodes

Test nodes are annotated with latest value, unit, and a status derived from reference range:
- `normal` → green
- `warning` → yellow (outside range)
- `critical` → red (>20% outside range)

---

#### Feature 5: Disease journey view (expandable tree)

Diseases act as cluster roots. Clicking a disease node expands its connected medications and tests in a tree layout:

```
Diabetes
 ├── Metformin (treated_by, 0.85)
 │     └── HbA1c [7.2% ↑ critical] (ordered, 0.80)
 ├── Glipizide (treated_by, 0.70)
 └── Glucose [105 mg/dL ↑ warning] (monitored_by, 0.80)
```

---

### 7.3 Technology

**Backend:**
- `app/services/graph.py` — pure Python, no new dependencies
  - `build_graph(db, patient_id)` → `{nodes, edges, alerts}`
  - `detect_alerts(test_series, ...)` → list of alert dicts
- `app/api/routes_graph.py` — FastAPI router
  - `GET /api/patients/{patient_id}/graph`
- Registered in `app/api/server.py`

**Frontend:**
- `src/graph.ts` — vanilla TypeScript (no React, no D3)
  - SVG force-directed layout using simple spring physics (~150 lines)
  - Alert cards rendered as HTML
  - Timeline strip as SVG
- `src/types.ts` — new types: `GraphNode`, `GraphEdge`, `GraphAlert`, `MedicalGraph`
- `src/api.ts` — new: `getGraph(patientId: string)`
- `index.html` — new "Graph" tab button in right panel

**No new npm packages.** Pure SVG + TypeScript.

---

### 7.4 Assumptions

| # | Assumption | Risk if wrong |
|---|-----------|--------------|
| A1 | `DocumentEntity` records are populated for all uploaded documents | Graph will be sparse for unprocessed docs |
| A2 | `Document.doc_type` / `classification` reliably distinguishes prescriptions from lab reports | Temporal edge inference fails silently |
| A3 | `Document.report_date` is populated (not null) for most documents | Timeline and temporal edges can't be computed |
| A4 | Users understand confidence scores intuitively | May need a legend or tooltip |
| A5 | SVG force layout is performant for < 100 nodes | May need optimization for patients with 100+ entities |

---

## 8. Release Plan

### V1 (this branch — `feat/graph`)

**Scope:**
- Graph construction from existing DB data
- Relationship inference (co-occurrence + temporal)
- Smart alerts (duplicate, dangerous trend, repeated abnormal)
- Graph tab UI: SVG graph view + alert dashboard
- Timeline strip

**Out of scope for V1:**
- Cross-patient comparison
- AI-assisted reasoning on graph (LLM annotation of edges)
- Editing / correcting inferred relationships
- Export / share graph
- Mobile-optimized touch interaction on graph canvas
- Graph persistence to DB (recomputed fresh on each request)

**Estimated complexity:** ~400–600 lines total across 4 files (backend service, API route, TS types + API, TS graph renderer)

### V2 (future)

- LLM-generated natural language summary of disease journey
- User can confirm / reject inferred relationships (HITL for graph edges)
- "Missing test" alerts: prescription with no corresponding test in 30 days
- DB-persisted graph with invalidation on new document upload
- Recurring pattern detection across longitudinal history
- Mobile touch-friendly graph navigation

---

## Appendix: API Contract

### `GET /api/patients/{patient_id}/graph`

**Response:**
```json
{
  "nodes": [
    {
      "id": "disease-1",
      "type": "disease",
      "label": "Diabetes",
      "date": "2024-01-15",
      "status": "normal"
    },
    {
      "id": "test_result-42",
      "type": "test",
      "label": "HbA1c",
      "date": "2024-03-10",
      "value": "7.2",
      "unit": "%",
      "status": "warning"
    },
    {
      "id": "medication-7",
      "type": "medication",
      "label": "Metformin",
      "date": "2024-01-15",
      "status": "normal"
    }
  ],
  "edges": [
    {
      "from": "disease-1",
      "to": "medication-7",
      "type": "treated_by",
      "confidence": 0.85,
      "temporal": false
    },
    {
      "from": "medication-7",
      "to": "test_result-42",
      "type": "ordered",
      "confidence": 0.85,
      "temporal": true,
      "days_apart": 7
    }
  ],
  "alerts": [
    {
      "type": "repeated_abnormal",
      "severity": "warning",
      "message": "HbA1c out of normal range in last 2 results",
      "test": "HbA1c"
    }
  ]
}
```

---

*This PRD was generated from the feature spec at `prompts/feat-graphical_arrangement.xml` on 2026-06-21.*
