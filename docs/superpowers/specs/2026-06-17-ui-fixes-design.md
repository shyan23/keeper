# Spec — fix/UI (caveman)

Date 2026-06-17. Branch `fix/UI` → PR into `feat/agentic_chatbot`.

Medical-doc RAG chatbot. Fix 10 issues from `prompts/Fix.xml`. Decisions: detect+ask patient
picker; citations open new browser tab; no pencil; one spec, one plan, TDD on backend logic.

## Root-cause findings

- `report_date` extracted (`doc_date`) but NEVER saved. Date bug source.
- Doc name = `{id}.ext`. Meaningless. No original filename stored.
- No file-serving endpoint exist. "Docs disappear" = cannot open them.
- Storage paths relative (`./data/files`). cwd change after restart → `os.path.exists` false → 404/size dash.
- `generate_answer_node` glues raw `#chunk_id` + OCR text into answer body. Garbage citations.
- `resolve_patient_node` exact `ilike` only. Slight name change → duplicate patient profile.

## Tasks

### Backend

1. **Dates.** `create_document` gets `report_date` param. `create_document_node` parse
   `extracted.doc_date` via `parse_doc_date`, store. `list_documents_timeline` + `document_to_out`
   return `report_date` (fallback `uploaded_at`). Migration `0004` add `original_name` (text, null).
   Store uploaded filename at upload → staged → doc.

2. **Citations clean.** `generate_answer_node`: answer body only in `content`. Attach `sources`
   list collapsed per document: `{document_id, name, doc_type, date}`. SSE already forward
   `message.sources`. `search_chunks` return `document_id, doc_type, report_date, original_name`.
   No hits → refusal "not in this patient's records".

3. **File serve.** `GET /api/documents/{id}/file` → `FileResponse`, existence-checked, 404 if gone.
   Store ABSOLUTE path in `save_bytes`/`save_staging`. Missing file → surfaced, not crash.

4. **Patient picker (chat).** New node before retrieve. Patient unresolved/ambiguous →
   `interrupt({type:"patient_pick", patients:[{id,name}]})`. Resume `{patient_id}`. Sidebar-selected
   patient = default. Empty/unknown query field → tell user not available.

5. **Fuzzy naming (ingest).** `resolve_patient_node` use `difflib.SequenceMatcher` ratio.
   ratio ≥ 0.85 and not exact → candidate shown in existing confirm gate "Same person as X?".
   Never auto-create dup. Exact match (ilike) still auto-resolves single.

### Frontend (vanilla TS, existing Tailwind palette, `esc()` XSS guard kept)

6. **Citation chips.** Clickable `DocType · date`. Open `/api/documents/{id}/file` new tab. Kill
   `[REF]` + raw OCR. One chip per document.

7. **Docs/Knowledge redesign.** Docs = table: title, type, report-date, icon, Open action.
   Remove "Active Context". Add search box + type filter + sort (newest/oldest/type) + group-by-type.
   Scale 100+. Knowledge panel shows only title/date/type/open.

8. **Ingestion stepper.** Replace "OCR 1/4" line with vertical stepper:
   Upload → OCR → Extract → Review → Index. Spinner active, check done. Driven by SSE node labels.

9. **Resizable divider.** Draggable handle dashboard ↔ chat panel. Persist width localStorage.
   Min/max clamp. Smooth. Mobile stacks (existing mobile tabs).

## Migrations

- `0004_document_original_name`: add `document.original_name` text null.
- `report_date` already live (0003).

## Out of scope

- MemorySaver still in-memory → chat threads reset on restart. Acceptable; documents/records persist (Supabase).
- No new RAG technique. No auth change.

## Testing

- Backend TDD: dates persist, citation source shape, fuzzy match candidates, file endpoint 200/404,
  patient-pick interrupt. Frontend: `tsc` clean + manual.
