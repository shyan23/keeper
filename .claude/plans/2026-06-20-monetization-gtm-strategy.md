# MedAgentic — Monetization, 10X & Go-To-Market Plan

_Date: 2026-06-20 · Owner: shyan · Lens: PM / business-model strategist_

> One-line product: **"Your family's paper medical history, turned into a
> private, searchable, trend-charted timeline that your AI assistant can read —
> without the records ever leaving your machine."**

---

## 0. The strategic insight (read this first)

You did not build "another medical chatbot." You built two things that almost
nobody else has shipped together:

1. **A personal health data layer** — OCR → structured extraction → patient
   timeline → trends → source-backed RAG, with a human approval gate.
2. **A local, no-egress MCP server** over that data.

The combination is the moat. Every cloud health app forces you to upload PHI to
their servers. MedAgentic's pitch is the inverse: **"We literally cannot read
your records. They stay on your machine, and now Claude can use them."**

That is the wedge. Privacy is not a feature here — it is the entire reason a
nervous person trusts you with their dad's cancer reports. Lead with it.

**Two products, one codebase:**

| | Standalone app | MCP server |
|---|---|---|
| Buyer | Caregiver / chronic patient | AI power-user (Claude Desktop, Cursor) |
| Job | "Stop drowning in my shoebox of reports" | "Let my AI answer from my real records" |
| Channel | App store / direct / word of mouth | MCP directories, AI early-adopter crowd |
| Role | The revenue product | The **distribution hack** (free reach) |

The MCP server is your customer-acquisition engine; the standalone app is where
the $1–5 comes from.

---

## 1. Who actually pays (target segments)

Price point of $1–5 = **consumer / prosumer**, not enterprise. Forget hospitals
and clinical sales for v1 — that's a 2-year regulatory slog. Sell to the person
holding the shoebox:

- **The sandwich-generation caregiver** (primary): managing an aging parent's
  reports across multiple doctors. High pain, high anxiety, will pay to feel in
  control. This is your beachhead.
- **The chronic patient**: diabetes, thyroid, kidney, cancer follow-up —
  recurring labs they need to track over years. Built-in retention.
- **The mover**: switched doctors / insurers / countries, needs to carry their
  history. Expats especially (records in multiple languages/systems).
- **The AI tinkerer** (MCP funnel): wants their personal data in Claude. Low
  willingness to pay directly, high willingness to evangelize.

All four share one trait: **they have a literal pile of paper and no system.**
That's the wound.

---

## 2. The 10X thesis — what turns this from a tool into a product people miss

A scanner + chat is a 10% improvement over a folder of PDFs. The 10X is
**proactive, personalized intelligence** that you can only deliver because you've
structured the data:

- **"Your dad's creatinine is up across his last 3 reports — chart attached.
  Worth raising at his nephrology visit."** (trend alerts, unprompted)
- **Pre-appointment brief**: one tap → a PDF of "here's what changed since last
  visit + 3 questions to ask." (You already have PDF generation — point it here.)
- **Medication sanity flags**: same drug under two brand names, overlapping
  prescriptions, gaps. (You have the medication entities already.)
- **Refill / re-test reminders** driven off extracted dates.
- **Multi-member rollup**: "the whole family's health, one screen."

These are not new infrastructure — they are **new outputs from data you already
extract.** That's why this is cheap to build and hard to copy: a competitor
would have to rebuild your whole extraction pipeline first.

Personalization is your retention moat. A scanner is used once and forgotten;
"your health, watched for you" is opened every month. **Recurring value → you've
earned recurring revenue.**

---

## 3. Monetization strategies (3 candidates, framework-scored)

COGS reality check: your real cost is **cloud LLM extraction + OCR per
document.** The core can run free on local Ollama/Tesseract (you built that
fallback). So: **don't charge for the software — charge for convenience, cloud
processing, and intelligence.** The free local mode is the funnel, not lost
revenue.

### Strategy A — Freemium consumer subscription ⭐ (recommended primary)

- **How it works**: Free tier = 1 family member, capped docs/month, local
  processing only (BYO Ollama). Paid **"Family" tier $3–5/mo** = unlimited
  members, managed cloud extraction (no GPU needed), trend alerts, PDF briefs,
  optional encrypted sync, support.
- **Audience fit**: Caregivers want it to "just work" without running a model.
  $5/mo is below the decision threshold — cheaper than one parking fee at the
  hospital. Recurring intelligence (§2) justifies recurring charge.
- **Unit economics**: COGS is metered cloud extraction; cap heavy users with a
  fair-use credit ceiling (see C). CAC near-zero via the MCP funnel + word of
  mouth. LTV strong because chronic/caregiver use is multi-year. Target ≥80%
  gross margin once extraction is cached (you already cache OCR/extraction).
- **Risks**: free-local mode cannibalizes paid → mitigate by making cloud +
  alerts + sync the things people actually want; classic 1–5% freemium
  conversion → mitigate with the proactive hooks that only paid gets.
- **Validation**: ship a landing page with the $5 "Family" plan and a "start
  free" button; measure email capture → free signup → paid conversion.

### Strategy B — Usage credits / pay-per-document (recommended secondary, hybrid)

- **How it works**: $1 = a pack of processing credits; 1 credit ≈ 1 document
  through cloud OCR+extraction. Sits *under* the subscription as the fair-use
  meter and as the entry point for people who refuse subscriptions.
- **Audience fit**: aligns price directly to value ("I dumped 40 reports, I pay
  for 40"); great for the one-time "digitize my shoebox" burst.
- **Risks**: revenue is lumpy; users batch then churn → that's why it's a
  *companion* to A, not the main model.
- **Validation**: instrument cost-per-document now; pilot a $1 credit pack with
  10 beta users, watch repeat purchase.

### Strategy C — One-time "Pro unlock" $5 (fallback, app-store fit)

- **How it works**: single $5 purchase unlocks unlimited local processing +
  PDF + trends, forever. No cloud, no recurring.
- **Audience fit**: privacy maximalists who want zero ongoing relationship; easy
  to sell, easy to trust.
- **Risks**: no recurring revenue, no funding for cloud COGS → keep it strictly
  local-only so it has no ongoing cost to you.
- **Validation**: list as a paid local app, measure conversion vs. the freemium
  landing page.

**Verdict: ship A as the headline, with B as the meter underneath, and offer C
as a one-time "local forever" option for the privacy crowd.** This hybrid is
normal and de-risks all three.

---

## 4. The MCP play (your unfair distribution advantage)

The MCP server is read-only and local today (`app/mcp/server.py`:
`list_patients`, `get_records`, `search_records`). That is exactly right for
trust. Use it as the top of the funnel:

- **Package it as "MedAgentic for Claude Desktop"** — a one-command install that
  puts your real medical history into your AI assistant, locally.
- **List it in MCP directories / awesome-mcp lists.** The AI crowd is hungry for
  *useful, personal, private* MCP servers; "talk to your own medical records" is
  a standout demo that gets shared.
- **Free MCP, paid app**: the MCP server is free and open — it drives installs.
  But the *data* it serves is only as rich as what the app extracted, and the
  good extraction (cloud, multi-member, trends) is the paid app. The MCP server
  is the taste; the app is the meal.
- **Future paid MCP tier**: a hosted MCP relay so you can query your home vault
  from your phone/work laptop — that's a clean $5/mo add-on later. Keep the
  default purely local.

Do **not** add an MCP ingest/write tool until the HITL safety gate has a
headless contract (the code comment already flags this — respect it; it's also
good positioning: "we never silently write to your health record").

---

## 5. Positioning & messaging

- **Headline**: "The private home for your family's medical records — that your
  AI can actually read."
- **Privacy line (repeat everywhere)**: "Your records never leave your machine.
  We can't see them. Neither can anyone else."
- **Against cloud health apps**: they upload your PHI; you don't.
- **Against a folder of PDFs**: it can't tell you your dad's kidney numbers are
  trending the wrong way. MedAgentic can.
- **Proof you already have**: HITL approval, dedup, source-backed citations,
  provider fallback — these are your "we're serious and safe" credibility props.
  Put the screenshots from `README_STUFF/` on the landing page.

---

## 6. 90-day validation roadmap (cheapest test first)

1. **Week 1–2 — Smoke test demand.** Landing page: the privacy headline, the
   `README_STUFF` screenshots, two CTAs ("Start free" + "$5/mo Family"). Drive
   traffic from caregiver subreddits & an MCP-directory listing. **Metric:
   email→signup intent.** Decision: >X% signup intent → continue.
2. **Week 3–4 — Instrument COGS.** Log real cost-per-document for cloud
   extraction. This sets the credit price (B) and the fair-use cap (A). Without
   this number you're pricing blind.
3. **Week 4–6 — Ship MCP package + directory listing.** Free funnel live.
   **Metric: installs + retention of MCP users.**
4. **Week 6–10 — Closed beta of paid Family tier** with 10–20 caregivers. Turn
   on **one** proactive hook (trend alert) — the cheapest 10X feature — and
   measure whether it drives weekly opens. **Metric: free→paid conversion +
   D30 retention.** Decision: retention strong → public launch.
5. **Week 10–12 — Public launch** on the segment that converted best.

**Kill criteria**: if the privacy/caregiver angle doesn't convert and only AI
tinkerers (who won't pay) show up, pivot the standalone app toward a paid hosted
MCP product for that crowd instead.

---

## 7. What NOT to do (scope discipline)

- No clinical / hospital B2B sales in v1 — regulatory cost kills a $5 product.
- No "AI medical advice" claims — you surface *your own* records with citations;
  you don't diagnose. Keep the HITL gate and the source links front and center.
- No new monetization infra before §6.2 (cost-per-document) is known.
- Don't break the local-first promise to chase cloud features — it's the moat.

---

## TL;DR

Sell the **privacy + proactive-intelligence** version of a family health vault.
Standalone app = revenue (**freemium ~$5/mo Family tier + $1 usage credits**,
one-time $5 local-forever for purists). MCP server = free distribution into the
AI crowd. The 10X is turning extracted data into *unprompted* trend alerts and
pre-appointment briefs — built from data you already have, hard for anyone to
copy without rebuilding your pipeline. Validate with a landing page and a
10-caregiver beta before writing more code.
