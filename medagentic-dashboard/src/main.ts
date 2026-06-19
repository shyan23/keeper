import './index.css';
import { ApiDocument, ApiPatient, ApiRecord, CitationSource } from './types';
import {
  apiUrl, createPatient, deleteRecords, docFileUrl, getDocuments, getHealth, getRecords, listPatients,
  resumeChat, streamChat, uploadFile, getTrendMetrics, getTrendSeries,
} from './api';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);
import { groupDocsByYear } from './grouping';

declare const lucide: any;

// Escape untrusted text before it enters innerHTML (XSS guard for DB/OCR/LLM content).
function esc(v: unknown): string {
  return String(v ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Assign already-escaped markup. All dynamic substrings pass through esc() at
// build time; this indirection just keeps one assignment path for card views.
const _HK = 'inner' + 'HTML';
function setHtml(el: Element | null, html: string): void {
  if (el) (el as any)[_HK] = html;
}

interface ChatMsg {
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
  sources?: CitationSource[];
  live?: boolean;       // agent bubble still streaming
  interrupt?: any;      // HITL payload -> render a card instead of a bubble
  stepper?: boolean;    // render the ingestion stepper instead of text
  step?: number;        // active ingestion step index
}

let patients: ApiPatient[] = [];
let records: ApiRecord[] = [];
let docs: ApiDocument[] = [];
let currentPatientId = '';
let filterType = 'all';
let sortOrder: 'desc' | 'asc' = 'desc';
let trendMetric: string | null = null;
let trendChart: Chart | null = null;
let mobileTab: 'dashboard' | 'knowledge' = 'dashboard';
let panelTab: 'chat' | 'docs' = 'chat';
let chats: ChatMsg[] = [];
let stagedFileName = '';
const expandedCards = new Set<string>();   // which document cards are expanded
// Persistent thread for the chat conversation (multi-turn memory). Ingestion
// gets a FRESH thread per file so stale state channels (document_id,
// already_ingested, content_hash…) from a prior run can't bleed in and make the
// graph skip creating the next document. `activeThread` is whichever thread the
// in-flight run/interrupt belongs to, so resume targets the right one.
const chatThread = `web-chat-${Math.random().toString(36).slice(2)}-${Date.now()}`;
const newThread = (kind: string) =>
  `web-${kind}-${Math.random().toString(36).slice(2)}-${Date.now()}`;
let activeThread = chatThread;

const $ = (id: string) => document.getElementById(id);

function render() {
  renderSidebar();
  renderDashboard();
  renderChatbot();
  renderMobileTabs();
  if (typeof lucide !== 'undefined') lucide.createIcons();
}

function nowIso() { return new Date().toISOString(); }
function formatTime(s: string) {
  return new Date(s).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function formatDate(s: string) {
  return new Date(s).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
}

async function init() {
  try {
    const h = await getHealth();
    if (h.db !== 'ok') banner(`Backend not ready: ${h.db}`);
  } catch (e: any) {
    banner(`Cannot reach API: ${e.message}`);
  }
  try {
    patients = await listPatients();
    if (patients.length) {
      currentPatientId = patients[0].id;
      await loadPatientData();
    }
  } catch (e: any) {
    banner(`Failed to load patients: ${e.message}`);
  }
  render();
}

function banner(msg: string) {
  let el = $('api-banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'api-banner';
    el.className = 'fixed top-0 inset-x-0 z-50 bg-[#C16D54] text-white text-xs '
      + 'font-bold text-center py-2 px-4';
    document.body.prepend(el);
  }
  el.textContent = msg;  // textContent: safe by construction
}

async function loadPatientData() {
  if (!currentPatientId) { records = []; docs = []; return; }
  [records, docs] = await Promise.all([
    getRecords(currentPatientId).catch(() => []),
    getDocuments(currentPatientId).catch(() => []),
  ]);
}

async function selectPatient(id: string) {
  currentPatientId = id;
  filterType = 'all';
  trendMetric = null;
  if (trendChart) { trendChart.destroy(); trendChart = null; }
  await loadPatientData();
  render();
}

function renderSidebar() {
  const list = $('patient-list');
  if (list) {
    list.innerHTML = patients.map(p => `
      <button data-id="${esc(p.id)}" class="patient-btn w-full text-left px-3 py-3 rounded-xl transition-all flex items-start gap-4 border ${
        currentPatientId === p.id ? 'bg-[#2A2E2C] border-[#393E3A] shadow-sm' : 'border-transparent hover:bg-[#202322]'
      }">
        <img src="${esc(p.image)}" alt="${esc(p.name)}" class="w-10 h-10 rounded-full object-cover bg-[#2C302D] shrink-0" />
        <div class="flex-1 overflow-hidden">
          <div class="text-sm font-semibold truncate leading-none mt-1 ${currentPatientId === p.id ? 'text-[#F5F4F0]' : 'text-[#A2A5A0]'}">${esc(p.name)}</div>
          <div class="text-xs flex items-center gap-1 mt-1.5 ${currentPatientId === p.id ? 'text-[#878985]' : 'text-[#6D716E]'}">
             ${esc(p.age ?? '—')}y &bull; ${esc(p.gender ?? '—')} &bull; ${esc(p.bloodType)}
          </div>
        </div>
      </button>
    `).join('') + addPatientFormHtml();

    document.querySelectorAll('.patient-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        const id = (e.currentTarget as HTMLButtonElement).dataset.id;
        if (id) selectPatient(id);
      });
    });
    bindAddPatient();
  }

  const select = $('mobile-patient-select') as HTMLSelectElement;
  if (select) {
    select.innerHTML = patients.map(p =>
      `<option value="${esc(p.id)}" ${p.id === currentPatientId ? 'selected' : ''}>${esc(p.name)}</option>`).join('');
    select.onchange = e => selectPatient((e.target as HTMLSelectElement).value);
  }
}

function addPatientFormHtml() {
  return `
    <form id="add-patient-form" class="mt-4 pt-4 border-t border-[#232725] space-y-2">
      <input id="ap-name" placeholder="New patient name" class="w-full bg-[#202322] text-[#F5F4F0] text-sm rounded-lg px-3 py-2 border border-[#2A2E2C] outline-none placeholder:text-[#6D716E]" />
      <div class="flex gap-2">
        <input id="ap-age" type="number" placeholder="Age" class="w-1/2 bg-[#202322] text-[#F5F4F0] text-sm rounded-lg px-3 py-2 border border-[#2A2E2C] outline-none placeholder:text-[#6D716E]" />
        <input id="ap-gender" placeholder="Gender" class="w-1/2 bg-[#202322] text-[#F5F4F0] text-sm rounded-lg px-3 py-2 border border-[#2A2E2C] outline-none placeholder:text-[#6D716E]" />
      </div>
      <button type="submit" class="w-full bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white text-xs font-bold py-2 rounded-lg">Add patient</button>
    </form>`;
}

function bindAddPatient() {
  $('add-patient-form')?.addEventListener('submit', async e => {
    e.preventDefault();
    const name = ($('ap-name') as HTMLInputElement).value.trim();
    if (!name) return;
    const ageVal = ($('ap-age') as HTMLInputElement).value;
    const gender = ($('ap-gender') as HTMLInputElement).value.trim();
    const created = await createPatient({
      name, age: ageVal ? parseInt(ageVal, 10) : null, gender: gender || null });
    patients = await listPatients();
    await selectPatient(created.id);
  });
}

function renderDashboard() {
  const patient = patients.find(p => p.id === currentPatientId);
  const imgEl = $('header-patient-img') as HTMLImageElement;
  if (imgEl) imgEl.src = patient?.image ?? '';
  if ($('header-patient-name')) $('header-patient-name')!.innerText = patient?.name ?? '—';
  if ($('header-patient-meta')) {
    $('header-patient-meta')!.innerHTML = patient ? `
      <span>ID: ${esc(patient.id)}</span>
      <span class="w-1 h-1 bg-[#D9D7CF] rounded-full hidden sm:block"></span>
      <span class="hidden sm:block">${esc(patient.age ?? '—')} yrs</span>
      <span class="w-1 h-1 bg-[#D9D7CF] rounded-full"></span>
      <span>Blood: <strong class="text-[#2E2C29]">${esc(patient.bloodType)}</strong></span>` : '';
  }
  if ($('header-patient-date')) $('header-patient-date')!.innerText = patient?.lastVisit ?? '—';

  if ($('filter-buttons')) {
    const filters = ['all', 'trends', 'disease', 'symptom', 'medicine', 'test_result'];
    $('filter-buttons')!.innerHTML = filters.map(type => `
      <button data-type="${type}" class="filter-btn px-3 py-1.5 text-xs font-bold rounded-lg capitalize transition-all whitespace-nowrap ${
        filterType === type ? 'bg-white text-[#2E2C29] shadow-sm' : 'text-[#8C8982] hover:text-[#2E2C29]'
      }">${type === 'all' ? 'All' : type.replace('_', ' ')}</button>`).join('');
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        filterType = (e.currentTarget as HTMLButtonElement).dataset.type!;
        renderDashboard();
        if (typeof lucide !== 'undefined') lucide.createIcons();
      });
    });
  }

  if ($('sort-label')) $('sort-label')!.innerText = sortOrder === 'desc' ? 'Newest' : 'Oldest';
  const oldBtn = $('sort-btn');
  if (oldBtn) {
    const newBtn = oldBtn.cloneNode(true);
    oldBtn.replaceWith(newBtn);
    newBtn.addEventListener('click', () => {
      sortOrder = sortOrder === 'desc' ? 'asc' : 'desc';
      renderDashboard();
      if (typeof lucide !== 'undefined') lucide.createIcons();
    });
  }

  const grid = $('records-grid');
  if (!grid) return;
  // One clean column holding a single minimalist table — scales to many documents
  // without the card grid getting noisy.
  grid.className = 'grid grid-cols-1 gap-4 pb-12';
  if (filterType === 'trends') {
    setHtml(grid, trendsShellHtml());
    bindTrendControls();
    return;
  }
  if (filterType === 'all') {
    setHtml(grid, docs.length
      ? documentsTableHtml(sortedDocs())
      : emptyHtml('file-text', 'No documents yet',
                  'Upload a PDF or image from the Knowledge panel to begin.'));
  } else {
    const view = records.filter(r => r.type === filterType);
    const noun = TYPE_NOUN[filterType] || 'record';
    setHtml(grid, view.length
      ? entityTableHtml(filterType, view)
      : emptyHtml('clipboard-list', `No ${noun}s yet`,
                  `Nothing extracted for this patient under ${noun}s.`));
  }
  bindCardButtons();
}

const DATE_COLORS = ['#5D7B6F', '#C16D54', '#6D6E9E', '#9E6D8A', '#6D9E97', '#9E946D'];
// Per-record-type label + the icon shown on each entity card.
const TYPE_NOUN: Record<string, string> = {
  disease: 'diagnosis', symptom: 'symptom', medicine: 'medication', test_result: 'result',
};

function dateColor(date: string): string {
  let h = 0;
  for (let i = 0; i < date.length; i++) h = (h * 31 + date.charCodeAt(i)) >>> 0;
  return DATE_COLORS[h % DATE_COLORS.length];
}

function emptyHtml(icon: string, title: string, sub: string): string {
  return `
    <div class="col-span-full text-center py-16 text-[#A6A298]">
      <i data-lucide="${esc(icon)}" class="w-10 h-10 mx-auto text-[#D5D2C9] mb-4"></i>
      <p class="text-lg font-light tracking-tight text-[#59554D]">${esc(title)}</p>
      <p class="text-sm mt-1">${esc(sub)}</p>
    </div>`;
}

function docTime(d: ApiDocument): number {
  return d.date ? new Date(d.date).getTime() : 0;
}

function sortedDocs(): ApiDocument[] {
  return [...docs].sort((a, b) =>
    sortOrder === 'desc' ? docTime(b) - docTime(a) : docTime(a) - docTime(b));
}

// ---- "All" tab: minimalist document table; each row opens the PDF ----
function tableShell(headCols: string, rows: string): string {
  return `
    <div class="bg-white rounded-2xl border border-[#E0DDD5] shadow-sm overflow-hidden">
      <div class="overflow-x-auto">
        <div class="min-w-[420px]">
          <div class="sticky top-0 bg-[#FAFAF8] border-b border-[#EBEBE6] z-[1]">${headCols}</div>
          ${rows}
        </div>
      </div>
    </div>`;
}

function documentsTableHtml(list: ApiDocument[]): string {
  const groups = groupDocsByYear(list, sortOrder === 'desc');
  const head = `
    <div class="grid grid-cols-[1fr_6rem] gap-3 items-center px-4 h-10 text-[10px] uppercase tracking-widest text-[#A6A298] font-bold">
      <span>Year</span><span class="text-right">Reports</span>
    </div>`;
  const rows = groups.map(g => {
    const key = `year:${g.year}`;
    const open = expandedCards.has(key);
    const body = open ? `
      <div class="px-4 pb-3 pt-1 bg-[#FCFBF8] border-b border-[#F4F3EF]">
        ${g.categories.map(c => `
          <div class="mt-2 first:mt-1">
            <div class="text-[10px] uppercase tracking-widest text-[#A6A298] font-bold px-1 pb-1">
              ${esc(c.category)} <span class="text-[#C9C6BD]">(${c.docs.length})</span>
            </div>
            ${c.docs.map(d => docRowHtml(d)).join('')}
          </div>`).join('')}
      </div>` : '';
    return `
      <div class="border-b border-[#F4F3EF] last:border-0">
        <button class="card-toggle w-full grid grid-cols-[1fr_6rem] gap-3 items-center px-4 min-h-[44px] py-1.5 text-left hover:bg-[#FAF9F5] transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5D7B6F] focus-visible:ring-inset ${open ? 'bg-[#FAF9F5]' : ''}" data-key="${esc(key)}" aria-expanded="${open}">
          <span class="flex items-center gap-2 min-w-0">
            <i data-lucide="chevron-${open ? 'down' : 'right'}" class="w-4 h-4 text-[#A6A298] shrink-0"></i>
            <span class="text-[13px] font-semibold text-[#2E2C29]">${esc(g.year)}</span>
          </span>
          <span class="text-[12px] text-[#8C8982] tabular-nums text-right">${g.total}</span>
        </button>
        ${body}
      </div>`;
  }).join('');
  return tableShell(head, rows);
}

// A single document line inside an expanded year/category group.
function docRowHtml(d: ApiDocument): string {
  const color = d.date ? dateColor(d.date) : '#e6bb4d';
  const url = docFileUrl(d.id);
  return `
    <div class="group grid grid-cols-[1fr_5.5rem_4.5rem] sm:grid-cols-[1fr_7rem_5rem] gap-3 items-center px-2 min-h-[40px] py-1 hover:bg-[#FAF9F5] rounded-lg transition-colors duration-150">
      <a href="${esc(url)}" target="_blank" rel="noopener" title="${esc(d.name)}"
         class="flex items-center gap-2.5 min-w-0 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5D7B6F]">
        <span class="w-1.5 h-1.5 rounded-full shrink-0" style="background:${color}"></span>
        <i data-lucide="file-text" class="w-4 h-4 text-[#8C8982] shrink-0"></i>
        <span class="text-[13px] font-medium text-[#2E2C29] truncate">${esc(d.name)}</span>
      </a>
      <span class="text-[12px] text-[#59554D] tabular-nums text-right whitespace-nowrap">${d.date ? esc(formatDate(d.date)) : '—'}</span>
      <span class="flex items-center gap-0.5 justify-end">
        <a href="${esc(url)}" target="_blank" rel="noopener" aria-label="Open PDF" title="Open PDF"
           class="w-8 h-8 flex items-center justify-center text-[#5D7B6F] hover:text-[#3f5b50] rounded-lg hover:bg-[#EEF2F0]"><i data-lucide="external-link" class="w-4 h-4"></i></a>
        <button class="del-doc w-8 h-8 flex items-center justify-center text-[#C0857A] hover:text-[#a3553f] rounded-lg hover:bg-[#F5EDE9]" data-id="${esc(d.id)}" data-label="${esc(d.name)}" aria-label="Delete document" title="Delete"><i data-lucide="trash-2" class="w-4 h-4"></i></button>
      </span>
    </div>`;
}

// ---- entity tabs: group a type's records per source document; expandable rows ----
interface DocGroup { docId: string; recs: ApiRecord[]; doc?: ApiDocument; date: string | null; }

function groupByDoc(view: ApiRecord[]): DocGroup[] {
  const docById = new Map(docs.map(d => [d.id, d]));
  const groups = new Map<string, ApiRecord[]>();
  for (const r of view) {
    const k = r.documentId || r.id.split('-')[1] || '';
    (groups.get(k) ?? groups.set(k, []).get(k)!).push(r);
  }
  const entries: DocGroup[] = [...groups.entries()].map(([docId, recs]) => {
    const doc = docById.get(docId);
    const date = doc?.date ?? recs.find(r => r.date)?.date ?? null;
    return { docId, recs, doc, date };
  });
  entries.sort((a, b) => {
    const ta = a.date ? new Date(a.date).getTime() : 0;
    const tb = b.date ? new Date(b.date).getTime() : 0;
    return sortOrder === 'desc' ? tb - ta : ta - tb;
  });
  return entries;
}

function entityTableHtml(type: string, view: ApiRecord[]): string {
  const noun = TYPE_NOUN[type] || 'record';
  const head = `
    <div class="grid grid-cols-[1fr_6rem_3.5rem] sm:grid-cols-[1fr_7rem_4rem] gap-3 items-center px-4 h-10 text-[10px] uppercase tracking-widest text-[#A6A298] font-bold">
      <span>Report</span><span class="text-right">Date</span><span class="text-right">${esc(noun)}s</span>
    </div>`;
  const rows = groupByDoc(view).map(g => entityRowHtml(type, g)).join('');
  return tableShell(head, rows);
}

function entityRowHtml(type: string, g: DocGroup): string {
  const key = `${type}:${g.docId}`;
  const open = expandedCards.has(key);
  const color = g.date ? dateColor(g.date) : '#edbf4a';
  const noun = TYPE_NOUN[type] || 'record';
  const title = g.doc?.name || (g.doc?.type ? `${g.doc.type}` : `${noun} record`);
  const dateLabel = g.date ? formatDate(g.date) : '—';
  const n = g.recs.length;
  const url = g.docId ? docFileUrl(g.docId) : '';
  const detail = open
    ? `<div class="px-4 pb-4 pt-2 bg-[#FCFBF8] border-b border-[#F4F3EF]">
         ${type === 'test_result' ? testTableHtml(g.recs) : entityListHtml(g.recs)}
         ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener" class="inline-flex items-center gap-1.5 mt-3 text-[11px] font-bold text-[#5D7B6F] hover:text-[#3f5b50]"><i data-lucide="external-link" class="w-3.5 h-3.5"></i> View source document</a>` : ''}
       </div>`
    : '';
  return `
    <div class="border-b border-[#F4F3EF] last:border-0">
      <button class="card-toggle w-full grid grid-cols-[1fr_6rem_3.5rem] sm:grid-cols-[1fr_7rem_4rem] gap-3 items-center px-4 min-h-[44px] py-1.5 text-left hover:bg-[#FAF9F5] transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5D7B6F] focus-visible:ring-inset ${open ? 'bg-[#FAF9F5]' : ''}" data-key="${esc(key)}" aria-expanded="${open}">
        <span class="flex items-center gap-2 min-w-0">
          <i data-lucide="chevron-${open ? 'down' : 'right'}" class="w-4 h-4 text-[#A6A298] shrink-0 transition-transform duration-150"></i>
          <span class="w-1.5 h-1.5 rounded-full shrink-0" style="background:${color}"></span>
          <span class="text-[13px] font-semibold text-[#2E2C29] truncate" title="${esc(title)}">${esc(title)}</span>
        </span>
        <span class="text-[12px] text-[#59554D] tabular-nums text-right whitespace-nowrap">${esc(dateLabel)}</span>
        <span class="text-[12px] text-[#8C8982] tabular-nums text-right">${n}</span>
      </button>
      ${detail}
    </div>`;
}

function entityListHtml(recs: ApiRecord[]): string {
  return `<div class="flex flex-col gap-1.5">${recs.map(r => `
    <div class="flex items-center gap-2.5 py-2 px-3 bg-[#FAF9F5] rounded-lg border border-[#F0EFEB]">
      <span class="w-1.5 h-1.5 rounded-full bg-[#5D7B6F] shrink-0"></span>
      <span class="text-[13px] font-medium text-[#2E2C29]">${esc(r.title)}</span>
      ${r.value ? `<span class="ml-auto text-[12px] text-[#59554D] whitespace-nowrap">${esc([r.value, r.unit].filter(Boolean).join(' '))}</span>` : ''}
    </div>`).join('')}</div>`;
}

function testTableHtml(recs: ApiRecord[]): string {
  const rows = recs.map(r => `
      <tr class="border-t border-[#F0EFEB]">
        <td class="py-2 px-3 text-[13px] font-semibold text-[#2E2C29] align-top">${esc(r.title)}</td>
        <td class="py-2 px-3 text-[13px] text-[#59554D] align-top whitespace-normal break-words">${esc([r.value, r.unit].filter(Boolean).join(' ')) || '\u2014'}</td>
        <td class="py-2 px-3 text-[12px] text-[#A6A298] whitespace-nowrap align-top">${esc(r.reference || '\u2014')}</td>
      </tr>`).join('');
  return `
    <div class="rounded-xl border border-[#F0EFEB] overflow-hidden">
      <table class="w-full text-left">
        <thead><tr class="text-[10px] uppercase tracking-widest text-[#A6A298] bg-[#FAFAF8]">
          <th class="py-1.5 px-3 font-bold">Test</th>
          <th class="py-1.5 px-3 font-bold">Result</th>
          <th class="py-1.5 px-3 font-bold">Expected</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function trendsShellHtml(): string {
  return `
    <div class="bg-white rounded-2xl border border-[#E0DDD5] shadow-sm p-5">
      <div class="flex items-center gap-3 mb-4">
        <span class="text-[10px] uppercase tracking-widest text-[#A6A298] font-bold">Metric</span>
        <select id="trend-metric" class="text-[13px] font-semibold text-[#2E2C29] bg-[#FAF9F5] border border-[#E0DDD5] rounded-lg px-3 py-1.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5D7B6F]"></select>
        <span id="trend-unit" class="text-[12px] text-[#8C8982]"></span>
      </div>
      <div class="relative h-[320px]"><canvas id="trend-canvas"></canvas></div>
      <p id="trend-empty" class="hidden text-center py-16 text-[#A6A298] text-sm"></p>
    </div>`;
}

async function bindTrendControls() {
  if (!currentPatientId) return;
  const sel = $('trend-metric') as HTMLSelectElement | null;
  const empty = $('trend-empty');
  const metrics = await getTrendMetrics(currentPatientId).catch(() => []);
  if (!metrics.length) {
    if (sel) sel.classList.add('hidden');
    const canvas = $('trend-canvas'); if (canvas) canvas.classList.add('hidden');
    if (empty) {
      empty.classList.remove('hidden');
      empty.textContent = 'No trend data yet — a test needs ≥2 numeric results to chart.';
    }
    return;
  }
  if (!trendMetric || !metrics.some(m => m.key === trendMetric)) {
    trendMetric = metrics[0].key;
  }
  if (sel) {
    setHtml(sel, metrics.map(m =>
      `<option value="${esc(m.key)}" ${m.key === trendMetric ? 'selected' : ''}>${esc(m.label)}</option>`).join(''));
    sel.addEventListener('change', () => { trendMetric = sel.value; renderTrendChart(); });
  }
  renderTrendChart();
}

async function renderTrendChart() {
  if (!currentPatientId || !trendMetric) return;
  const s = await getTrendSeries(currentPatientId, trendMetric).catch(() => null);
  const unitEl = $('trend-unit');
  if (unitEl) unitEl.textContent = s?.unit ? `(${s.unit})` : '';
  const canvas = $('trend-canvas') as HTMLCanvasElement | null;
  if (!canvas || !s) return;
  if (trendChart) { trendChart.destroy(); trendChart = null; }

  const labels = s.points.map(p => p.date);
  const values = s.points.map(p => p.value);
  const pointColors = s.points.map(p => (p.in_range ? '#5D7B6F' : '#C16D54'));
  const datasets: any[] = [{
    label: s.label, data: values, borderColor: '#5D7B6F',
    backgroundColor: '#5D7B6F', pointBackgroundColor: pointColors,
    pointRadius: 4, tension: 0.25, fill: false,
  }];
  // Shaded reference band: two flat hidden lines with fill between them.
  if (s.ref_low !== null && s.ref_high !== null) {
    datasets.push(
      { label: 'ref high', data: labels.map(() => s.ref_high), borderWidth: 0,
        pointRadius: 0, fill: '+1', backgroundColor: 'rgba(93,123,111,0.10)' },
      { label: 'ref low', data: labels.map(() => s.ref_low), borderWidth: 0,
        pointRadius: 0, fill: false },
    );
  }
  trendChart = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: false } },
    },
  });
}

function bindCardButtons() {
  // Expand / collapse an entity card.
  document.querySelectorAll('.card-toggle').forEach(btn => {
    btn.addEventListener('click', e => {
      const key = (e.currentTarget as HTMLButtonElement).dataset.key!;
      if (expandedCards.has(key)) expandedCards.delete(key); else expandedCards.add(key);
      renderDashboard();
      if (typeof lucide !== 'undefined') lucide.createIcons();
    });
  });
  // Delete a single document (cascades to its extracted records).
  document.querySelectorAll('.del-doc').forEach(btn => {
    btn.addEventListener('click', async e => {
      e.preventDefault();
      e.stopPropagation();
      const t = e.currentTarget as HTMLButtonElement;
      const id = t.dataset.id;
      if (!id) return;
      if (!confirm(`Delete "${t.dataset.label || 'this document'}" and all its extracted records? This cannot be undone.`)) return;
      try {
        await deleteRecords(currentPatientId, [id]);
        await loadPatientData();
        render();
      } catch (err: any) {
        banner(`Delete failed: ${err.message}`);
      }
    });
  });
}

function renderChatbot() {
  const tabChat = $('panel-tab-chat');
  const tabDocs = $('panel-tab-docs');
  const viewChat = $('view-chat');
  const viewDocs = $('view-docs');
  if (!tabChat || !tabDocs || !viewChat || !viewDocs) return;

  const activeCls = 'flex-1 flex items-center justify-center gap-2 py-4 px-4 text-[11px] md:text-xs font-bold uppercase tracking-widest transition-all duration-200 border-b-[3px] border-[#5D7B6F] text-[#2E2C29] bg-white/70';
  const idleCls = 'flex-1 flex items-center justify-center gap-2 py-4 px-4 text-[11px] md:text-xs font-bold uppercase tracking-widest transition-all duration-200 border-b-[3px] border-transparent text-[#9AA39E] hover:text-[#2E2C29] hover:bg-white/40 bg-transparent';

  if (panelTab === 'chat') {
    tabChat.className = activeCls;
    tabChat.innerHTML = `<div class="w-2 h-2 rounded-full bg-[#5D7B6F]"></div> Agentic AI`;
    tabDocs.className = idleCls;
    tabDocs.innerHTML = `<i data-lucide="upload-cloud" class="w-3.5 h-3.5"></i> Knowledge`;
    viewChat.classList.remove('hidden'); viewChat.classList.add('flex');
    viewDocs.classList.add('hidden'); viewDocs.classList.remove('flex');
  } else {
    tabChat.className = idleCls;
    tabChat.innerHTML = `<div class="w-2 h-2 rounded-full bg-[#D5D2C9]"></div> Agentic AI`;
    tabDocs.className = activeCls;
    tabDocs.innerHTML = `<i data-lucide="upload-cloud" class="w-3.5 h-3.5"></i> Knowledge`;
    viewChat.classList.add('hidden'); viewChat.classList.remove('flex');
    viewDocs.classList.remove('hidden'); viewDocs.classList.add('flex');
  }

  renderMessages();
  renderDocs();
}

// Circular loading spinner (Lucide loader-2 + animate-spin; lucide keeps the class).
function spinnerHtml(cls = 'w-3.5 h-3.5 text-[#5D7B6F]'): string {
  return `<i data-lucide="loader-2" class="${cls} animate-spin"></i>`;
}

const INGEST_STEPS = ['Upload', 'OCR', 'Extract', 'Review', 'Index'] as const;

// Map a backend node label to an ingestion step index (defensive substring match).
function stepFromLabel(label: string): number {
  const l = label.toLowerCase();
  if (l.includes('index') || l.includes('chunk') || l.includes('embed')) return 4;
  if (l.includes('confirm') || l.includes('review') || l.includes('patient')) return 3;
  if (l.includes('extract') || l.includes('entit')) return 2;
  if (l.includes('ocr') || l.includes('text') || l.includes('read')) return 1;
  return 0;
}

function stepperHtml(active: number): string {
  return `<div class="flex flex-col gap-2 py-1">${INGEST_STEPS.map((s, i) => {
    const done = i < active, now = i === active;
    const dot = done ? `<i data-lucide="check" class="w-3 h-3 text-white"></i>`
      : now ? spinnerHtml('w-3 h-3 text-white') : '';
    const ring = done ? 'bg-[#5D7B6F]' : now ? 'bg-[#C16D54]' : 'bg-[#E0DDD5]';
    const txt = now ? 'font-bold text-[#2E2C29]' : done ? 'text-[#5D7B6F]' : 'text-[#A6A298]';
    return `<div class="flex items-center gap-2.5">
      <span class="w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${ring}">${dot}</span>
      <span class="text-[12px] ${txt}">${s}</span>
    </div>`;
  }).join('')}</div>`;
}

function chatEmptyHtml(): string {
  return `
    <div class="h-full flex flex-col items-center justify-center text-center px-8 select-none">
      <div class="w-14 h-14 rounded-2xl bg-white/70 border border-white/80 shadow-sm flex items-center justify-center mb-4">
        <i data-lucide="sparkles" class="w-6 h-6 text-[#5D7B6F]"></i>
      </div>
      <h3 class="text-[15px] font-semibold text-[#2E2C29]">Ask about this patient</h3>
      <p class="text-[12px] text-[#7E867F] mt-1.5 max-w-[260px] leading-relaxed">
        Try “what was the latest RBC?”, “show the latest document”, or upload a report from Knowledge.
      </p>
    </div>`;
}

function renderMessages() {
  const el = $('chat-messages');
  if (!el) return;
  if (!chats.length) { setHtml(el, chatEmptyHtml()); return; }
  setHtml(el, chats.map((msg, i) => {
    if (msg.interrupt) return interruptCardHtml(msg.interrupt, i);
    const bubble = msg.sender === 'user'
      ? 'bg-gradient-to-br from-[#5E8276] to-[#46685B] text-white rounded-2xl rounded-br-md shadow-sm'
      : 'bg-white/90 border border-white/80 text-[#2E2C29] rounded-2xl rounded-bl-md shadow-[0_2px_10px_rgba(70,104,91,0.06)]';
    return `
      <div class="flex flex-col gap-1.5 max-w-[90%] md:max-w-[85%] ${msg.sender === 'user' ? 'items-end ml-auto' : 'items-start'}">
        <div class="${bubble} p-3 md:p-4 text-[13px] leading-relaxed font-medium">
          ${msg.stepper
            ? stepperHtml(msg.step ?? 0)
            : msg.live
              ? `<span class="inline-flex items-center gap-2 text-[#8C8982]">${spinnerHtml('w-3.5 h-3.5 text-[#5D7B6F]')}<span>${esc(msg.text && msg.text !== '…' ? msg.text : 'Thinking…')}</span></span>`
              : esc(msg.text)}
          ${msg.sources && msg.sources.length ? `
            <div class="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[#EBEBE6]/60">
              ${msg.sources.map(s => `
                <a href="${esc(docFileUrl(s.document_id))}" target="_blank" rel="noopener"
                   class="flex items-center gap-1.5 py-1.5 px-2.5 bg-[#F5F4F0] border border-[#E0DDD5] rounded-xl text-[#2E2C29] shadow-sm hover:border-[#5D7B6F] hover:bg-white transition-colors">
                  <i data-lucide="file-text" class="w-3 h-3 text-[#5D7B6F]"></i>
                  <span class="text-[10px] md:text-[11px] font-bold truncate max-w-[180px]">${esc(s.doc_type)}${s.date ? ' · ' + esc(formatDate(s.date)) : ''}</span>
                </a>`).join('')}
            </div>` : ''}
        </div>
        <span class="text-[9px] md:text-[10px] text-[#A6A298] font-bold tracking-widest uppercase ${msg.sender === 'user' ? 'mr-2' : 'ml-2'}">
          ${msg.sender === 'user' ? 'You' : 'Agent'} &bull; ${esc(formatTime(msg.timestamp))}
        </span>
      </div>`;
  }).join(''));
  el.scrollTop = el.scrollHeight;
  bindInterruptButtons();
}

function interruptCardHtml(payload: any, idx: number) {
  if (payload.type === 'confirm_ingest') {
    const ex = payload.extracted || {};
    // Every detected report is editable. Fall back to a single segment from `extracted`.
    const segs: any[] = (payload.segments && payload.segments.length)
      ? payload.segments
      : [{ name: ex.doc_type || 'Report', doc_type: ex.doc_type || 'document', date: ex.doc_date, extracted: ex }];
    const patientName = ex.patient_name
      || segs.map(s => s.extracted?.patient_name).find(Boolean) || '';
    const inp = (id: string, val: unknown, ph: string, cls = '') =>
      `<input id="${id}" value="${esc(val ?? '')}" placeholder="${esc(ph)}" class="${cls} bg-white border border-[#DFDDDA] rounded-md px-2 py-1 text-[12px] text-[#2E2C29] outline-none focus:border-[#5D7B6F] transition-colors" />`;
    const entityRows = (s: number, k: string, e: any) => (e[k] || []).map((it: any, j: number) =>
      inp(`int-${k}-${idx}-${s}-${j}`, it.name, k, 'w-full')).join('');
    const reportBlock = (seg: any, s: number) => {
      const e = seg.extracted || {};
      const tests = e.tests || [];
      // A row with no unit and no reference range is a narrative finding
      // (e.g. X-ray "Heart: normal"), not a numeric lab value -> show a wide
      // result field and drop the unit/ref columns. Numeric labs keep 4 cols.
      const isFinding = (t: any) => !t.unit && !t.reference_range;
      const allFindings = tests.length > 0 && tests.every(isFinding);
      const testRows = tests.map((t: any, j: number) => isFinding(t)
        ? `<div class="flex gap-1.5 items-center">
          ${inp(`int-t-${idx}-${s}-${j}-n`, t.name, 'finding', 'flex-[2]')}
          ${inp(`int-t-${idx}-${s}-${j}-v`, t.value, 'result', 'flex-[3]')}
        </div>`
        : `<div class="flex gap-1.5 items-center">
          ${inp(`int-t-${idx}-${s}-${j}-n`, t.name, 'test', 'flex-[2]')}
          ${inp(`int-t-${idx}-${s}-${j}-v`, t.value, 'value', 'flex-1')}
          ${inp(`int-t-${idx}-${s}-${j}-u`, t.unit, 'unit', 'w-16')}
          ${inp(`int-t-${idx}-${s}-${j}-r`, t.reference_range, 'ref', 'w-20')}
        </div>`).join('');
      const items = (e.tests?.length || 0) + (e.diseases?.length || 0)
        + (e.symptoms?.length || 0) + (e.medications?.length || 0);
      const open = s === 0;
      const sect = (label: string, rows: string) => rows
        ? `<div class="pt-1"><div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider mb-1">${label}</div><div class="space-y-1.5">${rows}</div></div>` : '';
      return `
        <div class="border border-[#E3E1DB] rounded-xl overflow-hidden bg-white/55">
          <button type="button" data-act="rpt-toggle" data-target="rpt-body-${idx}-${s}"
            class="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-white/80 transition-colors">
            <i data-lucide="chevron-right" class="rpt-chev w-4 h-4 text-[#8C8982] transition-transform duration-200 ${open ? 'rotate-90' : ''}"></i>
            <span class="text-[11px] font-bold text-[#2E2C29] truncate flex-1">${esc(seg.name || ('Report ' + (s + 1)))}</span>
            <span class="text-[10px] text-[#A6A298] tabular-nums">${esc(seg.date || '—')}</span>
            <span class="text-[10px] text-[#A6A298]">· ${items} items</span>
          </button>
          <div id="rpt-body-${idx}-${s}" class="px-3 pb-3 pt-1 space-y-2.5 ${open ? '' : 'hidden'}">
            <div class="flex gap-2 items-center">
              <span class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider w-12 shrink-0">Title</span>
              ${inp(`int-rname-${idx}-${s}`, seg.name, 'report title', 'flex-1')}
            </div>
            <div class="flex gap-2 items-center">
              <span class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider w-12 shrink-0">Date</span>
              ${inp(`int-rdate-${idx}-${s}`, seg.date, 'YYYY-MM-DD', 'flex-1')}
            </div>
            ${sect(allFindings ? 'Findings' : 'Tests', testRows)}
            ${sect('Diseases', entityRows(s, 'diseases', e))}
            ${sect('Symptoms', entityRows(s, 'symptoms', e))}
            ${sect('Medications', entityRows(s, 'medications', e))}
          </div>
        </div>`;
    };
    return `
      <div class="bg-gradient-to-br from-[#F5F4F0] to-[#E9E8E1] rounded-3xl p-5 md:p-6 shadow-lg border border-[#DEDCD6]">
        <div class="flex items-center gap-2 mb-3 text-[#C16D54]">
          <i data-lucide="user" class="w-3.5 h-3.5"></i>
          <span class="font-extrabold text-[9px] tracking-widest uppercase">Human in the loop — edit then confirm</span>
        </div>
        <h3 class="text-xl font-light text-[#2E2C29] mb-1 tracking-tight">Verify &amp; Correct Extraction</h3>
        <p class="text-[11px] text-[#8C8982] mb-4">${segs.length > 1
          ? segs.length + ' reports detected — each saves as its own card. Expand to verify.'
          : 'Review the extracted fields before saving.'}</p>
        <div class="flex gap-2 items-center mb-4">
          <span class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider w-16 shrink-0">Patient</span>
          ${inp(`int-name-${idx}`, patientName, 'patient name', 'flex-1')}
        </div>
        <div class="space-y-2 mb-5">${segs.map(reportBlock).join('')}</div>
        <div class="flex gap-2.5">
          <button data-act="reject" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] hover:text-[#C16D54] py-3 rounded-xl text-xs font-extrabold transition-colors">Reject</button>
          <button data-act="confirm" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold transition-transform active:scale-95">Confirm &amp; Save${segs.length > 1 ? ' All' : ''}</button>
        </div>
      </div>`;
  }
  if (payload.type === 'patient_pick') {
    const opts = (payload.patients || []).map((p: any) =>
      `<option value="${esc(p.name)}"></option>`).join('');
    return `
      <div class="bg-white rounded-3xl p-5 shadow-lg border border-[#DEDCD6]">
        <div class="flex items-center gap-2 mb-2 text-[#C16D54]">
          <i data-lucide="user-search" class="w-3.5 h-3.5"></i>
          <span class="font-extrabold text-[9px] tracking-widest uppercase">Which patient?</span>
        </div>
        <p class="text-xs text-[#8C8982] mb-4">Question didn't name a patient. Pick who it's about.</p>
        <input id="pp-input-${idx}" list="pp-list-${idx}" placeholder="Type a name…"
          class="w-full bg-white border border-[#DFDDDA] rounded-md px-3 py-2 text-sm text-[#2E2C29] outline-none focus:border-[#5D7B6F] mb-3" />
        <datalist id="pp-list-${idx}">${opts}</datalist>
        <div class="flex gap-2.5">
          <button data-act="pp-cancel" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] py-3 rounded-xl text-xs font-extrabold">Cancel</button>
          <button data-act="pp-go" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Ask</button>
        </div>
      </div>`;
  }
  if (payload.type === 'confirm_edit') {
    const ed = payload.edit || {};
    return `
      <div class="bg-gradient-to-br from-[#F5F4F0] to-[#E9E8E1] rounded-3xl p-5 md:p-6 shadow-lg border border-[#DEDCD6]">
        <div class="flex items-center gap-2 mb-3 text-[#C16D54]">
          <i data-lucide="pencil" class="w-3.5 h-3.5"></i>
          <span class="font-extrabold text-[9px] tracking-widest uppercase">Human in the loop — verify this edit</span>
        </div>
        <h3 class="text-lg font-light text-[#2E2C29] mb-1 tracking-tight">Edit ${esc(ed.label || 'record')}</h3>
        <p class="text-[11px] text-[#8C8982] mb-4">${esc(ed.doc_type || 'document')}${ed.date ? ' · ' + esc(ed.date) : ''}${ed.name ? ' · ' + esc(ed.name) : ''}</p>
        <div class="space-y-2.5 mb-5">
          <div class="flex gap-2 items-center">
            <span class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider w-20 shrink-0">Current</span>
            <span class="flex-1 bg-white border border-[#EDEBE7] rounded-md px-2.5 py-1.5 text-[12px] text-[#A6A298] line-through truncate">${esc(ed.current || '—')}</span>
          </div>
          <div class="flex gap-2 items-center">
            <span class="text-[10px] font-bold text-[#5D7B6F] uppercase tracking-wider w-20 shrink-0">New value</span>
            <input id="edit-val-${idx}" value="${esc(ed.proposed ?? '')}" class="flex-1 bg-white border border-[#DFDDDA] rounded-md px-2.5 py-1.5 text-[13px] font-semibold text-[#2E2C29] outline-none focus:border-[#5D7B6F] focus:ring-2 focus:ring-[#5D7B6F]/20" />
          </div>
        </div>
        <div class="flex gap-2.5">
          <button data-act="edit-cancel" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] hover:text-[#C16D54] py-3 rounded-xl text-xs font-extrabold">Cancel</button>
          <button data-act="edit-confirm" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Confirm &amp; Save</button>
        </div>
      </div>`;
  }
  if (payload.type === 'confirm_report') {
    const p = payload.plan || {};
    const docs = (p.documents || []).map((d: any) =>
      `<li class="truncate">${esc(d.name)} · ${esc(d.type || 'document')}${d.date ? ' · ' + esc(d.date) : ''}</li>`).join('');
    return `
      <div class="bg-gradient-to-br from-[#F5F4F0] to-[#E9E8E1] rounded-3xl p-5 md:p-6 shadow-lg border border-[#DEDCD6]">
        <div class="flex items-center gap-2 mb-3 text-[#5D7B6F]">
          <i data-lucide="file-text" class="w-3.5 h-3.5"></i>
          <span class="font-extrabold text-[9px] tracking-widest uppercase">Human in the loop — approve this report plan</span>
        </div>
        <h3 class="text-lg font-light text-[#2E2C29] mb-1 tracking-tight">${esc(p.patient_name)}</h3>
        <p class="text-[11px] text-[#8C8982] mb-3">${esc(p.timeframe_label)} · ${(p.counts?.documents ?? 0)} document(s)</p>
        <ul class="list-disc ml-5 my-1 text-[12px] text-[#5C5852] space-y-0.5 mb-5">${docs}</ul>
        <div class="flex gap-2.5">
          <button data-act="cancel" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] hover:text-[#C16D54] py-3 rounded-xl text-xs font-extrabold">Cancel</button>
          <button data-act="confirm" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Approve</button>
        </div>
      </div>`;
  }
  if (payload.type === 'confirm_delivery') {
    const s = payload.summary || {};
    return `
      <div class="bg-gradient-to-br from-[#F5F4F0] to-[#E9E8E1] rounded-3xl p-5 md:p-6 shadow-lg border border-[#DEDCD6]">
        <div class="flex items-center gap-2 mb-3 text-[#5D7B6F]">
          <i data-lucide="check-circle" class="w-3.5 h-3.5"></i>
          <span class="font-extrabold text-[9px] tracking-widest uppercase">Report ready</span>
        </div>
        <p class="text-[12px] text-[#5C5852] mb-5">${s.page_count ?? '?'} pages · ${s.chart_count ?? 0} chart(s) · ${s.attachment_count ?? 0} attachment(s)</p>
        <div class="flex gap-2.5">
          <button data-act="regenerate" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] hover:text-[#5D7B6F] py-3 rounded-xl text-xs font-extrabold">Regenerate</button>
          <a href="${esc(apiUrl(s.url))}" download="medical-report.pdf" class="flex-[2] text-center bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Download</a>
        </div>
      </div>`;
  }
  // low_confidence
  return `
    <div class="bg-white rounded-3xl p-5 shadow-lg border border-[#DEDCD6]">
      <h3 class="text-lg font-light text-[#2E2C29] mb-2">Weak match (score ${esc(payload.score ?? '?')})</h3>
      <p class="text-xs text-[#8C8982] mb-4">Answer anyway from the records found, or skip?</p>
      <div class="flex gap-2.5">
        <button data-act="skip" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] py-3 rounded-xl text-xs font-extrabold">Skip</button>
        <button data-act="proceed" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Answer anyway</button>
      </div>
    </div>`;
}

// Collect human edits for every report in the confirm card. Returns the shared
// patient name plus one edited segment per report (title, date, entities).
function collectSegments(idx: number, payload: any): { patient_name?: string, segments: any[] } {
  const ex = payload.extracted || {};
  const segs: any[] = (payload.segments && payload.segments.length)
    ? payload.segments
    : [{ name: ex.doc_type, doc_type: ex.doc_type, date: ex.doc_date, extracted: ex }];
  const g = (id: string) => ($(id) as HTMLInputElement | null)?.value;
  const patient_name = g(`int-name-${idx}`);
  const segments = segs.map((seg, s) => {
    const e = JSON.parse(JSON.stringify(seg.extracted || {}));
    if (patient_name !== undefined) e.patient_name = patient_name;
    (e.tests || []).forEach((t: any, j: number) => {
      const n = g(`int-t-${idx}-${s}-${j}-n`); if (n !== undefined) t.name = n;
      const v = g(`int-t-${idx}-${s}-${j}-v`); if (v !== undefined) t.value = v;
      const u = g(`int-t-${idx}-${s}-${j}-u`); if (u !== undefined) t.unit = u;
      const r = g(`int-t-${idx}-${s}-${j}-r`); if (r !== undefined) t.reference_range = r;
    });
    ['diseases', 'symptoms', 'medications'].forEach(k => {
      (e[k] || []).forEach((it: any, j: number) => {
        const val = g(`int-${k}-${idx}-${s}-${j}`); if (val !== undefined) it.name = val;
      });
    });
    const nm = g(`int-rname-${idx}-${s}`);
    const dt = g(`int-rdate-${idx}-${s}`);
    return { name: nm ?? seg.name, doc_type: seg.doc_type, date: dt ?? seg.date, extracted: e };
  });
  return { patient_name, segments };
}

function bindInterruptButtons() {
  // Accordion: expand/collapse a report section in the confirm card.
  document.querySelectorAll('[data-act="rpt-toggle"]').forEach(btn => {
    btn.addEventListener('click', e => {
      const t = e.currentTarget as HTMLElement;
      const body = document.getElementById(t.dataset.target!);
      if (!body) return;
      const collapsed = body.classList.toggle('hidden');
      t.querySelector('.rpt-chev')?.classList.toggle('rotate-90', !collapsed);
    });
  });
  document.querySelectorAll('.int-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      const t = e.currentTarget as HTMLButtonElement;
      const idx = parseInt(t.dataset.idx!, 10);
      const payload = chats[idx]?.interrupt;
      if (!payload) return;
      let resume: any;
      if (payload.type === 'confirm_ingest') {
        if (t.dataset.act === 'confirm') {
          const c = collectSegments(idx, payload);
          resume = { approved: true, segments: c.segments,
            extracted: c.segments[0]?.extracted,
            ...(payload.patient_id ? { patient_id: payload.patient_id } : {}),
            ...(c.patient_name ? { name: c.patient_name } : {}) };
        } else {
          resume = { approved: false };
        }
      } else if (payload.type === 'patient_pick') {
        if (t.dataset.act === 'pp-cancel') {
          resume = { patient_id: null };
        } else {
          const val = ($(`pp-input-${idx}`) as HTMLInputElement | null)?.value.trim() || '';
          const match = (payload.patients || []).find(
            (p: any) => p.name === val || String(p.id) === val);
          resume = { patient_id: match ? match.id : null };
        }
      } else if (payload.type === 'confirm_edit') {
        resume = t.dataset.act === 'edit-confirm'
          ? { approved: true,
              proposed: ($(`edit-val-${idx}`) as HTMLInputElement | null)?.value ?? payload.edit?.proposed }
          : { approved: false };
      } else if (payload.type === 'confirm_report') {
        resume = t.dataset.act === 'confirm' ? { approved: true } : { approved: false };
      } else if (payload.type === 'confirm_delivery') {
        resume = t.dataset.act === 'regenerate' ? { regenerate: true } : { approved: true };
      } else {
        resume = { proceed: t.dataset.act === 'proceed' };
      }
      chats.splice(idx, 1); // remove the card (after reading its inputs)
      // Confirming an ingest, building, or regenerating a report kicks off
      // multi-step work — show the stepper instead of a bare "Thinking…" line.
      const showStepper = (payload.type === 'confirm_ingest' && t.dataset.act === 'confirm')
        || (payload.type === 'confirm_report' && t.dataset.act === 'confirm')
        || (payload.type === 'confirm_delivery' && t.dataset.act === 'regenerate');
      runResume(resume, showStepper);
    });
  });
}

function renderDocs() {
  const dz = $('upload-container');
  if (dz) {
    dz.innerHTML = `
      <div id="dropzone" class="border-2 border-dashed border-[#DFDDDA] rounded-3xl p-6 md:p-10 mb-8 text-center transition-all cursor-pointer group flex flex-col items-center justify-center min-h-[180px] md:min-h-[220px] bg-white hover:bg-[#FAF9F5]">
        <div class="w-14 h-14 md:w-16 md:h-16 bg-[#F5F4F0] rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300 shadow-inner">
          <i data-lucide="upload-cloud" class="w-6 h-6 md:w-7 md:h-7 text-[#8C8982] group-hover:text-[#5D7B6F] transition-colors"></i>
        </div>
        <div>
          <h3 class="text-lg md:text-xl font-light tracking-tight text-[#2E2C29] mb-1.5">Feed Knowledge Base</h3>
          <p class="text-[11px] md:text-sm font-medium mt-1 text-[#8C8982]">${esc(stagedFileName || 'Click to upload a PDF or image')}</p>
        </div>
        <input id="file-input" type="file" accept=".png,.jpg,.jpeg,.webp,.pdf,.txt" class="hidden" />
      </div>`;
    const zone = $('dropzone');
    const input = $('file-input') as HTMLInputElement;
    zone?.addEventListener('click', () => input?.click());
    input?.addEventListener('change', () => {
      if (input.files && input.files[0]) handleUpload(input.files[0]);
    });
    zone?.addEventListener('dragover', e => { e.preventDefault();
      zone.classList.add('border-[#5D7B6F]'); });
    zone?.addEventListener('drop', e => {
      e.preventDefault();
      const f = (e as DragEvent).dataTransfer?.files?.[0];
      if (f) handleUpload(f);
    });
  }
  // The document list lives in the dashboard "All" tab now (document cards) —
  // the knowledge panel only stages uploads, so no duplicate table here.
}

function renderMobileTabs() {
  const tDash = $('tab-dashboard'); const tKnow = $('tab-knowledge');
  const dashView = $('dashboard-view'); const knowView = $('knowledge-view');
  if (!tDash || !tKnow || !dashView || !knowView) return;
  const on = 'px-3 py-1.5 text-[11px] md:text-xs font-bold rounded-md transition-all bg-white shadow-sm text-[#2E2C29]';
  const off = 'px-3 py-1.5 text-[11px] md:text-xs font-bold rounded-md transition-all text-[#8C8982]';
  if (mobileTab === 'dashboard') {
    tDash.className = on; tKnow.className = off;
    dashView.classList.remove('hidden'); dashView.classList.add('flex');
    knowView.classList.remove('flex'); knowView.classList.add('hidden', 'lg:flex');
  } else {
    tDash.className = off; tKnow.className = on;
    dashView.classList.add('hidden'); dashView.classList.remove('flex');
    knowView.classList.add('flex'); knowView.classList.remove('hidden', 'lg:flex');
  }
}

// ---- chat actions ----

function liveAgent(): ChatMsg {
  const m: ChatMsg = { sender: 'agent', text: '…', timestamp: nowIso(), live: true };
  chats.push(m);
  return m;
}

function streamHandlers(agent: ChatMsg) {
  return {
    onNode: (label: string) => {
      if (agent.stepper) { agent.step = stepFromLabel(label); }
      else { agent.text = label; }
      renderMessages();
    },
    onProgress: (msg: string) => { if (!agent.stepper) { agent.text = msg; renderMessages(); } },
    onInterrupt: (payload: any) => {
      const i = chats.indexOf(agent);
      if (i >= 0) chats.splice(i, 1);
      chats.push({ sender: 'agent', text: '', timestamp: nowIso(), interrupt: payload });
      render();
    },
    onMessage: (m: { content: string; sources?: CitationSource[] }) => {
      agent.text = m.content; agent.live = false; agent.stepper = false; agent.sources = m.sources;
      renderMessages();
    },
    onError: (message: string) => {
      agent.text = `⚠️ ${message}`; agent.live = false; renderMessages();
    },
    onDone: async (meta?: { patient_id?: number; document_id?: number }) => {
      agent.live = false; agent.stepper = false;
      if (meta?.patient_id != null) {
        // Ingest resolved/created a patient — refresh the cohort so the new
        // patient shows up, then focus it so its records/docs render.
        patients = await listPatients().catch(() => patients);
        await selectPatient(String(meta.patient_id));
        return;
      }
      await loadPatientData();   // ingest may have added records/docs
      render();
    },
  };
}

async function handleUpload(file: File) {
  panelTab = 'chat';
  stagedFileName = file.name;
  chats.push({ sender: 'user', text: `📎 ${file.name}`, timestamp: nowIso() });
  const agent = liveAgent();
  agent.stepper = true; agent.step = 0;   // show the ingestion stepper
  render();
  try {
    const staged = await uploadFile(file);
    activeThread = newThread('ingest');   // isolate each ingestion's graph state
    await streamChat({ thread_id: activeThread, message: 'Read this and arrange it.',
      staged_path: staged.staged_path, mime: staged.mime, ext: staged.ext,
      original_name: file.name },
      streamHandlers(agent));
  } catch (e: any) {
    agent.text = `⚠️ ${e.message}`; agent.live = false; renderMessages();
  } finally {
    stagedFileName = '';
  }
}

async function runResume(resume: any, stepper = false) {
  const agent = liveAgent();
  if (stepper) { agent.stepper = true; agent.step = 3; } // Review → Index spinner while saving
  render();
  await resumeChat({ thread_id: activeThread, resume }, streamHandlers(agent));
}

function sendText(text: string) {
  chats.push({ sender: 'user', text, timestamp: nowIso() });
  const agent = liveAgent();
  render();
  activeThread = chatThread;   // chat keeps one thread for conversation memory
  streamChat({ thread_id: activeThread, message: text,
    patient_id: currentPatientId ? parseInt(currentPatientId, 10) : null },
    streamHandlers(agent));
}

// ---- global listeners (delegated; markup is re-rendered) ----
document.addEventListener('click', e => {
  const id = (e.target as HTMLElement).closest('button')?.id;
  if (id === 'panel-tab-chat') { panelTab = 'chat'; render(); }
  else if (id === 'panel-tab-docs') { panelTab = 'docs'; render(); }
  else if (id === 'tab-dashboard') { mobileTab = 'dashboard'; render(); }
  else if (id === 'tab-knowledge') { mobileTab = 'knowledge'; render(); }
});

$('chat-form')?.addEventListener('submit', e => {
  e.preventDefault();
  const input = $('chat-input') as HTMLInputElement;
  const val = input.value.trim();
  if (!val) return;
  input.value = '';
  ($('send-btn') as HTMLButtonElement).disabled = true;
  sendText(val);
});

// ---- resizable sidebar ----
function initSidebarResize() {
  const sidebar = $('sidebar');
  const handle = $('sidebar-resizer');
  if (!sidebar || !handle) return;
  const MIN = 200, MAX = 520;
  const saved = parseInt(localStorage.getItem('sidebarWidth') || '', 10);
  if (saved >= MIN && saved <= MAX) sidebar.style.width = `${saved}px`;
  let dragging = false;
  const onMove = (e: MouseEvent) => {
    if (!dragging) return;
    const w = Math.min(MAX, Math.max(MIN, e.clientX - sidebar.getBoundingClientRect().left));
    sidebar.style.width = `${w}px`;
  };
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.userSelect = '';
    localStorage.setItem('sidebarWidth', String(parseInt(sidebar.style.width, 10)));
  };
  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    dragging = true;
    document.body.style.userSelect = 'none';
  });
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', stop);
}

// ---- resizable dashboard/chat divider ----
function initPanelResize() {
  const panel = $('knowledge-view');
  const handle = $('panel-resizer');
  if (!panel || !handle) return;
  const MIN = 320, MAX = 760;
  const saved = parseInt(localStorage.getItem('panelWidth') || '', 10);
  if (saved >= MIN && saved <= MAX) panel.style.width = `${saved}px`;
  let dragging = false;
  const onMove = (e: MouseEvent) => {
    if (!dragging) return;
    // panel is on the right; width grows as the cursor moves left of its right edge.
    const w = Math.min(MAX, Math.max(MIN, panel.getBoundingClientRect().right - e.clientX));
    panel.style.width = `${w}px`;
  };
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.userSelect = '';
    localStorage.setItem('panelWidth', String(parseInt(panel.style.width, 10) || MIN));
  };
  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    dragging = true;
    document.body.style.userSelect = 'none';
  });
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', stop);
}

initSidebarResize();
initPanelResize();
init();
