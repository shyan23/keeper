import './index.css';
import { ApiDocument, ApiPatient, ApiRecord, CitationSource } from './types';
import {
  createPatient, deleteRecords, docFileUrl, getDocuments, getHealth, getRecords, listPatients,
  resumeChat, streamChat, uploadFile,
} from './api';

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
let mobileTab: 'dashboard' | 'knowledge' = 'dashboard';
let panelTab: 'chat' | 'docs' = 'chat';
let chats: ChatMsg[] = [];
let stagedFileName = '';
let docSearch = '';
let docType = 'all';
let docSort: 'newest' | 'oldest' | 'type' = 'newest';
const expandedCards = new Set<string>();   // which document cards are expanded
const threadId = `web-${Math.random().toString(36).slice(2)}-${Date.now()}`;

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
    const filters = ['all', 'disease', 'symptom', 'medicine', 'test_result'];
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
  if (filterType === 'all') {
    // "All" = the uploaded documents themselves (no entities). Each card opens its PDF.
    grid.className = 'grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-5 auto-rows-max pb-12';
    setHtml(grid, docs.length
      ? sortedDocs().map(documentCardHtml).join('')
      : emptyHtml('file-text', 'No documents yet',
                  'Upload a PDF or image from the Knowledge panel to begin.'));
  } else {
    // Entity tabs = one expandable card per source document, dated from OCR.
    grid.className = 'grid grid-cols-1 lg:grid-cols-2 gap-5 auto-rows-max pb-12';
    const view = records.filter(r => r.type === filterType);
    const noun = TYPE_NOUN[filterType] || 'record';
    setHtml(grid, view.length
      ? entityCardsHtml(filterType, view)
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
const TYPE_ICON: Record<string, string> = {
  disease: 'stethoscope', symptom: 'activity', medicine: 'pill', test_result: 'flask-conical',
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

// ---- "All" tab: one card per uploaded document; click opens the PDF ----
function documentCardHtml(d: ApiDocument): string {
  const color = d.date ? dateColor(d.date) : '#A6A298';
  const url = docFileUrl(d.id);
  return `
    <div class="group bg-white rounded-2xl border border-[#E0DDD5] shadow-sm hover:shadow-md hover:border-[#5D7B6F] transition-all overflow-hidden flex flex-col" style="border-top:3px solid ${color}">
      <a href="${esc(url)}" target="_blank" rel="noopener" class="flex-1 p-5 flex flex-col gap-3 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5D7B6F]">
        <div class="flex items-start justify-between gap-3">
          <div class="w-11 h-11 rounded-xl bg-[#F5F4F0] flex items-center justify-center shrink-0">
            <i data-lucide="file-text" class="w-5 h-5" style="color:${color}"></i>
          </div>
          <span class="text-[9px] font-bold uppercase tracking-widest text-[#A6A298] bg-[#F5F4F0] px-2 py-1 rounded-md whitespace-nowrap">${esc(d.type || 'file')}</span>
        </div>
        <div>
          <h3 class="text-[15px] font-semibold text-[#2E2C29] leading-snug truncate" title="${esc(d.name)}">${esc(d.name)}</h3>
          <div class="flex items-center gap-1.5 mt-2 text-[12px] text-[#8C8982] font-medium">
            <i data-lucide="calendar" class="w-3.5 h-3.5" style="color:${color}"></i>
            ${d.date ? esc(formatDate(d.date)) : 'Undated'}
          </div>
        </div>
      </a>
      <div class="flex items-center justify-between px-4 py-2.5 border-t border-[#F0EFEB] bg-[#FAFAF8]">
        <a href="${esc(url)}" target="_blank" rel="noopener" class="flex items-center gap-1.5 text-[11px] font-bold text-[#5D7B6F] hover:text-[#3f5b50]">
          <i data-lucide="external-link" class="w-3.5 h-3.5"></i> Open PDF
        </a>
        <button class="del-doc text-[#C16D54] hover:text-[#a3553f] p-1.5 rounded-lg hover:bg-[#F5EDE9]" data-id="${esc(d.id)}" data-label="${esc(d.name)}" aria-label="Delete document" title="Delete document">
          <i data-lucide="trash-2" class="w-4 h-4"></i>
        </button>
      </div>
    </div>`;
}

// ---- entity tabs: group a type's records by source document into dated cards ----
interface DocGroup { docId: string; recs: ApiRecord[]; doc?: ApiDocument; date: string | null; }

function entityCardsHtml(type: string, view: ApiRecord[]): string {
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
  return entries.map(g => entityCardHtml(type, g)).join('');
}

function entityCardHtml(type: string, g: DocGroup): string {
  const key = `${type}:${g.docId}`;
  const open = expandedCards.has(key);
  const color = g.date ? dateColor(g.date) : '#A6A298';
  const noun = TYPE_NOUN[type] || 'record';
  const icon = TYPE_ICON[type] || 'file-text';
  const title = g.doc?.name || (g.doc?.type ? `${g.doc.type}` : `${noun} record`);
  const dateLabel = g.date ? formatDate(g.date) : 'Undated';
  const n = g.recs.length;
  const url = g.docId ? docFileUrl(g.docId) : '';
  const body = open
    ? `<div class="px-4 pb-4 pt-1">${type === 'test_result' ? testTableHtml(g.recs) : entityListHtml(g.recs)}
         ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener" class="inline-flex items-center gap-1.5 mt-3 text-[11px] font-bold text-[#5D7B6F] hover:text-[#3f5b50]"><i data-lucide="external-link" class="w-3.5 h-3.5"></i> View source document</a>` : ''}
       </div>`
    : '';
  return `
    <div class="bg-white rounded-2xl border border-[#E0DDD5] shadow-sm hover:shadow-md transition-shadow overflow-hidden" style="border-left:4px solid ${color}">
      <button class="card-toggle w-full text-left px-4 py-3.5 flex items-center justify-between gap-3 hover:bg-[#FAF9F5] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5D7B6F]" data-key="${esc(key)}" aria-expanded="${open}">
        <div class="flex items-center gap-3 min-w-0">
          <span class="w-9 h-9 rounded-xl bg-[#F5F4F0] flex items-center justify-center shrink-0"><i data-lucide="${esc(icon)}" class="w-4 h-4" style="color:${color}"></i></span>
          <div class="min-w-0">
            <div class="text-[14px] font-semibold text-[#2E2C29] truncate" title="${esc(title)}">${esc(title)}</div>
            <div class="flex items-center gap-1.5 text-[11px] text-[#8C8982] font-medium mt-0.5">
              <i data-lucide="calendar" class="w-3 h-3"></i>${esc(dateLabel)}
              <span class="w-1 h-1 rounded-full bg-[#D9D7CF]"></span>
              ${n} ${esc(noun)}${n === 1 ? '' : 's'}
            </div>
          </div>
        </div>
        <i data-lucide="chevron-${open ? 'up' : 'down'}" class="w-4 h-4 text-[#A6A298] shrink-0"></i>
      </button>
      ${body}
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
        <td class="py-2 px-3 text-[13px] font-semibold text-[#2E2C29]">${esc(r.title)}</td>
        <td class="py-2 px-3 text-[13px] text-[#59554D] whitespace-nowrap">${esc([r.value, r.unit].filter(Boolean).join(' ')) || '\u2014'}</td>
        <td class="py-2 px-3 text-[12px] text-[#A6A298] whitespace-nowrap">${esc(r.reference || '\u2014')}</td>
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

  const activeCls = 'flex-1 flex items-center justify-center gap-2 py-4 px-4 text-[11px] md:text-xs font-bold uppercase tracking-widest transition-all border-b-[3px] border-[#5D7B6F] text-[#2E2C29] bg-white';
  const idleCls = 'flex-1 flex items-center justify-center gap-2 py-4 px-4 text-[11px] md:text-xs font-bold uppercase tracking-widest transition-all border-b-[3px] border-[#EBEBE6] text-[#A6A298] hover:text-[#2E2C29] hover:bg-[#F5F4F0] bg-[#FAFAF8] shadow-inner';

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
      : now ? `<span class="w-2 h-2 rounded-full bg-white animate-pulse"></span>` : '';
    const ring = done ? 'bg-[#5D7B6F]' : now ? 'bg-[#C16D54]' : 'bg-[#E0DDD5]';
    const txt = now ? 'font-bold text-[#2E2C29]' : done ? 'text-[#5D7B6F]' : 'text-[#A6A298]';
    return `<div class="flex items-center gap-2.5">
      <span class="w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${ring}">${dot}</span>
      <span class="text-[12px] ${txt}">${s}</span>
    </div>`;
  }).join('')}</div>`;
}

function renderMessages() {
  const el = $('chat-messages');
  if (!el) return;
  el.innerHTML = chats.map((msg, i) => {
    if (msg.interrupt) return interruptCardHtml(msg.interrupt, i);
    const bubble = msg.sender === 'user'
      ? 'bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white rounded-[20px] rounded-tr-[4px] shadow-sm'
      : 'bg-white border border-[#EBEBE6] text-[#2E2C29] rounded-[20px] rounded-tl-[4px] shadow-[0_2px_8px_rgba(0,0,0,0.02)]';
    return `
      <div class="flex flex-col gap-1.5 max-w-[90%] md:max-w-[85%] ${msg.sender === 'user' ? 'items-end ml-auto' : 'items-start'}">
        <div class="${bubble} p-3 md:p-4 text-[13px] leading-relaxed font-medium">
          ${msg.stepper ? stepperHtml(msg.step ?? 0) : `${esc(msg.text)}${msg.live ? ' <span class="animate-pulse">▍</span>' : ''}`}
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
  }).join('');
  el.scrollTop = el.scrollHeight;
  bindInterruptButtons();
}

function interruptCardHtml(payload: any, idx: number) {
  if (payload.type === 'confirm_ingest') {
    const ex = payload.extracted || {};
    const name = ex.patient_name || '';
    const tests = ex.tests || [];
    const inp = (id: string, val: unknown, ph: string, cls = '') =>
      `<input id="${id}" value="${esc(val ?? '')}" placeholder="${esc(ph)}" class="${cls} bg-white border border-[#DFDDDA] rounded-md px-2 py-1 text-[12px] text-[#2E2C29] outline-none focus:border-[#5D7B6F]" />`;
    const testRows = tests.map((t: any, j: number) => `
        <div class="flex gap-1.5 items-center">
          ${inp(`int-t-${idx}-${j}-n`, t.name, 'test', 'flex-[2]')}
          ${inp(`int-t-${idx}-${j}-v`, t.value, 'value', 'flex-1')}
          ${inp(`int-t-${idx}-${j}-u`, t.unit, 'unit', 'w-16')}
          ${inp(`int-t-${idx}-${j}-r`, t.reference_range, 'ref', 'w-20')}
        </div>`).join('');
    const nameRows = (k: string) => (ex[k] || []).map((it: any, j: number) =>
      inp(`int-${k}-${idx}-${j}`, it.name, k, 'w-full')).join('');
    return `
      <div class="bg-gradient-to-br from-[#F5F4F0] to-[#E9E8E1] rounded-3xl p-5 md:p-6 shadow-lg border border-[#DEDCD6]">
        <div class="flex items-center gap-2 mb-3 text-[#C16D54]">
          <i data-lucide="user" class="w-3.5 h-3.5"></i>
          <span class="font-extrabold text-[9px] tracking-widest uppercase">Human in the loop — edit then confirm</span>
        </div>
        <h3 class="text-xl font-light text-[#2E2C29] mb-4 tracking-tight">Verify &amp; Correct Extraction</h3>
        <div class="space-y-3 mb-5">
          <div class="flex gap-2 items-center">
            <span class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider w-16">Patient</span>
            ${inp(`int-name-${idx}`, name, 'patient name', 'flex-1')}
          </div>
          <div class="flex gap-2 items-center">
            <span class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider w-16">Date</span>
            ${inp(`int-date-${idx}`, ex.doc_date, 'YYYY-MM-DD', 'flex-1')}
          </div>
          ${tests.length ? `<div class="pt-1"><div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider mb-1">Tests</div><div class="space-y-1.5">${testRows}</div></div>` : ''}
          ${(ex.diseases || []).length ? `<div class="pt-1"><div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider mb-1">Diseases</div><div class="space-y-1.5">${nameRows('diseases')}</div></div>` : ''}
          ${(ex.symptoms || []).length ? `<div class="pt-1"><div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider mb-1">Symptoms</div><div class="space-y-1.5">${nameRows('symptoms')}</div></div>` : ''}
          ${(ex.medications || []).length ? `<div class="pt-1"><div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider mb-1">Medications</div><div class="space-y-1.5">${nameRows('medications')}</div></div>` : ''}
        </div>
        <div class="flex gap-2.5">
          <button data-act="reject" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] hover:text-[#C16D54] py-3 rounded-xl text-xs font-extrabold">Reject</button>
          <button data-act="confirm" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Confirm &amp; Feed Layer</button>
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

function collectExtracted(idx: number, base: any): any {
  const ex = JSON.parse(JSON.stringify(base || {}));
  const g = (id: string) => ($(id) as HTMLInputElement | null)?.value;
  const nm = g(`int-name-${idx}`); if (nm !== undefined) ex.patient_name = nm;
  const dt = g(`int-date-${idx}`); if (dt !== undefined) ex.doc_date = dt;
  (ex.tests || []).forEach((t: any, j: number) => {
    const n = g(`int-t-${idx}-${j}-n`); if (n !== undefined) t.name = n;
    const v = g(`int-t-${idx}-${j}-v`); if (v !== undefined) t.value = v;
    const u = g(`int-t-${idx}-${j}-u`); if (u !== undefined) t.unit = u;
    const r = g(`int-t-${idx}-${j}-r`); if (r !== undefined) t.reference_range = r;
  });
  ['diseases', 'symptoms', 'medications'].forEach(k => {
    (ex[k] || []).forEach((it: any, j: number) => {
      const val = g(`int-${k}-${idx}-${j}`); if (val !== undefined) it.name = val;
    });
  });
  return ex;
}

function bindInterruptButtons() {
  document.querySelectorAll('.int-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      const t = e.currentTarget as HTMLButtonElement;
      const idx = parseInt(t.dataset.idx!, 10);
      const payload = chats[idx]?.interrupt;
      if (!payload) return;
      let resume: any;
      if (payload.type === 'confirm_ingest') {
        resume = t.dataset.act === 'confirm'
          ? { approved: true, extracted: collectExtracted(idx, payload.extracted),
              ...(payload.patient_id ? { patient_id: payload.patient_id } : {}) }
          : { approved: false };
      } else if (payload.type === 'patient_pick') {
        if (t.dataset.act === 'pp-cancel') {
          resume = { patient_id: null };
        } else {
          const val = ($(`pp-input-${idx}`) as HTMLInputElement | null)?.value.trim() || '';
          const match = (payload.patients || []).find(
            (p: any) => p.name === val || String(p.id) === val);
          resume = { patient_id: match ? match.id : null };
        }
      } else {
        resume = { proceed: t.dataset.act === 'proceed' };
      }
      chats.splice(idx, 1); // remove the card (after reading its inputs)
      runResume(resume);
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

  const docsList = $('docs-list');
  if (docsList) {
    const types = ['all', ...Array.from(new Set(docs.map(d => d.type).filter(Boolean)))];
    const view = docs.filter(d =>
      (docType === 'all' || d.type === docType) &&
      (!docSearch || d.name.toLowerCase().includes(docSearch.toLowerCase())));
    view.sort((a, b) => {
      if (docSort === 'type') return (a.type || '').localeCompare(b.type || '');
      const ta = a.date ? new Date(a.date).getTime() : 0;
      const tb = b.date ? new Date(b.date).getTime() : 0;
      return docSort === 'newest' ? tb - ta : ta - tb;
    });
    const rows = view.map(d => `
      <tr class="border-t border-[#F0EFEB] hover:bg-[#FAF9F5]">
        <td class="py-2 px-3 text-[13px] font-semibold text-[#2E2C29] truncate max-w-[220px]" title="${esc(d.name)}">
          <i data-lucide="file-text" class="w-3.5 h-3.5 inline text-[#C16D54] mr-1.5"></i>${esc(d.name)}</td>
        <td class="py-2 px-3 text-[11px] uppercase tracking-wider text-[#A6A298] font-bold whitespace-nowrap">${esc(d.type)}</td>
        <td class="py-2 px-3 text-[12px] text-[#59554D] whitespace-nowrap">${d.date ? esc(formatDate(d.date)) : '—'}</td>
        <td class="py-2 px-3 text-right"><a href="${esc(docFileUrl(d.id))}" target="_blank" rel="noopener"
          class="text-[11px] font-bold text-[#5D7B6F] hover:text-[#3f5b50]">Open</a></td>
      </tr>`).join('');
    docsList.innerHTML = `
      <div class="flex flex-wrap gap-2 mb-3 items-center">
        <input id="doc-search" value="${esc(docSearch)}" placeholder="Search documents…"
          class="flex-1 min-w-[140px] bg-white border border-[#E0DDD5] rounded-lg px-3 py-1.5 text-xs outline-none focus:border-[#5D7B6F]" />
        <select id="doc-type" class="bg-white border border-[#E0DDD5] rounded-lg px-2 py-1.5 text-xs">
          ${types.map(t => `<option value="${esc(t)}" ${t === docType ? 'selected' : ''}>${t === 'all' ? 'All types' : esc(t)}</option>`).join('')}
        </select>
        <select id="doc-sort" class="bg-white border border-[#E0DDD5] rounded-lg px-2 py-1.5 text-xs">
          <option value="newest" ${docSort === 'newest' ? 'selected' : ''}>Newest</option>
          <option value="oldest" ${docSort === 'oldest' ? 'selected' : ''}>Oldest</option>
          <option value="type" ${docSort === 'type' ? 'selected' : ''}>Type</option>
        </select>
      </div>
      ${view.length ? `<div class="bg-white rounded-2xl border border-[#E0DDD5] overflow-hidden">
        <table class="w-full text-left">
          <thead><tr class="text-[10px] uppercase tracking-widest text-[#A6A298]">
            <th class="py-1.5 px-3 font-bold">Document</th><th class="py-1.5 px-3 font-bold">Type</th>
            <th class="py-1.5 px-3 font-bold">Date</th><th class="py-1.5 px-3 font-bold text-right">Action</th>
          </tr></thead><tbody>${rows}</tbody></table></div>`
        : `<div class="text-center py-12 text-[#A6A298] text-sm">No documents.</div>`}`;
    const si = $('doc-search') as HTMLInputElement | null;
    if (si) si.oninput = () => { docSearch = si.value; renderDocs(); if (typeof lucide !== 'undefined') lucide.createIcons(); si.focus(); };
    const ts = $('doc-type') as HTMLSelectElement | null;
    if (ts) ts.onchange = () => { docType = ts.value; renderDocs(); if (typeof lucide !== 'undefined') lucide.createIcons(); };
    const so = $('doc-sort') as HTMLSelectElement | null;
    if (so) so.onchange = () => { docSort = so.value as typeof docSort; renderDocs(); if (typeof lucide !== 'undefined') lucide.createIcons(); };
  }
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
    await streamChat({ thread_id: threadId, message: 'Read this and arrange it.',
      staged_path: staged.staged_path, mime: staged.mime, ext: staged.ext,
      original_name: file.name },
      streamHandlers(agent));
  } catch (e: any) {
    agent.text = `⚠️ ${e.message}`; agent.live = false; renderMessages();
  } finally {
    stagedFileName = '';
  }
}

async function runResume(resume: any) {
  const agent = liveAgent();
  render();
  await resumeChat({ thread_id: threadId, resume }, streamHandlers(agent));
}

function sendText(text: string) {
  chats.push({ sender: 'user', text, timestamp: nowIso() });
  const agent = liveAgent();
  render();
  streamChat({ thread_id: threadId, message: text,
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
