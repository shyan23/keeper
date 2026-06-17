import './index.css';
import { ApiDocument, ApiPatient, ApiRecord } from './types';
import {
  createPatient, getDocuments, getHealth, getRecords, listPatients,
  resumeChat, streamChat, uploadFile,
} from './api';

declare const lucide: any;

// Escape untrusted text before it enters innerHTML (XSS guard for DB/OCR/LLM content).
function esc(v: unknown): string {
  return String(v ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

interface ChatMsg {
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
  sources?: string[];
  live?: boolean;       // agent bubble still streaming
  interrupt?: any;      // HITL payload -> render a card instead of a bubble
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
    const filters = ['all', 'disease', 'symptom', 'medicine', 'test_result', 'treatment_plan'];
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

  const view = records
    .filter(r => filterType === 'all' || r.type === filterType)
    .sort((a, b) => {
      const da = new Date(a.date ?? 0).getTime();
      const db = new Date(b.date ?? 0).getTime();
      return sortOrder === 'desc' ? db - da : da - db;
    });

  const grid = $('records-grid');
  if (grid) {
    grid.innerHTML = view.length === 0 ? `
      <div class="col-span-full text-center py-16 md:py-20 text-[#A6A298]">
        <i data-lucide="filter" class="w-10 h-10 mx-auto text-[#D5D2C9] mb-4"></i>
        <p class="text-lg font-light tracking-tight">No records found for this filter.</p>
      </div>` : view.map(record => `
      <div class="bg-white rounded-3xl border border-[#E0DDD5] shadow-sm p-5 md:p-6 hover:shadow-md hover:-translate-y-1 transition-all flex flex-col">
        <div class="flex justify-between items-start mb-4">
          <div>
            <h3 class="text-[17px] font-bold text-[#2E2C29] leading-tight flex flex-col items-start gap-1.5">
              ${esc(record.title)}
              <span class="text-[9px] font-black text-[#A6A298] uppercase tracking-widest bg-[#F5F4F0] px-2 py-0.5 rounded-full">${esc(record.type.replace('_', ' '))}</span>
            </h3>
          </div>
          ${record.severity ? `
            <span class="px-2 py-1 text-[9px] font-bold rounded uppercase tracking-widest shadow-sm whitespace-nowrap mt-0.5 ${
              record.severity === 'High' || record.severity === 'Critical' ? 'bg-[#FF7373] text-white' : 'bg-[#E5B567] text-white'
            }">${esc(record.severity)}</span>` : `
            <span class="px-2 py-1 bg-[#F5F4F0] text-[#8C8982] text-[9px] font-bold rounded uppercase tracking-widest border border-[#EBEBE6] whitespace-nowrap mt-0.5">${esc(record.status)}</span>`}
        </div>
        <div class="flex-1 mt-1 mb-2">
          <p class="font-medium text-[#59554D] text-[13px] md:text-sm leading-relaxed">${esc(record.description)}</p>
        </div>
        <div class="mt-5 pt-4 border-t border-[#F0EFEB] flex flex-wrap gap-2 items-center justify-between">
          <p class="text-[10px] text-[#A6A298] font-bold uppercase tracking-wider flex items-center gap-1.5">
            <i data-lucide="calendar" class="w-3.5 h-3.5 text-[#5D7B6F]"></i> ${esc(record.date ?? '—')}
          </p>
          ${record.doctor ? `
            <p class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider flex items-center gap-1.5 bg-[#F5F4F0] px-2 py-1 rounded-md">
              <i data-lucide="stethoscope" class="w-3.5 h-3.5 text-[#5D7B6F]"></i> <span class="truncate max-w-[120px]">${esc(record.doctor)}</span>
            </p>` : ''}
        </div>
      </div>`).join('');
  }
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
          ${esc(msg.text)}${msg.live ? ' <span class="animate-pulse">▍</span>' : ''}
          ${msg.sources && msg.sources.length ? `
            <div class="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[#EBEBE6]/60">
              ${msg.sources.map(s => `
                <div class="flex items-center gap-1.5 py-1.5 px-2.5 bg-[#F5F4F0] border border-[#E0DDD5] rounded-xl text-[#2E2C29] shadow-sm">
                  <span class="text-[8px] md:text-[9px] font-bold text-[#5D7B6F] uppercase tracking-wider">[REF]</span>
                  <span class="text-[10px] md:text-[11px] font-bold truncate max-w-[150px]">${esc(s)}</span>
                </div>`).join('')}
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
    const name = ex.patient_name || 'New patient';
    const summary: string[] = [];
    (ex.tests || []).forEach((t: any) =>
      summary.push(`${t.name}: ${t.value ?? ''}${t.unit ?? ''}`));
    ['diseases', 'symptoms', 'medications'].forEach(k =>
      (ex[k] || []).forEach((i: any) => summary.push(i.name)));
    return `
      <div class="bg-gradient-to-br from-[#F5F4F0] to-[#E9E8E1] rounded-3xl p-5 md:p-6 shadow-lg border border-[#DEDCD6]">
        <div class="flex items-center gap-2 mb-3 text-[#C16D54]">
          <i data-lucide="user" class="w-3.5 h-3.5"></i>
          <span class="font-extrabold text-[9px] md:text-[10px] tracking-widest uppercase">Human in the loop</span>
        </div>
        <h3 class="text-xl font-light text-[#2E2C29] mb-4 tracking-tight">Verify Extraction</h3>
        <div class="space-y-1.5 mb-5 bg-white/70 p-2 rounded-2xl border border-white">
          <div class="flex justify-between items-center p-2.5">
            <div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider">Patient</div>
            <div class="text-sm font-bold text-[#2E2C29] bg-[#EBE9E4] px-2.5 py-1 rounded-md">${esc(name)}</div>
          </div>
          ${summary.slice(0, 6).map(s => `
            <div class="flex justify-between items-center p-2.5 text-[#59554D] text-xs font-semibold">${esc(s)}</div>`).join('')}
        </div>
        <div class="flex gap-2.5">
          <button data-act="reject" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] hover:text-[#C16D54] py-3 rounded-xl text-xs font-extrabold">Reject</button>
          <button data-act="confirm" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Confirm & Feed Layer</button>
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

function bindInterruptButtons() {
  document.querySelectorAll('.int-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      const t = e.currentTarget as HTMLButtonElement;
      const idx = parseInt(t.dataset.idx!, 10);
      const payload = chats[idx]?.interrupt;
      if (!payload) return;
      chats.splice(idx, 1); // remove the card
      let resume: any;
      if (payload.type === 'confirm_ingest') {
        resume = t.dataset.act === 'confirm'
          ? { approved: true, extracted: payload.extracted,
              ...(payload.patient_id ? { patient_id: payload.patient_id } : {}) }
          : { approved: false };
      } else {
        resume = { proceed: t.dataset.act === 'proceed' };
      }
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
    docsList.innerHTML = docs.map(doc => `
      <div class="bg-white border border-[#EBEBE6] p-3.5 md:p-4 rounded-2xl flex items-start gap-3 md:gap-4 shadow-sm">
        <div class="bg-[#FAF9F5] text-[#C16D54] p-3 rounded-xl shrink-0 hidden sm:block">
          <i data-lucide="file-text" class="w-5 h-5"></i>
        </div>
        <div class="flex-1 overflow-hidden pt-0.5">
          <div class="text-[13px] md:text-sm font-bold text-[#2E2C29] truncate tracking-tight" title="${esc(doc.name)}">${esc(doc.name)}</div>
          <div class="flex flex-wrap items-center gap-1.5 mt-1.5 text-[10px] md:text-[11px] text-[#A6A298] font-bold uppercase tracking-wider">
            <span>${esc(doc.type)}</span>
            <span class="w-1 h-1 rounded-full bg-[#D5D2C9]"></span>
            <span>${esc(doc.size)}</span>
          </div>
          ${doc.date ? `
          <div class="mt-2.5 text-[9px] md:text-[10px] font-bold text-[#8C8982] uppercase tracking-wider flex items-center gap-1.5 bg-[#FAF9F5] inline-flex px-2 py-1 rounded-md border border-[#EBEBE6]">
            <i data-lucide="upload-cloud" class="w-3 h-3"></i> ${esc(formatDate(doc.date))}
          </div>` : ''}
        </div>
      </div>`).join('');
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
    onNode: (label: string) => { agent.text = label; renderMessages(); },
    onProgress: (msg: string) => { agent.text = msg; renderMessages(); },
    onInterrupt: (payload: any) => {
      const i = chats.indexOf(agent);
      if (i >= 0) chats.splice(i, 1);
      chats.push({ sender: 'agent', text: '', timestamp: nowIso(), interrupt: payload });
      render();
    },
    onMessage: (m: { content: string; sources?: string[] }) => {
      agent.text = m.content; agent.live = false; agent.sources = m.sources;
      renderMessages();
    },
    onError: (message: string) => {
      agent.text = `⚠️ ${message}`; agent.live = false; renderMessages();
    },
    onDone: async () => {
      agent.live = false;
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
  render();
  try {
    const staged = await uploadFile(file);
    await streamChat({ thread_id: threadId, message: 'Read this and arrange it.',
      staged_path: staged.staged_path, mime: staged.mime, ext: staged.ext },
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

init();
