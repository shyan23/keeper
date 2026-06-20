import {
  ApiDocument, ApiPatient, ApiRecord, Health, SseHandlers, TrendMetric, TrendSeries,
} from './types';

const API = (import.meta as any).env?.VITE_API_BASE ?? 'http://localhost:8000';

// Absolute URL to stream a document's original file (citations / docs table "Open").
export const docFileUrl = (id: string | number) => `${API}/api/documents/${id}/file`;

// Backend emits a root-relative report path (/api/chat/report/<id>.pdf). The dev
// server runs on a different origin than the API, so prefix the API base.
export const apiUrl = (path: string) =>
  path.startsWith('http') ? path : `${API}${path}`;

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const getHealth = () => json<Health>('/api/health');
export const listPatients = () => json<ApiPatient[]>('/api/patients');
export const createPatient = (body: {
  name: string; age?: number | null; gender?: string | null; relationship?: string | null;
}) => json<ApiPatient>('/api/patients', { method: 'POST', body: JSON.stringify(body) });
export const getRecords = (patientId: string) =>
  json<ApiRecord[]>(`/api/patients/${patientId}/records`);
export const getDocuments = (patientId: string) =>
  json<ApiDocument[]>(`/api/patients/${patientId}/documents`);
export const getTrendMetrics = (patientId: string) =>
  json<TrendMetric[]>(`/api/patients/${patientId}/trends`);
export const getTrendSeries = (patientId: string, key: string) =>
  json<TrendSeries>(`/api/patients/${patientId}/trends/${encodeURIComponent(key)}`);
export const deleteRecords = (patientId: string, documentIds: string[]) =>
  json<{ deleted: number }>(`/api/patients/${patientId}/records/delete`, {
    method: 'POST', body: JSON.stringify({ document_ids: documentIds }),
  });

export async function uploadFile(file: File):
  Promise<{ staged_path: string; mime: string; ext: string }> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API}/api/chat/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

// Read an SSE stream from a POST response and dispatch to handlers.
async function readSse(res: Response, h: SseHandlers): Promise<void> {
  const t0 = performance.now();
  console.debug('[sse] stream opened', res.status);
  if (!res.ok || !res.body) {
    h.onError?.(`${res.status} ${res.statusText}`);
    h.onDone?.();
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const blocks = buf.split('\n\n');
    buf = blocks.pop() ?? '';
    for (const block of blocks) {
      let event = 'message';
      let data = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      let parsed: any = {};
      try { parsed = data ? JSON.parse(data) : {}; } catch { parsed = {}; }
      console.debug(`[sse +${((performance.now() - t0) / 1000).toFixed(2)}s] ${event}`, parsed);
      if (event === 'node') h.onNode?.(parsed.label);
      else if (event === 'progress') h.onProgress?.(parsed.msg);
      else if (event === 'interrupt') h.onInterrupt?.(parsed);
      else if (event === 'message') h.onMessage?.(parsed);
      else if (event === 'error') h.onError?.(parsed.message);
      else if (event === 'done') h.onDone?.(parsed);
    }
  }
}

export async function streamChat(body: {
  thread_id: string; message?: string; patient_id?: number | null;
  staged_path?: string; mime?: string; ext?: string; original_name?: string;
}, handlers: SseHandlers): Promise<void> {
  const res = await fetch(`${API}/api/chat/stream`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  await readSse(res, handlers);
}

export const getActivity = () => json<any>('/api/tracer/activity');

export async function resumeChat(body: { thread_id: string; resume: any },
                                 handlers: SseHandlers): Promise<void> {
  const res = await fetch(`${API}/api/chat/resume`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  await readSse(res, handlers);
}
