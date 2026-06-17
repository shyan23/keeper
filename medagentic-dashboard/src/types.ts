export type RecordType = 'disease' | 'symptom' | 'medicine' | 'test_result' | 'treatment_plan';

export interface Document {
  id: string;
  name: string;
  date: string;
  type: string;
  size: string;
}

export interface PatientRecord {
  id: string;
  patientId: string;
  type: RecordType;
  title: string;
  description: string;
  date: string;
  status: string;
  severity?: 'Low' | 'Medium' | 'High' | 'Critical';
  doctor?: string;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  bloodType: string;
  image: string;
  lastVisit: string;
  status: string;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
  sources?: string[];
}

export interface Health {
  status: string;
  db: string;
  pgvector: boolean;
  version: string;
}

export interface ApiPatient {
  id: string;
  name: string;
  age: number | null;
  gender: string | null;
  bloodType: string;
  image: string;
  lastVisit: string;
  status: string;
}

export interface ApiRecord {
  id: string;
  documentId: string;
  patientId: string;
  type: string;
  title: string;
  description: string;
  value: string;
  unit: string;
  reference: string;
  date: string | null;
  status: string;
  severity: string | null;
  doctor: string | null;
}

export interface ApiDocument {
  id: string;
  name: string;
  date: string | null;
  type: string;
  size: string;
}

export interface CitationSource {
  document_id: number;
  name: string;
  doc_type: string;
  date: string | null;
}

export interface SseHandlers {
  onNode?: (label: string) => void;
  onProgress?: (msg: string) => void;
  onInterrupt?: (payload: any) => void;
  onMessage?: (msg: { role: string; content: string; sources?: CitationSource[] }) => void;
  onError?: (message: string) => void;
  onDone?: (meta?: { patient_id?: number; document_id?: number }) => void;
}
