import cytoscape from 'cytoscape';
import { MedicalGraph, GraphNode } from './types';

// ── helpers ───────────────────────────────────────────────────────────────────

function esc(v: unknown): string {
  return String(v ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function nodeStyle(type: string, status: string) {
  if (type === 'disease')    return { bg: '#5D7B6F', border: '#3D5B4F' };
  if (type === 'medication') return { bg: '#7B6F8A', border: '#5B4F6A' };
  if (status === 'critical') return { bg: '#E53935', border: '#B71C1C' };
  if (status === 'warning')  return { bg: '#FB8C00', border: '#E65100' };
  return { bg: '#43A047', border: '#2E7D32' };
}

function edgeColor(confidence: number): string {
  if (confidence >= 0.8) return '#5D7B6F';
  if (confidence >= 0.6) return '#A0B4AD';
  return '#D0DAD7';
}

// ── public API ────────────────────────────────────────────────────────────────

export function renderGraph(container: HTMLElement, graph: MedicalGraph): void {
  // Destroy any previous Cytoscape instance bound to this container.
  (container as any)._cy?.destroy();
  // Remove previous child nodes (re-render).
  while (container.firstChild) container.removeChild(container.firstChild);

  const nodes  = graph.nodes  ?? [];
  const edges  = graph.edges  ?? [];
  const alerts = graph.alerts ?? [];

  _renderAlerts(container, alerts);

  const graphDiv = document.createElement('div');
  graphDiv.style.cssText = 'width:100%;height:400px;border-radius:12px;overflow:hidden;background:#FAFAF8;border:1px solid #E0DDD5;';
  container.appendChild(graphDiv);

  if (!nodes.length) {
    graphDiv.style.cssText += 'display:flex;align-items:center;justify-content:center;';
    const msg = document.createElement('div');
    msg.style.cssText = 'text-align:center;color:#A6A298;font-size:13px;line-height:1.8;font-family:ui-sans-serif,system-ui,sans-serif';
    const line1 = document.createElement('div');
    line1.textContent = 'No medical relationships found yet.';
    const line2 = document.createElement('div');
    line2.style.cssText = 'font-size:11px;color:#C0BCB4';
    line2.textContent = 'Upload more documents to build your health graph.';
    msg.appendChild(line1);
    msg.appendChild(line2);
    graphDiv.appendChild(msg);
    return;
  }

  const elements = [
    ...nodes.map(n => {
      const s = nodeStyle(n.type, n.status ?? 'normal');
      return {
        data: {
          id: n.id,
          label: n.label,
          type: n.type,
          value: n.value ?? null,
          unit: n.unit ?? null,
          status: n.status ?? 'normal',
          bg: s.bg,
          border: s.border,
        },
      };
    }),
    ...edges.map((e, i) => ({
      data: {
        id: `e${i}`,
        source: e.from,
        target: e.to,
        label: e.type.replace(/_/g, ' '),
        confidence: e.confidence,
        color: edgeColor(e.confidence),
      },
    })),
  ];

  const cy = cytoscape({
    container: graphDiv,
    elements,
    style: [
      {
        selector: 'node',
        style: {
          'background-color': 'data(bg)',
          'border-color': 'data(border)',
          'border-width': 2,
          'label': 'data(label)',
          'color': '#ffffff',
          'font-size': 11,
          'font-weight': 'bold',
          'font-family': 'ui-sans-serif, system-ui, sans-serif',
          'text-valign': 'center',
          'text-halign': 'center',
          'text-wrap': 'wrap',
          'text-max-width': '72px',
          'width': 68,
          'height': 68,
          'text-outline-color': 'data(bg)',
          'text-outline-width': 1,
          'transition-property': 'border-width, width, height',
          'transition-duration': 150,
        } as any,
      },
      {
        selector: 'node:selected',
        style: { 'border-width': 4, 'width': 76, 'height': 76 } as any,
      },
      {
        selector: 'node[type="test"]',
        style: { 'shape': 'round-rectangle' } as any,
      },
      {
        selector: 'edge',
        style: {
          'width': 1.5,
          'line-color': 'data(color)',
          'target-arrow-color': 'data(color)',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          'label': 'data(label)',
          'font-size': 9,
          'color': '#8C8982',
          'font-family': 'ui-sans-serif, system-ui, sans-serif',
          'text-rotation': 'autorotate',
          'text-background-color': '#FAFAF8',
          'text-background-opacity': 0.9,
          'text-background-padding': '2px',
          'text-background-shape': 'round-rectangle',
          'opacity': 0.85,
        } as any,
      },
      {
        selector: 'edge:selected',
        style: { 'width': 3, 'opacity': 1 } as any,
      },
    ],
    layout: {
      name: 'cose',
      animate: true,
      animationDuration: 500,
      padding: 30,
      nodeOverlap: 20,
      fit: true,
      randomize: false,
      componentSpacing: 100,
      nodeRepulsion: () => 4500,
      edgeElasticity: () => 100,
      gravity: 80,
      numIter: 1000,
      initialTemp: 200,
      coolingFactor: 0.95,
      minTemp: 1.0,
    } as any,
    minZoom: 0.3,
    maxZoom: 3,
    wheelSensitivity: 0.3,
  });

  // Tooltip — built with safe DOM methods, no innerHTML for dynamic content.
  const tooltip = document.createElement('div');
  tooltip.style.cssText =
    'position:absolute;background:rgba(20,20,20,0.9);color:#fff;font-size:11px;padding:7px 11px;' +
    'border-radius:8px;pointer-events:none;display:none;z-index:999;max-width:200px;line-height:1.5;' +
    'font-family:ui-sans-serif,system-ui,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,.2)';
  container.style.position = 'relative';
  container.appendChild(tooltip);

  cy.on('mouseover', 'node', e => {
    const d = e.target.data();
    // Safe DOM construction — no innerHTML for user-derived strings.
    while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);

    const name = document.createElement('strong');
    name.textContent = d.label;
    tooltip.appendChild(name);

    const typeEl = document.createElement('span');
    typeEl.style.cssText = 'display:block;opacity:.65;margin-top:1px';
    typeEl.textContent = d.type;
    tooltip.appendChild(typeEl);

    if (d.value) {
      const valEl = document.createElement('span');
      valEl.style.cssText = 'display:block;margin-top:2px';
      valEl.textContent = d.value + (d.unit ? ' ' + d.unit : '');
      tooltip.appendChild(valEl);
    }
    if (d.status && d.status !== 'normal') {
      const stEl = document.createElement('span');
      stEl.style.cssText = 'display:block;color:#FFA726;margin-top:2px;font-weight:bold;text-transform:capitalize';
      stEl.textContent = d.status;
      tooltip.appendChild(stEl);
    }
    tooltip.style.display = 'block';
  });
  cy.on('mouseout', 'node', () => { tooltip.style.display = 'none'; });
  cy.on('mousemove', (e: any) => {
    const pos = e.originalEvent as MouseEvent;
    const rect = container.getBoundingClientRect();
    tooltip.style.left = (pos.clientX - rect.left + 14) + 'px';
    tooltip.style.top  = (pos.clientY - rect.top  - 10) + 'px';
  });

  (container as any)._cy = cy;
  _renderTimeline(container, nodes);
}

// ── alerts banner ─────────────────────────────────────────────────────────────

function _renderAlerts(container: HTMLElement, alerts: MedicalGraph['alerts']): void {
  if (!alerts.length) return;
  const wrap = document.createElement('div');
  wrap.style.cssText = 'margin-bottom:12px';
  for (const a of alerts) {
    const crit = a.severity === 'critical';
    const row = document.createElement('div');
    row.style.cssText =
      `display:flex;align-items:flex-start;gap:8px;padding:10px 12px;border-radius:10px;` +
      `border:1px solid ${crit ? '#FFCDD2' : '#FFE0B2'};` +
      `background:${crit ? '#FFF5F5' : '#FFF8F0'};font-size:12px;margin-bottom:6px;` +
      `font-family:ui-sans-serif,system-ui,sans-serif`;
    const icon = document.createElement('span');
    icon.style.cssText = `color:${crit ? '#E53935' : '#FB8C00'};font-weight:bold;flex-shrink:0`;
    icon.textContent = '⚠';
    const text = document.createElement('span');
    text.style.cssText = `color:${crit ? '#C62828' : '#E65100'}`;
    text.textContent = a.message;   // textContent: safe
    row.appendChild(icon);
    row.appendChild(text);
    wrap.appendChild(row);
  }
  container.appendChild(wrap);
}

// ── timeline strip ────────────────────────────────────────────────────────────

function _renderTimeline(container: HTMLElement, nodes: GraphNode[]): void {
  const dated = nodes
    .filter(n => n.date)
    .sort((a, b) => new Date(a.date!).getTime() - new Date(b.date!).getTime());
  if (!dated.length) return;

  const wrap = document.createElement('div');
  wrap.style.cssText = 'margin-top:12px';

  const heading = document.createElement('div');
  heading.style.cssText = 'font-size:10px;font-weight:700;color:#8C8982;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;font-family:ui-sans-serif,system-ui,sans-serif';
  heading.textContent = 'Timeline';
  wrap.appendChild(heading);

  const strip = document.createElement('div');
  strip.style.cssText = 'display:flex;gap:12px;overflow-x:auto;padding-bottom:6px';

  for (const n of dated) {
    const d = new Date(n.date!).toLocaleDateString([], { month: 'short', year: 'numeric' });
    const chip = document.createElement('div');
    chip.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:4px;flex-shrink:0';

    const dot = document.createElement('div');
    dot.style.cssText = 'width:8px;height:8px;border-radius:50%;background:#5D7B6F';

    const dateEl = document.createElement('div');
    dateEl.style.cssText = 'font-size:9px;color:#8C8982;font-family:ui-sans-serif,system-ui,sans-serif';
    dateEl.textContent = d;

    const labelEl = document.createElement('div');
    labelEl.style.cssText = 'font-size:9px;font-weight:600;color:#2E2C29;max-width:60px;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-sans-serif,system-ui,sans-serif';
    labelEl.title = n.label;
    labelEl.textContent = n.label;

    chip.appendChild(dot);
    chip.appendChild(dateEl);
    chip.appendChild(labelEl);
    strip.appendChild(chip);
  }

  wrap.appendChild(strip);
  container.appendChild(wrap);
}
