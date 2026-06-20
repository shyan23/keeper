import { MedicalGraph, GraphNode, GraphEdge, GraphAlert } from './types';

// ── helpers ──────────────────────────────────────────────────────────────────

function esc(v: unknown): string {
  return String(v ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function truncate(s: string, max = 12): string {
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}

function nodeColor(node: GraphNode): string {
  if (node.type === 'disease') return '#698A7D';
  if (node.type === 'medication') return '#8A7D98';
  if (node.status === 'critical') return '#F44336';
  if (node.status === 'warning') return '#FF9800';
  return '#4CAF50';
}

function edgeColor(confidence: number): string {
  if (confidence >= 0.80) return '#5D7B6F';
  if (confidence >= 0.60) return '#A0B4AD';
  return '#D0DAD7';
}

function edgeOpacity(confidence: number): number {
  if (confidence >= 0.80) return 0.9;
  if (confidence >= 0.60) return 0.7;
  return 0.5;
}

function formatMonthYear(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString([], { month: 'short', year: 'numeric' });
  } catch {
    return dateStr;
  }
}

// ── force-directed layout ─────────────────────────────────────────────────────

interface SimNode {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

function runForceLayout(
  nodes: GraphNode[],
  edges: GraphEdge[],
  width: number,
  height: number,
): SimNode[] {
  const PAD = 30;
  const KR = 8000;
  const KS = 0.08;
  const REST = 120;
  const KG = 0.01;
  const DAMP = 0.85;
  const ITER = 80;

  const cx = width / 2;
  const cy = height / 2;

  const sims: SimNode[] = nodes.map(() => ({
    id: '',
    x: PAD + Math.random() * (width - PAD * 2),
    y: PAD + Math.random() * (height - PAD * 2),
    vx: 0,
    vy: 0,
  }));
  nodes.forEach((n, i) => { sims[i].id = n.id; });

  const idx = new Map(sims.map((s, i) => [s.id, i]));

  for (let iter = 0; iter < ITER; iter++) {
    const fx = new Array(sims.length).fill(0);
    const fy = new Array(sims.length).fill(0);

    for (let a = 0; a < sims.length; a++) {
      for (let b = a + 1; b < sims.length; b++) {
        const dx = sims[a].x - sims[b].x;
        const dy = sims[a].y - sims[b].y;
        const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        const f = KR / (dist * dist);
        const ux = dx / dist;
        const uy = dy / dist;
        fx[a] += f * ux;
        fy[a] += f * uy;
        fx[b] -= f * ux;
        fy[b] -= f * uy;
      }
    }

    for (const edge of edges) {
      const ai = idx.get(edge.from);
      const bi = idx.get(edge.to);
      if (ai === undefined || bi === undefined) continue;
      const dx = sims[bi].x - sims[ai].x;
      const dy = sims[bi].y - sims[ai].y;
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const f = KS * (dist - REST);
      const ux = dx / dist;
      const uy = dy / dist;
      fx[ai] += f * ux;
      fy[ai] += f * uy;
      fx[bi] -= f * ux;
      fy[bi] -= f * uy;
    }

    for (let i = 0; i < sims.length; i++) {
      fx[i] += KG * (cx - sims[i].x);
      fy[i] += KG * (cy - sims[i].y);
    }

    for (let i = 0; i < sims.length; i++) {
      sims[i].vx = (sims[i].vx + fx[i]) * DAMP;
      sims[i].vy = (sims[i].vy + fy[i]) * DAMP;
      sims[i].x += sims[i].vx;
      sims[i].y += sims[i].vy;
      sims[i].x = Math.min(width - PAD, Math.max(PAD, sims[i].x));
      sims[i].y = Math.min(height - PAD, Math.max(PAD, sims[i].y));
    }
  }

  return sims;
}

// ── render alert banners ──────────────────────────────────────────────────────

function buildAlertsHtml(alerts: GraphAlert[]): string {
  if (!alerts.length) return '';
  const items = alerts.map(alert => {
    const isCritical = alert.severity === 'critical';
    const bg = isCritical ? 'bg-red-50 border-red-200' : 'bg-yellow-50 border-yellow-200';
    const textCls = isCritical ? 'text-red-700' : 'text-yellow-700';
    const iconCls = isCritical ? 'text-red-500' : 'text-yellow-500';
    return (
      '<div class="flex items-start gap-2 p-3 rounded-lg ' + bg + ' border text-sm">' +
      '<span class="' + iconCls + ' font-bold">⚠</span>' +
      '<span class="' + textCls + '">' + esc(alert.message) + '</span>' +
      '</div>'
    );
  }).join('');
  return '<div class="space-y-2 mb-4">' + items + '</div>';
}

// ── render SVG graph ──────────────────────────────────────────────────────────

function buildSvgGraph(nodes: GraphNode[], edges: GraphEdge[], width: number): string {
  const height = 340;
  const R = 22;

  if (nodes.length === 0) {
    const hw = (width / 2).toFixed(1);
    const hh1 = (height / 2 - 10).toFixed(1);
    const hh2 = (height / 2 + 10).toFixed(1);
    return (
      '<svg width="' + width + '" height="' + height + '" viewBox="0 0 ' + width + ' ' + height + '"' +
      ' xmlns="http://www.w3.org/2000/svg" class="rounded-xl border border-[#E0DDD5] bg-[#FAFAF8] w-full">' +
      '<text x="' + hw + '" y="' + hh1 + '" text-anchor="middle"' +
      ' font-size="13" fill="#A6A298" font-family="sans-serif">' +
      'No medical relationships found yet.' +
      '</text>' +
      '<text x="' + hw + '" y="' + hh2 + '" text-anchor="middle"' +
      ' font-size="12" fill="#C0BCB4" font-family="sans-serif">' +
      'Upload more documents to build your health graph.' +
      '</text>' +
      '</svg>'
    );
  }

  const positions = runForceLayout(nodes, edges, width, height);
  const posMap = new Map(positions.map(p => [p.id, p]));

  const edgeSvg = edges.map(edge => {
    const a = posMap.get(edge.from);
    const b = posMap.get(edge.to);
    if (!a || !b) return '';
    const mx = ((a.x + b.x) / 2).toFixed(1);
    const my = ((a.y + b.y) / 2).toFixed(1);
    const color = edgeColor(edge.confidence);
    const opacity = edgeOpacity(edge.confidence).toFixed(1);
    const label = '(' + edge.confidence.toFixed(2) + ')';
    return (
      '<line x1="' + a.x.toFixed(1) + '" y1="' + a.y.toFixed(1) + '"' +
      ' x2="' + b.x.toFixed(1) + '" y2="' + b.y.toFixed(1) + '"' +
      ' stroke="' + color + '" stroke-opacity="' + opacity + '" stroke-width="1.5" />' +
      '<text x="' + mx + '" y="' + my + '" text-anchor="middle"' +
      ' font-size="9" fill="' + color + '" fill-opacity="' + opacity + '"' +
      ' font-family="sans-serif">' + esc(label) + '</text>'
    );
  }).join('');

  const nodeSvg = nodes.map(node => {
    const pos = posMap.get(node.id);
    if (!pos) return '';
    const color = nodeColor(node);
    const labelText = truncate(node.label);
    const labelY = (pos.y + R + 13).toFixed(1);
    const valueText = node.type === 'test' && node.value
      ? truncate((node.value + (node.unit ? ' ' + node.unit : '')), 12)
      : null;
    const valueY = (pos.y + R + 24).toFixed(1);
    return (
      '<circle cx="' + pos.x.toFixed(1) + '" cy="' + pos.y.toFixed(1) + '" r="' + R + '"' +
      ' fill="' + color + '" fill-opacity="0.85" stroke="white" stroke-width="2" />' +
      '<text x="' + pos.x.toFixed(1) + '" y="' + labelY + '" text-anchor="middle"' +
      ' font-size="11" fill="#2E2C29" font-family="sans-serif" font-weight="500">' +
      esc(labelText) + '</text>' +
      (valueText
        ? '<text x="' + pos.x.toFixed(1) + '" y="' + valueY + '" text-anchor="middle"' +
          ' font-size="9" fill="#8C8982" font-family="sans-serif">' + esc(valueText) + '</text>'
        : '')
    );
  }).join('');

  return (
    '<svg width="' + width + '" height="' + height + '" viewBox="0 0 ' + width + ' ' + height + '"' +
    ' xmlns="http://www.w3.org/2000/svg"' +
    ' class="rounded-xl border border-[#E0DDD5] bg-[#FAFAF8] w-full" style="display:block">' +
    edgeSvg +
    nodeSvg +
    '</svg>'
  );
}

// ── render timeline strip ─────────────────────────────────────────────────────

function buildTimelineHtml(nodes: GraphNode[]): string {
  const dated = nodes
    .filter(n => n.date)
    .slice()
    .sort((a, b) => new Date(a.date!).getTime() - new Date(b.date!).getTime());

  if (!dated.length) return '';

  const chips = dated.map(n =>
    '<div class="shrink-0 flex flex-col items-center gap-1">' +
    '<div class="w-2 h-2 rounded-full bg-[#5D7B6F]"></div>' +
    '<div class="text-[10px] text-[#8C8982]">' + esc(formatMonthYear(n.date!)) + '</div>' +
    '<div class="text-[10px] font-medium text-[#2E2C29] max-w-[60px] text-center truncate"' +
    ' title="' + esc(n.label) + '">' + esc(n.label) + '</div>' +
    '</div>'
  ).join('');

  return (
    '<div class="mt-4">' +
    '<div class="text-xs font-bold text-[#8C8982] uppercase tracking-wider mb-2">Timeline</div>' +
    '<div class="flex gap-3 overflow-x-auto pb-2 hide-scrollbar">' + chips + '</div>' +
    '</div>'
  );
}

// ── public export ─────────────────────────────────────────────────────────────

export function renderGraph(container: HTMLElement, graph: MedicalGraph): void {
  const safeGraph = {
    nodes:  graph.nodes  ?? [],
    edges:  graph.edges  ?? [],
    alerts: graph.alerts ?? [],
  };

  const width = container.clientWidth || 400;
  const alertsHtml = buildAlertsHtml(safeGraph.alerts);
  const svgHtml = buildSvgGraph(safeGraph.nodes, safeGraph.edges, width);
  const timelineHtml = buildTimelineHtml(safeGraph.nodes);
  // eslint-disable-next-line -- safe: all dynamic strings pass through esc()
  container.innerHTML = alertsHtml + svgHtml + timelineHtml;
}
