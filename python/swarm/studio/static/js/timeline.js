// Timeline scrubber: channel load and detections over the run, fault / election markers, the playhead.

import { fmtClock } from './util.js';

const MARK = { kill: '#ff3b4d', revive: '#3dffa0', leader: '#ffb627' };

export class Timeline {
  constructor(canvas, onSeek) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.onSeek = onSeek;
    this.T = 1; this.t = 0; this.bins = null; this.marks = []; this.hover = null;
    const seekAt = (e) => {
      const r = this.c.getBoundingClientRect();
      const x = Math.min(Math.max(0, e.clientX - r.left - this.pad), r.width - 2 * this.pad);
      this.onSeek((x / (r.width - 2 * this.pad)) * this.T);
    };
    canvas.addEventListener('pointerdown', (e) => { canvas.setPointerCapture(e.pointerId); this.drag = true; seekAt(e); });
    canvas.addEventListener('pointermove', (e) => {
      if (this.drag) seekAt(e);
      const r = this.c.getBoundingClientRect();
      this.hover = e.clientX - r.left;
      this.draw();
    });
    canvas.addEventListener('pointerup', () => { this.drag = false; });
    canvas.addEventListener('pointerleave', () => { this.hover = null; this.draw(); });
    this.pad = 14;
  }

  setData(pb) {
    this.T = pb.T;
    const nb = 240, bw = this.T / nb;
    const air = new Float64Array(nb), det = new Float64Array(nb), lost = new Float64Array(nb);
    for (const [s, e] of pb.air) {
      for (let b = Math.max(0, Math.floor(s / bw)); b < nb && b * bw < e; b++) {
        air[b] += Math.max(0, Math.min(e, (b + 1) * bw) - Math.max(s, b * bw));
      }
    }
    for (const t of pb.repT) { const b = Math.min(nb - 1, Math.floor(t / bw)); det[b] += 1; }
    for (const r of pb.tx) if (r[8]) { const b = Math.min(nb - 1, Math.floor(r[1] / bw)); lost[b] += 1; }
    const dmax = Math.max(1, ...det), lmax = Math.max(1, ...lost);
    this.bins = { nb, air: Array.from(air, (v) => Math.min(1.2, v / bw)), det: Array.from(det, (v) => v / dmax), lost: Array.from(lost, (v) => v / lmax) };
    const firstSuspect = new Map();
    this.marks = [];
    for (const m of pb.marks) {
      if (m.kind === 'suspect') {
        if (!firstSuspect.has(m.node) || m.t - firstSuspect.get(m.node) > 1) { firstSuspect.set(m.node, m.t); this.marks.push(m); }
      } else if (m.kind !== 'readmit') this.marks.push(m);
    }
    this.draw();
  }

  set(t) { this.t = t; this.draw(); }

  draw() {
    const c = this.c, ctx = this.ctx;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = c.clientWidth, H = c.clientHeight;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const pad = this.pad, w = W - 2 * pad, top = 18, bh = H - top - 18;
    const X = (t) => pad + (t / this.T) * w;
    ctx.fillStyle = '#04080c';
    ctx.fillRect(pad, top, w, bh);
    ctx.strokeStyle = '#15232d';
    ctx.strokeRect(pad + 0.5, top + 0.5, w - 1, bh - 1);
    // ticks
    ctx.font = '10px Cascadia Mono, Consolas, monospace';
    ctx.fillStyle = '#4c6575';
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300].find((s) => this.T / s <= 12) || 600;
    for (let s = 0; s <= this.T + 1e-9; s += step) {
      const x = X(s);
      ctx.fillRect(x, top + bh, 1, 4);
      ctx.fillText(fmtClock(s, false).slice(0, 5), x - 14, H - 3);
    }
    if (this.bins) {
      const { nb, air, det, lost } = this.bins, bwpx = w / nb;
      for (let b = 0; b < nb; b++) {
        const a = air[b];
        if (a > 0) {
          ctx.fillStyle = a > 0.9 ? 'rgba(255,59,77,0.55)' : 'rgba(26,169,194,0.45)';
          const hh = Math.max(1, Math.min(1, a) * bh * 0.7);
          ctx.fillRect(pad + b * bwpx, top + bh - hh, Math.max(1, bwpx - 0.5), hh);
        }
        if (det[b] > 0) { ctx.fillStyle = 'rgba(61,255,160,0.75)'; ctx.fillRect(pad + b * bwpx, top + 2, Math.max(1, bwpx - 0.5), 2 + det[b] * 6); }
        if (lost[b] > 0) { ctx.fillStyle = 'rgba(255,59,77,0.8)'; ctx.fillRect(pad + b * bwpx, top + bh - 3, Math.max(1, bwpx - 0.5), 3); }
      }
    }
    for (const m of this.marks) {
      const x = X(m.t);
      if (m.kind === 'suspect') { ctx.fillStyle = 'rgba(255,138,149,0.7)'; ctx.fillRect(x, top, 1, bh); continue; }
      ctx.fillStyle = MARK[m.kind] || '#888';
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x - 0.5, top, 1.5, bh);
      ctx.globalAlpha = 1;
      ctx.beginPath();
      if (m.kind === 'leader') { ctx.moveTo(x, top - 1); ctx.lineTo(x + 5, top - 7); ctx.lineTo(x, top - 13); ctx.lineTo(x - 5, top - 7); }
      else if (m.kind === 'kill') { ctx.moveTo(x - 5, top - 12); ctx.lineTo(x + 5, top - 12); ctx.lineTo(x, top - 2); }
      else { ctx.moveTo(x - 5, top - 2); ctx.lineTo(x + 5, top - 2); ctx.lineTo(x, top - 12); }
      ctx.fill();
    }
    // played part + playhead
    const px = X(this.t);
    ctx.fillStyle = 'rgba(43,228,255,0.06)';
    ctx.fillRect(pad, top, px - pad, bh);
    ctx.fillStyle = '#2be4ff';
    ctx.shadowColor = '#2be4ff'; ctx.shadowBlur = 8;
    ctx.fillRect(px - 1, top - 4, 2, bh + 8);
    ctx.shadowBlur = 0;
    if (this.hover !== null && this.hover >= pad && this.hover <= pad + w) {
      const ht = ((this.hover - pad) / w) * this.T;
      ctx.fillStyle = 'rgba(228,241,250,0.35)';
      ctx.fillRect(this.hover, top, 1, bh);
      const near = this.marks.filter((m) => Math.abs(X(m.t) - this.hover) < 5).slice(0, 2);
      const label = fmtClock(ht) + near.map((m) => `  ${m.kind.toUpperCase()} ${m.node}`).join('');
      ctx.font = '10.5px Cascadia Mono, Consolas, monospace';
      const tw = ctx.measureText(label).width + 10;
      const lx = Math.min(W - tw - 2, Math.max(2, this.hover - tw / 2));
      ctx.fillStyle = 'rgba(5,10,15,0.95)'; ctx.fillRect(lx, 1, tw, 14);
      ctx.fillStyle = '#e4f1fa'; ctx.fillText(label, lx + 5, 11);
    }
  }
}
