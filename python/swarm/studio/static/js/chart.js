// A small line chart on a 2D canvas: learning curves of the training tab (no libraries, works offline).

export class LineChart {
  /** opts: {yMin, yMax, yFmt(v) -> label, xLabel} */
  constructor(holder, opts = {}) {
    this.opts = { yMin: 0, yMax: 1, yFmt: (v) => `${Math.round(v * 100)}%`, ...opts };
    this.canvas = document.createElement('canvas');
    holder.appendChild(this.canvas);
    this.series = [];
    this.marks = [];
    new ResizeObserver(() => this.draw()).observe(holder);
  }

  /** series: [{name, color: css, points: [[x, y], ...], dash: bool, width}]; marks: [{x, label}].
   *  A long, noisy series is drawn faintly with its moving average on top, so the trend stays readable. */
  set(series, marks = []) { this.series = series; this.marks = marks; this.draw(); }

  draw() {
    const c = this.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const { yMin, yMax, yFmt } = this.opts;
    const pad = { l: 38, r: 10, t: 10, b: 20 };
    let lo = Infinity, hi = -Infinity;  // loops, not Math.max(...points): a long run has tens of thousands
    for (const s of this.series) for (const p of s.points) { if (p[0] < lo) lo = p[0]; if (p[0] > hi) hi = p[0]; }
    const xMax = Math.max(1, Number.isFinite(hi) ? hi : 1);
    const xMin = Math.min(xMax - 1, Number.isFinite(lo) ? lo : 1, 1);
    const X = (x) => pad.l + (x - xMin) / Math.max(1e-9, xMax - xMin) * (w - pad.l - pad.r);
    const Y = (y) => pad.t + (1 - (y - yMin) / (yMax - yMin)) * (h - pad.t - pad.b);
    g.font = '10px ui-monospace, Consolas, monospace';
    g.textBaseline = 'middle';
    for (let k = 0; k <= 4; k++) {
      const v = yMin + (yMax - yMin) * k / 4, y = Y(v);
      g.strokeStyle = k === 0 ? '#22343f' : '#132029';
      g.beginPath(); g.moveTo(pad.l, y); g.lineTo(w - pad.r, y); g.stroke();
      g.fillStyle = '#5d7381'; g.textAlign = 'right'; g.fillText(yFmt(v), pad.l - 5, y);
    }
    const step = Math.max(1, Math.ceil((xMax - xMin) / 8));
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (let x = Math.ceil(xMin); x <= xMax; x += step) g.fillText(String(x), X(x), h - pad.b + 5);
    for (const m of this.marks) {
      g.strokeStyle = 'rgba(255,178,30,.35)'; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(X(m.x), pad.t); g.lineTo(X(m.x), h - pad.b); g.stroke(); g.setLineDash([]);
    }
    const yc = (y) => Y(Math.max(yMin, Math.min(yMax, y)));
    const line = (p) => { g.beginPath(); p.forEach(([x, y], k) => (k ? g.lineTo(X(x), yc(y)) : g.moveTo(X(x), yc(y)))); g.stroke(); };
    for (const s of this.series) {
      const p = s.points.filter((q) => q[1] !== null && Number.isFinite(q[1]));
      if (!p.length) continue;
      g.strokeStyle = s.color; g.setLineDash(s.dash ? [5, 4] : []);
      const smooth = !s.dash && p.length > SMOOTH_FROM ? movingAverage(p, Math.max(3, Math.round(p.length / 40))) : null;
      if (smooth) {
        g.globalAlpha = 0.28; g.lineWidth = 1; line(p);
        g.globalAlpha = 1; g.lineWidth = s.width || 2; line(smooth);
      } else {
        g.lineWidth = s.width || 2; line(p);
      }
      if (p.length === 1 || !s.dash) {
        const [x, y] = (smooth || p)[(smooth || p).length - 1];
        g.fillStyle = s.color; g.beginPath(); g.arc(X(x), yc(y), 2.6, 0, Math.PI * 2); g.fill();
      }
    }
    g.setLineDash([]); g.lineWidth = 1; g.globalAlpha = 1;
  }
}

const SMOOTH_FROM = 80;  // points: from here on a series is drawn smoothed

/** Centred moving average of [[x, y], ...] over ``win`` points (shorter windows at the ends). */
function movingAverage(p, win) {
  const half = Math.floor(win / 2), sum = new Float64Array(p.length + 1);
  p.forEach((q, i) => { sum[i + 1] = sum[i] + q[1]; });
  return p.map((q, i) => {
    const a = Math.max(0, i - half), b = Math.min(p.length, i + half + 1);
    return [q[0], (sum[b] - sum[a]) / (b - a)];
  });
}
