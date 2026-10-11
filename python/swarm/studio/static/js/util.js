// Small shared helpers: DOM building, formatting that mirrors the Python reports, toasts, modals, events.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** h('div', {class: 'x', onclick: fn}, 'text', child, [more]) - text children are always escaped. */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'value') el.value = v;
    else if (k === 'checked' || k === 'disabled' || k === 'selected' || k === 'hidden') el[k] = !!v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
export function setChildren(el, ...children) { clear(el); append(el, children); return el; }

export const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
export const lerp = (a, b, k) => a + (b - a) * k;
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Deterministic hash of an integer to [0, 1). */
export function hash01(n, salt = 0) {
  let x = (n * 374761393 + salt * 668265263) | 0;
  x = Math.imul(x ^ (x >>> 13), 1274126177);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

// ------------------------------------------------------------------ events
const handlers = new Map();
export const bus = {
  on(name, fn) { (handlers.get(name) || handlers.set(name, new Set()).get(name)).add(fn); return () => handlers.get(name).delete(fn); },
  emit(name, ...args) { for (const fn of handlers.get(name) || []) { try { fn(...args); } catch (e) { console.error(e); } } },
};

// ------------------------------------------------------------------ formatting
export function fmtClock(t, plus = true) {
  if (!Number.isFinite(t)) return '--:--.---';
  const neg = t < 0; t = Math.abs(t);
  const m = Math.floor(t / 60), s = t - m * 60;
  const txt = `${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(6, '0')}`;
  return (plus ? (neg ? 'T-' : 'T+') : '') + txt;
}

export function isNum(v) { return typeof v === 'number' && Number.isFinite(v); }

/** Python-style "{:.1f}" / "{:.0%}" / "{:.3g}" formatting. */
export function pyfmt(v, fmt) {
  if (v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))) return 'n/a';
  if (v === 'inf' || v === Infinity) return 'never';
  const m = /\{:\.?(\d*)([fg%])\}/.exec(fmt || '{:.3g}');
  const p = m && m[1] !== '' ? +m[1] : 3, kind = m ? m[2] : 'g';
  if (kind === '%') return (v * 100).toFixed(p) + '%';
  if (kind === 'f') return v.toFixed(p);
  return Number(v.toPrecision(Math.max(1, p))).toString();
}

export function fmtRatio(v, b) {
  if (!isNum(v) || !isNum(b)) return '';
  if (b === 0) return v === 0 ? '' : 'inf';
  const r = v / b;
  return r >= 100 ? `${r.toFixed(0)}x` : r >= 10 ? `${r.toFixed(1)}x` : `${r.toFixed(2)}x`;
}

export function si(n, digits = 3) {
  if (!isNum(n)) return 'n/a';
  const a = Math.abs(n);
  for (const [k, s] of [[1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k']]) if (a >= k) return (n / k).toPrecision(digits) + s;
  return Number(n.toPrecision(digits)).toString();
}

export function bytes(n) {
  if (!isNum(n)) return 'n/a';
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(1) + ' GiB';
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(1) + ' MiB';
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + ' KiB';
  return n + ' B';
}

export function bps(n) {
  if (!isNum(n)) return 'auto';
  if (n >= 1e9) return (n / 1e9).toPrecision(3) + ' Gbps';
  if (n >= 1e6) return (n / 1e6).toPrecision(3) + ' Mbps';
  if (n >= 1e3) return (n / 1e3).toPrecision(3) + ' kbps';
  return n + ' bps';
}

// ------------------------------------------------------------------ swarm vocabulary
const PREFIX = {
  'drone-a72': 'HAWK', 'drone-a53': 'RAVEN', 'mcu-m7': 'WASP', 'mcu-m4': 'GNAT',
  'cubesat-obc': 'CUBE', 'sat-payload': 'ATLAS', host: 'BASE',
};
/** Callsign = device class + node id, so "node 2" in a log line is RAVEN-2 on screen. */
export function callsign(device, id) { return `${PREFIX[device] || 'UNIT'}-${id}`; }

export const CLASS_NAMES = ['nothing', 'vehicle', 'building', 'road', 'aircraft'];
export function className(i, names = CLASS_NAMES) { return i >= 0 && i < names.length ? names[i] : `class ${i}`; }

// ------------------------------------------------------------------ toasts & modal
export function toast(msg, kind = 'info', title = '') {
  const el = h('div', { class: `toast ${kind}` }, title ? h('b', {}, title) : null, msg);
  $('#toasts').appendChild(el);
  setTimeout(() => { el.style.transition = 'opacity .4s'; el.style.opacity = '0'; setTimeout(() => el.remove(), 450); },
    kind === 'err' ? 9000 : 4500);
  return el;
}

/** Shows a modal; buttons = [{label, cls, value}]; resolves with the clicked value (null on backdrop click). */
export function modal(title, body, buttons = [{ label: 'CLOSE', value: null }]) {
  const root = $('#modal');
  setChildren($('#modal-title'), title);
  setChildren($('#modal-body'), body);
  return new Promise((resolve) => {
    const done = (v) => { root.classList.remove('show'); root.onclick = null; resolve(v); };
    setChildren($('#modal-foot'), buttons.map((b) => h('button', { class: `btn ${b.cls || ''}`, onclick: () => done(b.value) }, b.label)));
    root.onclick = (e) => { if (e.target === root) done(null); };
    root.classList.add('show');
  });
}

export function download(name, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h('a', { href: url, download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** Minimal Python highlighter for the source viewer (escapes everything, then wraps tokens). */
export function highlightPython(src) {
  const out = h('pre', { class: 'code' });
  const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|\b(def|class|return|if|elif|else|for|while|in|not|and|or|import|from|as|with|try|except|finally|raise|pass|break|continue|lambda|yield|None|True|False|self|is)\b|\b(int|float|str|len|range|min|max|sum|abs|print|isinstance|super|dict|list|tuple|set|sorted|enumerate|zip)\b/g;
  let last = 0, m;
  while ((m = re.exec(src))) {
    if (m.index > last) out.appendChild(document.createTextNode(src.slice(last, m.index)));
    const cls = m[1] ? 'cm' : m[2] ? 'st' : m[3] ? 'kw' : 'nb';
    out.appendChild(h('span', { class: cls }, m[0]));
    last = m.index + m[0].length;
  }
  out.appendChild(document.createTextNode(src.slice(last)));
  return out;
}
