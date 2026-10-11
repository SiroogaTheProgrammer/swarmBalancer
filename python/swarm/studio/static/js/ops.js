// OPS view: plays a run back - 3D theatre, timeline, ORBAT, inspector, sensor feed, event log, live metrics.

import { api } from './api.js';
import { Playback } from './playback.js';
import { COL, CONTACT_COLOR, Theatre } from './theatre.js';
import { Timeline } from './timeline.js';
import { $, $$, bus, bytes, callsign, className, clear, fmtClock, fmtRatio, h, isNum, pyfmt, setChildren, si, toast } from './util.js';

const css = (c, a = 1) => `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${a})`;

export class Ops {
  constructor() {
    this.theatre = new Theatre($('#gl'), $('#labels'));
    this.timeline = new Timeline($('#timeline'), (t) => this.seek(t));
    this.run = null; this.pbs = []; this.pb = null; this.si = 0;
    this.t = 0; this.playing = false; this.speed = 1; this.loop = false;
    this.sel = null; this.last = performance.now(); this.lastPanel = 0; this.lastMap = 0;
    this.frameCache = new Map(); this.sensorKey = ''; this.sensorBusy = false;
    this.theatre.onPick = (s) => this.select(s);
    this.theatre.onMode = (m) => $$('[data-cam]').forEach((b) => b.classList.toggle('on', b.dataset.cam === m));
    $('#tl-play').onclick = () => this.toggle();
    $('#tl-start').onclick = () => this.seek(0);
    $('#tl-prev').onclick = () => this.jump(-1);
    $('#tl-next').onclick = () => this.jump(1);
    $('#tl-speed').onchange = (e) => { this.speed = +e.target.value; };
    $('#tl-loop').onchange = (e) => { this.loop = e.target.checked; };
    $$('[data-cam]').forEach((b) => { b.onclick = () => this.setCam(b.dataset.cam); });
    $('#legend-btn').onclick = () => { $('#legend').classList.toggle('show'); $('#legend-btn').classList.toggle('on'); };
    window.addEventListener('keydown', (e) => this.key(e));
    requestAnimationFrame((now) => this.tick(now));
  }

  setCam(mode) {
    if (mode === 'asset' && (!this.sel || this.sel.type !== 'node')) { toast('Select an asset first (click a drone or an ORBAT row).', 'warn'); return; }
    this.theatre.setMode(mode);
  }

  // ------------------------------------------------------------------ loading
  async loadRun(id, { strategy = null, autoplay = true } = {}) {
    const run = await api(`runs/${id}`);
    this.run = run;
    this.pbs = [];
    this.frameCache.clear();
    const names = run.strategies.map((s) => s.name);
    let i = strategy !== null && names.includes(strategy) ? names.indexOf(strategy) : -1;
    if (i < 0) i = names.findIndex((n) => !['local', 'central/none', 'central/downscale', 'central/features', 'striped'].includes(n));
    if (i < 0) i = names.indexOf('striped');
    if (i < 0) i = 0;
    this.sel = null;
    this.t = 0;
    this.setStrategy(i, true);
    $('#ops-empty').classList.add('hidden');
    $('#st-op').textContent = run.scenario.name.toUpperCase();
    $('#ops-runid').textContent = run.id;
    this.renderMission();
    if (autoplay) this.play(); else this.pause();
    bus.emit('run-loaded', run);
    return run;
  }

  playback(i) {
    if (!this.pbs[i]) this.pbs[i] = new Playback(this.run, i);
    return this.pbs[i];
  }

  setStrategy(i, fresh = false) {
    this.si = i;
    this.pb = this.playback(i);
    if (fresh) this.theatre.load(this.run, this.pb); else this.theatre.setPlayback(this.pb);
    this.timeline.setData(this.pb);
    this.renderStrategyTabs();
    this.renderDesign();
    this.renderOrbat(true);
    this.buildLog();
    this.sensorKey = '';
    this.updatePanels(true);
  }

  // ------------------------------------------------------------------ time
  play() {
    if (!this.run) return;
    if (this.t >= this.pb.T - 1e-6) this.t = 0;
    this.playing = true;
    $('#tl-play').innerHTML = '&#10074;&#10074;';
  }

  pause() { this.playing = false; $('#tl-play').innerHTML = '&#9654;'; }
  toggle() { if (this.playing) this.pause(); else this.play(); }

  seek(t) {
    if (!this.pb) return;
    this.t = Math.min(Math.max(0, t), this.pb.T);
    this.updatePanels(true);
  }

  jump(dir) {
    if (!this.pb) return;
    const marks = this.timeline.marks.map((m) => m.t);
    const next = dir > 0 ? marks.find((m) => m > this.t + 0.02) : [...marks].reverse().find((m) => m < this.t - 0.25);
    this.seek(next !== undefined ? Math.max(0, next - 0.6) : dir > 0 ? this.pb.T : 0);
  }

  tick(now) {
    if (this.stopped) return;
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    if (this.playing && this.pb) {
      this.t += dt * this.speed;
      if (this.t >= this.pb.T) {
        if (this.loop) this.t = 0;
        else { this.t = this.pb.T; this.pause(); }
      }
    }
    try {
      this.theatre.frame(this.t, dt, this.speed);
    } catch (e) {
      console.error(e);
    }
    if (this.pb) {
      const clock = fmtClock(this.t);
      $('#st-clock').textContent = clock;
      setChildren($('#tl-time'), clock, h('small', {}, '/ ' + fmtClock(this.pb.T, false)));
      this.timeline.set(this.t);
      if (now - this.lastPanel > 120) { this.lastPanel = now; this.updatePanels(false); }
      if (now - this.lastMap > 70) { this.lastMap = now; this.minimap(); }
    }
    requestAnimationFrame((n) => this.tick(n));
  }

  key(e) {
    if (!$('#view-ops').classList.contains('active') || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    const k = e.key;
    if (k === ' ') { e.preventDefault(); this.toggle(); }
    else if (k === 'ArrowRight') this.seek(this.t + (e.shiftKey ? 1 : 0.1));
    else if (k === 'ArrowLeft') this.seek(this.t - (e.shiftKey ? 1 : 0.1));
    else if (k === ']') this.jump(1);
    else if (k === '[') this.jump(-1);
    else if (k === 'Home') this.seek(0);
    else if (k === 'End') this.seek(this.pb ? this.pb.T : 0);
    else if (k === 'Escape') this.select(null);
    else if (k === 'v' || k === 'V') this.setCam('swarm');
    else if (k === 't' || k === 'T') this.setCam('top');
    else if (k === 'f' || k === 'F') this.setCam(this.sel && this.sel.type === 'node' ? 'asset' : 'free');
    else if (k === 'l' || k === 'L') $('#legend-btn').click();
    else if (/^[1-9]$/.test(k) && this.run && +k <= this.run.strategies.length) this.setStrategy(+k - 1);
  }

  // ------------------------------------------------------------------ selection
  select(sel) {
    this.sel = sel;
    this.theatre.selected = sel && sel.type === 'node' ? sel.id : null;
    this.theatre.selectedContact = sel && sel.type === 'contact' ? sel.id : null;
    if (!sel && this.theatre.mode === 'asset') this.theatre.setMode('swarm');
    this.sensorKey = '';
    this.renderOrbat(true);
    this.updatePanels(true);
  }

  // ------------------------------------------------------------------ panels
  renderMission() {
    const sc = this.run.scenario, r = this.run;
    const faults = (sc.faults.kill || []).map(([w, t]) => `KILL ${w === 'leader' ? 'LEADER' : callsign(r.profiles[w].name, w)} @${t}s`)
      .concat((sc.faults.revive || []).map(([w, t]) => `REVIVE ${callsign(r.profiles[w].name, w)} @${t}s`));
    setChildren($('#mission-card'),
      h('div', { class: 't' }, sc.name.toUpperCase()),
      h('div', { class: 'd' }, sc.description || 'no description'),
      h('div', { class: 'kv', style: { marginTop: '7px' } },
        h('span', { class: 'k' }, 'Brain'), h('span', { class: 'v', title: r.brain.description }, `${r.brain.kind}${r.brain.model_path ? ' · ' + r.brain.model_path.split('/').pop() : ''}`),
        h('span', { class: 'k' }, 'Link'), h('span', { class: 'v' }, `${si(r.resolved.bps)}bps · ${(r.resolved.latency_s * 1e3).toFixed(1)} ms · loss ${(sc.link.loss * 100).toFixed(1)}%`),
        h('span', { class: 'k' }, 'Camera'), h('span', { class: 'v' }, `${sc.camera.size}px · ${sc.camera.fps} fps · obj ${(sc.camera.p_object * 100).toFixed(0)}%`),
        h('span', { class: 'k' }, 'Faults'), h('span', { class: 'v', title: faults.join(', ') }, faults.join(', ') || 'none'),
        h('span', { class: 'k' }, 'Checks'), h('span', { class: 'v ' + (r.checks.every((c) => c.passed) ? 'good' : 'bad') },
          r.checks.length ? `${r.checks.filter((c) => c.passed).length}/${r.checks.length} pass` : 'none')));
  }

  renderStrategyTabs() {
    const r = this.run, sc = r.scenario;
    setChildren($('#hud-tl'),
      h('div', { class: 'op' }, `OP ${sc.name.toUpperCase()}`),
      h('div', { class: 'sub' }, `${r.profiles.length} ASSETS · SEED ${sc.seed} · DETERMINISTIC SIMULATED TIME${this.pb.truncated ? ' · TRACE TRUNCATED' : ''}`),
      h('div', { class: 'strats pe' }, r.strategies.map((s, i) => h('button', {
        class: `btn small ${i === this.si ? 'on' : ''}`, title: s.note, onclick: () => this.setStrategy(i),
      }, `${i + 1} ${s.name.toUpperCase()}${s.name === sc.baseline ? ' (BASE)' : ''}`))));
  }

  renderDesign() {
    const r = this.run;
    const base = r.strategies.find((s) => s.name === r.scenario.baseline) || r.strategies[0];
    const rows = [['recall', 'RECALL', '{:.0%}', 1], ['latency_p50_ms', 'P50 MS', '{:.1f}', -1], ['kbps', 'KBIT/S', '{:.1f}', -1],
      ['gmac_ps', 'GMAC/S', '{:.3f}', -1], ['busy_max', 'MAX CPU', '{:.0%}', -1], ['work_retained', 'RETAINED', '{:.0%}', 1]];
    const s = r.strategies[this.si];
    setChildren($('#design-card'),
      h('div', { style: { fontWeight: 700, letterSpacing: '.1em', color: 'var(--bright)' } }, s.name.toUpperCase()),
      h('div', { class: 'muted', style: { fontSize: '11px', margin: '3px 0 8px' } }, s.note || ''),
      h('div', { class: 'kv' }, rows.flatMap(([k, label, fmt, dir]) => {
        const v = s.metrics[k], b = base.metrics[k];
        if (v === null || v === undefined) return [];
        const ratio = s === base ? '' : fmtRatio(v, b);
        const better = isNum(v) && isNum(b) && s !== base ? (dir > 0 ? v > b * 1.02 : v < b * 0.98) : null;
        const worse = isNum(v) && isNum(b) && s !== base ? (dir > 0 ? v < b * 0.98 : v > b * 1.02) : null;
        return [h('span', { class: 'k' }, label), h('span', { class: 'v' }, pyfmt(v, fmt),
          ratio ? h('span', { class: better ? 'good' : worse ? 'bad' : 'dimc', style: { marginLeft: '6px', fontSize: '10px' } }, ratio) : null)];
      })));
  }

  renderOrbat(force) {
    if (!this.run) return;
    const ul = $('#orbat');
    if (force || ul.children.length !== this.run.profiles.length) {
      clear(ul);
      this.run.profiles.forEach((p, i) => {
        ul.appendChild(h('li', { onclick: () => this.select({ type: 'node', id: i }), title: p.description },
          h('div', { class: 'st' }),
          h('div', {}, h('div', { class: 'cs' }, callsign(p.name, i)), h('div', { class: 'dv' }, p.name)),
          h('div', { class: 'right' }, h('b', {}, ''), h('div', { class: 'cpu' }, h('i'))),
        ));
      });
      $('#orbat-count').textContent = `${this.run.profiles.length} ASSETS`;
    }
    const t = this.t;
    [...ul.children].forEach((li, i) => {
      const alive = this.pb.aliveAt(i, t), lead = this.pb.leaderAt(i, t), cam = this.pb.camAt(i, t);
      const cpu = this.pb.cpuAt(i, t, 1);
      li.className = (lead ? 'leader ' : '') + (!alive ? 'dead ' : '') + (this.sel && this.sel.type === 'node' && this.sel.id === i ? 'sel' : '');
      li.querySelector('.right b').textContent = !alive ? 'OFFLINE' : `${lead ? 'LEAD' : 'NODE'}${cam ? ' · CAM' : ''} ${Math.round(cpu.util * 100)}%`;
      const bar = li.querySelector('.cpu');
      bar.className = 'cpu' + (cpu.util > 0.9 ? ' over' : cpu.util > 0.6 ? ' hot' : '');
      bar.firstChild.style.width = `${Math.round(cpu.util * 100)}%`;
    });
  }

  buildLog() {
    const r = this.run, pb = this.pb;
    const cs = (i) => (i >= 0 && i < r.profiles.length ? callsign(r.profiles[i].name, i) : `node ${i}`);
    const rows = [];
    for (const m of pb.marks) {
      const text = { kill: `${cs(m.node)} OFFLINE - signal lost`, revive: `${cs(m.node)} back ONLINE`, leader: `${cs(m.node)} assumes COMMAND`,
        suspect: `${cs(m.by)} suspects ${cs(m.node)} (heartbeats missing)`, readmit: `${cs(m.by)} re-admits ${cs(m.node)}` }[m.kind];
      if (text) rows.push([m.t, m.kind, text]);
    }
    for (const [t, text] of pb.logs) {
      if (/rotation now|camera (ON|OFF)|demoted|leads the stripe/.test(text)) {
        rows.push([t, 'cam', text.replace(/node (\d+)( \([^)]*\))?/g, (_, d) => cs(+d))]);
      }
    }
    const rej = pb.rejects.length;
    rows.sort((a, b) => a[0] - b[0]);
    const box = $('#eventlog');
    clear(box);
    this.logRows = rows.map(([t, kind, text]) => {
      const el = h('div', { class: kind, onclick: () => this.seek(t), title: text }, h('span', { class: 't' }, fmtClock(t)), text);
      box.appendChild(el);
      return { t, el };
    });
    if (rej) box.appendChild(h('div', { class: 'dimc' }, h('span', { class: 't' }, 'note'), `${rej} frames rejected by full CPU queues during the run`));
    $('#log-count').textContent = `${rows.length} EVENTS`;
    this.logCursor = -2;
  }

  updatePanels(force) {
    if (!this.pb) return;
    this.renderOrbat(false);
    // event log: dim the future, keep the latest past event in view
    let k = -1;
    this.logRows.forEach((r, i) => { const fut = r.t > this.t; r.el.classList.toggle('future', fut); if (!fut) k = i; });
    if (k !== this.logCursor) {
      this.logCursor = k;
      const box = $('#eventlog');
      if (k >= 0) {
        const el = this.logRows[k].el, top = el.offsetTop - box.offsetTop;
        if (top < box.scrollTop || top > box.scrollTop + box.clientHeight - el.offsetHeight) box.scrollTop = Math.max(0, top - box.clientHeight / 2);
      } else box.scrollTop = 0;
    }
    this.hudMetrics();
    this.inspector();
    this.sensor(force);
  }

  hudMetrics() {
    const c = this.pb.countsAt(this.t);
    const m = (l, v, sub = '') => h('div', { class: 'm' }, h('div', { class: 'l' }, l), h('div', { class: 'v' }, v, sub ? h('small', {}, ' ' + sub) : null));
    setChildren($('#hud-bl'),
      m('FRAMES', `${c.proc}`, `/ ${c.cap}`),
      m('OBJECTS', `${c.rep}`, `/ ${c.obj} · ${isNum(c.recall) ? Math.round(c.recall * 100) + '%' : '--'}`),
      m('LATENCY P50', isNum(c.p50) ? (c.p50 < 1 ? c.p50.toFixed(2) : c.p50.toFixed(1)) : '--', 'ms'),
      m('RADIO', c.kbps.toFixed(1), 'kbit/s'),
      m('CHANNEL', `${Math.round(c.chan * 100)}%`, ''),
      m('COMPUTE', c.gmacs.toFixed(3), 'GMAC/s'));
  }

  inspector() {
    const box = $('#inspector'), pb = this.pb, r = this.run, t = this.t;
    if (!this.sel) { setChildren($('#insp-hdr'), 'Inspector'); setChildren(box, h('div', { class: 'muted' }, 'Click an asset or a track in the theatre (or an ORBAT row).')); return; }
    if (this.sel.type === 'node') {
      const i = this.sel.id, p = r.profiles[i];
      const alive = pb.aliveAt(i, t), lead = pb.leaderAt(i, t), cam = pb.camAt(i, t), cpu = pb.cpuAt(i, t, 1);
      const sus = pb.suspicionsAt(t).filter(([by]) => by === i).map(([, n]) => callsign(r.profiles[n].name, n));
      const work = cpu.cur ? (cpu.cur.tag === 'brain' ? `INFERENCE frame #${cpu.cur.fid}` : `${cpu.cur.tag.toUpperCase()} frame #${cpu.cur.fid}`) : 'idle';
      setChildren($('#insp-hdr'), `Asset ${callsign(p.name, i)}`);
      setChildren(box, h('div', { class: 'kv' },
        h('span', { class: 'k' }, 'Status'), h('span', { class: 'v ' + (alive ? 'good' : 'bad') }, alive ? 'ONLINE' : 'OFFLINE'),
        h('span', { class: 'k' }, 'Role'), h('span', { class: 'v' }, !alive ? '-' : lead ? 'SWARM LEADER' : 'MEMBER'),
        h('span', { class: 'k' }, 'Camera'), h('span', { class: 'v' }, cam ? 'ON' : 'OFF'),
        h('span', { class: 'k' }, 'Device'), h('span', { class: 'v', title: p.description }, p.name),
        h('span', { class: 'k' }, 'CPU'), h('span', { class: 'v' }, `${p.cpu_mhz} MHz · ${si(p.cpu_mhz * 1e6 * p.macs_per_cycle)}MAC/s`),
        h('span', { class: 'k' }, 'RAM'), h('span', { class: 'v' }, bytes(p.ram_bytes)),
        h('span', { class: 'k' }, 'Now'), h('span', { class: 'v' }, work),
        h('span', { class: 'k' }, 'Load 1 s'), h('span', { class: 'v ' + (cpu.util > 0.9 ? 'bad' : cpu.util > 0.6 ? 'warnc' : '') }, `${(cpu.util * 100).toFixed(1)}% · queue ${cpu.queued}`),
        h('span', { class: 'k' }, 'Work done'), h('span', { class: 'v' }, `${si(cpu.macs)}MAC`),
        h('span', { class: 'k' }, 'Suspects'), h('span', { class: 'v ' + (sus.length ? 'bad' : '') }, sus.join(', ') || 'nobody'),
      ));
      return;
    }
    const f = pb.frames.get(this.sel.id);
    if (!f) { setChildren(box, h('div', { class: 'muted' }, 'track not in this strategy')); return; }
    const state = pb.contactState(f, t);
    const cs = (n) => callsign(r.profiles[n].name, n);
    setChildren($('#insp-hdr'), `Track TRK-${String(f.fid).padStart(4, '0')}`);
    setChildren(box, h('div', { class: 'kv' },
      h('span', { class: 'k' }, 'State'), h('span', { class: 'v', style: { color: css(CONTACT_COLOR[state] || COL.gray) } }, state.toUpperCase()),
      h('span', { class: 'k' }, 'Ground truth'), h('span', { class: 'v' }, className(f.label).toUpperCase()),
      h('span', { class: 'k' }, 'Sighted'), h('span', { class: 'v' }, `${fmtClock(f.t)} by ${cs(f.node)}`),
      h('span', { class: 'k' }, 'Classified'), h('span', { class: 'v' }, f.inf ? `${className(f.inf.pred).toUpperCase()} ${(f.inf.conf * 100).toFixed(0)}% on ${cs(f.inf.node)} +${((f.inf.t - f.t) * 1e3).toFixed(1)} ms` : 'never (dropped)'),
      h('span', { class: 'k' }, 'Leader knew'), h('span', { class: 'v' }, f.learn ? `${cs(f.learn.node)} +${((f.learn.t - f.t) * 1e3).toFixed(1)} ms` : 'never'),
    ));
  }

  // ------------------------------------------------------------------ sensor feed (the actual recorded frame)
  async sensor(force) {
    const pb = this.pb, r = this.run;
    let fid = -1, src = '', node = -1;
    if (this.sel && this.sel.type === 'contact') { fid = this.sel.id; const f = pb.frames.get(fid); node = f ? f.node : -1; }
    else {
      node = this.sel && this.sel.type === 'node' ? this.sel.id : pb.lead.findIndex((_, i) => pb.camAt(i, this.t));
      if (node < 0) node = [...Array(pb.n).keys()].find((i) => pb.camAt(i, this.t)) ?? -1;
      if (node >= 0 && pb.camAt(node, this.t)) fid = pb.lastFrame(node, this.t);
    }
    src = node >= 0 ? callsign(r.profiles[node].name, node) : '';
    $('#sensor-src').textContent = src ? `EO/IR ${src}` : '';
    const key = `${r.id}:${this.si}:${fid}`;
    if (key === this.sensorKey && !force) { this._sensorOverlay(fid); return; }
    if (fid < 0) { this.sensorKey = key; this._drawSensor(null); this._sensorOverlay(-1); return; }
    if (this.frameCache.has(key)) { this.sensorKey = key; this._drawSensor(this.frameCache.get(key)); this._sensorOverlay(fid); return; }
    if (this.sensorBusy) return;
    this.sensorBusy = true;
    try {
      const img = await api(`runs/${r.id}/frame?s=${this.si}&fid=${fid}`);
      const raw = Uint8Array.from(atob(img.data), (c) => c.charCodeAt(0));
      const frame = { w: img.w, h: img.h, raw };
      if (this.frameCache.size > 400) this.frameCache.clear();
      this.frameCache.set(key, frame);
      if (this.run && this.run.id === r.id) { this.sensorKey = key; this._drawSensor(frame); this._sensorOverlay(fid); }
    } catch {
      this.sensorKey = key;
      this._drawSensor(null);
    } finally {
      this.sensorBusy = false;
    }
  }

  _drawSensor(frame) {
    const c = $('#sensor-canvas'), ctx = c.getContext('2d');
    $('#sensor-nosig').classList.toggle('hidden', !!frame);
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, c.width, c.height);
    if (!frame) return;
    const off = document.createElement('canvas');
    off.width = frame.w; off.height = frame.h;
    const octx = off.getContext('2d'), img = octx.createImageData(frame.w, frame.h);
    for (let i = 0; i < frame.raw.length; i++) {
      const v = frame.raw[i];
      img.data[i * 4] = v * 0.55; img.data[i * 4 + 1] = Math.min(255, v * 1.05 + 8); img.data[i * 4 + 2] = v * 0.62; img.data[i * 4 + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    const s = Math.min(c.width, c.height);
    ctx.drawImage(off, (c.width - s) / 2, (c.height - s) / 2, s, s);
    ctx.fillStyle = 'rgba(0,0,0,0.18)';
    for (let y = 0; y < c.height; y += 3) ctx.fillRect(0, y, c.width, 1);
  }

  _sensorOverlay(fid) {
    const ov = $('#sensor-ov');
    if (fid < 0) { clear(ov); return; }
    const f = this.pb.frames.get(fid), r = this.run, t = this.t;
    const cs = (n) => callsign(r.profiles[n].name, n);
    const res = f && f.inf && f.inf.t <= t ? `${className(f.inf.pred).toUpperCase()} ${(f.inf.conf * 100).toFixed(0)}% · ${cs(f.inf.node)}` : 'CLASSIFYING...';
    setChildren(ov,
      h('div', { class: 'tl' }, `FRAME #${fid}`, h('br'), f ? fmtClock(f.t) : ''),
      h('div', { class: 'tr' }, f ? cs(f.node) : '', h('br'), `${r.scenario.camera.size}px > ${(this.pb.s.frames.shape || []).join('x')}`),
      h('div', { class: 'bl' }, 'BRAIN: ' + res),
      h('div', { class: 'br' }, f ? `GT ${className(f.label).toUpperCase()}` : ''),
      h('div', { class: 'xh' }));
  }

  // ------------------------------------------------------------------ minimap
  minimap() {
    const c = $('#minimap'), ctx = c.getContext('2d'), th = this.theatre, pb = this.pb;
    if (!th.kin) return;
    const W = c.width, H = c.height, t = this.t;
    const ext = th.kin.extent * 1.05, s = W / (2 * ext);
    const X = (x) => W / 2 + x * s, Y = (z) => H / 2 + z * s;
    ctx.fillStyle = 'rgba(2,6,9,1)'; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(43,228,255,0.07)'; ctx.lineWidth = 1;
    for (let g = -ext; g <= ext; g += ext / 4) { ctx.beginPath(); ctx.moveTo(X(g), 0); ctx.lineTo(X(g), H); ctx.moveTo(0, Y(g)); ctx.lineTo(W, Y(g)); ctx.stroke(); }
    ctx.strokeStyle = 'rgba(43,228,255,0.3)'; ctx.setLineDash([6, 6]); ctx.beginPath();
    th.kin.routeLine().forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[2])) : ctx.moveTo(X(p[0]), Y(p[2]))));
    ctx.stroke(); ctx.setLineDash([]);
    for (const f of th.visibleContacts || []) {
      ctx.fillStyle = css(CONTACT_COLOR[f.state] || COL.gray, f.state === 'track' ? 1 : 0.7);
      ctx.fillRect(X(f.x) - 2, Y(f.z) - 2, 4, 4);
    }
    const fp = th.kin.footprint / 2;
    th.poses.forEach(({ p, h: hd }, i) => {
      if (pb.camAt(i, t)) {
        ctx.strokeStyle = 'rgba(43,228,255,0.6)';
        ctx.beginPath();
        const rx = -hd[1], rz = hd[0];
        [[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]].forEach(([a, b], k) => {
          const x = X(p[0] + hd[0] * fp * a + rx * fp * 0.78 * b), y = Y(p[2] + hd[1] * fp * a + rz * fp * 0.78 * b);
          if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        });
        ctx.stroke();
      }
    });
    th.poses.forEach(({ p, h: hd }, i) => {
      const alive = pb.aliveAt(i, t), lead = pb.leaderAt(i, t);
      ctx.save();
      ctx.translate(X(p[0]), Y(p[2]));
      ctx.rotate(Math.atan2(hd[1], hd[0]));
      ctx.fillStyle = !alive ? '#ff3b4d' : lead ? '#ffb627' : '#2be4ff';
      ctx.beginPath(); ctx.moveTo(9, 0); ctx.lineTo(-6, 5); ctx.lineTo(-3, 0); ctx.lineTo(-6, -5); ctx.closePath(); ctx.fill();
      ctx.restore();
      if (this.sel && this.sel.type === 'node' && this.sel.id === i) { ctx.strokeStyle = '#fff'; ctx.strokeRect(X(p[0]) - 10, Y(p[2]) - 10, 20, 20); }
    });
    const tg = th.cam.target;
    ctx.strokeStyle = 'rgba(228,241,250,0.5)';
    ctx.beginPath(); ctx.arc(X(tg[0]), Y(tg[2]), 7, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = '#5f7787'; ctx.font = '18px Cascadia Mono, Consolas, monospace';
    ctx.fillText(`${Math.round(ext * 2)} m`, 10, H - 10);
    ctx.fillText('N', W - 24, 24);
  }
}
