// MISSION PLANNER: edit a scenario (the battery's Scenario as JSON), validate it live, save it as a runnable
// preset (scenarios/studio/ or the battery), or EXECUTE it - simulated in a worker, played back in OPS.

import { api } from './api.js';
import { $, bps, bus, bytes, callsign, clear, download, h, isNum, modal, setChildren, si, toast } from './util.js';

const BUILTIN = ['local', 'central/none', 'central/downscale', 'central/features', 'striped'];

export function blankScenario() {
  return {
    name: 'op_new_mission', description: 'Describe the question this mission answers.',
    swarm: { leader: 'drone-a72', worker: 'drone-a53', size: 5, nodes: null },
    camera: { fps: 10, size: 128, p_object: 0.3, striped_fps: 'match' },
    link: { bps: null, latency_s: null, loss: 0.01, qos: true, hb_interval: 0.5, missed_beats: 3 },
    brain: { kind: 'oracle', model: 'models/tiny_cnn_int8.swm', macs: 401568, accuracy: 0.87, input_size: 32, frontend_fraction: 0.92 },
    faults: { kill: [['leader', 10]], revive: [] }, duration: 30, seed: 0, strategies: [...BUILTIN], baseline: 'local',
    queue_limit: 4, confidence_threshold: 0.5, include_leader: true, stress: { devices: null, frames: 50 }, checks: [], tags: [],
  };
}

const clone = (o) => JSON.parse(JSON.stringify(o));

export class Planner {
  constructor(app) {
    this.app = app;
    this.catalog = null;
    this.sc = blankScenario();
    this.th = { terrain: 'auto', formation: 'auto', route: 'auto', altitude_m: 0, spacing_m: 0, speed_mps: 0, area_m: 1600 };
    this.preset = null;
    this.valid = null;
    this.timer = 0;
    $('#preset-filter').oninput = () => this.renderPresets();
    $('#pl-new').onclick = () => this.load(null);
    $('#pl-exec').onclick = () => this.execute();
    $('#pl-save').onclick = () => this.save('studio');
    $('#pl-battery').onclick = () => this.save('battery');
    $('#pl-export').onclick = () => download(`${this.sc.name}.json`, JSON.stringify({ scenario: this.sc, theatre: this.th }, null, 2));
    $('#pl-delete').onclick = () => this.remove();
  }

  setCatalog(cat) {
    this.catalog = cat;
    this.renderPresets();
    if (!this.rendered) { this.rendered = true; this.renderForm(); }
  }

  // ------------------------------------------------------------------ presets
  renderPresets() {
    if (!this.catalog) return;
    const q = $('#preset-filter').value.trim().toLowerCase();
    const ul = clear($('#preset-list'));
    const groups = [['studio', 'STUDIO MISSIONS'], ['battery', 'STANDARDIZED BATTERY']];
    let count = 0;
    for (const [g, title] of groups) {
      const items = this.catalog.presets.filter((p) => p.group === g && (!q || `${p.name} ${p.description || ''} ${(p.tags || []).join(' ')}`.toLowerCase().includes(q)));
      if (!items.length) continue;
      ul.appendChild(h('li', { class: 'grp' }, title));
      for (const p of items) {
        count++;
        ul.appendChild(h('li', {
          class: (this.preset && this.preset.id === p.id ? 'sel ' : '') + (p.error ? 'err' : ''), title: p.error || p.file,
          onclick: () => (p.error ? toast(p.error, 'err', p.file) : this.load(p)),
        }, h('div', { class: 'n' }, p.name), h('div', { class: 'd' }, p.error || p.description || '')));
      }
    }
    if (this.catalog.presets_pending) ul.appendChild(h('li', { class: 'grp' }, 'INDEXING PRESET FILES...'));
    $('#preset-count').textContent = `${count}`;
  }

  load(p) {
    this.preset = p;
    this.sc = p ? clone(p.scenario) : blankScenario();
    if (p && !p.editable) this.sc.name = p.name;
    this.th = Object.assign({ terrain: 'auto', formation: 'auto', route: 'auto', altitude_m: 0, spacing_m: 0, speed_mps: 0, area_m: 1600 }, (p && p.theatre) || {});
    if (!this.sc.swarm.nodes) {
      const s = this.sc.swarm;
      this.sc.swarm.nodes = [s.leader, ...Array(Math.max(0, s.size - 1)).fill(s.worker)];
    }
    this.renderPresets();
    this.renderForm();
    $('#pl-delete').disabled = !(p && p.editable);
  }

  // ------------------------------------------------------------------ form
  changed(structural = false) {
    if (structural) this.renderForm();
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.check(), 250);
    this.opord();
  }

  payload() {
    const sc = clone(this.sc);
    const nodes = sc.swarm.nodes || [];
    if (nodes.length && nodes.slice(1).every((n) => n === nodes[1])) sc.swarm = { leader: nodes[0], worker: nodes[1] || nodes[0], size: nodes.length, nodes: null };
    if (sc.tags && !Array.isArray(sc.tags)) sc.tags = [];
    return sc;
  }

  async check() {
    try {
      await api('scenario/check', { method: 'POST', body: { scenario: this.payload(), theatre: this.th } });
      this.valid = null;
      setChildren($('#pl-valid'), h('span', { class: 'badge pass' }, 'VALID'), ' ready to execute');
      $('#opord-state').textContent = 'READY';
    } catch (e) {
      this.valid = e.message;
      setChildren($('#pl-valid'), h('span', { class: 'badge fail' }, 'INVALID'), ' ', e.message);
      $('#opord-state').textContent = 'NOT READY';
    }
    $('#pl-exec').disabled = !!this.valid;
  }

  renderForm() {
    if (!this.catalog) return;
    const sc = this.sc, cat = this.catalog;
    if (!sc.swarm.nodes || !sc.swarm.nodes.length) {
      sc.swarm.nodes = [sc.swarm.leader, ...Array(Math.max(0, sc.swarm.size - 1)).fill(sc.swarm.worker)];
    }
    const f = $('#planner-form');
    const scroll = f.scrollTop;
    clear(f);
    const set = (path, v, structural = false) => {
      const keys = path.split('.');
      let o = sc;
      for (const k of keys.slice(0, -1)) o = o[k];
      o[keys[keys.length - 1]] = v;
      this.changed(structural);
    };
    const num = (label, path, opts = {}) => {
      const v = path.split('.').reduce((o, k) => o[k], sc);
      return h('label', { class: 'f' }, label, h('input', {
        type: 'number', value: v === null || v === undefined ? '' : opts.scale ? +(v * opts.scale).toPrecision(6) : v,
        step: opts.step || 'any', min: opts.min, max: opts.max, placeholder: opts.placeholder || '',
        oninput: (e) => set(path, e.target.value === '' ? (opts.nullable ? null : 0) : +e.target.value / (opts.scale || 1)),
      }), opts.hint ? h('span', { class: 'hint' }, opts.hint) : null);
    };
    const card = (title, right, ...body) => h('div', { class: 'card' }, h('div', { class: 'hdr' }, title, right ? h('span', { class: 'r' }, right) : null), h('div', { class: 'cbody' }, ...body));

    // identification
    f.appendChild(card('1 · Identification', this.preset ? this.preset.file : 'unsaved',
      h('div', { class: 'cols3' },
        h('label', { class: 'f', style: { gridColumn: 'span 2' } }, 'Operation name (file name)', h('input', { value: sc.name, oninput: (e) => set('name', e.target.value.trim()) })),
        h('label', { class: 'f' }, 'Tags', h('input', { value: (sc.tags || []).join(', '), placeholder: 'leader, survivability', oninput: (e) => set('tags', e.target.value.split(',').map((s) => s.trim()).filter(Boolean)) }))),
      h('label', { class: 'f' }, 'Question this mission answers', h('textarea', { rows: 2, oninput: (e) => set('description', e.target.value.replace(/\s+/g, ' ')) }, sc.description)),
      h('div', { class: 'cols4' }, num('Duration (s)', 'duration', { min: 0.5, max: 900 }), num('Seed', 'seed', { step: 1, min: 0, hint: 'same seed -> same run' }),
        num('Queue limit', 'queue_limit', { step: 1, min: 1, max: 64 }), num('Report threshold', 'confidence_threshold', { step: 0.05, min: 0, max: 1 }))));

    // ORBAT
    const devs = cat.devices.filter((d) => d.name !== 'host');
    const nodes = sc.swarm.nodes;
    const devOpts = (v) => devs.map((d) => h('option', { value: d.name, selected: d.name === v }, d.name));
    f.appendChild(card('2 · ORBAT - order of battle', `${nodes.length} assets`,
      h('table', { class: 'grid nodes-table' },
        h('tr', {}, h('th', {}, '#'), h('th', {}, 'Callsign'), h('th', {}, 'Device'), h('th', {}, 'Compute'), h('th', {}, 'RAM'), h('th', {}, 'Radio'), h('th', {})),
        nodes.map((n, i) => {
          const d = devs.find((x) => x.name === n) || {};
          return h('tr', {},
            h('td', { class: 'dimc' }, i), h('td', {}, h('b', {}, callsign(n, i))),
            h('td', {}, h('select', { onchange: (e) => { nodes[i] = e.target.value; this.changed(true); } }, devOpts(n))),
            h('td', { class: 'num' }, `${si(d.gmacs * 1e9)}MAC/s`), h('td', { class: 'num' }, bytes(d.ram_bytes)), h('td', { class: 'num' }, bps(d.radio_bps)),
            h('td', {}, nodes.length > 1 ? h('button', { class: 'btn small danger', onclick: () => { nodes.splice(i, 1); this._fixFaults(); this.changed(true); } }, 'x') : null));
        })),
      h('div', { class: 'btnrow' },
        h('button', { class: 'btn small', onclick: () => { if (nodes.length < 24) { nodes.push(nodes[nodes.length - 1] || 'drone-a53'); this.changed(true); } } }, '+ add asset'),
        ...['drone-a53', 'mcu-m4', 'cubesat-obc'].map((d) => h('button', { class: 'btn small', onclick: () => { if (nodes.length < 24) { nodes.push(d); this.changed(true); } } }, `+ ${d}`)),
        h('span', { class: 'muted', style: { marginLeft: 'auto', fontSize: '11px' } }, 'the leader is elected by capability, not by row'))));

    // sensors + comms
    const sf = sc.camera.striped_fps;
    f.appendChild(card('3 · Sensors', null,
      h('div', { class: 'cols4' }, num('Frames / s per camera', 'camera.fps', { min: 0.01, max: 240 }), num('Camera px (per side)', 'camera.size', { step: 1, min: 8, max: 2048 }),
        h('label', { class: 'f' }, `Objects in frames: ${Math.round(sc.camera.p_object * 100)}%`, h('input', { type: 'range', min: 0, max: 1, step: 0.01, value: sc.camera.p_object, oninput: (e) => { set('camera.p_object', +e.target.value); e.target.previousSibling.textContent = `Objects in frames: ${Math.round(e.target.value * 100)}%`; } })),
        h('label', { class: 'f' }, 'Single-camera designs', h('select', { onchange: (e) => set('camera.striped_fps', e.target.value === 'match' ? 'match' : e.target.value === 'per' ? null : sc.camera.fps * 2, true) },
          h('option', { value: 'match', selected: sf === 'match' }, 'match: fps x swarm size'), h('option', { value: 'per', selected: sf === null }, 'per-camera fps (low power)'),
          h('option', { value: 'num', selected: isNum(sf) }, 'custom fps')), isNum(sf) ? h('input', { type: 'number', value: sf, oninput: (e) => set('camera.striped_fps', +e.target.value) }) : null))));
    const ln = sc.link;
    f.appendChild(card('4 · Comms - shared radio channel', null,
      h('div', { class: 'cols6' },
        num('Bit rate (bps)', 'link.bps', { nullable: true, placeholder: 'auto: slowest radio', hint: ln.bps ? bps(ln.bps) : 'weakest radio' }),
        num('Latency (ms)', 'link.latency_s', { nullable: true, scale: 1000, placeholder: 'auto' }),
        num('Loss (%)', 'link.loss', { scale: 100, min: 0, max: 90 }),
        num('Heartbeat (s)', 'link.hb_interval', { min: 0.01 }),
        num('Missed beats', 'link.missed_beats', { step: 1, min: 1, max: 50, hint: `detect in ~${(ln.hb_interval * ln.missed_beats).toFixed(1)} s` }),
        h('label', { class: 'f' }, 'Control QoS', h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: ln.qos, onchange: (e) => set('link.qos', e.target.checked) }), 'priority class')))));

    // brain
    const br = sc.brain;
    const brainId = br.kind === 'oracle' ? 'oracle' : br.kind === 'package' ? `package:${br.model}` : `model:${br.model}`;
    const brains = cat.brains;
    const cur = brains.find((b) => b.id === brainId);
    const pickBrain = (id) => {
      const b = brains.find((x) => x.id === id);
      if (!b) return;
      if (b.kind === 'oracle') br.kind = 'oracle';
      else if (b.kind === 'package') { br.kind = 'package'; br.model = b.brain.model; }
      else { br.kind = 'auto'; br.model = b.brain.model; }
      this.changed(true);
    };
    const brainInfo = cur && cur.kind === 'model' && cur.info ? `${si(cur.info.macs)}MAC/frame · ${cur.info.input_shape.join('x')} · ${cur.info.int8 ? 'int8' : 'f32'} · ${bytes(cur.info.model_bytes)}`
      : cur && cur.kind === 'package' ? `${cur.format} package · validation ${cur.status}` : '';
    f.appendChild(card('5 · Brain - what each drone thinks', null,
      h('div', { class: 'cols3' },
        h('label', { class: 'f', style: { gridColumn: 'span 2' } }, 'Perception brain', h('select', { onchange: (e) => pickBrain(e.target.value) },
          h('optgroup', { label: 'parametric' }, brains.filter((b) => b.kind === 'oracle').map((b) => h('option', { value: b.id, selected: b.id === brainId }, b.label))),
          h('optgroup', { label: 'trained models (models/)' }, brains.filter((b) => b.kind === 'model').map((b) => h('option', { value: b.id, selected: b.id === brainId }, b.path))),
          h('optgroup', { label: 'armory packages' }, brains.filter((b) => b.kind === 'package').map((b) => h('option', { value: b.id, selected: b.id === brainId }, `${b.label} [${b.status}]`)))),
        h('span', { class: 'hint' }, brainInfo)),
        br.kind !== 'oracle' && br.kind !== 'package' ? h('label', { class: 'f' }, 'Engine', h('select', { onchange: (e) => set('brain.kind', e.target.value) },
          ['auto', 'native', 'numpy'].map((k) => h('option', { value: k, selected: br.kind === k }, k === 'auto' ? 'auto (C++ if built)' : k === 'native' ? 'C++ engine' : 'numpy')))) : h('span')),
      br.kind === 'oracle' ? h('div', { class: 'cols4' },
        num('MACs / frame', 'brain.macs', { step: 1, min: 1, hint: `${si(br.macs)}MAC` }), num('Accuracy', 'brain.accuracy', { step: 0.01, min: 0, max: 1 }),
        num('Input px', 'brain.input_size', { step: 1, min: 4, max: 512 }), num('Frontend share', 'brain.frontend_fraction', { step: 0.01, min: 0, max: 1, hint: 'for central/features' })) : null));

    // tactics
    const strategies = cat.strategies;
    const chosen = new Set(sc.strategies);
    f.appendChild(card('6 · Swarm designs under test', `baseline: ${sc.baseline}`,
      h('div', { class: 'strat-list' }, strategies.map((s) => h('div', {},
        h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: chosen.has(s.name) || s.name === sc.baseline, disabled: s.name === sc.baseline, onchange: (e) => {
          if (e.target.checked) chosen.add(s.name); else chosen.delete(s.name);
          sc.strategies = strategies.map((x) => x.name).filter((n) => chosen.has(n));
          this.changed();
        } }), h('b', {}, s.name), s.builtin ? null : h('span', { class: `badge ${s.status === 'pass' ? 'pass' : s.status === 'fail' ? 'fail' : 'pending'}`, style: { marginLeft: '6px' } }, 'plugin')),
        h('div', { class: 'note' }, s.note)))),
      h('label', { class: 'f', style: { maxWidth: '280px' } }, 'Baseline (every design is compared with it)', h('select', { onchange: (e) => set('baseline', e.target.value, true) },
        strategies.map((s) => h('option', { value: s.name, selected: s.name === sc.baseline }, s.name)))),
      h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: sc.include_leader, onchange: (e) => set('include_leader', e.target.checked) }), 'single-camera designs: the leader also computes a share of frames')));

    // faults
    const targets = (allowLeader) => [...(allowLeader ? [['leader', 'LEADER (whoever leads)']] : []), ...nodes.map((n, i) => [String(i), callsign(n, i)])];
    const frow = (list, idx, action) => {
      const [who, at] = list[idx];
      return h('div', { class: 'fault-row' },
        h('span', { class: `badge ${action === 'kill' ? 'fail' : 'pass'}` }, action === 'kill' ? 'KILL' : 'REVIVE'),
        h('select', { onchange: (e) => { list[idx][0] = e.target.value === 'leader' ? 'leader' : +e.target.value; this.changed(true); } },
          targets(action === 'kill').map(([v, l]) => h('option', { value: v, selected: String(who) === v }, l))),
        h('input', { type: 'number', value: at, step: 'any', min: 0, oninput: (e) => { list[idx][1] = +e.target.value; this.changed(); this.drawFaults(); } }),
        h('button', { class: 'btn small danger', onclick: () => { list.splice(idx, 1); this.changed(true); } }, 'x'));
    };
    f.appendChild(card('7 · Fault injection', `${sc.faults.kill.length + sc.faults.revive.length} events`,
      h('canvas', { id: 'faultbar', width: 1200, height: 88 }),
      sc.faults.kill.map((_, i) => frow(sc.faults.kill, i, 'kill')),
      sc.faults.revive.map((_, i) => frow(sc.faults.revive, i, 'revive')),
      h('div', { class: 'btnrow' },
        h('button', { class: 'btn small', onclick: () => { sc.faults.kill.push(['leader', +(sc.duration / 3).toFixed(1)]); this.changed(true); } }, '+ kill'),
        h('button', { class: 'btn small', onclick: () => { sc.faults.revive.push([Math.min(1, nodes.length - 1), +(sc.duration * 2 / 3).toFixed(1)]); this.changed(true); } }, '+ revive'),
        h('span', { class: 'muted', style: { fontSize: '11px' } }, 'fault scenarios also run a fault-free twin, so survivability is "retained vs undisturbed"'))));

    // theatre
    const tc = cat.theatre.choices;
    const thSel = (k, label) => h('label', { class: 'f' }, label, h('select', { onchange: (e) => { this.th[k] = e.target.value; this.changed(); } }, tc[k].map((v) => h('option', { value: v, selected: this.th[k] === v }, v))));
    const thNum = (k, label, hint) => h('label', { class: 'f' }, label, h('input', { type: 'number', value: this.th[k], min: 0, step: 'any', oninput: (e) => { this.th[k] = +e.target.value || 0; this.changed(); } }), h('span', { class: 'hint' }, hint));
    f.appendChild(card('8 · Theatre - 3D presentation only', 'never changes results',
      h('div', { class: 'cols3' }, thSel('terrain', 'Terrain'), thSel('formation', 'Formation'), thSel('route', 'Route')),
      h('div', { class: 'cols4' }, thNum('altitude_m', 'Altitude m', '0 = auto'), thNum('spacing_m', 'Spacing m', '0 = auto'), thNum('speed_mps', 'Speed m/s', '0 = auto'), thNum('area_m', 'Area m', 'route size'))));

    // checks
    const metrics = cat.check_metrics;
    const stratNames = ['*', ...strategies.map((s) => s.name)];
    f.appendChild(card('9 · Measures of effectiveness (checks)', `${sc.checks.length}`,
      sc.checks.map((c, i) => h('div', { class: 'check-row' },
        h('select', { onchange: (e) => { c.strategy = e.target.value; this.changed(); } }, stratNames.map((s) => h('option', { value: s, selected: s === c.strategy }, s))),
        h('select', { onchange: (e) => { c.metric = e.target.value; this.changed(); } }, metrics.map((m) => h('option', { value: m, selected: m === c.metric }, m))),
        h('select', { onchange: (e) => { c.op = e.target.value; this.changed(); } }, cat.check_ops.map((o) => h('option', { value: o, selected: o === c.op }, o))),
        h('input', { type: 'number', step: 'any', value: c.value, oninput: (e) => { c.value = +e.target.value; this.changed(); } }),
        h('label', { class: 'chk', title: 'divide by the baseline first' }, h('input', { type: 'checkbox', checked: c.ratio, onchange: (e) => { c.ratio = e.target.checked; this.changed(); } }), 'ratio'),
        h('input', { value: c.note || '', placeholder: 'why', oninput: (e) => { c.note = e.target.value; this.changed(); } }),
        h('button', { class: 'btn small danger', onclick: () => { sc.checks.splice(i, 1); this.changed(true); } }, 'x'))),
      h('div', { class: 'btnrow' }, h('button', { class: 'btn small', onclick: () => { sc.checks.push({ strategy: '*', metric: 'recall', op: '>=', value: 0.5, ratio: false, note: '' }); this.changed(true); } }, '+ check'),
        h('span', { class: 'muted', style: { fontSize: '11px' } }, 'checks turn a mission into a regression test (required for the battery)'))));

    this.drawFaults();
    this.opord();
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.check(), 50);
    f.scrollTop = scroll;
  }

  _fixFaults() {
    const n = this.sc.swarm.nodes.length;
    this.sc.faults.kill = this.sc.faults.kill.filter(([w]) => w === 'leader' || w < n);
    this.sc.faults.revive = this.sc.faults.revive.filter(([w]) => w < n);
  }

  drawFaults() {
    const c = $('#faultbar');
    if (!c) return;
    const ctx = c.getContext('2d'), W = c.width, H = c.height, T = this.sc.duration || 1;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#03070b'; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#1e3442'; ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
    ctx.font = '18px Cascadia Mono, Consolas, monospace'; ctx.fillStyle = '#4c6575';
    for (let k = 0; k <= 10; k++) { const x = 20 + (W - 40) * k / 10; ctx.fillRect(x, H - 18, 2, 8); ctx.fillText(`${(T * k / 10).toFixed(T < 20 ? 1 : 0)}s`, x - 14, H - 22); }
    ctx.fillStyle = 'rgba(43,228,255,0.25)'; ctx.fillRect(20, 30, W - 40, 6);
    const mark = (t, col, label, up) => {
      const x = 20 + (W - 40) * Math.min(1, Math.max(0, t / T));
      ctx.fillStyle = col; ctx.fillRect(x - 2, 14, 4, 30);
      ctx.font = 'bold 17px Bahnschrift, sans-serif'; ctx.fillText(label, Math.min(W - 180, x + 6), up ? 28 : 52);
    };
    const nodes = this.sc.swarm.nodes;
    this.sc.faults.kill.forEach(([w, t]) => mark(t, '#ff3b4d', `KILL ${w === 'leader' ? 'LEADER' : callsign(nodes[w], w)}`, true));
    this.sc.faults.revive.forEach(([w, t]) => mark(t, '#3dffa0', `REVIVE ${callsign(nodes[w], w)}`, false));
  }

  opord() {
    const sc = this.sc, th = this.th, nodes = sc.swarm.nodes || [];
    const counts = {};
    nodes.forEach((n) => { counts[n] = (counts[n] || 0) + 1; });
    const lines = [];
    const H = (t) => lines.push(h('span', { class: 'h' }, t), '\n');
    const L = (t) => lines.push(t + '\n');
    H(`OPORD ${sc.name.toUpperCase()}`);
    L(`DTG: simulated, deterministic (seed ${sc.seed})\n`);
    H('1. SITUATION');
    L(`   Terrain ${th.terrain}, route ${th.route}, formation ${th.formation}`);
    L(`   Objects of interest in ${Math.round(sc.camera.p_object * 100)}% of frames`);
    L(`   (vehicle, building, road, aircraft)\n`);
    H('2. MISSION');
    L(`   ${sc.description}\n`);
    H('3. EXECUTION');
    L(`   a. Forces: ${nodes.length} assets`);
    Object.entries(counts).forEach(([n, c]) => L(`      ${c} x ${n}`));
    L(`   b. Sensors: ${sc.camera.size}px @ ${sc.camera.fps} fps per camera`);
    L(`      single-camera designs: ${sc.camera.striped_fps === 'match' ? `1 camera @ ${(sc.camera.fps * nodes.length).toFixed(1)} fps` : sc.camera.striped_fps === null ? `1 camera @ ${sc.camera.fps} fps` : `1 camera @ ${sc.camera.striped_fps} fps`}`);
    const br = sc.brain;
    L(`   c. Brain: ${br.kind === 'oracle' ? `oracle, ${si(br.macs)}MAC/frame, ${Math.round(br.accuracy * 100)}% accurate` : `${br.model} (${br.kind})`}`);
    L(`   d. Designs: ${[sc.baseline + ' (baseline)', ...sc.strategies.filter((s) => s !== sc.baseline)].join(', ')}`);
    const fl = [...sc.faults.kill.map(([w, t]) => `KILL ${w === 'leader' ? 'LEADER' : callsign(nodes[w], w)} @${t}s`), ...sc.faults.revive.map(([w, t]) => `REVIVE ${callsign(nodes[w], w)} @${t}s`)];
    L(`   e. Faults: ${fl.join(', ') || 'none'}`);
    L(`   f. Duration: ${sc.duration} s simulated\n`);
    H('4. COMMS');
    L(`   Shared channel ${sc.link.bps ? bps(sc.link.bps) : 'auto (slowest radio)'}, latency ${sc.link.latency_s !== null ? (sc.link.latency_s * 1e3).toFixed(1) + ' ms' : 'auto'}`);
    L(`   loss ${(sc.link.loss * 100).toFixed(1)}%, QoS ${sc.link.qos ? 'ON' : 'OFF'}, heartbeat ${sc.link.hb_interval}s x ${sc.link.missed_beats}\n`);
    H('5. MEASURES OF EFFECTIVENESS');
    if (!sc.checks.length) L('   none (add checks to make this a regression test)');
    sc.checks.forEach((c) => L(`   - ${c.strategy} ${c.metric}${c.ratio ? '/baseline' : ''} ${c.op} ${c.value}${c.note ? '  # ' + c.note : ''}`));
    setChildren($('#opord'), lines);
  }

  // ------------------------------------------------------------------ actions
  async execute() {
    if (this.valid) { toast(this.valid, 'err', 'MISSION NOT READY'); return; }
    const body = { scenario: this.payload(), theatre: this.th, with_stress: $('#pl-stress').checked, preset: this.preset ? this.preset.id : null };
    $('#pl-exec').disabled = true;
    try {
      const { job, run } = await api('runs', { method: 'POST', body });
      bus.emit('job', job, `simulate ${body.scenario.name}`);
      await this.app.runJob(job, run, `EXECUTING ${body.scenario.name.toUpperCase()}`);
    } catch (e) {
      toast(e.message, 'err', 'EXECUTION FAILED');
    } finally {
      $('#pl-exec').disabled = !!this.valid;
    }
  }

  async save(where) {
    const body = { scenario: this.payload(), theatre: this.th, where };
    try {
      let res;
      try { res = await api('scenario/save', { method: 'POST', body }); }
      catch (e) {
        if (!/exists/.test(e.message)) throw e;
        const ok = await modal('OVERWRITE PRESET?', h('div', {}, e.message), [{ label: 'CANCEL', value: false }, { label: 'OVERWRITE', cls: 'danger', value: true }]);
        if (!ok) return;
        res = await api('scenario/save', { method: 'POST', body: { ...body, overwrite: true } });
      }
      toast(`saved as ${res.file} - runnable with: python ${res.file}`, 'ok', where === 'battery' ? 'ADDED TO BATTERY' : 'PRESET SAVED');
      await this.app.refreshCatalog(true);
      const p = this.catalog.presets.find((x) => x.file === res.file);
      if (p) { this.preset = p; this.renderPresets(); $('#pl-delete').disabled = !p.editable; }
    } catch (e) {
      toast(e.message, 'err', 'SAVE REFUSED');
    }
  }

  async remove() {
    const p = this.preset;
    if (!p || !p.editable) return;
    const ok = await modal('DELETE PRESET?', h('div', {}, `Delete ${p.file}? This removes the file.`), [{ label: 'CANCEL', value: false }, { label: 'DELETE', cls: 'danger', value: true }]);
    if (!ok) return;
    try {
      await api('scenario/delete', { method: 'POST', body: { file: p.file } });
      toast(`${p.file} deleted`, 'ok');
      this.preset = null;
      await this.app.refreshCatalog(true);
    } catch (e) { toast(e.message, 'err'); }
  }

  /** ARMORY -> "use in mission": a brain or a tactic into the current scenario. */
  useBrain(id) {
    const b = this.catalog.brains.find((x) => x.id === id);
    if (!b) return;
    if (b.kind === 'oracle') this.sc.brain.kind = 'oracle';
    else if (b.kind === 'package') { this.sc.brain.kind = 'package'; this.sc.brain.model = b.brain.model; }
    else { this.sc.brain.kind = 'auto'; this.sc.brain.model = b.brain.model; }
    this.changed(true);
  }

  useTactic(name) {
    if (!this.sc.strategies.includes(name)) this.sc.strategies.push(name);
    this.changed(true);
  }
}
