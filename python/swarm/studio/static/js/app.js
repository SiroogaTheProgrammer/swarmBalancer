// SWARM//OPS test bench: RECON / DEFENCE / OFFENCE tests and TRAIN, on one page.

import { api, hasToken, setToken, waitJob } from './api.js';
import { ArenaView } from './arena3d.js';
import { LineChart } from './chart.js';
import { markdown } from './md.js';
import { $, $$, h, highlightPython, modal, setChildren, toast } from './util.js';

const TABS = ['recon', 'defence', 'offence', 'train'];
const TONE = { recon: 'green', defender: 'blue', attacker: 'red' };
const STORE_KEY = 'swarm.bench.v1';
const SHORT = {
  intercepted: 'Intercepted', leaked: 'Through', overkill: 'Per target', kill_time: 'Time', cpu_def: 'CPU', cpu_att: 'CPU',
  held_off: 'Flying', misses: 'Misses', low_share: 'Masked', coverage: 'Surveyed', t90: '90% at', reported: 'Reported',
  connected: 'Contact', camera_duty: 'Camera', cpu: 'CPU',
};
const EMPTY = {
  recon: ['RECON', 'Pick a recon brain and press RUN TEST. The first episode replays here in 3D: the coverage map fills in (brighter = seen in more detail), each camera\'s cone widens as its drone climbs, and the radio chain back to the lead drone stretches - or breaks. Drag to orbit, right-drag to pan, wheel to zoom.'],
  defence: ['DEFENCE', 'Pick the wall\'s brain and the incoming swarm, then RUN TEST. The wall guards a plane up to the ceiling; amber lines show which attacker each interceptor goes for - magenta when two or more pile onto the same one. Attackers below the terrain mask are drawn dark: they are hard to see.'],
  offence: ['OFFENCE', 'Pick your attack swarm\'s brain and the wall it has to beat, then RUN TEST. Watch how the swarm uses height, the terrain mask and its radio to find a way through.'],
  train: ['TRAIN', 'Start a training run. Every few generations a match between the current brains is recorded and plays here, so you can watch the tactics change.'],
};

const S = {
  cat: null, tab: 'defence', view: null,
  picks: { recon: {}, defence: {}, offence: {} },
  values: { recon: {}, defence: {}, offence: {}, train: {}, trainEnv: { clash: {}, recon: {} } },
  run: { episodes: 8, seed: 0 },
  report: {}, runs: [], testJob: null, testProgress: '',
  train: {
    form: { mode: 'clash', learn_defenders: true, learn_attackers: true, start_defender: 'fresh', start_attacker: 'fresh', start_recon: 'fresh' },
    id: null, job: null, status: null, history: [], refs: null, since: 0, follow: true, match: null, shown: null,
    list: [], ui: null,
  },
  alive: true,
};

// ------------------------------------------------------------------ helpers
function fmtVal(v, fmt) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-';
  switch (fmt) {
    case 'pct': return v > 0 && v < 0.001 ? '<0.1%' : `${(v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0)}%`;
    case 'x': return `${v.toFixed(2)}x`;
    case 's': return `${v.toFixed(1)} s`;
    case 'kbps': return `${v.toFixed(2)} kbit/s`;
    case 'j': return `${v.toFixed(1)} J`;
    case 'km': return `${v.toFixed(2)} km`;
    case 'm': return `${v.toFixed(0)} m`;
    default: return v >= 10 ? v.toFixed(0) : v.toFixed(1);
  }
}
const pct = (v) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '-');
const clock = (t) => `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
const ago = (ts) => {
  const s = (Date.now() - (typeof ts === 'number' ? ts * 1000 : Date.parse(ts))) / 1000;
  return s < 60 ? 'just now' : s < 3600 ? `${Math.round(s / 60)} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`;
};
const brainById = (id) => S.cat.brains.find((b) => b.id === id);
const titleOf = (id) => (id === 'fresh' ? 'Fresh network' : (brainById(id) || { title: id }).title);

function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ tab: S.tab, picks: S.picks, values: S.values, run: S.run, trainForm: S.train.form }));
  } catch { /* storage full or blocked: settings just will not persist */ }
}

function restore() {
  try {
    const d = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
    if (TABS.includes(d.tab)) S.tab = d.tab;
    for (const k of ['recon', 'defence', 'offence']) Object.assign(S.picks[k], (d.picks || {})[k] || {});
    if (d.values) {
      for (const k of ['recon', 'defence', 'offence', 'train']) Object.assign(S.values[k], d.values[k] || {});
      for (const m of ['clash', 'recon']) Object.assign(S.values.trainEnv[m], (d.values.trainEnv || {})[m] || {});
    }
    if (d.run) Object.assign(S.run, d.run);
    if (d.trainForm) Object.assign(S.train.form, d.trainForm);
  } catch { /* ignore a corrupt store */ }
}

/** Only fields the server knows, so settings saved by an older version never break a run. */
function payload(schema, values) {
  const out = {};
  for (const f of schema) if (values[f.name] !== undefined) out[f.name] = values[f.name];
  return out;
}

function fixPicks() {
  for (const [tab, t] of Object.entries(S.cat.tests)) {
    for (const role of t.sides) {
      const b = brainById(S.picks[tab][role]);
      if (!b || b.role !== role) S.picks[tab][role] = S.cat.defaults[role];
    }
  }
  const F = S.train.form;
  for (const role of ['defender', 'attacker', 'recon']) {
    const key = `start_${role}`, b = brainById(F[key]);
    const learns = role === 'recon' || F[`learn_${role}s`];
    if (F[key] !== 'fresh' && (!b || b.role !== role || (learns && b.kind !== 'neural'))) F[key] = learns ? 'fresh' : S.cat.defaults[role];
  }
}

function badges(b) {
  const out = [];
  if (b.kind === 'neural') out.push(h('span', { class: 'badge neural' }, 'neural'));
  else if (b.kind === 'python') out.push(h('span', { class: 'badge python' }, 'python'));
  else out.push(h('span', { class: 'badge' }, 'built-in'));
  if (b.comms) out.push(' ', h('span', { class: 'badge radio', title: 'coordinates over the mesh radio' }, 'radio'));
  return out;
}

// ------------------------------------------------------------------ boot
function initToken() {
  const grab = () => {
    const m = /[#&]t=([A-Za-z0-9_-]+)/.exec(location.hash);
    if (m) sessionStorage.setItem('studio-token', m[1]);
    return !!m;
  };
  if (grab()) history.replaceState(null, '', location.pathname);
  setToken(sessionStorage.getItem('studio-token'));
  window.addEventListener('hashchange', () => { if (grab()) location.replace(location.pathname); });
}

function offline(msg) {
  S.alive = false;
  if (S.view) S.view.stop();
  if (msg) $('#offline-msg').textContent = msg;
  $('#offline').classList.add('show');
}

async function boot() {
  initToken();
  if (!hasToken()) {
    offline('No session token. Open the studio through the link printed by `python -m swarm.studio`.');
    return;
  }
  try { S.cat = await api('catalog'); } catch (e) { offline(e.message); return; }
  S.view = new ArenaView($('#gl'), { hud: $('#hud'), ticker: $('#ticker'), legend: $('#legend'), labels: $('#labels'), empty: $('#empty') });
  S.view.onTick = onTick;
  restore();
  fixPicks();
  wireTop();
  wirePlaybar();
  await Promise.all([refreshRuns(), refreshTrainings()]);
  await reattach();
  switchTab(S.tab);
  pingLoop();
  pollLoop();
}

function wireTop() {
  $$('#tabs button').forEach((b) => { b.onclick = () => switchTab(b.dataset.tab); });
  $('#btn-brains').onclick = openBrains;
  $('#btn-help').onclick = () => openHelp('arena');
  $('#btn-power').onclick = shutdown;
  $('#drawer-close').onclick = closeDrawer;
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDrawer();
    if (e.key === ' ' && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)) {
      e.preventDefault();
      togglePlay();
    }
  });
}

function switchTab(tab) {
  S.tab = tab;
  save();
  $$('#tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
  const [title, text] = EMPTY[tab];
  setChildren($('#empty'), h('div', {}, h('b', {}, title), text));
  renderSetup();
  renderResults();
  const replay = tab === 'train' ? (S.train.shown && S.train.shown.replay) : S.report[tab] && S.report[tab].replay;
  S.view.load(replay || null);
  if (replay) S.view.playing = true;
}

function renderSetup() { if (S.tab === 'train') renderTrainSetup(); else renderTestSetup(S.tab); }
function renderResults() { if (S.tab === 'train') renderTrainResults(); else renderTestResults(S.tab); }

// ------------------------------------------------------------------ playback bar
function wirePlaybar() {
  const speeds = [1, 2, 4, 8, 16, 32];
  const seg = $('#pb-speed');
  setChildren(seg, speeds.map((s) => h('button', {
    class: s === S.view.speed ? 'on' : '', onclick: (e) => {
      S.view.speed = s;
      $$('button', seg).forEach((b) => b.classList.toggle('on', b === e.currentTarget));
    },
  }, `${s}x`)));
  $('#pb-play').onclick = togglePlay;
  const scrub = $('#pb-scrub');
  scrub.oninput = () => { S.view.playing = false; S.view.setTime(scrub.value / 10); };
  $('#pb-links').onchange = (e) => { S.view.show.links = e.target.checked; };
  $('#pb-locks').onchange = (e) => { S.view.show.locks = e.target.checked; };
}

function togglePlay() {
  const v = S.view;
  if (!v.replay) return;
  if (!v.playing && v.t >= v.duration - 1e-6) v.setTime(0);
  v.playing = !v.playing;
}

let lastTick = '';
function onTick(t, dur, playing) {
  const key = `${Math.round(t * 10)}|${dur}|${playing}`;
  if (key === lastTick) return;
  lastTick = key;
  const scrub = $('#pb-scrub');
  scrub.max = String(Math.round(dur * 10));
  if (document.activeElement !== scrub) scrub.value = String(Math.round(t * 10));
  $('#pb-time').textContent = `${clock(t)} / ${clock(dur)}`;
  $('#pb-play').innerHTML = playing ? '&#10074;&#10074;' : '&#9654;';
}

// ------------------------------------------------------------------ forms
function settingsForm(schema, values, { exclude = [] } = {}) {
  const field = (f) => {
    if (exclude.includes(f.name)) return null;
    const cur = values[f.name] ?? f.default;
    if (f.type === 'bool') {
      const box = h('input', { type: 'checkbox', checked: cur, onchange: () => { values[f.name] = box.checked; save(); } });
      return h('div', { class: 'field check', title: f.help }, h('label', {}, box, f.label));
    }
    let input;
    if (f.type === 'choice') {
      input = h('select', { onchange: () => { values[f.name] = input.value; save(); } },
        f.choices.map((c) => h('option', { value: c, selected: c === cur }, c)));
    } else {
      input = h('input', {
        type: 'number', value: cur, min: f.min ?? undefined, max: f.max ?? undefined, step: f.type === 'int' ? 1 : 'any',
        onchange: () => {
          let v = Number(input.value);
          if (!Number.isFinite(v)) { input.value = cur; return; }
          if (f.min !== null && f.min !== undefined) v = Math.max(f.min, v);
          if (f.max !== null && f.max !== undefined) v = Math.min(f.max, v);
          if (f.type === 'int') v = Math.round(v);
          input.value = v;
          values[f.name] = v;
          save();
        },
      });
    }
    return h('div', { class: 'field', title: f.help || f.label }, h('label', {}, f.label, f.unit ? h('u', {}, ` ${f.unit}`) : null), input);
  };
  const basic = schema.filter((f) => f.basic).map(field).filter(Boolean);
  const adv = schema.filter((f) => !f.basic).map(field).filter(Boolean);
  const reset = h('button', {
    class: 'btn', style: { marginTop: '10px' },
    onclick: () => { for (const k of Object.keys(values)) delete values[k]; save(); renderSetup(); },
  }, 'Reset to defaults');
  return [
    basic.length ? h('div', { class: 'fields' }, basic) : null,
    adv.length ? h('details', { class: 'adv' }, h('summary', {}, `Advanced (${adv.length})`), h('div', { class: 'fields' }, adv), reset) : null,
  ];
}

function numField(label, value, min, max, onChange) {
  const input = h('input', {
    type: 'number', value, min, max, step: 1,
    onchange: () => {
      const v = Math.round(Math.min(max, Math.max(min, Number(input.value) || min)));
      input.value = v;
      onChange(v);
      save();
    },
  });
  return h('div', { class: 'field' }, h('label', {}, label), input);
}

function brainSelect(role, value, onChange, { fresh = false, neuralOnly = false } = {}) {
  const brains = S.cat.brains.filter((b) => b.role === role && b.kind !== 'broken' && (!neuralOnly || b.kind === 'neural'));
  const sel = h('select', { onchange: () => onChange(sel.value) });
  if (fresh) sel.appendChild(h('option', { value: 'fresh' }, 'Fresh network (untrained)'));
  for (const [label, list] of [['Built-in tactics', brains.filter((b) => b.kind === 'builtin')], ['Library', brains.filter((b) => b.kind !== 'builtin')]]) {
    if (!list.length) continue;
    sel.appendChild(h('optgroup', { label }, list.map((b) => h('option', { value: b.id },
      b.title + (b.kind === 'neural' ? '  - neural' : b.kind === 'python' ? '  - python' : '')))));
  }
  sel.value = value;
  return sel;
}

// ------------------------------------------------------------------ test tabs
function sidePicker(tab, role, subject) {
  const names = {
    recon: ['YOUR RECON SWARM', 'under test'],
    defender: subject ? ['YOUR WALL - DEFENDERS', 'under test'] : ['THE WALL', 'opponent'],
    attacker: subject ? ['YOUR SWARM - ATTACKERS', 'under test'] : ['INCOMING SWARM', 'opponent'],
  }[role];
  const desc = h('div', { class: 'desc' });
  const show = () => {
    const b = brainById(S.picks[tab][role]);
    setChildren(desc, b ? [badges(b), ' ', b.summary] : '');
  };
  const sel = brainSelect(role, S.picks[tab][role], (v) => { S.picks[tab][role] = v; save(); show(); });
  show();
  const load = h('button', {
    class: 'btn', title: 'load your own brain: a .py policy, trained .npz weights or a package .zip',
    onclick: () => pickUpload(role, (id) => { S.picks[tab][role] = id; save(); renderSetup(); }),
  }, 'LOAD');
  return h('div', { class: `side ${TONE[role]}` }, h('div', { class: 'who' }, h('b', {}, names[0]), h('span', {}, names[1])),
    h('div', { class: 'row' }, sel, load), desc);
}

function renderTestSetup(tab) {
  const t = S.cat.tests[tab];
  const running = S.testJob && S.testJob.tab === tab;
  const noun = { recon: 'recon brain', defender: 'defender brain', attacker: 'attacker brain' }[t.subject];
  setChildren($('#setup'),
    h('div', { class: 'kicker' }, `TEST 0${TABS.indexOf(tab) + 1}`),
    h('h1', { class: 'title' }, t.title),
    h('p', { class: 'about' }, t.about),
    t.sides.map((role, k) => sidePicker(tab, role, k === 0)),
    h('div', { class: 'sect' }, h('h4', {}, tab === 'recon' ? 'MISSION' : 'ENGAGEMENT'), settingsForm(t.settings, S.values[tab])),
    h('div', { class: 'sect' }, h('h4', {}, 'RUN'),
      h('div', { class: 'fields' },
        numField('Episodes', S.run.episodes, 1, 64, (v) => { S.run.episodes = v; }),
        numField('Seed', S.run.seed, 0, 2147483647, (v) => { S.run.seed = v; })),
      h('div', { class: 'runbox' }, h('button', { class: 'btn primary big', disabled: running, onclick: runTest }, running ? 'RUNNING...' : 'RUN TEST')),
      h('div', { class: 'hint' }, `Plays every episode at once, then replays the same situations with each built-in ${noun} so you can compare.`)));
}

async function runTest() {
  const tab = S.tab, t = S.cat.tests[tab];
  const body = { test: tab, settings: payload(t.settings, S.values[tab]), brains: { ...S.picks[tab] }, episodes: S.run.episodes, seed: S.run.seed };
  try {
    const r = await api('test', { method: 'POST', body });
    S.testJob = { id: r.job, tab };
    S.testProgress = 'queued';
    renderSetup();
    renderResults();
  } catch (e) {
    toast(e.message, 'err', 'CANNOT START THE TEST');
  }
}

async function pollTest() {
  const tj = S.testJob;
  if (!tj) return;
  let job;
  try { job = await api(`jobs/${tj.id}`); } catch { return; }
  if (job.state === 'queued' || job.state === 'running') {
    S.testProgress = job.progress || job.state;
    const el = $('#jobprog');
    if (el) el.textContent = S.testProgress;
    return;
  }
  S.testJob = null;
  if (job.state === 'done') {
    try {
      const doc = await api(`runs/${job.result.run}`);
      S.report[tj.tab] = doc;
      await refreshRuns();
      if (S.tab === tj.tab) { S.view.load(doc.replay); S.view.playing = true; }
    } catch (e) { toast(e.message, 'err', 'CANNOT LOAD THE REPORT'); }
  } else {
    toast([job.error || job.state, ...(job.log || []).slice(-4)].join('\n'), 'err', 'TEST FAILED');
  }
  if (S.tab === tj.tab) { renderSetup(); renderResults(); }
}

function cardsAndColumns(doc) {
  const meta = {};
  for (const c of doc.cards) meta[c.key] = c;
  return meta;
}

function renderTestResults(tab) {
  const doc = S.report[tab];
  const out = [];
  if (S.testJob && S.testJob.tab === tab) {
    out.push(h('div', { class: 'verdict info' }, h('div', { class: 'lbl' }, 'SIMULATING...'), h('p', { id: 'jobprog' }, S.testProgress)));
  }
  if (doc) {
    const meta = cardsAndColumns(doc);
    const rows = doc.compare;
    const mine = rows.find((r) => r.selected);
    const v = doc.verdict;
    out.push(h('div', { class: `verdict ${v.tone}` }, h('div', { class: 'lbl' }, v.label), h('p', {}, v.line)));
    const vs = Object.values(doc.brains).map((b) => b.title).join('  vs  ');
    out.push(h('div', { class: 'meta' }, `${vs} | ${doc.episodes} episodes, seed ${doc.seed} | ${doc.elapsed_s} s`));
    out.push(h('div', { class: 'cards' }, doc.cards.map((c) => {
      let cls = 'card';
      if (c.better && rows.length > 1 && mine && Number.isFinite(mine.metrics[c.key])) {
        const vals = rows.map((r) => r.metrics[c.key]).filter(Number.isFinite);
        const best = c.better === 'high' ? Math.max(...vals) : Math.min(...vals);
        const worst = c.better === 'high' ? Math.min(...vals) : Math.max(...vals);
        if (mine.metrics[c.key] === best && best !== worst) cls += ' best';
        else if (mine.metrics[c.key] === worst && best !== worst) cls += ' worst';
      }
      const spread = c.n > 1 && Number.isFinite(c.std) ? `+/- ${fmtVal(c.std, c.fmt)} across episodes` : '';
      return h('div', { class: cls, title: cls.includes('best') ? 'best of all brains compared' : cls.includes('worst') ? 'worst of all brains compared' : '' },
        h('div', { class: 'k' }, c.label), h('div', { class: 'v' }, fmtVal(c.mean, c.fmt)), h('div', { class: 's' }, spread));
    })));
    out.push(h('h3', { class: 'sub' }, h('span', {}, 'Same situations, other brains'), h('span', {}, 'rank by score')));
    const cols = doc.columns;
    out.push(h('table', { class: 'cmp' },
      h('tr', {}, h('th', {}, 'Brain'), cols.map((k) => h('th', { title: (meta[k] || {}).label || k }, SHORT[k] || k))),
      rows.map((r) => h('tr', { class: r.selected ? 'sel' : '' },
        h('td', {}, h('span', { class: 'rank' }, `${r.rank}.`), r.brain.title, r.selected ? h('span', { class: 'badge', style: { marginLeft: '6px' } }, 'tested') : null),
        cols.map((k) => {
          const m = meta[k] || { fmt: k.startsWith('cpu') || ['camera_duty', 'held_off', 'connected', 'reported', 'coverage'].includes(k) ? 'pct' : k === 't90' || k === 'kill_time' ? 's' : 'num' };
          const vals = rows.map((q) => q.metrics[k]).filter(Number.isFinite);
          const best = m.better === 'high' ? Math.max(...vals) : m.better === 'low' ? Math.min(...vals) : null;
          const isBest = best !== null && r.metrics[k] === best && vals.length > 1;
          return h('td', { class: isBest ? 'tone-good' : '' }, fmtVal(r.metrics[k], m.fmt));
        })))));
  } else if (!S.testJob || S.testJob.tab !== tab) {
    out.push(h('div', { class: 'verdict info' }, h('div', { class: 'lbl' }, 'READY'),
      h('p', {}, 'Choose the brains on the left and press RUN TEST. Results land here: a verdict, the key numbers, and how every built-in brain does on exactly the same situations.')));
  }
  const runs = S.runs.filter((r) => r.test === tab).slice(0, 14);
  out.push(h('h3', { class: 'sub' }, h('span', {}, 'Recent runs'), h('span', {}, runs.length ? '' : 'none yet')));
  out.push(h('div', { class: 'list' }, runs.map((r) => h('div', {
    class: `item ${doc && doc.id === r.id ? 'on' : ''}`, onclick: () => openRun(r.id),
  },
  h('span', { class: `t tone-${r.verdict.tone}` }, r.verdict.label),
  h('span', { class: 'x' }, ago(r.created)),
  h('span', { class: 'd' }, `${Object.values(r.brains).join(' vs ')} | ${r.headline.map((c) => `${c.label} ${fmtVal(c.mean, c.fmt)}`).join(', ')}`)))));
  setChildren($('#results'), out);
}

async function openRun(id) {
  try {
    const doc = await api(`runs/${id}`);
    S.report[doc.test] = doc;
    if (S.tab !== doc.test) switchTab(doc.test);
    else { renderResults(); S.view.load(doc.replay); S.view.playing = true; }
  } catch (e) { toast(e.message, 'err'); }
}

async function refreshRuns() {
  try { S.runs = (await api('runs')).runs; } catch { /* keep the old list */ }
}

// ------------------------------------------------------------------ training
const trainRunning = () => !!S.train.job;

function trainSide(role) {
  const F = S.train.form, key = `start_${role}`, learnKey = `learn_${role}s`;
  const learns = role === 'recon' || F[learnKey];
  const sel = brainSelect(role, F[key], (v) => { F[key] = v; save(); }, { fresh: true, neuralOnly: learns });
  const head = role === 'recon' ? ['RECON SWARM', 'learns'] : [role === 'defender' ? 'DEFENDERS - THE WALL' : 'ATTACKERS - THE SWARM', ''];
  let toggle = null;
  if (role !== 'recon') {
    const box = h('input', {
      type: 'checkbox', checked: F[learnKey], disabled: trainRunning(),
      onchange: () => {
        F[learnKey] = box.checked;
        const b = brainById(F[key]);
        if (box.checked && F[key] !== 'fresh' && (!b || b.kind !== 'neural')) F[key] = 'fresh';
        if (!box.checked && F[key] === 'fresh') F[key] = S.cat.defaults[role];
        save();
        renderTrainSetup();
      },
    });
    toggle = h('label', { class: 'tog' }, box, 'learns');
  }
  return h('div', { class: `side ${TONE[role]}` }, h('div', { class: 'who' }, h('b', {}, head[0]), toggle || h('span', {}, head[1])),
    h('div', { class: 'row' }, h('span', { class: 'hint', style: { margin: '0', alignSelf: 'center', minWidth: '62px' } }, learns ? 'start from' : 'fixed'), sel),
    h('div', { class: 'desc' }, learns
      ? 'Its mesh network evolves every generation. Start fresh, or keep training a saved network.'
      : 'Stays as it is - the opponent the other side trains against.'));
}

function renderTrainSetup() {
  const F = S.train.form, running = trainRunning();
  const mode = F.mode;
  const seg = h('div', { class: 'seg', style: { marginBottom: '12px' } },
    [['clash', 'TWO SWARMS'], ['recon', 'RECON SWARM']].map(([m, label]) => h('button', {
      class: mode === m ? 'on' : '', disabled: running,
      onclick: () => { F.mode = m; save(); renderTrainSetup(); },
    }, label)));
  const envSchema = mode === 'recon' ? S.cat.tests.recon.settings : S.cat.tests.defence.settings;
  setChildren($('#setup'),
    h('div', { class: 'kicker' }, 'TEST 04'),
    h('h1', { class: 'title' }, 'TRAIN'),
    h('p', { class: 'about' }, mode === 'recon'
      ? 'Evolve a recon swarm brain: every generation tries variants of its network and keeps what covers more ground, stays in contact and saves load.'
      : 'Let the smart wall and the attacking swarm learn against each other. Each generation both sides try variants of their networks; the charts show who learns faster and who is winning.'),
    seg,
    mode === 'recon' ? trainSide('recon') : [trainSide('defender'), trainSide('attacker')],
    h('div', { class: 'sect' }, h('h4', {}, 'TRAINING'),
      settingsForm(S.cat.train, S.values.train, { exclude: ['mode', 'learn_defenders', 'learn_attackers', 'start_defender', 'start_attacker', 'start_recon'] })),
    h('details', { class: 'adv sect' }, h('summary', {}, mode === 'recon' ? 'Mission settings' : 'Engagement settings'),
      settingsForm(envSchema, S.values.trainEnv[mode])),
    h('div', { class: 'runbox' }, running
      ? h('div', { class: 'btnrow' },
        h('button', { class: 'btn warn', style: { flex: '1' }, onclick: stopTraining, title: 'finish this generation and keep the brains' }, 'STOP & KEEP'),
        h('button', { class: 'btn danger', onclick: abortTraining, title: 'kill it now (brains of finished generations are kept)' }, 'ABORT'))
      : h('button', { class: 'btn primary big', onclick: startTraining }, 'START TRAINING')),
    h('div', { class: 'hint' }, `Simulates on ${S.cat.cores > 1 ? `all ${S.cat.cores} CPU cores` : 'one CPU core'} (Advanced: CPU cores) - with the default settings about ${mode === 'recon' ? '0.4' : '0.7'} s per generation on 8 cores. The brains are saved after every generation, and the best one on the fixed skill test is kept too; save them to the library to use them in the tests.`),
    S.cat.native === false ? h('div', { class: 'hint warn' }, 'This Python runs emulated (an x64 build on an ARM64 PC), so training runs at about half speed. Run "python dev.py native-python" once: training then uses a native interpreter automatically.') : null);
}

async function startTraining() {
  const F = S.train.form;
  if (F.mode === 'clash' && !F.learn_defenders && !F.learn_attackers) {
    toast('Tick "learns" for the defenders, the attackers, or both.', 'warn');
    return;
  }
  const envSchema = F.mode === 'recon' ? S.cat.tests.recon.settings : S.cat.tests.defence.settings;
  const body = {
    train: { ...payload(S.cat.train, S.values.train), mode: F.mode, learn_defenders: F.learn_defenders, learn_attackers: F.learn_attackers,
      start_defender: F.start_defender, start_attacker: F.start_attacker, start_recon: F.start_recon },
    settings: payload(envSchema, S.values.trainEnv[F.mode]),
  };
  try {
    const r = await api('train', { method: 'POST', body });
    Object.assign(S.train, { id: r.training, job: r.job, status: null, history: [], refs: null, since: 0, match: null, shown: null, follow: true });
    await openTraining(r.training, { keepJob: true });
    renderTrainSetup();
    toast('Training started - the charts update every generation.', 'ok', 'TRAINING');
  } catch (e) {
    toast(e.message, 'err', 'CANNOT START TRAINING');
  }
}

async function stopTraining() {
  if (!S.train.job) return;
  try { await api(`jobs/${S.train.job}/stop`, { method: 'POST' }); toast('Stopping after this generation...', 'warn'); } catch (e) { toast(e.message, 'err'); }
}

async function abortTraining() {
  if (!S.train.job) return;
  const ok = await modal('Abort training?', h('p', {}, 'The run is killed now. Brains from finished generations stay saved.'),
    [{ label: 'KEEP TRAINING', value: false }, { label: 'ABORT', cls: 'danger', value: true }]);
  if (ok) { try { await api(`jobs/${S.train.job}/cancel`, { method: 'POST' }); } catch (e) { toast(e.message, 'err'); } }
}

async function refreshTrainings() {
  try { S.train.list = (await api('training')).training; } catch { /* keep */ }
}

/** Opens a training run: status, history and its latest recorded match. */
async function openTraining(id, { keepJob = false } = {}) {
  const T = S.train;
  try {
    const d = await api(`training/${id}`);
    T.id = id;
    T.status = d.status;
    T.history = d.history;
    T.refs = d.status.references || null;
    if (!keepJob) T.job = ['queued', 'running'].includes(d.status.state) ? d.status.job : null;
    const last = [...T.history].reverse().find((r) => r.match);
    T.match = last ? last.match : null;
    T.shown = null;
    if (T.match) await showMatch(T.match, true);
  } catch (e) {
    toast(e.message, 'err');
  }
  if (S.tab === 'train') { renderTrainSetup(); renderTrainResults(); }
}

async function showMatch(name, fit = false) {
  const T = S.train;
  try {
    const doc = await api(`training/${T.id}/matches/${name}`);
    T.shown = { name, gen: doc.gen, replay: doc.replay, result: doc.result };
    if (S.tab === 'train') {
      S.view.load(doc.replay, { fit: fit || !S.view.replay });
      S.view.playing = true;
      const el = $('#match-label');
      if (el) el.textContent = matchLabel();
    }
  } catch (e) { toast(e.message, 'err'); }
}

function matchLabel() {
  const m = S.train.shown;
  if (!m) return 'no match recorded yet';
  const r = m.result || {};
  if (S.train.status && S.train.status.mode === 'recon') return `generation ${m.gen}: ${pct(r.coverage)} surveyed, contact ${pct(r.connected)}`;
  return `generation ${m.gen}: ${pct(r.intercepted)} intercepted, ${pct(r.leaked)} got through`;
}

async function pollTraining() {
  const T = S.train;
  if (!T.job) return;
  let job;
  try { job = await api(`jobs/${T.job}?since=${T.since}`); } catch { return; }
  T.since = job.next;
  let fresh = false;
  for (const e of job.events) {
    if (e.type === 'start') { T.refs = e.references || T.refs; fresh = true; }
    if (e.type === 'generation') {
      if (!T.history.some((r) => r.gen === e.gen)) T.history.push(e);
      if (e.match) { T.match = e.match; if (T.follow) await showMatch(e.match, !T.shown); }
      fresh = true;
    }
  }
  if (T.status) Object.assign(T.status, { state: job.state === 'running' ? 'running' : T.status.state, generation: T.history.length });
  if (job.state !== 'queued' && job.state !== 'running') {
    T.job = null;
    if (job.state === 'done') toast(job.result && job.result.stopped ? 'Stopped - the brains are kept.' : 'Training finished.', 'ok', 'TRAINING');
    else if (job.state === 'error') toast(job.error || 'training failed', 'err', 'TRAINING FAILED');
    await refreshTrainings();
    await openTraining(T.id, { keepJob: true });
    return;
  }
  if (S.tab === 'train') {
    if (fresh) updateTrainCharts();
    const el = $('#train-status');
    if (el) el.textContent = trainStatusText();
  }
}

function trainStatusText() {
  const T = S.train, st = T.status;
  if (!st) return '';
  const n = T.history.length, total = st.generations;
  const per = n ? T.history.slice(-5).reduce((a, r) => a + r.seconds, 0) / Math.min(5, n) : null;
  const state = T.job ? (n ? 'TRAINING' : 'STARTING') : (st.state || '').toUpperCase();
  let txt = `${state} | generation ${n} of ${total}`;
  if (per) txt += ` | ${per.toFixed(1)} s each`;
  if (T.job && per) txt += ` | about ${clock(per * (total - n))} left`;
  return txt;
}

function renderTrainResults() {
  const T = S.train;
  const out = [];
  if (!T.id) {
    out.push(h('div', { class: 'verdict info' }, h('div', { class: 'lbl' }, 'NOTHING TRAINING'),
      h('p', {}, 'Start a run on the left. Two swarms: the wall and the attackers evolve at the same time, each against the other\'s current brain, past versions of it and the built-in tactics.')));
  } else {
    const st = T.status || {};
    const recon = st.mode === 'recon';
    out.push(h('div', { class: 'meta', id: 'train-status', style: { margin: '0 0 6px', color: 'var(--text)' } }, trainStatusText()));
    out.push(h('div', { class: 'progress' }, h('div', { id: 'train-bar' })));
    const ui = { charts: {} };
    if (recon) {
      out.push(h('h3', { class: 'sub' }, h('span', {}, 'Score vs the built-in recon brains')));
      out.push(h('div', { class: 'chart', id: 'chart-a' }), h('div', { class: 'chart-legend', id: 'legend-a' }));
      out.push(h('h3', { class: 'sub' }, h('span', {}, 'What the current brain does')));
      out.push(h('div', { class: 'chart', id: 'chart-b' }), h('div', { class: 'chart-legend', id: 'legend-b' }));
    } else {
      out.push(h('h3', { class: 'sub' }, h('span', {}, 'Who is winning - head to head')));
      out.push(h('div', { class: 'tug' }, h('span', { style: { color: 'var(--cyan)' }, id: 'tug-l' }, '-'), h('div', { class: 'bar' }, h('div', { id: 'tug-bar' })), h('span', { style: { color: 'var(--red)' }, id: 'tug-r' }, '-')));
      out.push(h('div', { class: 'chart', id: 'chart-a' }), h('div', { class: 'chart-legend', id: 'legend-a' }));
      out.push(h('h3', { class: 'sub' }, h('span', {}, 'Who learns faster - vs built-in opponents')));
      out.push(h('div', { class: 'chart', id: 'chart-b' }), h('div', { class: 'chart-legend', id: 'legend-b' }));
    }
    out.push(h('div', { class: 'verdict info', id: 'insight', style: { marginTop: '10px' } }));
    out.push(h('h3', { class: 'sub' }, h('span', {}, 'Recorded matches'), h('span', { id: 'match-label' }, matchLabel())));
    const follow = h('input', { type: 'checkbox', checked: T.follow, onchange: () => { T.follow = follow.checked; } });
    out.push(h('div', { class: 'btnrow' },
      h('button', { class: 'btn', onclick: () => { if (T.match) showMatch(T.match); } }, 'WATCH LATEST MATCH'),
      h('label', { class: 'tog' }, follow, 'follow new matches')));
    const roles = recon ? ['recon'] : ['defender', 'attacker'].filter((r) => (st.train || {})[`learn_${r}s`]);
    out.push(h('h3', { class: 'sub' }, h('span', {}, 'Keep the trained brains')));
    out.push(h('div', { class: 'btnrow' }, roles.map((role) => h('button', {
      class: 'btn primary', disabled: !T.history.length, onclick: () => saveTrained(role),
    }, `SAVE ${role.toUpperCase()} BRAIN`))));
    ui.ready = true;
    T.ui = ui;
  }
  out.push(h('h3', { class: 'sub' }, h('span', {}, 'Training runs'), h('span', {}, T.list.length ? '' : 'none yet')));
  out.push(h('div', { class: 'list' }, T.list.slice(0, 12).map((r) => h('div', { class: `item ${r.id === T.id ? 'on' : ''}`, onclick: () => openTraining(r.id) },
    h('span', { class: 't' }, r.mode === 'recon' ? 'Recon swarm' : (r.train && r.train.learn_defenders && r.train.learn_attackers ? 'Both swarms' : r.train && r.train.learn_defenders ? 'Defenders' : 'Attackers')),
    h('span', { class: 'x' }, `${r.state} | ${ago(r.created)}`),
    h('span', { class: 'd' }, `${r.generation || 0} of ${r.generations} generations`, h('a', {
      href: '#', style: { float: 'right', color: 'var(--faint)' }, title: 'delete this training run',
      onclick: (e) => { e.preventDefault(); e.stopPropagation(); deleteTraining(r.id); },
    }, 'delete'))))));
  setChildren($('#results'), out);
  if (T.id) {
    T.ui.charts.a = new LineChart($('#chart-a'));
    T.ui.charts.b = new LineChart($('#chart-b'));
    updateTrainCharts();
  }
}

function legend(el, items) {
  setChildren(el, items.map(([color, label, dash]) => h('span', {}, h('i', { style: { background: dash ? 'transparent' : color, borderTop: dash ? `2px dashed ${color}` : '' } }), label)));
}

function updateTrainCharts() {
  const T = S.train, ui = T.ui;
  if (!ui || !ui.charts.a || !T.status) return;
  const H = T.history, st = T.status;
  const bar = $('#train-bar');
  if (bar) bar.style.width = `${Math.min(100, 100 * H.length / Math.max(1, st.generations))}%`;
  const series = (f) => H.map((r) => [r.gen, f(r)]);
  const insight = $('#insight');
  if (st.mode === 'recon') {
    const refs = T.refs || {};
    const palette = ['#ffb21e', '#9aa9b5', '#ff7ab6', '#7d8cff'];
    const refSeries = Object.entries(refs).map(([name, m], k) => ({ name, color: palette[k % palette.length], dash: true, width: 1.4,
      points: H.length ? [[H[0].gen, m.fitness], [H[H.length - 1].gen, m.fitness]] : [] }));
    const fit = H.map((r) => r.current.fitness).concat(Object.values(refs).map((m) => m.fitness)).filter(Number.isFinite);
    const lo = fit.reduce((a, v) => Math.min(a, v), 0), hi = fit.reduce((a, v) => Math.max(a, v), 1.2);
    ui.charts.a.opts = { ...ui.charts.a.opts, yMin: Math.floor(lo * 5) / 5, yMax: Math.ceil(hi * 5) / 5, yFmt: (v) => v.toFixed(1) };
    ui.charts.a.set([{ name: 'mesh', color: '#43f59b', points: series((r) => r.current.fitness) }, ...refSeries]);
    legend($('#legend-a'), [['#43f59b', 'trained mesh brain'], ...refSeries.map((s) => [s.color, s.name, true])]);
    ui.charts.b.set([
      { name: 'coverage', color: '#43f59b', points: series((r) => r.current.coverage) },
      { name: 'contact', color: '#36dcff', points: series((r) => r.current.connected) },
      { name: 'camera', color: '#ffb21e', points: series((r) => r.current.camera_duty) },
    ]);
    legend($('#legend-b'), [['#43f59b', 'area surveyed'], ['#36dcff', 'in radio contact'], ['#ffb21e', 'camera on (load)']]);
    if (insight && H.length) {
      const cur = H[H.length - 1].current;
      const best = Object.entries(refs).sort((a, b) => b[1].fitness - a[1].fitness)[0];
      const ahead = best && cur.fitness >= best[1].fitness;
      const top = (H[H.length - 1].best || {}).recon;
      setChildren(insight, h('div', { class: 'lbl' }, ahead ? 'BEATS THE BUILT-INS' : 'STILL LEARNING'),
        h('p', {}, `Score ${cur.fitness.toFixed(2)}${best ? ` vs best built-in ${best[0]} ${best[1].fitness.toFixed(2)}` : ''}: surveys ${pct(cur.coverage)}, in contact ${pct(cur.connected)} of the time, cameras on ${pct(cur.camera_duty)}.${top ? ` Best so far: generation ${top.gen} (score ${top.skill.toFixed(2)}).` : ''}`));
      insight.className = `verdict ${ahead ? 'good' : 'info'}`;
    }
    return;
  }
  const train = st.train || {};
  ui.charts.a.set([
    { name: 'intercepted', color: '#36dcff', points: series((r) => r.h2h.intercepted) },
    { name: 'through', color: '#ff4e3e', points: series((r) => r.h2h.leaked) },
  ]);
  legend($('#legend-a'), [['#36dcff', 'intercepted (wall wins)'], ['#ff4e3e', 'got through (swarm wins)']]);
  const bSeries = [];
  if (train.learn_defenders) bSeries.push({ name: 'def', color: '#36dcff', points: series((r) => r.defender_skill) });
  if (train.learn_attackers) bSeries.push({ name: 'att', color: '#ff4e3e', points: series((r) => r.attacker_skill) });
  ui.charts.b.set(bSeries);
  legend($('#legend-b'), [
    ...(train.learn_defenders ? [['#36dcff', 'defenders: % of built-in attackers intercepted']] : []),
    ...(train.learn_attackers ? [['#ff4e3e', 'attackers: % through built-in walls']] : []),
  ]);
  if (!H.length) return;
  const avg = (rows, f) => rows.reduce((a, r) => a + (f(r) ?? 0), 0) / Math.max(1, rows.length);
  const win = Math.max(3, Math.min(50, Math.round(H.length / 10)));  // long runs: compare averages, not single noisy rows
  const recent = H.slice(-win);
  const tugL = $('#tug-l'), tugR = $('#tug-r'), tug = $('#tug-bar');
  if (tug) {
    const icp = avg(recent, (r) => r.h2h.intercepted), lk = avg(recent, (r) => r.h2h.leaked);
    tug.style.width = `${100 * icp / Math.max(1e-9, icp + lk)}%`;
    tugL.textContent = `WALL ${pct(icp)}`;
    tugR.textContent = `${pct(lk)} SWARM`;
  }
  if (insight) {
    const first = H.slice(0, win);
    const dGain = train.learn_defenders ? avg(recent, (r) => r.defender_skill) - avg(first, (r) => r.defender_skill) : null;
    const aGain = train.learn_attackers ? avg(recent, (r) => r.attacker_skill) - avg(first, (r) => r.attacker_skill) : null;
    const leak = avg(recent, (r) => r.h2h.leaked);
    const lines = [];
    if (dGain !== null) lines.push(`Defenders ${dGain >= 0 ? '+' : ''}${Math.round(dGain * 100)} pts against the built-in attackers.`);
    if (aGain !== null) lines.push(`Attackers ${aGain >= 0 ? '+' : ''}${Math.round(aGain * 100)} pts against the built-in walls.`);
    let label = 'CONTESTED', tone = 'warn';
    if (leak < 0.1) { label = 'THE WALL IS WINNING'; tone = 'good'; } else if (leak > 0.3) { label = 'THE SWARM IS WINNING'; tone = 'bad'; }
    if (dGain !== null && aGain !== null && H.length >= 3) {
      lines.push(Math.abs(dGain - aGain) < 0.03 ? 'Both sides are learning at about the same pace.'
        : dGain > aGain ? 'The defenders are learning faster.' : 'The attackers are learning faster.');
    }
    lines.push(`Head to head, ${pct(leak)} of the attackers get through.`);
    const best = H[H.length - 1].best || {};
    if (best.defender) lines.push(`Best wall so far: generation ${best.defender.gen} (${pct(best.defender.skill)} intercepted).`);
    if (best.attacker) lines.push(`Best swarm so far: generation ${best.attacker.gen} (${pct(best.attacker.skill)} through).`);
    insight.className = `verdict ${tone}`;
    setChildren(insight, h('div', { class: 'lbl' }, label), h('p', {}, lines.join(' ')));
  }
}

async function saveTrained(role) {
  const T = S.train;
  const last = T.history.length ? T.history[T.history.length - 1] : null;
  const best = ((last && last.best) || (T.status && T.status.best) || {})[role];
  const skill = (v) => (role === 'recon' ? `score ${v.toFixed(2)} on the fixed test`
    : `${pct(v)} ${role === 'defender' ? 'intercepted against the built-in attacks' : 'through the built-in walls'}`);
  let which = best ? 'best' : 'latest';
  const opt = (value, text) => h('label', { class: 'tog', style: { display: 'block', margin: '4px 0' } },
    h('input', { type: 'radio', name: 'which', checked: which === value, onchange: () => { which = value; } }), text);
  const name = h('input', { type: 'text', value: `my_${role}`, maxlength: 40 });
  const title = h('input', { type: 'text', value: '', placeholder: 'optional display name', maxlength: 60 });
  const ok = await modal(`Save the ${role} brain`, h('div', {},
    h('p', { class: 'hint' }, 'It goes into the library (brains/) and appears in the brain lists of every test.'),
    best ? h('div', { style: { margin: '6px 0 10px' } },
      opt('best', `Best on the fixed skill test: generation ${best.gen} (${skill(best.skill)})`),
      opt('latest', `Latest: generation ${last ? last.gen : T.history.length}`)) : null,
    h('div', { class: 'field wide' }, h('label', {}, 'Name (letters, digits, _)'), name),
    h('div', { class: 'field wide', style: { marginTop: '8px' } }, h('label', {}, 'Title'), title)),
  [{ label: 'CANCEL', value: false }, { label: 'SAVE', cls: 'primary', value: true }]);
  if (!ok) return;
  try {
    const r = await api(`training/${T.id}/save`, { method: 'POST', body: { role, which, name: name.value, title: title.value } });
    await refreshCatalog();
    toast(`${r.brain.title} is in the library.`, 'ok', 'BRAIN SAVED');
  } catch (e) { toast(e.message, 'err', 'NOT SAVED'); }
}

async function deleteTraining(id) {
  const ok = await modal('Delete this training run?', h('p', {}, 'Its history, recorded matches and unsaved brains are removed. Brains you saved to the library stay.'),
    [{ label: 'CANCEL', value: false }, { label: 'DELETE', cls: 'danger', value: true }]);
  if (!ok) return;
  try {
    await api(`training/${id}`, { method: 'DELETE' });
    if (S.train.id === id) Object.assign(S.train, { id: null, status: null, history: [], match: null, shown: null });
    await refreshTrainings();
    if (S.tab === 'train') { renderTrainResults(); S.view.load(null); }
  } catch (e) { toast(e.message, 'err'); }
}

/** After a reload: pick up a training that is still running. */
async function reattach() {
  try {
    const ping = await api('ping');
    const tj = ping.jobs.find((j) => j.kind === 'train');
    if (tj && tj.meta && tj.meta.training) {
      S.train.job = tj.id;
      S.train.since = 0;  // replays every event; generations already read from disk are skipped
      await openTraining(tj.meta.training, { keepJob: true });
    } else if (S.train.list.length) {
      await openTraining(S.train.list[0].id);
    }
    const test = ping.jobs.find((j) => j.kind === 'test');
    if (test && test.meta && test.meta.test) S.testJob = { id: test.id, tab: test.meta.test };
  } catch { /* first ping failing is handled by the ping loop */ }
}

// ------------------------------------------------------------------ brains library
async function refreshCatalog() {
  try {
    const cat = await api('catalog');
    S.cat.brains = cat.brains;
    fixPicks();
    renderSetup();
    if ($('#drawer').classList.contains('show') && $('#drawer-title').textContent === 'BRAINS') renderBrains();
  } catch (e) { toast(e.message, 'err'); }
}

function pickUpload(role, onInstalled) {
  const input = $('#file');
  input.value = '';
  input.onchange = () => { if (input.files[0]) upload(input.files[0], role, onInstalled); };
  input.click();
}

async function upload(file, role, onInstalled) {
  if (!/\.(py|npz|zip)$/i.test(file.name)) { toast('Load a .py policy, trained .npz weights or a package .zip.', 'err', 'UNSUPPORTED FILE'); return; }
  try {
    const data = await file.arrayBuffer();
    const r = await api(`brains/upload?filename=${encodeURIComponent(file.name)}`, { method: 'POST', body: data, raw: true });
    await refreshCatalog();
    toast(`${r.brain.title} (${r.brain.role}) is in the library - running its checks...`, 'ok', 'BRAIN LOADED');
    if (role && r.brain.role !== role) toast(`It is a ${r.brain.role} brain, so it was not selected for the ${role} side.`, 'warn');
    else if (onInstalled) onInstalled(r.brain.id);
    watchValidation(r.job, r.brain.title);
  } catch (e) { toast(e.message, 'err', 'BRAIN REJECTED'); }
}

async function watchValidation(jobId, title) {
  try {
    const job = await waitJob(jobId, null, 600);
    const rep = job.result.report;
    const fails = rep.findings.filter((f) => f.level === 'fail');
    const rank = rep.findings.find((f) => f.check === 'ranking');
    const t = toast(fails.length ? fails[0].detail : (rank ? rank.detail : 'all checks passed'), fails.length ? 'err' : rep.status === 'warn' ? 'warn' : 'ok',
      `${title}: ${rep.status.toUpperCase()}`);
    t.style.cursor = 'pointer';
    t.onclick = () => showReport(title, rep);
  } catch (e) { toast(e.message, 'err', `${title}: CHECKS FAILED TO RUN`); }
}

function showReport(title, rep) {
  modal(`Checks: ${title}`, h('div', { class: 'report' }, rep.findings.map((f) => h('div', { class: 'f' },
    h('span', { class: f.level }, f.level.toUpperCase()), h('span', {}, f.check), h('span', {}, f.detail)))));
}

function openDrawer(title, body) {
  setChildren($('#drawer-title'), title);
  setChildren($('#drawer-body'), body);
  $('#drawer').classList.add('show');
}
function closeDrawer() { $('#drawer').classList.remove('show'); }

function openBrains() { renderBrains(); $('#drawer').classList.add('show'); }

function renderBrains() {
  const drop = h('div', { class: 'uploadbox' },
    h('div', { class: 'btnrow', style: { marginTop: 0, marginBottom: '6px' } },
      h('button', { class: 'btn primary', onclick: () => pickUpload(null) }, 'LOAD BRAIN'),
      h('button', { class: 'btn', onclick: () => openHelp('sdk') }, 'HOW TO WRITE ONE')),
    'Drop or load a .py file with a class that has act(obs) and role = "defender" / "attacker" / "recon", trained .npz weights from TRAIN, or a package .zip. It is checked and ranked against the built-ins right away.');
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('drag'); };
  drop.ondragleave = () => drop.classList.remove('drag');
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('drag'); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0], null); };
  const groups = [['recon', 'Recon swarm'], ['defender', 'Defenders - the wall'], ['attacker', 'Attackers - the swarm']];
  const row = (b) => {
    const acts = [];
    if (b.kind !== 'builtin') {
      const name = b.id.slice(4);
      acts.push(h('button', { class: 'btn', onclick: async () => {
        try { const r = await api(`brains/${name}/validate`, { method: 'POST' }); toast(`Checking ${b.title}...`); watchValidation(r.job, b.title); } catch (e) { toast(e.message, 'err'); }
      } }, 'CHECK'));
      acts.push(h('a', { class: 'btn', href: '#', onclick: (e) => { e.preventDefault(); downloadBrain(name); } }, 'DOWNLOAD'));
      if (b.kind === 'python') acts.push(h('button', { class: 'btn', onclick: () => showSource(name, b.title) }, 'SOURCE'));
      if (b.removable) acts.push(h('button', { class: 'btn danger', onclick: () => deleteBrain(name, b.title) }, 'DELETE'));
    }
    const use = { recon: 'recon', defender: 'defence', attacker: 'offence' }[b.role];
    if (use) acts.push(h('button', { class: 'btn', onclick: () => { S.picks[use][b.role] = b.id; save(); closeDrawer(); switchTab(use); } }, `TEST IT`));
    const trained = b.trained && b.trained.generations ? ` | trained ${b.trained.generations} generations` : '';
    return h('div', { class: `brain ${b.kind === 'broken' ? 'err' : ''}` },
      h('div', { class: 'top' }, h('b', {}, b.title), h('span', {}, badges(b))),
      h('p', {}, b.summary || ''),
      h('p', { style: { fontFamily: 'var(--mono)', fontSize: '11px' } }, `${b.id} | ${b.macs ? `${b.macs} MACs per decision` : 'cost not declared'}${trained}`),
      acts.length ? h('div', { class: 'acts' }, acts) : null);
  };
  const broken = S.cat.brains.filter((b) => b.kind === 'broken');
  openDrawer('BRAINS', [drop,
    groups.map(([role, label]) => [h('div', { class: 'role-h' }, label), S.cat.brains.filter((b) => b.role === role).map(row)]),
    broken.length ? [h('div', { class: 'role-h' }, 'Broken packages'), broken.map(row)] : null]);
}

async function downloadBrain(name) {
  try {
    const r = await fetch(`/api/brains/${name}/download`, { headers: { 'X-Studio-Token': sessionStorage.getItem('studio-token') || '' } });
    if (!r.ok) throw new Error((await r.json()).error);
    const url = URL.createObjectURL(await r.blob());
    const a = h('a', { href: url, download: `${name}.zip` });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (e) { toast(e.message, 'err'); }
}

async function showSource(name, title) {
  try {
    const r = await api(`brains/${name}/source`);
    modal(`${title} - ${r.file}`, highlightPython(r.text));
  } catch (e) { toast(e.message, 'err'); }
}

async function deleteBrain(name, title) {
  const ok = await modal(`Delete ${title}?`, h('p', {}, 'It is removed from the library (brains/).'),
    [{ label: 'CANCEL', value: false }, { label: 'DELETE', cls: 'danger', value: true }]);
  if (!ok) return;
  try { await api(`brains/${name}`, { method: 'DELETE' }); await refreshCatalog(); toast(`${title} deleted.`, 'ok'); } catch (e) { toast(e.message, 'err'); }
}

// ------------------------------------------------------------------ manual
async function openHelp(id) {
  try {
    const list = (await api('docs')).docs;
    const doc = await api(`docs/${id}`);
    openDrawer('MANUAL', [
      h('div', { class: 'doc-tabs' }, list.map((d) => h('button', { class: `btn ${d.id === id ? 'primary' : ''}`, onclick: () => openHelp(d.id) }, d.title))),
      markdown(doc.markdown)]);
    $('#drawer-body').scrollTop = 0;
  } catch (e) { toast(e.message, 'err'); }
}

// ------------------------------------------------------------------ server status
async function pingLoop() {
  let misses = 0;
  while (S.alive) {
    try {
      const p = await api('ping');
      misses = 0;
      const el = $('#jobstate');
      const busy = p.jobs.filter((j) => j.state === 'running' || j.state === 'queued');
      el.classList.toggle('busy', busy.length > 0);
      el.textContent = busy.map((j) => `${j.label}${j.progress ? `: ${j.progress}` : ''}`).join(' | ');
    } catch {
      if (++misses >= 2) { offline(); return; }
    }
    await new Promise((r) => setTimeout(r, 3000));
  }
}

async function pollLoop() {
  while (S.alive) {
    await pollTest();
    await pollTraining();
    await new Promise((r) => setTimeout(r, 700));
  }
}

async function shutdown() {
  const ok = await modal('Shut the studio down?', h('p', {}, 'Stops the local server and any test or training still running. Saved brains and reports stay on disk.'),
    [{ label: 'CANCEL', value: false }, { label: 'SHUT DOWN', cls: 'danger', value: true }]);
  if (!ok) return;
  try { await api('shutdown', { method: 'POST' }); } catch { /* it is going away */ }
  offline('The studio was shut down. Start it again with `python -m swarm.studio`.');
}

boot();
