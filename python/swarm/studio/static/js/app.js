// Boot, routing and shared state. The session token arrives in the launch URL's #fragment (never sent to the
// server in a URL, never in a Referer) and is kept in sessionStorage for reloads of this tab.

import { AAR } from './aar.js';
import { api, hasToken, setToken, waitJob } from './api.js';
import { Armory } from './armory.js';
import { Manual } from './manual.js';
import { Ops } from './ops.js';
import { Planner } from './planner.js';
import { $, $$, bus, h, modal, setChildren, sleep, toast } from './util.js';

const VIEWS = ['ops', 'planner', 'armory', 'aar', 'manual'];
const DEFAULT_MISSION = 's02_leader_loss';

class App {
  constructor() {
    const m = /[#&]t=([\w-]+)/.exec(location.hash);
    if (m) { sessionStorage.setItem('swarm-studio-token', m[1]); history.replaceState(null, '', location.pathname); }
    setToken(sessionStorage.getItem('swarm-studio-token'));
    // a restarted studio's link opened in this tab only changes the #fragment: take the new token and reboot
    window.addEventListener('hashchange', () => {
      const t = /[#&]t=([\w-]+)/.exec(location.hash);
      if (t) { sessionStorage.setItem('swarm-studio-token', t[1]); history.replaceState(null, '', location.pathname); location.reload(); }
    });
    this.catalog = null;
    this.jobs = new Map();
    $$('#tabs .tab').forEach((b) => { b.onclick = () => this.show(b.dataset.view); });
    window.addEventListener('keydown', (e) => {
      const i = ['F1', 'F2', 'F3', 'F4', 'F5'].indexOf(e.key);
      if (i >= 0) { e.preventDefault(); this.show(VIEWS[i]); }
    });
    $('#menu-btn').onclick = (e) => {
      const menu = $('#menu');
      const r = e.currentTarget.getBoundingClientRect();
      menu.style.top = `${r.bottom + 2}px`;
      menu.style.right = `${window.innerWidth - r.right}px`;
      menu.classList.toggle('show');
      e.stopPropagation();
    };
    document.addEventListener('click', () => $('#menu').classList.remove('show'));
    $$('#menu button').forEach((b) => { b.onclick = () => this.menu(b.dataset.act); });
    bus.on('job', (id, label) => this.trackJob(id, label));
  }

  async line(text, cls = '', pause = 70) {
    const el = h('div', {}, cls ? h('span', { class: cls }, text.slice(0, 6)) : null, cls ? text.slice(6) : text);
    $('#boot-lines').appendChild(el);
    await sleep(pause);
    return el;
  }

  async boot() {
    await this.line('SWARM//OPS TACTICAL CONSOLE - SECURE LOCAL SESSION', '', 120);
    if (!hasToken()) {
      await this.line('[FAIL] NO SESSION TOKEN', 'er');
      $('#boot-deny').style.display = 'block';
      return;
    }
    let cat;
    try {
      cat = await api('catalog');
    } catch (e) {
      await this.line(`[FAIL] ${e.message}`, 'er');
      $('#boot-deny').style.display = e.status === 403 ? 'block' : 'none';
      return;
    }
    await this.line('[ OK ] LINK 127.0.0.1 AUTHENTICATED (SESSION TOKEN)', 'ok');
    await this.line('[ OK ] SIMULATION KERNEL: DETERMINISTIC DISCRETE-EVENT, COSTS FROM DEVICE PROFILES', 'ok');
    const eng = cat.engine;
    const info = (eng.info || '').replace(/\s*\([^)]*\)/g, '');
    await this.line(eng.native ? `[ OK ] BRAIN ENGINE: NATIVE C++ (${info})` : '[WARN] BRAIN ENGINE: NUMPY FALLBACK (python dev.py build for the C++ engine)', eng.native ? 'ok' : 'wn');
    await this.line(`[ OK ] ARMORY: ${cat.brains.length} BRAINS, ${cat.strategies.length} SWARM DESIGNS, ${cat.packages.length} PACKAGES`, 'ok');
    this.catalog = cat;
    if (cat.presets_pending) {
      const el = await this.line('[ .. ] INDEXING MISSION LIBRARY', 'wn');
      cat = await this.refreshCatalog(true, false);
      el.firstChild.textContent = '[ OK ]';
      el.firstChild.className = 'ok';
    }
    await this.line(`[ OK ] MISSION LIBRARY: ${cat.presets.length} PRESETS`, 'ok');
    this.ops = new Ops();
    this.planner = new Planner(this);
    this.armory = new Armory(this);
    this.aar = new AAR(this);
    this.manual = new Manual(this);
    this.distribute(cat);
    $('#st-engine').textContent = eng.native ? 'NATIVE C++' : 'NUMPY';
    $('#st-engine').className = 'v ' + (eng.native ? 'ok' : 'warn');
    this.manual.init();
    this.aar.refreshRuns();
    this.ping();

    const runs = (await api('runs')).runs;
    const preset = cat.presets.find((p) => p.name === DEFAULT_MISSION) || cat.presets.find((p) => !p.error);
    if (preset) this.planner.load(preset);
    if (runs.length) {
      await this.line(`[ OK ] RESUMING LAST OPERATION ${runs[0].name.toUpperCase()}`, 'ok', 200);
      this.finishBoot();
      await this.openRun(runs[0].id, 'ops');
    } else if (preset) {
      await this.line(`[ .. ] FIRST LAUNCH: EXECUTING ${preset.name.toUpperCase()}`, 'wn', 200);
      this.finishBoot();
      this.planner.execute();
    } else {
      this.finishBoot();
      this.show('planner');
    }
  }

  finishBoot() {
    $('#boot').classList.add('gone');
    setTimeout(() => $('#boot').remove(), 600);
  }

  show(view) {
    $$('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${view}`));
    $$('#tabs .tab').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
    if (view === 'manual' && this.manual && !this.manual.current) this.manual.open('studio');
    if (view === 'aar' && this.aar) this.aar.refreshRuns();
  }

  distribute(cat) {
    this.catalog = cat;
    if (this.planner) this.planner.setCatalog(cat);
    if (this.armory) this.armory.setCatalog(cat);
  }

  async refreshCatalog(waitPresets = false, distribute = true) {
    let cat = await api('catalog');
    for (let k = 0; waitPresets && cat.presets_pending && k < 120; k++) { await sleep(500); cat = await api('catalog'); }
    if (distribute) this.distribute(cat); else this.catalog = cat;
    if (cat.presets_pending) setTimeout(() => this.refreshCatalog(true), 1500);
    return cat;
  }

  async openRun(id, view = 'ops') {
    try {
      const run = await this.ops.loadRun(id);
      this.aar.show(run);
      this.show(view);
    } catch (e) {
      toast(e.message, 'err', 'CANNOT OPEN RUN');
    }
  }

  /** Waits for a simulation job with a progress banner over the theatre, then opens the run. */
  async runJob(jobId, runId, label) {
    this.show('ops');
    const center = $('#hud-center');
    const banner = h('div', { class: 'banner-msg', style: { minWidth: '420px' } }, h('div', {}, label), h('div', { class: 'progressbar', style: { margin: '8px 0 6px' } }),
      h('div', { class: 'mono', style: { fontSize: '11px', letterSpacing: '.05em', color: 'var(--text)' } }, 'queued'));
    setChildren(center, banner);
    $('#ops-empty').classList.add('hidden');
    try {
      await waitJob(jobId, (j) => { banner.lastChild.textContent = `${j.progress || j.state} · ${j.elapsed.toFixed(1)} s`; });
      setChildren(center);
      await this.openRun(runId, 'ops');
      toast(`${label.replace('EXECUTING ', '')}: simulation complete - every design replayable`, 'ok', 'MISSION COMPLETE');
    } catch (e) {
      setChildren(center, h('div', { class: 'banner-msg red' }, 'MISSION FAILED', h('div', { class: 'mono', style: { fontSize: '11px', marginTop: '6px', letterSpacing: 0, maxWidth: '640px', whiteSpace: 'pre-wrap', textAlign: 'left' } },
        e.message + (e.job && e.job.log ? '\n' + e.job.log.slice(-8).join('\n') : ''))));
      setTimeout(() => setChildren(center), 12000);
      throw e;
    }
  }

  trackJob(id, label) {
    this.jobs.set(id, label);
    this.updateJobs();
    waitJob(id).catch(() => {}).finally(() => { this.jobs.delete(id); this.updateJobs(); });
  }

  updateJobs() {
    const pill = $('#jobs-pill');
    pill.classList.toggle('show', this.jobs.size > 0);
    $('#jobs-text').textContent = this.jobs.size ? [...this.jobs.values()].slice(-1)[0] + (this.jobs.size > 1 ? ` +${this.jobs.size - 1}` : '') : '-';
  }

  async ping() {
    while (!this.stopped) {
      try { await api('ping'); $('#st-link').textContent = 'LOCAL'; $('#st-link').className = 'v ok'; }
      catch { $('#st-link').textContent = 'LOST'; $('#st-link').className = 'v bad'; }
      await sleep(5000);
    }
  }

  async menu(act) {
    $('#menu').classList.remove('show');
    if (act === 'rerun') {
      const run = this.ops && this.ops.run;
      if (!run) { toast('no operation loaded', 'warn'); return; }
      try {
        const { job, run: rid } = await api('runs', { method: 'POST', body: { scenario: run.scenario, theatre: run.theatre, with_stress: false } });
        bus.emit('job', job, `simulate ${run.scenario.name}`);
        await this.runJob(job, rid, `EXECUTING ${run.scenario.name.toUpperCase()}`);
      } catch (e) { toast(e.message, 'err', 'RE-RUN FAILED'); }
    } else if (act === 'runs') this.show('aar');
    else if (act === 'catalog') { await this.refreshCatalog(true); toast('catalog refreshed', 'ok'); }
    else if (act === 'help') {
      const rows = [['F1..F5', 'switch view'], ['Space', 'play / pause'], ['Left / Right', 'step 0.1 s (Shift: 1 s)'], ['[ / ]', 'previous / next event'],
        ['Home / End', 'start / end'], ['1..9', 'switch swarm design'], ['V', 'track swarm'], ['T', 'top-down'], ['F', 'follow selected asset'],
        ['L', 'legend'], ['Esc', 'clear selection'], ['drag / right-drag / wheel', 'orbit / pan / zoom']];
      modal('KEYBOARD', h('table', { class: 'grid' }, rows.map(([k, v]) => h('tr', {}, h('td', { class: 'cyanc' }, k), h('td', {}, v)))));
    } else if (act === 'shutdown') {
      const ok = await modal('SHUT DOWN STUDIO?', h('div', {}, 'Stops the local server and any running jobs. Start it again with python -m swarm.studio.'),
        [{ label: 'CANCEL', value: false }, { label: 'SHUT DOWN', cls: 'danger', value: true }]);
      if (!ok) return;
      try { await api('shutdown', { method: 'POST' }); } catch { /* server is going away */ }
      this.stopped = true;
      if (this.ops) this.ops.stopped = true;
      document.body.appendChild(h('div', { style: { position: 'fixed', inset: '0', zIndex: 200, background: '#010305', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '10px', letterSpacing: '.3em' } },
        h('b', { style: { color: '#ff3b4d', fontSize: '20px' } }, 'STUDIO OFFLINE'), h('span', { class: 'muted' }, 'python -m swarm.studio to start a new session')));
    }
  }
}

const app = new App();
window.addEventListener('error', (e) => toast(String(e.message || e), 'err', 'CONSOLE ERROR'));
app.boot().catch((e) => { console.error(e); toast(e.message, 'err', 'BOOT FAILED'); });
