// ARMORY: the brain & tactic library - browse, upload, scaffold, validate, compile, export for drones.

import { api, waitJob } from './api.js';
import { $, bytes, bus, clear, h, highlightPython, modal, setChildren, si, toast } from './util.js';

const GLYPH = {
  brain: '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1aa9c2" stroke-width="1.3"><circle cx="6" cy="7" r="2"/><circle cx="6" cy="17" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="12" cy="5" r="1.5"/><circle cx="12" cy="19" r="1.5"/><path d="M8 7l8 4.5M8 17l8-4.5M6 9v6M13.3 5.8l3.6 4.6M13.3 18.2l3.6-4.6"/></svg>',
  tactic: '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffb627" stroke-width="1.3"><path d="M12 3l3 5h-6z"/><circle cx="5" cy="18" r="2"/><circle cx="12" cy="18" r="2"/><circle cx="19" cy="18" r="2"/><path d="M12 8v8M11 9l-5 7M13 9l5 7"/></svg>',
  oracle: '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.3"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/></svg>',
};

function glyph(kind) {
  const s = h('span', { class: 'glyph' });
  s.innerHTML = GLYPH[kind]; // constant SVG markup, no user data
  return s;
}

const statusBadge = (s) => h('span', { class: `badge ${s === 'pass' ? 'pass' : s === 'warn' ? 'warn' : s === 'fail' ? 'fail' : 'pending'}` },
  s === 'pending' ? 'unvalidated' : s);

export class Armory {
  constructor(app) {
    this.app = app;
    this.catalog = null;
    this.sel = null;
    this.jobs = new Map();
    const dz = $('#dropzone'), input = $('#upload-input');
    input.onchange = () => { this.upload([...input.files]); input.value = ''; };
    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('over'));
    dz.addEventListener('drop', (e) => { e.preventDefault(); dz.classList.remove('over'); this.upload([...e.dataTransfer.files]); });
    $('#np-create').onclick = () => this.create();
  }

  setCatalog(cat) {
    this.catalog = cat;
    this.render();
    if (this.sel) this.detail(this.sel);
  }

  items() {
    const cat = this.catalog, out = [];
    for (const b of cat.brains) {
      if (b.kind === 'package') {
        const p = cat.packages.find((x) => x.root_rel === b.package);
        if (p) out.push({ id: b.id, group: 'brain', kind: 'package', pkg: p, brainId: b.id });
      } else out.push({ id: b.id, group: 'brain', kind: b.kind, brain: b, brainId: b.id });
    }
    for (const s of cat.strategies) {
      if (s.builtin) out.push({ id: `strategy:${s.name}`, group: 'tactic', kind: 'builtin', strategy: s });
      else {
        const p = cat.packages.find((x) => x.root_rel === s.package);
        if (p) out.push({ id: `package:${p.root_rel}`, group: 'tactic', kind: 'package', pkg: p, strategy: s });
      }
    }
    for (const p of cat.packages.filter((x) => x.broken)) out.push({ id: `broken:${p.root_rel}`, group: 'broken', kind: 'broken', pkg: p });
    return out;
  }

  render() {
    const box = clear($('#cards'));
    const items = this.items();
    $('#armory-count').textContent = `${items.length} ITEMS`;
    const section = (title, list) => {
      if (!list.length) return;
      box.appendChild(h('div', { class: 'section-t' }, title));
      for (const it of list) box.appendChild(this.card(it));
    };
    section('PERCEPTION BRAINS - what each drone concludes from its camera', items.filter((i) => i.group === 'brain'));
    section('COORDINATION TACTICS - how the swarm shares the work', items.filter((i) => i.group === 'tactic'));
    section('BROKEN PACKAGES', items.filter((i) => i.group === 'broken'));
  }

  card(it) {
    let name, desc, badges = [], facts = [], g = 'brain';
    if (it.kind === 'oracle') {
      name = 'ORACLE'; desc = it.brain.description; g = 'oracle';
      badges = [h('span', { class: 'badge violet' }, 'parametric')];
      facts = [['cost', 'any'], ['accuracy', 'any']];
    } else if (it.kind === 'model') {
      const inf = it.brain.info || {};
      name = it.brain.label; desc = it.brain.path;
      badges = [h('span', { class: 'badge info' }, '.swm'), h('span', { class: 'badge dim' }, 'models/')];
      facts = inf.error ? [['error', inf.error]] : [['MAC', si(inf.macs)], ['in', (inf.input_shape || []).join('x')], ['params', si(inf.params)], [inf.int8 ? 'int8' : 'f32', bytes(inf.model_bytes)]];
    } else if (it.kind === 'builtin') {
      name = it.strategy.name.toUpperCase(); desc = it.strategy.note; g = 'tactic';
      badges = [h('span', { class: 'badge dim' }, 'built-in'), it.strategy.baseline ? h('span', { class: 'badge info' }, 'baseline') : null];
    } else if (it.kind === 'broken') {
      name = it.pkg.name; desc = it.pkg.broken; badges = [h('span', { class: 'badge fail' }, 'broken')];
    } else {
      const p = it.pkg, v = p.validation || {}, info = v.info || {};
      name = p.title || p.name; desc = p.description;
      g = p.type === 'tactic' ? 'tactic' : 'brain';
      badges = [h('span', { class: 'badge info' }, p.type === 'tactic' ? 'tactic' : p.format), statusBadge(p.format === 'swm' && !p.model_built ? 'pending' : p.status),
        p.library === 'library' ? h('span', { class: 'badge dim' }, 'uploaded') : h('span', { class: 'badge dim' }, 'example')];
      if (p.format === 'swm' && !p.model_built) badges.push(h('span', { class: 'badge warn' }, 'not compiled'));
      if (p.validating) badges.push(h('span', { class: 'badge info' }, 'validating...'));
      facts = p.type === 'tactic' ? [['strategy', p.strategy]] : [['MAC', si(p.macs || info.macs)], ['in', (p.input_shape || info.input_shape || []).join('x') || '?'],
        ['acc', info.accuracy !== undefined && info.accuracy !== null ? `${Math.round(info.accuracy * 100)}%` : '?']];
    }
    return h('div', { class: `pcard ${this.sel === it.id ? 'sel' : ''}`, onclick: () => this.detail(it.id) },
      glyph(g),
      h('div', { class: 'top' }, h('span', { class: 'nm' }, name)),
      h('div', { class: 'btnrow' }, badges),
      h('div', { class: 'ds' }, desc || ''),
      h('div', { class: 'facts' }, facts.map(([k, v]) => h('span', {}, h('b', {}, k), v))));
  }

  detail(id) {
    this.sel = id;
    this.render();
    const it = this.items().find((x) => x.id === id);
    const box = $('#armory-detail');
    if (!it) { setChildren(box, h('div', { class: 'hdr' }, 'Dossier'), h('div', { class: 'body muted' }, 'Select a brain or a tactic.')); return; }
    const body = h('div', { class: 'body', style: { display: 'grid', gap: '12px', alignContent: 'start' } });
    setChildren(box, h('div', { class: 'hdr' }, 'Dossier', h('span', { class: 'r' }, id)), h('div', { class: 'scroll', style: { flex: 1 } }, body));
    const sect = (t) => h('div', { class: 'hdr', style: { padding: '6px 0', border: 0, borderBottom: '1px solid var(--line)' } }, t);

    if (it.kind === 'oracle' || it.kind === 'model') {
      const b = it.brain, inf = b.info || {};
      body.append(h('div', { class: 'nm', style: { font: '700 17px var(--font-ui)', letterSpacing: '.1em', color: 'var(--bright)' } }, b.label),
        h('div', { class: 'muted' }, b.description), h('div', { class: 'kv' },
          h('span', { class: 'k' }, 'deploys to'), h('span', { class: 'v' }, b.deploy),
          ...(it.kind === 'model' && !inf.error ? [
            h('span', { class: 'k' }, 'file'), h('span', { class: 'v' }, b.path),
            h('span', { class: 'k' }, 'MACs / frame'), h('span', { class: 'v' }, `${inf.macs} (${si(inf.macs)})`),
            h('span', { class: 'k' }, 'input'), h('span', { class: 'v' }, inf.input_shape.join(' x ')),
            h('span', { class: 'k' }, 'classes'), h('span', { class: 'v' }, inf.n_classes),
            h('span', { class: 'k' }, 'parameters'), h('span', { class: 'v' }, inf.params),
            h('span', { class: 'k' }, 'weights'), h('span', { class: 'v' }, `${inf.int8 ? 'int8' : 'f32'} · ${bytes(inf.weights_bytes)}`),
            h('span', { class: 'k' }, 'split point'), h('span', { class: 'v' }, `${si(inf.frontend_macs)}MAC frontend · ${bytes(inf.feature_bytes)} features`)] : [])),
        h('div', { class: 'btnrow' }, h('button', { class: 'btn primary', onclick: () => this.use(it) }, 'USE IN MISSION')));
      if (it.kind === 'model' && !inf.error) {
        body.append(sect('Estimated time per frame on each device'), this.estimateGrid(inf.macs),
          sect('Deploy'), h('pre', { class: 'code' }, `python setup_device.py prepare --config <node.json> \\\n  --model ${b.path} \\\n  --library <Pi-built swarm_brain.so> --output <NEW folder> --target linux-aarch64\n\n# or package it first for validation/export:\npython -m swarm.sdk install ${b.path}`));
      }
      return;
    }
    if (it.kind === 'builtin') {
      const s = it.strategy;
      body.append(h('div', { style: { font: '700 17px var(--font-ui)', letterSpacing: '.1em', color: 'var(--bright)' } }, s.name.toUpperCase()),
        h('div', {}, s.note),
        h('div', { class: 'muted' }, 'Built into the simulator (python/swarm/sim/strategies.py). To try a variation, scaffold a tactic package: it starts as a copy of the striped design with every decision marked.'),
        h('div', { class: 'btnrow' }, h('button', { class: 'btn primary', onclick: () => this.use(it) }, 'USE IN MISSION'),
          h('button', { class: 'btn', onclick: () => { $('#np-type').value = 'tactic'; $('#np-name').focus(); } }, 'WRITE MY OWN')));
      return;
    }
    const p = it.pkg;
    if (it.kind === 'broken') {
      body.append(h('div', { class: 'bad' }, p.broken), h('div', { class: 'muted' }, p.root_rel));
      return;
    }
    const v = p.validation;
    const actions = [h('button', { class: 'btn primary', onclick: () => this.use(it) }, 'USE IN MISSION'),
      h('button', { class: 'btn', onclick: () => this.validate(p) }, 'VALIDATE')];
    if (p.build) actions.push(h('button', { class: 'btn execute', style: { padding: '6px 11px', fontSize: '11px' }, onclick: () => this.compile(p) }, p.model_built ? 'RECOMPILE' : 'COMPILE'));
    if (p.type === 'perception') actions.push(h('button', { class: 'btn', onclick: () => this.export(p) }, 'EXPORT FOR DEVICE'));
    if (p.removable) actions.push(h('button', { class: 'btn danger', onclick: () => this.remove(p) }, 'DELETE'));
    body.append(
      h('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
        h('span', { style: { font: '700 17px var(--font-ui)', letterSpacing: '.08em', color: 'var(--bright)' } }, p.title || p.name),
        statusBadge(p.format === 'swm' && !p.model_built ? 'pending' : p.status)),
      h('div', { class: 'muted' }, p.description || ''),
      h('div', { class: 'kv' },
        h('span', { class: 'k' }, 'package'), h('span', { class: 'v' }, `${p.name} v${p.version}`),
        h('span', { class: 'k' }, 'type'), h('span', { class: 'v' }, p.type === 'tactic' ? `tactic -> strategy "${p.strategy}"` : `perception / ${p.format}`),
        h('span', { class: 'k' }, 'entry'), h('span', { class: 'v' }, p.entry || p.model || '-'),
        h('span', { class: 'k' }, 'location'), h('span', { class: 'v' }, p.root_rel),
        h('span', { class: 'k' }, 'author'), h('span', { class: 'v' }, p.author || '-')),
      h('div', { class: 'btnrow' }, actions),
      h('div', { id: 'armory-job' }));
    const job = this.jobs.get(p.root_rel);
    if (job) this.showJob(job);
    if (p.format === 'swm' && !p.model_built) {
      body.append(h('div', { class: 'warnc' }, 'This network is not compiled yet: COMPILE runs its build script (' + (p.build || 'train.py') + ') which trains it and writes the .swm the drones run.'));
    }
    if (v) {
      body.append(sect(`Validation · ${v.status.toUpperCase()} · ${v.seconds ? v.seconds.toFixed(1) + ' s' : ''}`),
        h('div', { class: 'findings' }, v.findings.map((fd) => h('div', {}, h('span', { class: `lv ${fd.level}` }, fd.level.toUpperCase()), h('span', { class: 'ck' }, fd.check), h('span', {}, fd.detail)))));
      const info = v.info || {};
      if (info.devices) {
        body.append(sect('Device fit - can each drone run it?'), h('div', { class: 'fitgrid' }, info.devices.map((d) => h('div', {
          class: `c ${d.fits === true ? 'yes' : d.fits === false ? 'no' : 'na'}`, title: `${d.runtime}${d.ram_needed ? ` · needs ${bytes(d.ram_needed)} of ${bytes(d.ram_cap)}` : ''}`,
        }, h('b', {}, d.device), d.fits === false ? (d.runtime === 'needs Linux' ? 'NO LINUX' : 'NO FIT') : `${d.ms_per_frame < 0.1 ? d.ms_per_frame.toFixed(3) : d.ms_per_frame.toFixed(2)} ms`, h('br'), h('span', { style: { color: 'var(--dim)' } }, d.runtime)))));
      }
      if (info.conformance) {
        body.append(sect('Conformance runs (oracle brain)'), h('table', { class: 'grid' },
          h('tr', {}, ['run', 'frames', 'recall', 'kbit/s', 'busy max', 'recover s'].map((c) => h('th', { class: c === 'run' ? '' : 'num' }, c))),
          info.conformance.map((r) => h('tr', {}, h('td', {}, r.run), h('td', { class: 'num' }, r.frames_processed),
            h('td', { class: 'num' }, r.recall === null ? 'n/a' : `${Math.round(r.recall * 100)}%`), h('td', { class: 'num' }, r.kbps.toFixed(1)),
            h('td', { class: 'num' }, `${Math.round(r.busy_max * 100)}%`), h('td', { class: 'num' }, r.recover_s === null ? '-' : r.recover_s === 'inf' ? 'never' : r.recover_s.toFixed(2))))));
      }
    } else if (!(p.format === 'swm' && !p.model_built)) {
      body.append(h('div', { class: 'muted' }, p.validating ? 'Validation running...' : 'Not validated yet.'));
    }
    body.append(sect('Files'), h('ul', { class: 'files' }, p.files.map((fl) => h('li', {
      onclick: () => /\.(py|json|md|txt)$/.test(fl.path) ? this.source(p, fl.path) : toast(`${fl.path}: binary file (${bytes(fl.bytes)})`, 'info'),
    }, fl.path, h('span', {}, bytes(fl.bytes))))));
  }

  estimateGrid(macs) {
    return h('div', { class: 'fitgrid' }, this.catalog.devices.filter((d) => d.name !== 'host').map((d) => {
      const ms = macs / (d.gmacs * 1e9) * 1e3;
      return h('div', { class: `c ${ms < 100 ? 'yes' : 'no'}` }, h('b', {}, d.name), `${ms < 0.1 ? ms.toFixed(3) : ms.toFixed(2)} ms`);
    }));
  }

  showJob(job) {
    const el = $('#armory-job');
    if (!el) return;
    const running = job.state === 'queued' || job.state === 'running';
    setChildren(el, h('div', { class: 'row', style: { marginBottom: '5px' } }, running ? h('div', { class: 'spinner' }) : h('span', { class: `badge ${job.state === 'done' ? 'pass' : 'fail'}` }, job.state),
      h('span', { class: 'muted' }, `${job.label} · ${job.progress || job.state} · ${job.elapsed ? job.elapsed.toFixed(1) : 0} s`)),
      job.log && job.log.length ? h('div', { class: 'joblog' }, job.log.slice(-40).join('\n')) : null,
      job.error ? h('div', { class: 'bad', style: { marginTop: '5px' } }, job.error) : null);
    const log = el.querySelector('.joblog');
    if (log) log.scrollTop = log.scrollHeight;
  }

  async track(p, jobId, what) {
    try {
      const job = await waitJob(jobId, (j) => { this.jobs.set(p.root_rel, j); if (this.sel === `package:${p.root_rel}`) this.showJob(j); });
      const st = job.result && job.result.report ? job.result.report.status : 'done';
      toast(`${p.name}: ${what} finished - ${st.toUpperCase()}`, st === 'fail' ? 'err' : st === 'warn' ? 'warn' : 'ok', 'ARMORY');
    } catch (e) {
      toast(`${p.name}: ${e.message}`, 'err', `${what.toUpperCase()} FAILED`);
    }
    await this.app.refreshCatalog();
  }

  async validate(p) {
    const { job } = await api('packages/validate', { method: 'POST', body: { path: p.root_rel } });
    bus.emit('job', job, `validate ${p.name}`);
    this.track(p, job, 'validation');
  }

  async compile(p) {
    try {
      const { job } = await api('packages/build', { method: 'POST', body: { path: p.root_rel } });
      bus.emit('job', job, `compile ${p.name}`);
      toast(`compiling ${p.name}: training the network and exporting the .swm`, 'info', 'COMPILE');
      this.track(p, job, 'compile');
    } catch (e) { toast(e.message, 'err', 'COMPILE'); }
  }

  async export(p) {
    try {
      const res = await api('packages/export', { method: 'POST', body: { path: p.root_rel } });
      await modal(`EXPORT FOR DEVICE · ${p.name}`, h('div', {},
        h('div', { class: 'muted', style: { marginBottom: '8px' } }, `Artefacts written to ${res.out_rel} (DEPLOY.txt inside). Nothing was signed, copied or run on a device.`),
        h('pre', { class: 'code', style: { maxHeight: '60vh', whiteSpace: 'pre-wrap' } }, res.instructions)));
    } catch (e) { toast(e.message, 'err', 'EXPORT REFUSED'); }
  }

  async remove(p) {
    const ok = await modal('DELETE PACKAGE?', h('div', {}, `Delete ${p.root_rel}? The folder is removed from the library.`),
      [{ label: 'CANCEL', value: false }, { label: 'DELETE', cls: 'danger', value: true }]);
    if (!ok) return;
    try { await api('packages/delete', { method: 'POST', body: { path: p.root_rel } }); toast(`${p.name} deleted`, 'ok'); this.sel = null; await this.app.refreshCatalog(); }
    catch (e) { toast(e.message, 'err'); }
  }

  async source(p, file) {
    try {
      const res = await api(`packages/source?path=${encodeURIComponent(p.root_rel)}&file=${encodeURIComponent(file)}`);
      await modal(`${p.name} / ${file}`, file.endsWith('.py') ? highlightPython(res.text) : h('pre', { class: 'code' }, res.text));
    } catch (e) { toast(e.message, 'err'); }
  }

  use(it) {
    const planner = this.app.planner;
    if (it.group === 'brain') { planner.useBrain(it.brainId); toast(`brain set in MISSION PLANNER`, 'ok'); }
    else { planner.useTactic(it.strategy.name); toast(`"${it.strategy.name}" added to the designs under test`, 'ok'); }
    this.app.show('planner');
  }

  async create() {
    const name = $('#np-name').value.trim(), type = $('#np-type').value;
    try {
      const res = await api('packages/new', { method: 'POST', body: { name, type } });
      toast(`brains/${name} created - edit it in your editor; the SDK manual is in FIELD MANUAL`, 'ok', 'PACKAGE CREATED');
      $('#np-name').value = '';
      await this.app.refreshCatalog();
      this.detail(`package:${res.package.root_rel}`);
      if (res.job) this.track(res.package, res.job, 'validation');
    } catch (e) { toast(e.message, 'err', 'CREATE REFUSED'); }
  }

  async upload(files) {
    for (const file of files) {
      const send = (replace) => api(`packages/upload?filename=${encodeURIComponent(file.name)}&replace=${replace ? 1 : 0}`, { method: 'POST', body: file, raw: true });
      try {
        let res;
        try { res = await send(false); }
        catch (e) {
          if (!/already installed/.test(e.message)) throw e;
          const ok = await modal('REPLACE PACKAGE?', h('div', {}, e.message), [{ label: 'CANCEL', value: false }, { label: 'REPLACE', cls: 'danger', value: true }]);
          if (!ok) continue;
          res = await send(true);
        }
        toast(`${file.name} installed as ${res.package.name} - validating`, 'ok', 'UPLOAD');
        await this.app.refreshCatalog();
        this.detail(`package:${res.package.root_rel}`);
        this.track(res.package, res.job, 'validation');
      } catch (e) {
        toast(`${file.name}: ${e.message}`, 'err', 'UPLOAD REFUSED');
      }
    }
  }
}
