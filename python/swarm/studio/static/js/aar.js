// AFTER-ACTION: the battery's report for a run - every design vs the baseline - plus the run archive.

import { api } from './api.js';
import { $, bytes, clear, download, fmtRatio, h, isNum, modal, pyfmt, setChildren, toast } from './util.js';

const TOL = 0.02;

export class AAR {
  constructor(app) {
    this.app = app;
    this.run = null;
  }

  async refreshRuns() {
    let runs = [];
    try { runs = (await api('runs')).runs; } catch (e) { toast(e.message, 'err'); }
    const ul = clear($('#runs-list'));
    $('#runs-count').textContent = `${runs.length}`;
    for (const r of runs) {
      const d = new Date(r.created * 1000);
      ul.appendChild(h('li', { class: this.run && this.run.id === r.id ? 'sel' : '', onclick: () => this.app.openRun(r.id, 'aar') },
        h('div', { class: 'n' }, r.name),
        h('div', { class: 'd' }, `${d.toLocaleDateString()} ${d.toLocaleTimeString()} · ${r.nodes} assets · ${r.strategies.length} designs`),
        h('div', { class: 'd' }, `brain ${r.brain} · faults ${r.faults} · checks `, h('span', { class: r.checks_passed === r.checks ? 'good' : 'bad' }, r.checks ? `${r.checks_passed}/${r.checks}` : '-'))));
    }
    if (!runs.length) ul.appendChild(h('li', { class: 'muted' }, 'No runs yet - EXECUTE a mission.'));
  }

  show(run) {
    this.run = run;
    const cat = this.app.catalog;
    const box = clear($('#aar'));
    const sc = run.scenario;
    const base = run.strategies.find((s) => s.name === sc.baseline) || run.strategies[0];
    const others = run.strategies.filter((s) => s !== base);
    const passed = run.checks.filter((c) => c.passed).length;
    box.append(
      h('div', {}, h('div', { class: 'muted', style: { letterSpacing: '.3em', fontSize: '10px' } }, 'AFTER-ACTION REPORT'),
        h('h2', {}, sc.name.toUpperCase()), h('div', { class: 'lead' }, sc.description)),
      h('div', { class: 'btnrow' },
        h('button', { class: 'btn primary', onclick: () => this.app.openRun(run.id, 'ops') }, 'REPLAY IN OPS'),
        h('button', { class: 'btn', onclick: () => download(`${sc.name}-${run.id}.json`, JSON.stringify(run)) }, 'DOWNLOAD JSON'),
        h('button', { class: 'btn danger', onclick: () => this.remove(run) }, 'DELETE RUN'),
        h('span', { class: 'muted' }, `run ${run.id} · brain ${run.brain.kind} (${run.brain.description}) · ${run.resolved.duration} s simulated in ${run.wall_s.toFixed(1)} s wall`)),
      run.brain.note ? h('div', { class: 'warnc' }, 'note: ' + run.brain.note) : null);

    if (run.checks.length) {
      box.append(h('div', { class: 'card' }, h('div', { class: 'hdr' }, `Measures of effectiveness · ${passed}/${run.checks.length} passed`),
        h('div', { class: 'cbody findings' }, run.checks.map((c) => h('div', {},
          h('span', { class: `lv ${c.passed ? 'pass' : 'fail'}` }, c.passed ? 'PASS' : 'FAIL'),
          h('span', { class: 'ck' }, c.strategy),
          h('span', {}, `${c.metric}${c.ratio ? '/baseline' : ''} ${c.op} ${c.value}   (actual ${isNum(c.actual) ? Number(c.actual.toPrecision(4)) : 'n/a'})${c.note ? '  # ' + c.note : ''}`))))));
    }

    const table = (title, rows) => {
      const t = h('table', { class: 'grid metrics' },
        h('tr', {}, h('th', {}, 'metric'), h('th', { class: 'strat' }, `${base.name} (baseline)`), others.map((o) => h('th', { class: 'strat' }, o.name))));
      for (const r of rows) {
        const b = base.metrics[r.key];
        const tag = r.better === '+' ? ' [+]' : r.better === '-' ? ' [-]' : '';
        t.appendChild(h('tr', {}, h('td', {}, r.label + tag), h('td', { class: 'v base' }, pyfmt(b, r.fmt)),
          others.map((o) => {
            const v = o.metrics[r.key];
            const ratio = fmtRatio(v, b);
            let cls = 'dimc';
            if (isNum(v) && isNum(b) && r.better) {
              const better = r.better === '+' ? v > b * (1 + TOL) : v < b * (1 - TOL);
              const worse = r.better === '+' ? v < b * (1 - TOL) : v > b * (1 + TOL);
              cls = better ? 'good' : worse ? 'bad' : 'dimc';
            }
            return h('td', { class: 'v' }, pyfmt(v, r.fmt), ratio ? h('span', { class: `ratio ${cls}` }, ratio) : null);
          })));
      }
      return h('div', { class: 'card' }, h('div', { class: 'hdr' }, title, h('span', { class: 'r' }, '[+] higher is better · [-] lower is better')), h('div', { class: 'scroll' }, t));
    };
    box.append(table('Mission & speed', cat.metrics.mission), table('Compute, radio, energy', cat.metrics.cost));
    if ((sc.faults.kill || []).length || (sc.faults.revive || []).length) box.append(table('Survivability (vs a fault-free twin run)', cat.metrics.survival));

    const charts = [['recall', 'Recall', '{:.0%}', '+'], ['latency_p50_ms', 'Detection latency p50 (ms)', '{:.1f}', '-'], ['kbps', 'Radio (kbit/s)', '{:.1f}', '-'],
      ['gmac_ps', 'Compute (GMAC/s)', '{:.3f}', '-'], ['busy_max', 'Busiest CPU', '{:.0%}', '-'], ['energy_j', 'Energy (J)', '{:.3f}', '-']];
    if (sc.faults.kill && sc.faults.kill.length) charts.push(['work_retained', 'Work retained vs no-fault', '{:.0%}', '+']);
    box.append(h('div', { class: 'charts' }, charts.map(([k, title, fmt, dir]) => {
      const vals = run.strategies.map((s) => s.metrics[k]).filter(isNum);
      if (!vals.length) return null;
      const max = Math.max(...vals, 1e-12);
      const best = dir === '+' ? Math.max(...vals) : Math.min(...vals);
      return h('div', { class: 'chart' }, h('div', { class: 't' }, title), run.strategies.map((s) => {
        const v = s.metrics[k];
        const w = isNum(v) ? Math.max(1, (v / max) * 100) : 0;
        return h('div', { class: 'bar' }, h('span', { class: 'mono', style: { overflow: 'hidden', textOverflow: 'ellipsis' } }, s.name),
          h('div', { class: 'track' }, h('i', { class: s === base ? 'base' : v === best ? 'best' : '', style: { width: `${w}%` } })),
          h('span', { class: 'v' }, pyfmt(v, fmt)));
      }));
    })));

    if (run.stress && run.stress.length) {
      const rows = run.stress;
      box.append(h('div', { class: 'card' }, h('div', { class: 'hdr' }, run.brain.model_path ? `Device stress · ${run.brain.model_path} under each device's RAM cap` : 'Compute budget per device'),
        h('div', { class: 'scroll' }, h('table', { class: 'grid' },
          h('tr', {}, ['device', 'RAM', 'need / cap', 'ms/frame', 'max fps', 'host ms', 'accuracy'].map((c) => h('th', {}, c))),
          rows.map((s) => h('tr', {}, h('td', {}, s.device),
            h('td', { class: s.fits_ram === false ? 'bad' : s.fits_ram ? 'good' : 'dimc' }, s.fits_ram === null ? '?' : s.fits_ram ? 'FITS' : 'OVERFLOW'),
            h('td', { class: 'num' }, s.ram_needed ? `${bytes(s.ram_needed)} / ${bytes(s.ram_cap)}` : bytes(s.ram_cap)),
            h('td', { class: 'num' }, pyfmt(s.device_ms_per_frame, '{:.3f}')), h('td', { class: 'num' }, pyfmt(s.max_fps, '{:.1f}')),
            h('td', { class: 'num' }, pyfmt(s.host_ms_per_frame, '{:.3f}')), h('td', { class: 'num' }, pyfmt(s.accuracy, '{:.0%}'))))))));
    }
    box.append(h('div', { class: 'muted', style: { fontSize: '11px' } },
      'Same numbers as `python -m swarm.bench`: costs come only from device profiles and the shared channel model, never from wall-clock time.'));
    this.refreshRuns();
  }

  async remove(run) {
    const ok = await modal('DELETE RUN?', h('div', {}, `Delete run ${run.id} (${run.scenario.name}) from out/studio/runs?`),
      [{ label: 'CANCEL', value: false }, { label: 'DELETE', cls: 'danger', value: true }]);
    if (!ok) return;
    try { await api(`runs/${run.id}`, { method: 'DELETE' }); toast('run deleted', 'ok'); this.run = null; setChildren($('#aar'), h('div', { class: 'muted' }, 'Run deleted.')); this.refreshRuns(); }
    catch (e) { toast(e.message, 'err'); }
  }
}
