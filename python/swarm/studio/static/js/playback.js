// Turns one strategy's trace into time-indexed queries for the renderer, the panels and the HUD.
// Row layouts are swarm.sim.trace.SCHEMA; every timestamp is simulated time (seconds).

export const KIND_COLOR = {
  frame: [0.17, 0.89, 1.0], features: [0.65, 0.55, 0.98], result: [0.24, 1.0, 0.63], cmd: [1.0, 0.71, 0.15],
  hb: [0.8, 0.91, 1.0],
};
const OTHER = [0.85, 0.85, 0.85];

function lastAtOrBefore(times, t) {
  let lo = 0, hi = times.length - 1, ans = -1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (times[mid] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1; }
  return ans;
}

/** Piecewise-constant boolean timeline: [[t, value], ...] sorted by t. */
class Track {
  constructor(initial) { this.t = []; this.v = []; this.init = initial; }
  push(t, v) { this.t.push(t); this.v.push(v); }
  at(t) { const i = lastAtOrBefore(this.t, t); return i < 0 ? this.init : this.v[i]; }
}

export class Playback {
  constructor(run, si) {
    this.run = run;
    this.si = si;
    this.s = run.strategies[si];
    const tr = this.s.trace;
    this.n = run.profiles.length;
    this.T = run.resolved.duration;
    this.kinds = tr.kinds || [];
    this.tags = tr.tags || [];
    this.truncated = !!tr.truncated;
    const n = this.n;

    // --- membership, roles, cameras
    this.alive = Array.from({ length: n }, () => new Track(1));
    this.lead = Array.from({ length: n }, () => new Track(0));
    this.cam = Array.from({ length: n }, () => new Track(0));
    this.sus = new Map();
    this.marks = [];
    for (const [t, kind, node, by] of tr.event) {
      if (kind === 'kill') this.alive[node].push(t, 0);
      if (kind === 'revive') this.alive[node].push(t, 1);
      if (kind === 'suspect' || kind === 'readmit') {
        const key = by * 1000 + node;
        if (!this.sus.has(key)) this.sus.set(key, new Track(0));
        this.sus.get(key).push(t, kind === 'suspect' ? 1 : 0);
      }
      this.marks.push({ t, kind, node, by });
    }
    for (const [t, node, on] of tr.role) this.lead[node].push(t, on);
    for (const [t, node, on] of tr.camera) this.cam[node].push(t, on);

    // --- CPU bookings per node (FIFO single core: starts are increasing)
    this.cpu = Array.from({ length: n }, () => ({ s: [], e: [], b: [], tag: [], fid: [], macs: [] }));
    for (const [tb, ts, te, node, macs, tag, fid] of tr.cpu) {
      const c = this.cpu[node];
      c.b.push(tb); c.s.push(ts); c.e.push(te); c.tag.push(tag); c.fid.push(fid); c.macs.push(macs);
    }
    for (const c of this.cpu) {
      const order = c.s.map((_, i) => i).sort((a, b) => c.s[a] - c.s[b]);
      for (const k of ['s', 'e', 'b', 'tag', 'fid', 'macs']) c[k] = order.map((i) => c[k][i]);
      c.cum = new Float64Array(c.s.length + 1);
      c.cmac = new Float64Array(c.s.length + 1);
      for (let i = 0; i < c.s.length; i++) { c.cum[i + 1] = c.cum[i] + (c.e[i] - c.s[i]); c.cmac[i + 1] = c.cmac[i] + c.macs[i]; }
    }

    // --- radio: unicasts and broadcasts sorted by start of transmission
    this.tx = tr.tx.slice().sort((a, b) => a[1] - b[1]);
    this.txStart = this.tx.map((r) => r[1]);
    this.bc = tr.bcast.slice().sort((a, b) => a[1] - b[1]);
    this.bcStart = this.bc.map((r) => r[1]);
    this.maxFlight = 0;
    for (const r of this.tx) this.maxFlight = Math.max(this.maxFlight, r[3] - r[1]);
    for (const r of this.bc) this.maxFlight = Math.max(this.maxFlight, r[3] - r[1]);
    const sent = [...tr.tx.map((r) => [r[0], r[7] + 24, r[2] - r[1], r[1]]), ...tr.bcast.map((r) => [r[0], r[6] + 24, r[2] - r[1], r[1]])]
      .sort((a, b) => a[0] - b[0]);
    this.sendT = sent.map((r) => r[0]);
    this.sendCum = new Float64Array(sent.length + 1);
    sent.forEach((r, i) => { this.sendCum[i + 1] = this.sendCum[i] + r[1]; });
    this.air = sent.map((r) => [r[3], r[3] + r[2]]).sort((a, b) => a[0] - b[0]);
    this.airStart = this.air.map((r) => r[0]);
    this.maxAir = this.air.reduce((m, r) => Math.max(m, r[1] - r[0]), 0);

    // --- frames: capture -> inference -> leader learns
    this.frames = new Map();
    for (const [t, node, fid, label] of tr.capture) this.frames.set(fid, { fid, t, node, label, inf: null, learn: null, x: 0, z: 0 });
    for (const [t, node, fid, pred, conf, label] of tr.infer) {
      const f = this.frames.get(fid);
      if (f && !f.inf) f.inf = { t, node, pred, conf, label };
    }
    for (const [t, node, fid, pred, conf, label, tcap] of tr.learn) {
      const f = this.frames.get(fid);
      if (f && !f.learn) f.learn = { t, node, pred, conf, tcap };
    }
    this.rejects = tr.reject;
    this.capT = tr.capture.map((r) => r[0]);
    this.objT = tr.capture.filter((r) => r[3] !== 0).map((r) => r[0]);
    this.procT = tr.infer.map((r) => r[0]).sort((a, b) => a - b);
    const learnDet = tr.learn.filter((r) => r[3] !== 0);
    this.detT = learnDet.map((r) => r[0]);
    this.repT = learnDet.filter((r) => r[5] !== 0).map((r) => r[0]);
    this.lat = learnDet.map((r) => [r[0], r[0] - r[6]]);
    this.contacts = [];
    for (const f of this.frames.values()) {
      if (f.label !== 0 || (f.learn && f.learn.pred !== 0)) this.contacts.push(f);
    }
    this.contacts.sort((a, b) => a.t - b.t);
    this.contactT = this.contacts.map((c) => c.t);
    this.lastCap = Array.from({ length: n }, () => []);
    for (const [t, node, fid] of tr.capture) this.lastCap[node].push([t, fid]);
    this.logs = tr.log || [];
    this.logT = this.logs.map((r) => r[0]);
  }

  aliveAt(i, t) { return this.alive[i].at(t) === 1; }
  leaderAt(i, t) { return this.alive[i].at(t) === 1 && this.lead[i].at(t) === 1; }
  camAt(i, t) { return this.alive[i].at(t) === 1 && this.cam[i].at(t) === 1; }

  /** [[observer, suspected], ...] at time t (only live observers keep opinions). */
  suspicionsAt(t) {
    const out = [];
    for (const [key, track] of this.sus) {
      if (track.at(t) === 1) { const by = Math.floor(key / 1000), node = key % 1000; if (this.aliveAt(by, t)) out.push([by, node]); }
    }
    return out;
  }

  /** The booking running on node i at t, the number waiting behind it, and the CPU share over the last `w` s. */
  cpuAt(i, t, w = 1.0) {
    const c = this.cpu[i];
    const k = lastAtOrBefore(c.s, t);
    let cur = null;
    if (k >= 0 && c.e[k] > t) cur = { s: c.s[k], e: c.e[k], tag: this.tags[c.tag[k]] || '?', fid: c.fid[k], macs: c.macs[k] };
    let queued = 0;
    for (let j = k + 1; j < c.s.length && c.b[j] <= t; j++) queued++;
    const busy = (x) => {
      const j = lastAtOrBefore(c.s, x);
      if (j < 0) return 0;
      return c.cum[j] + Math.max(0, Math.min(x, c.e[j]) - c.s[j]);
    };
    const t0 = Math.max(0, t - w);
    const util = t > t0 ? (busy(t) - busy(t0)) / (t - t0) : 0;
    const macsDone = k < 0 ? 0 : c.cmac[k] + (c.e[k] <= t ? c.macs[k] : c.macs[k] * Math.max(0, (t - c.s[k]) / Math.max(1e-12, c.e[k] - c.s[k])));
    return { cur, queued, util: Math.min(1, util), macs: macsDone };
  }

  /** Packets to draw at t. Short flights are stretched to `dwell` seconds so they stay visible. */
  packets(t, dwell) {
    const out = [];
    const window = Math.max(this.maxFlight, dwell) + 0.001;
    const scan = (rows, starts, bcast) => {
      let i = lastAtOrBefore(starts, t);
      for (; i >= 0 && starts[i] >= t - window; i--) {
        const r = rows[i];
        const ts = r[1], ta = r[3], vis = Math.max(ta - ts, dwell);
        if (t > ts + vis) continue;
        const kind = this.kinds[bcast ? r[5] : r[6]] || '?';
        out.push({ bcast, src: r[4], dst: bcast ? -1 : r[5], kind, bytes: bcast ? r[6] : r[7],
          lost: bcast ? r[7] : r[8], k: (t - ts) / vis, fid: bcast ? r[8] : r[9], ts, ta, color: KIND_COLOR[kind] || OTHER });
      }
    };
    scan(this.tx, this.txStart, false);
    scan(this.bc, this.bcStart, true);
    return out;
  }

  /** Mission counters at t, like the final metrics but cumulative. */
  countsAt(t) {
    const c = (arr) => lastAtOrBefore(arr, t) + 1;
    const cap = c(this.capT), obj = c(this.objT), proc = c(this.procT), det = c(this.detT), rep = c(this.repT);
    const lats = [];
    for (const [tl, l] of this.lat) { if (tl > t) break; lats.push(l); }
    lats.sort((a, b) => a - b);
    const p50 = lats.length ? lats[Math.min(lats.length - 1, Math.floor(0.5 * lats.length))] * 1e3 : NaN;
    const sentAt = (x) => this.sendCum[lastAtOrBefore(this.sendT, x) + 1];
    const w = 1.0; // rates are "over the last second", like a radio's activity meter
    const kbps = ((sentAt(t) - sentAt(t - w)) * 8) / w / 1e3;
    let airBusy = 0;
    for (let i = lastAtOrBefore(this.airStart, t); i >= 0 && this.air[i][0] >= t - w - this.maxAir; i--) {
      const [s, e] = this.air[i];
      airBusy += Math.max(0, Math.min(e, t) - Math.max(s, t - w));
    }
    let macs = 0;
    for (let i = 0; i < this.n; i++) macs += this.cpuAt(i, t, w).util * this._speed(i) * w;
    return { cap, obj, proc, det, rep, p50, kbps, chan: airBusy / w, gmacs: macs / w / 1e9,
      recall: obj ? rep / obj : NaN, bytes: sentAt(t) };
  }

  _speed(i) { const p = this.run.profiles[i]; return p.cpu_mhz * 1e6 * p.macs_per_cycle; }

  /** Most recent frame captured by node i at or before t (fid or -1). */
  lastFrame(i, t) {
    const arr = this.lastCap[i];
    let lo = 0, hi = arr.length - 1, ans = -1;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (arr[mid][0] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1; }
    return ans < 0 ? -1 : arr[ans][1];
  }

  /** future | hidden | sighted | classified | track | wrong | missed | lost | ghost */
  contactState(f, t) {
    if (t < f.t) return 'future';
    if (f.label === 0) return f.learn && f.learn.t <= t ? 'ghost' : 'hidden';
    if (f.learn && f.learn.t <= t) return f.learn.pred === f.label ? 'track' : f.learn.pred === 0 ? 'missed' : 'wrong';
    if (f.inf && f.inf.t <= t) return f.inf.pred === 0 ? 'missed' : 'classified';
    if (!f.inf) return t - f.t > 1.5 ? 'lost' : 'sighted';
    return 'sighted';
  }

  eventsUpTo(t) { return this.marks.filter((m) => m.t <= t); }
  logIndex(t) { return lastAtOrBefore(this.logT, t); }
}

export { lastAtOrBefore };
