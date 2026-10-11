// The 3D theatre: terrain, the swarm, camera frustums, radio traffic, CPU load and ground tracks at time t.

import { Batch, OrbitCamera, Renderer, mat4, yawOf } from './gl.js';
import { Kinematics, resolveTheatre } from './kinematics.js';
import { callsign, className, hash01 } from './util.js';
import * as W from './world.js';

export const COL = {
  cyan: [0.17, 0.89, 1.0], amber: [1.0, 0.71, 0.15], red: [1.0, 0.23, 0.3], green: [0.24, 1.0, 0.63],
  yellow: [1.0, 0.88, 0.3], magenta: [1.0, 0.31, 0.85], gray: [0.29, 0.36, 0.42], white: [0.85, 0.93, 1.0],
  orange: [1.0, 0.55, 0.2], violet: [0.65, 0.55, 0.98],
};
export const CONTACT_COLOR = {
  sighted: COL.yellow, classified: COL.orange, track: COL.red, wrong: COL.amber, missed: COL.gray, lost: COL.gray,
  ghost: COL.magenta,
};
const rgba = (c, a) => [c[0], c[1], c[2], a];
const mix = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
const MAX_CONTACTS = 900;
const TRACK_PERSIST = 30; // seconds a contact stays on the map
const MESH_SECONDS = 3;   // ...of which it is drawn as a 3D object, then as a ground mark
const MAX_MESHES = 48;    // high-fps single cameras sight dozens of objects per second: only the newest get bodies

function airframeFor(profile) {
  const tags = profile.tags || [];
  if (tags.includes('satellite')) return profile.name === 'sat-payload' ? 'sat' : 'cube';
  if (profile.name === 'drone-a72') return 'hex';
  if (tags.includes('tiny') || profile.name.startsWith('mcu')) return 'micro';
  return 'quad';
}

export class Theatre {
  constructor(canvas, labelsEl) {
    this.canvas = canvas;
    this.labelsEl = labelsEl;
    this.r = new Renderer(canvas);
    this.cam = new OrbitCamera(canvas);
    const r = this.r;
    // airframes are drawn ~3x life size, like map symbology, so a swarm reads at theatre scale
    this.geo = {
      quad: r.geometry(W.airframe(4, 4.2)), hex: r.geometry(W.airframe(6, 5.2)), micro: r.geometry(W.airframe(4, 3.0)),
      sat: r.geometry(W.satellite(30, 2)), cube: r.geometry(W.satellite(20, 1)),
      rotors: { quad: r.geometry(W.rotors(4, 4.2)), hex: r.geometry(W.rotors(6, 5.2)), micro: r.geometry(W.rotors(4, 3.0)) },
      contacts: W.contacts().map((g) => r.geometry(g)),
    };
    this.lines = new Batch(7);
    this.glow = new Batch(7);
    this.tris = new Batch(7);
    this.pts = new Batch(8);
    this.m = new Float32Array(16);
    this.ident = mat4.identity();
    this.mode = 'swarm';
    this.selected = null;          // node index
    this.selectedContact = null;   // fid
    this.onPick = null;
    this.onMode = null;
    this.nodeLabels = [];
    this.trackLabels = [];
    this.poses = [];
    this.cam.onClick = (x, y) => this.pick(x, y);
    this.cam.onPan = () => { if (this.mode !== 'free') this.setMode('free'); };
  }

  setMode(mode) {
    this.mode = mode;
    if (mode === 'top') { this.cam.want.pitch = 1.53; }
    else if (this.cam.want.pitch > 1.4) this.cam.want.pitch = 0.62;
    if (this.onMode) this.onMode(mode);
  }

  /** Builds the stage for a run; `pb` is the strategy being shown. */
  load(run, pb) {
    const r = this.r;
    this.run = run;
    this.th = resolveTheatre(run);
    this.kin = new Kinematics(this.th, run.profiles.length);
    this.frames = run.profiles.map(airframeFor);
    const T = W.terrain(this.th.terrain, this.kin.extent, (run.scenario.seed || 0) + 11);
    this.terrain = { ground: r.geometry(T.ground), props: T.props ? r.geometry(T.props) : null, grid: r.staticLines(T.grid), pal: T.palette };
    const rb = new Batch(7);
    const pts = this.kin.routeLine();
    for (let i = 0; i + 1 < pts.length; i += 2) rb.line(pts[i], pts[i + 1], rgba(COL.cyan, 0.22));
    this.route = r.staticLines(rb);
    this.labelsEl.replaceChildren();
    this.nodeLabels = run.profiles.map((p, i) => {
      const el = document.createElement('div');
      el.className = 'lbl node';
      el.innerHTML = '<div class="cs"></div><div class="meta"></div><div class="bar"><i></i></div>';
      this.labelsEl.appendChild(el);
      return { el, cs: el.children[0], meta: el.children[1], bar: el.children[2].firstChild, key: '' };
    });
    this.nodeLabels.forEach((l, i) => { l.cs.textContent = callsign(run.profiles[i].name, i); });
    this.trackLabels = [];
    this.setPlayback(pb);
    const n = run.profiles.length;
    this.cam.want.dist = this.th.space ? 2400 : Math.min(2600, 180 + this.th.spacing * Math.sqrt(n) * 3.5);
    this.cam.want.pitch = this.th.space ? 0.5 : 0.58;
    this.cam.dist = this.cam.want.dist * 1.6;
    this.cam.pitch = 1.2;
    this.cam.maxDist = this.kin.extent * 5;
    const c = this.centroid(0);
    this.cam.target = [c[0], c[1], c[2]];
    this.cam.want.target = [c[0], c[1], c[2]];
  }

  setPlayback(pb) {
    this.pb = pb;
    const fp = this.kin.footprint;
    for (const f of pb.contacts) {
      const { p, h } = this.kin.pose(f.node, f.t);
      const u = (hash01(f.fid, 1) - 0.5) * fp * 0.9, v = (hash01(f.fid, 2) - 0.5) * fp * 0.9;
      f.x = p[0] + h[0] * u - h[1] * v;
      f.z = p[2] + h[1] * u + h[0] * v;
      f.yaw = hash01(f.fid, 3) * Math.PI * 2;
    }
    for (const l of this.trackLabels) l.el.remove();
    this.trackLabels = [];
  }

  centroid(t) {
    let x = 0, z = 0, k = 0;
    const n = this.run.profiles.length;
    for (let i = 0; i < n; i++) {
      if (this.pb && !this.pb.aliveAt(i, t) && k > 0) continue;
      const { p } = this.kin.pose(i, t);
      x += p[0]; z += p[2]; k++;
    }
    return k ? [x / k, this.th.altitude * 0.45, z / k] : [0, 0, 0];
  }

  pick(x, y) {
    let best = null, bd = 30;
    this.poses.forEach((pose, i) => {
      const s = this.r.project(pose.p);
      if (s) { const d = Math.hypot(s[0] - x, s[1] - y); if (d < bd) { bd = d; best = { type: 'node', id: i }; } }
    });
    if (!best && this.visibleContacts) {
      let cd = 16;
      for (const f of this.visibleContacts) {
        const s = this.r.project([f.x, 2, f.z]);
        if (s) { const d = Math.hypot(s[0] - x, s[1] - y); if (d < cd) { cd = d; best = { type: 'contact', id: f.fid }; } }
      }
    }
    if (this.onPick) this.onPick(best);
  }

  frame(t, dt, playSpeed = 1) {
    const r = this.r;
    const aspect = r.resize();
    if (!this.run) {
      this.cam.update(dt, aspect);
      r.begin(this.cam.view, this.cam.proj, this.cam.eye, [0.01, 0.02, 0.03], 0.0004, this.cam.fovy);
      return;
    }
    const pb = this.pb, kin = this.kin, th = this.th, n = this.run.profiles.length;
    this.poses = Array.from({ length: n }, (_, i) => kin.pose(i, t));

    if (!this.cam.dragging) {
      if (this.mode === 'swarm' || this.mode === 'top') {
        const c = this.centroid(t);
        this.cam.want.target = c;
      } else if (this.mode === 'asset' && this.selected !== null) {
        const p = this.poses[this.selected].p;
        this.cam.want.target = [p[0], p[1], p[2]];
      }
    }
    this.cam.update(dt, aspect);
    const pal = this.terrain.pal;
    const fogDen = 1 / Math.max(this.cam.dist * 3.2, kin.extent * 1.6);
    r.begin(this.cam.view, this.cam.proj, this.cam.eye, pal.fog, fogDen, this.cam.fovy);
    r.mesh(this.terrain.ground, this.ident, pal.ground, { emissive: 0.2 });
    if (this.terrain.props) r.mesh(this.terrain.props, this.ident, [1, 1, 1]);
    r.drawStaticLines(this.terrain.grid);
    r.drawStaticLines(this.route);

    const L = this.lines, G = this.glow, T = this.tris, P = this.pts;
    L.reset(); G.reset(); T.reset(); P.reset();

    // ---------------- node states
    const st = [];
    for (let i = 0; i < n; i++) {
      st.push({ alive: pb.aliveAt(i, t), leader: pb.leaderAt(i, t), cam: pb.camAt(i, t), cpu: pb.cpuAt(i, t, 1.0) });
    }
    const suspicions = pb.suspicionsAt(t);
    const suspected = new Set(suspicions.map((s) => s[1]));
    const leaders = st.map((s, i) => (s.leader ? i : -1)).filter((i) => i >= 0);

    // ---------------- ground contacts: fresh ones as 3D objects, older ones as fading ground marks
    const lo = Math.max(0, this._firstContact(t - TRACK_PERSIST));
    const hi = this._firstContact(t + 1e-9);
    const vis = [];
    for (let k = Math.max(lo, hi - MAX_CONTACTS); k < hi; k++) {
      const f = pb.contacts[k];
      const state = pb.contactState(f, t);
      if (state === 'future' || state === 'hidden') continue;
      f.state = state;
      vis.push(f);
      const age = t - f.t;
      const since = f.learn ? t - f.learn.t : age;
      const col = CONTACT_COLOR[state] || COL.gray;
      const live = (Math.min(age, since) < MESH_SECONDS && k >= hi - MAX_MESHES) || f.fid === this.selectedContact;
      if (live) {
        const geoIdx = state === 'ghost' ? 0 : f.label;
        const emissive = state === 'track' ? 0.35 : state === 'sighted' ? 0.5 : state === 'ghost' ? 0.8 : 0.2;
        mat4.trs(this.m, f.x, 0, f.z, f.yaw, 0, 0, state === 'ghost' ? 1.0 : 0.65);
        r.mesh(this.geo.contacts[geoIdx] || this.geo.contacts[0], this.m, col, { emissive });
      } else {
        const a = 0.55 * Math.max(0, 1 - (age - MESH_SECONDS) / (TRACK_PERSIST - MESH_SECONDS));
        const s = 2.2;
        L.line([f.x - s, 0.4, f.z], [f.x + s, 0.4, f.z], rgba(col, a));
        L.line([f.x, 0.4, f.z - s], [f.x, 0.4, f.z + s], rgba(col, a));
      }
      const fresh = Math.max(0, 1 - age / 3);
      if (state === 'track' || state === 'wrong' || state === 'ghost') {
        const since = t - f.learn.t;
        if (since >= 0 && since < 8) {
          const k = 1 - since / 8;
          const s = 6 + (1 - Math.min(1, since / 0.6)) * 16; // lock-on: the marker closes onto the target
          const c = rgba(col, 0.85 * k);
          L.line([f.x - s, 0.8, f.z], [f.x, 0.8, f.z - s], c); L.line([f.x, 0.8, f.z - s], [f.x + s, 0.8, f.z], c);
          L.line([f.x + s, 0.8, f.z], [f.x, 0.8, f.z + s], c); L.line([f.x, 0.8, f.z + s], [f.x - s, 0.8, f.z], c);
          L.line([f.x, 0.8, f.z], [f.x, 18, f.z], rgba(col, 0.35 * k));
        }
      }
      if (age < 6) P.point([f.x, 3, f.z], state === 'track' ? 9 : 6, rgba(col, (state === 'sighted' ? 0.6 : 0.4) * fresh + 0.08));
      if (f.fid === this.selectedContact) this._brackets(L, [f.x, 4, f.z], 16, rgba(COL.white, 0.9));
    }
    this.visibleContacts = vis;

    // ---------------- camera frustums and footprints
    const fp = kin.footprint / 2;
    for (let i = 0; i < n; i++) {
      if (!st[i].cam) continue;
      const { p, h } = this.poses[i];
      const rx = -h[1], rz = h[0];
      const g = [p[0], 0.6, p[2]];
      const corners = [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => [g[0] + h[0] * fp * a + rx * fp * 0.78 * b, 0.6, g[2] + h[1] * fp * a + rz * fp * 0.78 * b]);
      const apex = [p[0], p[1] - 2, p[2]];
      for (let k = 0; k < 4; k++) {
        T.tri(apex, corners[k], corners[(k + 1) % 4], rgba(COL.cyan, 0.035));
        L.line(corners[k], corners[(k + 1) % 4], rgba(COL.cyan, 0.6));
        L.line(apex, corners[k], rgba(COL.cyan, 0.12));
      }
      T.tri(corners[0], corners[1], corners[2], rgba(COL.cyan, 0.05));
      T.tri(corners[0], corners[2], corners[3], rgba(COL.cyan, 0.05));
      const sweep = ((t * 0.9 + i * 0.37) % 1) * 2 - 1;
      const a = [g[0] + h[0] * fp * sweep + rx * fp * 0.78, 0.8, g[2] + h[1] * fp * sweep + rz * fp * 0.78];
      const b = [g[0] + h[0] * fp * sweep - rx * fp * 0.78, 0.8, g[2] + h[1] * fp * sweep - rz * fp * 0.78];
      G.line(a, b, rgba(COL.cyan, 0.35));
    }

    // ---------------- command links and suspicion
    for (let i = 0; i < n; i++) {
      if (!st[i].alive) continue;
      for (const l of leaders) {
        if (l === i) continue;
        L.line(this.poses[i].p, this.poses[l].p, rgba(COL.cyan, 0.16));
      }
    }
    for (const [by, node] of suspicions) {
      const a = this.poses[by].p, b = this.poses[node].p;
      const segs = 14;
      for (let k = 0; k < segs; k += 2) {
        const p0 = [a[0] + (b[0] - a[0]) * k / segs, a[1] + (b[1] - a[1]) * k / segs, a[2] + (b[2] - a[2]) * k / segs];
        const p1 = [a[0] + (b[0] - a[0]) * (k + 1) / segs, a[1] + (b[1] - a[1]) * (k + 1) / segs, a[2] + (b[2] - a[2]) * (k + 1) / segs];
        G.line(p0, p1, rgba(COL.red, 0.55));
      }
    }

    // ---------------- radio traffic
    const dwell = Math.min(1.2, Math.max(0.035, 0.14 * playSpeed));
    const span = th.spacing * Math.max(2, Math.sqrt(n)) * 1.6;
    for (const pk of pb.packets(t, dwell)) {
      const a = this.poses[pk.src].p;
      if (pk.bcast) {
        const rad = pk.k * span;
        const alpha = (1 - pk.k) * (pk.kind === 'hb' ? 0.22 : 0.55);
        if (alpha > 0.01) L.circle(a[0], a[1], a[2], Math.max(1, rad), rgba(pk.color, alpha), 40);
        continue;
      }
      if (pk.dst < 0 || pk.dst >= n) continue;
      const b = this.poses[pk.dst].p;
      const k = pk.lost ? Math.min(pk.k, 0.55) : pk.k;
      const lift = Math.sin(Math.PI * k) * Math.hypot(b[0] - a[0], b[2] - a[2]) * 0.08;
      const at = (q) => [a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q + Math.sin(Math.PI * q) * Math.hypot(b[0] - a[0], b[2] - a[2]) * 0.08, a[2] + (b[2] - a[2]) * q];
      const pos = [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k + lift, a[2] + (b[2] - a[2]) * k];
      let col = pk.color, alpha = 0.95;
      if (pk.lost) { col = COL.red; alpha = pk.k > 0.55 ? Math.max(0, 1 - (pk.k - 0.55) * 3) : 0.9; }
      const size = pk.kind === 'frame' || pk.kind === 'features' ? 6 + Math.min(10, Math.log2(1 + pk.bytes / 512) * 2.5) : 4.5;
      P.point(pos, size, rgba(col, alpha));
      G.line(at(Math.max(0, k - 0.12)), pos, rgba(col, 0.0), rgba(col, alpha * 0.55));
    }

    // ---------------- airframes
    for (let i = 0; i < n; i++) {
      const s = st[i], { p, h } = this.poses[i];
      const yaw = yawOf(h[0], h[1]);
      const kind = this.frames[i];
      let col = s.leader ? COL.amber : COL.cyan;
      if (!s.alive) col = [0.22, 0.26, 0.3];
      const sel = this.selected === i;
      const flash = suspected.has(i) && s.alive ? 0.5 + 0.5 * Math.sin(t * 20) : 0;
      if (flash) col = mix(col, COL.red, flash);
      const bank = th.space ? 0 : Math.sin(t * 0.9 + i) * 0.05;
      mat4.trs(this.m, p[0], p[1], p[2], yaw, bank * 0.5, bank, 1);
      r.mesh(this.geo[kind], this.m, col, { emissive: s.alive ? (sel ? 0.55 : 0.28) : 0 });
      if (s.alive && this.geo.rotors[kind]) {
        mat4.trs(this.m, p[0], p[1], p[2], yaw, bank * 0.5, bank, 1);
        r.mesh(this.geo.rotors[kind], this.m, mix(col, COL.white, 0.4), { alpha: 0.22, blend: true, cull: false, emissive: 0.6 });
      }
      const ringR = kind === 'sat' || kind === 'cube' ? 60 : kind === 'micro' ? 10 : kind === 'hex' ? 16 : 14;
      if (s.alive) {
        const u = s.cpu.util;
        const cc = u > 0.9 ? COL.red : u > 0.6 ? COL.amber : COL.cyan;
        L.circle(p[0], p[1] - 0.5, p[2], ringR, rgba(COL.cyan, 0.12), 36);
        if (u > 0.002) G.circle(p[0], p[1] - 0.5, p[2], ringR, rgba(cc, 0.9), Math.max(3, Math.ceil(48 * u)), -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * u);
        if (s.cpu.cur) P.point([p[0], p[1], p[2]], s.cpu.cur.tag === 'brain' ? 16 : 9, rgba(s.cpu.cur.tag === 'brain' ? cc : COL.white, 0.5 + 0.3 * Math.sin(t * 30)));
      } else {
        const x = 6;
        G.line([p[0] - x, p[1] + 8, p[2] - x], [p[0] + x, p[1] + 8, p[2] + x], rgba(COL.red, 0.9));
        G.line([p[0] - x, p[1] + 8, p[2] + x], [p[0] + x, p[1] + 8, p[2] - x], rgba(COL.red, 0.9));
      }
      if (s.leader) {
        const y = p[1] + ringR + 6, s2 = 4.5, a0 = t * 1.6;
        const q = [0, 1, 2, 3].map((k) => [p[0] + s2 * Math.cos(a0 + k * Math.PI / 2), y, p[2] + s2 * Math.sin(a0 + k * Math.PI / 2)]);
        const top = [p[0], y + 6, p[2]], bot = [p[0], y - 6, p[2]];
        for (let k = 0; k < 4; k++) { G.line(q[k], q[(k + 1) % 4], rgba(COL.amber, 0.95)); G.line(q[k], top, rgba(COL.amber, 0.6)); G.line(q[k], bot, rgba(COL.amber, 0.6)); }
        L.circle(p[0], 0.7, p[2], ringR * 2.2, rgba(COL.amber, 0.35), 48);
      }
      if (sel) this._brackets(L, p, ringR + 5, rgba(COL.white, 0.85));
      L.line([p[0], 0.5, p[2]], [p[0], p[1] - 2, p[2]], rgba(s.alive ? col : COL.red, 0.08));
    }
    // fault pulses: an expanding ring where a node went down / came back
    for (const m of pb.marks) {
      if (m.t > t || t - m.t > 1.6 || (m.kind !== 'kill' && m.kind !== 'revive' && m.kind !== 'leader')) continue;
      const k = (t - m.t) / 1.6, { p } = this.poses[m.node];
      const c = m.kind === 'kill' ? COL.red : m.kind === 'revive' ? COL.green : COL.amber;
      G.circle(p[0], p[1], p[2], 5 + k * 70, rgba(c, (1 - k) * 0.9), 48);
    }

    r.tris(T);
    r.lines(L);
    r.lines(G, true);
    r.points(P);
    this._labels(t, st, vis);
  }

  _firstContact(t) {
    const a = this.pb.contactT;
    let lo = 0, hi = a.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < t) lo = mid + 1; else hi = mid; }
    return lo;
  }

  _brackets(B, p, s, c) {
    const k = s * 0.35;
    for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const x = p[0] + sx * s, z = p[2] + sz * s;
      B.line([x, p[1], z], [x - sx * k, p[1], z], c);
      B.line([x, p[1], z], [x, p[1], z - sz * k], c);
    }
  }

  _labels(t, st, vis) {
    const placed = [];
    // greedy de-overlap in screen space: a label that would cover another moves down a row, or hides
    const place = (x, y, w, hgt) => {
      for (let k = 0; k < 4; k++) {
        const yy = y + k * (hgt + 2);
        if (!placed.some((b) => x < b.x + b.w && x + w > b.x && yy < b.y + b.h && yy + hgt > b.y)) {
          placed.push({ x, y: yy, w, h: hgt });
          return yy;
        }
      }
      return null;
    };
    st.forEach((s, i) => {
      const l = this.nodeLabels[i];
      const scr = this.r.project(this.poses[i].p);
      const y = scr ? place(scr[0] + 16, scr[1] - 22, 112, 36) : null;
      if (y === null) { l.el.style.display = 'none'; return; }
      l.el.style.display = '';
      l.el.style.transform = `translate(${Math.round(scr[0] + 16)}px, ${Math.round(y)}px)`;
      const role = !s.alive ? 'OFFLINE' : s.leader ? 'LEAD' : s.cam ? 'SENSOR' : 'NODE';
      const work = s.cpu.cur ? (s.cpu.cur.tag === 'brain' ? ` INF#${s.cpu.cur.fid}` : ' PREP') : '';
      const key = `${role}|${Math.round(s.cpu.util * 100)}|${work}|${this.selected === i}`;
      if (key !== l.key) {
        l.key = key;
        l.el.className = 'lbl node' + (s.leader ? ' leader' : '') + (!s.alive ? ' dead' : '') + (this.selected === i ? ' sel' : '');
        l.meta.textContent = `${role} ${s.alive ? Math.round(s.cpu.util * 100) + '%' : ''}${work}`;
        l.bar.style.width = `${Math.round(s.cpu.util * 100)}%`;
      }
    });
    const near = this.cam.dist < 1500;
    const tracks = near ? vis.filter((f) => (f.state === 'track' || f.state === 'wrong' || f.state === 'ghost') &&
      (t - f.learn.t < 5 || f.fid === this.selectedContact)).slice(-7) : vis.filter((f) => f.fid === this.selectedContact);
    while (this.trackLabels.length < tracks.length) {
      const el = document.createElement('div');
      el.className = 'lbl trk';
      this.labelsEl.appendChild(el);
      this.trackLabels.push({ el, key: '' });
    }
    this.trackLabels.forEach((l, k) => {
      const f = tracks[k];
      const scr = f ? this.r.project([f.x, 18, f.z]) : null;
      const y = scr ? place(scr[0] + 6, scr[1] - 14, 150, 13) : null;
      if (y === null) { l.el.style.display = 'none'; return; }
      l.el.style.display = '';
      l.el.style.transform = `translate(${Math.round(scr[0] + 6)}px, ${Math.round(y)}px)`;
      const key = `${f.fid}|${f.state}|${this.selectedContact === f.fid}`;
      if (key !== l.key) {
        l.key = key;
        const cls = f.state === 'ghost' ? ' ghost' : f.state === 'wrong' ? ' wrong' : '';
        l.el.className = 'lbl trk' + cls + (this.selectedContact === f.fid ? ' sel' : '');
        const name = className(f.learn.pred).toUpperCase();
        l.el.textContent = f.state === 'ghost' ? `FALSE ALARM ${name}?` : `TRK-${String(f.fid).padStart(4, '0')} ${name} ${Math.round(f.learn.conf * 100)}%`;
      }
    });
  }
}
