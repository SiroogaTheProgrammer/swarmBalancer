// 3D replay of one arena episode: the clash (interceptor wall vs incoming swarm) or the recon survey.
// Simulation coordinates (x, y, z) in metres map to the scene as X = x, Y = z (altitude), Z = -y: the
// attackers come in from -Z towards the protected plane Z = 0.

import { Batch, OrbitCamera, Renderer, mat4, yawOf } from './gl.js';
import { airframe, box, rotors, terrain } from './world.js';

const C = {
  cyan: [0.21, 0.86, 1.0], red: [1.0, 0.3, 0.22], redDim: [0.55, 0.17, 0.14], amber: [1.0, 0.7, 0.12],
  green: [0.26, 0.96, 0.6], grey: [0.33, 0.4, 0.46], white: [0.9, 0.95, 1.0], magenta: [1, 0.25, 0.5],
};

function dart() {
  const g = { pos: [], nor: [], col: [] };
  box(g, 2.2, 0.35, 0.55, 0, 0, 0, [1, 1, 1]);            // fuselage, facing +X
  box(g, 0.7, 0.1, 2.4, -0.2, 0, 0, [0.85, 0.85, 0.85]);   // wing
  box(g, 0.45, 0.6, 0.1, -0.95, 0.3, 0, [0.7, 0.7, 0.7]);  // fin
  box(g, 0.35, 0.3, 0.4, 1.15, 0, 0, [0.5, 0.5, 0.5]);     // seeker / warhead
  return g;
}

const lerp = (a, b, k) => a + (b - a) * k;
const rgba = (c, a) => [c[0], c[1], c[2], a];
const clamp01 = (v) => Math.max(0, Math.min(1, v));

/** Vertical ring in the plane Z = z (where an attacker went through the protected plane). */
function wallRing(L, cx, cy, z, r, col, seg = 24) {
  let px = cx + r, py = cy;
  for (let i = 1; i <= seg; i++) {
    const a = (i / seg) * Math.PI * 2, x = cx + r * Math.cos(a), y = Math.max(0.5, cy + r * Math.sin(a));
    L.line([px, py, z], [x, y, z], col);
    px = x; py = y;
  }
}

// Replays recorded before the arena went 3-D carry (x, y) per drone: lift them to a fixed height.
function upgrade(r) {
  const lift = (arr, z) => { const out = []; for (let i = 0; i + 1 < arr.length; i += 2) out.push(arr[i], arr[i + 1], z); return out; };
  if (r.kind === 'clash' && r.ceiling === undefined) {
    Object.assign(r, {
      ceiling: 150, low_altitude: 0, station: r.station.map((s) => [s[0], s[1], 35]),
      frames: r.frames.map((f) => ({ ...f, d: lift(f.d, 35), a: lift(f.a, 55) })),
      events: (r.events || []).map((e) => ({ z: 55, ...e })),
    });
  } else if (r.kind === 'recon' && r.camera_tan === undefined) {
    Object.assign(r, {
      camera_tan: (r.footprint || 40) / 70, detail_altitude: 70, max_altitude: 140, quality_step: 0.1, base: r.base || [0, 0],
      frames: r.frames.map((f) => ({ ...f, p: lift(f.p, 70), nq: f.nq || f.nc.map(() => 10) })),
    });
  }
  return r;
}

export class ArenaView {
  constructor(canvas, dom) {
    this.canvas = canvas;
    this.dom = dom;                     // {hud, ticker, legend, labels, empty}
    this.r = new Renderer(canvas);
    this.cam = new OrbitCamera(canvas);
    this.geo = {
      drone: this.r.geometry(airframe(4, 1)), rotor: this.r.geometry(rotors(4, 1)), dart: this.r.geometry(dart()),
    };
    this.lines = new Batch(7, 8192);
    this.tris = new Batch(7, 8192);
    this.pts = new Batch(8, 1024);
    this.m = new Float32Array(16);
    this.replay = null;
    this.t = 0;
    this.playing = false;
    this.speed = 4;
    this.show = { links: true, locks: true };
    this.onTick = null;
    this._last = performance.now();
    this._labels = [];
    this._hudKey = '';
    this._tickKey = '';
    const loop = (now) => { this._frame(now); this._raf = requestAnimationFrame(loop); };
    this._raf = requestAnimationFrame(loop);
  }

  stop() { cancelAnimationFrame(this._raf); }

  get duration() {
    const r = this.replay;
    return r ? (r.frames.length - 1) * r.dt : 0;
  }

  load(replay, { fit = true } = {}) {
    this.replay = replay ? upgrade(replay) : null;
    this.t = 0;
    this.kind = replay ? replay.kind : null;
    this.dom.empty.classList.toggle('hide', !!replay);
    if (!replay) {
      this._static = null;
      this.playing = false;
      for (const el of [this.dom.hud, this.dom.ticker, this.dom.legend, this.dom.labels]) el.innerHTML = '';
      this._hudKey = this._tickKey = '';
      if (this.onTick) this.onTick(0, 0, false);
      return;
    }
    const extent = this.kind === 'clash' ? Math.max(replay.width, replay.depth) : replay.area;
    const ter = terrain('plains', extent * 0.9, 11);
    this._static = {
      ground: this.r.geometry(ter.ground), grid: this.r.staticLines(ter.grid), props: ter.props ? this.r.geometry(ter.props) : null,
      palette: ter.palette, extent,
    };
    if (this.kind === 'recon') this._prepRecon(replay); else this._prepClash(replay);
    this._legend();
    if (fit) this.fit();
  }

  fit() {
    const r = this.replay;
    if (!r) return;
    const w = this.cam.want;
    if (this.kind === 'clash') {
      // from behind the wall, a little to the side and low enough to read heights
      w.target = [0, r.ceiling * 0.2, -r.depth * 0.5];
      w.dist = Math.max(r.width, r.depth) * 1.1;
      w.yaw = 0.5; w.pitch = 0.42;
    } else {
      w.target = [0, 0, 0];
      w.dist = r.area * 1.2;
      w.yaw = 0.4; w.pitch = 0.75;
    }
    this.cam.maxDist = this._static.extent * 6;
  }

  setTime(t) { this.t = Math.max(0, Math.min(this.duration, t)); }

  // ------------------------------------------------------------------ preparation
  _prepClash(r) {
    const n = r.frames.length;
    this._events = (r.events || []).slice().sort((a, b) => a.t - b.t);
    // per attacker: time it died / crossed, for fading; per frame: how many defenders lock each attacker
    this._aEnd = new Array(r.attackers).fill(Infinity);
    for (const e of this._events) if (e.type === 'kill' || e.type === 'breach') this._aEnd[e.a] = Math.min(this._aEnd[e.a], e.t);
    this._count = [];
    for (let i = 0; i < n; i++) {
      const f = r.frames[i], c = {};
      for (const tg of f.tg) if (tg >= 0) c[tg] = (c[tg] || 0) + 1;
      this._count.push(c);
    }
  }

  _prepRecon(r) {
    const cells = r.grid * r.grid;
    this._level = new Uint8Array(cells);      // best detail each cell was seen in, in steps of quality_step
    this._levelT = new Float32Array(cells);   // when that last improved
    this._cellFrame = -1;
    this._events = (r.events || []).slice().sort((a, b) => a.t - b.t);
    this._found = new Array(r.targets.length).fill(Infinity);
    this._reported = new Array(r.targets.length).fill(Infinity);
    for (const e of this._events) {
      if (e.type === 'found') this._found[e.target] = e.t;
      if (e.type === 'reported') this._reported[e.target] = e.t;
    }
  }

  /** Coverage map at time `now`: the frames only carry the cells that improved, so replay them in order. */
  _cellsAt(now) {
    const r = this.replay, fi = Math.min(Math.floor(now / r.dt + 1e-6), r.frames.length - 1);
    if (fi < this._cellFrame) { this._level.fill(0); this._cellFrame = -1; }
    for (let f = this._cellFrame + 1; f <= fi; f++) {
      const fr = r.frames[f];
      for (let u = 0; u < fr.nc.length; u++) { this._level[fr.nc[u]] = fr.nq[u]; this._levelT[fr.nc[u]] = f * r.dt; }
    }
    this._cellFrame = fi;
  }

  _legend() {
    const items = this.kind === 'clash'
      ? [[C.cyan, 'interceptors'], [C.red, 'attackers'], [C.redDim, 'attacker under the terrain mask'],
        [C.amber, 'intent (who is going for whom)'], [C.magenta, 'two or more on one target'],
        [C.cyan, 'protected plane, up to the ceiling', 0.5]]
      : [[C.amber, 'lead drone (uplink)'], [C.cyan, 'camera on: cone to its footprint'], [C.grey, 'camera off'],
        [C.green, 'surveyed ground (brighter = more detail)'], [C.green, 'radio chain to the lead', 0.6]];
    const css = (c, a = 1) => `rgba(${c.map((v) => Math.round(v * 255)).join(',')},${a})`;
    this.dom.legend.innerHTML = '';
    for (const [c, label, a] of items) {
      const row = document.createElement('div');
      const sw = document.createElement('i');
      sw.style.background = css(c, a || 1);
      row.append(sw, label);
      this.dom.legend.appendChild(row);
    }
  }

  // ------------------------------------------------------------------ per frame
  _frame(now) {
    const dt = Math.min(0.1, (now - this._last) / 1000);
    this._last = now;
    const r = this.r;
    const aspect = r.resize();
    this.cam.update(dt, aspect);
    const st = this._static;
    const fog = st ? st.palette.fog : [0.01, 0.02, 0.03];
    r.begin(this.cam.view, this.cam.proj, this.cam.eye, fog, st ? 0.35 / (st.extent * 2.4) : 0.0004, this.cam.fovy);
    if (!this.replay || !st) return;
    if (this.playing) {
      this.t += dt * this.speed;
      if (this.t >= this.duration) { this.t = this.duration; this.playing = false; }
    }
    r.mesh(st.ground, mat4.trs(this.m, 0, 0, 0), st.palette.ground, { emissive: 0.0 });
    r.drawStaticLines(st.grid);
    if (st.props) r.mesh(st.props, mat4.trs(this.m, 0, 0, 0), [1, 1, 1]);
    this.lines.reset(); this.tris.reset(); this.pts.reset();
    this._labels = [];
    if (this.kind === 'clash') this._drawClash(); else this._drawRecon();
    r.tris(this.tris);
    r.lines(this.lines);
    r.points(this.pts);
    this._placeLabels();
    if (this.onTick) this.onTick(this.t, this.duration, this.playing);
  }

  /** Drones are drawn as icons about 1/120 of the view distance across, so they stay visible over a field
   *  kilometres wide, and shrink towards life size as you zoom in. */
  _scale() { return Math.max(2.5, Math.min(this.cam.dist, this._static.extent * 1.5) / 120); }

  _sample() {
    const r = this.replay, n = r.frames.length;
    const x = Math.min(this.t / r.dt, n - 1);
    const i = Math.min(Math.floor(x), n - 1), j = Math.min(i + 1, n - 1);
    // headings come from the motion between two frames; on the last frame, from the two before it
    const h0 = j > i ? i : Math.max(0, i - 1);
    return { i, k: x - i, a: r.frames[i], b: r.frames[j], ha: r.frames[h0], hb: r.frames[h0 === i ? j : i] };
  }

  /** Scene position of drone `idx` from flat [x, y, z, ...] frame arrays; (ox, oy) is the ground origin. */
  _pos(fa, fb, idx, k, ox = 0, oy = 0) {
    const o = 3 * idx;
    return [lerp(fa[o], fb[o], k) - ox, lerp(fa[o + 2], fb[o + 2], k), oy - lerp(fa[o + 1], fb[o + 1], k)];
  }

  /** [yaw, climb angle] from the motion between two frames. */
  _dir(fa, fb, idx, fallback) {
    const o = 3 * idx, dx = fb[o] - fa[o], dy = fb[o + 1] - fa[o + 1], dz = fb[o + 2] - fa[o + 2];
    const hz = Math.hypot(dx, dy);
    return hz > 1e-3 ? [yawOf(dx, -dy), Math.atan2(dz, hz)] : [fallback, 0];
  }

  _links(ps, range, colour) {
    const L = this.lines, R2 = range * range;
    for (let u = 0; u < ps.length; u++) {
      const p = ps[u];
      if (!p) continue;
      for (let v = u + 1; v < ps.length; v++) {
        const q = ps[v];
        if (!q) continue;
        const dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
        if (dx * dx + dy * dy + dz * dz <= R2) L.line(p, q, typeof colour === 'function' ? colour(u, v) : colour);
      }
    }
  }

  _drawClash() {
    const R = this.replay, r = this.r, L = this.lines, T = this.tris, P = this.pts, m = this.m;
    const { i, k, a, b, ha, hb } = this._sample();
    const W = R.width / 2, depth = R.depth, H = R.ceiling, low = R.low_altitude || 0, now = this.t, s = this._scale();
    const far = -depth * 1.2;
    // the protected line is a plane up to the ceiling: crossing it at any height scores
    const pulse = 0.55 + 0.25 * Math.sin(performance.now() / 380);
    L.line([-W, 1, 0], [W, 1, 0], rgba(C.cyan, 0.95));
    L.line([-W, H, 0], [W, H, 0], rgba(C.cyan, 0.45));
    for (let q = 0; q <= 8; q++) {
      const x = -W + (2 * W * q) / 8;
      L.line([x, 1, 0], [x, H, 0], rgba(C.cyan, q % 8 ? 0.12 : 0.45));
    }
    T.tri([-W, 0, 0], [W, 0, 0], [W, H, 0], rgba(C.cyan, 0.045 * pulse));
    T.tri([-W, 0, 0], [W, H, 0], [-W, H, 0], rgba(C.cyan, 0.045 * pulse));
    // the airspace both sides fly in, and where the attackers enter
    for (const x of [-W, W]) {
      L.line([x, 1, 0], [x, 1, far], rgba(C.grey, 0.4));
      L.line([x, H, 0], [x, H, far], rgba(C.grey, 0.16));
      L.line([x, 1, far], [x, H, far], rgba(C.grey, 0.16));
    }
    L.line([-W, H, far], [W, H, far], rgba(C.grey, 0.16));
    L.line([-W, 1, -depth], [W, 1, -depth], rgba(C.red, 0.35));
    // the terrain mask: below it a drone is hidden from sensors until close
    if (low > 0) {
      const mc = rgba(C.amber, 0.035);
      T.tri([-W, low, 0], [W, low, 0], [W, low, far], mc);
      T.tri([-W, low, 0], [W, low, far], [-W, low, far], mc);
      L.line([-W, low, 0], [W, low, 0], rgba(C.amber, 0.3));
      for (const x of [-W, W]) L.line([x, low, 0], [x, low, far], rgba(C.amber, 0.2));
      this._labels.push([[W, low, far * 0.5], `TERRAIN MASK ${Math.round(low)} m`]);
    }
    // guard posts of the interceptors still in the fight (the line re-spaces as they are spent; a brain may
    // post some or all of them forward, as pickets)
    let stations = R.station;
    for (const rs of R.restations || []) { if (rs.t > now) break; stations = rs.station; }
    stations.forEach((st, d) => { if (a.ds[d] === '1') L.circle(st[0], st[2], -st[1], 1.4 * s, rgba(C.cyan, 0.3), 14); });
    this._labels.push([[0, H, 0], 'PROTECTED LINE'], [[0, 5, -depth], 'ATTACKERS ENTER'], [[W, H, 0], `CEILING ${Math.round(H)} m`]);

    const D = R.defenders, A = R.attackers;
    const dpos = [], apos = [];
    let masked = 0;
    for (let d = 0; d < D; d++) {
      const p = this._pos(a.d, b.d, d, k);
      const [yaw] = this._dir(ha.d, hb.d, d, Math.PI / 2);
      if (a.ds[d] !== '1') {  // out of shots: it lands
        dpos.push(null);
        r.mesh(this.geo.drone, mat4.trs(m, p[0], 2, p[2], yaw, 0, 0, s * 0.8), C.grey, { emissive: 0 });
        continue;
      }
      dpos.push(p);
      r.mesh(this.geo.drone, mat4.trs(m, p[0], p[1], p[2], yaw, 0, 0, s), C.cyan, { emissive: 0.35 });
      r.mesh(this.geo.rotor, mat4.trs(m, p[0], p[1], p[2], yaw + now * 30, 0, 0, s), C.cyan, { alpha: 0.35, blend: true, cull: false });
      P.point(p, 2.9 * s, rgba(C.cyan, 0.55));
      L.line(p, [p[0], 0.5, p[2]], rgba(C.cyan, 0.14));
    }
    for (let q = 0; q < A; q++) {
      if (now > this._aEnd[q]) { apos.push(null); continue; }
      const p = this._pos(a.a, b.a, q, k);
      apos.push(p);
      const [yaw, climb] = this._dir(ha.a, hb.a, q, -Math.PI / 2);
      const hidden = p[1] < low;
      if (hidden) masked++;
      r.mesh(this.geo.dart, mat4.trs(m, p[0], p[1], p[2], yaw, 0, climb, s * 1.1), hidden ? C.redDim : C.red, { emissive: hidden ? 0.15 : 0.45 });
      P.point(p, 3.2 * s, rgba(C.red, hidden ? 0.25 : 0.6));
      L.line(p, [p[0], 0.5, p[2]], rgba(C.red, 0.1));
    }
    // radio mesh: friends within radio range (3-D distance)
    if (this.show.links) {
      this._links(dpos, R.comm_range, rgba(C.cyan, 0.14));
      this._links(apos, R.comm_range, rgba(C.red, 0.12));
    }
    // intent: who is going for whom; magenta when several pile onto one attacker
    if (this.show.locks) {
      const count = this._count[i];
      a.tg.forEach((tg, d) => {
        if (tg < 0 || !dpos[d] || !apos[tg]) return;
        L.line(dpos[d], apos[tg], count[tg] > 1 ? rgba(C.magenta, 0.95) : rgba(C.amber, 0.7));
      });
    }
    // flashes for kills / breaches in the last 1.5 s of simulated time
    for (const e of this._events) {
      if (e.t > now) break;
      const age = now - e.t;
      if (age > 1.5) continue;
      const f = 1 - age / 1.5, z = e.z === undefined ? 50 : e.z;
      if (e.type === 'kill') {
        P.point([e.x, z, -e.y], (10 + 30 * (1 - f)) * s, [1, 0.85, 0.5, f]);
        L.circle(e.x, z, -e.y, (1 + 5 * (1 - f)) * s, rgba(C.white, f), 20);
      } else if (e.type === 'breach') {
        P.point([e.x, z, 0], (16 + 34 * (1 - f)) * s, [1, 0.2, 0.15, f]);
        wallRing(L, e.x, z, 0, (1.5 + 7 * (1 - f)) * s, rgba(C.red, f));
      }
    }
    this._hudClash(a, now, masked);
  }

  _hudClash(f, now, masked) {
    const R = this.replay;
    let kills = 0, through = 0;
    for (const e of this._events) {
      if (e.t > now) break;
      if (e.type === 'kill') kills++;
      else if (e.type === 'breach') through++;
    }
    const spent = [...f.ds].filter((c) => c !== '1').length;
    this._hud([['blue', R.defenders - spent, 'interceptors up'], ['red', R.attackers - kills - through, 'attackers inbound'],
      ['red', masked, 'under the mask'], ['green', kills, 'intercepted'], ['amber', through, 'got through']]);
    this._ticker(now, (e) => {
      if (e.type === 'kill') return ['kill', `INTERCEPT  I-${e.d} took out A-${e.a} at ${Math.round(e.z ?? 0)} m`];
      if (e.type === 'breach') return ['breach', `BREACH     A-${e.a} crossed the line at ${Math.round(e.z ?? 0)} m`];
      return ['miss', `miss       I-${e.d} missed A-${e.a}`];
    });
  }

  _drawRecon() {
    const R = this.replay, r = this.r, L = this.lines, T = this.tris, P = this.pts, m = this.m;
    const { k, a, b, ha, hb } = this._sample();
    const S = R.area, half = S / 2, G = R.grid, cs = S / G, now = this.t, s = this._scale();
    const X = (x) => x - half, Z = (y) => half - y;
    const top = R.max_altitude, det = R.detail_altitude;
    // the area on the ground, the full-detail height and the highest the swarm may fly
    const rect = (y, col) => {
      const c = [[-half, y, half], [half, y, half], [half, y, -half], [-half, y, -half]];
      for (let q = 0; q < 4; q++) L.line(c[q], c[(q + 1) % 4], col);
    };
    rect(1, rgba(C.cyan, 0.5));
    rect(det, rgba(C.green, 0.22));
    rect(top, rgba(C.grey, 0.16));
    for (const [x, z] of [[-half, half], [half, half], [half, -half], [-half, -half]]) L.line([x, 1, z], [x, top, z], rgba(C.grey, 0.14));
    this._labels.push([[half, det, half], `FULL DETAIL BELOW ${Math.round(det)} m`], [[half, top, half], `MAX ${Math.round(top)} m`]);
    const bx = X(R.base[0]), bz = Z(R.base[1]);
    L.circle(bx, 1, bz, 2.5 * s, rgba(C.amber, 0.8), 24);
    this._labels.push([[bx, 2 * s, bz], 'BASE']);
    // surveyed ground: brighter = seen in more detail; cells glow for a few seconds when they improve
    this._cellsAt(now);
    const qs = R.quality_step;
    let detail = 0;
    for (let c = 0; c < G * G; c++) {
      const lv = this._level[c];
      if (!lv) continue;
      const q = Math.min(1, lv * qs);
      detail += q;
      const fresh = Math.max(0, 1 - (now - this._levelT[c]) / 6);
      const ix = (c / G) | 0, iy = c - ix * G;
      const x0 = X(ix * cs) + 1, x1 = X((ix + 1) * cs) - 1, z0 = Z(iy * cs) - 1, z1 = Z((iy + 1) * cs) + 1;
      const col = [0.1 + 0.15 * q + 0.5 * fresh, 0.3 + 0.5 * q, 0.42 + 0.1 * q + 0.3 * fresh, 0.07 + 0.2 * q + 0.25 * fresh];
      T.tri([x0, 0.6, z0], [x1, 0.6, z0], [x1, 0.6, z1], col);
      T.tri([x0, 0.6, z0], [x1, 0.6, z1], [x0, 0.6, z1], col);
    }
    // targets: faint until found (amber), green once the report reached the operator
    const ts = Math.max(4, 0.6 * s);
    R.targets.forEach((p, q) => {
      const x = X(p[0]), z = Z(p[1]);
      const found = this._found[q] <= now, rep = this._reported[q] <= now;
      const col = rep ? C.green : found ? C.amber : C.grey;
      r.mesh(this.geo.dart, mat4.trs(m, x, 3, z, q, 0, 0, ts), col, { emissive: found ? 0.6 : 0.0, alpha: found ? 1 : 0.5, blend: !found });
      if (found) P.point([x, 8, z], (rep ? 1.6 : 2.4) * s, rgba(col, 0.8));
    });
    // drones, camera cones, radio links
    const N = R.drones, pos = [];
    let alt = 0, cams = 0;
    for (let d = 0; d < N; d++) {
      const p = this._pos(a.p, b.p, d, k, half, half);
      pos.push(p);
      alt += p[1];
      const lead = d === 0, cam = a.c[d] === '1';
      if (cam) cams++;
      const col = lead ? C.amber : cam ? C.cyan : C.grey;
      const [yaw] = this._dir(ha.p, hb.p, d, 0);
      r.mesh(this.geo.drone, mat4.trs(m, p[0], p[1], p[2], yaw, 0, 0, s), col, { emissive: 0.35 });
      r.mesh(this.geo.rotor, mat4.trs(m, p[0], p[1], p[2], yaw + now * 30, 0, 0, s), col, { alpha: 0.35, blend: true, cull: false });
      P.point(p, 2.9 * s, rgba(col, 0.6));
      if (cam) {
        // higher = a wider footprint but less detail (fainter cone)
        const fp = p[1] * R.camera_tan, q = clamp01((3 * det - p[1]) / (2 * det));
        L.circle(p[0], 1.2, p[2], fp, rgba(C.cyan, 0.25 + 0.55 * q), 32);
        for (let u = 0; u < 4; u++) {
          const ang = yaw + (u + 0.5) * Math.PI / 2;
          L.line(p, [p[0] + fp * Math.cos(ang), 1.2, p[2] - fp * Math.sin(ang)], rgba(C.cyan, 0.08 + 0.2 * q));
        }
      } else {
        L.line(p, [p[0], 0.5, p[2]], rgba(C.grey, 0.25));
      }
      if (lead) this._labels.push([[p[0], p[1] + 2 * s, p[2]], 'LEAD']);
    }
    if (this.show.links) {
      this._links(pos, R.comm_range, (u, v) => (a.l[u] === '1' && a.l[v] === '1' ? rgba(C.green, 0.55) : rgba(C.grey, 0.4)));
    }
    const found = this._found.filter((t) => t <= now).length, rep = this._reported.filter((t) => t <= now).length;
    const linked = [...a.l].filter((c) => c === '1').length;
    this._hud([['green', `${Math.round(100 * detail / (G * G))}%`, 'surveyed'], ['amber', `${found}/${R.targets.length}`, 'targets found'],
      ['green', rep, 'reported'], ['blue', `${linked}/${N}`, 'in contact'], ['blue', `${cams}/${N}`, 'cameras on'],
      ['blue', `${Math.round(alt / N)} m`, 'mean altitude']]);
    this._ticker(now, (e) => (e.type === 'found'
      ? ['found', `FOUND      target ${e.target} by drone ${e.drone}`]
      : ['reported', `REPORTED   target ${e.target} reached the operator`]));
  }

  // ------------------------------------------------------------------ overlays
  _hud(boxes) {
    const key = JSON.stringify(boxes);
    if (key === this._hudKey) return;
    this._hudKey = key;
    this.dom.hud.innerHTML = '';
    for (const [tone, v, label] of boxes) {
      const el = document.createElement('div');
      el.className = `hudbox ${tone}`;
      const bv = document.createElement('b'); bv.textContent = v;
      const sp = document.createElement('span'); sp.textContent = label;
      el.append(bv, sp);
      this.dom.hud.appendChild(el);
    }
  }

  _ticker(now, describe) {
    const last = [];
    for (const e of this._events) { if (e.t > now) break; last.push(e); }
    const shown = last.slice(-6);
    const key = shown.length + ':' + (shown.length ? shown[shown.length - 1].t : '');
    if (key === this._tickKey) return;
    this._tickKey = key;
    this.dom.ticker.innerHTML = '';
    for (const e of shown) {
      const [cls, text] = describe(e);
      const el = document.createElement('div');
      el.className = cls;
      el.textContent = `T+${e.t.toFixed(1).padStart(5, '0')}  ${text}`;
      this.dom.ticker.appendChild(el);
    }
  }

  _placeLabels() {
    const box = this.dom.labels;
    while (box.children.length < this._labels.length) box.appendChild(document.createElement('span'));
    [...box.children].forEach((el, q) => {
      const item = this._labels[q];
      const p = item ? this.r.project(item[0]) : null;
      if (!p) { el.style.display = 'none'; return; }
      el.style.display = '';
      el.textContent = item[1];
      el.style.left = `${p[0]}px`;
      el.style.top = `${p[1]}px`;
    });
  }
}
