// Where things are. The simulator has no geography - its costs come from device profiles and the shared
// channel - so the theatre gives the swarm a deterministic, purely visual flight: a route, a formation and
// nadir camera footprints. Everything that happens on this stage (who looks, who computes, what is sent,
// who fails) comes from the simulation trace.

export const AUTO = { terrain: 'urban', formation: 'wedge', route: 'racetrack' };

/** Fills in "auto" theatre settings from the swarm: satellites get an orbital track over the ocean. */
export function resolveTheatre(run) {
  const t = Object.assign({ terrain: 'auto', formation: 'auto', route: 'auto', altitude_m: 0, spacing_m: 0, speed_mps: 0, area_m: 1600 }, run.theatre || {});
  const profiles = run.profiles || [];
  const n = profiles.length;
  const space = n > 0 && profiles.every((p) => (p.tags || []).includes('satellite'));
  const pick = (k, v) => (t[k] && t[k] !== 'auto' ? t[k] : v);
  return {
    space,
    terrain: pick('terrain', space ? 'ocean' : 'urban'),
    formation: pick('formation', space ? 'column' : n <= 3 ? 'line' : 'wedge'),
    route: pick('route', space ? 'transit' : 'racetrack'),
    altitude: t.altitude_m || (space ? 560 : 110),
    spacing: t.spacing_m || (space ? 260 : 44),
    speed: t.speed_mps || (space ? 150 : 15),
    area: t.area_m || 1600,
    fov: space ? 0.55 : 0.8, // full nadir field of view (rad)
  };
}

function chaikin(pts, iterations = 3) {
  let p = pts;
  for (let k = 0; k < iterations; k++) {
    const q = [];
    for (let i = 0; i < p.length; i++) {
      const a = p[i], b = p[(i + 1) % p.length];
      q.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]], [0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
    }
    p = q;
  }
  return p;
}

function routePoints(route, area) {
  const pts = [];
  if (route === 'orbit') {
    const r = area * 0.3;
    for (let i = 0; i < 160; i++) { const a = (i / 160) * Math.PI * 2; pts.push([r * Math.cos(a), r * Math.sin(a)]); }
    return pts;
  }
  if (route === 'transit') {
    const a = area * 3.2, b = area * 0.45;
    for (let i = 0; i < 400; i++) { const t = (i / 400) * Math.PI * 2; pts.push([a * Math.cos(t), b * Math.sin(t)]); }
    return pts;
  }
  if (route === 'lawnmower') {
    const w = area * 0.62, lanes = 5, gap = area * 0.13, z0 = -gap * (lanes - 1) / 2;
    for (let i = 0; i < lanes; i++) {
      const z = z0 + i * gap, dir = i % 2 === 0 ? 1 : -1;
      pts.push([-dir * w / 2, z], [dir * w / 2, z]);
    }
    const zEnd = z0 + (lanes - 1) * gap;
    const xEnd = (lanes - 1) % 2 === 0 ? w / 2 : -w / 2;
    pts.push([xEnd + Math.sign(xEnd) * gap, zEnd + gap], [-w / 2 - gap, zEnd + gap], [-w / 2 - gap, z0 - gap * 0.6]);
    return chaikin(pts, 4);
  }
  // racetrack (also the fallback): two straights joined by half circles
  const L = area * 0.5, R = area * 0.17;
  const seg = 60;
  for (let i = 0; i <= seg; i++) pts.push([-L / 2 + (L * i) / seg, -R]);
  for (let i = 1; i < seg; i++) { const a = -Math.PI / 2 + (Math.PI * i) / seg; pts.push([L / 2 + R * Math.cos(a), R * Math.sin(a)]); }
  for (let i = 0; i <= seg; i++) pts.push([L / 2 - (L * i) / seg, R]);
  for (let i = 1; i < seg; i++) { const a = Math.PI / 2 + (Math.PI * i) / seg; pts.push([-L / 2 + R * Math.cos(a), R * Math.sin(a)]); }
  return pts;
}

/** Formation slot of member k (of n): [forward, right] metres relative to the formation reference. */
export function slot(formation, k, n, spacing) {
  if (formation === 'line') return [0, (k - (n - 1) / 2) * spacing];
  if (formation === 'column') return [-k * spacing, 0];
  if (formation === 'grid') {
    const cols = Math.ceil(Math.sqrt(n)), row = Math.floor(k / cols), col = k % cols;
    return [-row * spacing, (col - (cols - 1) / 2) * spacing];
  }
  if (formation === 'ring') {
    if (n === 1) return [0, 0];
    const r = Math.max(spacing, (spacing * n) / (2 * Math.PI)), a = (k / n) * Math.PI * 2;
    return [r * Math.cos(a), r * Math.sin(a)];
  }
  if (k === 0) return [0, 0]; // wedge: node 0 at the apex
  const row = Math.ceil(k / 2), side = k % 2 ? 1 : -1;
  return [-row * spacing * 0.9, side * row * spacing * 0.8];
}

export class Kinematics {
  constructor(theatre, n) {
    this.th = theatre;
    this.n = n;
    this.hold = theatre.route === 'hold';
    const pts = this.hold ? [[0, 0], [1, 0]] : routePoints(theatre.route, theatre.area);
    this.px = new Float64Array(pts.length + 1);
    this.pz = new Float64Array(pts.length + 1);
    this.cum = new Float64Array(pts.length + 1);
    for (let i = 0; i <= pts.length; i++) {
      const p = pts[i % pts.length];
      this.px[i] = p[0]; this.pz[i] = p[1];
      if (i) this.cum[i] = this.cum[i - 1] + Math.hypot(p[0] - this.px[i - 1], p[1] - this.pz[i - 1]);
    }
    this.length = this.cum[pts.length];
    this.slots = Array.from({ length: n }, (_, k) => slot(theatre.formation, k, n, theatre.spacing));
    this.footprint = 2 * theatre.altitude * Math.tan(theatre.fov / 2);
    let ext = 0;
    for (let i = 0; i < pts.length; i++) ext = Math.max(ext, Math.abs(pts[i][0]), Math.abs(pts[i][1]));
    this.extent = ext + theatre.spacing * n + this.footprint;
  }

  /** Formation reference point and unit heading at time t. */
  base(t) {
    if (this.hold) {
      const a = t * 0.04;
      return [0, 0, Math.cos(a), Math.sin(a)];
    }
    let s = (this.th.speed * t) % this.length;
    if (s < 0) s += this.length;
    let lo = 0, hi = this.cum.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (this.cum[mid] <= s) lo = mid; else hi = mid; }
    const segLen = this.cum[hi] - this.cum[lo] || 1, k = (s - this.cum[lo]) / segLen;
    const x = this.px[lo] + (this.px[hi] - this.px[lo]) * k, z = this.pz[lo] + (this.pz[hi] - this.pz[lo]) * k;
    const dx = this.px[hi] - this.px[lo], dz = this.pz[hi] - this.pz[lo], l = Math.hypot(dx, dz) || 1;
    return [x, z, dx / l, dz / l];
  }

  /** World position [x, y, z] and heading [hx, hz] of member i at time t. */
  pose(i, t) {
    const [bx, bz, hx, hz] = this.base(t);
    const [f, r] = this.slots[i] || [0, 0];
    const rx = -hz, rz = hx; // right-hand side of the heading on the ground
    const bob = this.th.space ? 0 : Math.sin(t * 1.3 + i * 1.7) * 1.4;
    const sway = this.th.space ? 0 : Math.sin(t * 0.7 + i * 2.3) * 1.2;
    return { p: [bx + hx * f + rx * (r + sway), this.th.altitude + bob + (this.th.space ? i * 6 : 0), bz + hz * f + rz * (r + sway)], h: [hx, hz] };
  }

  /** Polyline of the route for drawing (y = 0.6 m). */
  routeLine() {
    const out = [];
    for (let i = 0; i < this.px.length; i++) out.push([this.px[i], 0.6, this.pz[i]]);
    return out;
  }
}
