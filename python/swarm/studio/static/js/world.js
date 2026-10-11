// Procedural geometry: airframes, ground contacts and terrain. Everything is built from boxes and discs at
// load time, so the console ships no model files. Units are metres; models face +X, Y is up.

import { Batch, mat4 } from './gl.js';

function pushFace(g, a, b, c, d, n, col) {
  for (const p of [a, b, c, a, c, d]) {
    g.pos.push(p[0], p[1], p[2]); g.nor.push(n[0], n[1], n[2]);
    if (g.col) g.col.push(col[0], col[1], col[2]);
  }
}

/** Axis-aligned box centred at (cx, cy, cz). */
export function box(g, w, hgt, d, cx = 0, cy = 0, cz = 0, col = [1, 1, 1]) {
  const x0 = cx - w / 2, x1 = cx + w / 2, y0 = cy - hgt / 2, y1 = cy + hgt / 2, z0 = cz - d / 2, z1 = cz + d / 2;
  pushFace(g, [x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [0, 1, 0], col);   // top
  pushFace(g, [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [0, -1, 0], col);  // bottom
  pushFace(g, [x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [1, 0, 0], col);   // +x
  pushFace(g, [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1, 0, 0], col);  // -x
  pushFace(g, [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], col);   // +z
  pushFace(g, [x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1], col);  // -z
  return g;
}

/** Flat disc facing up (and down), radius r at height y. */
export function disc(g, r, cx = 0, cy = 0, cz = 0, seg = 20, col = [1, 1, 1]) {
  for (let i = 0; i < seg; i++) {
    const a0 = (i / seg) * Math.PI * 2, a1 = ((i + 1) / seg) * Math.PI * 2;
    const p0 = [cx, cy, cz], p1 = [cx + r * Math.cos(a0), cy, cz + r * Math.sin(a0)], p2 = [cx + r * Math.cos(a1), cy, cz + r * Math.sin(a1)];
    for (const [p, n] of [[p0, 1], [p2, 1], [p1, 1], [p0, -1], [p1, -1], [p2, -1]]) {
      g.pos.push(p[0], p[1], p[2]); g.nor.push(0, n, 0);
      if (g.col) g.col.push(col[0], col[1], col[2]);
    }
  }
  return g;
}

/** Appends `src` transformed by matrix m into `dst`. */
export function merge(dst, src, m) {
  for (let i = 0; i < src.pos.length; i += 3) {
    const p = mat4.apply(m, src.pos[i], src.pos[i + 1], src.pos[i + 2], 1);
    const n = mat4.apply(m, src.nor[i], src.nor[i + 1], src.nor[i + 2], 0);
    const l = Math.hypot(n[0], n[1], n[2]) || 1;
    dst.pos.push(p[0], p[1], p[2]); dst.nor.push(n[0] / l, n[1] / l, n[2] / l);
    if (dst.col) dst.col.push(...(src.col ? src.col.slice(i, i + 3) : [1, 1, 1]));
  }
  return dst;
}

const G = (withCol = true) => ({ pos: [], nor: [], col: withCol ? [] : null });

/** Multirotor airframe: body, arms, motor pods, a nadir camera. Rotor discs are drawn separately (they spin). */
export function airframe(arms = 4, span = 1) {
  const g = G();
  box(g, 0.9 * span, 0.28 * span, 0.62 * span, 0, 0, 0, [1, 1, 1]);
  box(g, 0.32 * span, 0.12 * span, 0.32 * span, 0.12 * span, 0.2 * span, 0, [0.65, 0.65, 0.65]);   // GPS / radio puck
  box(g, 0.2 * span, 0.2 * span, 0.2 * span, 0.18 * span, -0.22 * span, 0, [0.35, 0.35, 0.35]);   // camera gimbal
  for (let i = 0; i < arms; i++) {
    const a = (i + 0.5) / arms * Math.PI * 2;
    const arm = box(G(), 1.25 * span, 0.08 * span, 0.09 * span, 0.62 * span, 0, 0, [0.55, 0.55, 0.55]);
    merge(g, arm, mat4.trs(new Float32Array(16), 0, 0, 0, a));
    const pod = box(G(), 0.16 * span, 0.18 * span, 0.16 * span, 1.24 * span, 0.04 * span, 0, [0.8, 0.8, 0.8]);
    merge(g, pod, mat4.trs(new Float32Array(16), 0, 0, 0, a));
  }
  return g;
}

/** Rotor discs at the arm tips (drawn translucent and spinning). */
export function rotors(arms = 4, span = 1) {
  const g = G(false);
  for (let i = 0; i < arms; i++) {
    const a = (i + 0.5) / arms * Math.PI * 2;
    disc(g, 0.48 * span, Math.cos(a) * 1.24 * span, 0.14 * span, -Math.sin(a) * 1.24 * span, 18);
  }
  return g;
}

export function satellite(scale = 1, panels = 2) {
  const g = G();
  box(g, 1.1 * scale, 1.1 * scale, 1.1 * scale, 0, 0, 0, [1, 1, 1]);
  box(g, 0.3 * scale, 0.5 * scale, 0.3 * scale, 0, -0.75 * scale, 0, [0.4, 0.4, 0.4]);       // imager
  for (let s = -1; s <= 1; s += 2) {
    for (let k = 0; k < panels; k++) {
      box(g, 0.9 * scale, 0.05 * scale, 1.0 * scale, 0, 0, s * (1.1 + k * 1.05) * scale, [0.35, 0.5, 1.0]);
    }
  }
  box(g, 0.06 * scale, 0.9 * scale, 0.06 * scale, 0, 0.9 * scale, 0, [0.7, 0.7, 0.7]);         // antenna
  return g;
}

/** Ground contacts, one mesh per class of the simulator's world (index = class). */
export function contacts() {
  const vehicle = box(G(false), 5, 2.0, 2.4, 0, 1.0, 0);
  box(vehicle, 2.2, 1.0, 2.0, -0.6, 2.4, 0);
  const building = box(G(false), 16, 9, 16, 0, 4.5, 0);
  box(building, 11, 2.5, 11, 0, 10.2, 0);
  const road = box(G(false), 34, 0.4, 4, 0, 0.2, 0);
  const aircraft = box(G(false), 16, 1.6, 2.2, 0, 1.2, 0);
  box(aircraft, 3.2, 0.5, 15, 0.8, 1.3, 0);
  box(aircraft, 2.0, 0.4, 5.5, -6.8, 1.4, 0);
  box(aircraft, 1.6, 2.8, 0.4, -7.0, 2.6, 0);
  const ghost = box(G(false), 3, 3, 3, 0, 1.5, 0);
  return [ghost, vehicle, building, road, aircraft];
}

// ------------------------------------------------------------------ terrain
function rng(seed) {
  let s = (seed * 2654435761) >>> 0 || 1;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
}

export const TERRAIN = {
  urban: { ground: [0.035, 0.05, 0.065], grid: [0.06, 0.17, 0.21], fog: [0.012, 0.022, 0.032] },
  plains: { ground: [0.03, 0.055, 0.04], grid: [0.06, 0.18, 0.15], fog: [0.012, 0.024, 0.026] },
  desert: { ground: [0.07, 0.06, 0.04], grid: [0.2, 0.16, 0.09], fog: [0.03, 0.026, 0.02] },
  arctic: { ground: [0.08, 0.1, 0.12], grid: [0.2, 0.3, 0.36], fog: [0.04, 0.05, 0.065] },
  ocean: { ground: [0.01, 0.03, 0.06], grid: [0.04, 0.14, 0.26], fog: [0.006, 0.014, 0.03] },
};

/** Ground plane, a coordinate grid and static props for a terrain type, covering roughly +-extent metres. */
export function terrain(type, extent, seed = 7) {
  const pal = TERRAIN[type] || TERRAIN.urban;
  const R = rng(seed);
  const ground = G(false);
  const big = extent * 6;
  box(ground, big * 2, 1, big * 2, 0, -0.5, 0);

  const grid = new Batch(7, 8192);
  const step = extent > 4000 ? 500 : 100, major = step * 5, span = Math.ceil(extent * 1.6 / major) * major;
  const gc = pal.grid;
  for (let v = -span; v <= span; v += step) {
    const isMajor = Math.abs(v % major) < 1e-6;
    const c = [gc[0], gc[1], gc[2], isMajor ? 0.55 : 0.22];
    grid.line([v, 0.05, -span], [v, 0.05, span], c);
    grid.line([-span, 0.05, v], [span, 0.05, v], c);
  }

  const props = G();
  const add = (w, hgt, d, x, z, col) => box(props, w, hgt, d, x, hgt / 2, z, col);
  const lim = extent * 1.5;
  if (type === 'urban') {
    const block = 90, street = 26;
    for (let bx = -lim; bx < lim; bx += block + street) {
      for (let bz = -lim; bz < lim; bz += block + street) {
        if (R() < 0.18) continue; // parks and plazas
        const n = 1 + Math.floor(R() * 4);
        for (let k = 0; k < n; k++) {
          const w = 14 + R() * 32, d = 14 + R() * 32, hgt = 6 + Math.pow(R(), 2.2) * 70;
          const x = bx + w / 2 + R() * Math.max(1, block - w), z = bz + d / 2 + R() * Math.max(1, block - d);
          const s = 0.08 + R() * 0.05;
          add(w, hgt, d, x, z, [s, s * 1.25, s * 1.5]);
        }
      }
    }
  } else if (type === 'plains') {
    for (let i = 0; i < 260; i++) {
      const w = 60 + R() * 160, d = 60 + R() * 160, s = 0.05 + R() * 0.05;
      add(w, 0.6, d, (R() * 2 - 1) * lim, (R() * 2 - 1) * lim, [s * 0.8, s * 1.4, s * 0.7]);
    }
    for (let i = 0; i < 40; i++) add(10 + R() * 12, 5 + R() * 6, 8 + R() * 10, (R() * 2 - 1) * lim, (R() * 2 - 1) * lim, [0.16, 0.15, 0.13]);
  } else if (type === 'desert') {
    for (let i = 0; i < 140; i++) {
      const w = 80 + R() * 220, s = 0.1 + R() * 0.05;
      add(w, 2 + R() * 9, w * (0.3 + R() * 0.3), (R() * 2 - 1) * lim, (R() * 2 - 1) * lim, [s * 1.3, s * 1.1, s * 0.75]);
    }
  } else if (type === 'arctic') {
    for (let i = 0; i < 160; i++) {
      const w = 20 + R() * 70, s = 0.18 + R() * 0.1;
      add(w, 2 + R() * 14, w * (0.5 + R() * 0.6), (R() * 2 - 1) * lim, (R() * 2 - 1) * lim, [s * 0.9, s * 1.05, s * 1.2]);
    }
  }
  return { ground, grid, props: props.pos.length ? props : null, palette: pal };
}
