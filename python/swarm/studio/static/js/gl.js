// A small WebGL renderer - just what the theatre needs, no libraries (the console runs offline):
// lit meshes with rim light and distance fog, colored lines / translucent triangles, glowing points.
// Matrices are column-major Float32Array(16), like WebGL expects. World: metres, Y up, ground = XZ plane.

export const mat4 = {
  identity() { const m = new Float32Array(16); m[0] = m[5] = m[10] = m[15] = 1; return m; },

  perspective(out, fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    out.fill(0);
    out[0] = f / aspect; out[5] = f; out[10] = (far + near) * nf; out[11] = -1; out[14] = 2 * far * near * nf;
    return out;
  },

  lookAt(out, eye, center, up) {
    let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let len = Math.hypot(zx, zy, zz) || 1; zx /= len; zy /= len; zz /= len;
    let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
    len = Math.hypot(xx, xy, xz) || 1; xx /= len; xy /= len; xz /= len;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
    out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
    out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[15] = 1;
    return out;
  },

  multiply(out, a, b) {
    const r = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let i = 0; i < 4; i++) {
        r[c * 4 + i] = a[i] * b[c * 4] + a[4 + i] * b[c * 4 + 1] + a[8 + i] * b[c * 4 + 2] + a[12 + i] * b[c * 4 + 3];
      }
    }
    out.set(r);
    return out;
  },

  /** Translation * Ry(yaw) * Rx(pitch) * Rz(roll) * Scale. Models face +X; yaw turns +X towards -Z. */
  trs(out, x, y, z, yaw = 0, pitch = 0, roll = 0, sx = 1, sy = sx, sz = sx) {
    const cy = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cr = Math.cos(roll), sr = Math.sin(roll);
    out[0] = (cy * cr + syw * sp * sr) * sx; out[1] = cp * sr * sx; out[2] = (-syw * cr + cy * sp * sr) * sx; out[3] = 0;
    out[4] = (-cy * sr + syw * sp * cr) * sy; out[5] = cp * cr * sy; out[6] = (syw * sr + cy * sp * cr) * sy; out[7] = 0;
    out[8] = syw * cp * sz; out[9] = -sp * sz; out[10] = cy * cp * sz; out[11] = 0;
    out[12] = x; out[13] = y; out[14] = z; out[15] = 1;
    return out;
  },

  apply(m, x, y, z, w = 1) {
    return [m[0] * x + m[4] * y + m[8] * z + m[12] * w, m[1] * x + m[5] * y + m[9] * z + m[13] * w,
      m[2] * x + m[6] * y + m[10] * z + m[14] * w, m[3] * x + m[7] * y + m[11] * z + m[15] * w];
  },
};

/** Heading vector (dx, dz) on the ground -> yaw for models that face +X. */
export const yawOf = (dx, dz) => Math.atan2(-dz, dx);

const VS_SOLID = `
attribute vec3 aPos; attribute vec3 aNor; attribute vec3 aCol;
uniform mat4 uVP; uniform mat4 uM;
varying vec3 vN; varying vec3 vW; varying vec3 vC;
void main() {
  vec4 w = uM * vec4(aPos, 1.0);
  vW = w.xyz; vN = (uM * vec4(aNor, 0.0)).xyz; vC = aCol;
  gl_Position = uVP * w;
}`;
const FS_SOLID = `
precision mediump float;
uniform vec3 uColor; uniform float uAlpha; uniform float uEmissive; uniform vec3 uEye; uniform vec3 uFog; uniform float uFogDen;
varying vec3 vN; varying vec3 vW; varying vec3 vC;
void main() {
  vec3 n = normalize(vN);
  float d = max(dot(n, normalize(vec3(0.35, 0.9, 0.25))), 0.0);
  vec3 v = normalize(uEye - vW);
  float rim = pow(1.0 - max(dot(n, v), 0.0), 2.5);
  vec3 base = uColor * vC;
  vec3 c = base * (0.30 + 0.70 * d) + base * rim * 0.6 + base * uEmissive;
  float f = clamp(1.0 - exp(-pow(length(uEye - vW) * uFogDen, 2.0)), 0.0, 1.0);
  gl_FragColor = vec4(mix(c, uFog, f), uAlpha);
}`;
const VS_FLAT = `
attribute vec3 aPos; attribute vec4 aCol; uniform mat4 uVP;
varying vec4 vC; varying vec3 vW;
void main() { vC = aCol; vW = aPos; gl_Position = uVP * vec4(aPos, 1.0); }`;
const FS_FLAT = `
precision mediump float;
uniform vec3 uEye; uniform vec3 uFog; uniform float uFogDen;
varying vec4 vC; varying vec3 vW;
void main() {
  float f = clamp(1.0 - exp(-pow(length(uEye - vW) * uFogDen, 2.0)), 0.0, 1.0);
  gl_FragColor = vec4(mix(vC.rgb, uFog, f), vC.a * (1.0 - 0.9 * f));
}`;
const VS_POINT = `
attribute vec3 aPos; attribute float aSize; attribute vec4 aCol;
uniform mat4 uVP; uniform float uScale;
varying vec4 vC;
void main() {
  vec4 p = uVP * vec4(aPos, 1.0);
  gl_Position = p;
  gl_PointSize = clamp(aSize * uScale / max(p.w, 0.001), 2.5, 56.0);
  vC = aCol;
}`;
const FS_POINT = `
precision mediump float;
varying vec4 vC;
void main() {
  float r = length(gl_PointCoord - 0.5) * 2.0;
  float a = smoothstep(1.0, 0.0, r);
  gl_FragColor = vec4(vC.rgb * (0.7 + 0.9 * (1.0 - r)), vC.a * a * a);
}`;

const A_POS = 0, A_NOR = 1, A_COL = 2, A_SIZE = 3;

function program(gl, vs, fs) {
  const mk = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, A_POS, 'aPos');
  gl.bindAttribLocation(p, A_NOR, 'aNor');
  gl.bindAttribLocation(p, A_COL, 'aCol');
  gl.bindAttribLocation(p, A_SIZE, 'aSize');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name] = gl.getUniformLocation(p, info.name); }
  return { p, u };
}

/** A growable Float32Array that dynamic geometry is written into every frame. */
export class Batch {
  constructor(stride, capacity = 4096) { this.stride = stride; this.data = new Float32Array(stride * capacity); this.n = 0; }
  reset() { this.n = 0; }
  _room(k) {
    const need = (this.n + k) * this.stride;
    if (need > this.data.length) { const d = new Float32Array(Math.max(need, this.data.length * 2)); d.set(this.data); this.data = d; }
  }
  /** Line segment from a to b, colours [r,g,b,a]. */
  line(a, b, ca, cb = ca) {
    this._room(2);
    const d = this.data; let o = this.n * 7;
    d[o++] = a[0]; d[o++] = a[1]; d[o++] = a[2]; d[o++] = ca[0]; d[o++] = ca[1]; d[o++] = ca[2]; d[o++] = ca[3];
    d[o++] = b[0]; d[o++] = b[1]; d[o++] = b[2]; d[o++] = cb[0]; d[o++] = cb[1]; d[o++] = cb[2]; d[o++] = cb[3];
    this.n += 2;
  }
  tri(a, b, c, col) {
    this._room(3);
    const d = this.data; let o = this.n * 7;
    for (const p of [a, b, c]) { d[o++] = p[0]; d[o++] = p[1]; d[o++] = p[2]; d[o++] = col[0]; d[o++] = col[1]; d[o++] = col[2]; d[o++] = col[3]; }
    this.n += 3;
  }
  point(p, size, col) {
    this._room(1);
    const d = this.data; let o = this.n * 8;
    d[o++] = p[0]; d[o++] = p[1]; d[o++] = p[2]; d[o++] = size; d[o++] = col[0]; d[o++] = col[1]; d[o++] = col[2]; d[o++] = col[3];
    this.n += 1;
  }
  /** Ground circle (or arc from a0 to a1) at height y. */
  circle(cx, y, cz, r, col, seg = 48, a0 = 0, a1 = Math.PI * 2) {
    let px = cx + r * Math.cos(a0), pz = cz + r * Math.sin(a0);
    for (let i = 1; i <= seg; i++) {
      const a = a0 + (a1 - a0) * (i / seg), x = cx + r * Math.cos(a), z = cz + r * Math.sin(a);
      this.line([px, y, pz], [x, y, z], col);
      px = x; pz = z;
    }
  }
}

export class Renderer {
  constructor(canvas) {
    const opts = { antialias: true, alpha: false, depth: true, premultipliedAlpha: false, powerPreference: 'high-performance' };
    const gl = canvas.getContext('webgl2', opts) || canvas.getContext('webgl', opts);
    if (!gl) throw new Error('WebGL is not available in this browser');
    this.gl = gl;
    this.canvas = canvas;
    this.solid = program(gl, VS_SOLID, FS_SOLID);
    this.flat = program(gl, VS_FLAT, FS_FLAT);
    this.pts = program(gl, VS_POINT, FS_POINT);
    this.dyn = gl.createBuffer();
    this.vp = mat4.identity();
    this.eye = [0, 0, 0];
    this.fog = [0, 0, 0];
    this.fogDen = 0.0004;
    this.pxScale = 1;
    this.dpr = 1;
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr)), hgt = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== hgt) { this.canvas.width = w; this.canvas.height = hgt; }
    this.dpr = dpr;
    this.gl.viewport(0, 0, w, hgt);
    return w / hgt;
  }

  begin(view, proj, eye, fog, fogDen, fovy) {
    const gl = this.gl;
    mat4.multiply(this.vp, proj, view);
    this.eye = eye; this.fog = fog; this.fogDen = fogDen;
    this.pxScale = this.canvas.height / (2 * Math.tan(fovy / 2));
    gl.clearColor(fog[0], fog[1], fog[2], 1);
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }

  /** Static mesh from {pos, nor, col?} arrays. */
  geometry({ pos, nor, col }) {
    const gl = this.gl;
    const mk = (arr) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.STATIC_DRAW); return b; };
    return { pos: mk(pos), nor: mk(nor), col: col ? mk(col) : null, count: pos.length / 3 };
  }

  /** Static line set from a Batch (lines). */
  staticLines(batch) {
    const gl = this.gl, b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, batch.data.subarray(0, batch.n * 7), gl.STATIC_DRAW);
    return { buf: b, count: batch.n };
  }

  _uniforms(prog) {
    const gl = this.gl, u = prog.u;
    gl.useProgram(prog.p);
    gl.uniformMatrix4fv(u.uVP, false, this.vp);
    if (u.uEye) gl.uniform3fv(u.uEye, this.eye);
    if (u.uFog) gl.uniform3fv(u.uFog, this.fog);
    if (u.uFogDen) gl.uniform1f(u.uFogDen, this.fogDen);
  }

  mesh(geo, model, color, { alpha = 1, emissive = 0, blend = false, cull = true } = {}) {
    const gl = this.gl, u = this.solid.u;
    this._uniforms(this.solid);
    gl.uniformMatrix4fv(u.uM, false, model);
    gl.uniform3fv(u.uColor, color);
    gl.uniform1f(u.uAlpha, alpha);
    gl.uniform1f(u.uEmissive, emissive);
    gl.bindBuffer(gl.ARRAY_BUFFER, geo.pos); gl.enableVertexAttribArray(A_POS); gl.vertexAttribPointer(A_POS, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, geo.nor); gl.enableVertexAttribArray(A_NOR); gl.vertexAttribPointer(A_NOR, 3, gl.FLOAT, false, 0, 0);
    if (geo.col) { gl.bindBuffer(gl.ARRAY_BUFFER, geo.col); gl.enableVertexAttribArray(A_COL); gl.vertexAttribPointer(A_COL, 3, gl.FLOAT, false, 0, 0); }
    else { gl.disableVertexAttribArray(A_COL); gl.vertexAttrib3f(A_COL, 1, 1, 1); }
    if (blend) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); }
    if (!cull) gl.disable(gl.CULL_FACE);
    gl.drawArrays(gl.TRIANGLES, 0, geo.count);
    if (!cull) gl.enable(gl.CULL_FACE);
    if (blend) { gl.disable(gl.BLEND); gl.depthMask(true); }
    gl.disableVertexAttribArray(A_NOR);
    gl.disableVertexAttribArray(A_COL);
  }

  _flat(buf, data, count, mode, additive) {
    if (!count) return;
    const gl = this.gl;
    this._uniforms(this.flat);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    if (data) gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(A_POS); gl.vertexAttribPointer(A_POS, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(A_COL); gl.vertexAttribPointer(A_COL, 4, gl.FLOAT, false, 28, 12);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, additive ? gl.ONE : gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    gl.disable(gl.CULL_FACE);
    gl.drawArrays(mode, 0, count);
    gl.enable(gl.CULL_FACE);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.disableVertexAttribArray(A_COL);
  }

  drawStaticLines(h) { this._flat(h.buf, null, h.count, this.gl.LINES, false); }
  lines(batch, additive = false) { this._flat(this.dyn, batch.data.subarray(0, batch.n * 7), batch.n, this.gl.LINES, additive); }
  tris(batch, additive = false) { this._flat(this.dyn, batch.data.subarray(0, batch.n * 7), batch.n, this.gl.TRIANGLES, additive); }

  points(batch) {
    if (!batch.n) return;
    const gl = this.gl;
    this._uniforms(this.pts);
    gl.uniform1f(this.pts.u.uScale, this.pxScale);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dyn);
    gl.bufferData(gl.ARRAY_BUFFER, batch.data.subarray(0, batch.n * 8), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(A_POS); gl.vertexAttribPointer(A_POS, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(A_SIZE); gl.vertexAttribPointer(A_SIZE, 1, gl.FLOAT, false, 32, 12);
    gl.enableVertexAttribArray(A_COL); gl.vertexAttribPointer(A_COL, 4, gl.FLOAT, false, 32, 16);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.depthMask(false);
    gl.drawArrays(gl.POINTS, 0, batch.n);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.disableVertexAttribArray(A_SIZE);
    gl.disableVertexAttribArray(A_COL);
  }

  /** World -> CSS pixels of the canvas; null when behind the camera. */
  project(p) {
    const c = mat4.apply(this.vp, p[0], p[1], p[2], 1);
    if (c[3] <= 0.01) return null;
    const x = c[0] / c[3], y = c[1] / c[3];
    return [(x * 0.5 + 0.5) * this.canvas.clientWidth, (0.5 - y * 0.5) * this.canvas.clientHeight, c[3]];
  }
}

/** Orbit camera: drag to rotate, right-drag / shift-drag to pan, wheel to zoom; values ease towards targets. */
export class OrbitCamera {
  constructor(el) {
    this.el = el;
    this.target = [0, 0, 0]; this.want = { target: [0, 0, 0], yaw: 0.85, pitch: 0.62, dist: 750 };
    this.yaw = 0.85; this.pitch = 0.62; this.dist = 750;
    this.fovy = 48 * Math.PI / 180;
    this.minDist = 20; this.maxDist = 20000;
    this.onPan = null; this.onClick = null;
    this.view = mat4.identity(); this.proj = mat4.identity();
    this._drag = null;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => this._down(e));
    el.addEventListener('pointermove', (e) => this._move(e));
    el.addEventListener('pointerup', (e) => this._up(e));
    el.addEventListener('pointercancel', () => { this._drag = null; el.classList.remove('dragging'); });
    el.addEventListener('wheel', (e) => { e.preventDefault(); this.zoom(Math.exp(e.deltaY * 0.0012)); }, { passive: false });
  }

  zoom(k) { this.want.dist = Math.min(this.maxDist, Math.max(this.minDist, this.want.dist * k)); }

  _down(e) {
    this.el.setPointerCapture(e.pointerId);
    this._drag = { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, pan: e.button === 2 || e.shiftKey, moved: false };
    this.el.classList.add('dragging');
  }

  _move(e) {
    const d = this._drag;
    if (!d) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y;
    d.x = e.clientX; d.y = e.clientY;
    if (Math.abs(e.clientX - d.sx) + Math.abs(e.clientY - d.sy) > 4) d.moved = true;
    if (d.pan) {
      const s = this.want.dist * 0.0016;
      const fx = Math.sin(this.yaw), fz = Math.cos(this.yaw);  // ground direction away from the camera
      this.want.target[0] += (-dx * fz - dy * fx) * s;
      this.want.target[2] += (dx * fx - dy * fz) * s;
      if (this.onPan) this.onPan();
    } else {
      this.want.yaw -= dx * 0.0055;
      this.want.pitch = Math.min(1.55, Math.max(0.06, this.want.pitch + dy * 0.004));
    }
  }

  _up(e) {
    const d = this._drag;
    this._drag = null;
    this.el.classList.remove('dragging');
    if (d && !d.moved && this.onClick) {
      const r = this.el.getBoundingClientRect();
      this.onClick(e.clientX - r.left, e.clientY - r.top, e);
    }
  }

  get dragging() { return !!this._drag; }

  update(dt, aspect) {
    const k = 1 - Math.exp(-dt * 7);
    for (let i = 0; i < 3; i++) this.target[i] += (this.want.target[i] - this.target[i]) * k;
    this.yaw += (this.want.yaw - this.yaw) * k;
    this.pitch += (this.want.pitch - this.pitch) * k;
    this.dist += (this.want.dist - this.dist) * k;
    const cp = Math.cos(this.pitch);
    this.eye = [this.target[0] + this.dist * cp * Math.sin(this.yaw), this.target[1] + this.dist * Math.sin(this.pitch),
      this.target[2] + this.dist * cp * Math.cos(this.yaw)];
    mat4.lookAt(this.view, this.eye, this.target, [0, 1, 0]);
    mat4.perspective(this.proj, this.fovy, aspect, Math.max(0.5, this.dist * 0.01), Math.max(30000, this.dist * 30));
  }
}
