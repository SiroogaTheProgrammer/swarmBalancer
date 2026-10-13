"""RECON: a swarm surveys a square area from a base in its middle and reports what it finds - in 3-D.

Height is the survey trade-off. A camera sees the ground within ``altitude * tan(fov / 2)``, so flying
higher covers ground faster - but the detail it resolves falls with height: full detail up to
``detail_altitude``, none at three times that. Every cell of the shared coverage map keeps the best detail
any camera saw it in (0..1), and "area surveyed" is the mean of that. Each hidden target needs its own
amount of detail to be recognised (some stand out, some hardly do); a camera passing over it finds it if
it resolves at least that much.

The other goals pull against coverage, which is the point of the test:
    stay together       reports reach the operator only through the lead drone (index 0, it carries the
                        long-range link), relayed hop by hop over the one-hop mesh (3-D distances); a drone
                        outside the chain keeps its reports until it is back in contact
    keep the load low   the camera + image brain is the big CPU and energy cost; radio and flying count too

The map is ``grid x grid`` cells. Every array has a leading batch axis B of independent episodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.devices import get_profile

from .config import ReconConfig
from .core import (HEADER, M, MSG, SITE_RADIO, SITE_START, SITE_TARGETS, Noise, clip_hv, clip_norm,
                   device_macs_per_s, gather, heard_mean, links, nearest, norm, tag, take_batch)

DIRS = np.array([[np.cos(a), np.sin(a)] for a in np.arange(8) * np.pi / 4])  # E, NE, N, ... (x east, y north)
LOOK_CELLS = 4        # the 8 look-ahead windows are centred this many cells away ...
LOOK_HALF = 2         # ... and are 2 * LOOK_HALF + 1 cells wide
STATE_BYTES = 11      # radioed with every decision: 3-D position, camera and link flags
REPORT_BYTES = 16     # one target report: id, position, class, confidence
START_ALTITUDE = 40.0  # the swarm lifts off from the base to this height
QUALITY_STEP = 0.1    # the replay records each cell's detail in steps of this much


def detail_at(altitude, detail_altitude: float):
    """Detail a camera resolves from ``altitude``: 1 up to ``detail_altitude``, falling to 0 at 3x."""
    return np.clip((3.0 * detail_altitude - np.asarray(altitude)) / (2.0 * detail_altitude), 0.0, 1.0)


@dataclass
class ReconObs:
    """What each drone knows when it decides. Arrays are [B, N, ...]; vectors are 3-D (x east, y north,
    z altitude) except the horizontal ``gap_rel``."""

    cfg: ReconConfig
    t: float
    index: np.ndarray        # [N] own id; 0 is the lead
    pos: np.ndarray          # [B, N, 3] metres from the area's south-west corner; z = altitude
    vel: np.ndarray          # [B, N, 3]
    linked: np.ndarray       # [B, N] in radio contact with the lead, directly or through friends
    lead_rel: np.ndarray     # [B, N, 3] where the lead was when last heard, relative to me
    friend_rel: np.ndarray   # [B, N, M, 3] nearest friends heard this step, relative to me
    friend_mask: np.ndarray  # [B, N, M]
    friends_n: np.ndarray    # [B, N]
    look: np.ndarray         # [B, N, 8] detail still missing in a window towards E, NE, N, ... (0..1)
    gap_rel: np.ndarray      # [B, N, 2] nearest ground no camera has seen yet, relative to me (0: none left)
    uncovered_here: np.ndarray  # [B, N] share of the ground in my camera's view it would still improve
    footprint: np.ndarray    # [B, N] radius of the ground my camera sees from my altitude
    detail: np.ndarray       # [B, N] detail my camera resolves from my altitude (1 = full)
    msg: np.ndarray          # [B, N, MSG] mean learned message heard

    FEATURES = 6 + 4 + 4 * M + 1 + 8 + 2 + 1 + 1 + MSG + 3

    def take(self, idx) -> "ReconObs":
        return take_batch(self, idx)

    def role(self) -> np.ndarray:
        """A fixed code in [-1, 1] per drone (see ``AttackerObs.role``)."""
        return (self.index * 0.6180339887) % 1.0 * 2.0 - 1.0

    def features(self) -> np.ndarray:
        c = self.cfg
        B, N = self.linked.shape
        R, S = c.comm_range, c.area
        own = np.concatenate([self.vel[..., :2] / c.speed, self.vel[..., 2:3] / c.climb, self.pos[..., :2] / S * 2 - 1,
                              self.pos[..., 2:3] / c.max_altitude], -1)
        lead = np.concatenate([np.clip(self.lead_rel / R, -3, 3), self.linked[..., None].astype(float)], -1)
        fm = self.friend_mask[..., None].astype(float)
        fr = np.concatenate([np.clip(self.friend_rel / R, -2, 2) * fm, fm], -1).reshape(B, N, -1)
        friends = np.concatenate([fr, self.friends_n[..., None] / max(1, N - 1)], -1)
        gap = np.clip(self.gap_rel / (S / 4), -2, 2)
        tail = np.stack([np.full((B, N), self.t / c.duration), np.broadcast_to(self.role(), (B, N)),
                         np.broadcast_to((self.index == 0).astype(float), (B, N))], -1)
        return np.concatenate([own, lead, friends, self.look, gap, self.uncovered_here[..., None],
                               self.detail[..., None], self.msg, tail], -1)


class ReconEnv:
    """``keys``: as in :class:`~swarm.arena.clash.ClashEnv` (same key = same hidden targets and random numbers)."""

    def __init__(self, cfg: ReconConfig, batch: int = 1, seed: int = 0, record: bool = False,
                 keys: np.ndarray | None = None):
        self.cfg = cfg
        self.B, self.N = int(batch), cfg.drones
        self.rng = np.random.default_rng(seed)
        self.record = record
        if keys is not None and np.shape(keys) != (self.B,):
            raise ValueError("keys needs one non-negative int per episode")
        self.noise = Noise(self.rng, keys)
        self.G = cfg.grid
        self.cs = cfg.area / cfg.grid
        self.tan = float(np.tan(np.radians(cfg.camera_fov / 2)))
        # footprint stencils: cell offsets (and their centres, relative to the drone's own cell) for every
        # radius up to the widest footprint; each step uses the smallest one that fits the highest drone
        self.stencils = []
        for r in range(min(int(np.ceil(cfg.max_altitude * self.tan / self.cs)) + 1, self.G) + 1):
            o = np.arange(-r, r + 1)
            st = np.stack(np.meshgrid(o, o, indexing="ij"), -1).reshape(-1, 2)
            self.stencils.append((st, (st + 0.5) * self.cs))
        cc = (np.arange(self.G) + 0.5) * self.cs
        cx, cy = np.meshgrid(cc, cc, indexing="ij")
        self.cx, self.cy = cx.ravel(), cy.ravel()  # cell centres; flat cell index = ix * grid + iy
        self.cc = cc.astype(np.float32)            # float32: the nearest-gap search is the hot loop
        self.lo = np.array([0.0, 0.0, cfg.min_altitude])
        self.hi = np.array([cfg.area, cfg.area, cfg.max_altitude])
        self._reset()

    def _hidden(self, rng: np.random.Generator, n: int) -> list[np.ndarray]:
        """``n`` sets of hidden targets: positions [n, T, 2] and the detail each needs to be recognised [n, T]."""
        c = self.cfg
        return [rng.uniform(0.03, 0.97, (n, c.targets, 2)) * c.area, rng.uniform(0.05, 1.0, (n, c.targets))]

    def _reset(self) -> None:
        c, B, N = self.cfg, self.B, self.N
        k = np.arange(N)
        cols = 4
        start = np.stack([c.area / 2 + (k % cols - (cols - 1) / 2) * 30.0,
                          c.area / 2 + (k // cols - ((N - 1) // cols) / 2) * 30.0,
                          np.full(N, np.clip(START_ALTITUDE, c.min_altitude, c.max_altitude))], -1)
        self.pos = (np.repeat(start[None], B, 0)
                    + self.noise.normal(0, 2.0, (B, N, 3), tag(SITE_START)) * np.array([1.0, 1.0, 0.0]))
        self.vel = np.zeros((B, N, 3))
        self.quality = np.zeros((B, self.G * self.G))   # best detail each cell was seen in
        T = c.targets
        self.tpos, self.tneed = self.noise.scenarios(B, SITE_TARGETS, self._hidden)
        self.found = np.zeros((B, T), bool)
        self.found_t = np.full((B, T), np.nan)
        self.finder = np.full((B, T), -1)
        self.reported = np.zeros((B, T), bool)
        self.report_t = np.full((B, T), np.nan)
        self.lead_seen = np.repeat(self.pos[:, :1], N, 1)
        self.msg = np.zeros((B, N, MSG))
        self.dir = np.zeros((B, N, 2))
        self.speed = np.zeros((B, N))
        self.cam = np.zeros((B, N), bool)
        self.alt_cmd = self.pos[..., 2].copy()
        self.t, self.steps = 0.0, 0
        self.done = np.zeros(B, bool)
        z = lambda: np.zeros(B)  # noqa: E731
        self.stat = {"cam": z(), "linked": z(), "spacing": z(), "bytes": z(), "dec": z(), "dist": z(), "effort": z(),
                     "auc": z(), "alt": z(), "t_end": np.full(B, c.duration), "t50": np.full(B, np.nan),
                     "t90": np.full(B, np.nan)}
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self._rec_level = np.zeros(self.G * self.G, int)
        self._radio()
        self._fp = self._footprint()
        if self.record:
            self._frame()

    # ------------------------------------------------------------------ helpers
    def _radio(self) -> None:
        """Who hears whom this step, and who is in the lead's radio chain (multi-hop)."""
        c, B, N = self.cfg, self.B, self.N
        self.hear = links(self.pos, np.ones((B, N), bool), c.comm_range, c.comm_loss, self.noise,
                          tag(SITE_RADIO, self.steps))
        reach = np.zeros((B, N), bool)
        reach[:, 0] = True
        for _ in range(N - 1):
            nxt = reach | (self.hear & reach[:, None, :]).any(2)
            if (nxt == reach).all():
                break
            reach = nxt
        self.linked = reach
        self.lead_seen = np.where(reach[..., None], self.pos[:, :1], self.lead_seen)

    def _detail(self) -> np.ndarray:
        return detail_at(self.pos[..., 2], self.cfg.detail_altitude)

    def _footprint(self) -> tuple[np.ndarray, np.ndarray]:
        """Flat cell ids [B, N, O] around each drone and which of them its camera sees from its altitude."""
        G, cs = self.G, self.cs
        xy = self.pos[..., :2]
        radius = self.pos[..., 2] * self.tan
        st, st_c = self.stencils[min(int(np.ceil(radius.max() / cs)) + 1, len(self.stencils) - 1)]
        base = np.floor(xy / cs).astype(int)                    # each drone's own cell
        frac = xy - base * cs                                   # where in that cell it is
        dx = st_c[:, 0] - frac[..., 0:1]                        # [B, N, O] cell centre relative to the drone
        dy = st_c[:, 1] - frac[..., 1:2]
        ix = base[..., 0:1] + st[:, 0]
        iy = base[..., 1:2] + st[:, 1]
        within = ((dx * dx + dy * dy <= (radius * radius)[..., None])
                  & (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G))
        return np.clip(ix, 0, G - 1) * G + np.clip(iy, 0, G - 1), within

    def _look(self, missing: np.ndarray) -> np.ndarray:
        B, G = self.B, self.G
        sat = np.zeros((B, G + 1, G + 1))
        sat[:, 1:, 1:] = missing.reshape(B, G, G).cumsum(1).cumsum(2)
        ci = np.clip(np.floor(self.pos[..., :2] / self.cs).astype(int), 0, G - 1)
        ctr = ci[:, :, None, :] + np.rint(DIRS * LOOK_CELLS).astype(int)[None, None]
        lo = np.clip(ctr - LOOK_HALF, 0, G)
        hi = np.clip(ctr + LOOK_HALF + 1, 0, G)
        b = np.arange(B)[:, None, None]
        s = (sat[b, hi[..., 0], hi[..., 1]] - sat[b, lo[..., 0], hi[..., 1]]
             - sat[b, hi[..., 0], lo[..., 1]] + sat[b, lo[..., 0], lo[..., 1]])
        area = (hi - lo).prod(-1)
        return np.where(area > 0, s / np.maximum(area, 1), 0.0)

    def _gap(self, unseen: np.ndarray) -> np.ndarray:
        B, N, G = self.B, self.N, self.G
        # squared distance to every cell centre, separable on the regular grid: dx^2 + dy^2 [B, N, G, G]
        dx2 = (self.cc - self.pos[..., 0:1].astype(np.float32)) ** 2
        dy2 = (self.cc - self.pos[..., 1:2].astype(np.float32)) ** 2
        d2 = (dx2[..., :, None] + dy2[..., None, :]).reshape(B, N, G * G)
        j = np.where(unseen[:, None, :], d2, np.float32(np.inf)).argmin(-1)
        gap = np.stack([self.cx[j] - self.pos[..., 0], self.cy[j] - self.pos[..., 1]], -1)
        return np.where(unseen.any(1)[:, None, None], gap, 0.0)

    # ------------------------------------------------------------------ observations and decisions
    def observe(self) -> ReconObs:
        c, B, N = self.cfg, self.B, self.N
        rel = self.pos[:, None, :, :] - self.pos[:, :, None, :]          # [b, i, j] = pos_j - pos_i
        idx, ok = nearest(np.where(self.hear, norm(rel), np.inf), M)
        msg, n = heard_mean(self.hear, self.msg)
        flat, within = self._fp  # the footprint of the current positions, from the last step
        q = self._detail()
        gain = within & (self.quality[np.arange(B)[:, None, None], flat] < q[..., None] - 1e-6)
        here = gain.sum(-1) / np.maximum(within.sum(-1), 1)
        return ReconObs(c, self.t, np.arange(N), self.pos.copy(), self.vel.copy(), self.linked.copy(),
                        self.lead_seen - self.pos, np.where(ok[..., None], gather(rel, idx), 0.0), ok, n,
                        self._look(1.0 - self.quality), self._gap(self.quality <= 0.0), here,
                        self.pos[..., 2] * self.tan, q, msg)

    def apply(self, obs: ReconObs, direction: np.ndarray, speed: np.ndarray, camera: np.ndarray,
              altitude: np.ndarray | None = None, msg: np.ndarray | None = None,
              due: np.ndarray | None = None) -> None:
        """``direction`` [B, N, 2] horizontal (any length; zero = hover), ``speed`` 0..1 of top speed,
        ``camera`` on/off, ``altitude`` metres to climb or descend to (None: keep the last one); ``due`` [B]:
        the episodes whose drones decided (default: all) - the others keep their last commands."""
        c = self.cfg
        live = ~self.done if due is None else ~self.done & due
        L = live[:, None]
        d = np.asarray(direction, dtype=float)[..., :2]
        n = norm(d)[..., None]
        self.dir = np.where(L[..., None], np.where(n > 1e-6, d / np.maximum(n, 1e-9), 0.0), self.dir)
        self.speed = np.where(L, np.clip(speed, 0.0, 1.0), self.speed)
        self.cam = np.where(L, np.asarray(camera, dtype=bool), self.cam)
        if altitude is not None:
            self.alt_cmd = np.where(L, np.clip(altitude, c.min_altitude, c.max_altitude), self.alt_cmd)
        if msg is not None:
            self.msg = np.where(L[..., None], np.clip(msg, -1, 1), self.msg)
        self.stat["bytes"] += live * self.N * (HEADER + STATE_BYTES + MSG)
        self.stat["dec"] += live * self.N

    # ------------------------------------------------------------------ physics
    def step(self) -> None:
        c, dt, B, N = self.cfg, self.cfg.dt, self.B, self.N
        act = ~self.done
        live = act[:, None, None]
        vz = np.clip(0.5 * (self.alt_cmd - self.pos[..., 2]), -c.climb, c.climb)
        v_des = np.concatenate([self.dir * (self.speed * c.speed)[..., None], vz[..., None]], -1)
        acc = clip_norm(v_des - self.vel, c.accel) * live
        p0 = self.pos
        self.vel = clip_hv(self.vel + acc * dt, c.speed, c.climb) * live
        self.pos = np.minimum(np.maximum(self.pos + self.vel * dt, self.lo), self.hi)  # np.clip: 25x slower here
        self.vel = (self.pos - p0) / dt  # the edge of the area, the floor and the ceiling stop you
        self.stat["dist"] += norm(self.pos - p0).sum(1)
        self.stat["effort"] += norm(acc).sum(1) * dt
        self.stat["alt"] += self.pos[..., 2].mean(1) * dt * act
        self.t += dt
        self.steps += 1
        self._radio()
        cam = self.cam & act[:, None]
        self._fp = flat, within = self._footprint()
        q = self._detail()
        b, n, o = np.nonzero(within & cam[..., None])
        np.maximum.at(self.quality, (b, flat[b, n, o]), q[b, n])
        cov = self.quality.mean(1)
        self.stat["auc"] += cov * dt * act
        for key, level in (("t50", 0.5), ("t90", 0.9)):
            self.stat[key] = np.where(np.isnan(self.stat[key]) & (cov >= level) & act, self.t, self.stat[key])
        if c.targets:
            self._targets(cam, q, act)
        self.stat["cam"] += cam.sum(1) * dt
        self.stat["linked"] += self.linked.mean(1) * dt * act
        if N > 1:
            d = norm(self.pos[:, None] - self.pos[:, :, None]) + np.eye(N)[None] * 1e12
            self.stat["spacing"] += d.min(2).mean(1) * dt * act
        complete = (cov >= 0.999) & self.reported.all(1)
        finished = act & ((self.t >= c.duration - 1e-9) | complete)
        self.stat["t_end"] = np.where(finished, self.t, self.stat["t_end"])
        self.done |= finished
        if self.record:
            self._frame()

    def _targets(self, cam: np.ndarray, q: np.ndarray, act: np.ndarray) -> None:
        dist = norm(self.tpos[:, None, :, :] - self.pos[:, :, None, :2])        # [B, N, T] over the ground
        sees = ((dist <= (self.pos[..., 2] * self.tan)[..., None]) & cam[..., None]
                & (q[..., None] >= self.tneed[:, None, :]))
        new = sees.any(1) & ~self.found
        if new.any():
            who = np.where(sees, dist, np.inf).argmin(1)
            self.finder = np.where(new, who, self.finder)
            self.found_t = np.where(new, self.t, self.found_t)
            self.found |= new
            self.stat["bytes"] += new.sum(1) * REPORT_BYTES
            if self.record:
                for k in np.nonzero(new[0])[0]:
                    self.events.append({"t": round(self.t, 2), "type": "found", "target": int(k),
                                        "drone": int(self.finder[0, k])})
        # store and forward: a report gets out once its finder is in the lead's radio chain
        chain = self.linked[np.arange(self.B)[:, None], np.maximum(self.finder, 0)]
        rep = self.found & ~self.reported & chain & act[:, None]
        if rep.any():
            self.reported |= rep
            self.report_t = np.where(rep, self.t, self.report_t)
            if self.record:
                for k in np.nonzero(rep[0])[0]:
                    self.events.append({"t": round(self.t, 2), "type": "reported", "target": int(k),
                                        "drone": int(self.finder[0, k])})

    def _frame(self) -> None:
        level = np.floor(self.quality[0] / QUALITY_STEP + 1e-9).astype(int)
        changed = np.nonzero(level > self._rec_level)[0]
        self._rec_level[changed] = level[changed]
        self.frames.append({"p": np.rint(self.pos[0]).astype(int).ravel().tolist(),
                            "c": "".join(np.where(self.cam[0], "1", "0")),
                            "l": "".join(np.where(self.linked[0], "1", "0")),
                            "nc": changed.tolist(), "nq": level[changed].tolist()})

    # ------------------------------------------------------------------ results
    def summary(self, brain_macs=0.0) -> dict[str, np.ndarray]:
        """Per-episode results. ``brain_macs``: cost of the swarm brain per drone per decision (one number,
        or one per episode)."""
        c, N = self.cfg, self.N
        t_end = self.stat["t_end"]
        T = max(1, c.targets)
        cov = self.quality.mean(1)
        n_rep = self.reported.sum(1)
        delay = np.where(self.reported, self.report_t - np.where(self.reported, self.found_t, 0.0), 0.0).sum(1)
        cam_duty = self.stat["cam"] / (N * t_end)
        perception = c.perception_macs * c.camera_fps * cam_duty            # MAC/s per drone
        brain = brain_macs * self.stat["dec"] / (N * t_end)
        dev = get_profile(c.device)
        energy = ((perception + brain) * t_end * dev.energy_per_mac_nj
                  + self.stat["bytes"] / N * 8 * dev.energy_per_bit_nj) * 1e-9
        connected = self.stat["linked"] / t_end
        # mean coverage over the whole mission time (finishing early keeps the final coverage)
        auc = (self.stat["auc"] + cov * (c.duration - t_end)) / c.duration
        effort = self.stat["effort"] / (N * t_end) / c.accel
        reported = n_rep / T if c.targets else np.ones_like(cov)
        fitness = (0.5 * cov + 0.5 * auc + 0.5 * reported
                   - c.w_cohesion * 0.8 * (1 - connected) - c.w_load * (0.2 * cam_duty + 0.1 * effort))
        return {
            "coverage": cov, "seen": (self.quality > 0).mean(1), "coverage_avg": auc,
            "t50": self.stat["t50"], "t90": self.stat["t90"],
            "found": self.found.sum(1) / T, "reported": reported,
            "report_delay": np.where(n_rep > 0, delay / np.maximum(n_rep, 1), np.nan),
            "connected": connected, "spacing_m": self.stat["spacing"] / t_end, "camera_duty": cam_duty,
            "altitude": self.stat["alt"] / t_end,
            "cpu": (perception + brain) / device_macs_per_s(c.device),
            "radio_kbps": self.stat["bytes"] * 8 / N / t_end / 1e3, "energy_j": energy,
            "flown_km": self.stat["dist"] / N / 1e3, "effort": effort, "duration": t_end, "fitness": fitness,
        }

    def replay(self) -> dict:
        c = self.cfg
        return {"kind": "recon", "dt": c.dt, "frames": self.frames, "events": self.events, "area": c.area,
                "grid": self.G, "quality_step": QUALITY_STEP, "camera_tan": round(self.tan, 4),
                "detail_altitude": c.detail_altitude, "max_altitude": c.max_altitude, "comm_range": c.comm_range,
                "drones": self.N, "base": [c.area / 2, c.area / 2],
                "targets": np.round(self.tpos[0], 1).tolist()}
