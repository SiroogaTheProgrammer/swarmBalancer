"""DEFENCE / OFFENCE: a wall of interceptor drones against an incoming swarm, in 3-D, many episodes at once.

The airspace is a box: x across the front (|x| <= width/2), y towards the enemy, z altitude (up to the
ceiling). The protected line is the vertical plane y = 0 - the wall. Attackers start around y = depth and
score by crossing it at any height; interceptors score by intercepting. An interceptor that is not chasing
anyone waits at its guard post: by default its station in a two-layer line just in front of the wall (when
one is spent, the line re-spaces over the rest), but its brain may post it anywhere up to ``guard_depth``
of the way out - pickets far forward spot a low attack long before the line could. Each side knows only
what its sensors see and what its friends radio it - range-limited, lossy, one decision late: the mesh.

Height matters three ways:
    changing height is slower than flying level (a climb rate, limited separately from the top speed);
    below ``low_altitude`` terrain hides a drone - it is seen only within a third of the sensor range -
    but terrain following costs it a quarter of its speed;
    sensors and radios measure true 3-D distances.

What is radioed (range-limited, lossy, every decision):
    both sides  each drone's sightings of the enemy (its tracks), so a drone knows about everything the
                friends it hears can see - the mesh's shared picture, which is what lets a whole wall react
                to an attack concentrated where only a few interceptors can see it. With ``relay`` > 1 every
                drone also repeats the tracks it heard, so a sighting crosses up to that many hops;
    defenders   which attacker each is going for and how soon it can get there (its "claim"), plus MSG
                learned channels - enough to stop several interceptors piling onto one target, if the
                brain uses it;
    attackers   whether each is being chased, plus MSG learned channels.

Every array has a leading batch axis B of independent episodes, so a whole training population is
evaluated at once. ``record=True`` keeps a replay of episode 0 for the 3D view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import ClashConfig
from .core import (HEADER, K, KD, MSG, SITE_ARRIVALS, SITE_HITS, SITE_RADIO, SITE_START, T_NORM, TRACK_BYTES,
                   Noise, clip_hv, clip_norm, device_macs_per_s, dot, gather, heard_mean, intercept_eta, links,
                   nearest, norm, relay_tracks, shared, sq_dists, tag, take_batch, toward, unit)

STATION_FRACTION = 0.1         # interceptor stations: this fraction of the depth in front of the line, ...
STATION_LAYERS = (0.15, 0.55)  # ... alternating between these fractions of the ceiling (a two-layer wall)
CHASE_RANGE = 400.0            # an attacker feels "chased" by an interceptor closing in within this range
LOW_VISIBLE = 1 / 3            # a drone below the terrain mask is seen at this fraction of the sensor range ...
LOW_SPEED = 0.75               # ... and flies at most this fraction of its top speed (terrain following)
FLOOR = 5.0                    # nobody flies lower than this (m)
SUBSTEP = 0.5                  # s: the longest physics step (the autopilot loop); ``dt`` is the decision interval
CLAIM_BYTES = 4                # a claim: attacker id and ETA (sent, and repeated once by every friend that hears it)


@dataclass
class DefenderObs:
    """What each defender knows when it decides. Arrays are [B, D, ...]; slots = the KD nearest attackers it
    knows about - seen by its own sensors or by a friend it heard. Vectors are 3-D: x across the front,
    y towards the enemy, z altitude."""

    cfg: ClashConfig
    t: float
    index: np.ndarray       # [D] own id
    alive: np.ndarray       # [B, D]
    pos: np.ndarray         # [B, D, 3]
    vel: np.ndarray         # [B, D, 3]
    station: np.ndarray     # [B, D, 3] own default station in the line (it re-spaces as interceptors are spent)
    slot_id: np.ndarray     # [B, D, KD] attacker (track) id
    slot_mask: np.ndarray   # [B, D, KD] slot holds a known attacker
    slot_rel: np.ndarray    # [B, D, KD, 3] attacker position relative to me
    slot_vel: np.ndarray    # [B, D, KD, 3] attacker velocity
    slot_dist: np.ndarray   # [B, D, KD]
    slot_eta: np.ndarray    # [B, D, KD] my time to intercept it, climbing included (inf: I cannot catch it)
    slot_wall: np.ndarray   # [B, D, KD] its distance to the line
    slot_wall_time: np.ndarray  # [B, D, KD] its time to the line at its current speed
    claims: np.ndarray      # [B, D, KD] friends going for it (claims travel two hops)
    claim_eta: np.ndarray   # [B, D, KD] the best ETA among them, aged to "from now" (inf: unclaimed)
    slot_mine: np.ndarray   # [B, D, KD] it is the attacker I am going for now
    my_claim_eta: np.ndarray  # [B, D] the ETA I radioed with my claim, aged like theirs (inf: no claim)
    friends_n: np.ndarray   # [B, D] friends heard this step
    friends_rel: np.ndarray  # [B, D, 3] mean friend position relative to me
    msg: np.ndarray         # [B, D, MSG] mean learned message heard

    SLOT_FEATURES = 14
    CONTEXT_FEATURES = 6 + 3 + 4 + MSG + 3 + 1 + 1
    FEATURES = SLOT_FEATURES + 1 + CONTEXT_FEATURES  # per row; see features()

    def take(self, idx) -> "DefenderObs":
        return take_batch(self, idx)

    def role(self) -> np.ndarray:
        """A fixed code in [-1, 1] per interceptor (golden-ratio spread), so one shared brain can still post
        some of them forward as pickets and keep the rest in the line - by thresholding it."""
        return (self.index * 0.6180339887) % 1.0 * 2.0 - 1.0

    def claim_reference(self) -> np.ndarray:
        """[B, D, KD] the ETA a friend's claim must beat for that attacker to count as theirs.

        For my own target I compare the claim I radioed with theirs (made at the same decision and aged
        the same way, so both drones reach the same verdict); for any other one, a friend that is about
        as quick already has it.
        """
        return np.where(self.slot_mine, self.my_claim_eta[..., None], self.slot_eta + 0.25)

    def features(self) -> np.ndarray:
        """One row per option a defender has - each tracked attacker, then "hold station" - as
        [B, D, KD + 1, FEATURES]: the option's own features, a hold flag, then the defender's context
        (repeated on every row). The mesh brain scores every row with the same small network and takes
        the best, so "go for a free, urgent attacker" is one rule however many it tracks."""
        c = self.cfg
        B, D = self.alive.shape
        S, v, vz, W, L, H = c.sensor_range, c.defender_speed, c.defender_climb, c.width, c.depth, c.ceiling
        m = self.slot_mask.astype(float)
        cap = lambda x: np.minimum(np.where(np.isfinite(x), x, 2 * T_NORM), 2 * T_NORM)  # noqa: E731
        se = cap(self.slot_eta)
        ref = cap(self.claim_reference())
        ce = np.where(np.isfinite(self.claim_eta), self.claim_eta, ref + T_NORM)
        adv = np.clip((ce - ref) / 10.0, -1, 1)  # > 0: nobody quicker has it
        r, u = self.slot_rel, self.slot_vel
        slots = np.stack([
            r[..., 0] / S, r[..., 1] / S, r[..., 2] / S, u[..., 0] / v, u[..., 1] / v, u[..., 2] / vz,
            self.slot_dist / S, se / T_NORM, self.slot_wall / L, np.minimum(self.slot_wall_time, 2 * T_NORM) / T_NORM,
            self.claims / 3.0, adv, self.slot_mine.astype(float), m], -1) * m[..., None]
        options = np.concatenate([
            np.concatenate([slots, np.zeros((B, D, KD, 1))], -1),
            np.concatenate([np.zeros((B, D, 1, self.SLOT_FEATURES)), np.ones((B, D, 1, 1))], -1)], 2)
        free = (self.slot_mask & (adv > 0)).sum(-1, keepdims=True) / KD
        p, w = self.pos, self.vel
        context = np.concatenate([
            w[..., :2] / v, w[..., 2:3] / vz, p[..., 1:2] / L, p[..., 0:1] / (W / 2), p[..., 2:3] / H,
            np.isfinite(self.my_claim_eta)[..., None].astype(float), m.sum(-1, keepdims=True) / KD, free,
            self.friends_n[..., None] / max(1, D - 1), self.friends_rel / c.comm_range, self.msg,
            (self.station - p) / np.array([W, W, H]), np.full((B, D, 1), self.t / c.duration),
            np.broadcast_to(self.role()[:, None], (B, D, 1))], -1)
        return np.concatenate([options, np.broadcast_to(context[:, :, None, :], (B, D, KD + 1, context.shape[-1]))], -1)


@dataclass
class AttackerObs:
    """What each attacker knows when it decides. Arrays are [B, A, ...]; slots = the K nearest defenders it
    knows about (seen by itself or by a friend it heard). Vectors are 3-D: x across the front, y towards the
    enemy (the line is at y = 0), z altitude."""

    cfg: ClashConfig
    t: float
    index: np.ndarray       # [A] own id
    alive: np.ndarray       # [B, A]
    pos: np.ndarray         # [B, A, 3]
    vel: np.ndarray         # [B, A, 3]
    slot_mask: np.ndarray   # [B, A, K]
    slot_rel: np.ndarray    # [B, A, K, 3] defender position relative to me
    slot_vel: np.ndarray    # [B, A, K, 3] defender velocity
    slot_dist: np.ndarray   # [B, A, K]
    slot_closing: np.ndarray  # [B, A, K] +1: flying straight at me
    chased: np.ndarray      # [B, A] I am being chased (radioed to friends)
    friends_n: np.ndarray   # [B, A]
    friends_rel: np.ndarray  # [B, A, 3]
    friends_chased: np.ndarray  # [B, A] share of heard friends that are being chased
    msg: np.ndarray         # [B, A, MSG]

    FEATURES = 7 + 9 * K + 5 + MSG + 3

    def take(self, idx) -> "AttackerObs":
        return take_batch(self, idx)

    def role(self) -> np.ndarray:
        """A fixed code in [-1, 1] per attacker (golden-ratio spread), so one shared brain can still give
        drones different jobs - e.g. a third of them decoys - by thresholding it."""
        return (self.index * 0.6180339887) % 1.0 * 2.0 - 1.0

    def masked(self) -> np.ndarray:
        """[B, A] flying below the terrain mask: hidden from all but close sensors, but slower."""
        return self.pos[..., 2] < self.cfg.low_altitude

    def features(self) -> np.ndarray:
        c = self.cfg
        B, A = self.alive.shape
        S, v, W, L, H = c.sensor_range, c.attacker_speed, c.width, c.depth, c.ceiling
        m = self.slot_mask.astype(float)
        r, u = self.slot_rel, self.slot_vel
        slots = np.stack([
            r[..., 0] / S, r[..., 1] / S, r[..., 2] / S, u[..., 0] / (2 * v), u[..., 1] / (2 * v),
            u[..., 2] / (2 * c.defender_climb), self.slot_dist / S, self.slot_closing, m], -1) * m[..., None]
        p, w = self.pos, self.vel
        own = np.concatenate([w[..., :2] / v, w[..., 2:3] / c.attacker_climb, p[..., 1:2] / L, p[..., 0:1] / (W / 2),
                              p[..., 2:3] / H, self.masked()[..., None].astype(float)], -1)
        friends = np.concatenate([self.friends_n[..., None] / max(1, A - 1), self.friends_rel / c.comm_range,
                                  self.friends_chased[..., None]], -1)
        tail = np.stack([np.full((B, A), self.t / c.duration), self.chased.astype(float),
                         np.broadcast_to(self.role(), (B, A))], -1)
        return np.concatenate([own, slots.reshape(B, A, -1), friends, self.msg, tail], -1)


class ClashEnv:
    """``keys`` (optional, one non-negative int per episode, see :class:`~swarm.arena.core.Noise`): each
    episode's arrivals and in-flight random numbers then come from its key alone. Episodes with the same key
    start from the same situation and meet the same radio losses and hit rolls - training compares brain
    variants that way - and a generation can be split across processes without changing any result."""

    def __init__(self, cfg: ClashConfig, batch: int = 1, seed: int = 0, record: bool = False,
                 keys: np.ndarray | None = None):
        self.cfg = cfg
        self.B, self.D, self.A = int(batch), cfg.defenders, cfg.attackers
        self.rng = np.random.default_rng(seed)
        self.record = record
        if keys is not None and np.shape(keys) != (self.B,):
            raise ValueError("keys needs one non-negative int per episode")
        self.noise = Noise(self.rng, keys)
        W, L, H = cfg.width, cfg.depth, cfg.ceiling
        # the two layers of the wall; the low one flies above the terrain mask
        self.layer_z = np.maximum(np.array(STATION_LAYERS) * H, min(2.0 * cfg.low_altitude, 0.5 * H))
        self.station0 = self._stations(np.ones((1, self.D), bool))[0]   # the line before anyone is spent
        # where a brain may post its interceptors: anywhere across the front, from the line out to guard_depth
        self.post_lo = np.array([-W / 2, 0.0, FLOOR])
        self.post_hi = np.array([W / 2, cfg.guard_depth * L, H])
        # the airspace box; attackers are not stopped at the line (crossing it is how they score)
        self.hi = np.array([W / 2, 1.5 * L, H])
        self.lo_d = np.array([-W / 2, -STATION_FRACTION * L, FLOOR])
        self.lo_a = np.array([-W / 2, -np.inf, FLOOR])
        self._reset()

    def _stations(self, alive: np.ndarray) -> np.ndarray:
        """Guard stations [B, D, 3] spaced evenly along the front over the interceptors still in the fight,
        in their order along the line, alternating the two layers: a spent interceptor leaves no hole."""
        c = self.cfg
        rank = np.cumsum(alive, 1) - 1
        n = np.maximum(alive.sum(1, keepdims=True), 1)
        x = (rank + 0.5) / n * c.width - c.width / 2
        z = np.where(rank % 2 == 0, self.layer_z[0], self.layer_z[1])
        return np.stack([x, np.full(x.shape, STATION_FRACTION * c.depth), z], -1)

    def posts(self) -> np.ndarray:
        """[B, D, 3] where each interceptor waits when it is not chasing anyone: the post its brain chose, else
        its station in the line."""
        return np.where(np.isnan(self.post), self.station, self.post)

    def _show_posts(self) -> None:
        """Replay: notes when episode 0's guard posts move - by more than a few metres, since a learned
        formation shifts a little with every decision."""
        if self.record:
            p = self.posts()[0]
            if np.abs(p - self._shown).max() > 10.0:
                self._shown = p
                self.restations.append({"t": round(self.t, 2), "station": np.round(p, 1).tolist()})

    # ------------------------------------------------------------------ setup
    def _arrivals(self, rng: np.random.Generator, n: int) -> list[np.ndarray]:
        """``n`` arrival patterns, [n, A, 3]."""
        c, A = self.cfg, self.A
        W, L, H = c.width, c.depth, c.ceiling
        z_hi = 0.75 * H
        z_lo = min(max(2.0 * c.low_altitude, 0.1 * H), z_hi)
        if c.spawn == "cluster":
            x = rng.uniform(-0.3, 0.3, (n, 1)) * W + rng.normal(0, 0.05 * W, (n, A))
            y = L + rng.uniform(0, 0.08 * L, (n, A))
            z = rng.uniform(z_lo, z_hi, (n, 1)) + rng.normal(0, 0.04 * H, (n, A))
        elif c.spawn == "waves":
            x = rng.uniform(-0.45, 0.45, (n, A)) * W
            y = L + (np.arange(A) % 2)[None] * 0.35 * L + rng.uniform(0, 0.08 * L, (n, A))
            z = rng.uniform(z_lo, z_hi, (n, A))
        else:
            x = rng.uniform(-0.45, 0.45, (n, A)) * W
            y = L + rng.uniform(0, 0.1 * L, (n, A))
            z = rng.uniform(z_lo, z_hi, (n, A))
        return [np.stack([np.clip(x, -W / 2, W / 2), y, np.clip(z, FLOOR, H)], -1)]

    def _reset(self) -> None:
        c, B, D, A = self.cfg, self.B, self.D, self.A
        self.station = np.repeat(self.station0[None], B, 0)
        self.dp = self.station + self.noise.normal(0, 3.0, (B, D, 3), tag(SITE_START))
        self.dv = np.zeros((B, D, 3))
        self.d_alive = np.ones((B, D), bool)
        self.ammo = np.full((B, D), c.shots)
        self.ap = self.noise.scenarios(B, SITE_ARRIVALS, self._arrivals)[0]
        self.a_y0 = self.ap[..., 1].copy()
        self.a_min_y = self.a_y0.copy()  # deepest point each attacker reached (for "advance")
        self.av = np.zeros((B, A, 3))
        self.av[..., 1] = -0.6 * c.attacker_speed
        self.alt_cmd = self.ap[..., 2].copy()  # attackers hold their height until a brain says otherwise
        self.a_alive = np.ones((B, A), bool)
        self.a_killed = np.zeros((B, A), bool)
        self.a_breached = np.zeros((B, A), bool)
        self.a_seen = np.full((B, A), np.inf)
        self.a_end = np.full((B, A), np.nan)
        self.target = np.full((B, D), -1)
        self.post = np.full((B, D, 3), np.nan)  # guard posts the brain chose (NaN: its station in the line)
        self.claim_eta = np.full((B, D), np.inf)
        self.claim_t = np.zeros((B, D))
        # each drone's track table when sightings are relayed: decisions since someone saw that enemy
        self.track_age = {"d": np.full((B, D, A), np.inf), "a": np.full((B, A, D), np.inf)}
        self.d_msg = np.zeros((B, D, MSG))
        self.a_msg = np.zeros((B, A, MSG))
        self.heading = np.zeros((B, A))
        self.speed = np.ones((B, A))
        self.a_chased = np.zeros((B, A), bool)
        self.steps, self.t = 0, 0.0
        self.done = np.zeros(B, bool)
        z = lambda: np.zeros(B)  # noqa: E731
        self.stat = {"locks": z(), "lock_pairs": z(), "effort_d": z(), "effort_a": z(), "dist_d": z(), "dist_a": z(),
                     "bytes_d": z(), "bytes_a": z(), "dec_d": z(), "dec_a": z(), "spent": z(), "misses": z(),
                     "air_a": z(), "low_a": z(), "alt_a": z(), "t_end": np.full(B, c.duration)}
        self.events: list[dict] = []
        self.frames: list[dict] = []
        self.restations: list[dict] = []   # replay: when the guard posts moved, and where (episode 0)
        self._shown = self.station0        # the guard posts the replay shows now
        self._pair()
        if self.record:
            self._frame()

    def _pair(self) -> None:
        """Distance from every defender to every attacker [B, D, A] - once per step, shared by both sides'
        sensors and the intercept check."""
        self.dist = np.sqrt(sq_dists(self.dp, self.ap))

    def _seen_within(self, pos: np.ndarray) -> np.ndarray:
        """How far away a drone at ``pos`` can be seen: the sensor range, less for one hidden by terrain."""
        c = self.cfg
        return np.where(pos[..., 2] < c.low_altitude, LOW_VISIBLE * c.sensor_range, c.sensor_range)

    def _known(self, seen: np.ndarray, hear: np.ndarray, side: str,
               due: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Which enemies each drone of one side (``"d"`` or ``"a"``) knows about, and how many track reports the
        side radios per episode. A sighting travels ``cfg.relay`` radio hops, one per decision (1: only to the
        friends that hear the drone that sees it)."""
        hops = self.cfg.relay
        if hops <= 1:
            return shared(seen, hear), seen.sum((1, 2))
        own_alive, enemy_alive = (self.d_alive, self.a_alive) if side == "d" else (self.a_alive, self.d_alive)
        age = self.track_age[side]
        broadcast, new = relay_tracks(seen, hear, age)
        new = np.where(own_alive[:, :, None] & enemy_alive[:, None, :], new, np.inf)
        if due is not None:  # only the episodes that decide now exchange reports
            new = np.where(due[:, None, None], new, age)
        self.track_age[side] = new
        return new < hops, ((broadcast < hops) & own_alive[:, :, None]).sum((1, 2))

    # ------------------------------------------------------------------ observations
    def defender_obs(self, due: np.ndarray | None = None) -> DefenderObs:
        """What every defender knows now. ``due`` [B]: the episodes whose defenders decide now (default:
        all) - only their radio traffic and first sightings are counted."""
        c, B, D = self.cfg, self.B, self.D
        bi = np.arange(B)[:, None, None]
        counting = ~self.done if due is None else ~self.done & due
        hear = links(self.dp, self.d_alive, c.comm_range, c.comm_loss, self.noise, tag(SITE_RADIO, self.steps, 0))
        seen = self.d_alive[:, :, None] & self.a_alive[:, None, :] & (self.dist <= self._seen_within(self.ap)[:, None, :])
        first = seen.any(1) & np.isinf(self.a_seen)
        if due is not None:
            first &= due[:, None]
        self.a_seen = np.where(first, self.t, self.a_seen)
        known, sent = self._known(seen, hear, "d", due)
        self.stat["bytes_d"] += sent * TRACK_BYTES * counting
        idx, ok = nearest(np.where(known, self.dist, np.inf), KD)
        s_rel = self.ap[bi, idx] - self.dp[:, :, None, :]
        s_vel = self.av[bi, idx]
        s_dist = np.where(ok, gather(self.dist, idx), 0.0)
        eta = np.where(ok, intercept_eta(s_rel, s_vel, c.defender_speed, c.defender_climb), np.inf)
        wall = np.maximum(self.ap[bi, idx, 1], 0.0)
        wall_time = wall / np.maximum(-s_vel[..., 1], 1.0)
        slot_id = np.where(ok, idx, -1)
        # a claim is stamped with the time it was made, so it arrives aged to "ETA from now"
        claim = np.where(self.target >= 0, np.maximum(self.claim_eta - (self.t - self.claim_t), 0.0), np.inf)
        # claims travel two hops (every interceptor repeats the claims it heard): one that learns of an
        # attacker through a friend's sighting also learns who is already going for it
        hf = hear.astype(np.float32)
        relay = (hear | (np.matmul(hf, hf) > 0.5)) & ~np.eye(D, dtype=bool)
        self.stat["bytes_d"] += hear.sum((1, 2)) * CLAIM_BYTES * counting
        # friends that claim each of my slots, and the best ETA among their claims
        same = relay[:, :, None, :] & (self.target[:, None, None, :] == slot_id[..., None]) & ok[..., None]
        claims = same.sum(-1).astype(float)
        best = np.where(same, claim[:, None, None, :], np.inf).min(-1)
        heard, n = heard_mean(hear, np.concatenate([self.d_msg, self.dp], -1))
        friends_rel = np.where(n[..., None] > 0, heard[..., MSG:] - self.dp, 0.0)
        mine = ok & (slot_id == self.target[..., None]) & (self.target[..., None] >= 0)
        return DefenderObs(c, self.t, np.arange(D), self.d_alive.copy(), self.dp.copy(), self.dv.copy(),
                           self.station.copy(), slot_id, ok, np.where(ok[..., None], s_rel, 0.0),
                           np.where(ok[..., None], s_vel, 0.0), s_dist, eta, np.where(ok, wall, 0.0),
                           np.where(ok, wall_time, 0.0), claims, best, mine, claim, n, friends_rel,
                           heard[..., :MSG])

    def attacker_obs(self, due: np.ndarray | None = None) -> AttackerObs:
        """What every attacker knows now; ``due`` as in :meth:`defender_obs`."""
        c, B, A = self.cfg, self.B, self.A
        bi, ai = np.arange(B)[:, None, None], np.arange(A)[None, :, None]
        counting = ~self.done if due is None else ~self.done & due
        dist = self.dist.transpose(0, 2, 1)                                  # [B, A, D]
        hear = links(self.ap, self.a_alive, c.comm_range, c.comm_loss, self.noise, tag(SITE_RADIO, self.steps, 1))
        seen = self.a_alive[:, :, None] & self.d_alive[:, None, :] & (dist <= self._seen_within(self.dp)[:, None, :])
        known, sent = self._known(seen, hear, "a", due)
        self.stat["bytes_a"] += sent * TRACK_BYTES * counting
        idx, ok = nearest(np.where(known, dist, np.inf), K)
        s_rel = np.where(ok[..., None], self.dp[bi, idx] - self.ap[:, :, None, :], 0.0)  # defender relative to me
        s_vel = np.where(ok[..., None], self.dv[bi, idx], 0.0)
        s_dist = np.where(ok, self.dist[bi, idx, ai], 0.0)
        closing = np.where(ok, -dot(s_vel, unit(s_rel)) / np.maximum(norm(s_vel), 1.0), 0.0)
        chased = (ok & (s_dist < CHASE_RANGE) & (closing > 0.6)).any(-1) & self.a_alive
        heard, n = heard_mean(hear, np.concatenate([self.a_msg, self.ap, self.a_chased[..., None]], -1))
        return AttackerObs(c, self.t, np.arange(A), self.a_alive.copy(), self.ap.copy(), self.av.copy(), ok, s_rel,
                           s_vel, s_dist, closing, chased, n,
                           np.where(n[..., None] > 0, heard[..., MSG:MSG + 3] - self.ap, 0.0), heard[..., -1],
                           heard[..., :MSG])

    # ------------------------------------------------------------------ decisions
    def apply_defenders(self, obs: DefenderObs, choice: np.ndarray, msg: np.ndarray | None = None,
                        due: np.ndarray | None = None, station: np.ndarray | None = None) -> None:
        """``choice`` [B, D]: slot index 0..KD-1 to go for, or -1 to hold station; ``due`` [B]: the episodes
        whose defenders decided (default: all) - the others keep their targets. ``station`` [B, D, 3]: where
        each one waits when it holds - kept inside the front, between the line and ``guard_depth``; NaN or
        None: in its station in the line (None also keeps any post chosen earlier)."""
        choice = np.asarray(choice).astype(int)
        ci = np.clip(choice, 0, KD - 1)
        bi, di = np.arange(self.B)[:, None], np.arange(self.D)[None, :]
        picked = (choice >= 0) & (choice < KD) & obs.slot_mask[bi, di, ci]
        tid = obs.slot_id[bi, di, ci]
        live = self.d_alive & ~self.done[:, None]
        if due is not None:
            live &= due[:, None]
        self.target = np.where(live & picked, tid, np.where(live, -1, self.target))
        eta = obs.slot_eta[bi, di, ci]
        self.claim_eta = np.where(live & picked, eta, np.where(live, np.inf, self.claim_eta))
        self.claim_t = np.where(live & picked, obs.t, self.claim_t)
        if msg is not None:
            self.d_msg = np.where(live[..., None], np.clip(msg, -1, 1), self.d_msg)
        if station is not None:
            # np.minimum / np.maximum keep NaN ("back to the line"); np.clip with per-axis bounds is slower
            post = np.minimum(np.maximum(np.asarray(station, dtype=float), self.post_lo), self.post_hi)
            self.post = np.where(live[..., None], post, self.post)
            self._show_posts()
        self.stat["bytes_d"] += live.sum(1) * (HEADER + CLAIM_BYTES + MSG)
        self.stat["dec_d"] += live.sum(1)

    def apply_attackers(self, obs: AttackerObs, heading: np.ndarray, speed: np.ndarray,
                        altitude: np.ndarray | None = None, msg: np.ndarray | None = None,
                        due: np.ndarray | None = None) -> None:
        """``heading`` [B, A]: radians off "straight at the line" (+ = towards +x); ``speed`` 0..1 of max;
        ``altitude`` metres to climb or descend to (None, or NaN for a drone: keep the last one); ``due`` [B]:
        the episodes whose attackers decided (default: all) - the others keep their last commands."""
        live = self.a_alive & ~self.done[:, None]
        if due is not None:
            live &= due[:, None]
        self.heading = np.where(live, np.clip(heading, -np.pi / 2, np.pi / 2), self.heading)
        self.speed = np.where(live, np.clip(speed, 0.2, 1.0), self.speed)
        if altitude is not None:
            alt = np.asarray(altitude, dtype=float)
            self.alt_cmd = np.where(live & ~np.isnan(alt), np.clip(alt, FLOOR, self.cfg.ceiling), self.alt_cmd)
        self.a_chased = obs.chased & live if due is None else np.where(due[:, None], obs.chased & live, self.a_chased)
        if msg is not None:
            self.a_msg = np.where(live[..., None], np.clip(msg, -1, 1), self.a_msg)
        self.stat["bytes_a"] += live.sum(1) * (HEADER + 1 + MSG)
        self.stat["dec_a"] += live.sum(1)

    # ------------------------------------------------------------------ physics
    def step(self) -> None:
        """One decision interval ``dt``. The autopilots fly the brains' last commands in physics sub-steps
        of at most SUBSTEP seconds - re-aiming at the intercept point every time - so decisions can be
        rarer than the flight control loop without making flight or intercepts coarser."""
        c, A = self.cfg, self.A
        act = ~self.done
        n = max(1, math.ceil(c.dt / SUBSTEP - 1e-9))
        for i in range(n):
            self._fly(c.dt / n, act, (self.steps + (i + 1) / n) * c.dt, i)
        self.steps += 1
        # who is locked onto what (piling onto one attacker is the failure a smart wall avoids)
        locked = (self.target[:, :, None] == np.arange(A)) & self.d_alive[:, :, None]
        counts = locked.sum(1) * self.a_alive
        self.stat["locks"] += (counts > 0).sum(1) * act
        self.stat["lock_pairs"] += counts.sum(1) * act
        finished = act & (~self.a_alive.any(1) | (self.t >= c.duration - 1e-9))
        self.stat["t_end"] = np.where(finished, self.t, self.stat["t_end"])
        self.done |= finished
        if self.record:
            self._frame()

    def _fly(self, dt: float, act: np.ndarray, t_end: float, sub: int) -> None:
        c, B = self.cfg, self.B
        bi = np.arange(B)[:, None]
        # terrain following below the mask costs speed
        vmax_d = np.where(self.dp[..., 2] < c.low_altitude, LOW_SPEED, 1.0) * c.defender_speed
        vmax_a = np.where(self.ap[..., 2] < c.low_altitude, LOW_SPEED, 1.0) * c.attacker_speed
        # defenders: straight at the predicted intercept point of the target they hold, else back to their post
        tid = np.maximum(self.target, 0)
        has = (self.target >= 0) & self.a_alive[bi, tid]
        self.target = np.where(has, self.target, -1)
        rel = self.ap[bi, tid] - self.dp
        tv = self.av[bi, tid]
        tau = intercept_eta(rel, tv, c.defender_speed, c.defender_climb)
        lead = rel + tv * np.where(np.isfinite(tau), np.minimum(tau, 30.0), 0.0)[..., None]
        v_hold = clip_hv(0.5 * (self.posts() - self.dp), vmax_d, c.defender_climb)
        v_des = np.where(has[..., None], toward(lead, vmax_d, c.defender_climb), v_hold)
        live_d = (self.d_alive & act[:, None])[..., None]
        acc_d = clip_norm((v_des - self.dv) / 0.5, c.defender_accel) * live_d
        # attackers: the commanded heading and speed, climbing or descending to the commanded altitude
        vh = self.speed * vmax_a
        vz = np.clip(0.5 * (self.alt_cmd - self.ap[..., 2]), -c.attacker_climb, c.attacker_climb)
        v_des_a = np.stack([np.sin(self.heading) * vh, -np.cos(self.heading) * vh, vz], -1)
        live_a = (self.a_alive & act[:, None])[..., None]
        acc_a = clip_norm((v_des_a - self.av) / 0.6, c.attacker_accel) * live_a
        dp0, ap0 = self.dp, self.ap
        self.dv = clip_hv(self.dv + acc_d * dt, vmax_d, c.defender_climb) * live_d
        self.av = clip_hv(self.av + acc_a * dt, vmax_a, c.attacker_climb) * live_a
        self.dp = self._inside(self.dp + self.dv * dt, self.dv, self.lo_d)
        self.ap = self._inside(self.ap + self.av * dt, self.av, self.lo_a)
        self.stat["effort_d"] += norm(acc_d).sum(1) * dt
        self.stat["effort_a"] += norm(acc_a).sum(1) * dt
        self.stat["dist_d"] += norm(self.dp - dp0).sum(1)
        self.stat["dist_a"] += norm(self.ap - ap0).sum(1)
        flying = self.a_alive & act[:, None]
        self.stat["air_a"] += flying.sum(1) * dt
        self.stat["low_a"] += (flying & (self.ap[..., 2] < c.low_altitude)).sum(1) * dt
        self.stat["alt_a"] += (self.ap[..., 2] * flying).sum(1) * dt
        self.a_min_y = np.where(self.a_alive, np.minimum(self.a_min_y, self.ap[..., 1]), self.a_min_y)
        self.t = t_end
        d0 = self.dist
        self._pair()
        self._engagements(d0, dp0, ap0, act, dt, sub)

    def _inside(self, p: np.ndarray, v: np.ndarray, lo: np.ndarray) -> np.ndarray:
        """Keeps positions inside the airspace box and stops the velocity component that hit its wall."""
        v[(p < lo) | (p > self.hi)] = 0.0
        return np.minimum(np.maximum(p, lo), self.hi)  # np.clip with per-axis bounds is ~25x slower

    def _engagements(self, d0: np.ndarray, dp0: np.ndarray, ap0: np.ndarray, act: np.ndarray, dt: float,
                     sub: int) -> None:
        c, B, D, A = self.cfg, self.B, self.D, self.A
        # closest approach during the step, so fast closures cannot tunnel through the capture radius. Two
        # drones can only pass that close if they started or ended the step within the capture radius plus
        # the furthest both can move in one step, so only those few pairs are checked
        reach = c.capture_radius + dt * (np.hypot(c.defender_speed, c.defender_climb)
                                         + np.hypot(c.attacker_speed, c.attacker_climb))
        near = ((np.minimum(d0, self.dist) <= reach) & self.d_alive[:, :, None] & self.a_alive[:, None, :]
                & act[:, None, None])
        b, d, a = np.nonzero(near)
        close = np.full((B, D, A), np.inf)
        frac = np.zeros((B, D, A))
        if b.size:
            r0 = ap0[b, a] - dp0[b, d]
            dr = (self.ap[b, a] - self.dp[b, d]) - r0
            s = np.clip(-dot(r0, dr) / np.maximum(dot(dr, dr), 1e-9), 0, 1)
            close[b, d, a] = norm(r0 + s[:, None] * dr)
            frac[b, d, a] = s
        cand = close <= c.capture_radius
        if cand.any():
            # one engagement per interceptor per step: the closest attacker it passes
            pick = (np.arange(A) == close.argmin(2)[..., None]) & cand
            roll = self.noise.uniform(self.d_alive.shape, tag(SITE_HITS, self.steps, sub)) < c.p_kill
            hit = pick & roll[..., None]
            miss = pick & ~roll[..., None]
            killed = hit.any(1)
            self.stat["misses"] += miss.sum((1, 2))
            self.a_killed |= killed
            self.a_alive &= ~killed
            self.a_end = np.where(killed, self.t, self.a_end)
            fired = hit.any(2)
            self.ammo -= fired
            spent = fired & (self.ammo <= 0)
            self.d_alive &= ~spent
            self.stat["spent"] += spent.sum(1)
            self.dv = self.dv * self.d_alive[..., None]
            if spent.any():
                self.station = np.where(spent.any(1)[:, None, None], self._stations(self.d_alive), self.station)
                self._show_posts()
            if self.record:
                for i, j in zip(*np.nonzero(hit[0])):
                    p = ap0[0, j] + frac[0, i, j] * (self.ap[0, j] - ap0[0, j])
                    self.events.append({"t": round(self.t, 2), "type": "kill", "d": int(i), "a": int(j),
                                        "x": round(float(p[0]), 1), "y": round(float(p[1]), 1),
                                        "z": round(float(p[2]), 1)})
                for i, j in zip(*np.nonzero(miss[0])):
                    self.events.append({"t": round(self.t, 2), "type": "miss", "d": int(i), "a": int(j)})
        crossed = self.a_alive & (self.ap[..., 1] <= 0) & act[:, None]
        if crossed.any():
            self.a_breached |= crossed
            self.a_alive &= ~crossed
            self.a_end = np.where(crossed, self.t, self.a_end)
            if self.record:
                for a in np.nonzero(crossed[0])[0]:
                    self.events.append({"t": round(self.t, 2), "type": "breach", "a": int(a),
                                        "x": round(float(self.ap[0, a, 0]), 1), "y": 0.0,
                                        "z": round(float(self.ap[0, a, 2]), 1)})

    def _frame(self) -> None:
        a_state = np.where(self.a_alive[0], "1", np.where(self.a_killed[0], "k", np.where(self.a_breached[0], "b", "0")))
        self.frames.append({"d": np.rint(self.dp[0]).astype(int).ravel().tolist(),
                            "a": np.rint(self.ap[0]).astype(int).ravel().tolist(),
                            "ds": "".join(np.where(self.d_alive[0], "1", "s")), "as": "".join(a_state),
                            "tg": self.target[0].tolist()})

    # ------------------------------------------------------------------ results
    def summary(self, def_macs=0.0, att_macs=0.0) -> dict[str, np.ndarray]:
        """Per-episode results; rates are fractions of the attacking swarm. ``*_macs``: brain cost per
        drone per decision (one number, or one per episode), for the CPU load on each side's device."""
        c, A, D = self.cfg, self.A, self.D
        kills = self.a_killed.sum(1).astype(float)
        leaks = self.a_breached.sum(1).astype(float)
        held = A - kills - leaks
        t_end = self.stat["t_end"]
        timed = self.a_killed & np.isfinite(self.a_seen)
        react = np.where(timed, self.a_end - np.where(timed, self.a_seen, 0.0), 0.0)
        n_timed = timed.sum(1)
        kill_time = np.where(n_timed > 0, react.sum(1) / np.maximum(n_timed, 1), np.nan)
        locks = self.stat["locks"]
        advance = np.clip(1.0 - np.maximum(self.a_min_y, 0.0) / self.a_y0, 0.0, 1.0)
        air = np.maximum(self.stat["air_a"], 1e-9)
        return {
            "intercepted": kills / A, "leaked": leaks / A, "held_off": held / A,
            "advance": advance.mean(1),
            "overkill": np.where(locks > 0, self.stat["lock_pairs"] / np.maximum(locks, 1.0), 1.0),
            "interceptors_spent": self.stat["spent"] / D, "misses": self.stat["misses"],
            "kill_time": kill_time, "duration": t_end,
            "low_share": self.stat["low_a"] / air, "alt_att": self.stat["alt_a"] / air,
            "cpu_def": def_macs * self.stat["dec_d"] / (D * t_end) / device_macs_per_s(c.defender_device),
            "cpu_att": att_macs * self.stat["dec_a"] / (A * t_end) / device_macs_per_s(c.attacker_device),
            "radio_def_kbps": self.stat["bytes_d"] * 8 / t_end / 1e3 / D,
            "radio_att_kbps": self.stat["bytes_a"] * 8 / t_end / 1e3 / A,
            "effort_def": self.stat["effort_d"] / (D * t_end), "effort_att": self.stat["effort_a"] / (A * t_end),
            "flown_def_km": self.stat["dist_d"] / D / 1e3, "flown_att_km": self.stat["dist_a"] / A / 1e3,
            # training scores: defenders want kills and no leaks; attackers want to get through, and get a
            # little for ground gained so that hanging back out of reach is never the safe answer
            "def_fitness": (kills + 0.25 * held - leaks) / A,
            "att_fitness": leaks / A + 0.1 * advance.mean(1),
        }

    def replay(self) -> dict:
        c = self.cfg
        return {"kind": "clash", "dt": c.dt, "frames": self.frames, "events": self.events,
                "station": np.round(self.station0, 1).tolist(), "restations": self.restations,
                "width": c.width, "depth": c.depth,
                "ceiling": c.ceiling, "low_altitude": c.low_altitude, "sensor_range": c.sensor_range,
                "comm_range": c.comm_range, "capture_radius": c.capture_radius,
                "defenders": self.D, "attackers": self.A}
