"""Shared pieces of the arena: vector helpers, the mesh radio, intercept geometry, compute timing.

Vectors are 3-D - x across the front (east), y forward (north), z altitude - unless a helper says
"horizontal"; most helpers also accept 2-D vectors.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace
from functools import lru_cache

import numpy as np

from swarm.devices import get_profile

K = 4           # interceptors each attacker tracks (nearest first)
KD = 8          # attackers each interceptor's tracker shows it (nearest first): enough options that a dense
                # crowd does not hide the free attackers behind the few every interceptor sees first
M = 3           # friends each drone keeps an eye on
MSG = 2         # learned message channels a brain may radio to its friends
HEADER = 24     # bytes of framing per radio message (addressing, CRC)
TRACK_BYTES = 10  # one radioed sighting of an enemy: track id, 3-D position and velocity, compressed
T_NORM = 60.0   # s: time scale used to normalise observations

# Everything here runs hundreds of times per simulated step on small arrays, so the helpers avoid numpy
# calls with high fixed overhead (einsum, take_along_axis) in favour of plain arithmetic and indexing.


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot product along the last axis (2-D or 3-D vectors)."""
    out = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    if a.shape[-1] > 2:
        out = out + a[..., 2] * b[..., 2]
    return out


def norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(dot(v, v))  # np.hypot / linalg.norm are several times slower on emulated x64 numpy


def hnorm(v: np.ndarray) -> np.ndarray:
    """Horizontal length (x, y) of a 3-D vector."""
    x, y = v[..., 0], v[..., 1]
    return np.sqrt(x * x + y * y)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(norm(v)[..., None], 1e-9)


def clip_norm(v: np.ndarray, limit) -> np.ndarray:
    n = norm(v)[..., None]
    return v * np.minimum(1.0, np.asarray(limit)[..., None] / np.maximum(n, 1e-9))


def clip_hv(v: np.ndarray, h_limit, v_limit: float) -> np.ndarray:
    """A multirotor's velocity limits: horizontal speed and climb / descent rate are limited separately.
    ``h_limit`` may be one number or one per drone."""
    k = np.minimum(1.0, np.asarray(h_limit) / np.maximum(hnorm(v), 1e-9))
    return np.stack([v[..., 0] * k, v[..., 1] * k, np.clip(v[..., 2], -v_limit, v_limit)], -1)


def toward(d: np.ndarray, h_limit, v_limit: float) -> np.ndarray:
    """Velocity along ``d`` (straight at a point) at the fastest speed both limits allow."""
    s = np.minimum(np.asarray(h_limit) / np.maximum(hnorm(d), 1e-9), v_limit / np.maximum(np.abs(d[..., 2]), 1e-9))
    return d * s[..., None]


@lru_cache(maxsize=128)
def _grid(b: int, x: int) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(b)[:, None, None], np.arange(x)[None, :, None]


def gather(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``arr`` [B, X, Y, ...] taken along axis 2 at ``idx`` [B, X, K] -> [B, X, K, ...]."""
    bi, xi = _grid(*idx.shape[:2])
    return arr[bi, xi, idx]


def nearest(dist: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the ``k`` smallest entries of the last axis of ``dist`` [B, X, Y], and which of them are
    real (finite). Fewer than ``k`` candidates are padded with index 0 and ``valid = False``."""
    n = dist.shape[-1]
    order = np.argsort(dist, axis=-1)[..., :k]
    if n < k:
        pad = np.zeros(order.shape[:-1] + (k - n,), dtype=order.dtype)
        order = np.concatenate([order, pad], -1)
    valid = np.isfinite(gather(dist, order))
    if n < k:
        valid[..., n:] = False
    return order, valid


def sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared distances between every point of ``a`` [B, X, C] and every point of ``b`` [B, Y, C] -> [B, X, Y],
    one coordinate at a time: cheaper than a [B, X, Y, C] difference array or the |a|^2 + |b|^2 - 2ab matmul."""
    d2 = a[:, :, None, 0] - b[:, None, :, 0]
    d2 *= d2
    for k in range(1, a.shape[-1]):
        d = a[:, :, None, k] - b[:, None, :, k]
        d *= d
        d2 += d
    return d2


def links(pos: np.ndarray, alive: np.ndarray, radio_range: float, loss: float, noise: "Noise",
          tag: int) -> np.ndarray:
    """Who hears whom this step: ``[B, N, N]`` with [b, i, j] = i received j's message.

    One hop, range-limited, each link independently lossy; dead drones neither talk nor listen. ``tag``
    names this draw for keyed noise (see :class:`Noise`).
    """
    hear = (sq_dists(pos, pos) <= radio_range * radio_range) & alive[:, :, None] & alive[:, None, :]
    hear &= ~np.eye(pos.shape[1], dtype=bool)[None]
    if loss > 0:
        # a 16-bit random number per link (loss resolved to 1/65536): several times cheaper than floats
        hear &= noise.bits16(hear.shape, tag) >= int(round(loss * 65536))
    return hear


# ---------------------------------------------------------------------------------------------- randomness
SITE_START, SITE_ARRIVALS, SITE_TARGETS, SITE_RADIO, SITE_HITS = 1, 2, 3, 4, 5  # what a keyed draw is for
_M1, _M2, _M3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S48, _S11 = (np.uint64(s) for s in (30, 27, 31, 48, 11))


def _mix(x: np.ndarray) -> np.ndarray:
    """The splitmix64 finaliser on every element: a well-mixed 64-bit hash (uint64 arithmetic wraps around)."""
    x = (x ^ (x >> _S30)) * _M2
    x = (x ^ (x >> _S27)) * _M3
    return x ^ (x >> _S31)


@lru_cache(maxsize=64)
def _cells(tail: tuple[int, ...]) -> np.ndarray:
    return (np.arange(math.prod(tail), dtype=np.uint64) * _M1 + _M2).reshape(tail)


def tag(site: int, *counters: int) -> int:
    """A draw's name for keyed noise: what it is for (``SITE_*``) and when (step, physics sub-step, ...)."""
    t = site
    for c in counters:
        t = t * 1_000_003 + int(c)
    return t & 0x7FFF_FFFF_FFFF_FFFF


def derive(*parts: int) -> int:
    """A 63-bit key from integer parts, e.g. (training seed, generation, scenario): stable on every machine."""
    s = np.random.SeedSequence([int(p) for p in parts]).generate_state(2, np.uint32)
    return int(s[0]) | (int(s[1]) & 0x7FFF_FFFF) << 32


class Noise:
    """The random numbers of a batch of episodes.

    Without keys they come from one generator, in batch order: a test is reproducible from its seed.
    With one key per episode (training) each draw is a hash of (the episode's key, the draw's tag, which
    drone or link): an episode draws the same numbers whatever else shares its batch, so a generation can
    be cut into pieces that run on different CPU cores; and episodes that share a key share their numbers
    - common random numbers: brain variants on the same situation meet the same radio losses and hit
    rolls, so their scores differ by what they did, not by luck.
    """

    def __init__(self, rng: np.random.Generator, keys: np.ndarray | None = None):
        self.rng = rng
        self.keys = None if keys is None else np.asarray(keys, dtype=np.uint64)

    def _hash(self, tail: tuple[int, ...], t: int) -> np.ndarray:
        base = _mix(self.keys * _M3 + np.uint64(t))
        return _mix(base.reshape((-1,) + (1,) * len(tail)) ^ _cells(tail))

    def bits16(self, shape: tuple[int, ...], t: int) -> np.ndarray:
        if self.keys is None:
            return np.frombuffer(self.rng.bytes(2 * math.prod(shape)), np.uint16).reshape(shape)
        return (self._hash(tuple(shape[1:]), t) >> _S48).astype(np.uint16)

    def uniform(self, shape: tuple[int, ...], t: int) -> np.ndarray:
        """Uniform in [0, 1)."""
        if self.keys is None:
            return self.rng.random(shape)
        return (self._hash(tuple(shape[1:]), t) >> _S11) * (2.0 ** -53)

    def normal(self, loc: float, scale: float, shape: tuple[int, ...], t: int) -> np.ndarray:
        if self.keys is None:
            return self.rng.normal(loc, scale, shape)
        return np.stack([np.random.default_rng([int(k), t]).normal(loc, scale, shape[1:]) for k in self.keys])

    def scenarios(self, n: int, t: int, draw) -> list[np.ndarray]:
        """Per-episode starting situations from ``draw(generator, rows) -> [arrays with that many rows]``: one
        call on the batch's generator for all ``n`` episodes, or one generator per distinct key (episodes
        that share a key share their situation)."""
        if self.keys is None:
            return draw(self.rng, n)
        uniq, inv = np.unique(self.keys, return_inverse=True)
        parts = [draw(np.random.default_rng([int(k), t]), 1) for k in uniq]
        return [np.concatenate([p[i] for p in parts])[inv] for i in range(len(parts[0]))]


def heard_mean(hear: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of what each drone heard (``values`` [B, N, C]) and how many it heard from. To average
    several things, stack them into one ``values``: the bool -> float cast costs more than the product."""
    h = hear.astype(float)
    n = h.sum(2)
    return np.matmul(h, values) / np.maximum(n, 1.0)[..., None], n


def shared(seen: np.ndarray, hear: np.ndarray) -> np.ndarray:
    """Which enemies each drone knows about, [B, N, E]: those it sees itself plus those any friend it heard
    this step sees - sightings are radioed over one hop, so the whole mesh can react to a concentrated attack."""
    relayed = np.matmul(hear.astype(np.float32), seen.astype(np.float32)) > 0.5
    return seen | relayed


def relay_tracks(seen: np.ndarray, hear: np.ndarray, age: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Track tables passed on through a mesh, one hop per decision.

    ``seen`` [B, N, E]: what each drone sees now; ``hear`` [B, N, N]: who hears whom; ``age`` [B, N, E]: each
    drone's table after its last decision - for every enemy, how many decisions old its freshest report is
    (``inf``: never heard of it). Every drone radios its own sightings (age 0) and the reports in its table,
    one decision older; a drone keeps the freshest report it has or hears. Returns ``(broadcast, new age)``:
    a report of age ``a`` has travelled ``a + 1`` hops, so "known within ``h`` hops" is ``new age < h``.
    """
    broadcast = np.where(seen, 0.0, age + 1.0)
    via = np.where(hear[..., None], broadcast[:, None, :, :], np.inf).min(2)
    return broadcast, np.minimum(broadcast, via)


def intercept_time(rel: np.ndarray, target_vel: np.ndarray, speed: float) -> np.ndarray:
    """Time for a pursuer at full ``speed`` to meet a target at ``rel`` moving at ``target_vel``.

    Solves |rel + v t| = speed t; ``inf`` when the target is as fast or faster and cannot be caught.
    """
    a = dot(target_vel, target_vel) - speed * speed
    b = 2.0 * dot(rel, target_vel)
    c = dot(rel, rel)
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = (-b - np.sqrt(disc)) / (2.0 * a)
    return np.where((a < -1e-6) & (tau >= 0), tau, np.inf)


def intercept_eta(rel: np.ndarray, target_vel: np.ndarray, speed: float, climb: float) -> np.ndarray:
    """Time to intercept for a pursuer with horizontal top ``speed`` and climb rate ``climb``: the
    horizontal intercept, stretched when the height gap at that moment takes longer to close."""
    th = intercept_time(rel[..., :2], target_vel[..., :2], speed)
    ok = np.isfinite(th)
    t = np.where(ok, th, 0.0)
    dz = np.abs(rel[..., 2] + target_vel[..., 2] * t)
    return np.where(ok, np.maximum(th, dz / climb), np.inf)


UNBATCHED = ("cfg", "t", "index")
_BATCHED: dict[type, tuple[str, ...]] = {}


def take_batch(obs, idx):
    """The same observation restricted to episodes ``idx`` (every field except the shared ones above)."""
    names = _BATCHED.get(type(obs))
    if names is None:
        names = _BATCHED[type(obs)] = tuple(f.name for f in fields(obs) if f.name not in UNBATCHED)
    return replace(obs, **{n: getattr(obs, n)[idx] for n in names})


def device_macs_per_s(device: str) -> float:
    p = get_profile(device)
    return p.cpu_mhz * 1e6 * p.macs_per_cycle


def decide_every(macs: float, device: str, dt: float) -> int:
    """Steps between decisions: a brain that needs longer than one step on its device decides less often."""
    return max(1, math.ceil(macs / device_macs_per_s(device) / dt - 1e-9))
