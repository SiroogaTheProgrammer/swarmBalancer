"""Brains for the arena tests: what every drone of one side decides, for all episodes of a batch at once.

``act(obs)`` gets the batched observation of its role and returns a dict of actions, arrays [B, N, ...]:

    defender  {"target": int}                     slot 0..KD-1 of the observation to go for, -1 = hold station
    attacker  {"heading": float, "speed": float}  radians off straight-at-the-line (+ = towards +x); 0..1
    recon     {"direction": [.., 2], "speed": float, "camera": bool}   horizontal direction; 0..1; on/off
    attackers and recon drones may add "altitude": metres to climb or descend to (otherwise they keep
    their height); defenders may add "station": [B, D, 3] where each waits when it holds - its guard post,
    anywhere from the line out to ``guard_depth`` (otherwise its station in the line, ``obs.station``); any
    role may add "msg": [B, N, MSG] in [-1, 1] - radio channels its friends hear on the next step

Three kinds of brain:
    built-in  hand-written tactics: the baselines every test compares a brain against
    neural    the "mesh" brain - one small Dense-ReLU-Dense network that every drone runs on its own
              observation (shared weights), trained by evolution (``swarm.arena.train``); stored as ``.npz``,
              ``to_swm`` compiles it for the C++ engine
    python    a ``swarm.sdk`` package of type "policy": a class with ``act(obs)``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .clash import AttackerObs, DefenderObs
from .core import K, KD, MSG, norm, unit
from .recon import DIRS as RECON_DIRS
from .recon import ReconObs

ROLES = ("recon", "defender", "attacker")
OBS = {"defender": DefenderObs, "attacker": AttackerObs, "recon": ReconObs}
OUTPUTS = {"defender": 1 + MSG + 3, "attacker": 3 + MSG, "recon": 8 + MSG}
ROWS = {"defender": KD + 1, "attacker": 1, "recon": 1}  # network evaluations per drone per decision
REQUIRED = {"defender": ("target",), "attacker": ("heading", "speed"), "recon": ("direction", "speed", "camera")}
OPTIONAL = {"defender": ("msg", "station"), "attacker": ("altitude", "msg"), "recon": ("altitude", "msg")}
VECTORS = {"direction": 2, "msg": MSG, "station": 3}   # actions with a last axis of this length
KEEP = ("altitude", "station")  # optional actions where NaN means "no new command" (see MixedBrain)
NPZ_FORMAT = "swarm.mesh/2"
# mesh defenders saved before guard posts: no "role" input, no post outputs. They load with zero weights
# there - the same decisions as before, from the station in the line
OLD_DEFENDER = (DefenderObs.FEATURES - 1, 1 + MSG)


class BrainError(ValueError):
    """A brain cannot be found or loaded, or returned malformed actions."""


class Brain:
    """Base class: subclasses set the class attributes and implement :meth:`act`."""

    role = ""
    name = ""
    title = ""
    summary = ""
    kind = "builtin"
    comms = False   # coordinates over the mesh radio (beyond the position broadcast every drone makes)
    macs = 200      # compute per drone per decision; a scripted rule is a few hundred operations

    @property
    def id(self) -> str:
        return self.name

    def reset(self, batch: int, seed: int = 0) -> None:
        """Called before each run with the number of episodes; per-episode memory belongs here."""

    def act(self, obs) -> dict:
        raise NotImplementedError

    def info(self) -> dict:
        return {"id": self.id, "name": self.name, "role": self.role, "kind": self.kind,
                "title": self.title or self.name, "summary": self.summary, "comms": self.comms,
                "macs": int(self.macs)}


def _choose(cost: np.ndarray) -> np.ndarray:
    """Cheapest finite slot per drone, -1 where every slot is infinite."""
    return np.where(np.isfinite(cost.min(-1)), cost.argmin(-1), -1)


def _preference(index: np.ndarray, slot_id: np.ndarray, spread: float) -> np.ndarray:
    """A fixed pseudo-random preference in [0, spread) for every (interceptor, attacker) pair, [B, D, KD].

    Interceptors that decide at the same moment with the same picture would otherwise all bid for the same
    "best" attacker; only one keeps it, the rest move on together to the next one, and a dense crowd is
    worked through one attacker per round. Personal preferences split their first bids instead - the
    randomised-priority trick of distributed auctions - and being fixed, they do not make a drone flip-flop.
    """
    x = np.sin(index[None, :, None] * 12.9898 + slot_id * 78.233) * 43758.5453
    return (x - np.floor(x)) * spread


def _away(rel: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
    """Sum of pushes away from neighbours closer than ``radius`` (stronger when closer)."""
    d = norm(rel)
    w = np.where(mask & (d < radius), 1.0 - d / radius, 0.0)
    return -(unit(rel) * w[..., None]).sum(-2)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _pick(arr: np.ndarray, k: np.ndarray) -> np.ndarray:
    """``arr`` [B, N, S, ...] at slot ``k`` [B, N] -> [B, N, ...]."""
    B, N = k.shape
    return arr[np.arange(B)[:, None], np.arange(N)[None, :], k]


def _h(v: np.ndarray) -> np.ndarray:
    """The horizontal (x, y) part of 3-D vectors."""
    return v[..., :2]


def footprint_at(cfg, altitude):
    """Radius of the ground a recon camera sees from ``altitude``."""
    return np.asarray(altitude) * math.tan(math.radians(cfg.camera_fov / 2))


# ---------------------------------------------------------------------------------------------- defenders
class Greedy(Brain):
    role, name, title = "defender", "greedy", "Greedy"
    summary = ("Every interceptor goes for the attacker it can reach first. No coordination: neighbours "
               "pile onto the same one while others slip through.")

    def act(self, o: DefenderObs) -> dict:
        return {"target": _choose(np.where(o.slot_mask, o.slot_eta, np.inf))}


class Auction(Brain):
    role, name, title = "defender", "auction", "Smart wall (auction)"
    comms = True
    summary = ("Each interceptor radios which attacker it is going for and how soon it gets there, drops one a "
               "friend reaches sooner and takes the most urgent free one: a leaderless auction - one interceptor "
               "per attacker, the rest stay on the line as reserve.")
    SPREAD = 10.0  # s: the largest personal preference added to an attacker's cost (see _preference)

    def act(self, o: DefenderObs) -> dict:
        taken = o.claim_eta < o.claim_reference()
        # most urgent first; stick to my pick; a fixed personal preference per attacker keeps interceptors
        # that see the same crowd from all bidding for the same one, round after round
        cost = (o.slot_eta + 0.25 * o.slot_wall_time - 1.0 * o.slot_mine
                + _preference(o.index, o.slot_id, self.SPREAD))
        return {"target": _choose(np.where(o.slot_mask & ~taken, cost, np.inf))}


class Pickets(Auction):
    role, name, title = "defender", "pickets", "Pickets + auction"
    summary = ("Every third interceptor waits far out in front as a picket, so even a terrain-hugging attack is "
               "spotted a kilometre before it reaches the line and the radio warns the rest in time. Targets are "
               "shared out by the same leaderless auction as the smart wall.")
    FORWARD = 0.35  # pickets wait this share of the start distance in front of the line

    def act(self, o: DefenderObs) -> dict:
        rank = np.cumsum(o.alive, 1) - 1   # order along the line among those still in the fight
        post = o.station.copy()
        post[..., 1] = np.where(o.alive & (rank % 3 == 1), self.FORWARD * o.cfg.depth, post[..., 1])
        return {**super().act(o), "station": post}


class Zone(Brain):
    role, name, title = "defender", "zone", "Zones"
    summary = ("The line is cut into one sector per interceptor; each takes the most urgent attacker heading "
               "for its own sector. Needs no radio, but a crowd aimed at one sector overwhelms its guard.")

    def act(self, o: DefenderObs) -> dict:
        W = o.cfg.width
        # one sector per interceptor still in the fight, centred on its (re-spaced) station
        half = W / (2 * np.maximum(o.alive.sum(1), 1))[:, None, None] * 1.2
        ax = o.pos[..., None, 0] + o.slot_rel[..., 0]
        cross = np.clip(ax + o.slot_vel[..., 0] * np.minimum(o.slot_wall_time, 60.0), -W / 2, W / 2)
        mine = np.abs(cross - o.station[..., None, 0]) <= half
        return {"target": _choose(np.where(o.slot_mask & mine, o.slot_wall_time + 0.1 * o.slot_eta, np.inf))}


# ---------------------------------------------------------------------------------------------- attackers
def _jink(o: AttackerObs, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Break hard away from the closest interceptor closing in within ``radius`` - sideways and in height
    (dive if it comes from above, climb if from below). Returns (heading, altitude, threatened)."""
    threat = o.slot_mask & (o.slot_closing > 0.5) & (o.slot_dist < radius)
    d = np.where(threat, o.slot_dist, np.inf)
    r = _pick(o.slot_rel, d.argmin(-1))           # [B, A, 3] the closest threat, relative to me
    has = np.isfinite(d.min(-1))
    heading = np.where(has, np.where(r[..., 0] > 0, -1.25, 1.25), 0.0)
    altitude = o.pos[..., 2] + np.where(has, np.where(r[..., 2] > 0, -60.0, 60.0), 0.0)
    return heading, altitude, has


class Rush(Brain):
    role, name, title = "attacker", "rush", "Rush"
    summary = "Everyone flies straight at the line at full speed, at the height it arrived at. No radio, no tricks."

    def act(self, o: AttackerObs) -> dict:
        return {"heading": np.zeros(o.alive.shape), "speed": np.ones(o.alive.shape)}


class Spread(Brain):
    role, name, title = "attacker", "spread", "Fan out"
    summary = ("The swarm fans out across the front and through the whole height band while it closes in, so "
               "the wall must cover every sector and every altitude at once.")

    def act(self, o: AttackerObs) -> dict:
        c = o.cfg
        far = o.pos[..., 1] > 0.35 * c.depth
        fan = np.clip(-o.friends_rel[..., 0] / 150.0, -1, 1) * 0.8
        lo, hi = min(2.0 * c.low_altitude + 20.0, 0.5 * c.ceiling), 0.9 * c.ceiling
        layer = lo + (o.role() + 1.0) / 2.0 * (hi - lo)
        return {"heading": np.where(far & (o.friends_n > 0), fan, 0.0), "speed": np.ones(o.alive.shape),
                "altitude": np.broadcast_to(layer, o.alive.shape)}


class Evade(Brain):
    role, name, title = "attacker", "evade", "Jink"
    summary = ("Straight at the line at full speed, but a drone that sees an interceptor about to hit it "
               "breaks hard to the side and dives or climbs away at the last moment.")

    def act(self, o: AttackerObs) -> dict:
        heading, altitude, _ = _jink(o, 120.0)
        return {"heading": heading, "speed": np.ones(o.alive.shape), "altitude": altitude}


class Sneak(Brain):
    role, name, title = "attacker", "sneak", "Terrain hug"
    summary = ("Everyone dives below the terrain mask and comes in low: hidden from all but close sensors, but "
               "a quarter slower. A drone that sees an interceptor about to hit breaks to the side.")

    def act(self, o: AttackerObs) -> dict:
        heading, _, _ = _jink(o, 120.0)
        return {"heading": heading, "speed": np.ones(o.alive.shape),
                "altitude": np.full(o.alive.shape, 0.6 * o.cfg.low_altitude)}


class Decoy(Brain):
    role, name, title = "attacker", "decoy", "Decoys + strike"
    comms = True
    summary = ("A third of the swarm flies high and ahead as decoys and, once chased, drags the interceptors "
               "up and sideways. The strike group hugs the terrain, hidden from all but close sensors, and holds "
               "back until the radio says the interceptors have committed - then it sprints for the line. The "
               "'go' call is relayed drone to drone.")

    def act(self, o: AttackerObs) -> dict:
        c = o.cfg
        shape = o.alive.shape
        decoy = np.broadcast_to(o.role() < -1 / 3, shape)
        go = o.chased | (o.friends_chased > 0.05) | (o.msg[..., 0] > 0.05) | (o.t > 0.35 * c.duration)
        jink, _, _ = _jink(o, 120.0)
        # decoys flee sideways once chased (towards the far side of the corridor), strikers jink
        side = np.where(o.pos[..., 0] > 0, -1.0, 1.0)
        heading = np.where(decoy, np.where(o.chased, side * 1.3, 0.0), jink)
        speed = np.where(decoy | go, 1.0, 0.35)
        altitude = np.where(decoy, 0.85 * c.ceiling, 0.6 * c.low_altitude)
        msg = np.zeros(shape + (MSG,))
        msg[..., 0] = go
        return {"heading": heading, "speed": speed, "altitude": altitude, "msg": msg}


# ---------------------------------------------------------------------------------------------- recon
def _sweep_point(c, swath: float, t: float, lead: float = 15.0) -> np.ndarray:
    """Formation centre of a line-abreast boustrophedon sweep (north-south lanes ``swath`` wide), ``lead``
    metres ahead of where the route has got to by time ``t``. The swarm first flies from the base in the
    middle of the area to the start of the first lane."""
    S = c.area
    lanes = max(1, math.ceil(S / swath))
    xs = np.linspace(swath / 2, S - swath / 2, lanes) if lanes > 1 else np.array([S / 2])
    margin = min(0.8 * float(footprint_at(c, c.detail_altitude)), 0.25 * S)
    lo, hi = margin, S - margin
    start = np.array([xs[0], lo])
    s = 0.8 * c.speed * t + lead - float(np.hypot(*(start - S / 2)))
    if s <= 0:
        return start
    leg = hi - lo
    for k in range(lanes):
        if s <= leg:
            return np.array([xs[k], lo + s if k % 2 == 0 else hi - s])
        s -= leg
        if k + 1 < lanes:
            shift = xs[k + 1] - xs[k]
            if s <= shift:
                return np.array([xs[k] + s, hi if k % 2 == 0 else lo])
            s -= shift
    return np.array([xs[-1], hi if (lanes - 1) % 2 == 0 else lo])


class Lawnmower(Brain):
    role, name, title = "recon", "lawnmower", "Lawnmower"
    summary = ("A pre-planned sweep: the drones fly line abreast up and down the area at full-detail height, "
               "cameras always on. Thorough and always in radio contact, but slow to start (it first flies to a "
               "corner) and every drone flies every leg at full sensing load.")

    def act(self, o: ReconObs) -> dict:
        c, n = o.cfg, len(o.index)
        sp = 1.8 * float(footprint_at(c, c.detail_altitude))
        centre = _sweep_point(c, n * sp, o.t)
        goal = np.clip(np.stack([centre[0] + (o.index - (n - 1) / 2) * sp, np.full(n, centre[1])], -1), 0, c.area)
        d = goal[None] - _h(o.pos)
        return {"direction": d, "speed": np.clip(norm(d) / 30.0, 0.0, 1.0), "camera": np.ones(o.linked.shape, bool),
                "altitude": np.full(o.linked.shape, c.detail_altitude)}


class Flock(Brain):
    role, name, title = "recon", "flock", "Flock"
    summary = ("Classic boids at full-detail height: keep close to your neighbours, don't collide, follow the "
               "shared route. Never loses contact, but the cameras overlap, so it surveys slowly.")

    def act(self, o: ReconObs) -> dict:
        c, n = o.cfg, len(o.index)
        r = float(footprint_at(c, c.detail_altitude))
        to_route = unit(_sweep_point(c, n * r, o.t, lead=60.0)[None, None] - _h(o.pos))
        fm = o.friend_mask
        frel = _h(o.friend_rel)
        centre = (frel * fm[..., None]).sum(-2) / np.maximum(fm.sum(-1), 1)[..., None]
        d = to_route + 0.5 * centre / r + 1.5 * _away(frel, fm, 0.9 * r)
        return {"direction": d, "speed": np.full(o.linked.shape, 0.8), "camera": np.ones(o.linked.shape, bool),
                "altitude": np.full(o.linked.shape, c.detail_altitude)}


class Frontier(Brain):
    role, name, title = "recon", "frontier", "Frontier (high)"
    summary = ("Each drone flies high - a wide view at about 70% detail - to the nearest ground nobody has seen "
               "and keeps away from its neighbours. Sees everything fast, but misses the targets that need a close "
               "look, and the swarm scatters out of radio range, so reports wait on board.")

    def act(self, o: ReconObs) -> dict:
        c = o.cfg
        alt = 1.6 * c.detail_altitude
        d = unit(o.gap_rel) + 1.2 * _away(_h(o.friend_rel), o.friend_mask, 2.2 * float(footprint_at(c, alt)))
        return {"direction": d, "speed": np.ones(o.linked.shape), "camera": np.ones(o.linked.shape, bool),
                "altitude": np.full(o.linked.shape, alt)}


class RelaySweep(Brain):
    role, name, title = "recon", "relay_sweep", "Relay sweep (load-aware)"
    comms = True
    summary = ("Frontier search at full-detail height on a radio leash: no drone strays past 70% of radio range "
               "from its nearest friend, anyone cut off flies back to the lead, the lead trails the group, and "
               "cameras run only while there is ground in view they would still improve - about the same "
               "coverage for far less sensing load.")

    def act(self, o: ReconObs) -> dict:
        c = o.cfg
        R = c.comm_range
        fm = o.friend_mask
        frel = _h(o.friend_rel)
        fd = np.where(fm, norm(o.friend_rel), np.inf)            # the radio cares about 3-D distance
        near_rel = _pick(frel, fd.argmin(-1))
        d_near = fd.min(-1)
        leash = unit(near_rel) * np.clip((d_near - 0.55 * R) / (0.15 * R), 0, 2)[..., None]
        leash = np.where(np.isfinite(d_near)[..., None], leash, 0.0)
        explore = unit(o.gap_rel) + 1.2 * _away(frel, fm, 2.0 * float(footprint_at(c, c.detail_altitude))) + leash
        centre = (frel * fm[..., None]).sum(-2) / np.maximum(fm.sum(-1), 1)[..., None]
        is_lead = (o.index == 0)[None, :, None]
        d = np.where(is_lead, 0.6 * unit(o.gap_rel) + centre / (0.3 * R), explore)
        d = np.where(o.linked[..., None] | is_lead, d, unit(_h(o.lead_rel)))
        speed = np.where(is_lead[..., 0], 0.75, 1.0) * np.ones(o.linked.shape)
        return {"direction": d, "speed": speed, "camera": o.uncovered_here > 0.0,
                "altitude": np.full(o.linked.shape, c.detail_altitude)}


BUILTINS: dict[str, type[Brain]] = {cls.name: cls for cls in (
    Greedy, Auction, Pickets, Zone, Rush, Spread, Evade, Sneak, Decoy, Lawnmower, Flock, Frontier, RelaySweep)}
DEFAULTS = {"defender": "auction", "attacker": "decoy", "recon": "relay_sweep"}


# ---------------------------------------------------------------------------------------------- neural
def decode_defender(out: np.ndarray, o: DefenderObs) -> dict:
    """``out`` [B, D, KD + 1, 1 + MSG + 3]: a score per option row (each tracked attacker, then hold); the hold
    row also carries the radio message and the guard post, as a shift from the station in the line."""
    logits = out[..., 0].copy()
    logits[..., :KD] = np.where(o.slot_mask & np.isfinite(o.slot_eta), logits[..., :KD], -np.inf)
    choice = logits.argmax(-1)  # all-zero scores pick slot 0: an untrained network starts out greedy
    hold = out[..., KD, :]
    act = {"target": np.where(choice == KD, -1, choice), "msg": np.tanh(hold[..., 1:1 + MSG])}
    c = o.cfg
    shift = np.tanh(hold[..., 1 + MSG:]) * np.array([c.width / 4, c.depth / 2, c.ceiling / 2])
    if shift.any():  # a network trained before guard posts has zero weights there and keeps the line exactly
        act["station"] = o.station + shift
    return act


def decode_attacker(out: np.ndarray, o: AttackerObs) -> dict:
    """An untrained network flies straight at the line, at the height it arrived at."""
    return {"heading": np.tanh(out[..., 0]) * (np.pi / 2), "speed": 0.2 + 0.8 * _sigmoid(out[..., 1] + 2.0),
            "altitude": o.pos[..., 2] + 150.0 * np.tanh(out[..., 2]), "msg": np.tanh(out[..., 3:])}


def decode_recon(out: np.ndarray, o: ReconObs) -> dict:
    """The network weighs five steering urges and sets its height (an untrained network is pure "frontier"
    - nearest unseen ground only - at full-detail height)."""
    c = o.cfg
    fm = o.friend_mask
    frel = _h(o.friend_rel)
    urges = np.stack([
        unit(o.gap_rel),                                               # nearest unseen ground
        unit(_h(o.lead_rel)),                                          # back towards the lead
        unit((frel * fm[..., None]).sum(-2)),                          # towards the friends I hear
        _away(frel, fm, 2.0 * float(footprint_at(c, c.detail_altitude))),  # away from friends too close
        (o.look[..., None] * RECON_DIRS).sum(-2) / 2.0,                # towards windows still missing detail
    ], -2)                                                             # [B, N, 5, 2]
    w = out[..., :5] + np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    # the camera threshold sits close to an untrained network's output so that exploration flips it and
    # training can find out when the camera is worth its load
    return {"direction": (w[..., None] * urges).sum(-2), "speed": _sigmoid(out[..., 5] + 2.0),
            "camera": out[..., 6] > -0.25, "altitude": c.detail_altitude * 2.0 ** (1.6 * np.tanh(out[..., 7])),
            "msg": np.tanh(out[..., 8:])}


DECODE = {"defender": decode_defender, "attacker": decode_attacker, "recon": decode_recon}

_SLOT_D = ["rel_x/sensor", "rel_y/sensor", "rel_z/sensor", "vel_x/v", "vel_y/v", "vel_z/climb", "dist/sensor",
           "eta/60s", "wall_dist/depth", "wall_time/60s", "claims/3", "claim_advantage", "mine", "visible"]
_SLOT_A = ["rel_x/sensor", "rel_y/sensor", "rel_z/sensor", "vel_x/2v", "vel_y/2v", "vel_z/(2 interceptor climb)",
           "dist/sensor", "closing", "visible"]
FEATURE_NAMES = {
    # one row per option (KD tracked attackers, then hold); the network scores each row
    "defender": [f"option.{n}" for n in _SLOT_D] + ["option.is_hold"] + [
        "vel_x/v", "vel_y/v", "vel_z/climb", "pos_y/depth", "pos_x/(width/2)", "altitude/ceiling", "has_target",
        "visible/KD", "free/KD", "friends_heard/(D-1)", "friends_rel_x/radio", "friends_rel_y/radio",
        "friends_rel_z/radio"] + [f"msg{i}" for i in range(MSG)] + [
        "station_dx/width", "station_dy/width", "station_dz/ceiling", "t/duration", "role"],
    "attacker": ["vel_x/v", "vel_y/v", "vel_z/climb", "pos_y/depth", "pos_x/(width/2)", "altitude/ceiling",
                 "below_terrain_mask"] + [f"threat{k}.{n}" for k in range(K) for n in _SLOT_A] + [
        "friends_heard/(A-1)", "friends_rel_x/radio", "friends_rel_y/radio", "friends_rel_z/radio",
        "friends_chased"] + [f"msg{i}" for i in range(MSG)] + ["t/duration", "chased", "role"],
    "recon": ["vel_x/v", "vel_y/v", "vel_z/climb", "pos_x*2/area-1", "pos_y*2/area-1", "altitude/max_altitude",
              "lead_rel_x/radio", "lead_rel_y/radio", "lead_rel_z/radio", "linked"] + [
        f"friend{k}.{n}" for k in range(3) for n in ("rel_x/radio", "rel_y/radio", "rel_z/radio", "heard")] + [
        "friends_heard/(N-1)"] + [f"missing_detail_{d}" for d in ("E", "NE", "N", "NW", "W", "SW", "S", "SE")] + [
        "gap_x/(area/4)", "gap_y/(area/4)", "unsurveyed_here", "detail"] + [f"msg{i}" for i in range(MSG)] + [
        "t/duration", "role", "is_lead"],
}
OUTPUT_NAMES = {
    "defender": ["score", *(f"msg{i}" for i in range(MSG)), "post_x", "post_y", "post_z"],
    "attacker": ["heading", "speed", "altitude_change", *(f"msg{i}" for i in range(MSG))],
    "recon": ["w_gap", "w_lead", "w_friends", "w_spread", "w_unsurveyed", "speed", "camera", "altitude",
              *(f"msg{i}" for i in range(MSG))],
}
DECODE_NOTES = {
    "defender": "run the network on each option row; skip rows of attackers not visible or not catchable; "
                "go for the highest score (the last row = hold station); radio tanh(msg) of the hold row; "
                "wait at the station in the line shifted by (width/4 * tanh(post_x), depth/2 * tanh(post_y), "
                "ceiling/2 * tanh(post_z)) of the hold row - the arena keeps it between the line and guard_depth",
    "attacker": "heading = tanh(heading) * pi/2 radians off straight-at-the-line; "
                "speed = 0.2 + 0.8 * sigmoid(speed + 2) of top speed; "
                "fly to altitude = current altitude + 150 m * tanh(altitude_change); radio tanh(msg)",
    "recon": "steer (horizontally) along (1 + w_gap) * unit(gap) + w_lead * unit(lead) + w_friends * "
             "unit(sum of friend offsets) + w_spread * push away from friends within 2 full-detail footprints + "
             "w_unsurveyed * sum(missing_detail_k * compass_k) / 2; speed = sigmoid(speed + 2) of top speed; "
             "camera on when camera > -0.25; fly at detail_altitude * 2 ** (1.6 * tanh(altitude)); radio tanh(msg)",
}


class NeuralBrain(Brain):
    """The mesh brain: features -> Dense(hidden) -> ReLU -> Dense(outputs), decoded per role (``DECODE``).

    Defenders run the network once per option row (``DefenderObs.features``) and take the best score;
    attackers and recon drones run it once on their feature vector.
    """

    kind = "neural"
    comms = True

    def __init__(self, role: str, hidden: int = 16, theta: np.ndarray | None = None, seed: int = 0,
                 name: str = "mesh", title: str = "", summary: str = "", meta: dict | None = None):
        if role not in ROLES:
            raise BrainError(f"role must be one of {ROLES}, got {role!r}")
        self.role, self.name = role, name
        self.title = title or f"Mesh network ({role})"
        self.summary = summary or ("A small neural network every drone runs on what it senses and hears from its "
                                   "friends; trained by evolution in the arena.")
        self.n_in, self.hidden, self.n_out = OBS[role].FEATURES, int(hidden), OUTPUTS[role]
        self.size = self.n_in * self.hidden + self.hidden + self.hidden * self.n_out + self.n_out
        self.macs = ROWS[role] * (self.n_in * self.hidden + self.hidden * self.n_out)
        self.meta = dict(meta or {})
        self.lib_id: str | None = None
        theta = self.initial(seed) if theta is None else np.asarray(theta, dtype=float).ravel()
        if theta.size != self.size:
            raise BrainError(f"{name}: {theta.size} weights, a {role} network with {hidden} hidden units needs {self.size}")
        self.theta = theta

    @property
    def id(self) -> str:
        return self.lib_id or self.name

    def initial(self, seed: int) -> np.ndarray:
        """Random hidden layer, zero output layer: an untrained brain starts from the role's neutral action."""
        rng = np.random.default_rng(seed)
        w1 = rng.standard_normal((self.n_in, self.hidden)) * math.sqrt(2.0 / self.n_in)
        return np.concatenate([w1.ravel(), np.zeros(self.hidden + self.hidden * self.n_out + self.n_out)])

    def unpack(self, p: np.ndarray):
        i, h, o = self.n_in, self.hidden, self.n_out
        s = p.shape[:-1]
        a, b, c = i * h, i * h + h, i * h + h + h * o
        return p[..., :a].reshape(s + (i, h)), p[..., a:b], p[..., b:c].reshape(s + (h, o)), p[..., c:]

    def forward(self, x: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """``x`` [B, ..., features] -> [B, ..., outputs]; ``params`` [B, size] gives each episode its own weights."""
        if params is None:
            w1, b1, w2, b2 = self.unpack(self.theta)
            return np.maximum(x @ w1 + b1, 0.0) @ w2 + b2
        w1, b1, w2, b2 = self.unpack(params)
        xb = x.reshape(x.shape[0], -1, x.shape[-1])
        h = np.maximum(np.matmul(xb, w1) + b1[:, None, :], 0.0)
        out = np.matmul(h, w2) + b2[:, None, :]
        return out.reshape(x.shape[:-1] + (self.n_out,))

    def act(self, obs, params: np.ndarray | None = None) -> dict:
        return DECODE[self.role](self.forward(obs.features(), params), obs)

    def with_theta(self, theta: np.ndarray, **meta) -> "NeuralBrain":
        b = NeuralBrain(self.role, self.hidden, theta, name=self.name, title=self.title, summary=self.summary,
                        meta={**self.meta, **meta})
        b.lib_id = self.lib_id
        return b

    def info(self) -> dict:
        return {**super().info(), "hidden": self.hidden, "params": self.size,
                "trained": {k: self.meta[k] for k in ("generations", "trained_against", "seconds") if k in self.meta}}

    def save(self, path: str | Path, **meta) -> Path:
        path = Path(path)
        m = {**self.meta, **meta, "format": NPZ_FORMAT, "role": self.role, "hidden": self.hidden, "n_in": self.n_in,
             "n_out": self.n_out, "name": self.name, "title": self.title, "summary": self.summary}
        with open(path, "wb") as f:
            np.savez(f, theta=self.theta.astype(np.float32), meta=np.array(json.dumps(m)))
        return path

    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> "NeuralBrain":
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                theta = z["theta"].astype(float)
        except (OSError, ValueError, KeyError) as exc:
            raise BrainError(f"{Path(path).name}: not a mesh brain .npz ({exc})") from None
        if not isinstance(meta, dict) or meta.get("format") != NPZ_FORMAT:
            raise BrainError(f"{Path(path).name}: not a {NPZ_FORMAT} file")
        role = meta.get("role")
        if role not in ROLES:
            raise BrainError(f"{Path(path).name}: unknown role {role!r}")
        if role == "defender" and (meta.get("n_in"), meta.get("n_out")) == OLD_DEFENDER:
            theta = _widen(theta, int(meta.get("hidden", 0)), OLD_DEFENDER, (OBS[role].FEATURES, OUTPUTS[role]),
                           Path(path).name)
            meta = {**meta, "n_in": OBS[role].FEATURES, "n_out": OUTPUTS[role]}
        if meta.get("n_in") != OBS[role].FEATURES or meta.get("n_out") != OUTPUTS[role]:
            raise BrainError(f"{Path(path).name}: built for a different {role} observation layout "
                             f"({meta.get('n_in')} inputs, this version has {OBS[role].FEATURES}); retrain it")
        return cls(role, int(meta.get("hidden", 0)), theta, name=name or str(meta.get("name") or "mesh"),
                   title=str(meta.get("title") or ""), summary=str(meta.get("summary") or ""), meta=meta)

    def to_swm(self, path: str | Path) -> Path:
        """The network as a ``.swm`` file for the C++ engine (Dense -> ReLU -> Dense, f32, no softmax).

        A defender runs it once per option row and takes the row with the highest first output."""
        from swarm.brain.layers import Dense, Flatten, ReLU
        from swarm.brain.model import Sequential

        w1, b1, w2, b2 = self.unpack(self.theta)
        d1, d2 = Dense(self.n_in, self.hidden), Dense(self.hidden, self.n_out)
        d1.W[...], d1.b[...], d2.W[...], d2.b[...] = w1, b1, w2, b2
        return Sequential([Flatten(), d1, ReLU(), d2], (self.n_in, 1, 1)).save(path, with_softmax=False)


def _widen(theta: np.ndarray, hidden: int, old: tuple[int, int], new: tuple[int, int], who: str) -> np.ndarray:
    """A network's weights with inputs and outputs appended (zero weights: the new inputs change nothing and
    the new outputs read 0)."""
    (i0, o0), (i1, o1), h = old, new, hidden
    if h <= 0 or theta.size != i0 * h + h + h * o0 + o0:
        raise BrainError(f"{who}: {theta.size} weights do not fit {i0} inputs, {h} hidden, {o0} outputs")
    w1 = np.zeros((i1, h))
    w1[:i0] = theta[:i0 * h].reshape(i0, h)
    b1 = theta[i0 * h:i0 * h + h]
    w2 = np.zeros((h, o1))
    w2[:, :o0] = theta[i0 * h + h:i0 * h + h + h * o0].reshape(h, o0)
    b2 = np.zeros(o1)
    b2[:o0] = theta[i0 * h + h + h * o0:]
    return np.concatenate([w1.ravel(), b1, w2.ravel(), b2])


# ---------------------------------------------------------------------------------------------- python packages
def check_actions(role: str, out, obs, who: str = "brain") -> dict:
    """Validates and normalises what a (user-written) brain returned; raises :class:`BrainError`."""
    if not isinstance(out, dict):
        raise BrainError(f"{who}: act() must return a dict with keys {REQUIRED[role]}, got {type(out).__name__}")
    missing = [k for k in REQUIRED[role] if k not in out]
    if missing:
        raise BrainError(f"{who}: act() result is missing {missing} (a {role} brain returns {REQUIRED[role]})")
    shape = obs.pos.shape[:2]
    res = {}
    for key, v in out.items():
        if key not in REQUIRED[role] + OPTIONAL[role]:
            raise BrainError(f"{who}: unknown action {key!r} for a {role} brain "
                             f"(it may return {REQUIRED[role] + OPTIONAL[role]})")
        a = np.asarray(v)
        if key == "direction" and a.ndim and a.shape[-1] == 3:
            a = a[..., :2]  # a 3-D direction: its horizontal part steers, "altitude" sets the height
        want = shape + ((VECTORS[key],) if key in VECTORS else ())
        try:
            a = np.broadcast_to(a, want)
        except ValueError:
            raise BrainError(f"{who}: {key!r} has shape {a.shape}, expected {want}") from None
        if key == "camera":
            res[key] = a.astype(bool)
            continue
        if not np.issubdtype(a.dtype, np.number) and a.dtype != bool:
            raise BrainError(f"{who}: {key!r} must be numbers")
        a = a.astype(float)
        if not np.isfinite(a).all():
            raise BrainError(f"{who}: {key!r} contains NaN or infinity")
        if key == "target":
            if (a != np.round(a)).any() or (a < -1).any() or (a >= KD).any():
                raise BrainError(f"{who}: 'target' must be whole numbers from -1 (hold) to {KD - 1}")
            a = a.astype(int)
        res[key] = a
    return res


class PythonBrain(Brain):
    """Wraps a policy object from a package: ``act(obs)`` required, ``reset(batch, seed)`` optional."""

    kind = "python"

    def __init__(self, impl, role: str, name: str, title: str = "", summary: str = ""):
        if not callable(getattr(impl, "act", None)):
            raise BrainError(f"{name}: the policy class has no act(obs) method")
        self.impl, self.role, self.name = impl, role, name
        self.title = title or getattr(impl, "title", "") or name
        doc = (type(impl).__doc__ or "").strip()
        self.summary = summary or getattr(impl, "summary", "") or (doc.splitlines()[0] if doc else "")
        self.macs = int(getattr(impl, "macs", 1000))
        self.comms = bool(getattr(impl, "comms", False))
        self.lib_id: str | None = None

    @property
    def id(self) -> str:
        return self.lib_id or self.name

    def reset(self, batch: int, seed: int = 0) -> None:
        fn = getattr(self.impl, "reset", None)
        if callable(fn):
            fn(batch, seed)

    def act(self, obs) -> dict:
        return check_actions(self.role, self.impl.act(obs), obs, self.name)

    def __reduce_ex__(self, protocol):
        # a library policy's class lives in a module only this process has loaded: a training worker
        # process reloads the package by its id instead
        if self.lib_id:
            return _library_brain, (self.lib_id, self.role)
        return super().__reduce_ex__(protocol)


_LOADED: dict[tuple[str, str], Brain] = {}


def _library_brain(spec: str, role: str) -> Brain:
    """``load_brain`` once per process (training workers unpickle the same policy for every chunk)."""
    brain = _LOADED.get((spec, role))
    if brain is None:
        brain = _LOADED[(spec, role)] = load_brain(spec, role)
    return brain


# ---------------------------------------------------------------------------------------------- mixing
class MixedBrain(Brain):
    """Different brains - or different weights of one neural brain - for different episodes of one batch.

    ``groups``: ``(brain, episode indices, per-episode weights or None)``. Training uses it to play a whole
    population against several opponents in a single simulation run.
    """

    kind = "mixed"

    def __init__(self, role: str, groups: list[tuple[Brain, np.ndarray, np.ndarray | None]]):
        self.role, self.name = role, "mixed"
        self.groups = [(b, np.asarray(i, dtype=int), p) for b, i, p in groups if len(i)]
        self.macs = max(b.macs for b, _, _ in self.groups)

    def reset(self, batch: int, seed: int = 0) -> None:
        for b, idx, _ in self.groups:
            b.reset(len(idx), seed)

    def act(self, obs, due: np.ndarray | None = None) -> dict:
        """``due`` [B]: the episodes deciding now; a brain whose episodes are not due is not asked at all, so
        it advances its own memory only at its own pace. An optional action only some brains return reads
        "no new command" (NaN, see ``KEEP``) or zero for the others' episodes."""
        B = obs.pos.shape[0]
        out: dict[str, np.ndarray] = {}
        for brain, idx, params in self.groups:
            if due is not None and not due[idx].any():
                continue
            sub = obs.take(idx)
            a = brain.act(sub, params) if params is not None else brain.act(sub)
            for k, v in a.items():
                v = np.asarray(v)
                if k not in out:
                    out[k] = (np.full((B,) + v.shape[1:], np.nan) if k in KEEP
                              else np.zeros((B,) + v.shape[1:], dtype=v.dtype))
                out[k][idx] = v
        return out


# ---------------------------------------------------------------------------------------------- registry
def _policy_packages():
    from swarm.sdk.packages import discover_packages

    return [(folder, pkg, err) for folder, pkg, err in discover_packages()
            if pkg is None or pkg.type == "policy"]


def list_brains(role: str | None = None) -> list[dict]:
    """Built-ins first, then the library (``brains/`` and ``examples/brains/``; a name in ``brains/``
    shadows an example of the same name). Reads manifests and weight headers only - runs no package code."""
    rows = [cls().info() for cls in BUILTINS.values() if role in (None, cls.role)]
    seen: set[str] = set()
    for folder, pkg, err in _policy_packages():
        name = pkg.name if pkg is not None else folder.name
        if name in seen:
            continue
        seen.add(name)
        if pkg is None:
            rows.append({"id": f"lib:{folder.name}", "name": folder.name, "role": None, "kind": "broken",
                         "title": folder.name, "summary": err, "error": err, "origin": str(folder)})
        elif role in (None, pkg.role):
            rows.append(pkg.policy_info())
    return rows


def load_brain(spec: str, role: str) -> Brain:
    """``spec``: a built-in name, ``lib:<package>`` or a path to a mesh ``.npz``."""
    if not isinstance(spec, str) or not spec:
        raise BrainError("no brain selected")
    if spec in BUILTINS:
        brain: Brain = BUILTINS[spec]()
    elif spec.startswith("lib:"):
        from swarm.sdk.packages import PackageError, discover_packages

        name = spec[4:]
        found = [pkg for _f, pkg, _e in discover_packages() if pkg is not None and pkg.name == name]
        if not found:
            raise BrainError(f"no brain package {name!r} in the library")
        try:
            brain = found[0].make_policy()
        except PackageError as exc:
            raise BrainError(str(exc)) from None
    elif spec.endswith(".npz") and Path(spec).is_file():
        brain = NeuralBrain.load(spec)
    else:
        raise BrainError(f"unknown brain {spec!r}; built-in {role} brains: "
                         f"{[n for n, c in BUILTINS.items() if c.role == role]}")
    if brain.role != role:
        raise BrainError(f"{spec} is {a_or_an(brain.role)} brain; this side needs {a_or_an(role)} brain")
    return brain


def a_or_an(role: str) -> str:
    return f"an {role}" if role[:1] in "aeiou" else f"a {role}"
