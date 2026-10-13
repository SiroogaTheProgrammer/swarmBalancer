"""Settings of the arena tests and of training: frozen dataclasses with bounded values.

They arrive as JSON from the browser (``from_dict`` checks every field) or are built in Python. Distances
are metres, speeds m/s, times seconds; the world is 3-D (x across the front, y forward, z altitude). The
defaults are a sensible small-drone engagement over a few kilometres.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

from swarm.devices import DEVICE_PROFILES


@dataclass(frozen=True)
class ClashConfig:
    """DEFENCE / OFFENCE: a wall of interceptors guards the plane y = 0 against a swarm coming from y = depth."""

    defenders: int = 24
    attackers: int = 48
    width: float = 3000.0          # the front: width of the airspace, and of the protected line
    depth: float = 3000.0          # attackers start this far from the line
    ceiling: float = 600.0         # top of the airspace both sides fly in
    low_altitude: float = 40.0     # below this height terrain hides a drone (seen at a third of the
                                   # sensor range) but terrain following costs a quarter of its speed
    defender_speed: float = 55.0
    attacker_speed: float = 40.0
    defender_climb: float = 20.0   # climb / descent rates: changing height is slower than flying level
    attacker_climb: float = 10.0
    defender_accel: float = 35.0
    attacker_accel: float = 20.0
    capture_radius: float = 8.0    # an interceptor that passes this close may take the attacker out
    p_kill: float = 0.85           # ...with this probability (a miss can try again)
    shots: int = 2                 # kills an interceptor can make before it is spent (1 = kamikaze interceptor)
    guard_depth: float = 0.5       # interceptors may wait at most this share of the depth in front of the line
    sensor_range: float = 1500.0   # both sides see enemies this far (3-D distance)
    comm_range: float = 800.0      # one-hop mesh radio range within each swarm
    comm_loss: float = 0.05        # per link, per message
    relay: int = 1                 # radio hops a sighting travels through each swarm's mesh, one per decision
    spawn: str = "line"            # how the attackers arrive: line | cluster | waves
    duration: float = 150.0
    dt: float = 1.0                # seconds between decisions; flight is simulated in sub-steps of <= 0.5 s
    defender_device: str = "mcu-m7"
    attacker_device: str = "mcu-m4"


@dataclass(frozen=True)
class ReconConfig:
    """RECON: a swarm surveys a square area from a base in its middle, ideally covering it in full detail,
    staying in radio contact with its lead drone and keeping the sensing / radio / flight load low."""

    drones: int = 12
    area: float = 3000.0           # side of the square to survey; the base is in its middle
    duration: float = 600.0
    dt: float = 2.0
    speed: float = 18.0
    climb: float = 5.0             # climb / descent rate
    accel: float = 4.0
    min_altitude: float = 20.0
    max_altitude: float = 400.0
    camera_fov: float = 90.0       # degrees: a camera sees ground within altitude * tan(fov / 2)
    detail_altitude: float = 100.0  # targets are recognised for sure up to this height; detail fades to none at 3x
    comm_range: float = 600.0
    comm_loss: float = 0.02
    targets: int = 30              # hidden objects to find
    grid: int = 40                 # coverage map cells per side
    device: str = "drone-a53"
    perception_macs: int = 150_000_000  # the detector each camera frame costs (MobileNet-SSD-lite class)
    camera_fps: float = 8.0
    w_cohesion: float = 1.0        # training weights: staying connected ...
    w_load: float = 1.0            # ... and keeping cameras/radios quiet, against coverage


@dataclass(frozen=True)
class TrainConfig:
    """Evolution-strategy training: one side, both sides (co-evolution), or a recon swarm."""

    mode: str = "clash"            # clash | recon
    generations: int = 200         # ~4 min for two swarms on 8 cores
    population: int = 16           # perturbations per generation (antithetic pairs)
    episodes: int = 2              # episodes per perturbation and opponent
    sigma: float = 0.1
    lr: float = 0.05
    hidden: int = 16               # neurons of the mesh brain's hidden layer
    learn_defenders: bool = True
    learn_attackers: bool = True
    start_defender: str = "fresh"  # "fresh" or a brain id; a non-neural brain is a frozen opponent
    start_attacker: str = "fresh"
    start_recon: str = "fresh"
    record_every: int = 5          # record a watchable match every N generations (at most ~120 per run)
    seed: int = 0
    workers: int = 0               # simulation processes; 0 = one per CPU core. Speed only: same result


LIMITS: dict[str, tuple[float, float]] = {
    "defenders": (1, 64), "attackers": (1, 128), "width": (200, 20000), "depth": (300, 20000),
    "ceiling": (50, 3000), "low_altitude": (0, 500),
    "defender_speed": (5, 150), "attacker_speed": (5, 150), "defender_climb": (1, 100), "attacker_climb": (1, 100),
    "defender_accel": (2, 200), "attacker_accel": (2, 200),
    "capture_radius": (1, 50), "p_kill": (0.05, 1), "shots": (1, 20), "guard_depth": (0.1, 0.9),
    "sensor_range": (50, 20000),
    "comm_range": (20, 20000), "relay": (1, 50),
    "comm_loss": (0, 0.9), "duration": (10, 3600), "dt": (0.05, 5), "drones": (1, 48), "area": (300, 20000),
    "speed": (2, 60), "climb": (0.5, 30), "accel": (0.5, 30), "min_altitude": (5, 1000), "max_altitude": (20, 3000),
    "camera_fov": (10, 150), "detail_altitude": (10, 1000), "targets": (0, 300), "grid": (8, 96),
    "perception_macs": (0, 1e10), "camera_fps": (0.1, 60), "w_cohesion": (0, 10), "w_load": (0, 10),
    "generations": (1, 20000), "population": (4, 256), "episodes": (1, 16), "sigma": (0.005, 1),
    "lr": (0.0005, 0.5), "hidden": (4, 256), "record_every": (1, 1000), "seed": (0, 2**31 - 1),
    "workers": (0, 256),
}
CHOICES = {"spawn": ("line", "cluster", "waves"), "mode": ("clash", "recon")}
DEVICE_FIELDS = ("defender_device", "attacker_device", "device")

# name: (label, unit, help, shown up front). Everything else sits under "Advanced" in the studio.
FIELD_INFO: dict[str, tuple[str, str, str, bool]] = {
    "defenders": ("Interceptors", "", "drones in the defending wall", True),
    "attackers": ("Attackers", "", "drones in the incoming swarm", True),
    "width": ("Front width", "m", "width of the airspace, and of the protected line", True),
    "depth": ("Start distance", "m", "how far from the line the attackers start", True),
    "ceiling": ("Ceiling", "m", "top of the airspace both sides fly in", True),
    "low_altitude": ("Terrain mask", "m", "below this height a drone is hidden by terrain - seen at a third of the "
                     "sensor range - but terrain following costs a quarter of its speed", True),
    "attacker_speed": ("Attacker speed", "m/s", "top speed of the attackers", True),
    "spawn": ("Arrival", "", "line: spread along the front, cluster: one tight group, waves: two waves", True),
    "shots": ("Shots each", "", "kills an interceptor can make before it is spent (1 = kamikaze interceptor)", True),
    "comm_range": ("Radio range", "m", "one-hop mesh radio range between friends (3-D distance)", True),
    "relay": ("Sighting relay", "hops", "how many radio hops a sighting travels through each swarm's mesh, one per "
              "decision (1: only to the friends that hear the drone that sees it)", False),
    "guard_depth": ("Forward limit", "", "how far in front of the line the interceptors' brains may post them, as a "
                    "share of the start distance (0.1: on the line)", False),
    "p_kill": ("Hit chance", "", "chance that a pass within the capture radius takes the attacker out", False),
    "comm_loss": ("Radio loss", "", "chance that a message is lost on a link", False),
    "defender_speed": ("Interceptor speed", "m/s", "top speed of the interceptors", False),
    "defender_climb": ("Interceptor climb rate", "m/s", "how fast an interceptor changes height", False),
    "attacker_climb": ("Attacker climb rate", "m/s", "how fast an attacker changes height", False),
    "defender_accel": ("Interceptor accel.", "m/s²", "", False),
    "attacker_accel": ("Attacker accel.", "m/s²", "", False),
    "capture_radius": ("Capture radius", "m", "how close an interceptor must pass", False),
    "sensor_range": ("Sensor range", "m", "both sides see enemies this far (3-D distance)", False),
    "duration": ("Time limit", "s", "", False),
    "dt": ("Decision interval", "s", "how often every brain decides (DEFENCE / OFFENCE fly in physics steps of "
           "at most 0.5 s in between)", False),
    "defender_device": ("Interceptor computer", "", "the board each interceptor's brain runs on", False),
    "attacker_device": ("Attacker computer", "", "the board each attacker's brain runs on", False),
    "drones": ("Drones", "", "size of the recon swarm", True),
    "area": ("Area", "m", "side of the square to survey; the base is in its middle", True),
    "targets": ("Hidden targets", "", "objects to find and report", True),
    "detail_altitude": ("Full-detail height", "m", "targets are recognised for sure up to this height; detail "
                        "fades to nothing at three times it", True),
    "max_altitude": ("Highest altitude", "m", "the highest the swarm may fly: a wider camera view, in less detail", True),
    "min_altitude": ("Lowest altitude", "m", "the lowest the swarm may fly", False),
    "camera_fov": ("Camera field of view", "°", "footprint radius = altitude x tan(fov / 2)", False),
    "speed": ("Cruise speed", "m/s", "top horizontal speed of a recon drone", False),
    "climb": ("Climb rate", "m/s", "how fast a recon drone changes height", False),
    "accel": ("Acceleration", "m/s²", "", False),
    "grid": ("Map cells per side", "", "resolution of the shared coverage map", False),
    "device": ("Drone computer", "", "the board the camera brain and swarm brain run on", False),
    "perception_macs": ("Detector cost", "MAC/frame", "compute of the image brain per camera frame", False),
    "camera_fps": ("Camera rate", "fps", "", False),
    "w_cohesion": ("Weight: staying together", "", "training score weight on radio contact", False),
    "w_load": ("Weight: low load", "", "training score weight on camera and flight load", False),
    "generations": ("Generations", "", "rounds of evolution", True),
    "population": ("Variants per generation", "", "brain variants tried each round (even)", False),
    "episodes": ("Episodes per variant", "", "", False),
    "sigma": ("Exploration noise", "", "how far variants stray from the current brain", False),
    "lr": ("Learning rate", "", "", False),
    "hidden": ("Hidden neurons", "", "size of a fresh mesh network", False),
    "record_every": ("Record a match every", "gen.", "long runs record less often: at most about 120 matches", False),
    "seed": ("Seed", "", "same seed + same settings = same result", False),
    "workers": ("CPU cores", "", "simulation processes run in parallel; 0 = all cores. Changes the speed only, "
                "never the result", False),
}


def schema(cls) -> list[dict]:
    """The fields of a settings class, with defaults, limits and labels, for building forms."""
    from swarm.devices import DEVICE_PROFILES

    out = []
    for f in fields(cls):
        label, unit, help_text, basic = FIELD_INFO.get(f.name, (f.name.replace("_", " "), "", "", False))
        d = f.default
        row = {"name": f.name, "default": d, "label": label, "unit": unit, "help": help_text, "basic": basic}
        if isinstance(d, bool):
            row["type"] = "bool"
        elif isinstance(d, (int, float)):
            row["type"] = "int" if isinstance(d, int) else "float"
            lo, hi = LIMITS.get(f.name, (None, None))
            row.update(min=lo, max=hi)
        elif f.name in CHOICES:
            row.update(type="choice", choices=list(CHOICES[f.name]))
        elif f.name in DEVICE_FIELDS:
            row.update(type="choice", choices=[n for n in DEVICE_PROFILES if n != "host"])
        else:
            row["type"] = "text"
        out.append(row)
    return out


class ConfigError(ValueError):
    """A setting has the wrong type or is out of range; the message names the field."""


def from_dict(cls, data: dict | None):
    """Builds ``cls`` from JSON, checking types and ranges; unknown keys are an error."""
    data = dict(data or {})
    names = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ConfigError(f"unknown settings {unknown}")
    out = {}
    for name, f in names.items():
        default = f.default
        v = data.get(name, default)
        if isinstance(default, bool):
            if not isinstance(v, bool):
                raise ConfigError(f"{name} must be true or false")
        elif isinstance(default, (int, float)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ConfigError(f"{name} must be a number")
            if isinstance(default, int):
                if float(v) != int(v):
                    raise ConfigError(f"{name} must be a whole number")
                v = int(v)
            else:
                v = float(v)
            lo, hi = LIMITS.get(name, (-math.inf, math.inf))
            if not lo <= v <= hi:
                raise ConfigError(f"{name} must be within [{lo:g}, {hi:g}], got {v:g}")
        else:
            if not isinstance(v, str) or len(v) > 200:
                raise ConfigError(f"{name} must be a short text")
            if name in CHOICES and v not in CHOICES[name]:
                raise ConfigError(f"{name} must be one of {CHOICES[name]}")
            if name in DEVICE_FIELDS and v not in DEVICE_PROFILES:
                raise ConfigError(f"{name}: unknown device {v!r}; known: {sorted(DEVICE_PROFILES)}")
        out[name] = v
    cfg = cls(**out)
    if isinstance(cfg, TrainConfig) and cfg.population % 2:
        raise ConfigError("population must be even (perturbations come in +/- pairs)")
    if isinstance(cfg, ClashConfig) and cfg.low_altitude >= 0.5 * cfg.ceiling:
        raise ConfigError("the terrain mask (low_altitude) must be below half the ceiling")
    if isinstance(cfg, ReconConfig) and cfg.min_altitude >= cfg.max_altitude:
        raise ConfigError("min_altitude must be below max_altitude")
    return cfg


def to_dict(cfg) -> dict:
    return asdict(cfg)
