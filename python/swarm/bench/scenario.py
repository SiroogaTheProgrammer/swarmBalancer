"""Scenario building blocks.

Everything is a frozen dataclass so a preset file reads like a config and a
scenario can be hashed / compared / serialised. Defaults describe the
reference swarm used throughout the docs: a Cortex-A72 command drone plus
four Cortex-A53 drones, 128 px cameras at 10 fps, a 6 Mbps shared link.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

ALL_STRATEGIES: tuple[str, ...] = ("local", "central/none", "central/downscale", "central/features", "striped")
BASELINE = "local"

STRATEGY_NOTES = {
    "local": "every drone runs the whole brain on its own camera; only detections go to the leader",
    "central/none": "every camera streams raw frames to the command drone, which computes for all",
    "central/downscale": "drones resize frames to the brain input before streaming to the command drone",
    "central/features": "drones run the conv frontend and ship int8 features; leader runs only the dense head",
    "striped": "one camera (the leader's); frame i goes to live member i mod n; workers answer only if useful",
}


def split_strategy(name: str) -> tuple[str, str]:
    """``"central/features"`` -> ``("central", "features")``; ``"striped"`` -> ``("striped", "none")``."""
    if "/" in name:
        s, p = name.split("/", 1)
        return s, p
    return name, "none"


@dataclass(frozen=True)
class Swarm:
    """Who is in the swarm. The leader is elected by capability, not by position in this list."""

    leader: str = "drone-a72"
    worker: str = "drone-a53"
    size: int = 5
    nodes: tuple[str, ...] | None = None  # explicit (heterogeneous) list; overrides leader/worker/size

    def profile_names(self) -> tuple[str, ...]:
        if self.nodes:
            return tuple(self.nodes)
        if self.size < 1:
            raise ValueError("swarm size must be >= 1")
        return (self.leader,) + (self.worker,) * (self.size - 1)

    def describe(self) -> str:
        names = self.profile_names()
        if self.nodes:
            counts: dict[str, int] = {}
            for n in names:
                counts[n] = counts.get(n, 0) + 1
            return f"{len(names)} nodes (" + ", ".join(f"{c}x {n}" for n, c in counts.items()) + ")"
        return f"{len(names)} nodes ({self.leader} + {len(names) - 1}x {self.worker})"


@dataclass(frozen=True)
class Camera:
    fps: float = 10.0                 # per active camera
    size: int = 128                   # pixels per side, 8-bit mono
    p_object: float = 0.3             # fraction of frames that contain something
    # Striped uses a single camera. "match" runs it at fps * swarm size so the swarm processes as many
    # frames per second as the all-cameras strategies (the "one fast camera, many brains" hypothesis);
    # None keeps the per-camera fps (a low-power configuration); a number sets it explicitly.
    striped_fps: float | str | None = "match"


@dataclass(frozen=True)
class Link:
    bps: float | None = None          # shared channel bit rate; None -> slowest radio in the swarm
    latency_s: float | None = None    # one-way latency; None -> worst radio in the swarm
    loss: float = 0.01                # per-message loss probability
    qos: bool = True                  # heartbeats/results/commands do not queue behind bulk frames
    hb_interval: float = 0.5          # heartbeat period (s)
    missed_beats: int = 3             # silent for this many periods -> suspected offline


@dataclass(frozen=True)
class BrainSpec:
    """Which brain the nodes run.

    ``auto`` uses ``model`` inside the C++ engine if both exist, else numpy, else the oracle.
    The oracle has no weights: it answers with the ground truth at ``accuracy`` and costs ``macs``
    per frame - use it to explore hypothetical model sizes without training anything.
    """

    kind: str = "auto"                # auto | oracle | numpy | native
    model: str = "models/tiny_cnn_int8.swm"
    macs: int = 401_568               # oracle: MACs per frame
    accuracy: float = 0.87            # oracle: P(correct class)
    input_size: int = 32              # oracle: brain input is input_size^2 bytes
    frontend_fraction: float = 0.92   # oracle: share of MACs in the conv frontend (split inference)


@dataclass(frozen=True)
class Faults:
    """Who goes offline and when. ``who`` is a node id or ``"leader"`` (whoever leads at that moment)."""

    kill: tuple[tuple[str | int, float], ...] = ()
    revive: tuple[tuple[int, float], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.kill or self.revive)

    def describe(self) -> str:
        parts = [f"kill {w}@{t:g}s" for w, t in self.kill] + [f"revive {w}@{t:g}s" for w, t in self.revive]
        return ", ".join(parts) if parts else "none"


@dataclass(frozen=True)
class Stress:
    devices: tuple[str, ...] | None = None  # None -> every distinct device in the swarm
    frames: int = 50                        # frames actually run on the host per device


@dataclass(frozen=True)
class Check:
    """A pass/fail expectation, evaluated on the per-strategy metrics of the result.

    ``strategy`` is a strategy name or ``"*"`` for all of them. With ``ratio=True`` the metric is
    divided by the baseline's value first, so ``Check("striped", "gmac_ps", "<=", 0.3, ratio=True)``
    reads "striped must use at most 30 % of the baseline's compute".
    """

    strategy: str
    metric: str
    op: str
    value: float
    ratio: bool = False
    note: str = ""

    def describe(self) -> str:
        m = f"{self.metric}/baseline" if self.ratio else self.metric
        return f"{self.strategy:<18} {m} {self.op} {self.value:g}" + (f"   # {self.note}" if self.note else "")


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str = ""
    swarm: Swarm = field(default_factory=Swarm)
    camera: Camera = field(default_factory=Camera)
    link: Link = field(default_factory=Link)
    brain: BrainSpec = field(default_factory=BrainSpec)
    faults: Faults = field(default_factory=Faults)
    duration: float = 30.0            # simulated seconds
    seed: int = 0
    strategies: tuple[str, ...] = ALL_STRATEGIES
    baseline: str = BASELINE          # always simulated; every other strategy is reported relative to it
    queue_limit: int = 4              # frames a node may have waiting for its CPU
    confidence_threshold: float = 0.5  # local/striped: report only detections at least this confident
    include_leader: bool = True       # striped: the leader also computes a share of frames
    stress: Stress = field(default_factory=Stress)
    checks: tuple[Check, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = [s for s in self.strategies if s not in ALL_STRATEGIES]
        if unknown or self.baseline not in ALL_STRATEGIES:
            raise ValueError(f"unknown strategies {unknown or [self.baseline]}; known: {ALL_STRATEGIES}")
        if self.duration <= 0 or self.camera.fps <= 0:
            raise ValueError("duration and fps must be positive")

    def strategy_order(self) -> tuple[str, ...]:
        """Baseline first, then the others in the declared order."""
        rest = tuple(s for s in self.strategies if s != self.baseline)
        return (self.baseline,) + rest

    def striped_fps(self) -> float:
        n = len(self.swarm.profile_names())
        sf = self.camera.striped_fps
        if sf is None:
            return self.camera.fps
        if sf == "match":
            return self.camera.fps * n
        return float(sf)


def to_plain(obj: Any) -> Any:
    """Dataclass -> JSON-friendly dict (tuples become lists)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    return obj
