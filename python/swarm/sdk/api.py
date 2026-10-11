"""The public classes of the SDK: what a perception brain and a tactic look like.

Perception brain (Python format)
    Any class with ``input_shape``, ``classes``, ``macs`` and ``classify(image) -> (class, confidence)``.
    Subclassing :class:`PerceptionBrain` is optional - a brain that should also run on a drone must not
    import ``swarm`` at all (the device payload only carries the runtime), so the templates write plain
    classes that need nothing but numpy.

Tactic
    A :class:`Tactic` is a swarm member's coordination logic - who looks, who thinks, who tells whom -
    written against the simulator's event hooks, exactly like the built-in ``local`` / ``central`` /
    ``striped`` designs in :mod:`swarm.sim.strategies`.
"""

from __future__ import annotations

import inspect
import random
from typing import Any, Callable

import numpy as np

from swarm.sim.brains import Brain, Frame, downscale_macs
from swarm.sim.channel import Message
from swarm.sim.strategies import RESULT_BYTES, SwarmNode
from swarm.train.dataset import CLASSES

__all__ = ["CLASSES", "Brain", "Frame", "Message", "PerceptionBrain", "PythonBrainAdapter", "RESULT_BYTES", "Tactic",
           "downscale_macs", "instantiate"]

EMPTY_CLASS_NAMES = ("nothing", "none", "empty", "background", "clear")


class PerceptionBrain:
    """Optional base class for Python perception brains (simulation-only if you keep the import).

    ``classify`` receives the frame exactly as the simulator's cameras deliver it: ``float32`` array of
    shape ``input_shape`` (channels, height, width) with values in ``[0, 1]``. It returns
    ``(class_index, confidence)``; class 0 must mean "nothing here" - only other classes are detections.
    """

    input_shape: tuple[int, int, int] = (1, 32, 32)
    classes: tuple[str, ...] = tuple(CLASSES)
    macs: int = 0                      # declared multiply-accumulates per frame; the simulator charges this
    frontend_fraction: float = 0.9     # share of macs before the split point (central/features)
    feature_bytes: int | None = None   # bytes shipped at the split point; default: one byte per input pixel
    ram_bytes: int | None = None       # declared working memory, informational

    def __init__(self, seed: int = 0, options: dict | None = None):
        self.seed = seed
        self.options = dict(options or {})

    def classify(self, image: np.ndarray) -> tuple[int, float]:
        raise NotImplementedError


def instantiate(cls: type, seed: int = 0, options: dict | None = None) -> Any:
    """Builds a brain, passing ``seed`` / ``options`` only if its constructor accepts them."""
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return cls()
    var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    kwargs: dict[str, Any] = {}
    if "seed" in params or var_kw:
        kwargs["seed"] = seed
    if "options" in params or var_kw:
        kwargs["options"] = dict(options or {})
    return cls(**kwargs)


class PythonBrainAdapter(Brain):
    """Makes a Python perception brain look like a simulator :class:`~swarm.sim.brains.Brain`."""

    def __init__(self, impl: Any, name: str = ""):
        self.impl = impl
        self.name = name or type(impl).__name__
        shape = tuple(int(d) for d in impl.input_shape)
        if len(shape) != 3:
            raise ValueError(f"{self.name}: input_shape must be (channels, height, width), got {shape}")
        self.input_shape = shape
        self.n_classes = len(impl.classes)
        self.macs = int(impl.macs)
        frac = float(getattr(impl, "frontend_fraction", 0.9))
        self.frontend_macs = int(self.macs * min(max(frac, 0.0), 1.0))
        self.head_macs = self.macs - self.frontend_macs
        fb = getattr(impl, "feature_bytes", None)
        self.feature_bytes = int(fb) if fb else int(np.prod(shape))

    def infer(self, frame: Frame) -> tuple[int, float]:
        pred, conf = self.impl.classify(frame.image)
        return int(pred), float(conf)

    def describe(self) -> str:
        return f"python brain {self.name}: " + super().describe().split(": ", 1)[1] + " (declared cost)"


class Tactic(SwarmNode):
    """Base class for coordination tactics. One instance runs on every swarm member.

    Hooks you override (see docs/SDK.md for the event order)::

        on_start()                  booted, or came back after a failure: decide whether the camera runs
        on_capture(frame)           this member's camera produced a frame (REQUIRED)
        on_frame(src, msg)          a message of kind "frame" arrived; any kind "k" goes to on_k(src, msg)
        on_result(src, msg)         a detection report arrived (default: the leader learns it)
        rebalance(node_id, alive)   a peer was declared dead, or came back

    Rules every tactic must keep (``python -m swarm.sdk validate`` checks the first and the last):

    * pay the physics: shrink every camera frame to the brain input somewhere (``preprocess`` /
      ``downscale_cost``), and run the full brain cost per processed frame (``run_brain``);
    * move data between members only with ``send`` / ``broadcast`` (never call another node's methods);
    * be deterministic: use ``self.rng`` for randomness, never wall-clock time.
    """

    strategy: str = ""             # name used in scenarios; lower-case identifier, not a built-in
    summary: str = ""              # one line shown in reports and in the studio
    camera_mode = "all"            # "all": every camera at the scenario fps; "single": one camera at striped fps

    # ------------------------------------------------------------ swarm view
    @property
    def now(self) -> float:
        """Simulated time in seconds."""
        return self.sim.now

    @property
    def is_leader(self) -> bool:
        return self.membership.is_leader()

    @property
    def leader_id(self) -> int:
        return self.membership.leader()

    def live_members(self) -> list[int]:
        """Members this node currently believes are alive (itself included), sorted by id."""
        return self.membership.live_members()

    def peer_load(self, node_id: int) -> float:
        """Queue fill (0..1) of a member as of its last heartbeat; this node's own value is current."""
        if node_id == self.id:
            return self.queue_depth() / max(1, self.queue_limit)
        return float(self.membership.load.get(node_id, 0.0))

    def node_speed(self, node_id: int) -> float:
        """Brain MACs per second a member's device sustains (device types are known before take-off)."""
        p = self.sim.nodes[node_id].profile
        return p.cpu_mhz * 1e6 * p.macs_per_cycle

    @property
    def rng(self) -> random.Random:
        """Per-member random stream seeded from the scenario: deterministic, and independent of the radio's."""
        r = self.__dict__.get("_tactic_rng")
        if r is None:
            r = self.__dict__["_tactic_rng"] = random.Random(self.seed * 1_000_003 + self.id)
        return r

    # ------------------------------------------------------------ costs
    def downscale_cost(self) -> int:
        """MACs to shrink one camera frame to the brain input (every design pays this once per frame)."""
        return downscale_macs(self.cfg.cam_size ** 2)

    def compute_time(self, macs: int, node_id: int | None = None) -> float:
        """Seconds ``macs`` take on a member's CPU (this node by default)."""
        return macs / self.node_speed(self.id if node_id is None else node_id)

    def cpu_free_at(self) -> float:
        """When this node's CPU will have finished everything already booked on it."""
        return max(self.sim.now, self._busy_until)

    def airtime(self, nbytes: int) -> float:
        """Seconds for an uncontended message of ``nbytes`` to reach a peer (serialisation + latency)."""
        return (nbytes + self.channel.HEADER_BYTES) * 8.0 / self.channel.bps + self.channel.latency

    @property
    def input_bytes(self) -> int:
        """Bytes of one frame at brain-input size - what a downscaled frame costs on the radio."""
        return self.brain.input_bytes

    def preprocess(self, frame: Frame, then: Callable, *args) -> bool:
        """Books the downscale on this CPU, then calls ``then(frame, *args)``. False if the queue was full."""
        return self.try_compute(self.downscale_cost(), then, frame, *args)

    # ------------------------------------------------------------ radio
    def send(self, dst: int, kind: str, payload: Any = None, size: int = 0) -> None:
        """Unicast over the shared channel; ``size`` bytes of airtime (+ header) are charged."""
        self.channel.send(self, self.sim.nodes[dst], Message(kind, payload=payload, size=int(size)))

    def broadcast(self, kind: str, payload: Any = None, size: int = 0) -> None:
        self.channel.broadcast(self, Message(kind, payload=payload, size=int(size)))

    def send_frame(self, dst: int, frame: Frame) -> None:
        """Ships a downscaled frame (``input_bytes``) to ``dst``, whose ``on_frame`` receives it."""
        self.send(dst, "frame", frame, self.input_bytes)
