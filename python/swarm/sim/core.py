"""Discrete-event simulation core.

The whole swarm is simulated in one process, in *simulated* time: every
compute step costs ``profile.compute_seconds(macs)`` and every radio message
costs channel time, so results are deterministic and independent of how fast
the development PC is. Real inference (numpy or the C++ engine) can still be
plugged in for the *content* of results; only its *duration* is modelled.
"""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from swarm.devices import DeviceProfile


class Event:
    __slots__ = ("time", "seq", "fn", "args")

    def __init__(self, time: float, seq: int, fn: Callable, args: tuple):
        self.time, self.seq, self.fn, self.args = time, seq, fn, args

    def __lt__(self, other: "Event") -> bool:
        return (self.time, self.seq) < (other.time, other.seq)


@dataclass
class Metrics:
    """Counters every strategy reports so they can be compared apples-to-apples."""

    frames_captured: int = 0
    frames_processed: int = 0      # brain ran on a frame somewhere in the swarm
    frames_rejected: int = 0       # a node's compute queue was full (explicit local drop)
    objects_captured: int = 0      # captured frames that really contained something (ground truth)
    objects_reported: int = 0      # ...for which the leader learnt of a detection (mission recall)
    false_alarms_reported: int = 0  # leader learnt of a detection on an empty frame
    detections: int = 0            # frames where the brain found something (class != 0)
    detections_reported: int = 0   # ...and the leader learnt about it
    correct: int = 0               # predicted class == ground truth
    bytes_sent: int = 0
    messages_sent: int = 0
    messages_lost: int = 0
    macs_total: int = 0
    macs_by_node: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    busy_by_node: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    latencies: list[float] = field(default_factory=list)   # capture -> leader knows about a detection
    learn_times: list[float] = field(default_factory=list)  # when the leader learnt something (recovery analysis)
    events: list[tuple[str, float, int]] = field(default_factory=list)  # (kill|revive|suspect|readmit|leader, t, node)
    leader_changes: int = 0
    failures_detected: int = 0
    rebalances: int = 0
    energy_j: float = 0.0

    def summary(self, sim_time: float, n_nodes: int) -> dict[str, Any]:
        import statistics

        lat = sorted(self.latencies)
        p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else float("nan")
        # work booked just before the end may finish after it; a saturated core reads 100%, not 100.3%
        busy = [min(1.0, self.busy_by_node.get(i, 0.0) / sim_time) for i in range(n_nodes)]
        return {
            "frames_captured": self.frames_captured,
            "frames_processed": self.frames_processed,
            # captured but never processed: queue full, radio loss, or sent to a node that died
            "frames_dropped": self.frames_captured - self.frames_processed,
            "frames_rejected": self.frames_rejected,
            "objects_captured": self.objects_captured,
            "objects_reported": self.objects_reported,
            "false_alarms_reported": self.false_alarms_reported,
            "recall": self.objects_reported / self.objects_captured if self.objects_captured else float("nan"),
            "detections": self.detections,
            "detections_reported": self.detections_reported,
            "accuracy": self.correct / self.frames_processed if self.frames_processed else float("nan"),
            "latency_p50_ms": p(0.5) * 1e3,
            "latency_p95_ms": p(0.95) * 1e3,
            "bytes_sent": self.bytes_sent,
            "kbps": self.bytes_sent * 8 / sim_time / 1e3,
            "bytes_per_frame": self.bytes_sent / self.frames_processed if self.frames_processed else float("nan"),
            "messages_lost": self.messages_lost,
            "macs_total": self.macs_total,
            "gmac_total": self.macs_total / 1e9,
            "macs_per_frame": self.macs_total / self.frames_processed if self.frames_processed else float("nan"),
            "busy_mean": statistics.fmean(busy) if busy else 0.0,
            "busy_max": max(busy) if busy else 0.0,
            "busy_by_node": [round(b, 3) for b in busy],
            "macs_by_node": [self.macs_by_node.get(i, 0) for i in range(n_nodes)],
            "leader_changes": self.leader_changes,
            "failures_detected": self.failures_detected,
            "rebalances": self.rebalances,
            "energy_j": self.energy_j,
            "events": list(self.events),
        }


class Simulator:
    def __init__(self, seed: int = 0):
        import random

        self.now = 0.0
        self._queue: list[Event] = []
        self._seq = itertools.count()
        self.rng = random.Random(seed)
        self.metrics = Metrics()
        self.log_lines: list[str] = []
        self.verbose = False
        self.nodes: list["Node"] = []

    def schedule(self, delay: float, fn: Callable, *args) -> Event:
        ev = Event(self.now + max(0.0, delay), next(self._seq), fn, args)
        heapq.heappush(self._queue, ev)
        return ev

    def run(self, until: float) -> None:
        while self._queue and self._queue[0].time <= until:
            ev = heapq.heappop(self._queue)
            self.now = ev.time
            ev.fn(*ev.args)
        self.now = until

    def log(self, msg: str) -> None:
        line = f"[{self.now:8.3f}] {msg}"
        self.log_lines.append(line)
        if self.verbose:
            print(line)

    def event(self, kind: str, node: int) -> None:
        """Records a membership/fault event for post-run survivability analysis."""
        self.metrics.events.append((kind, self.now, node))

    def add_node(self, node: "Node") -> "Node":
        node.sim = self
        node.id = len(self.nodes)
        self.nodes.append(node)
        return node

    def alive_nodes(self) -> list["Node"]:
        return [n for n in self.nodes if n.alive]


class Node:
    """A swarm member: a single-core device that is either idle or busy.

    Compute is modelled as a FIFO: ``compute(macs, cb)`` books
    ``profile.compute_seconds(macs)`` of CPU time after any already-queued
    work, then fires ``cb``. ``queue_limit`` bounds how much work may wait;
    beyond it work is rejected (the caller decides whether that is a drop).
    """

    def __init__(self, profile: DeviceProfile, name: str | None = None, queue_limit: int = 2):
        self.profile = profile
        self.name = name or profile.name
        self.sim: Simulator = None  # type: ignore[assignment]
        self.id = -1
        self.alive = True
        self._busy_until = 0.0
        self._queued = 0
        self._epoch = 0  # bumped on kill so completions of pre-death work are ignored
        self.queue_limit = queue_limit

    # ---------------------------------------------------------------- compute
    def busy(self) -> bool:
        return self._busy_until > self.sim.now

    def queue_depth(self) -> int:
        return self._queued

    def can_accept(self) -> bool:
        return self.alive and self._queued < self.queue_limit

    def compute(self, macs: int, cb: Callable, *args) -> bool:
        """Books ``macs`` of work; returns False (and does nothing) if the queue is full or the node is dead."""
        if not self.can_accept():
            return False
        dur = self.profile.compute_seconds(macs)
        start = max(self.sim.now, self._busy_until)
        self._busy_until = start + dur
        self._queued += 1
        m = self.sim.metrics
        m.macs_total += macs
        m.macs_by_node[self.id] += macs
        m.busy_by_node[self.id] += dur
        m.energy_j += self.profile.energy_joules(macs, 0)
        self.sim.schedule(self._busy_until - self.sim.now, self._finish, self._epoch, cb, args)
        return True

    def _finish(self, epoch: int, cb: Callable, args: tuple) -> None:
        if epoch != self._epoch:
            return  # node died (and maybe came back) since this work was booked
        self._queued -= 1
        cb(*args)

    # ---------------------------------------------------------------- lifecycle
    def kill(self) -> None:
        if self.alive:
            self.alive = False
            self._queued = 0
            self._busy_until = self.sim.now
            self._epoch += 1
            self.sim.log(f"node {self.id} ({self.name}) went OFFLINE")
            self.sim.event("kill", self.id)
            self.on_killed()

    def revive(self) -> None:
        if not self.alive:
            self.alive = True
            self.sim.log(f"node {self.id} ({self.name}) came back ONLINE")
            self.sim.event("revive", self.id)
            self.on_revived()

    def on_killed(self) -> None:
        pass

    def on_revived(self) -> None:
        pass

    # ---------------------------------------------------------------- messaging (filled in by Channel)
    def receive(self, src: int, msg: Any) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Node({self.id}, {self.name}, {'up' if self.alive else 'down'})"
