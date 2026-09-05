"""Membership: who is alive, and who leads.

Every node broadcasts a small heartbeat every ``interval`` seconds and keeps
a table ``last_seen[node_id]``. A peer that has been silent for
``timeout`` (= ``missed_beats * interval``) is declared *suspected*; the
node's ``on_member_change`` hook fires so the strategy can rebalance
(re-assign that peer's work, pick a new leader, ...). When the peer's
heartbeats return it is re-admitted and the hook fires again.

Leader election is deterministic and needs no extra messages: the leader is
the live node with the highest ``priority`` (ties -> lowest id). Because every
node runs the same rule on its own membership view, all nodes converge on the
same leader within one heartbeat timeout of a failure, without a vote.
Heartbeats carry the sender's current load so the leader can also do
load-aware assignment.
"""

from __future__ import annotations

from typing import Callable

from .channel import Channel, Message
from .core import Node

HEARTBEAT_BYTES = 8  # id, seq, load, flags - tiny by design


class Membership:
    def __init__(self, node: Node, channel: Channel, interval: float = 0.5, missed_beats: int = 3,
                 priority: Callable[[Node], float] | None = None):
        self.node = node
        self.channel = channel
        self.interval = interval
        self.timeout = interval * missed_beats
        self.priority = priority or (lambda n: n.profile.macs_per_cycle * n.profile.cpu_mhz)
        self.last_seen: dict[int, float] = {}
        self.load: dict[int, float] = {}
        self.suspected: set[int] = set()
        self.on_member_change: Callable[[int, bool], None] | None = None  # (node_id, alive)
        self._started = False

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        sim = self.node.sim
        for peer in sim.nodes:
            if peer is not self.node:
                self.last_seen[peer.id] = sim.now  # assume everyone is up at boot
        if not self._started:
            self._started = True
            # stagger by id so heartbeats don't all collide at t=0
            sim.schedule(self.interval * (self.node.id / max(1, len(sim.nodes))), self._beat)

    def reset_after_revive(self) -> None:
        """A node coming back has a stale view: trust nobody until heard from."""
        now = self.node.sim.now
        for pid in self.last_seen:
            self.last_seen[pid] = now
        self.suspected.clear()

    def _beat(self) -> None:
        sim = self.node.sim
        if self.node.alive:
            load = self.node.queue_depth() / max(1, self.node.queue_limit)
            self.channel.broadcast(self.node, Message("hb", payload=load, size=HEARTBEAT_BYTES))
            self._check_peers()
        sim.schedule(self.interval, self._beat)

    def _check_peers(self) -> None:
        now = self.node.sim.now
        for pid, t in self.last_seen.items():
            if pid not in self.suspected and now - t > self.timeout:
                self.suspected.add(pid)
                self.node.sim.metrics.failures_detected += 1
                self.node.sim.event("suspect", pid)
                self.node.sim.log(f"node {self.node.id} suspects node {pid} (silent {now - t:.2f}s)")
                if self.on_member_change:
                    self.on_member_change(pid, False)

    # ---------------------------------------------------------------- inbound
    def handle(self, src: int, msg: Message) -> bool:
        """Returns True if ``msg`` was a heartbeat (and consumed)."""
        if msg.kind != "hb":
            return False
        self.last_seen[src] = self.node.sim.now
        self.load[src] = float(msg.payload)
        if src in self.suspected:
            self.suspected.discard(src)
            self.node.sim.event("readmit", src)
            self.node.sim.log(f"node {self.node.id} re-admits node {src}")
            if self.on_member_change:
                self.on_member_change(src, True)
        return True

    # ---------------------------------------------------------------- views
    def live_peers(self) -> list[int]:
        return sorted(pid for pid in self.last_seen if pid not in self.suspected)

    def live_members(self) -> list[int]:
        """Live peers plus myself, sorted by id."""
        return sorted(self.live_peers() + [self.node.id])

    def leader(self) -> int:
        sim = self.node.sim
        members = self.live_members()
        return max(members, key=lambda i: (self.priority(sim.nodes[i]), -i))

    def is_leader(self) -> bool:
        return self.leader() == self.node.id
