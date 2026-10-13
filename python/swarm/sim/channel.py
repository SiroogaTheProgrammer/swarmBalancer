"""Shared radio channel.

All swarm members share one medium (like a Wi-Fi/mesh channel or a satellite
downlink slot): messages are serialised one after another at the channel's
bit rate, so a strategy that pushes raw frames from every drone saturates
the link and *everyone's* latency grows. Each message also pays the
one-way latency and may be lost with probability ``loss``.

The channel is what makes the "send everything to the command drone" vs
"pre-process first" vs "stripe frames across the swarm" trade-off visible.

With ``qos=True`` (default) small *control* messages - heartbeats, results,
commands - do not queue behind bulk frame data (think a separate low-rate
control channel or a priority MAC class). Without QoS a saturated link also
starves the heartbeats, so nodes falsely suspect each other and the swarm
flaps; run with ``--no-qos`` to see that failure mode.
"""

from __future__ import annotations

from typing import Any

from .core import Node, Simulator

CONTROL_KINDS = frozenset({"hb", "result", "cmd"})


class Message:
    __slots__ = ("kind", "payload", "size", "t_sent", "meta")

    def __init__(self, kind: str, payload: Any = None, size: int = 0, meta: dict | None = None):
        self.kind = kind
        self.payload = payload
        self.size = size  # bytes on the wire
        self.t_sent = 0.0
        self.meta = meta or {}

    def __repr__(self) -> str:
        return f"Message({self.kind}, {self.size}B)"


class Channel:
    HEADER_BYTES = 24  # framing, addressing, CRC

    def __init__(self, sim: Simulator, bps: float, latency_s: float, loss: float = 0.0, qos: bool = True):
        self.sim = sim
        self.bps = bps
        self.latency = latency_s
        self.loss = loss
        self.qos = qos
        self._free_at = 0.0
        self.busy_time = 0.0

    def utilisation(self) -> float:
        """Airtime demanded / wall time. Above 1.0 the medium is oversubscribed and queues grow without bound."""
        return self.busy_time / self.sim.now if self.sim.now > 0 else 0.0

    def _airtime(self, size: int) -> float:
        return (size + self.HEADER_BYTES) * 8.0 / self.bps

    def _book(self, msg: Message) -> float:
        """Reserves airtime on the shared medium; returns when the message has fully left the sender."""
        dur = self._airtime(msg.size)
        self.busy_time += dur
        if self.qos and msg.kind in CONTROL_KINDS:
            return self.sim.now + dur  # priority class: does not wait behind bulk data
        start = max(self.sim.now, self._free_at)
        self._free_at = start + dur
        return self._free_at

    def send(self, src: Node, dst: Node, msg: Message) -> None:
        if not src.alive:
            return
        m = self.sim.metrics
        m.messages_sent += 1
        m.bytes_sent += msg.size + self.HEADER_BYTES
        m.energy_j += src.profile.energy_joules(0, msg.size + self.HEADER_BYTES)
        msg.t_sent = self.sim.now
        done = self._book(msg)
        if self.sim.rng.random() < self.loss:
            m.messages_lost += 1
            return
        self.sim.schedule(done - self.sim.now + self.latency, self._deliver, src.id, dst, msg)

    def broadcast(self, src: Node, msg: Message) -> None:
        """One transmission, heard by every other live node (loss is rolled per receiver)."""
        if not src.alive:
            return
        m = self.sim.metrics
        m.messages_sent += 1
        m.bytes_sent += msg.size + self.HEADER_BYTES
        m.energy_j += src.profile.energy_joules(0, msg.size + self.HEADER_BYTES)
        msg.t_sent = self.sim.now
        done = self._book(msg)
        for dst in self.sim.nodes:
            if dst is src:
                continue
            if self.sim.rng.random() < self.loss:
                m.messages_lost += 1
                continue
            self.sim.schedule(done - self.sim.now + self.latency, self._deliver, src.id, dst, msg)

    def _deliver(self, src_id: int, dst: Node, msg: Message) -> None:
        if dst.alive:
            dst.receive(src_id, msg)
