"""Workload-distribution strategies.

Three designs of "how does a swarm process what its cameras see", all built
on the same nodes, channel and membership so they can be compared on equal
footing:

``local`` (baseline - "everyone is an independent thinker")
    Every drone runs the whole brain on its own frames and reports only
    detections to the leader. Max compute duplication, min radio.

``central`` (design 1 - "central command drone")
    Every drone streams its camera to the leader, which runs the brain for
    everybody and broadcasts a decision. ``preprocess`` controls how much the
    small drones do before transmitting:

    * ``none``      - raw camera frame (``cam_size^2`` bytes)
    * ``downscale`` - resize to the brain input on the drone (``input_bytes``)
    * ``features``  - also run the conv frontend on the drone and send the
      int8 feature map; the leader only runs the dense head (split inference)

``striped`` (design 2 - "front drone sees, swarm thinks")
    Only the leader's camera is on. Frame *i* goes to live member
    ``i mod n`` (round-robin; the leader can take a share too). A worker
    replies only when it found something useful, otherwise stays silent.
    Cameras of the workers are off - they rest and listen.

Fault tolerance is identical for all three: heartbeats -> suspicion ->
``on_member_change`` -> the rotation / leader is recomputed from the same
deterministic rule on every node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarm.devices import DeviceProfile
from swarm.train.dataset import FramePool, FrameStream

from .brains import Brain, Frame, downscale_macs
from .channel import Channel, Message
from .core import Node, Simulator
from .membership import Membership

RESULT_BYTES = 16   # frame id, class, confidence, position hint
CMD_BYTES = 16      # leader's decision broadcast


@dataclass
class SwarmConfig:
    strategy: str = "central"            # local | central | striped
    preprocess: str = "none"             # none | downscale | features (central only)
    fps: float = 10.0                    # per active camera
    cam_size: int = 128                  # camera resolution (pixels per side, 8-bit mono)
    include_leader: bool = True          # striped: leader also takes a share of frames
    confidence_threshold: float = 0.5    # striped/local: report only if conf >= this and class != 0
    hb_interval: float = 0.5
    missed_beats: int = 3
    queue_limit: int = 4
    p_object: float = 0.3                # fraction of frames that contain something
    frame_size: int = 32                 # rendered frame size fed to the brain
    extra: dict[str, Any] = field(default_factory=dict)


class SwarmNode(Node):
    """Common machinery: camera, brain, membership. Strategies subclass this."""

    def __init__(self, profile: DeviceProfile, brain: Brain, cfg: SwarmConfig, channel: Channel, seed: int,
                 name: str | None = None, pool: FramePool | None = None):
        super().__init__(profile, name, queue_limit=cfg.queue_limit)
        self.brain = brain
        self.cfg = cfg
        self.channel = channel
        self.seed = seed
        self.pool = pool
        self.camera: FrameStream | None = None
        self.camera_on = False
        self.membership = Membership(self, channel, cfg.hb_interval, cfg.missed_beats)
        self.membership.on_member_change = self.on_member_change
        self.frames_captured = 0
        self._camera_epoch = 0  # bumps every time the camera (re)starts so stale ticks stop
        self._was_leader = False

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.camera = FrameStream(self.cfg.frame_size, seed=self.seed * 1000 + self.id, p_object=self.cfg.p_object,
                                  pool=self.pool)
        self.membership.start()
        self._was_leader = self.membership.is_leader()
        if self._was_leader:
            self.sim.event("leader", self.id)
        self.on_start()

    def on_start(self) -> None:
        pass

    def on_killed(self) -> None:
        self.camera_on = False
        self._was_leader = False

    def on_revived(self) -> None:
        self.membership.reset_after_revive()
        self._track_leadership()
        self.on_start()

    def on_member_change(self, node_id: int, alive: bool) -> None:
        self.sim.metrics.rebalances += 1
        self._track_leadership()
        self.rebalance(node_id, alive)

    def _track_leadership(self) -> None:
        now_leader = self.membership.is_leader()
        if now_leader and not self._was_leader:
            self.sim.metrics.leader_changes += 1
            self.sim.event("leader", self.id)
            self.sim.log(f"node {self.id} ({self.name}) is now LEADER")
        self._was_leader = now_leader

    def rebalance(self, node_id: int, alive: bool) -> None:
        pass

    # ------------------------------------------------------------ camera
    def start_camera(self) -> None:
        if self.camera_on:
            return
        self.camera_on = True
        self._camera_epoch += 1
        self.sim.schedule(0.0, self._tick, self._camera_epoch)

    def stop_camera(self) -> None:
        self.camera_on = False

    def _tick(self, epoch: int) -> None:
        if not self.alive or not self.camera_on or epoch != self._camera_epoch:
            return
        self.sim.schedule(1.0 / self.cfg.fps, self._tick, epoch)
        self.on_capture(self.capture())

    def capture(self) -> Frame:
        assert self.camera is not None
        image, label = self.camera.next()
        self.frames_captured += 1
        self.sim.metrics.frames_captured += 1
        if label != 0:
            self.sim.metrics.objects_captured += 1
        return Frame(id=self.sim.metrics.frames_captured, t_capture=self.sim.now, src=self.id, image=image, label=label)

    def on_capture(self, frame: Frame) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------ brain
    def try_compute(self, macs: int, cb, *args) -> bool:
        """``compute`` that records a rejected frame when this node's queue is full."""
        ok = self.compute(macs, cb, *args)
        if not ok:
            self.sim.metrics.frames_rejected += 1
            self.sim.log(f"node {self.id} queue full - rejected work")
        return ok

    def run_brain(self, macs: int, frame: Frame, then, *args) -> bool:
        """Books ``macs`` on this node's CPU and calls ``then(frame, pred, conf, *args)`` when done."""
        return self.try_compute(macs, self._brain_done, frame, then, args)

    def _brain_done(self, frame: Frame, then, args) -> None:
        pred, conf = self.brain.infer(frame)
        m = self.sim.metrics
        m.frames_processed += 1
        if pred == frame.label:
            m.correct += 1
        if pred != 0:
            m.detections += 1
        then(frame, pred, conf, *args)

    def useful(self, pred: int, conf: float) -> bool:
        return pred != 0 and conf >= self.cfg.confidence_threshold

    def report_to_leader(self, frame: Frame, pred: int, conf: float) -> None:
        leader = self.membership.leader()
        if leader == self.id:
            self.leader_learns(frame, pred, conf)
        else:
            # the label rides along for metrics only (ground truth is never used for decisions)
            self.channel.send(self, self.sim.nodes[leader],
                              Message("result", payload=(frame.id, pred, conf, frame.t_capture, frame.label),
                                      size=RESULT_BYTES))

    def leader_learns(self, frame: Frame | None, pred: int, conf: float, t_capture: float | None = None,
                      label: int | None = None) -> None:
        """The leader now knows the result for a frame - the moment the swarm can act on it."""
        m = self.sim.metrics
        if frame is not None:
            t_capture, label = frame.t_capture, frame.label
        m.learn_times.append(self.sim.now)
        if pred != 0:
            # latency is measured on detections only: "how long until the leader knows something is there";
            # counting every frame would favour designs whose leader inspects its own frames locally
            m.latencies.append(self.sim.now - t_capture)
            m.detections_reported += 1
            if label != 0:
                m.objects_reported += 1
            else:
                m.false_alarms_reported += 1

    # ------------------------------------------------------------ messaging
    def receive(self, src: int, msg: Message) -> None:
        if self.membership.handle(src, msg):
            return
        handler = getattr(self, f"on_{msg.kind}", None)
        if handler:
            handler(src, msg)

    def on_result(self, src: int, msg: Message) -> None:
        _fid, pred, conf, t_capture, label = msg.payload
        self.leader_learns(None, pred, conf, t_capture, label)

    def on_cmd(self, src: int, msg: Message) -> None:
        pass  # followers act on the leader's decision; nothing to model yet


# ----------------------------------------------------------------------------
# local: independent thinkers
# ----------------------------------------------------------------------------
class LocalNode(SwarmNode):
    def on_start(self) -> None:
        self.start_camera()

    def on_capture(self, frame: Frame) -> None:
        # every design has to shrink the camera frame to the brain input; here each drone does its own
        self.run_brain(self.brain.macs + downscale_macs(self.cfg.cam_size ** 2), frame, self._done)

    def _done(self, frame: Frame, pred: int, conf: float) -> None:
        if self.useful(pred, conf):
            self.report_to_leader(frame, pred, conf)
        elif self.membership.is_leader():
            self.leader_learns(frame, pred, conf)


# ----------------------------------------------------------------------------
# central: one command drone computes for everyone
# ----------------------------------------------------------------------------
class CentralNode(SwarmNode):
    def on_start(self) -> None:
        self.start_camera()

    def on_capture(self, frame: Frame) -> None:
        pre = self.cfg.preprocess
        cam_pixels = self.cfg.cam_size ** 2
        if pre == "none":
            self._ship(frame, size=cam_pixels, kind="frame")
        elif pre == "downscale":
            self.try_compute(downscale_macs(cam_pixels), self._ship, frame, self.brain.input_bytes, "frame")
        elif pre == "features":
            # downscale + conv frontend on the drone; only the dense head runs on the leader
            self.try_compute(downscale_macs(cam_pixels) + self.brain.frontend_macs, self._ship, frame,
                             self.brain.feature_bytes, "features")
        else:
            raise ValueError(f"unknown preprocess {pre!r}")

    def _ship(self, frame: Frame, size: int, kind: str) -> None:
        leader = self.membership.leader()
        if leader == self.id:
            self._process(frame, kind)
        else:
            self.channel.send(self, self.sim.nodes[leader], Message(kind, payload=frame, size=size))

    def _leader_macs(self, kind: str) -> int:
        if kind == "features":
            return self.brain.head_macs
        macs = self.brain.macs
        if self.cfg.preprocess == "none":
            macs += downscale_macs(self.cfg.cam_size ** 2)  # leader must downscale the raw frame itself
        return macs

    def _process(self, frame: Frame, kind: str) -> None:
        self.run_brain(self._leader_macs(kind), frame, self._decided)

    def _decided(self, frame: Frame, pred: int, conf: float) -> None:
        self.leader_learns(frame, pred, conf)
        # central decision goes back to everyone
        self.channel.broadcast(self, Message("cmd", payload=(frame.id, pred), size=CMD_BYTES))

    def on_frame(self, src: int, msg: Message) -> None:
        # Whoever receives a frame processes it - robust to briefly divergent leader views.
        self._process(msg.payload, "frame")

    def on_features(self, src: int, msg: Message) -> None:
        self._process(msg.payload, "features")


# ----------------------------------------------------------------------------
# striped: leader's camera, everyone's brain
# ----------------------------------------------------------------------------
class StripedNode(SwarmNode):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._rr = 0

    def on_start(self) -> None:
        self._ensure_role()

    def rebalance(self, node_id: int, alive: bool) -> None:
        self._ensure_role()
        self.sim.log(f"node {self.id}: rotation now {self._rotation()} (leader {self.membership.leader()})")

    def _ensure_role(self) -> None:
        if self.membership.is_leader():
            if not self.camera_on:
                self.sim.log(f"node {self.id} leads the stripe - camera ON")
            self.start_camera()
        else:
            if self.camera_on:
                self.sim.log(f"node {self.id} demoted - camera OFF")
            self.stop_camera()

    def _rotation(self) -> list[int]:
        members = self.membership.live_members()
        if not self.cfg.include_leader and len(members) > 1:
            members = [m for m in members if m != self.id]
        return members

    def on_capture(self, frame: Frame) -> None:
        # the leader always downsamples (its own camera) before shipping
        self.try_compute(downscale_macs(self.cfg.cam_size ** 2), self._dispatch, frame)

    def _dispatch(self, frame: Frame) -> None:
        if not self.membership.is_leader():
            return  # demoted while pre-processing; the new leader's camera has taken over
        rot = self._rotation()
        target = rot[self._rr % len(rot)]
        self._rr += 1
        if target == self.id:
            self.run_brain(self.brain.macs, frame, self._worker_done)
        else:
            self.channel.send(self, self.sim.nodes[target], Message("frame", payload=frame, size=self.brain.input_bytes))

    def on_frame(self, src: int, msg: Message) -> None:
        self.run_brain(self.brain.macs, msg.payload, self._worker_done)

    def _worker_done(self, frame: Frame, pred: int, conf: float) -> None:
        if self.useful(pred, conf):
            self.report_to_leader(frame, pred, conf)
            if self.membership.is_leader():
                self.channel.broadcast(self, Message("cmd", payload=(frame.id, pred), size=CMD_BYTES))
        elif self.membership.is_leader():
            self.leader_learns(frame, pred, conf)
        # else: nothing useful -> stay silent, save the radio

    def on_result(self, src: int, msg: Message) -> None:
        super().on_result(src, msg)
        fid, pred = msg.payload[0], msg.payload[1]
        self.channel.broadcast(self, Message("cmd", payload=(fid, pred), size=CMD_BYTES))


NODE_CLASSES = {"local": LocalNode, "central": CentralNode, "striped": StripedNode}


def build_swarm(sim: Simulator, cfg: SwarmConfig, profiles: list[DeviceProfile], brain: Brain, channel_bps: float,
                channel_latency: float, channel_loss: float, seed: int = 0,
                pool: FramePool | None = None, qos: bool = True) -> tuple[Channel, list[SwarmNode]]:
    channel = Channel(sim, channel_bps, channel_latency, channel_loss, qos=qos)
    cls = NODE_CLASSES[cfg.strategy]
    nodes = [sim.add_node(cls(p, brain, cfg, channel, seed, name=f"{p.name}#{i}", pool=pool))
             for i, p in enumerate(profiles)]
    for n in nodes:
        n.start()
    return channel, nodes
