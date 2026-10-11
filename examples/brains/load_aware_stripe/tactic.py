"""Load-aware stripe - one camera; frames are shared in proportion to each member's speed.

The built-in "striped" design deals frames round-robin, so on a mixed swarm (scenario s11: one A72, two
A53s, three Cortex-M4s) every MCU gets the same share as the A72 and drowns. This tactic deals them with a
smooth weighted round-robin: each member's weight is the brain MACs per second its device sustains, so
an A72 takes ~140x the frames of a Cortex-M4 and every member ends up about equally busy. It also listens
to backpressure: a member whose heartbeat reports a (nearly) full queue is skipped until it drains.
Everything else - who leads, whose camera runs, silence unless something was found - is the striped design.
"""

from swarm.sdk import Tactic

BUSY = 0.75  # queue fill (from heartbeats) above which a member gets no new frames


class LoadAwareStripe(Tactic):
    strategy = "load_aware"
    summary = "one camera; frames dealt in proportion to each member's speed, skipping members with full queues"
    camera_mode = "single"

    def on_start(self):
        self._credit = {}  # smooth weighted round-robin state, per member
        self._ensure_role()

    def rebalance(self, node_id, alive):
        self._credit.pop(node_id, None)  # gone: forget it; back: it starts with an empty queue
        self._ensure_role()

    def _ensure_role(self):
        if self.is_leader:
            self.start_camera()
        else:
            self.stop_camera()

    def on_capture(self, frame):
        self.preprocess(frame, self._dispatch)  # the camera drone pays the downscale, as in every design

    def _dispatch(self, frame):
        if not self.is_leader:
            return
        target = self._pick()
        if target == self.id:
            self.run_brain(self.brain.macs, frame, self._done)
        else:
            self.send_frame(target, frame)

    def _pick(self):
        members = self.live_members()
        ready = [m for m in members if self.peer_load(m) < BUSY] or members
        weights = {m: self.node_speed(m) for m in ready}
        total = sum(weights.values())
        for m, w in weights.items():
            self._credit[m] = self._credit.get(m, 0.0) + w
        best = max(ready, key=lambda m: (self._credit[m], -m))
        self._credit[best] -= total
        return best

    def on_frame(self, src, msg):
        self.run_brain(self.brain.macs, msg.payload, self._done)

    def _done(self, frame, pred, conf):
        if self.useful(pred, conf):
            self.report_to_leader(frame, pred, conf)
        elif self.is_leader:
            self.leader_learns(frame, pred, conf)
