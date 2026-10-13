"""Pincer attack - an attacker brain that coordinates over the radio (example arena policy).

The swarm splits in two by each drone's role code. Both halves drop under the terrain mask, where the
interceptors see them only up close, and each flies to its own flank of the corridor and waits there,
slowly, beyond the interceptors' sensor range, radioing whether it is in position. When most of the friends
it hears are in position - or one of them calls "go", or time is running out - it calls "go" too and
sprints for the line. The call floods through the mesh hop by hop, so both flanks hit the wall at nearly
the same moment and stretch it as wide as possible. Close to an interceptor, a drone breaks hard: to the
side, and away from it in height.

Radio channels it uses (obs.msg is the mean of what each drone heard from its friends):
    msg[..., 0]  "I am in position"      (1 or 0)
    msg[..., 1]  "go"                    (1 or 0)
"""

import numpy as np


class PincerAttack:
    role = "attacker"
    title = "Pincer (radio-coordinated)"
    summary = ("Splits into two flanks that sneak in under the terrain mask, form up out of sensor range, agree over "
               "the radio when both are ready, then strike both ends of the line at once.")
    comms = True
    macs = 400

    def reset(self, batch, seed):
        self.go = np.zeros((batch, 0), bool)  # sized on the first act(); remembers who has called "go"

    def act(self, obs):
        cfg = obs.cfg
        B, A = obs.alive.shape
        if self.go.shape != (B, A):
            self.go = np.zeros((B, A), bool)
        x, y, z = obs.pos[..., 0], obs.pos[..., 1], obs.pos[..., 2]
        left = np.broadcast_to(obs.role() < 0, (B, A))
        lane = np.where(left, -0.4, 0.4) * cfg.width
        altitude = np.full((B, A), 0.6 * cfg.low_altitude)  # under the terrain mask: seen only up close
        hold_y = 0.62 * cfg.depth                             # form up beyond the interceptors' sensor range
        ready = (np.abs(x - lane) < 100) & (y < hold_y + 150) & (np.abs(z - altitude) < 30)
        self.go |= (obs.msg[..., 0] > 0.6) | (obs.msg[..., 1] > 0.05) | obs.chased | (obs.t > 0.45 * cfg.duration)
        go = self.go

        # forming up: head for the flank lane, fly to the hold line, then loiter
        to_lane = np.clip((lane - x) / 150.0, -1.0, 1.0)
        heading = np.where(go, 0.25 * to_lane, 1.1 * to_lane)
        speed = np.where(go, 1.0, np.where(y > hold_y, 0.8, 0.25))

        # break away from an interceptor about to hit: sideways, and dive or climb away from it
        threat = obs.slot_mask & (obs.slot_closing > 0.5) & (obs.slot_dist < 100.0)
        dist = np.where(threat, obs.slot_dist, np.inf)
        k = dist.argmin(-1)[..., None, None]
        rel = np.take_along_axis(obs.slot_rel, k, -2)[..., 0, :]  # the closest threat, relative to me
        dodge = np.isfinite(dist.min(-1))
        heading = np.where(dodge, np.where(rel[..., 0] > 0, -1.25, 1.25), heading)
        altitude = np.where(dodge, z - 60.0 * np.sign(rel[..., 2]), altitude)

        msg = np.zeros((B, A, 2))
        msg[..., 0] = ready
        msg[..., 1] = go
        return {"heading": heading, "speed": speed, "altitude": altitude, "msg": msg}
