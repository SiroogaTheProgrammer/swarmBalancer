"""s02 - The command drone dies mid-mission and never comes back.

The most important survivability test: the node everything depends on disappears at t = 10 s.
Answers: how long until the swarm notices, agrees on a new leader and delivers results again,
and how much of the mission is retained compared with the same run without the failure?
Central designs lose the brain, striped loses the only camera, local loses one of five cameras.
"""

import _bootstrap  # noqa: F401

from swarm.bench import Camera, Check, Faults, Link, Scenario, Swarm

SCENARIO = Scenario(
    name="s02_leader_loss",
    description="Reference swarm; the leader is killed at t=10 s and stays dead.",
    swarm=Swarm(leader="drone-a72", worker="drone-a53", size=5),
    camera=Camera(fps=10, size=128, p_object=0.3),
    link=Link(loss=0.01, hb_interval=0.5, missed_beats=3),
    faults=Faults(kill=(("leader", 10.0),)),
    duration=30.0,
    checks=(
        Check("*", "leader_changes", "==", 1, note="exactly one re-election"),
        Check("*", "detect_s", "<=", 2.5, note="3 missed heartbeats of 0.5 s plus stagger"),
        Check("*", "reelect_s", "<=", 2.5, note="election is a pure function of the membership view"),
        Check("*", "recover_s", "<=", 4.0, note="results reach the new leader shortly after election"),
        Check("*", "work_retained", ">=", 0.6),
        Check("striped", "work_retained", ">=", 0.85, note="the camera role moves; only the blackout is lost"),
        Check("striped", "recall_retained", ">=", 0.8),
    ),
    tags=("survivability", "leader"),
)

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
