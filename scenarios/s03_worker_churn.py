"""s03 - Workers flap: two drones drop out and rejoin at different times.

Models intermittent links, battery swaps or a drone ducking behind a building. Node 2 dies twice,
node 3 once; all come back. Answers: does the swarm re-integrate members cleanly (rotation grows
again, no spurious leader changes) and how much work is lost in the detection windows?
"""

import _bootstrap  # noqa: F401

from swarm.bench import Camera, Check, Faults, Link, Scenario, Swarm

SCENARIO = Scenario(
    name="s03_worker_churn",
    description="Reference swarm; workers 2 and 3 go offline and come back (3 outages in 30 s).",
    swarm=Swarm(leader="drone-a72", worker="drone-a53", size=5),
    camera=Camera(fps=10, size=128, p_object=0.3),
    link=Link(loss=0.01),
    faults=Faults(kill=((2, 5.0), (3, 8.0), (2, 22.0)), revive=((2, 12.0), (3, 20.0), (2, 27.0))),
    duration=30.0,
    checks=(
        Check("*", "leader_changes", "==", 0, note="worker outages must not disturb the leader"),
        Check("*", "failures_detected", ">=", 12, note="3 outages x 4 survivors each notice"),
        Check("*", "detect_s", "<=", 2.5),
        Check("*", "work_retained", ">=", 0.8),
        Check("striped", "frames_lost_to_fault", "<=", 90,
              note="frames sent to a dead worker until it is suspected: ~3 x 1.7 s x 10 fps"),
    ),
    tags=("survivability", "churn"),
)

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
