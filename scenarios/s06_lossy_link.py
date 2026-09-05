"""s06 - A lossy link (15 % of messages vanish): does the failure detector cry wolf?

Two variants of the reference swarm on a bad link. With 3 missed heartbeats as the threshold
the swarm falsely suspects healthy members (and briefly runs two cameras / two leaders); with
5 it does not, at the price of slower detection of real failures (2.5 s instead of 1.5 s).
Answers: how should the heartbeat timeout be tuned for this radio?
"""

import _bootstrap  # noqa: F401

from swarm.bench import Camera, Check, Link, Scenario, Swarm

_common = dict(
    swarm=Swarm(leader="drone-a72", worker="drone-a53", size=5),
    camera=Camera(fps=10, size=128, p_object=0.3),
    duration=60.0,
    tags=("radio", "membership"),
)

SCENARIOS = [
    Scenario(
        name="s06_lossy_link_hb3",
        description="15 % message loss, suspect after 3 missed heartbeats -> false suspicions expected.",
        link=Link(loss=0.15, hb_interval=0.5, missed_beats=3),
        checks=(
            Check("*", "failures_detected", ">=", 1, note="a too-tight timeout produces false suspicions"),
            Check("*", "recall", ">=", 0.7),
        ),
        **_common,
    ),
    Scenario(
        name="s06_lossy_link_hb5",
        description="15 % message loss, suspect after 5 missed heartbeats -> stable membership.",
        link=Link(loss=0.15, hb_interval=0.5, missed_beats=5),
        checks=(
            Check("*", "failures_detected", "==", 0, note="no false suspicions"),
            Check("*", "leader_changes", "==", 0),
            Check("*", "recall", ">=", 0.7),
        ),
        **_common,
    ),
]

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(max(main(sc) for sc in SCENARIOS))
