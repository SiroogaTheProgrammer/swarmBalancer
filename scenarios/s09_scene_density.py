"""s09 - Camera pointed at a busy scene (80 % of frames contain something) vs. an empty one (5 %).

Striping's trick is that workers stay silent when a frame is empty. Answers: how much of the
radio advantage survives when almost every frame is a detection, and how do the designs compare
when the sky is empty and the swarm is mostly burning compute on nothing?
"""

import _bootstrap  # noqa: F401

from swarm.bench import Camera, Check, Link, Scenario, Swarm

_common = dict(
    swarm=Swarm(leader="drone-a72", worker="drone-a53", size=5),
    link=Link(loss=0.01),
    duration=30.0,
    tags=("scene",),
)

SCENARIOS = [
    Scenario(
        name="s09_busy_scene",
        description="Targets in 80 % of frames: nearly every worker result must be transmitted.",
        camera=Camera(fps=10, size=128, p_object=0.8),
        checks=(
            Check("*", "recall", ">=", 0.8),
            Check("local", "kbps", ">=", 5.0, note="many detections to report even for independent thinkers"),
            Check("striped", "kbps", "<=", 450.0, note="1 KB frames out + 16 B results and commands back"),
        ),
        **_common,
    ),
    Scenario(
        name="s09_empty_scene",
        description="Targets in 5 % of frames: the swarm mostly confirms 'nothing'.",
        camera=Camera(fps=10, size=128, p_object=0.05),
        checks=(
            Check("local", "kbps", "<=", 6.0, note="heartbeats plus a few detections"),
            Check("striped", "kbps", ">=", 20.0, ratio=True,
                  note="striping still ships every frame to a worker; only the return path is content-dependent"),
            Check("central/downscale", "kbps", "<=", 500.0),
        ),
        **_common,
    ),
]

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(max(main(sc) for sc in SCENARIOS))
