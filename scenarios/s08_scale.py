"""s08 - Scale: 3, 6 and 12 identical Cortex-A53 drones on a 6 Mbps link.

Same brain, same cameras, same link; only the swarm grows. Answers: how do compute, radio and
latency scale with N for each design? Central radio load grows with N; local compute grows with
N; the stripe keeps a single camera and a fixed per-node load. Run the three and compare rows.
"""

import _bootstrap  # noqa: F401

from swarm.bench import Camera, Check, Link, Scenario, Swarm


def _at(n: int) -> Scenario:
    return Scenario(
        name=f"s08_scale_{n:02d}",
        description=f"{n} identical drone-a53 nodes, 128 px @ 10 fps, 6 Mbps.",
        swarm=Swarm(nodes=("drone-a53",) * n),
        camera=Camera(fps=10, size=128, p_object=0.3),
        link=Link(bps=6e6, latency_s=0.002, loss=0.01),
        duration=20.0,
        checks=(
            Check("*", "leader_changes", "==", 0),
            Check("striped", "cameras_on", "==", 1),
            Check("striped", "busy_max", "<=", 1.6, ratio=True,
                  note="the stripe leader also downsamples every frame, a duty that grows with N (1.1x at 3, 1.4x at 12)"),
        ) + ((Check("central/none", "channel_util", ">=", 1.0, note="12 raw streams oversubscribe 6 Mbps"),) if n >= 12 else ()),
        tags=("scale",),
    )


SCENARIOS = [_at(3), _at(6), _at(12)]

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(max(main(sc) for sc in SCENARIOS))
