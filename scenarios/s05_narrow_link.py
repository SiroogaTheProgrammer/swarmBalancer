"""s05 - A narrow long-range link (100 kbps, 20 ms).

LoRa-class radio between drones that are far apart, so cameras run at 2 fps. Answers: which
designs are radio-bound? Raw streaming needs 13x the link; downscaled frames and the stripe use
~85 % of it; independent thinkers need almost nothing.
"""

import _bootstrap  # noqa: F401

from swarm.bench import Camera, Check, Link, Scenario, Swarm

SCENARIO = Scenario(
    name="s05_narrow_link",
    description="Reference drones on a 100 kbps / 20 ms link at 2 fps (radio-bound).",
    swarm=Swarm(leader="drone-a72", worker="drone-a53", size=5),
    camera=Camera(fps=2, size=128, p_object=0.3),
    link=Link(bps=100e3, latency_s=0.02, loss=0.02),
    duration=60.0,
    checks=(
        Check("central/none", "channel_util", ">=", 5.0, note="hopelessly oversubscribed"),
        Check("central/none", "latency_p95_ms", ">=", 5000,
              note="remote frames queue for tens of seconds (p50 hides it: the leader's own frames cost 0 ms)"),
        Check("central/none", "recall", "<=", 0.3, note="most frames never arrive in time"),
        Check("central/downscale", "channel_util", "<=", 0.95),
        Check("striped", "channel_util", "<=", 0.95),
        Check("local", "kbps", "<=", 5.0, note="detections only"),
        Check("local", "recall", ">=", 0.8),
    ),
    tags=("radio-bound",),
)

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
