"""s01 - Reference swarm, nothing goes wrong.

The yardstick every other scenario is a variation of: a Cortex-A72 command drone with four
Cortex-A53 drones, 128 px cameras at 10 fps, one shared 6 Mbps link, targets in 30 % of frames.
Answers: with everything working, what does each design cost in compute, radio and latency to
deliver the same detections as five independent thinkers?
"""

import _bootstrap  # noqa: F401

from swarm.bench import Camera, Check, Link, Scenario, Swarm

SCENARIO = Scenario(
    name="s01_reference",
    description="Reference swarm (a72 + 4x a53, 128 px @ 10 fps, 6 Mbps), no faults.",
    swarm=Swarm(leader="drone-a72", worker="drone-a53", size=5),
    camera=Camera(fps=10, size=128, p_object=0.3),
    link=Link(loss=0.01),
    duration=30.0,
    checks=(
        Check("*", "recall", ">=", 0.8, note="every design must report most of what is in view"),
        Check("*", "leader_changes", "==", 0, note="no false leader changes on a healthy link"),
        Check("*", "frames_dropped", "<=", 30, note="only radio loss and in-flight frames may be lost"),
        Check("central/none", "channel_util", ">=", 0.5, note="raw streaming must show up as a heavy link"),
        Check("central/downscale", "channel_util", "<=", 0.1, note="pre-processing shrinks the payload ~16x"),
        Check("striped", "cameras_on", "==", 1, note="striped runs a single camera"),
        Check("striped", "gmac_ps", "<=", 1.1, ratio=True, note="same frames/s -> about the same compute as local"),
    ),
    tags=("reference",),
)

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
