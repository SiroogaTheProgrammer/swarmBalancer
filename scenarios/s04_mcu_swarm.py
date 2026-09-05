"""s04 - A cheap swarm of six Cortex-M4 microcontrollers with a heavier brain.

No powerful command drone at all: every node is a 168 MHz MCU on a 250 kbps radio, and the brain
is a hypothetical 6.4 MMAC/frame network (76 ms per frame on an M4). Answers: when nobody can
compute for everybody, which design still keeps up? Central saturates the leader's CPU; raw
streaming is impossible on this radio; striping spreads the load evenly.
"""

import _bootstrap  # noqa: F401

from swarm.bench import BrainSpec, Camera, Check, Link, Scenario, Swarm

SCENARIO = Scenario(
    name="s04_mcu_swarm",
    description="6x mcu-m4 on a 250 kbps radio running a 6.4 MMAC brain at 3 fps per camera (compute-bound).",
    swarm=Swarm(nodes=("mcu-m4",) * 6),
    camera=Camera(fps=3, size=128, p_object=0.3),
    link=Link(loss=0.01),
    brain=BrainSpec(kind="oracle", macs=6_400_000, accuracy=0.9, input_size=32),
    duration=40.0,
    checks=(
        Check("central/downscale", "busy_max", ">=", 0.95, note="the single leader CPU is saturated (18 fps x 76 ms)"),
        Check("central/downscale", "frames_rejected", ">", 0, note="...and has to refuse frames"),
        Check("central/none", "channel_util", ">=", 2.0, note="16 KB raw frames cannot fit a 250 kbps link"),
        Check("local", "busy_max", "<=", 0.5),
        Check("striped", "busy_max", "<=", 0.5, note="the same total work spread over 6 MCUs"),
        Check("striped", "frames_rejected", "==", 0),
        Check("striped", "recall", ">=", 0.8),
    ),
    tags=("compute-bound", "mcu"),
)

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
