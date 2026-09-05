"""s11 - A mixed swarm: one A72, two A53s and three Cortex-M4 MCUs, with a 3.2 MMAC brain.

The MCUs need 38 ms per frame; at 30 fps they are asked for more than they can do. Answers: which
design copes with unequal members? Central puts everything on the A72 and is fine; independent
thinkers overload the M4s; the naive round-robin stripe hands the M4s the same share as the A72
and overloads them too - a load-aware stripe (heartbeats already carry each node's load) is the
obvious next strategy to add and this scenario is its yardstick.
"""

import _bootstrap  # noqa: F401

from swarm.bench import BrainSpec, Camera, Check, Link, Scenario, Swarm

SCENARIO = Scenario(
    name="s11_heterogeneous",
    description="a72 + 2x a53 + 3x mcu-m4 with a 3.2 MMAC brain at 30 fps on a 6 Mbps link.",
    swarm=Swarm(nodes=("drone-a72", "drone-a53", "drone-a53", "mcu-m4", "mcu-m4", "mcu-m4")),
    camera=Camera(fps=30, size=128, p_object=0.3),
    link=Link(bps=6e6, latency_s=0.003, loss=0.01),
    brain=BrainSpec(kind="oracle", macs=3_200_000, accuracy=0.9, input_size=32),
    duration=20.0,
    checks=(
        Check("central/downscale", "frames_rejected", "==", 0, note="the A72 absorbs 180 fps with ease"),
        Check("central/downscale", "busy_max", "<=", 0.3),
        Check("local", "frames_rejected", ">", 0, note="an M4 cannot run 30 fps x 38 ms on its own"),
        Check("local", "busy_max", ">=", 0.95),
        Check("striped", "busy_max", ">=", 0.95, note="round-robin gives the M4s the same share as the A72"),
    ),
    tags=("heterogeneous", "compute-bound"),
)

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
