"""op_mixed_swarm - Created in the swarmBalancer studio on 2026-10-11.

A preset like any other: run it with `python <this file>` (-h for options). THEATRE holds the 3D
view settings of the studio; the battery ignores it.
"""

import _bootstrap  # noqa: F401

from swarm.bench import BrainSpec, Camera, Check, Faults, Link, Scenario, Swarm
from swarm.sdk import use_package

use_package("examples/brains/load_aware_stripe")

SCENARIO = Scenario(
    name="op_mixed_swarm",
    description="Mixed A72/A53/M4 swarm on a desert survey: does the load-aware stripe keep the MCUs out of trouble when one of them is lost?",
    swarm=Swarm(nodes=("drone-a72", "drone-a53", "drone-a53", "mcu-m4", "mcu-m4", "mcu-m4")),
    camera=Camera(fps=30.0),
    link=Link(bps=6000000.0, latency_s=0.003),
    brain=BrainSpec(kind="oracle", macs=3_200_000, accuracy=0.9),
    faults=Faults(kill=((3, 8.0),)),
    duration=20.0,
    seed=0,
    strategies=("local", "striped", "load_aware"),
    tags=("heterogeneous", "sdk-example"),
    checks=(
        Check("load_aware", "busy_max", "<=", 0.2, note="speed-weighted shares keep every CPU light"),
        Check("load_aware", "frames_rejected", "==", 0.0),
        Check("striped", "busy_max", ">=", 0.95, note="round-robin overloads the M4s"),
        Check("load_aware", "objects_reported_ps", ">=", 1.0, ratio=True),
    ),
)

THEATRE = {"terrain": "desert", "formation": "wedge", "route": "lawnmower", "altitude_m": 0.0, "spacing_m": 0.0, "speed_mps": 0.0, "area_m": 1800.0}

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
