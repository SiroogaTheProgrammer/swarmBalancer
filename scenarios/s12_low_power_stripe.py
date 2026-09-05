"""s12 - Low-power stripe: one camera at the normal frame rate, the other four drones rest.

Instead of matching the throughput of five cameras, the stripe runs the leader's camera at the
ordinary 10 fps and every worker sees only every fifth frame. Answers: how much compute, radio and
energy does the swarm save when it accepts one viewpoint at 1/5 of the frame budget, and what does
it give up in observations? Note the energy row: with these radio profiles, transmitting a 1 KB
frame costs about three times the energy of classifying it, so striping saves compute but only
part of the energy - the case for pre-processing on the sensing node, or a cheaper radio.
"""

import _bootstrap  # noqa: F401

from swarm.bench import Camera, Check, Link, Scenario, Swarm

SCENARIO = Scenario(
    name="s12_low_power_stripe",
    description="Reference swarm; the stripe runs its single camera at 10 fps instead of 5x10.",
    swarm=Swarm(leader="drone-a72", worker="drone-a53", size=5),
    camera=Camera(fps=10, size=128, p_object=0.3, striped_fps=None),
    link=Link(loss=0.01),
    strategies=("local", "central/downscale", "striped"),
    duration=30.0,
    checks=(
        Check("striped", "cameras_on", "==", 1),
        Check("striped", "gmac_ps", "<=", 0.25, ratio=True, note="one fifth of the frames -> one fifth of the compute"),
        Check("striped", "energy_j", "<=", 0.8, ratio=True,
              note="only ~30% saved: shipping a 1 KB frame costs ~3x the energy of classifying it on these radios"),
        Check("striped", "objects_reported_ps", "<=", 0.3, ratio=True, note="...and one fifth of the observations"),
        Check("striped", "recall", ">=", 0.8, note="but everything the one camera sees is still found"),
    ),
    tags=("power",),
)

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
