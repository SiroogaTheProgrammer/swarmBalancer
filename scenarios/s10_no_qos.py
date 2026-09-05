"""s10 - Does the shared radio starve the heartbeats? (control-plane QoS off)

Same as s01 on a 1 Mbps link, but heartbeats, results and commands have to queue behind bulk frame
data like everything else. Raw streaming oversubscribes the link 5x, so the queue grows without
bound and heartbeats arrive 2.6 s apart - longer than the 1.5 s timeout: healthy nodes get
suspected, re-admitted, suspected again. (A milder overload only delays heartbeats without
opening gaps, which is why the link is set this slow.) Answers: is a priority class - or a
separate control channel - for the control plane a must-have in the radio design? Compare with s01.
"""

import _bootstrap  # noqa: F401

from swarm.bench import Camera, Check, Link, Scenario, Swarm

SCENARIO = Scenario(
    name="s10_no_qos",
    description="Reference swarm on 1 Mbps without control-plane QoS: control messages queue behind frames.",
    swarm=Swarm(leader="drone-a72", worker="drone-a53", size=5),
    camera=Camera(fps=10, size=128, p_object=0.3),
    link=Link(bps=1e6, loss=0.01, qos=False),
    duration=30.0,
    checks=(
        Check("central/none", "failures_detected", ">=", 1, note="saturated link -> late heartbeats -> false suspicions"),
        Check("central/downscale", "failures_detected", "==", 0, note="a light link keeps the control plane healthy"),
        Check("striped", "failures_detected", "==", 0),
        Check("local", "failures_detected", "==", 0),
    ),
    tags=("radio", "membership"),
)

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
