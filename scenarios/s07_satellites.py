"""s07 - Satellite constellation: a payload sat and three CubeSats sharing a 9.6 kbps link.

The ``sat-payload`` (Cortex-A class) leads; the ``cubesat-obc`` nodes are 100 MHz radiation-
tolerant MCUs with 256 KiB of RAM - the int8 brain needs 150 KiB and 8 ms per frame there.
Cameras take one frame every 4 s of 64 px imagery; heartbeats every 5 s because chatter is
expensive on this radio. Answers: is it better to send imagery to the capable satellite (or a
ground station) for classification, or to classify on orbit and send only detections? Raw frames
do not fit the link at all; downscaled frames fill most of it; detections cost almost nothing.
"""

import _bootstrap  # noqa: F401

from swarm.bench import Camera, Check, Link, Scenario, Swarm

SCENARIO = Scenario(
    name="s07_satellites",
    description="sat-payload + 3x cubesat-obc over a 9.6 kbps / 10 ms link, 64 px frames every 4 s.",
    swarm=Swarm(nodes=("sat-payload", "cubesat-obc", "cubesat-obc", "cubesat-obc")),
    camera=Camera(fps=0.25, size=64, p_object=0.3, striped_fps="match"),
    link=Link(bps=9600, latency_s=0.01, loss=0.02, hb_interval=5.0, missed_beats=3),
    duration=240.0,
    checks=(
        Check("local", "recall", ">=", 0.8, note="classify on orbit, send 16-byte detections"),
        Check("local", "channel_util", "<=", 0.1),
        Check("central/none", "channel_util", ">=", 2.0, note="4 KB raw frames do not fit 9.6 kbps"),
        Check("central/downscale", "channel_util", "<=", 0.95, note="1 KB per frame just about fits"),
        Check("central/downscale", "recall", ">=", 0.7),
        Check("*", "leader_changes", "==", 0),
    ),
    tags=("satellite", "radio-bound"),
)

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
