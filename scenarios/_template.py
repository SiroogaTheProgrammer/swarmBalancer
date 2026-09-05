"""TEMPLATE for a new standardized test - copy to ``scenarios/sNN_my_test.py`` and edit.

    python scenarios/sNN_my_test.py          run it (add -h for options: --brain, --fast, --strategies, --log ...)
    python -m swarm.bench                    run the whole battery; files starting with "_" are skipped

Every scenario always simulates the ``local`` baseline (every drone is an independent thinker) and
reports every other strategy as value + ratio vs that baseline. A file may define one ``SCENARIO``
or a list ``SCENARIOS`` (handy for A/B variants of the same setup).

Metrics you can use in ``Check`` (see swarm/bench/report.py for the full list and their meaning):
    fps_processed objects_seen_ps objects_reported_ps recall accuracy false_alarms_reported
    latency_p50_ms latency_p95_ms gmac_ps kmac_per_frame busy_mean busy_max kbps channel_util
    bytes_per_frame energy_j cameras_on frames_dropped frames_rejected messages_lost
    leader_changes failures_detected rebalances
    detect_s reelect_s recover_s work_retained recall_retained frames_lost_to_fault   (fault scenarios)
Device profiles: mcu-m4 mcu-m7 drone-a53 drone-a72 cubesat-obc sat-payload host   (swarm/devices/profiles.py)
"""

import _bootstrap  # noqa: F401  (makes `swarm` importable when this file is run directly)

from swarm.bench import BrainSpec, Camera, Check, Faults, Link, Scenario, Swarm

SCENARIO = Scenario(
    name="sNN_my_test",                                   # keep equal to the file name
    description="One sentence: which question does this scenario answer?",
    swarm=Swarm(leader="drone-a72", worker="drone-a53", size=5),
    # heterogeneous swarms list every node: Swarm(nodes=("drone-a72", "drone-a53", "mcu-m4", "mcu-m4")),
    camera=Camera(fps=10, size=128, p_object=0.3, striped_fps="match"),
    #   striped_fps: "match" -> one camera at fps*size (same frames/s as N cameras); None -> fps (low power)
    link=Link(bps=None, latency_s=None, loss=0.01, qos=True, hb_interval=0.5, missed_beats=3),
    #   bps/latency None -> taken from the weakest radio in the swarm
    brain=BrainSpec(kind="auto", model="models/tiny_cnn_int8.swm"),
    #   a hypothetical brain without training anything: BrainSpec(kind="oracle", macs=6_400_000, input_size=64)
    faults=Faults(kill=(("leader", 10.0),), revive=()),   # who="leader" or a node id; () for no faults
    duration=30.0,
    seed=0,
    checks=(
        Check("*", "leader_changes", "==", 1, note="every design must re-elect exactly once"),
        Check("*", "recover_s", "<=", 4.0, note="results must flow to the new leader within 4 s"),
        Check("striped", "gmac_ps", "<=", 1.2, ratio=True, note="striping must not cost more compute than local"),
    ),
    tags=("example",),
)

if __name__ == "__main__":
    from swarm.bench import main

    raise SystemExit(main(SCENARIO))
