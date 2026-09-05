"""Swarm simulator: strategies, failure detection, rebalancing, leader failover."""

import pytest

from swarm.devices import get_profile
from swarm.sim.brains import OracleBrain
from swarm.sim.core import Simulator
from swarm.sim.strategies import SwarmConfig, build_swarm

A72, A53, M4 = get_profile("drone-a72"), get_profile("drone-a53"), get_profile("mcu-m4")


def run(strategy, preprocess="none", n=5, duration=12.0, kill=None, revive=None, fps=10.0, bps=6e6, loss=0.0,
        brain_macs=401_568, profiles=None, seed=0, **cfg_kw):
    sim = Simulator(seed)
    brain = OracleBrain(macs=brain_macs, accuracy=1.0, seed=seed)
    cfg = SwarmConfig(strategy=strategy, preprocess=preprocess, fps=fps, **cfg_kw)
    profiles = profiles or [A72] + [A53] * (n - 1)
    channel, nodes = build_swarm(sim, cfg, profiles, brain, bps, 0.002, loss, seed)
    for who, at in (kill or []):
        sim.schedule(at, lambda w=who: nodes[w].kill())
    for who, at in (revive or []):
        sim.schedule(at, lambda w=who: nodes[w].revive())
    sim.run(duration)
    s = sim.metrics.summary(duration, len(nodes))
    s["channel_util"] = channel.utilisation()
    s["nodes"] = nodes
    s["log"] = sim.log_lines
    return s


@pytest.mark.parametrize("strategy,pre", [("local", "none"), ("central", "none"), ("central", "downscale"),
                                          ("central", "features"), ("striped", "none")])
def test_every_strategy_processes_frames_without_faults(strategy, pre):
    s = run(strategy, pre)
    assert s["frames_processed"] > 0
    assert s["frames_dropped"] <= 5  # at most one frame per camera still in flight at the end
    assert s["accuracy"] == 1.0
    assert s["leader_changes"] == 0 and s["failures_detected"] == 0
    assert s["detections_reported"] == s["detections"]


def test_striped_uses_one_camera_and_spreads_compute():
    s = run("striped", n=4)
    cams = [n.camera_on for n in s["nodes"]]
    assert cams == [True, False, False, False]
    macs = s["macs_by_node"]
    assert all(m > 0 for m in macs), "every member should take a share of frames"
    # leader also downsamples every frame, so allow it a little extra; frame shares are equal (round-robin)
    assert max(macs) / min(macs) < 1.5, macs


def test_preprocessing_cuts_payload_and_features_offload_leader():
    raw = run("central", "none")
    down = run("central", "downscale")
    feat = run("central", "features")
    assert raw["bytes_per_frame"] > 10 * down["bytes_per_frame"]
    assert feat["bytes_per_frame"] <= down["bytes_per_frame"] * 1.05
    # leader (node 0) does far less compute when workers run the conv frontend
    assert feat["macs_by_node"][0] < down["macs_by_node"][0] * 0.5


def test_raw_streaming_saturates_a_slow_link_and_drops_frames():
    s = run("central", "none", bps=1e6, duration=10.0)
    assert s["channel_util"] > 1.0
    assert s["frames_dropped"] > s["frames_processed"] * 0.3


def test_worker_failure_is_detected_and_rotation_rebalances():
    s = run("striped", n=5, duration=15.0, kill=[(2, 5.0)], revive=[(2, 10.0)], hb_interval=0.5, missed_beats=3)
    assert s["failures_detected"] == 4, "each of the 4 survivors suspects node 2 exactly once"
    log = "\n".join(s["log"])
    assert "rotation now [0, 1, 3, 4]" in log
    assert "re-admits node 2" in log and "rotation now [0, 1, 2, 3, 4]" in log
    # only frames sent to node 2 between death and detection (1.5 s * 10 fps / 5) are lost
    assert s["frames_dropped"] < 8, s["frames_dropped"]
    assert s["leader_changes"] == 0


def test_leader_failure_elects_next_best_and_camera_moves():
    s = run("striped", n=4, duration=15.0, kill=[(0, 4.0)])
    assert s["leader_changes"] == 1
    nodes = s["nodes"]
    assert not nodes[0].alive and nodes[1].camera_on and not nodes[2].camera_on
    # all survivors agree on the leader
    assert {n.membership.leader() for n in nodes[1:]} == {1}
    # work kept flowing after failover
    assert s["frames_processed"] > 10 * 4.0 * 0.8 + 10 * (15.0 - 6.0) * 0.8


def test_central_leader_failure_reroutes_frames():
    s = run("central", "downscale", n=4, duration=12.0, kill=[(0, 4.0)])
    assert s["leader_changes"] == 1
    processed_after_failover = s["frames_processed"] - 4 * 10 * 4.0
    assert processed_after_failover > 3 * 10 * (12.0 - 5.6) * 0.9  # 3 cameras once node 0 is gone


def test_revived_leader_takes_back_over():
    s = run("striped", n=3, duration=20.0, kill=[(0, 3.0)], revive=[(0, 9.0)])
    nodes = s["nodes"]
    assert nodes[0].alive and nodes[0].camera_on and not nodes[1].camera_on
    assert s["leader_changes"] == 2
    assert {n.membership.leader() for n in nodes} == {0}


def test_membership_priority_prefers_more_capable_node():
    # put the powerful node last: it must still be elected leader
    s = run("striped", n=3, profiles=[M4, A53, A72], duration=2.0)
    assert {n.membership.leader() for n in s["nodes"]} == {2}
    assert s["nodes"][2].camera_on


def test_slow_device_becomes_compute_bottleneck():
    # 6.4 MMAC brain on Cortex-M4s: 76 ms/frame vs 100 ms period per camera -> central leader cannot keep up
    s = run("central", "downscale", n=5, profiles=[M4] * 5, brain_macs=6_400_000, fps=10.0, duration=10.0)
    assert s["frames_rejected"] > 0
    assert s["busy_by_node"][0] > 0.9
    # striping the same load over 5 brains keeps up
    s2 = run("striped", n=5, profiles=[M4] * 5, brain_macs=6_400_000, fps=50.0, duration=10.0)
    assert s2["frames_rejected"] == 0 and s2["busy_max"] < 0.9
