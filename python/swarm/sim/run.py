"""Run the swarm simulator.

Examples::

    # one strategy, verbose event log
    python -m swarm.sim.run --strategy striped --drones 5 --duration 20 --kill 2@8 --revive 2@14 -v

    # compare all strategies side by side, with the leader dying at t=10s
    python -m swarm.sim.run --compare --drones 5 --fps 10 --duration 30 --kill leader@10

    # use the trained brain for real (numpy or the C++ engine) instead of the oracle
    python -m swarm.sim.run --compare --brain native --model models/tiny_cnn_int8.swm

    # a heavier hypothetical brain (6.4 MMAC/frame) on tiny MCUs with a slow radio
    python -m swarm.sim.run --compare --leader mcu-m7 --worker mcu-m4 --brain-macs 6400000 --bps 1e6

``--kill N@T`` takes a node id or ``leader`` (whoever leads at time T).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from swarm.devices import DEVICE_PROFILES, get_profile

from .brains import Brain, make_brain
from .core import Simulator
from .strategies import SwarmConfig, build_swarm

COMPARE_SET = [
    ("local", "none"),
    ("central", "none"),
    ("central", "downscale"),
    ("central", "features"),
    ("striped", "none"),
]


def parse_schedule(items: list[str]) -> list[tuple[str, float]]:
    out = []
    for it in items:
        who, at = it.split("@")
        out.append((who, float(at)))
    return out


def run_once(cfg: SwarmConfig, profiles, brain: Brain, duration: float, kills, revives, seed: int,
             bps: float, latency: float, loss: float, verbose: bool = False, pool=None, qos: bool = True) -> dict:
    sim = Simulator(seed)
    sim.verbose = verbose
    channel, nodes = build_swarm(sim, cfg, profiles, brain, bps, latency, loss, seed, pool, qos)

    def kill(who: str):
        if who == "leader":
            live = [n for n in nodes if n.alive]
            if not live:
                return
            idx = live[0].membership.leader()
        else:
            idx = int(who)
        nodes[idx].kill()

    for who, at in kills:
        sim.schedule(at, kill, who)
    for who, at in revives:
        sim.schedule(at, lambda w=who: nodes[int(w)].revive())

    sim.run(duration)
    s = sim.metrics.summary(duration, len(nodes))
    s["strategy"] = cfg.strategy + (f"/{cfg.preprocess}" if cfg.strategy == "central" else "")
    s["channel_util"] = channel.utilisation()
    s["cameras_on"] = sum(1 for n in nodes if n.camera_on)
    s["log"] = sim.log_lines
    return s


def fmt_row(s: dict) -> str:
    return (f"{s['strategy']:<18} {s['frames_processed']:>6} {s['frames_dropped']:>6} "
            f"{s['accuracy']*100:>5.1f}% {s['detections_reported']:>5}/{s['detections']:<5} "
            f"{s['latency_p50_ms']:>7.1f} {s['latency_p95_ms']:>7.1f} "
            f"{s['kbps']:>8.1f} {s['channel_util']*100:>5.1f}% {s['bytes_per_frame']:>7.0f} "
            f"{s['gmac_total']:>6.2f} {s['busy_mean']*100:>5.1f}% {s['busy_max']*100:>5.1f}% "
            f"{s['leader_changes']:>3} {s['rebalances']:>3}")


HEADER = (f"{'strategy':<18} {'proc':>6} {'drop':>6} {'acc':>6} {'det rep/all':>11} "
          f"{'p50ms':>7} {'p95ms':>7} {'kbps':>8} {'chan':>6} {'B/frame':>7} {'GMAC':>6} {'busy':>6} {'max':>6} "
          f"{'ldr':>3} {'rbl':>3}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["local", "central", "striped"], default="central")
    ap.add_argument("--preprocess", choices=["none", "downscale", "features"], default="none")
    ap.add_argument("--compare", action="store_true", help="run every strategy/preprocess combination")
    ap.add_argument("--drones", type=int, default=5, help="swarm size including the leader")
    ap.add_argument("--leader", default="drone-a72", help=f"device profile of node 0. Known: {sorted(DEVICE_PROFILES)}")
    ap.add_argument("--worker", default="drone-a53", help="device profile of the other nodes")
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second per active camera")
    ap.add_argument("--striped-fps", type=float, default=None,
                    help="camera fps for the striped strategy (default: --fps; use fps*drones to process as many "
                         "frames as the all-cameras strategies)")
    ap.add_argument("--cam-size", type=int, default=128, help="camera resolution (px per side, 8-bit)")
    ap.add_argument("--duration", type=float, default=30.0, help="simulated seconds")
    ap.add_argument("--p-object", type=float, default=0.3, help="fraction of frames containing something")
    ap.add_argument("--bps", type=float, default=None, help="shared channel bit rate (default: weakest radio)")
    ap.add_argument("--latency", type=float, default=None, help="channel one-way latency s (default: worst radio)")
    ap.add_argument("--loss", type=float, default=0.01, help="per-message loss probability")
    ap.add_argument("--no-qos", action="store_true",
                    help="control messages (heartbeats/results/cmds) queue behind bulk frames like everything else")
    ap.add_argument("--hb-interval", type=float, default=0.5)
    ap.add_argument("--missed-beats", type=int, default=3)
    ap.add_argument("--queue-limit", type=int, default=4)
    ap.add_argument("--no-leader-share", action="store_true", help="striped: leader only dispatches, never computes")
    ap.add_argument("--kill", action="append", default=[], metavar="WHO@T", help="node id or 'leader' @ time")
    ap.add_argument("--revive", action="append", default=[], metavar="ID@T")
    ap.add_argument("--brain", choices=["oracle", "numpy", "native"], default="oracle")
    ap.add_argument("--model", type=Path, default=None, help=".swm file for numpy/native brains")
    ap.add_argument("--brain-macs", type=int, default=401_568, help="oracle brain cost per frame")
    ap.add_argument("--brain-accuracy", type=float, default=0.87)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="write full results (incl. event logs) here")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the event log")
    args = ap.parse_args(argv)

    leader, worker = get_profile(args.leader), get_profile(args.worker)
    profiles = [leader] + [worker] * (args.drones - 1)
    bps = args.bps or min(p.radio_bps for p in profiles)
    latency = args.latency if args.latency is not None else max(p.radio_latency_s for p in profiles)

    brain = make_brain(args.brain, args.model, macs=args.brain_macs, accuracy=args.brain_accuracy, seed=args.seed) \
        if args.brain == "oracle" else make_brain(args.brain, args.model)
    frame_size = brain.input_shape[-1]

    pool = None
    if args.brain != "oracle":
        from swarm.train.dataset import FramePool
        pool = FramePool(frame_size, per_class=64, seed=args.seed + 999)

    kills, revives = parse_schedule(args.kill), parse_schedule(args.revive)
    combos = COMPARE_SET if args.compare else [(args.strategy, args.preprocess)]

    print(f"swarm: {args.drones} nodes  leader={leader.name}  workers={worker.name}  "
          f"channel={bps/1e6:.2f} Mbps / {latency*1e3:.1f} ms / loss {args.loss:.1%}")
    print(f"brain: {brain.describe()}")
    print(f"camera: {args.cam_size}x{args.cam_size} 8-bit @ {args.fps} fps  "
          f"(raw {args.cam_size**2} B, brain input {brain.input_bytes} B)")
    if kills or revives:
        print(f"faults: kill {kills}  revive {revives}")
    print()
    print(HEADER)
    print("-" * len(HEADER))

    results = []
    for strategy, pre in combos:
        fps = args.striped_fps if (strategy == "striped" and args.striped_fps) else args.fps
        cfg = SwarmConfig(strategy=strategy, preprocess=pre, fps=fps, cam_size=args.cam_size,
                          include_leader=not args.no_leader_share, hb_interval=args.hb_interval,
                          missed_beats=args.missed_beats, queue_limit=args.queue_limit, p_object=args.p_object,
                          frame_size=frame_size)
        s = run_once(cfg, profiles, brain, args.duration, kills, revives, args.seed, bps, latency, args.loss,
                     verbose=args.verbose and not args.compare, pool=pool, qos=not args.no_qos)
        results.append(s)
        print(fmt_row(s))
        sys.stdout.flush()

    print()
    print("proc/drop = frames processed / never processed; det rep/all = detections the leader learnt of / total;")
    print("p50/p95 = capture->leader-knows latency of detections; chan = shared channel utilisation; busy = mean/max CPU busy;")
    print("ldr/rbl = leader changes / rebalance events (membership changes seen).")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
