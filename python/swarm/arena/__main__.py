"""Command line for the arena test bench (the studio UI calls the same functions).

    python -m swarm.arena brains [--role defender]
    python -m swarm.arena test defence --defender auction --attacker decoy --episodes 8
    python -m swarm.arena test recon --recon relay_sweep --set drones=8 --set area=1500
    python -m swarm.arena train --generations 40 --out out/arena/coevo          (both swarms learn)
    python -m swarm.arena train --mode recon --generations 30 --out out/arena/recon
    python -m swarm.arena export out/arena/coevo/defender.npz --out deploy/defender.swm
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .brains import BrainError, NeuralBrain, list_brains, load_brain
from .config import ClashConfig, ConfigError, ReconConfig, TrainConfig, from_dict
from .run import TESTS, run_test


def _settings(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def _fmt(v, fmt: str) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return {"pct": f"{v:.0%}", "x": f"{v:.2f}x", "s": f"{v:.1f} s", "kbps": f"{v:.2f} kbit/s",
            "j": f"{v:.1f} J", "km": f"{v:.2f} km"}.get(fmt, f"{v:.2f}")


def cmd_brains(args) -> int:
    for b in list_brains(args.role):
        tag = b["kind"] if b["kind"] != "builtin" else ""
        print(f"{b['id']:<28} {str(b['role']):<9} {tag:<8} {b['title']}")
    return 0


def cmd_test(args) -> int:
    brains = {r: getattr(args, r) for r in ("recon", "defender", "attacker") if getattr(args, r)}
    doc = run_test(args.test, _settings(args.set), brains, args.episodes, args.seed, compare=not args.no_compare)
    v = doc["verdict"]
    print(f"{doc['title']}: {v['label']} - {v['line']}")
    for c in doc["cards"]:
        print(f"  {c['label']:<26} {_fmt(c.get('mean'), c['fmt'])}")
    if len(doc["compare"]) > 1:
        cols = doc["columns"]
        print("\n  rank  brain" + " " * 22 + "".join(f"{c[:12]:>13}" for c in cols))
        fmts = {c["key"]: c["fmt"] for c in doc["cards"]}
        for r in doc["compare"]:
            mark = "*" if r["selected"] else " "
            print(f"  {r['rank']:>3}{mark} {r['brain']['title'][:26]:<26}"
                  + "".join(f"{_fmt(r['metrics'][c], fmts.get(c, 'num')):>13}" for c in cols))
    if args.json:
        Path(args.json).write_text(json.dumps(doc), encoding="utf-8")
        print(f"\nreport (with replay) written to {args.json}")
    return 0


def cmd_train(args) -> int:
    from .train import train

    tc = from_dict(TrainConfig, {k: v for k, v in {
        "mode": args.mode, "generations": args.generations, "population": args.population,
        "episodes": args.episodes, "hidden": args.hidden, "seed": args.seed, "record_every": args.record_every,
        "workers": args.workers,
        "learn_defenders": not args.attackers_only, "learn_attackers": not args.defenders_only,
        "start_defender": args.start_defender, "start_attacker": args.start_attacker,
        "start_recon": args.start_recon}.items() if v is not None})
    env = from_dict(ReconConfig if tc.mode == "recon" else ClashConfig, _settings(args.set))
    pct = lambda v: "-" if v is None else f"{v:.0%}"  # noqa: E731

    def report(e: dict) -> None:
        if e["type"] == "start":
            print(f"simulating on {e['workers']} CPU core{'s' if e['workers'] > 1 else ''} (Python {e.get('python')}); "
                  f"a match is recorded every {e['record_every']} generations"
                  + ("; the computer stays awake until training ends" if e.get("awake") else ""), flush=True)
            return
        if tc.mode == "recon":
            c = e["current"]
            print(f"gen {e['gen']:>4}  fitness {c['fitness']:.3f}  coverage {pct(c['coverage'])}  "
                  f"contact {pct(c['connected'])}  camera {pct(c['camera_duty'])}  ({e['seconds']:.1f} s)", flush=True)
        else:
            h = e["h2h"]
            print(f"gen {e['gen']:>4}  head-to-head: intercepted {pct(h['intercepted'])} through {pct(h['leaked'])}  |  "
                  f"vs built-ins: defenders intercept {pct(e['defender_skill'])}, attackers get through "
                  f"{pct(e['attacker_skill'])}  ({e['seconds']:.1f} s)", flush=True)

    kw = {"recon": env} if tc.mode == "recon" else {"clash": env}
    res = train(tc, out_dir=args.out, report=report, **kw)
    print(f"\n{res['generations']} generations in {res['seconds']:.0f} s on {res['workers']} cores; "
          f"latest brains: {res['brains']}")
    for role, b in res["best"].items():
        print(f"best {role} on the fixed skill test: generation {b['gen']} ({b['skill']:.3f}) -> {b['path']}")
    return 0


def cmd_export(args) -> int:
    brain = NeuralBrain.load(args.brain) if args.brain.endswith(".npz") else load_brain(args.brain, args.role)
    if not isinstance(brain, NeuralBrain):
        raise SystemExit(f"{args.brain} is not a mesh network; only those compile to .swm")
    path = brain.to_swm(args.out)
    print(f"{path}: {brain.n_in} inputs -> {brain.hidden} ReLU -> {brain.n_out} outputs, {brain.macs} MACs per decision")
    return 0


def _on_native(argv: list[str]) -> int | None:
    """Runs this command again on the repo's native interpreter when this one runs emulated (Windows on ARM,
    see ``swarm._arch.native_python``) and waits for it; None when there is none."""
    import os
    import subprocess

    from swarm import _arch

    exe = _arch.native_python()
    if exe is None:
        return None
    env = dict(os.environ, SWARM_NATIVE_PYTHON="off")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"training on the native {_arch.machine_arch()} python in {exe.parent} - about twice as fast as this "
          f"emulated one (SWARM_NATIVE_PYTHON=off stays on this one)", flush=True)
    proc = subprocess.Popen([str(exe), "-m", "swarm.arena", *argv], env=env)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:  # Ctrl+C reaches the native run too: wait until it has stopped
            continue


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m swarm.arena", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("brains", help="list built-in and library brains")
    b.add_argument("--role", choices=("recon", "defender", "attacker"))
    t = sub.add_parser("test", help="run one test and print the report")
    t.add_argument("test", choices=list(TESTS))
    for role in ("recon", "defender", "attacker"):
        t.add_argument(f"--{role}", help=f"{role} brain: built-in name, lib:<package> or a .npz")
    t.add_argument("--episodes", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--set", action="append", metavar="KEY=VALUE", help="test setting, e.g. attackers=20")
    t.add_argument("--no-compare", action="store_true", help="skip the built-in comparison runs")
    t.add_argument("--json", help="write the full report (with replay) to this file")
    r = sub.add_parser("train", help="train mesh brains by evolution")
    r.add_argument("--mode", choices=("clash", "recon"))
    r.add_argument("--generations", type=int)
    r.add_argument("--population", type=int)
    r.add_argument("--episodes", type=int)
    r.add_argument("--hidden", type=int)
    r.add_argument("--seed", type=int)
    r.add_argument("--record-every", type=int)
    r.add_argument("--workers", type=int, help="simulation processes (default 0 = one per CPU core); speed only")
    r.add_argument("--defenders-only", action="store_true", help="attackers stay fixed (--start-attacker)")
    r.add_argument("--attackers-only", action="store_true", help="defenders stay fixed (--start-defender)")
    r.add_argument("--start-defender")
    r.add_argument("--start-attacker")
    r.add_argument("--start-recon")
    r.add_argument("--set", action="append", metavar="KEY=VALUE", help="engagement / recon setting")
    r.add_argument("--out", default="out/arena/train")
    e = sub.add_parser("export", help="compile a mesh brain to .swm for the C++ engine")
    e.add_argument("brain", help="a .npz or lib:<package>")
    e.add_argument("--role", default="defender", help="role when loading lib:<package>")
    e.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "train":
        rc = _on_native(list(sys.argv[1:] if argv is None else argv))
        if rc is not None:
            return rc
    try:
        return {"brains": cmd_brains, "test": cmd_test, "train": cmd_train, "export": cmd_export}[args.cmd](args)
    except (BrainError, ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
