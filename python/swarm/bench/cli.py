"""Command line: run one preset (``python scenarios/<preset>.py``) or the whole battery (``python -m swarm.bench``).

    python -m swarm.bench                      # every scenarios/*.py, scoreboard at the end, reports in out/bench/
    python -m swarm.bench --list
    python -m swarm.bench s02 s08              # by name prefix, or give file paths
    python -m swarm.bench --fast --no-stress   # smoke run (half duration)
    python -m swarm.bench --brain oracle       # ignore the trained model, use the oracle everywhere
    python scenarios/s02_leader_loss.py -v     # one preset with the simulator's event log

Exit code 1 if any check failed - the battery doubles as a regression test.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .report import dump_json, render_markdown, render_scoreboard, render_scoreboard_markdown, render_text
from .runner import REPO_ROOT, ScenarioResult, run_scenario
from .scenario import ALL_STRATEGIES, Scenario

DEFAULT_DIR = REPO_ROOT / "scenarios"
DEFAULT_OUT = REPO_ROOT / "out" / "bench"


def discover(directory: Path) -> list[Path]:
    """Preset files: every ``*.py`` in ``directory`` not starting with ``_``."""
    return sorted(p for p in directory.glob("*.py") if not p.name.startswith("_"))


def load_file(path: Path) -> list[Scenario]:
    """Imports a preset file and returns its ``SCENARIO`` / ``SCENARIOS``."""
    path = path.resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))  # so `import _bootstrap` inside the preset resolves
    spec = importlib.util.spec_from_file_location(f"swarm_scenario_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    found: list[Scenario] = []
    if hasattr(mod, "SCENARIOS"):
        found.extend(mod.SCENARIOS)
    if hasattr(mod, "SCENARIO"):
        found.append(mod.SCENARIO)
    if not found:
        raise ValueError(f"{path.name} defines neither SCENARIO nor SCENARIOS")
    for s in found:
        if not isinstance(s, Scenario):
            raise TypeError(f"{path.name}: expected swarm.bench.Scenario objects, got {type(s).__name__}")
    return found


def load_scenarios(directory: Path, selectors: list[str]) -> list[tuple[Path, Scenario]]:
    files = discover(directory)
    explicit = [Path(s) for s in selectors if Path(s).is_file()]
    names = [s for s in selectors if not Path(s).is_file()]
    for f in explicit:
        if f.resolve() not in [x.resolve() for x in files]:
            files.append(f)
    out: list[tuple[Path, Scenario]] = []
    for f in files:
        for sc in load_file(f):
            wanted = (not selectors or f in explicit
                      or any(sc.name == n or sc.name.startswith(n) or f.stem.startswith(n) for n in names))
            if wanted:
                out.append((f, sc))
    if selectors and not out:
        known = ", ".join(sc.name for f in files for sc in load_file(f))
        raise SystemExit(f"no scenario matches {selectors}; known: {known}")
    return out


def _write_reports(out_dir: Path, results: list[ScenarioResult], suite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        (out_dir / f"{r.scenario.name}.md").write_text(render_markdown(r), encoding="utf-8")
        (out_dir / f"{r.scenario.name}.json").write_text(dump_json(r), encoding="utf-8")
    if suite:
        parts = ["# swarmBalancer benchmark battery", "", render_scoreboard_markdown(results), "",
                 "Baseline = `local` (every drone is an independent thinker). Other columns: value (ratio vs baseline); "
                 "[+] higher is better, [-] lower is better.", ""]
        parts += [render_markdown(r) for r in results]
        (out_dir / "suite.md").write_text("\n".join(parts), encoding="utf-8")


def main(scenario: Scenario | None = None, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=None if scenario else "python -m swarm.bench",
        description=(f"Run the scenario preset '{scenario.name}'" if scenario else __doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    if scenario is None:
        ap.add_argument("selectors", nargs="*", metavar="NAME|FILE",
                        help="scenario names / prefixes or preset files (default: every file in --dir)")
        ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="preset directory")
        ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--brain", choices=["auto", "oracle", "numpy", "native"], default=None,
                    help="override the brain kind of every scenario")
    ap.add_argument("--strategies", default=None,
                    help=f"comma-separated subset of {', '.join(ALL_STRATEGIES)} (the baseline is always added)")
    ap.add_argument("--duration", type=float, default=None, help="override simulated seconds")
    ap.add_argument("--fast", action="store_true", help="simulate half the duration (smoke run; faults after that are skipped)")
    ap.add_argument("--no-stress", action="store_true", help="skip the device stress section")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for .md/.json reports")
    ap.add_argument("--no-report", action="store_true", help="do not write report files")
    ap.add_argument("--log", action="store_true", help="print the fault-related event log per strategy")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every simulator event (single scenario)")
    args = ap.parse_args(argv)

    if scenario is not None:
        items: list[tuple[Path | None, Scenario]] = [(None, scenario)]
    else:
        items = list(load_scenarios(args.dir, args.selectors))
        if args.list:
            for f, sc in items:
                print(f"{sc.name:<28} {f.name:<28} {sc.description}")
            return 0
        if not items:
            print(f"no preset files in {args.dir} (copy scenarios/_template.py to get started)")
            return 2

    strategies = [s.strip() for s in args.strategies.split(",")] if args.strategies else None
    results: list[ScenarioResult] = []
    for f, sc in items:
        duration = args.duration or (sc.duration / 2 if args.fast else None)
        res = run_scenario(sc, brain_kind=args.brain, duration=duration, strategies=strategies,
                           with_stress=not args.no_stress, verbose=args.verbose and len(items) == 1)
        results.append(res)
        print(render_text(res, show_log=args.log))
        print()
        sys.stdout.flush()

    if len(results) > 1:
        print(render_scoreboard(results))
    if not args.no_report:
        _write_reports(args.out, results, suite=scenario is None)
        print(f"\nreports: {args.out}" + ("\\suite.md" if scenario is None and len(results) > 1 else ""))
    return 0 if all(r.checks_passed() for r in results) else 1
