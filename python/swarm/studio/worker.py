"""Worker process: everything that runs package or preset code happens here, never in the server.

    python -m swarm.studio.worker <job dir>     (started by jobs.JobManager)

Reads ``<job dir>/spec.json``, reports progress as JSON lines on stdout (anything else a brain or a
training script prints is relayed as log text), and writes ``result.json`` or ``error.json``.
Job kinds: run (simulate a scenario with tracing), validate (a package), build (run a package's
build script, then validate), index (import every preset file).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def emit(**kw) -> None:
    sys.stdout.write("\x1e" + json.dumps(kw) + "\n")  # record separator marks protocol lines
    sys.stdout.flush()


def _sanitize(o):
    from swarm.bench.report import _sanitize as sanitize

    return sanitize(o)


def _profile_dict(p) -> dict:
    return {"name": p.name, "description": p.description, "cpu_mhz": p.cpu_mhz, "macs_per_cycle": p.macs_per_cycle,
            "ram_bytes": p.ram_bytes, "radio_bps": p.radio_bps, "radio_latency_s": p.radio_latency_s,
            "tags": list(p.tags)}


def do_run(spec: dict) -> dict:
    import numpy as np

    from swarm.bench.runner import run_scenario
    from swarm.bench.scenario import strategy_note, to_plain
    from swarm.devices import get_profile
    from swarm.sdk import use_package

    from .scenario_io import scenario_from_plain
    from .store import prune_runs

    for p in spec.get("packages") or []:
        emit(progress=f"loading tactic package {p}")
        use_package(p)
    sc = scenario_from_plain(spec["scenario"])
    res = run_scenario(sc, brain_kind=spec.get("brain_kind") or None, strategies=spec.get("strategies") or None,
                       with_stress=bool(spec.get("with_stress", True)), trace=True, keep_frames=True,
                       progress=lambda msg: emit(progress=msg))
    emit(progress="writing the playback files")
    out = Path(spec["run_dir"])
    out.mkdir(parents=True, exist_ok=True)
    strategies = []
    for i, r in enumerate(res.strategies):
        fids, images = r.trace.frame_arrays()
        np.save(out / f"fids_{i}.npy", fids)
        np.save(out / f"frames_{i}.npy", images)
        strategies.append({"name": r.strategy, "note": strategy_note(r.strategy), "metrics": r.metrics,
                           "twin_metrics": r.twin_metrics, "trace": r.trace.to_dict(),
                           "frames": {"count": int(len(fids)), "shape": list(images.shape[1:])}})
    run_id = out.name
    doc = {
        "id": run_id, "created": time.time(), "scenario": to_plain(sc), "theatre": spec.get("theatre"),
        "packages": spec.get("packages") or [], "preset": spec.get("preset"),
        "brain": {"kind": res.brain_kind, "description": res.brain_desc, "note": res.brain_note,
                  "model_path": res.model_path},
        "profiles": [_profile_dict(get_profile(n)) for n in res.resolved["profiles"]],
        "resolved": res.resolved, "strategies": strategies, "stress": res.stress,
        "checks": [{"strategy": c.strategy, "metric": c.check.metric, "op": c.check.op, "value": c.check.value,
                    "ratio": c.check.ratio, "note": c.check.note, "actual": c.actual, "passed": c.passed}
                   for c in res.checks],
        "wall_s": res.wall_s,
    }
    (out / "run.json").write_text(json.dumps(_sanitize(doc), separators=(",", ":")), encoding="utf-8")
    meta = {"id": run_id, "created": doc["created"], "name": sc.name, "description": sc.description,
            "strategies": [s["name"] for s in strategies], "brain": res.brain_kind,
            "nodes": len(res.resolved["profiles"]), "duration": res.resolved["duration"],
            "faults": sc.faults.describe(), "checks_passed": sum(c.passed for c in res.checks),
            "checks": len(res.checks), "wall_s": res.wall_s}
    (out / "meta.json").write_text(json.dumps(_sanitize(meta)), encoding="utf-8")
    prune_runs()
    return {"run_id": run_id}


def do_validate(spec: dict) -> dict:
    from swarm.sdk import load_package, validate

    from .store import save_validation

    emit(progress=f"validating {spec['path']}")
    rep = validate(spec["path"]).to_dict()
    pkg = load_package(spec["path"])
    save_validation(pkg.root, pkg.name, rep)
    return {"report": rep}


def do_build(spec: dict) -> dict:
    from swarm.sdk import load_package

    pkg = load_package(spec["path"])
    script = pkg.manifest.get("build")
    if not script:
        raise ValueError(f"{pkg.name} has no build script in its manifest")
    emit(progress=f"compiling {pkg.name}: python {script}")
    env = dict(os.environ)
    repo_python = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_python + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen([sys.executable, str(pkg.root / script)], cwd=pkg.root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        emit(log=line.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{script} exited with code {rc}")
    result = do_validate(spec)
    result["built"] = True
    return result


def do_index(spec: dict) -> dict:
    from .scenario_io import index_presets, preset_fingerprint

    emit(progress="reading preset files")
    return {"fingerprint": preset_fingerprint(), "presets": index_presets()}


HANDLERS = {"run": do_run, "validate": do_validate, "build": do_build, "index": do_index}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    job_dir = Path(argv[0])
    spec = json.loads((job_dir / "spec.json").read_text(encoding="utf-8"))
    try:
        result = HANDLERS[spec["kind"]](spec)
        (job_dir / "result.json").write_text(json.dumps(_sanitize(result)), encoding="utf-8")
        return 0
    except BaseException as exc:  # noqa: BLE001 - report everything to the server
        (job_dir / "error.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}",
                                                        "traceback": traceback.format_exc()}), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
