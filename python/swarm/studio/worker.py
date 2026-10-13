"""Worker process: everything that runs brain code happens here, never in the server.

    python -m swarm.studio.worker <job dir>     (started by jobs.JobManager)

Reads ``<job dir>/spec.json``, reports progress and structured events as JSON lines on stdout (anything
else a brain prints is relayed as log text) and writes ``result.json`` or ``error.json``.

Job kinds:
    test       run one arena test (with the built-in comparison) and save the report
    train      evolve mesh brains; stops gracefully, keeping what it learned, when ``<job dir>/stop`` appears
    validate   the SDK conformance checks of a brain package
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path


def emit(**kw) -> None:
    sys.stdout.write("\x1e" + json.dumps(_clean(kw)) + "\n")  # record separator marks protocol lines
    sys.stdout.flush()


def _clean(o):
    """JSON-safe: non-finite floats become null, numpy scalars plain numbers."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and callable(o.item):
        return _clean(o.item())
    return o


def do_test(spec: dict, job_dir: Path) -> dict:
    from swarm.arena import run_test

    from .store import save_run

    emit(progress="simulating")
    doc = run_test(spec["test"], spec.get("settings"), spec.get("brains"), int(spec.get("episodes", 8)),
                   int(spec.get("seed", 0)), compare=bool(spec.get("compare", True)),
                   progress=lambda i, n: emit(progress=f"simulated {i} of {n} brain line-ups"))
    run_id = save_run(_clean(doc))
    return {"run": run_id, "verdict": doc["verdict"], "elapsed_s": doc["elapsed_s"]}


def do_train(spec: dict, job_dir: Path) -> dict:
    from swarm.arena.config import ClashConfig, ReconConfig, TrainConfig, from_dict
    from swarm.arena.train import train

    from .store import write_json

    tc = from_dict(TrainConfig, spec["train"])
    env = from_dict(ReconConfig if tc.mode == "recon" else ClashConfig, spec.get("settings"))
    out = Path(spec["out_dir"])
    status = dict(spec["status"], state="running", started=time.time())
    write_json(out / "status.json", status)
    stop_file = job_dir / "stop"

    def report(event: dict) -> None:
        emit(event=event)
        if event["type"] == "start":
            status["sides"] = event.get("sides")
            status["references"] = event.get("references")
            status["workers"] = event.get("workers")
            status["python"] = event.get("python")
            write_json(out / "status.json", _clean(status))
        elif event["type"] == "generation":
            status["generation"] = event["gen"]
            status["last"] = {k: event[k] for k in ("h2h", "defender_skill", "attacker_skill", "current")
                              if k in event}
            status["best"] = event.get("best", {})
            write_json(out / "status.json", _clean(status))
            emit(progress=f"generation {event['gen']} of {tc.generations}")

    kw = {"recon": env} if tc.mode == "recon" else {"clash": env}
    res = train(tc, out_dir=out, report=report, should_stop=stop_file.exists, **kw)
    status.update(state="stopped" if res["stopped"] else "done", finished=time.time(),
                  generation=res["generations"], brains=sorted(Path(p).stem for p in res["brains"].values()),
                  best={role: {"gen": b["gen"], "skill": b["skill"]} for role, b in res["best"].items()})
    write_json(out / "status.json", _clean(status))
    return {"training": status["id"], "generations": res["generations"], "stopped": res["stopped"],
            "seconds": res["seconds"]}


def do_validate(spec: dict, job_dir: Path) -> dict:
    from swarm.sdk import validate

    emit(progress=f"validating {Path(spec['path']).name}")
    return {"report": validate(spec["path"]).to_dict()}


HANDLERS = {"test": do_test, "train": do_train, "validate": do_validate}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    job_dir = Path(argv[0])
    spec = json.loads((job_dir / "spec.json").read_text(encoding="utf-8"))
    try:
        result = HANDLERS[spec["kind"]](spec, job_dir)
        (job_dir / "result.json").write_text(json.dumps(_clean(result)), encoding="utf-8")
        return 0
    except BaseException as exc:  # noqa: BLE001 - report everything to the server
        (job_dir / "error.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}",
                                                        "traceback": traceback.format_exc()}), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
