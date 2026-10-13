"""Where the studio keeps things: ``out/studio/`` (git-ignored).

    runs/<id>.json          a test report: settings, brains, score cards, comparison, replay of episode 0
    runs/<id>.meta.json     the few fields the "recent runs" list needs
    training/<id>/          a training run: status.json, history.json, defender/attacker/recon.npz, matches/
    jobs/<id>/              scratch space of running worker processes
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("SWARM_STUDIO_OUT") or REPO_ROOT / "out" / "studio")  # tests point this elsewhere
RUNS = OUT / "runs"
TRAINING = OUT / "training"
JOBS = OUT / "jobs"
KEEP_RUNS = 60
KEEP_TRAINING = 24
ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
MATCH_RE = re.compile(r"^gen_\d{4,5}\.json$")   # runs go up to 20,000 generations


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def _check(run_id: str) -> str:
    if not ID_RE.match(run_id or ""):
        raise KeyError(f"bad id {run_id!r}")
    return run_id


def write_json(path: Path, doc) -> None:
    """Atomic write, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------------
# test reports
# ----------------------------------------------------------------------------
def save_run(doc: dict) -> str:
    run_id = new_id()
    doc["id"] = run_id
    write_json(RUNS / f"{run_id}.json", doc)
    meta = {"id": run_id, "test": doc["test"], "title": doc["title"], "created": doc["created"],
            "episodes": doc["episodes"], "verdict": doc["verdict"],
            "brains": {role: b.get("title") for role, b in doc["brains"].items()},
            "headline": [{"label": c["label"], "fmt": c["fmt"], "mean": c.get("mean")} for c in doc["cards"][:2]]}
    write_json(RUNS / f"{run_id}.meta.json", meta)
    _prune(sorted(RUNS.glob("*.meta.json"), reverse=True)[KEEP_RUNS:],
           lambda p: (p.unlink(missing_ok=True), (RUNS / p.name.replace(".meta", "")).unlink(missing_ok=True)))
    return run_id


def list_runs() -> list[dict]:
    out = []
    for p in sorted(RUNS.glob("*.meta.json"), reverse=True) if RUNS.is_dir() else []:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def run_file(run_id: str) -> Path:
    p = RUNS / f"{_check(run_id)}.json"
    if not p.is_file():
        raise KeyError(f"no run {run_id}")
    return p


def delete_run(run_id: str) -> None:
    run_file(run_id).unlink()
    (RUNS / f"{run_id}.meta.json").unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# training runs
# ----------------------------------------------------------------------------
def new_training() -> tuple[str, Path]:
    tid = new_id()
    d = TRAINING / tid
    (d / "matches").mkdir(parents=True)
    _prune(sorted((p for p in TRAINING.iterdir() if p.is_dir() and ID_RE.match(p.name)), reverse=True)[KEEP_TRAINING:],
           lambda p: shutil.rmtree(p, ignore_errors=True))
    return tid, d


def training_dir(tid: str) -> Path:
    d = TRAINING / _check(tid)
    if not (d / "status.json").is_file():
        raise KeyError(f"no training run {tid}")
    return d


def training_status(tid: str) -> dict:
    return json.loads((training_dir(tid) / "status.json").read_text(encoding="utf-8"))


def list_trainings() -> list[dict]:
    out = []
    for d in sorted(TRAINING.iterdir(), reverse=True) if TRAINING.is_dir() else []:
        if ID_RE.match(d.name) and (d / "status.json").is_file():
            try:
                out.append(json.loads((d / "status.json").read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return out


def match_file(tid: str, name: str) -> Path:
    if not MATCH_RE.match(name or ""):
        raise KeyError(f"bad match name {name!r}")
    p = training_dir(tid) / "matches" / name
    if not p.is_file():
        raise KeyError(f"no match {name} in {tid}")
    return p


def delete_training(tid: str) -> None:
    shutil.rmtree(training_dir(tid))


def _prune(items, remove) -> None:
    for p in items:
        try:
            remove(p)
        except OSError:
            pass
