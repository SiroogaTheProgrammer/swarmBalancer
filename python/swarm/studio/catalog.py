"""What the studio offers: devices, strategies and tactics, brains, packages, presets and engine status.

Built in the server process from manifests, ``.swm`` headers and cached validation reports - never by
running package or preset code. Packages without a fresh report and stale preset indexes are handed to
worker jobs; the UI polls the catalog until they are in.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from swarm.bench.report import COST_ROWS, MISSION_ROWS, SURVIVAL_ROWS
from swarm.bench.scenario import ALL_STRATEGIES, BASELINE, STRATEGY_NOTES
from swarm.devices import DEVICE_PROFILES
from swarm.sdk.packages import LIBRARY_DIRS, Package, PackageError, discover_packages, swm_info
from swarm.sdk.validate import runs_python

from .jobs import Job, JobManager
from .scenario_io import CHECK_METRICS, THEATRE_CHOICES, THEATRE_DEFAULTS, preset_fingerprint
from .store import INDEX, OUT, REPO_ROOT, load_validation

MODELS = REPO_ROOT / "models"


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


class Catalog:
    def __init__(self, jobs: JobManager):
        self.jobs = jobs
        self._lock = threading.Lock()
        self._swm: dict[str, tuple[int, dict]] = {}
        self._index_job: Job | None = None
        self._validating: dict[str, Job] = {}
        self.engine = self._engine()

    @staticmethod
    def _engine() -> dict[str, Any]:
        from swarm.brain import native

        ok = native.available()
        return {"native": ok, "info": native.build_info() if ok else "", "diagnosis": native.diagnosis()}

    # ------------------------------------------------------------------ models and packages
    def _swm_info(self, path: Path) -> dict:
        key, mtime = str(path), path.stat().st_mtime_ns
        hit = self._swm.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            info = swm_info(path)
        except (OSError, ValueError, KeyError) as exc:
            info = {"error": str(exc)}
        self._swm[key] = (mtime, info)
        return info

    def models(self) -> list[dict]:
        out = []
        for p in sorted(MODELS.rglob("*.swm")) if MODELS.is_dir() else []:
            info = self._swm_info(p)
            rel = _rel(p)
            out.append({"id": f"model:{rel}", "kind": "model", "label": p.stem, "path": rel, "info": info,
                        "brain": {"kind": "auto", "model": rel}, "deploy": "C++ engine on every device it fits",
                        "description": "trained network in models/ (python dev.py train)"})
        return out

    def packages(self) -> list[dict]:
        out = []
        for folder, pkg, err in discover_packages():
            lib = "library" if folder.parent == LIBRARY_DIRS[0] else "examples"
            if pkg is None:
                out.append({"name": folder.name, "root_rel": _rel(folder), "library": lib, "broken": err,
                            "status": "fail"})
                continue
            s = pkg.summary()
            s["library"] = lib
            s["removable"] = lib == "library"
            s.pop("root", None)
            report = load_validation(pkg.root, pkg.name)
            s["validation"] = report
            s["status"] = report["status"] if report else "pending"
            job = self._validating.get(s["root_rel"])
            s["validating"] = job.id if job is not None and job.state in ("queued", "running") else None
            out.append(s)
        return out

    def package(self, path: str) -> Package:
        """A discovered package by its repo-relative folder - the only packages the API acts on."""
        for folder, pkg, err in discover_packages():
            if _rel(folder) == path:
                if pkg is None:
                    raise PackageError(err)
                return pkg
        raise PackageError(f"no package at {path!r} in " + ", ".join(_rel(d) for d in LIBRARY_DIRS))

    def validate(self, pkg: Package, build: bool = False) -> Job:
        rel = _rel(pkg.root)
        with self._lock:
            job = self._validating.get(rel)
            if job is not None and job.state in ("queued", "running") and not build:
                return job
            job = self.jobs.submit("build" if build else "validate", {"path": rel},
                                   label=("compile " if build else "validate ") + pkg.name)
            self._validating[rel] = job
        return job

    def _auto_validate(self, packages: list[dict]) -> None:
        for s in packages:
            if s.get("broken") or s["status"] != "pending" or s.get("validating"):
                continue
            if s.get("format") == "swm" and not s.get("model_built"):
                continue  # nothing to validate until it is compiled
            try:
                self.validate(self.package(s["root_rel"]))
            except PackageError:
                continue

    # ------------------------------------------------------------------ presets
    def presets(self) -> tuple[list | None, bool]:
        """(presets, pending): the cached index, rebuilt by a worker when preset files changed."""
        fp = preset_fingerprint()
        cached = None
        try:
            data = json.loads(INDEX.read_text(encoding="utf-8"))
            cached = data.get("presets")
            if data.get("fingerprint") == json.loads(json.dumps(fp)):
                return cached, False
        except (OSError, json.JSONDecodeError):
            pass
        with self._lock:
            if self._index_job is None or self._index_job.state not in ("queued", "running"):
                self._index_job = self.jobs.submit("index", {}, label="index presets", on_done=self._store_index)
        return cached, True

    @staticmethod
    def _store_index(job: Job) -> None:
        if job.state == "done" and job.result:
            OUT.mkdir(parents=True, exist_ok=True)
            INDEX.write_text(json.dumps(job.result), encoding="utf-8")

    def invalidate_presets(self) -> None:
        try:
            INDEX.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------ everything
    def build(self) -> dict[str, Any]:
        packages = self.packages()
        self._auto_validate(packages)
        presets, pending = self.presets()
        devices = [{"name": p.name, "description": p.description, "cpu_mhz": p.cpu_mhz,
                    "macs_per_cycle": p.macs_per_cycle, "gmacs": p.cpu_mhz * p.macs_per_cycle / 1e3,
                    "ram_bytes": p.ram_bytes, "radio_bps": p.radio_bps, "radio_latency_s": p.radio_latency_s,
                    "tags": list(p.tags), "python": runs_python(p)} for p in DEVICE_PROFILES.values()]
        strategies = [{"name": s, "note": STRATEGY_NOTES[s], "builtin": True, "baseline": s == BASELINE}
                      for s in ALL_STRATEGIES]
        for s in packages:
            if s.get("type") == "tactic":
                strategies.append({"name": s["strategy"], "note": (s.get("validation") or {}).get("info", {}).get(
                    "summary") or s.get("description", ""), "builtin": False, "baseline": False,
                    "package": s["root_rel"], "status": s["status"], "title": s.get("title")})
        brains = [{"id": "oracle", "kind": "oracle", "label": "Oracle (parametric)", "brain": {"kind": "oracle"},
                   "deploy": "none - a stand-in with a chosen cost and accuracy",
                   "description": "no weights: answers with the ground truth at the chosen accuracy and costs the "
                                  "chosen MACs per frame - sweep hypothetical brain sizes without training"}]
        brains += self.models()
        for s in packages:
            if s.get("type") == "perception":
                brains.append({"id": f"package:{s['root_rel']}", "kind": "package", "label": s.get("title") or s["name"],
                               "brain": {"kind": "package", "model": s["root_rel"]}, "package": s["root_rel"],
                               "format": s.get("format"), "status": s["status"], "description": s.get("description", "")})
        rows = {"mission": MISSION_ROWS, "cost": COST_ROWS, "survival": SURVIVAL_ROWS}
        return {
            "engine": self.engine, "devices": devices, "strategies": strategies, "brains": brains,
            "packages": packages, "presets": presets or [], "presets_pending": pending,
            "metrics": {k: [{"key": r[0], "label": r[1], "fmt": r[2], "better": r[3]} for r in v]
                        for k, v in rows.items()},
            "check_metrics": list(CHECK_METRICS), "check_ops": ["<", "<=", ">", ">=", "==", "!="],
            "theatre": {"choices": THEATRE_CHOICES, "defaults": THEATRE_DEFAULTS},
            "jobs": [j.view(log_tail=0) for j in self.jobs.active()],
        }
