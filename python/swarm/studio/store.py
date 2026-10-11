"""Where the studio keeps things: ``out/studio/`` (git-ignored).

    runs/<id>/run.json        playback document: scenario, metrics, checks and each strategy's trace
    runs/<id>/meta.json       small summary for the run list
    runs/<id>/frames_<i>.npy  what the cameras saw (uint8 N x H x W) per strategy, fids_<i>.npy their frame ids
    validation/<pkg>.json     cached validation reports (stale once a package file changes)
    exports/                  device artefacts from swarm.sdk export
    jobs/<id>/                scratch space of running worker processes
    presets.json              index of the preset files (rebuilt by a worker when they change)
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "out" / "studio"
RUNS = OUT / "runs"
VALIDATION = OUT / "validation"
EXPORTS = OUT / "exports"
JOBS = OUT / "jobs"
INDEX = OUT / "presets.json"
KEEP_RUNS = 40
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id or ""):
        raise KeyError(f"bad run id {run_id!r}")
    d = RUNS / run_id
    if not (d / "run.json").is_file():
        raise KeyError(f"no run {run_id}")
    return d


def list_runs() -> list[dict]:
    out = []
    if RUNS.is_dir():
        for d in sorted(RUNS.iterdir(), reverse=True):
            meta = d / "meta.json"
            if RUN_ID_RE.match(d.name) and meta.is_file():
                try:
                    out.append(json.loads(meta.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
    return out


def prune_runs(keep: int = KEEP_RUNS) -> None:
    if not RUNS.is_dir():
        return
    runs = sorted((d for d in RUNS.iterdir() if d.is_dir() and RUN_ID_RE.match(d.name)), reverse=True)
    for d in runs[keep:]:
        shutil.rmtree(d, ignore_errors=True)


def delete_run(run_id: str) -> None:
    shutil.rmtree(run_dir(run_id))


# ----------------------------------------------------------------------------
# validation cache
# ----------------------------------------------------------------------------
def package_fingerprint(root: Path) -> str:
    h = hashlib.sha1()
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            st = p.stat()
            h.update(f"{p.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _validation_file(root: Path, name: str) -> Path:
    tag = hashlib.sha1(str(root.resolve()).lower().encode("utf-8")).hexdigest()[:10]
    return VALIDATION / f"{name}-{tag}.json"


def load_validation(root: Path, name: str) -> dict | None:
    """The cached report if no package file changed since it was made, else None."""
    f = _validation_file(root, name)
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("fingerprint") != package_fingerprint(root):
        return None
    return data.get("report")


def save_validation(root: Path, name: str, report: dict) -> None:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    payload = {"fingerprint": package_fingerprint(root), "saved": time.time(), "report": report}
    _validation_file(root, name).write_text(json.dumps(payload), encoding="utf-8")


# ----------------------------------------------------------------------------
# camera frames of a run, served one at a time to the sensor view
# ----------------------------------------------------------------------------
class FrameCache:
    def __init__(self, size: int = 6):
        self._lru: OrderedDict[tuple[str, int], tuple[dict[int, int], np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()
        self._size = size

    def frame(self, run_id: str, strategy: int, fid: int) -> np.ndarray | None:
        key = (run_id, int(strategy))
        with self._lock:
            entry = self._lru.get(key)
            if entry is None:
                d = run_dir(run_id)
                fpath, ipath = d / f"fids_{strategy}.npy", d / f"frames_{strategy}.npy"
                if not fpath.is_file():
                    return None
                fids = np.load(fpath)
                images = np.load(ipath)  # small (~1 KB per frame); no mmap, so a deleted run frees its files
                entry = ({int(f): i for i, f in enumerate(fids)}, images)
                self._lru[key] = entry
                while len(self._lru) > self._size:
                    self._lru.popitem(last=False)
            self._lru.move_to_end(key)
        index, images = entry
        row = index.get(int(fid))
        return None if row is None else np.asarray(images[row])

    def forget(self, run_id: str) -> None:
        with self._lock:
            for key in [k for k in self._lru if k[0] == run_id]:
                del self._lru[key]
