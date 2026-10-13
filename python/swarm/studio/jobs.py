"""Background jobs: each runs ``python -m swarm.studio.worker`` in its own process.

A separate process per job keeps brain code out of the server, lets a long simulation be cancelled, and
bounds every job by a timeout. Progress and structured events (training generations) arrive as protocol
lines on the worker's stdout; any other output (a brain's prints) is kept as log text for the UI.
``stop`` asks a training job to finish its generation and save (``cancel`` kills it).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from swarm._arch import native_python

from .store import JOBS, new_id

TIMEOUTS = {"test": 900.0, "validate": 300.0, "train": 48 * 3600.0}
PYTHON_DIR = Path(__file__).resolve().parents[2]
MAX_EVENTS = 5000   # events kept per job; older ones are dropped (a long run's history is on disk)


@dataclass
class Job:
    id: str
    kind: str
    label: str
    state: str = "queued"          # queued | running | done | error | cancelled
    progress: str = ""
    log: deque = field(default_factory=lambda: deque(maxlen=400))
    events: list = field(default_factory=list)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)
    on_done: Callable[["Job"], None] | None = None
    _proc: subprocess.Popen | None = None
    _cancel: bool = False
    events_base: int = 0           # number of the first event still kept

    def add_event(self, event: dict) -> None:
        self.events.append(event)
        if len(self.events) > MAX_EVENTS:
            drop = MAX_EVENTS // 2
            del self.events[:drop]
            self.events_base += drop

    def view(self, since: int = 0, log_tail: int = 40) -> dict[str, Any]:
        start = max(0, min(int(since) - self.events_base, len(self.events)))
        return {"id": self.id, "kind": self.kind, "label": self.label, "state": self.state, "progress": self.progress,
                "log": list(self.log)[-log_tail:], "created": self.created, "started": self.started,
                "finished": self.finished, "result": self.result, "error": self.error, "meta": self.meta,
                "events": self.events[start:], "next": self.events_base + len(self.events),
                "elapsed": (self.finished or time.time()) - (self.started or self.created)}


class JobManager:
    def __init__(self, max_parallel: int = 2):
        self._jobs: dict[str, Job] = {}
        self._queue: deque[tuple[Job, dict]] = deque()
        self._lock = threading.Lock()
        self._running = 0
        self._max = max_parallel
        self._closed = False

    def submit(self, kind: str, spec: dict, label: str = "", on_done: Callable[[Job], None] | None = None,
               meta: dict | None = None) -> Job:
        job = Job(new_id(), kind, label or kind, on_done=on_done, meta=dict(meta or {}))
        with self._lock:
            if self._closed:
                raise RuntimeError("the studio is shutting down")
            self._jobs[job.id] = job
            self._queue.append((job, dict(spec, kind=kind)))
            for old in [j for j in self._jobs.values() if j.finished and time.time() - j.finished > 3600]:
                del self._jobs[old.id]
        self._pump()
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            return self._jobs[job_id]

    def active(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.state in ("queued", "running")]

    def stop(self, job_id: str) -> Job:
        """Asks a running job to wrap up (training saves its brains); a queued one is simply cancelled."""
        job = self.get(job_id)
        if job.state == "queued":
            return self.cancel(job_id)
        if job.state == "running":
            try:
                (JOBS / job.id / "stop").touch()
            except OSError:
                pass
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job._cancel = True
        with self._lock:
            queued = [(j, s) for j, s in self._queue if j is job]
            for item in queued:
                self._queue.remove(item)
        if queued:
            self._finish(job, "cancelled", error="cancelled before it started")
        elif job._proc is not None and job._proc.poll() is None:
            job._proc.kill()
        return job

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            jobs = list(self._jobs.values())
            self._queue.clear()
        for j in jobs:
            if j._proc is not None and j._proc.poll() is None:
                j._proc.kill()

    # ------------------------------------------------------------------ internals
    def _pump(self) -> None:
        with self._lock:
            while self._running < self._max and self._queue and not self._closed:
                job, spec = self._queue.popleft()
                self._running += 1
                threading.Thread(target=self._run, args=(job, spec), name=f"job-{job.id}", daemon=True).start()

    def _finish(self, job: Job, state: str, result: dict | None = None, error: str | None = None) -> None:
        job.state, job.result, job.error = state, result, error
        job.finished = time.time()
        if job.on_done is not None:
            try:
                job.on_done(job)
            except Exception as exc:  # noqa: BLE001
                job.log.append(f"post-processing failed: {exc}")

    def _run(self, job: Job, spec: dict) -> None:
        job_dir = JOBS / job.id
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(PYTHON_DIR) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            # training is hours of numpy: on Windows on ARM it runs on the native interpreter if there is one
            exe = (native_python() if job.kind == "train" else None) or Path(sys.executable)
            if exe != Path(sys.executable):
                job.log.append(f"running on the native python in {exe.parent} (about twice as fast as emulated)")
            job.state, job.started = "running", time.time()
            job._proc = subprocess.Popen([str(exe), "-m", "swarm.studio.worker", str(job_dir)],
                                         cwd=PYTHON_DIR.parent, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                         creationflags=flags)
            timer = threading.Timer(TIMEOUTS.get(job.kind, 600.0), job._proc.kill)
            timer.daemon = True
            timer.start()
            assert job._proc.stdout is not None
            for line in job._proc.stdout:
                line = line.rstrip("\r\n")
                if line.startswith("\x1e"):
                    try:
                        msg = json.loads(line[1:])
                    except json.JSONDecodeError:
                        continue
                    if "progress" in msg:
                        job.progress = str(msg["progress"])
                    if "log" in msg:
                        job.log.append(str(msg["log"]))
                    if "event" in msg:
                        job.add_event(msg["event"])
                elif line:
                    job.log.append(line)
            rc = job._proc.wait()
            timer.cancel()
            if job._cancel:
                self._finish(job, "cancelled", error="cancelled")
            elif rc == 0 and (job_dir / "result.json").is_file():
                self._finish(job, "done", result=json.loads((job_dir / "result.json").read_text(encoding="utf-8")))
            else:
                err = f"worker exited with code {rc}"
                if (job_dir / "error.json").is_file():
                    info = json.loads((job_dir / "error.json").read_text(encoding="utf-8"))
                    err = info.get("error", err)
                    job.log.extend(info.get("traceback", "").splitlines()[-12:])
                elif rc != 0 and time.time() - (job.started or 0) >= TIMEOUTS.get(job.kind, 600.0) - 1:
                    err = f"timed out after {TIMEOUTS.get(job.kind, 600.0):.0f} s"
                self._finish(job, "error", error=err)
        except Exception as exc:  # noqa: BLE001
            self._finish(job, "error", error=f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            with self._lock:
                self._running -= 1
            self._pump()
