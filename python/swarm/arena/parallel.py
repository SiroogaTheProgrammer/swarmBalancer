"""Training on every CPU core: a generation's episodes are split into chunks that run in worker processes.

Training gives every episode a key (see :class:`~swarm.arena.core.Noise`): its situation and every random
number it draws come from that key alone, so an episode plays out the same whether it runs alone, in one
big batch or in a chunk on another core. How a generation is split - and on how many cores - changes the
speed, never the result. (A Python policy that draws its own random numbers per batch is the exception.)

Worker processes start once per training run (``spawn``: each imports the arena afresh), receive each
chunk's brains and weights by pickling, and exit with the run - or as soon as the process that started
them dies, so an aborted run leaves nothing behind.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from typing import Callable, Iterator

import numpy as np

from .brains import MixedBrain
from .clash import ClashEnv
from .config import ClashConfig, ReconConfig
from .recon import ReconEnv
from .run import play_clash, play_recon

MIN_CHUNK = 6   # episodes: a smaller simulation spends most of its time on per-step overhead

Entry = tuple  # (brain, its weights for this episode or None)


def cpu_workers(requested: int = 0) -> int:
    """Worker processes to use: ``requested``, or one per CPU core for 0."""
    return max(1, int(requested) if requested and requested > 0 else (os.cpu_count() or 1))


@contextmanager
def keep_awake() -> Iterator[bool]:
    """While the block runs, asks Windows not to go to sleep (or into Modern Standby) because nobody is using
    the computer: a long run on a laptop would otherwise pause as soon as it idles. Closing the lid or the
    power button still puts it to sleep (the run carries on after wake-up). Yields whether the request was
    made; elsewhere it does nothing (use caffeinate or systemd-inhibit there)."""
    if sys.platform != "win32":
        yield False
        return
    import ctypes

    continuous, system_required = 0x80000000, 0x00000001
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint
    ok = bool(kernel32.SetThreadExecutionState(continuous | system_required))
    try:
        yield ok
    finally:
        if ok:
            kernel32.SetThreadExecutionState(continuous)  # back to normal: the computer may sleep again


def split(n: int, workers: int) -> list[slice]:
    """``n`` episodes as up to ``workers`` chunks of near-equal size, none smaller than MIN_CHUNK."""
    k = max(1, min(workers, n // MIN_CHUNK))
    bounds = np.linspace(0, n, k + 1).round().astype(int)
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def mixed(role: str, entries: list[Entry]) -> MixedBrain:
    """Per-episode (brain, weights or None) -> one brain that groups the episodes sharing a brain."""
    groups: dict[int, list] = {}
    for b, (brain, row) in enumerate(entries):
        g = groups.setdefault(id(brain), [brain, [], []])
        g[1].append(b)
        g[2].append(row)
    return MixedBrain(role, [(brain, np.array(idx), None if rows[0] is None else np.stack(rows))
                             for brain, idx, rows in groups.values()])


def clash_chunk(cfg: ClashConfig, keys: np.ndarray, seed: int, defenders: list[Entry],
                attackers: list[Entry]) -> dict[str, np.ndarray]:
    env = ClashEnv(cfg, len(keys), seed, keys=keys)
    return play_clash(env, mixed("defender", defenders), mixed("attacker", attackers), seed)


def recon_chunk(cfg: ReconConfig, keys: np.ndarray, seed: int, entries: list[Entry]) -> dict[str, np.ndarray]:
    env = ReconEnv(cfg, len(keys), seed, keys=keys)
    return play_recon(env, mixed("recon", entries), seed)


def _watch_parent() -> None:
    """Runs in each worker: exit the moment the training process is gone (e.g. killed by ABORT)."""
    parent = mp.parent_process()
    if parent is None:
        return

    def watch() -> None:
        from multiprocessing.connection import wait

        wait([parent.sentinel])
        os._exit(0)

    threading.Thread(target=watch, name="watch-parent", daemon=True).start()


class Workers:
    """Plays batches of keyed episodes on ``n`` worker processes (``n == 1``: in this process)."""

    def __init__(self, n: int = 0):
        self.n = cpu_workers(n)
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> "Workers":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _start(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # one numpy thread per worker: the cores are already shared out between processes
            for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            self._pool = ProcessPoolExecutor(self.n, mp_context=mp.get_context("spawn"), initializer=_watch_parent)
        return self._pool

    def play(self, fn: Callable, cfg, keys, seed: int, *sides: list[Entry],
             meanwhile: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
        """Plays one episode per key with ``fn`` (:func:`clash_chunk` or :func:`recon_chunk`) and the per-episode
        brain entries of each side; returns the per-episode results in order. ``meanwhile()`` runs in this
        process while the workers simulate."""
        keys = np.asarray(keys, dtype=np.uint64)
        parts = split(len(keys), self.n)
        tasks = [(cfg, keys[s], seed, *[side[s] for side in sides]) for s in parts]
        if len(tasks) == 1:
            results = [fn(*tasks[0])]
            if meanwhile:
                meanwhile()
        else:
            try:
                futures = [self._start().submit(fn, *t) for t in tasks]
                if meanwhile:
                    meanwhile()
                results = [f.result() for f in futures]
            except BrokenProcessPool as exc:
                self._pool = None
                raise RuntimeError(f"a simulation worker process died ({exc}). Calling train() from your own "
                                   "script? Put the call under `if __name__ == '__main__':` (worker processes "
                                   "import your script), or set workers=1") from None
        return {k: np.concatenate([r[k] for r in results]) for k in results[0]}
