"""Training brains in the arena by evolution - one side, or both swarms against each other (co-evolution).

Every generation is one batch of simulations, spread over the CPU cores (``parallel``):
    each defender variant (weights + noise) against the current attackers and a sparring partner,
    each attacker variant against the current defenders and a sparring partner,
    the current defenders against every built-in attacker and the current attackers against every
    built-in defender (absolute skill: "who learns faster"), and the current pair head to head
    ("who is winning").
Variants meet identical situations with identical random numbers (common random numbers, ``core.Noise``),
so their ranking reflects the brain, not luck. Skill and head-to-head are played on the same fixed
situations every generation, so a curve moves only when a brain changes. Sparring partners are earlier
versions of the opponent (a hall of fame: its latest snapshots plus a uniform sample of its whole history,
so a tactic beaten long ago cannot quietly come back) and the built-ins, so neither side can win by
over-fitting the other's latest quirk.

The output folder holds defender.npz / attacker.npz / recon.npz (the latest weights, rewritten every
generation), defender_best.npz / attacker_best.npz / recon_best.npz (the weights that scored best on the
fixed skill test so far), history.json (one row per generation, rewritten every few seconds) and
matches/gen_NNNN.json (a recorded head-to-head every few generations, for the 3D view).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from swarm._arch import describe_python

from .brains import BUILTINS, Brain, BrainError, NeuralBrain, load_brain
from .clash import ClashEnv
from .config import ClashConfig, ConfigError, ReconConfig, TrainConfig, to_dict
from .core import derive
from .es import ES
from .parallel import Workers, clash_chunk, keep_awake, recon_chunk
from .recon import ReconEnv
from .run import play_clash, play_recon

EVAL_EPISODES = 3    # fixed situations per built-in opponent, every generation ("skill")
H2H_EPISODES = 6     # fixed situations: the current brains head to head
RECON_EVAL = 8       # fixed situations: the current recon brain, and the built-in references
HOF_EVERY = 5        # generations between hall-of-fame snapshots of each learning side
HOF_RECENT = 4       # sparring partners: the opponent's latest snapshots ...
HOF_ARCHIVE = 4      # ... and a uniform sample of all its older ones, so a long run cannot forget old tactics
MAX_MATCHES = 120    # recorded matches per run: a long run records less often
HISTORY_EVERY = 2.0  # s between rewrites of history.json (it grows with the run)
TRAIN, EVAL, H2H = 1, 2, 3  # key families: training situations (new every generation), the fixed tests

ROLE_SEED = {"recon": 37, "defender": 11, "attacker": 23}

Report = Callable[[dict], None]


def _builtins(role: str) -> list[Brain]:
    return [cls() for cls in BUILTINS.values() if cls.role == role]


def _start(role: str, spec: str, tc: TrainConfig, learn: bool) -> Brain:
    if spec in ("", "fresh"):
        return NeuralBrain(role, tc.hidden, seed=tc.seed * 1000 + ROLE_SEED[role], name=f"mesh_{role}")
    brain = load_brain(spec, role)
    if learn and not isinstance(brain, NeuralBrain):
        raise BrainError(f"{spec!r} is not a mesh network, so it cannot learn: start the {role}s from 'fresh' or "
                         f"a mesh brain, or stop training that side to use {spec!r} as a fixed opponent")
    return brain


def _mean(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return round(float(x.mean()), 4) if x.size else None


def _record_every(tc: TrainConfig) -> int:
    return max(tc.record_every, math.ceil(tc.generations / MAX_MATCHES))


class _Side:
    """One side of the training: its network and optimiser if it learns, else a fixed brain."""

    def __init__(self, role: str, brain: Brain, learn: bool, tc: TrainConfig, seed: int, builtins: list[Brain]):
        self.role, self.learn = role, learn
        self.brain = brain
        self.builtins = builtins  # shared instances, so a batch has as few brain groups as possible
        self.es = ES(brain.theta, tc.sigma, tc.lr, tc.population // 2, seed) if learn else None
        self.recent: list[np.ndarray] = []
        self.archive: list[np.ndarray] = []
        self._retired = 0                                  # snapshots that left the recent window so far
        self._pick = np.random.default_rng(seed + 7)       # reservoir sampling of the archive
        self.best: tuple[float, int] = (-math.inf, 0)   # best skill so far, and the generation that had it

    @property
    def hof(self) -> list[np.ndarray]:
        return self.recent + self.archive

    @property
    def current(self) -> tuple[Brain, np.ndarray | None]:
        return (self.brain, self.es.theta.copy()) if self.es else (self.brain, None)

    def standalone(self, theta: np.ndarray | None = None) -> Brain:
        if not self.es:
            return self.brain
        return self.brain.with_theta((self.es.theta if theta is None else theta).copy())

    def sparring(self, n: int, rng: np.random.Generator) -> list[tuple[Brain, np.ndarray | None]]:
        """Opponents for the other side's ``n`` scenarios: the current brain first, then past selves/built-ins."""
        if not self.es:
            return [self.current] * n
        pool = [(self.brain, h) for h in self.hof] + [(b, None) for b in self.builtins]
        return [self.current] + [pool[int(rng.integers(len(pool)))] for _ in range(n - 1)]

    def snapshot(self, gen: int) -> None:
        if not (self.es and gen % HOF_EVERY == 0):
            return
        if len(self.recent) == HOF_RECENT:
            old = self.recent.pop(0)
            self._retired += 1
            if len(self.archive) < HOF_ARCHIVE:
                self.archive.append(old)
            else:  # reservoir sampling: every retired snapshot ends up in the archive with equal chance
                j = int(self._pick.integers(self._retired))
                if j < HOF_ARCHIVE:
                    self.archive[j] = old
        self.recent.append(self.es.theta.copy())


def _write_json(path: Path, doc) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


class _History:
    """history.json: one row per generation, rewritten at most every HISTORY_EVERY seconds and at the end."""

    def __init__(self, path: Path, head: dict):
        self.path, self.head, self.rows = path, head, []
        self._written = -math.inf

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if time.perf_counter() - self._written >= HISTORY_EVERY:
            self.flush()

    def flush(self) -> None:
        _write_json(self.path, {**self.head, "history": self.rows})
        self._written = time.perf_counter()


def train(tc: TrainConfig, clash: ClashConfig | None = None, recon: ReconConfig | None = None,
          out_dir: str | Path = "out/arena/train", report: Report | None = None,
          should_stop: Callable[[], bool] | None = None) -> dict:
    """Runs ``tc.generations`` generations (or until ``should_stop()``); returns a summary of the run. The
    computer is kept awake meanwhile (see ``parallel.keep_awake``)."""
    out = Path(out_dir)
    (out / "matches").mkdir(parents=True, exist_ok=True)
    report = report or (lambda event: None)
    stop = should_stop or (lambda: False)
    if tc.mode == "clash" and not (tc.learn_defenders or tc.learn_attackers):
        raise ConfigError("nothing to train: tick defenders, attackers or both")
    with keep_awake() as awake:
        def emit(event: dict) -> None:
            report({**event, "awake": awake, "python": describe_python()} if event["type"] == "start" else event)

        if tc.mode == "recon":
            return _train_recon(tc, recon or ReconConfig(), out, emit, stop)
        return _train_clash(tc, clash or ClashConfig(), out, emit, stop)


def _save_match(out: Path, gen: int, env, result: dict) -> str:
    name = f"gen_{gen:04d}.json"
    _write_json(out / "matches" / name, {"kind": "match", "gen": gen, "replay": env.replay(),
                                          "result": {k: _mean(v) for k, v in result.items()}})
    return name


def _train_clash(tc: TrainConfig, cfg: ClashConfig, out: Path, emit: Report, stop) -> dict:
    t0 = time.perf_counter()
    ev_att, ev_def = _builtins("attacker"), _builtins("defender")
    d = _Side("defender", _start("defender", tc.start_defender, tc, tc.learn_defenders), tc.learn_defenders, tc,
              tc.seed * 2 + 1, ev_def)
    a = _Side("attacker", _start("attacker", tc.start_attacker, tc, tc.learn_attackers), tc.learn_attackers, tc,
              tc.seed * 2 + 2, ev_att)
    P, E = tc.population, tc.episodes
    every = _record_every(tc)
    eval_keys = [derive(tc.seed, EVAL, i) for i in range(EVAL_EPISODES)]
    h2h_keys = [derive(tc.seed, H2H, i) for i in range(H2H_EPISODES)]
    log = _History(out / "history.json", {"mode": "clash", "train": to_dict(tc), "config": to_dict(cfg)})
    matches: list[str] = []
    stopped = False
    with Workers(tc.workers) as pool:
        emit({"type": "start", "mode": "clash", "generations": tc.generations, "workers": pool.n,
              "record_every": every,
              "sides": {"defender": {"learn": d.learn, "brain": d.brain.info()},
                        "attacker": {"learn": a.learn, "brain": a.brain.info()}},
              "builtins": {"defender": [b.name for b in ev_def], "attacker": [b.name for b in ev_att]}})
        for gen in range(1, tc.generations + 1):
            if stop():
                stopped = True
                break
            g0 = time.perf_counter()
            seed = tc.seed * 100_003 + gen
            rng = np.random.default_rng(seed)
            ent_d, ent_a, keys, tags = [], [], [], []

            def add(ks, de, ae, tag, d_rows=None, a_rows=None):
                for i, k in enumerate(ks):
                    ent_d.append((de[0], de[1] if d_rows is None else d_rows[i]))
                    ent_a.append((ae[0], ae[1] if a_rows is None else a_rows[i]))
                    keys.append(k)
                    tags.append(tag)

            cur_d, cur_a = d.current, a.current   # this generation's tests all play these weights
            if d.es:
                pop = d.es.ask()
                for e, opp in enumerate(a.sparring(E, rng)):
                    add([derive(seed, TRAIN, e)] * P, (d.brain, None), opp, ("d",), d_rows=pop)
            if a.es:
                pop = a.es.ask()
                for e, opp in enumerate(d.sparring(E, rng)):
                    add([derive(seed, TRAIN, e)] * P, opp, (a.brain, None), ("a",), a_rows=pop)
            if d.es:
                for b in ev_att:
                    add(eval_keys, cur_d, (b, None), ("ed", b.name))
            if a.es:
                for b in ev_def:
                    add(eval_keys, (b, None), cur_a, ("ea", b.name))
            add(h2h_keys, cur_d, cur_a, ("h2h",))
            match: list[str] = []

            def record() -> None:   # in this process, while the workers simulate: head-to-head game 1, watchable
                if gen % every == 0 or gen == 1 or gen == tc.generations:
                    env = ClashEnv(cfg, 1, seed, record=True, keys=h2h_keys[:1])
                    r = play_clash(env, d.standalone(cur_d[1]), a.standalone(cur_a[1]), seed)
                    match.append(_save_match(out, gen, env, r))

            s = pool.play(clash_chunk, cfg, keys, seed, ent_d, ent_a, meanwhile=record)
            kind = np.array([t[0] for t in tags])
            row: dict = {"gen": gen}
            for side, key, fit in ((d, "d", "def_fitness"), (a, "a", "att_fitness")):
                if side.es:
                    f = s[fit][kind == key].reshape(E, P).mean(0)
                    row[f"{side.role}_step"] = round(side.es.tell(f), 4)
                    row[f"{side.role}_pop"] = _mean(f)
            h = kind == "h2h"
            row["h2h"] = {k: _mean(s[k][h]) for k in ("intercepted", "leaked", "held_off", "advance", "overkill", "misses")}
            row["defender_vs_builtin"] = {b.name: _mean(s["intercepted"][[t == ("ed", b.name) for t in tags]])
                                          for b in ev_att} if d.es else {}
            row["attacker_vs_builtin"] = {b.name: _mean(s["leaked"][[t == ("ea", b.name) for t in tags]])
                                          for b in ev_def} if a.es else {}
            row["defender_skill"] = _mean(s["intercepted"][kind == "ed"])
            row["attacker_skill"] = _mean(s["leaked"][kind == "ea"])
            for side, skill, cur in ((d, row["defender_skill"], cur_d), (a, row["attacker_skill"], cur_a)):
                if side.es and skill is not None and skill >= side.best[0]:
                    side.best = (skill, gen)
                    side.standalone(cur[1]).save(out / f"{side.role}_best.npz", generations=gen - 1, skill=skill,
                                                 seconds=round(time.perf_counter() - t0, 1))
            row["best"] = {side.role: {"gen": side.best[1], "skill": side.best[0]} for side in (d, a)
                           if side.es and side.best[1]}
            d.snapshot(gen)
            a.snapshot(gen)
            for side in (d, a):
                if side.es:
                    side.standalone().save(out / f"{side.role}.npz", generations=gen,
                                           seconds=round(time.perf_counter() - t0, 1),
                                           trained_against=("co-evolution" if d.es and a.es else
                                                            (a if side is d else d).brain.id))
            if match:
                matches.append(match[0])
                row["match"] = match[0]
            row["seconds"] = round(time.perf_counter() - g0, 2)
            log.add(row)
            emit({"type": "generation", **row})
    log.flush()
    return {"mode": "clash", "generations": len(log.rows), "stopped": stopped, "workers": pool.n,
            "seconds": round(time.perf_counter() - t0, 1), "history": log.rows, "matches": matches,
            "brains": {s.role: str(out / f"{s.role}.npz") for s in (d, a) if s.es},
            "best": {s.role: {"gen": s.best[1], "skill": s.best[0], "path": str(out / f"{s.role}_best.npz")}
                     for s in (d, a) if s.es and s.best[1]}}


def _train_recon(tc: TrainConfig, cfg: ReconConfig, out: Path, emit: Report, stop) -> dict:
    t0 = time.perf_counter()
    brain = _start("recon", tc.start_recon, tc, True)
    es = ES(brain.theta, tc.sigma, tc.lr, tc.population // 2, tc.seed * 2 + 3)
    P, E = tc.population, tc.episodes
    every = _record_every(tc)
    fields = ("fitness", "coverage", "t90", "reported", "connected", "camera_duty", "cpu")
    eval_keys = [derive(tc.seed, EVAL, i) for i in range(RECON_EVAL)]
    builtins = _builtins("recon")
    log: _History | None = None
    matches: list[str] = []
    stopped = False
    best: tuple[float, int] = (-math.inf, 0)
    with Workers(tc.workers) as pool:
        # the built-in references play the same fixed situations the current brain is measured on
        n = RECON_EVAL
        s = pool.play(recon_chunk, cfg, eval_keys * len(builtins), tc.seed, [(b, None) for b in builtins for _ in eval_keys])
        refs = {b.name: {k: _mean(s[k][i * n:(i + 1) * n]) for k in fields} for i, b in enumerate(builtins)}
        log = _History(out / "history.json", {"mode": "recon", "train": to_dict(tc), "config": to_dict(cfg),
                                              "references": refs})
        emit({"type": "start", "mode": "recon", "generations": tc.generations, "references": refs,
              "workers": pool.n, "record_every": every, "sides": {"recon": {"learn": True, "brain": brain.info()}}})
        for gen in range(1, tc.generations + 1):
            if stop():
                stopped = True
                break
            g0 = time.perf_counter()
            seed = tc.seed * 100_003 + gen
            pop = es.ask()
            cur = es.theta.copy()
            keys = [derive(seed, TRAIN, e) for e in range(E) for _p in range(P)] + eval_keys
            entries = [(brain, pop[p]) for _e in range(E) for p in range(P)] + [(brain, cur)] * n
            match: list[str] = []

            def record() -> None:   # in this process, while the workers simulate
                if gen % every == 0 or gen == 1 or gen == tc.generations:
                    env = ReconEnv(cfg, 1, seed, record=True, keys=eval_keys[:1])
                    match.append(_save_match(out, gen, env, play_recon(env, brain.with_theta(cur), seed)))

            s = pool.play(recon_chunk, cfg, keys, seed, entries, meanwhile=record)
            m = P * E
            f = s["fitness"][:m].reshape(E, P).mean(0)
            current = {k: _mean(s[k][m:]) for k in fields}
            row = {"gen": gen, "recon_step": round(es.tell(f), 4), "recon_pop": _mean(f), "current": current}
            if current["fitness"] is not None and current["fitness"] >= best[0]:
                best = (current["fitness"], gen)
                brain.with_theta(cur).save(out / "recon_best.npz", generations=gen - 1, skill=best[0],
                                           seconds=round(time.perf_counter() - t0, 1), trained_against="recon")
            row["best"] = {"recon": {"gen": best[1], "skill": best[0]}}
            brain.with_theta(es.theta.copy()).save(out / "recon.npz", generations=gen,
                                                   seconds=round(time.perf_counter() - t0, 1),
                                                   trained_against="recon")
            if match:
                matches.append(match[0])
                row["match"] = match[0]
            row["seconds"] = round(time.perf_counter() - g0, 2)
            log.add(row)
            emit({"type": "generation", **row})
    log.flush()
    return {"mode": "recon", "generations": len(log.rows), "stopped": stopped, "workers": pool.n,
            "seconds": round(time.perf_counter() - t0, 1), "history": log.rows, "matches": matches,
            "references": refs, "brains": {"recon": str(out / "recon.npz")},
            "best": {"recon": {"gen": best[1], "skill": best[0], "path": str(out / "recon_best.npz")}} if best[1] else {}}
