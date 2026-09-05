"""Runs one scenario end to end.

For every strategy (baseline first) the swarm is simulated with the
scenario's faults. If the scenario has faults, the *same* configuration is
also simulated **without** them (the "twin" run), so survivability can be
expressed as work / recall retained relative to an undisturbed swarm rather
than as raw counts. Afterwards the brain is stress-tested against the devices
in the swarm and the scenario's checks are evaluated.

All costs come from device profiles and the channel model, never from
wall-clock time, so a scenario gives identical numbers on every machine for
a given seed. Each strategy gets a fresh brain instance so results do not
depend on the order strategies run in.
"""

from __future__ import annotations

import math
import operator
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from swarm.brain import native
from swarm.devices import DeviceProfile, get_profile
from swarm.sim.brains import Brain, NativeBrain, NumpyBrain, OracleBrain
from swarm.sim.core import Simulator
from swarm.sim.strategies import SwarmConfig, build_swarm
from swarm.train.dataset import FramePool, make_dataset

from .scenario import Check, Faults, Scenario, split_strategy

REPO_ROOT = Path(__file__).resolve().parents[3]
INF = float("inf")
NAN = float("nan")


# ----------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------
@dataclass
class StrategyResult:
    strategy: str
    metrics: dict
    log: list[str]
    twin_metrics: dict | None = None  # the same run without faults (only when the scenario has faults)


@dataclass
class CheckResult:
    check: Check
    strategy: str
    actual: float
    passed: bool

    def describe(self) -> str:
        metric = f"{self.check.metric}/baseline" if self.check.ratio else self.check.metric
        actual = "n/a" if isinstance(self.actual, float) and math.isnan(self.actual) else f"{self.actual:.4g}"
        line = f"{'PASS' if self.passed else 'FAIL'}  {self.strategy:<18} {metric} {self.check.op} {self.check.value:g}"
        line += f"   (actual {actual})"
        if self.check.note:
            line += f"   # {self.check.note}"
        return line


@dataclass
class ScenarioResult:
    scenario: Scenario
    brain_kind: str
    brain_desc: str
    brain_note: str
    model_path: str | None
    resolved: dict
    strategies: list[StrategyResult]
    stress: list[dict]
    checks: list[CheckResult]
    wall_s: float

    def baseline(self) -> StrategyResult:
        return next(r for r in self.strategies if r.strategy == self.scenario.baseline)

    def by_strategy(self) -> dict[str, dict]:
        return {r.strategy: r.metrics for r in self.strategies}

    def checks_passed(self) -> bool:
        return all(c.passed for c in self.checks)


# ----------------------------------------------------------------------------
# brain resolution
# ----------------------------------------------------------------------------
def resolve_brain(sc: Scenario, override: str | None = None) -> tuple[str, Path | None, str, Callable[[], Brain]]:
    """Returns ``(kind, model_path, note, factory)``; ``factory()`` builds a fresh brain."""
    spec = sc.brain
    kind = override or spec.kind
    model: Path | None = None
    if spec.model:
        model = Path(spec.model)
        if not model.is_absolute():
            model = REPO_ROOT / model
    note = ""
    if kind == "auto":
        if model is not None and model.is_file():
            kind = "native" if native.available() else "numpy"
            if kind == "numpy":
                note = "C++ engine not built - numpy brain (RAM fit is estimated, not enforced)"
        else:
            kind = "oracle"
            note = (f"no model file at {spec.model!r} - oracle brain stands in "
                    f"(train one: python -m swarm.train.train_tiny_cnn)")

    if kind == "oracle":
        s = spec.input_size

        def make_oracle() -> Brain:
            return OracleBrain(macs=spec.macs, accuracy=spec.accuracy, input_shape=(1, s, s),
                               frontend_fraction=spec.frontend_fraction, feature_bytes=s * s, seed=sc.seed)

        return kind, None, note, make_oracle

    if model is None or not model.is_file():
        raise FileNotFoundError(f"brain kind {kind!r} needs a model file; {model} not found "
                                "(train one: python -m swarm.train.train_tiny_cnn)")
    if kind == "native":
        if not native.available():
            raise RuntimeError("brain kind 'native' requested but the C++ engine is not built "
                               "(cmake --preset mingw-arm64 && cmake --build --preset mingw-arm64)")
        return kind, model, note, lambda: NativeBrain(model)
    if kind == "numpy":
        return kind, model, note, lambda: NumpyBrain(model, int8="int8" in model.name)
    raise ValueError(f"unknown brain kind {kind!r}")


# ----------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------
def simulate(sc: Scenario, strategy: str, brain: Brain, profiles: list[DeviceProfile], bps: float, latency: float,
             faults: Faults, pool: FramePool | None, duration: float, verbose: bool = False):
    strat, pre = split_strategy(strategy)
    fps = sc.striped_fps() if strat == "striped" else sc.camera.fps
    cfg = SwarmConfig(strategy=strat, preprocess=pre, fps=fps, cam_size=sc.camera.size,
                      include_leader=sc.include_leader, confidence_threshold=sc.confidence_threshold,
                      hb_interval=sc.link.hb_interval, missed_beats=sc.link.missed_beats,
                      queue_limit=sc.queue_limit, p_object=sc.camera.p_object, frame_size=brain.input_shape[-1])
    sim = Simulator(sc.seed)
    sim.verbose = verbose
    channel, nodes = build_swarm(sim, cfg, profiles, brain, bps, latency, sc.link.loss, sc.seed, pool, sc.link.qos)

    def kill(who) -> None:
        if who == "leader":
            live = [n for n in nodes if n.alive]
            if not live:
                return
            idx = live[0].membership.leader()
        else:
            idx = int(who)
        nodes[idx].kill()

    for who, at in faults.kill:
        sim.schedule(at, kill, who)
    for who, at in faults.revive:
        sim.schedule(at, lambda w=who: nodes[int(w)].revive())
    sim.run(duration)
    return sim, channel, nodes


def _ratio(a: float, b: float) -> float:
    return a / b if b else NAN


def survivability(events: list[tuple[str, float, int]], learn_times: list[float]) -> dict[str, float]:
    """Timings around the *first* kill: how fast was it noticed, re-elected around, and worked around."""
    out = {"detect_s": NAN, "reelect_s": NAN, "recover_s": NAN}
    kills = [(t, n) for k, t, n in events if k == "kill"]
    if not kills:
        return out
    t_k, node_k = kills[0]
    suspects = [t for k, t, n in events if k == "suspect" and n == node_k and t >= t_k]
    out["detect_s"] = (min(suspects) - t_k) if suspects else INF
    leaders_before = [n for k, t, n in events if k == "leader" and t <= t_k]
    if leaders_before and leaders_before[-1] == node_k:
        after = [t for k, t, n in events if k == "leader" and t > t_k]
        out["reelect_s"] = (min(after) - t_k) if after else INF
    later = [t for t in learn_times if t > t_k]  # learn_times is chronological
    out["recover_s"] = (later[0] - t_k) if later else INF
    return out


def collect(sim: Simulator, channel, nodes, duration: float, twin: dict | None) -> dict:
    s = sim.metrics.summary(duration, len(nodes))
    d = {k: v for k, v in s.items() if k != "events"}
    d["fps_processed"] = s["frames_processed"] / duration
    d["objects_seen_ps"] = s["objects_captured"] / duration
    d["objects_reported_ps"] = s["objects_reported"] / duration
    d["gmac_ps"] = s["macs_total"] / duration / 1e9
    d["kmac_per_frame"] = s["macs_per_frame"] / 1e3
    d["channel_util"] = channel.utilisation()
    d["cameras_on"] = sum(1 for n in nodes if n.camera_on)
    d["energy_mj_per_frame"] = s["energy_j"] * 1e3 / s["frames_processed"] if s["frames_processed"] else NAN
    # where does the CPU load land? (leader = whoever led at the start; unrounded, unlike summary())
    leaders = [n for k, t, n in s["events"] if k == "leader"]
    leader = leaders[0] if leaders else 0
    busy = [min(1.0, sim.metrics.busy_by_node.get(i, 0.0) / duration) for i in range(len(nodes))]
    d["busy_leader"] = busy[leader] if leader < len(busy) else NAN
    workers = [b for i, b in enumerate(busy) if i != leader]
    d["busy_workers_max"] = max(workers) if workers else 0.0
    d.update(survivability(s["events"], sim.metrics.learn_times))
    if twin is not None:
        d["work_retained"] = _ratio(d["frames_processed"], twin["frames_processed"])
        d["recall_retained"] = _ratio(d["objects_reported"], twin["objects_reported"])
        d["frames_lost_to_fault"] = d["frames_dropped"] - twin["frames_dropped"]
    else:
        d["work_retained"] = d["recall_retained"] = d["frames_lost_to_fault"] = NAN
    return d


# ----------------------------------------------------------------------------
# stress
# ----------------------------------------------------------------------------
def run_stress(sc: Scenario, model_path: Path | None, profiles: list[DeviceProfile], brain_macs: int,
               frame_size: int) -> list[dict]:
    names = sc.stress.devices or tuple(dict.fromkeys(p.name for p in profiles))
    rows: list[dict] = []
    if model_path is None:
        for name in names:
            p = get_profile(name)
            dev_s = p.compute_seconds(brain_macs)
            rows.append({"device": name, "fits_ram": None, "ram_needed": None, "ram_cap": p.ram_bytes,
                         "device_ms_per_frame": dev_s * 1e3, "max_fps": 1.0 / dev_s if dev_s else INF,
                         "host_ms_per_frame": NAN, "accuracy": NAN, "engine": "oracle"})
        return rows
    from swarm.stress.device_stress import stress

    ds = make_dataset(sc.stress.frames, frame_size, seed=sc.seed + 7)
    for name in names:
        rows.append(asdict(stress(model_path, get_profile(name), fps=sc.camera.fps, x=ds.x, y=ds.y)))
    return rows


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne}


def _compare(actual: float, op: str, value: float) -> bool:
    if op not in _OPS:
        raise ValueError(f"unknown check operator {op!r}; use one of {sorted(_OPS)}")
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        return False
    return bool(_OPS[op](actual, value))


def evaluate_checks(sc: Scenario, results: list[StrategyResult]) -> list[CheckResult]:
    by = {r.strategy: r.metrics for r in results}
    base = by[sc.baseline]
    out: list[CheckResult] = []
    for c in sc.checks:
        targets = [r.strategy for r in results] if c.strategy == "*" else [c.strategy]
        for s in targets:
            if s not in by:
                out.append(CheckResult(c, s, NAN, False))
                continue
            if c.metric not in by[s]:
                raise KeyError(f"check refers to unknown metric {c.metric!r}; known: {sorted(by[s])}")
            v = by[s][c.metric]
            if c.ratio:
                v = _ratio(v, base[c.metric])
            out.append(CheckResult(c, s, v, _compare(v, c.op, c.value)))
    return out


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
def run_scenario(sc: Scenario, *, brain_kind: str | None = None, duration: float | None = None,
                 strategies: list[str] | None = None, with_stress: bool = True, verbose: bool = False) -> ScenarioResult:
    t0 = time.perf_counter()
    duration = duration or sc.duration
    names = list(strategies) if strategies else list(sc.strategy_order())
    names = [sc.baseline] + [n for n in names if n != sc.baseline]  # the baseline is always simulated, first

    profiles = [get_profile(n) for n in sc.swarm.profile_names()]
    bps = sc.link.bps or min(p.radio_bps for p in profiles)
    latency = sc.link.latency_s if sc.link.latency_s is not None else max(p.radio_latency_s for p in profiles)
    kind, model_path, note, factory = resolve_brain(sc, brain_kind)
    probe = factory()
    frame_size = probe.input_shape[-1]
    pool = FramePool(frame_size, per_class=64, seed=sc.seed + 999) if kind != "oracle" else None
    n = len(profiles)
    striped_fps = sc.striped_fps()
    resolved = {
        "profiles": [p.name for p in profiles],
        "bps": bps,
        "latency_s": latency,
        "duration": duration,
        "striped_fps": striped_fps,
        "brain_macs": probe.macs,
        "brain_input_bytes": probe.input_bytes,
        "brain_feature_bytes": probe.feature_bytes,
        "frontend_macs": probe.frontend_macs,
        "head_macs": probe.head_macs,
        # frames per second a single node's CPU must handle under each design
        "demand_fps": {
            "local": sc.camera.fps,
            "central leader": sc.camera.fps * n,
            "striped per node": striped_fps / (n if sc.include_leader or n == 1 else n - 1),
        },
    }

    results: list[StrategyResult] = []
    for name in names:
        sim, ch, nodes = simulate(sc, name, factory(), profiles, bps, latency, sc.faults, pool, duration, verbose)
        twin = None
        if sc.faults:
            tsim, tch, tnodes = simulate(sc, name, factory(), profiles, bps, latency, Faults(), pool, duration)
            twin = collect(tsim, tch, tnodes, duration, None)
        results.append(StrategyResult(name, collect(sim, ch, nodes, duration, twin), sim.log_lines, twin))

    stress_rows = run_stress(sc, model_path, profiles, probe.macs, frame_size) if with_stress else []
    checks = evaluate_checks(sc, results)
    shown_path = None
    if model_path:
        try:
            shown_path = model_path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            shown_path = str(model_path)
    return ScenarioResult(sc, kind, probe.describe(), note, shown_path, resolved, results, stress_rows, checks,
                          time.perf_counter() - t0)
