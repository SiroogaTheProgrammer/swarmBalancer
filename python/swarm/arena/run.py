"""Running a test: one call returns a JSON-ready report - score cards, a verdict, the same situations
replayed with every built-in alternative for comparison, and a replay of the first episode for the 3D view.

Tests:
    recon    the selected recon brain surveys the area
    defence  the selected defender brain (the brain under test) against the selected attacker
    offence  the same engagement seen from the attackers: the attacker brain is under test
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone

import numpy as np

from .brains import BUILTINS, DEFAULTS, Brain, MixedBrain, a_or_an, load_brain
from .clash import ClashEnv
from .config import ClashConfig, ReconConfig, from_dict, to_dict
from .core import decide_every
from .recon import ReconEnv

TESTS = {
    "recon": {"title": "RECON", "config": ReconConfig, "sides": ("recon",), "subject": "recon", "score": "fitness"},
    "defence": {"title": "DEFENCE", "config": ClashConfig, "sides": ("defender", "attacker"), "subject": "defender",
                "score": "def_fitness"},
    "offence": {"title": "OFFENCE", "config": ClashConfig, "sides": ("attacker", "defender"), "subject": "attacker",
                "score": "att_fitness"},
}

# key, label, format (pct | x | s | kbps | km | j | m | num), better (high | low | None)
CARDS = {
    "defence": [("intercepted", "Intercepted", "pct", "high"), ("leaked", "Got through", "pct", "low"),
                ("overkill", "Interceptors per target", "x", "low"), ("kill_time", "Time to intercept", "s", "low"),
                ("cpu_def", "CPU load per drone", "pct", "low"), ("radio_def_kbps", "Radio per drone", "kbps", "low")],
    "offence": [("leaked", "Got through", "pct", "high"), ("intercepted", "Shot down", "pct", "low"),
                ("held_off", "Still flying", "pct", None), ("low_share", "Flying under the terrain mask", "pct", None),
                ("alt_att", "Mean altitude", "m", None), ("misses", "Interceptor misses", "num", "high"),
                ("cpu_att", "CPU load per drone", "pct", "low"), ("radio_att_kbps", "Radio per drone", "kbps", "low")],
    "recon": [("coverage", "Area surveyed (detail)", "pct", "high"), ("t90", "Time to 90%", "s", "low"),
              ("reported", "Targets reported", "pct", "high"), ("connected", "In radio contact", "pct", "high"),
              ("altitude", "Mean altitude", "m", None), ("camera_duty", "Camera on", "pct", "low"),
              ("cpu", "CPU load per drone", "pct", "low"), ("energy_j", "Compute + radio energy", "j", "low")],
}
COLUMNS = {
    "defence": ["intercepted", "leaked", "overkill", "kill_time", "cpu_def"],
    "offence": ["leaked", "intercepted", "held_off", "low_share", "cpu_att"],
    "recon": ["coverage", "t90", "reported", "connected", "camera_duty", "cpu"],
}


def _cadence(brain: Brain, device: str, dt: float, batch: int) -> np.ndarray:
    """Steps between decisions for every episode: a brain too slow for its board decides less often. In a
    mixed batch each brain keeps its own pace, so an episode plays out the same in any batch."""
    k = np.full(batch, decide_every(brain.macs, device, dt))
    if isinstance(brain, MixedBrain):
        for b, idx, _ in brain.groups:
            k[idx] = decide_every(b.macs, device, dt)
    return k


def _act(brain: Brain, obs, due: np.ndarray) -> dict:
    return brain.act(obs, due) if isinstance(brain, MixedBrain) else brain.act(obs)


def _macs(brain: Brain, batch: int):
    """Compute per decision, per episode for a mixed batch (for each episode's CPU load figure)."""
    if not isinstance(brain, MixedBrain):
        return brain.macs
    m = np.zeros(batch)
    for b, idx, _ in brain.groups:
        m[idx] = b.macs
    return m


def play_clash(env: ClashEnv, defender: Brain, attacker: Brain, seed: int = 0) -> dict[str, np.ndarray]:
    """Runs every episode of ``env`` to the end. A brain too slow for its device decides less often."""
    c = env.cfg
    k_d = _cadence(defender, c.defender_device, c.dt, env.B)
    k_a = _cadence(attacker, c.attacker_device, c.dt, env.B)
    defender.reset(env.B, seed)
    attacker.reset(env.B, seed)
    while not env.done.all():
        due = env.steps % k_d == 0
        if due.any():
            o = env.defender_obs(due)
            a = _act(defender, o, due)
            env.apply_defenders(o, a["target"], a.get("msg"), due=due, station=a.get("station"))
        due = env.steps % k_a == 0
        if due.any():
            o = env.attacker_obs(due)
            a = _act(attacker, o, due)
            env.apply_attackers(o, a["heading"], a["speed"], altitude=a.get("altitude"), msg=a.get("msg"), due=due)
        env.step()
    return env.summary(_macs(defender, env.B), _macs(attacker, env.B))


def play_recon(env: ReconEnv, brain: Brain, seed: int = 0) -> dict[str, np.ndarray]:
    k = _cadence(brain, env.cfg.device, env.cfg.dt, env.B)
    brain.reset(env.B, seed)
    while not env.done.all():
        due = env.steps % k == 0
        if due.any():
            o = env.observe()
            a = _act(brain, o, due)
            env.apply(o, a["direction"], a["speed"], a["camera"], altitude=a.get("altitude"), msg=a.get("msg"),
                      due=due)
        env.step()
    return env.summary(_macs(brain, env.B))


def _play(test: str, cfg, brains: dict[str, Brain], episodes: int, seed: int, record: bool):
    if test == "recon":
        env = ReconEnv(cfg, episodes, seed, record=record)
        return play_recon(env, brains["recon"], seed), env
    env = ClashEnv(cfg, episodes, seed, record=record)
    return play_clash(env, brains["defender"], brains["attacker"], seed), env


def _clean(v: float) -> float | None:
    return None if v is None or not math.isfinite(v) else float(f"{float(v):.6g}")


def _stats(summary: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for key, arr in summary.items():
        a = np.asarray(arr, dtype=float)
        ok = a[np.isfinite(a)]
        out[key] = {"mean": _clean(ok.mean()) if ok.size else None, "std": _clean(ok.std()) if ok.size else None,
                    "min": _clean(ok.min()) if ok.size else None, "max": _clean(ok.max()) if ok.size else None,
                    "n": int(ok.size)}
    return out


def _mean(stats: dict, key: str) -> float:
    v = stats.get(key, {}).get("mean")
    return float("nan") if v is None else v


def verdict(test: str, stats: dict, cfg) -> dict:
    m = lambda k: _mean(stats, k)  # noqa: E731
    if test == "recon":
        cov, conn, rep = m("coverage"), m("connected"), m("reported")
        if cov >= 0.95 and conn >= 0.9:
            label, tone = "SURVEY COMPLETE", "good"
        elif cov < 0.7:
            label, tone = "GAPS LEFT", "bad"
        elif conn < 0.8:
            label, tone = "SWARM SCATTERED", "warn"
        else:
            label, tone = "PARTIAL SURVEY", "warn"
        t90 = m("t90")
        when = f", 90% after {t90:.0f} s" if math.isfinite(t90) else ", never reached 90%"
        line = (f"{cov:.0%} of the area surveyed in full detail{when} (seen at all: {m('seen'):.0%}); "
                f"{rep * cfg.targets:.1f} of {cfg.targets} targets reported; in radio contact {conn:.0%} of the time; "
                f"mean altitude {m('altitude'):.0f} m; cameras on {m('camera_duty'):.0%} of the time")
        return {"label": label, "tone": tone, "line": line}
    A = cfg.attackers
    through, down = m("leaked") * A, m("intercepted") * A
    if test == "defence":
        if m("leaked") < 0.01:
            label, tone = "WALL HELD", "good"
        elif m("leaked") <= 0.25:
            label, tone = "HELD WITH LEAKS", "warn"
        else:
            label, tone = "BREACHED", "bad"
        line = (f"{down:.1f} of {A} attackers intercepted and {through:.1f} got through per run, "
                f"{m('overkill'):.2f} interceptors per target")
    else:
        if m("leaked") >= 0.5:
            label, tone = "BREAKTHROUGH", "good"
        elif m("leaked") >= 0.2:
            label, tone = "PARTIAL BREACH", "warn"
        else:
            label, tone = "REPELLED", "bad"
        line = (f"{through:.1f} of {A} attackers got through and {down:.1f} were shot down per run; they flew at "
                f"{m('alt_att'):.0f} m on average, {m('low_share'):.0%} of the time under the terrain mask")
    return {"label": label, "tone": tone, "line": line}


def run_test(test: str, settings: dict | None = None, brains: dict | None = None, episodes: int = 8,
             seed: int = 0, compare: bool = True, progress=None) -> dict:
    """Plays ``episodes`` runs of ``test``; ``brains`` maps role -> brain id or :class:`Brain` (defaults per role)."""
    if test not in TESTS:
        raise ValueError(f"unknown test {test!r}; tests: {list(TESTS)}")
    if not 1 <= int(episodes) <= 64:
        raise ValueError("episodes must be within [1, 64]")
    spec = TESTS[test]
    cfg = from_dict(spec["config"], settings)
    chosen = {}
    for role in spec["sides"]:
        b = (brains or {}).get(role) or DEFAULTS[role]
        chosen[role] = b if isinstance(b, Brain) else load_brain(b, role)
        if chosen[role].role != role:
            raise ValueError(f"{chosen[role].id} is not {a_or_an(role)} brain")
    subject = spec["subject"]
    rivals = [cls() for name, cls in BUILTINS.items() if cls.role == subject and name != chosen[subject].id]
    total = 1 + (len(rivals) if compare else 0)
    t0 = time.perf_counter()
    summary, env = _play(test, cfg, chosen, episodes, seed, record=True)
    stats = _stats(summary)
    score = spec["score"]
    rows = [{"brain": chosen[subject].info(), "selected": True, "score": _mean(stats, score),
             "metrics": {k: _mean(stats, k) for k in COLUMNS[test]}}]
    if progress:
        progress(1, total)
    if compare:
        for i, rival in enumerate(rivals):
            s, _ = _play(test, cfg, {**chosen, subject: rival}, episodes, seed, record=False)
            st = _stats(s)
            rows.append({"brain": rival.info(), "selected": False, "score": _mean(st, score),
                         "metrics": {k: _mean(st, k) for k in COLUMNS[test]}})
            if progress:
                progress(2 + i, total)
    rows.sort(key=lambda r: -r["score"] if math.isfinite(r["score"]) else math.inf)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["score"] = _clean(r["score"])
        r["metrics"] = {k: _clean(v) for k, v in r["metrics"].items()}
    per_episode = [{k: _clean(float(summary[k][b])) for k in COLUMNS[test] if k in summary}
                   for b in range(env.B)]
    return {
        "kind": "test", "test": test, "title": spec["title"], "subject": subject,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episodes": int(episodes), "seed": int(seed), "settings": to_dict(cfg),
        "brains": {role: b.info() for role, b in chosen.items()},
        "cards": [{"key": k, "label": label, "fmt": fmt, "better": better, **stats.get(k, {})}
                  for k, label, fmt, better in CARDS[test]],
        "metrics": stats, "columns": COLUMNS[test], "verdict": verdict(test, stats, cfg),
        "compare": rows, "per_episode": per_episode, "replay": env.replay(),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
