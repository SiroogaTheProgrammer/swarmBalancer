"""Renders scenario results: text for the terminal, Markdown for ``out/bench``, JSON for tooling.

Every table is *metric x strategy*. The baseline column shows absolute values;
every other column shows the value **and its ratio to the baseline**, so a
reader can see at a glance where a design is faster, cheaper, or more fragile
than a swarm of independent thinkers. ``[+]`` marks metrics where higher is
better, ``[-]`` where lower is better.
"""

from __future__ import annotations

import json
import math
from typing import Sequence

from .runner import ScenarioResult, StrategyResult
from .scenario import STRATEGY_NOTES, to_plain

# key, label, format, direction ("+" higher is better, "-" lower is better, None neutral)
MISSION_ROWS: list[tuple[str, str, str, str | None]] = [
    ("fps_processed", "frames processed /s", "{:.1f}", "+"),
    ("objects_seen_ps", "objects in view /s", "{:.2f}", "+"),
    ("objects_reported_ps", "objects reported /s", "{:.2f}", "+"),
    ("recall", "recall (reported/in view)", "{:.0%}", "+"),
    ("accuracy", "class accuracy", "{:.0%}", "+"),
    ("false_alarms_reported", "false alarms", "{:.0f}", "-"),
    ("latency_p50_ms", "detection latency p50 ms", "{:.1f}", "-"),
    ("latency_p95_ms", "detection latency p95 ms", "{:.1f}", "-"),
]
COST_ROWS: list[tuple[str, str, str, str | None]] = [
    ("gmac_ps", "compute GMAC/s (swarm)", "{:.3f}", "-"),
    ("kmac_per_frame", "compute kMAC/frame", "{:.0f}", "-"),
    ("busy_max", "busiest node CPU", "{:.2%}", "-"),
    ("busy_leader", "initial leader CPU", "{:.2%}", "-"),
    ("busy_workers_max", "busiest worker CPU", "{:.2%}", "-"),
    ("kbps", "radio kbit/s", "{:.1f}", "-"),
    ("channel_util", "channel utilisation", "{:.0%}", "-"),
    ("bytes_per_frame", "radio bytes/frame", "{:.0f}", "-"),
    ("energy_j", "energy J", "{:.3f}", "-"),
    ("cameras_on", "cameras on", "{:.0f}", "-"),
    ("frames_dropped", "frames dropped", "{:.0f}", "-"),
    ("frames_rejected", "frames rejected (queue)", "{:.0f}", "-"),
    ("messages_lost", "messages lost", "{:.0f}", "-"),
    ("failures_detected", "suspicions raised", "{:.0f}", None),
    ("leader_changes", "leader changes", "{:.0f}", None),
]
SURVIVAL_ROWS: list[tuple[str, str, str, str | None]] = [
    ("detect_s", "failure detected after s", "{:.2f}", "-"),
    ("reelect_s", "new leader after s", "{:.2f}", "-"),
    ("recover_s", "results flowing after s", "{:.2f}", "-"),
    ("work_retained", "work retained vs no-fault", "{:.0%}", "+"),
    ("recall_retained", "recall retained vs no-fault", "{:.0%}", "+"),
    ("frames_lost_to_fault", "frames lost to the fault", "{:.0f}", "-"),
    ("rebalances", "rebalance events", "{:.0f}", None),
]

SHORT = {"central/downscale": "central/down", "central/features": "central/feat"}
LABEL_W, BASE_W, CELL_W = 32, 12, 15


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt_value(v, fmt: str) -> str:
    if _isnan(v):
        return "n/a"
    if isinstance(v, float) and math.isinf(v):
        return "never"
    return fmt.format(v)


def fmt_ratio(v, b) -> str:
    if _isnan(v) or _isnan(b) or math.isinf(v) or math.isinf(b):
        return ""
    if b == 0:
        return "" if v == 0 else "inf"
    r = v / b
    if r >= 100:
        return f"{r:.0f}x"
    if r >= 10:
        return f"{r:.1f}x"
    return f"{r:.2f}x"


def short(name: str) -> str:
    return SHORT.get(name, name)


# ----------------------------------------------------------------------------
# text
# ----------------------------------------------------------------------------
def _text_table(rows, results: Sequence[StrategyResult], baseline: str) -> list[str]:
    base = next(r for r in results if r.strategy == baseline)
    others = [r for r in results if r.strategy != baseline]
    head = f"{'metric':<{LABEL_W}}{short(base.strategy):>{BASE_W}}" + "".join(f"{short(o.strategy):>{CELL_W}}" for o in others)
    out = [head, "-" * len(head)]
    for key, label, fmt, direction in rows:
        b = base.metrics.get(key)
        tag = {"+": "[+]", "-": "[-]"}.get(direction, "   ")
        line = f"{label:<{LABEL_W - 4}}{tag} {fmt_value(b, fmt):>{BASE_W - 1}}"
        for o in others:
            v = o.metrics.get(key)
            line += f"{fmt_value(v, fmt):>{CELL_W - 7}} {fmt_ratio(v, b):>6}"
        out.append(line)
    return out


def _header_lines(res: ScenarioResult) -> list[str]:
    sc, r = res.scenario, res.resolved
    n = len(r["profiles"])
    title = f"=== {sc.name} ==="
    lines = [title]
    if sc.description:
        lines.append(sc.description)
    lines.append(f"swarm : {sc.swarm.describe()} | camera {sc.camera.size} px @ {sc.camera.fps:g} fps"
                 f" (striped: 1 camera @ {r['striped_fps']:g} fps) | objects in {sc.camera.p_object:.0%} of frames")
    lines.append(f"link  : {r['bps'] / 1e6:.3g} Mbps, {r['latency_s'] * 1e3:.1f} ms, loss {sc.link.loss:.1%}, "
                 f"heartbeat {sc.link.hb_interval:g} s x {sc.link.missed_beats}"
                 f"{'' if sc.link.qos else ', NO control-plane QoS'} | {r['duration']:g} s simulated | seed {sc.seed}")
    lines.append(f"brain : {res.brain_kind} - {res.brain_desc}" + (f" [{res.model_path}]" if res.model_path else ""))
    if res.brain_note:
        lines.append(f"        note: {res.brain_note}")
    lines.append(f"faults: {sc.faults.describe()}")
    d = r["demand_fps"]
    lines.append(f"per-node CPU demand: local {d['local']:g} fps | central leader {d['central leader']:g} fps"
                 f" | striped {d['striped per node']:.3g} fps/node   ({n} nodes)")
    lines.append(f"baseline = {sc.baseline}: {STRATEGY_NOTES.get(sc.baseline, '')}")
    lines.append("other columns: value and ratio vs baseline; [+] higher is better, [-] lower is better")
    return lines


def _stress_lines(res: ScenarioResult) -> list[str]:
    if not res.stress:
        return []
    out = []
    if res.model_path:
        out.append(f"--- device stress: {res.model_path} loaded in the engine under each device's RAM cap ---")
        out.append(f"{'device':<13}{'RAM':<10}{'need/cap KiB':>16}{'ms/frame':>10}{'max fps':>10}{'host ms':>9}{'acc':>6}")
        for s in res.stress:
            fit = "FITS" if s["fits_ram"] else "OVERFLOW"
            need = f"{s['ram_needed'] / 1024:.1f}/{s['ram_cap'] / 1024:.0f}" if s["ram_needed"] else "?"
            out.append(f"{s['device']:<13}{fit:<10}{need:>16}{s['device_ms_per_frame']:>10.2f}{s['max_fps']:>10.1f}"
                       f"{s['host_ms_per_frame']:>9.3f}{s['accuracy']:>6.0%}")
    else:
        out.append(f"--- compute budget: oracle brain, {res.resolved['brain_macs']} MACs/frame "
                   "(no model file -> RAM fit not checked) ---")
        out.append(f"{'device':<13}{'RAM cap KiB':>12}{'ms/frame':>10}{'max fps':>10}")
        for s in res.stress:
            out.append(f"{s['device']:<13}{s['ram_cap'] / 1024:>12.0f}{s['device_ms_per_frame']:>10.2f}{s['max_fps']:>10.1f}")
    return out


def _check_lines(res: ScenarioResult) -> list[str]:
    if not res.checks:
        return ["--- checks: none defined ---"]
    passed = sum(c.passed for c in res.checks)
    failed = len(res.checks) - passed
    out = [f"--- checks: {passed} passed" + (f", {failed} FAILED" if failed else "") + " ---"]
    out += [c.describe() for c in res.checks]
    return out


def fault_log(result: StrategyResult, max_lines: int = 12) -> list[str]:
    keys = ("OFFLINE", "ONLINE", "suspects", "re-admits", "LEADER", "camera", "rotation")
    lines = [l for l in result.log if any(k in l for k in keys)]
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [f"    ... {len(lines) - max_lines + 1} more"]
    return lines


def render_text(res: ScenarioResult, show_log: bool = False) -> str:
    sc = res.scenario
    out = _header_lines(res) + [""]
    out += ["--- mission & speed ---"] + _text_table(MISSION_ROWS, res.strategies, sc.baseline) + [""]
    out += ["--- compute, radio, energy ---"] + _text_table(COST_ROWS, res.strategies, sc.baseline) + [""]
    if sc.faults:
        out += [f"--- survivability: {sc.faults.describe()} (twin run without faults as reference) ---"]
        out += _text_table(SURVIVAL_ROWS, res.strategies, sc.baseline) + [""]
    out += _stress_lines(res) + [""]
    out += _check_lines(res)
    if show_log:
        for r in res.strategies:
            fl = fault_log(r)
            if fl:
                out += ["", f"--- event log: {r.strategy} ---"] + fl
    out.append(f"({res.wall_s:.1f} s wall time)")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------
def _md_table(rows, results: Sequence[StrategyResult], baseline: str) -> list[str]:
    base = next(r for r in results if r.strategy == baseline)
    others = [r for r in results if r.strategy != baseline]
    head = "| metric | " + f"{base.strategy} (baseline) | " + " | ".join(o.strategy for o in others) + " |"
    out = [head, "|---|" + "---:|" * (1 + len(others))]
    for key, label, fmt, direction in rows:
        b = base.metrics.get(key)
        tag = {"+": " [+]", "-": " [-]"}.get(direction, "")
        cells = [f"**{fmt_value(b, fmt)}**"]
        for o in others:
            v = o.metrics.get(key)
            ratio = fmt_ratio(v, b)
            cells.append(f"{fmt_value(v, fmt)}" + (f" ({ratio})" if ratio else ""))
        out.append(f"| {label}{tag} | " + " | ".join(cells) + " |")
    return out


def render_markdown(res: ScenarioResult) -> str:
    sc = res.scenario
    out = [f"## {sc.name}", ""]
    if sc.description:
        out += [sc.description, ""]
    out += ["```"] + _header_lines(res)[2 if sc.description else 1:] + ["```", ""]
    out += ["### Mission & speed", ""] + _md_table(MISSION_ROWS, res.strategies, sc.baseline) + [""]
    out += ["### Compute, radio, energy", ""] + _md_table(COST_ROWS, res.strategies, sc.baseline) + [""]
    if sc.faults:
        out += [f"### Survivability ({sc.faults.describe()})", ""]
        out += _md_table(SURVIVAL_ROWS, res.strategies, sc.baseline) + [""]
    if res.stress:
        out += ["### Device stress", "", "```"] + _stress_lines(res) + ["```", ""]
    out += ["### Checks", "", "```"] + _check_lines(res) + ["```", ""]
    logs = [(r.strategy, fault_log(r)) for r in res.strategies]
    if any(fl for _, fl in logs):
        out += ["<details><summary>fault event log</summary>", ""]
        for name, fl in logs:
            if fl:
                out += [f"**{name}**", "", "```"] + fl + ["```", ""]
        out += ["</details>", ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# json
# ----------------------------------------------------------------------------
def to_json(res: ScenarioResult) -> dict:
    return {
        "scenario": to_plain(res.scenario),
        "brain": {"kind": res.brain_kind, "description": res.brain_desc, "note": res.brain_note,
                  "model_path": res.model_path},
        "resolved": res.resolved,
        "strategies": [{"strategy": r.strategy, "metrics": r.metrics, "twin_metrics": r.twin_metrics, "log": r.log}
                       for r in res.strategies],
        "stress": res.stress,
        "checks": [{"check": to_plain(c.check), "strategy": c.strategy, "actual": c.actual, "passed": c.passed}
                   for c in res.checks],
        "wall_s": res.wall_s,
    }


def dump_json(res: ScenarioResult) -> str:
    return json.dumps(_sanitize(to_json(res)), indent=1, default=str)


def _sanitize(o):
    """NaN/inf are not valid JSON; map them to None / 'inf'."""
    if isinstance(o, float):
        if math.isnan(o):
            return None
        if math.isinf(o):
            return "inf"
        return o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


# ----------------------------------------------------------------------------
# scoreboard (suite summary)
# ----------------------------------------------------------------------------
MISSION_FRACTION = 0.8  # a design "delivers the mission" if it reports at least this share of the baseline's objects/s
TIE_TOLERANCE = 0.05    # values within 5 % of each other are a tie (a few % come from randomly lost frames)

SCORE_COLS = [("fastest", "latency_p50_ms", True), ("least compute", "gmac_ps", True),
              ("lightest workers", "busy_workers_max", True), ("least radio", "kbps", True),
              ("most objects/s", "objects_reported_ps", False), ("most survivable", "work_retained", False)]


def delivering(res: ScenarioResult) -> list[StrategyResult]:
    """Strategies that report >= MISSION_FRACTION of the baseline's objects/s. A design that drops most
    frames would otherwise 'win' least compute / least radio / fastest."""
    base = res.baseline().metrics.get("objects_reported_ps")
    if _isnan(base) or base == 0:
        return list(res.strategies)
    return [r for r in res.strategies
            if not _isnan(r.metrics.get("objects_reported_ps")) and r.metrics["objects_reported_ps"] >= MISSION_FRACTION * base]


def _best(res: ScenarioResult, key: str, lower: bool) -> str:
    pool = res.strategies if key in ("objects_reported_ps", "work_retained") else delivering(res)
    cands = [(r.metrics.get(key), r.strategy) for r in pool]
    cands = [(v, s) for v, s in cands if not _isnan(v) and not math.isinf(v)]
    if not cands:
        return "-"
    vals = [v for v, _ in cands]
    if max(vals) - min(vals) <= TIE_TOLERANCE * max(abs(max(vals)), 1e-12):
        return "(tie)"
    return short(min(cands)[1] if lower else max(cands)[1])


def _score_cells(r: ScenarioResult) -> list[str]:
    return ["-" if (key == "work_retained" and not r.scenario.faults) else _best(r, key, lower)
            for _, key, lower in SCORE_COLS]


def _failed_mission(r: ScenarioResult) -> list[str]:
    ok = {d.strategy for d in delivering(r)}
    return [short(s.strategy) for s in r.strategies if s.strategy not in ok]


def render_scoreboard(results: Sequence[ScenarioResult]) -> str:
    w_name = max(12, max(len(r.scenario.name) for r in results) + 2)
    w = 17
    head = f"{'scenario':<{w_name}}" + "".join(f"{c[0]:>{w}}" for c in SCORE_COLS) + f"{'checks':>10}  failed mission"
    out = ["=== scoreboard: which design wins each category (baseline = independent thinkers) ===", head, "-" * len(head)]
    total_pass = total = 0
    for r in results:
        p = sum(c.passed for c in r.checks)
        total_pass += p
        total += len(r.checks)
        chk = f"{p}/{len(r.checks)}" if r.checks else "-"
        if r.checks and p < len(r.checks):
            chk += " FAIL"
        out.append(f"{r.scenario.name:<{w_name}}" + "".join(f"{c:>{w}}" for c in _score_cells(r)) + f"{chk:>10}  "
                   + (", ".join(_failed_mission(r)) or "-"))
    out.append("")
    out.append(f"cost categories consider only designs that report >= {MISSION_FRACTION:.0%} of the baseline's objects/s "
               f"('failed mission' lists the others); values within {TIE_TOLERANCE:.0%} are a tie")
    out.append(f"checks: {total_pass}/{total} passed" + ("" if total_pass == total else f"  ({total - total_pass} FAILED)"))
    return "\n".join(out)


def render_scoreboard_markdown(results: Sequence[ScenarioResult]) -> str:
    out = ["| scenario | " + " | ".join(c[0] for c in SCORE_COLS) + " | checks | failed mission |",
           "|---|" + "---|" * (len(SCORE_COLS) + 2)]
    for r in results:
        p = sum(c.passed for c in r.checks)
        chk = (f"{p}/{len(r.checks)}" + ("" if p == len(r.checks) else " **FAIL**")) if r.checks else "-"
        out.append(f"| [{r.scenario.name}](#{r.scenario.name.lower().replace('_', '-')}) | " + " | ".join(_score_cells(r))
                   + f" | {chk} | {', '.join(_failed_mission(r)) or '-'} |")
    out.append("")
    out.append(f"Cost categories consider only designs that report at least {MISSION_FRACTION:.0%} of the baseline's "
               f"objects/s (*failed mission* lists the others); values within {TIE_TOLERANCE:.0%} of each other are a tie.")
    return "\n".join(out)
