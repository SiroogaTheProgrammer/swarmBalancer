"""Scenarios between the browser and the battery: JSON in, checked plain dict, runnable preset file back.

The studio edits scenarios as the plain JSON of ``swarm.bench.scenario.to_plain``. :func:`normalize` checks
every field (types, ranges, known devices and strategies) because that JSON comes from a browser.
:func:`render_preset` writes the same kind of file as ``scenarios/_template.py`` - a studio scenario is a
real preset that ``python <file>`` and ``python -m swarm.bench`` run - plus an optional ``THEATRE`` dict
with the 3D view settings (formation, route, terrain), which the bench ignores.

Only worker processes import preset files (:func:`index_presets`, :func:`scenario_from_plain` after the
tactic packages are registered); the server itself never executes preset or package code.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from swarm.bench.runner import _OPS
from swarm.bench.scenario import (ALL_STRATEGIES, BrainSpec, Camera, Check, Faults, Link, Scenario, Stress, Swarm,
                                  to_plain)
from swarm.devices import DEVICE_PROFILES

REPO_ROOT = Path(__file__).resolve().parents[3]
BATTERY_DIR = REPO_ROOT / "scenarios"
STUDIO_DIR = REPO_ROOT / "scenarios" / "studio"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
BATTERY_NAME_RE = re.compile(r"^s\d{2}_[a-z0-9_]+$")
MARKER = "Created in the swarmBalancer studio"
MAX_NODES = 24
BRAIN_KINDS = ("auto", "oracle", "numpy", "native", "package")
CHECK_METRICS = (
    "fps_processed", "objects_seen_ps", "objects_reported_ps", "recall", "accuracy", "false_alarms_reported",
    "latency_p50_ms", "latency_p95_ms", "gmac_ps", "kmac_per_frame", "busy_mean", "busy_max", "busy_leader",
    "busy_workers_max", "kbps", "channel_util", "bytes_per_frame", "energy_j", "energy_mj_per_frame", "cameras_on",
    "frames_captured", "frames_processed", "frames_dropped", "frames_rejected", "messages_lost", "leader_changes",
    "failures_detected", "rebalances", "detect_s", "reelect_s", "recover_s", "work_retained", "recall_retained",
    "frames_lost_to_fault",
)
THEATRE_CHOICES = {
    "terrain": ("auto", "plains", "urban", "desert", "arctic", "ocean"),
    "formation": ("auto", "wedge", "line", "column", "grid", "ring"),
    "route": ("auto", "racetrack", "lawnmower", "orbit", "transit", "hold"),
}
THEATRE_NUMBERS = {"altitude_m": (0.0, 5000.0), "spacing_m": (0.0, 2000.0), "speed_mps": (0.0, 8000.0),
                   "area_m": (200.0, 20000.0)}
THEATRE_DEFAULTS = {"terrain": "auto", "formation": "auto", "route": "auto", "altitude_m": 0.0, "spacing_m": 0.0,
                    "speed_mps": 0.0, "area_m": 1600.0}


class ScenarioError(ValueError):
    """The browser sent something that is not a valid scenario; the message names the field."""


# ----------------------------------------------------------------------------
# field checks
# ----------------------------------------------------------------------------
def _num(d: dict, key: str, default, lo: float, hi: float, where: str, integer: bool = False):
    v = d.get(key, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
        raise ScenarioError(f"{where}.{key} must be a number, got {v!r}")
    if integer:
        if float(v) != int(v):
            raise ScenarioError(f"{where}.{key} must be a whole number, got {v!r}")
        v = int(v)
    else:
        v = float(v)
    if not lo <= v <= hi:
        raise ScenarioError(f"{where}.{key} must be within [{lo:g}, {hi:g}], got {v:g}")
    return v


def _text(d: dict, key: str, default: str, where: str, limit: int = 400) -> str:
    v = d.get(key, default)
    if not isinstance(v, str) or len(v) > limit or any(ord(c) < 32 and c not in "\t" for c in v):
        raise ScenarioError(f"{where}.{key} must be a single-line string of at most {limit} characters")
    return v.strip()


def _obj(d: dict, key: str) -> dict:
    v = d.get(key) or {}
    if not isinstance(v, dict):
        raise ScenarioError(f"{key} must be an object")
    return v


def _device(name: Any, where: str) -> str:
    if name not in DEVICE_PROFILES:
        raise ScenarioError(f"{where}: unknown device {name!r}; known: {sorted(DEVICE_PROFILES)}")
    return name


def normalize(plain: dict, plugin_strategies: set[str] | frozenset[str] = frozenset()) -> dict:
    """Checks a scenario JSON and returns it in canonical form (what ``to_plain`` would give)."""
    if not isinstance(plain, dict):
        raise ScenarioError("scenario must be a JSON object")
    name = plain.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ScenarioError(f"name {name!r} must be a lower-case identifier of 2-48 characters")
    sw = _obj(plain, "swarm")
    nodes = sw.get("nodes")
    if nodes:
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_NODES:
            raise ScenarioError(f"swarm.nodes must list 1..{MAX_NODES} device names")
        swarm = {"leader": "drone-a72", "worker": "drone-a53", "size": len(nodes),
                 "nodes": [_device(n, "swarm.nodes") for n in nodes]}
    else:
        swarm = {"leader": _device(sw.get("leader", "drone-a72"), "swarm.leader"),
                 "worker": _device(sw.get("worker", "drone-a53"), "swarm.worker"),
                 "size": _num(sw, "size", 5, 1, MAX_NODES, "swarm", integer=True), "nodes": None}
    n_nodes = len(swarm["nodes"]) if swarm["nodes"] else swarm["size"]

    cam = _obj(plain, "camera")
    sf = cam.get("striped_fps", "match")
    if sf not in ("match", None):
        sf = _num(cam, "striped_fps", None, 0.01, 5000.0, "camera")
    camera = {"fps": _num(cam, "fps", 10.0, 0.01, 240.0, "camera"),
              "size": _num(cam, "size", 128, 8, 2048, "camera", integer=True),
              "p_object": _num(cam, "p_object", 0.3, 0.0, 1.0, "camera"), "striped_fps": sf}

    ln = _obj(plain, "link")
    link = {"bps": _num(ln, "bps", None, 100.0, 1e10, "link"),
            "latency_s": _num(ln, "latency_s", None, 0.0, 10.0, "link"),
            "loss": _num(ln, "loss", 0.01, 0.0, 0.9, "link"), "qos": bool(ln.get("qos", True)),
            "hb_interval": _num(ln, "hb_interval", 0.5, 0.01, 60.0, "link"),
            "missed_beats": _num(ln, "missed_beats", 3, 1, 50, "link", integer=True)}

    br = _obj(plain, "brain")
    kind = br.get("kind", "auto")
    if kind not in BRAIN_KINDS:
        raise ScenarioError(f"brain.kind must be one of {BRAIN_KINDS}")
    brain = {"kind": kind, "model": _text(br, "model", "models/tiny_cnn_int8.swm", "brain", 500),
             "macs": _num(br, "macs", 401_568, 1, 10**12, "brain", integer=True),
             "accuracy": _num(br, "accuracy", 0.87, 0.0, 1.0, "brain"),
             "input_size": _num(br, "input_size", 32, 4, 512, "brain", integer=True),
             "frontend_fraction": _num(br, "frontend_fraction", 0.92, 0.0, 1.0, "brain")}

    duration = _num(plain, "duration", 30.0, 0.5, 900.0, "scenario")
    fl = _obj(plain, "faults")
    kills, revives = [], []
    for item in fl.get("kill") or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ScenarioError("faults.kill entries are [who, time]")
        who, at = item
        if who != "leader":
            if isinstance(who, bool) or not isinstance(who, (int, float, str)) or not str(who).isdigit() \
                    or not 0 <= int(who) < n_nodes:
                raise ScenarioError(f"faults.kill: who must be 'leader' or a node id 0..{n_nodes - 1}, got {who!r}")
            who = int(who)
        kills.append([who, _num({"t": at}, "t", 0, 0.0, duration, "faults.kill")])
    for item in fl.get("revive") or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ScenarioError("faults.revive entries are [node id, time]")
        who, at = item
        if isinstance(who, bool) or not str(who).isdigit() or not 0 <= int(who) < n_nodes:
            raise ScenarioError(f"faults.revive: node id must be 0..{n_nodes - 1}, got {who!r}")
        revives.append([int(who), _num({"t": at}, "t", 0, 0.0, duration, "faults.revive")])

    known = set(ALL_STRATEGIES) | set(plugin_strategies)
    strategies = plain.get("strategies") or list(ALL_STRATEGIES)
    if not isinstance(strategies, list) or not all(isinstance(s, str) for s in strategies):
        raise ScenarioError("strategies must be a list of names")
    unknown = [s for s in strategies if s not in known]
    if unknown:
        raise ScenarioError(f"unknown strategies {unknown}; known: {sorted(known)}")
    baseline = plain.get("baseline", "local")
    if baseline not in known:
        raise ScenarioError(f"unknown baseline {baseline!r}")

    checks = []
    for c in plain.get("checks") or []:
        if not isinstance(c, dict):
            raise ScenarioError("checks entries are objects")
        strat = c.get("strategy", "*")
        if strat != "*" and strat not in known:
            raise ScenarioError(f"check strategy {strat!r} is not a known strategy")
        if c.get("metric") not in CHECK_METRICS:
            raise ScenarioError(f"check metric {c.get('metric')!r}; choose one of {CHECK_METRICS}")
        if c.get("op") not in _OPS:
            raise ScenarioError(f"check operator {c.get('op')!r}; choose one of {sorted(_OPS)}")
        checks.append({"strategy": strat, "metric": c["metric"], "op": c["op"],
                       "value": _num(c, "value", 0, -1e15, 1e15, "check"), "ratio": bool(c.get("ratio", False)),
                       "note": _text(c, "note", "", "check", 200)})

    st = _obj(plain, "stress")
    devices = st.get("devices")
    if devices is not None:
        if not isinstance(devices, list):
            raise ScenarioError("stress.devices must be a list of device names or null")
        devices = [_device(d, "stress.devices") for d in devices]
    tags = plain.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) and NAME_RE.match(t.replace("-", "_")) for t in tags):
        raise ScenarioError("tags must be short lower-case words")
    return {
        "name": name, "description": _text(plain, "description", "", "scenario"), "swarm": swarm, "camera": camera,
        "link": link, "brain": brain, "faults": {"kill": kills, "revive": revives}, "duration": duration,
        "seed": _num(plain, "seed", 0, 0, 2**31 - 1, "scenario", integer=True), "strategies": strategies,
        "baseline": baseline, "queue_limit": _num(plain, "queue_limit", 4, 1, 64, "scenario", integer=True),
        "confidence_threshold": _num(plain, "confidence_threshold", 0.5, 0.0, 1.0, "scenario"),
        "include_leader": bool(plain.get("include_leader", True)),
        "stress": {"devices": devices, "frames": _num(st, "frames", 50, 1, 1000, "stress", integer=True)},
        "checks": checks, "tags": tags,
    }


def normalize_theatre(t: dict | None) -> dict:
    t = dict(t or {})
    out = dict(THEATRE_DEFAULTS)
    for key, choices in THEATRE_CHOICES.items():
        v = t.get(key, out[key])
        if v not in choices:
            raise ScenarioError(f"theatre.{key} must be one of {choices}")
        out[key] = v
    for key, (lo, hi) in THEATRE_NUMBERS.items():
        out[key] = _num(t, key, out[key], lo, hi, "theatre")
    unknown = set(t) - set(out)
    if unknown:
        raise ScenarioError(f"unknown theatre settings {sorted(unknown)}")
    return out


# ----------------------------------------------------------------------------
# plain dict -> Scenario (worker side, after plugin tactics are registered)
# ----------------------------------------------------------------------------
def _tup(v):
    return tuple(_tup(x) for x in v) if isinstance(v, list) else v


def scenario_from_plain(plain: dict) -> Scenario:
    p = normalize(plain, _registered_plugins())
    sw = p["swarm"]
    return Scenario(
        name=p["name"], description=p["description"],
        swarm=Swarm(leader=sw["leader"], worker=sw["worker"], size=sw["size"],
                    nodes=tuple(sw["nodes"]) if sw["nodes"] else None),
        camera=Camera(**p["camera"]), link=Link(**p["link"]), brain=BrainSpec(**p["brain"]),
        faults=Faults(kill=_tup(p["faults"]["kill"]), revive=_tup(p["faults"]["revive"])),
        duration=p["duration"], seed=p["seed"], strategies=tuple(p["strategies"]), baseline=p["baseline"],
        queue_limit=p["queue_limit"], confidence_threshold=p["confidence_threshold"],
        include_leader=p["include_leader"],
        stress=Stress(devices=tuple(p["stress"]["devices"]) if p["stress"]["devices"] else None,
                      frames=p["stress"]["frames"]),
        checks=tuple(Check(**c) for c in p["checks"]), tags=tuple(p["tags"]))


def _registered_plugins() -> set[str]:
    from swarm.sim.strategies import plugin_strategies

    return set(plugin_strategies())


# ----------------------------------------------------------------------------
# preset files
# ----------------------------------------------------------------------------
def _lit(v: Any) -> str:
    if v is None or isinstance(v, bool):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)  # a JSON string literal is a valid Python string literal
    if isinstance(v, int):
        return f"{v:_}" if abs(v) >= 100_000 else str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_lit(x) for x in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_lit(k)}: {_lit(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"cannot write {type(v).__name__} into a preset")


def _same(a, b) -> bool:
    return _tup(a) == _tup(b) if isinstance(a, list) or isinstance(b, (list, tuple)) else a == b


def _ctor(cls_name: str, values: dict, defaults: dict) -> str:
    args = [f"{k}={_lit(_tup(v))}" for k, v in values.items() if not _same(v, defaults.get(k))]
    return f"{cls_name}(" + ", ".join(args) + ")"


def render_preset(plain: dict, theatre: dict | None = None, packages: list[str] | None = None) -> str:
    """Source of a runnable preset file for an already :func:`normalize`-d scenario."""
    d = to_plain(Scenario(name="defaults"))
    sw = dict(plain["swarm"])
    if sw.get("nodes"):
        sw = {"nodes": sw["nodes"]}
        swarm_defaults = {"nodes": None}
    else:
        swarm_defaults = d["swarm"]
    fields = [
        ("name", _lit(plain["name"])), ("description", _lit(plain["description"])),
        ("swarm", _ctor("Swarm", sw, swarm_defaults)), ("camera", _ctor("Camera", plain["camera"], d["camera"])),
        ("link", _ctor("Link", plain["link"], d["link"])), ("brain", _ctor("BrainSpec", plain["brain"], d["brain"])),
        ("faults", _ctor("Faults", plain["faults"], d["faults"])), ("duration", _lit(plain["duration"])),
        ("seed", _lit(plain["seed"])),
    ]
    for key in ("strategies", "baseline", "queue_limit", "confidence_threshold", "include_leader", "tags"):
        if not _same(plain[key], d[key]):
            fields.append((key, _lit(_tup(plain[key]))))
    if not _same(plain["stress"], d["stress"]):
        fields.append(("stress", _ctor("Stress", plain["stress"], d["stress"])))
    check_lines = []
    for c in plain["checks"]:
        extra = (", ratio=True" if c["ratio"] else "") + (f", note={_lit(c['note'])}" if c["note"] else "")
        check_lines.append(f"        Check({_lit(c['strategy'])}, {_lit(c['metric'])}, {_lit(c['op'])}, "
                           f"{_lit(c['value'])}{extra}),")
    used = {"Scenario"} | {v.split("(")[0] for _k, v in fields if "(" in v and v[0].isupper()}
    if check_lines:
        used.add("Check")
    out = [f'"""{plain["name"]} - {MARKER} on {time.strftime("%Y-%m-%d")}.',
           "",
           "A preset like any other: run it with `python <this file>` (-h for options). THEATRE holds the 3D",
           "view settings of the studio; the battery ignores it.",
           '"""',
           "",
           "import _bootstrap  # noqa: F401",
           "",
           f"from swarm.bench import {', '.join(sorted(used))}"]
    if packages:
        out += ["from swarm.sdk import use_package", ""]
        out += [f"use_package({_lit(p)})" for p in packages]
    out += ["", "SCENARIO = Scenario("]
    out += [f"    {k}={v}," for k, v in fields]
    if check_lines:
        out += ["    checks=("] + check_lines + ["    ),"]
    out += [")", ""]
    if theatre:
        out += ["THEATRE = " + _lit(normalize_theatre(theatre)), ""]
    out += ['if __name__ == "__main__":', "    from swarm.bench import main", "",
            "    raise SystemExit(main(SCENARIO))", ""]
    return "\n".join(out)


def preset_path(name: str, where: str) -> Path:
    if where == "battery":
        if not BATTERY_NAME_RE.match(name):
            raise ScenarioError("battery presets are named sNN_something (e.g. s13_my_test) - the number orders them")
        return BATTERY_DIR / f"{name}.py"
    if where == "studio":
        return STUDIO_DIR / f"{name}.py"
    raise ScenarioError("save to 'studio' (scenarios/studio/) or 'battery' (scenarios/)")


def save_preset(plain: dict, theatre: dict | None, packages: list[str], where: str = "studio",
                overwrite: bool = False) -> Path:
    path = preset_path(plain["name"], where)
    if where == "battery" and (not plain["description"] or not plain["checks"]):
        raise ScenarioError("a battery preset needs a description and at least one check (it is a regression test)")
    if path.exists():
        if not overwrite:
            raise ScenarioError(f"{_rel(path)} exists - choose another name or overwrite")
        if MARKER not in path.read_text(encoding="utf-8", errors="replace"):
            raise ScenarioError(f"{path.name} was not created by the studio; it will not be overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    if where == "studio" and not (path.parent / "_bootstrap.py").exists():
        (path.parent / "_bootstrap.py").write_text(BOOTSTRAP, encoding="utf-8")
    path.write_text(render_preset(plain, theatre, packages), encoding="utf-8")
    return path


BOOTSTRAP = '''"""Imported by every studio preset so `python scenarios/studio/<preset>.py` works without installation."""

import sys
from pathlib import Path

_PY = str(Path(__file__).resolve().parents[2] / "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)
'''


# ----------------------------------------------------------------------------
# discovery (worker side: importing a preset runs it)
# ----------------------------------------------------------------------------
def _import_preset(path: Path):
    path = path.resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"swarm_studio_preset_{path.parent.name}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preset_files() -> list[tuple[str, Path]]:
    files = [("battery", p) for p in sorted(BATTERY_DIR.glob("*.py")) if not p.name.startswith("_")]
    files += [("studio", p) for p in sorted(STUDIO_DIR.glob("*.py")) if not p.name.startswith("_")]
    return files


def preset_fingerprint() -> list[list]:
    return [[g, p.name, p.stat().st_mtime_ns, p.stat().st_size] for g, p in preset_files()]


def index_presets() -> list[dict]:
    from swarm.sim.strategies import plugin_strategies

    out: list[dict] = []
    for group, path in preset_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            mod = _import_preset(path)
            found = list(getattr(mod, "SCENARIOS", [])) + ([mod.SCENARIO] if hasattr(mod, "SCENARIO") else [])
            theatre = getattr(mod, "THEATRE", None)
            for sc in found:
                if not isinstance(sc, Scenario):
                    raise TypeError(f"{path.name}: SCENARIO must be a swarm.bench.Scenario")
                sources = plugin_strategies()
                packages = sorted({_rel(sources[s]) for s in sc.strategies if s in sources})
                out.append({"id": f"{group}:{path.stem}:{sc.name}", "group": group, "file": rel, "name": sc.name,
                            "description": sc.description, "tags": list(sc.tags),
                            "editable": group == "studio" and MARKER in path.read_text(encoding="utf-8"),
                            "scenario": to_plain(sc), "theatre": theatre if isinstance(theatre, dict) else None,
                            "packages": packages})
        except Exception as exc:  # noqa: BLE001 - one broken preset must not hide the others
            out.append({"id": f"{group}:{path.stem}", "group": group, "file": rel, "name": path.stem,
                        "error": f"{type(exc).__name__}: {exc}"})
    return out


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)
