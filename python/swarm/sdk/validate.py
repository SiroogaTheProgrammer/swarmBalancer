"""Conformance checks for brain, tactic and policy packages - the gate in front of the simulator and the drones.

    python -m swarm.sdk validate examples/brains/edge_detector

Perception brains: manifest, interface (shape, classes, declared cost), outputs on synthetic frames,
determinism, accuracy on the synthetic dataset, and a device table (time per frame, RAM fit, which
runtime can host it). Tactics: they must run, be deterministic, pay the same physics as every built-in
design (every processed frame costs at least brain + downscale MACs somewhere in the swarm), and keep
delivering results after losing the leader or a worker. Arena policies: they must load, return valid
actions through a short test, be deterministic, and are ranked against the built-in brains of their role.
A report with no ``fail`` finding is usable.
"""

from __future__ import annotations

import ast
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from swarm.devices import DEVICE_PROFILES, DeviceProfile

from .packages import Package, PackageError, load_package

LEVELS = ("pass", "info", "warn", "fail")


@dataclass
class Finding:
    level: str
    check: str
    detail: str


@dataclass
class Report:
    target: str
    name: str | None = None
    type: str | None = None
    format: str | None = None
    version: str | None = None
    findings: list[Finding] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def add(self, level: str, check: str, detail: str) -> None:
        self.findings.append(Finding(level, check, detail))

    @property
    def ok(self) -> bool:
        return not any(f.level == "fail" for f in self.findings)

    @property
    def status(self) -> str:
        levels = {f.level for f in self.findings}
        return "fail" if "fail" in levels else "warn" if "warn" in levels else "pass"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status
        return _jsonable(d)

    def render_text(self) -> str:
        title = f"{self.name or self.target}" + (f" v{self.version}" if self.version else "")
        kind = "/".join(x for x in (self.type, self.format) if x)
        out = [f"=== swarm.sdk validation: {title} ({kind or 'unknown'}) ==="]
        for f in self.findings:
            out.append(f"{f.level.upper():<5} {f.check:<18} {f.detail}")
        devices = self.info.get("devices")
        if devices:
            out += ["", f"{'device':<13}{'runtime':<16}{'ms/frame':>10}{'max fps':>9}{'RAM need/cap KiB':>20}  fits"]
            for d in devices:
                need = "?" if d["ram_needed"] is None else f"{d['ram_needed'] / 1024:.1f}"
                fits = {True: "yes", False: "NO", None: "?"}[d["fits"]]
                out.append(f"{d['device']:<13}{d['runtime']:<16}{d['ms_per_frame']:>10.3g}{d['max_fps']:>9.3g}"
                           f"{need + '/' + format(d['ram_cap'] / 1024, '.0f'):>20}  {fits}")
        runs = self.info.get("conformance")
        if runs:
            out += ["", f"{'run':<24}{'frames':>8}{'recall':>8}{'kbps':>9}{'busy max':>9}{'recover s':>10}"]
            for r in runs:
                v = r["recover_s"]
                rec = "-" if v is None else ("never" if v == "inf" or (isinstance(v, float) and math.isinf(v))
                                             else f"{v:.2f}")
                out.append(f"{r['run']:<24}{r['frames_processed']:>8}{_pct(r['recall']):>8}{r['kbps']:>9.1f}"
                           f"{_pct(r['busy_max']):>9}{rec:>10}")
        n_warn = sum(f.level == "warn" for f in self.findings)
        out.append("")
        out.append(f"RESULT: {self.status.upper()}" + (f" ({n_warn} warning{'s' * (n_warn != 1)})" if n_warn else "")
                   + f"  [{self.seconds:.1f} s]")
        return "\n".join(out)


def _pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0%}"


def _jsonable(o):
    if isinstance(o, float):
        if math.isnan(o):
            return None
        if math.isinf(o):
            return "inf"
        return o
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _jsonable(float(o))
    return o


def _where(exc: BaseException, root: Path) -> str:
    """'ValueError: msg (brain.py:12)' - the innermost frame inside the package, if any."""
    loc = ""
    for fs in traceback.extract_tb(exc.__traceback__):
        try:
            rel = Path(fs.filename).resolve().relative_to(root)
        except ValueError:
            continue
        loc = f" ({rel.as_posix()}:{fs.lineno})"
    return f"{type(exc).__name__}: {exc}{loc}"


# ----------------------------------------------------------------------------
# devices
# ----------------------------------------------------------------------------
def runs_python(profile: DeviceProfile) -> bool:
    """Linux-class companion computers can host a Python brain; MCUs and OBCs run only the C++ engine."""
    return "linux" in profile.tags


def device_table(macs: int, fmt: str, model: Path | None = None, ram_bytes: int | None = None,
                 portable: bool = True) -> list[dict[str, Any]]:
    rows = []
    for p in DEVICE_PROFILES.values():
        if p.name == "host":
            continue
        dev_s = p.compute_seconds(macs)
        if fmt == "swm":
            runtime, need, fits = "C++ engine", None, None
            if model is not None:
                from swarm.stress.device_stress import check_ram

                fits, need, _err = check_ram(model, p)
        else:
            ok = runs_python(p) and portable
            runtime = "python runtime" if ok else ("needs Linux" if portable else "sim only")
            need = ram_bytes
            fits = (ram_bytes <= p.ram_bytes if ram_bytes else None) if ok else False
        rows.append({"device": p.name, "runtime": runtime, "ms_per_frame": dev_s * 1e3,
                     "max_fps": 1.0 / dev_s if dev_s > 0 else float("inf"), "ram_needed": need,
                     "ram_cap": p.ram_bytes, "fits": fits})
    return rows


def scan_imports(root: Path) -> tuple[set[str], list[str]]:
    """Top-level modules imported by the package's own files, and files that do not parse."""
    names: set[str] = set()
    broken: list[str] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            broken.append(f"{path.relative_to(root).as_posix()}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    local = {p.stem for p in root.glob("*.py")} | {p.name for p in root.iterdir() if p.is_dir()}
    return names - local, broken


# ----------------------------------------------------------------------------
# perception
# ----------------------------------------------------------------------------
def _validate_perception(pkg: Package, rep: Report) -> None:
    from swarm.sim.brains import Frame
    from swarm.train.dataset import CLASSES, make_dataset

    from .api import EMPTY_CLASS_NAMES

    names = pkg.manifest.get("classes")
    portable = True
    if pkg.format == "swm":
        path = pkg.model_path
        if path is None or not path.is_file():
            rep.add("fail", "model", f"compiled model {pkg.manifest.get('model')!r} missing - build it "
                                     "(e.g. python train.py inside the package)")
            return
        try:
            facts = pkg.summary()
            if "model_error" in facts:
                raise ValueError(facts["model_error"])
            make = pkg.resolve_brain(seed=0)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a finding
            rep.add("fail", "model", f"not a loadable .swm network: {exc}")
            return
        rep.info.update({k: facts[k] for k in ("input_shape", "n_classes", "macs", "params", "int8", "model_bytes",
                                                "weights_bytes", "frontend_macs", "feature_bytes")})
        rep.add("pass", "model", f"{facts['layers']} layers, {facts['params']} params, {facts['macs']} MACs/frame, "
                                 f"{'int8' if facts['int8'] else 'f32'} weights, {facts['model_bytes']} bytes")
        n_classes = facts["n_classes"]
        if names and len(names) != n_classes:
            rep.add("fail", "classes", f"manifest lists {len(names)} classes, the network outputs {n_classes}")
            return
        names = names or (list(CLASSES) if n_classes == len(CLASSES) else [f"class_{i}" for i in range(n_classes)])
        factory = make[3]
        rep.info["engine"] = make[0]
    else:
        imports, broken = scan_imports(pkg.root)
        for b in broken:
            rep.add("fail", "syntax", b)
        if broken:
            return
        try:
            cls = pkg.load_class()
            from .api import PythonBrainAdapter, instantiate

            options = dict(pkg.manifest.get("options") or {})
            impl = instantiate(cls, 0, options)
            for attr in ("input_shape", "classes", "macs", "classify"):
                if not hasattr(impl, attr):
                    raise PackageError(f"{cls.__name__} has no attribute {attr!r}")
            brain = PythonBrainAdapter(impl, pkg.name)
        except Exception as exc:  # noqa: BLE001
            rep.add("fail", "load", _where(exc, pkg.root))
            return
        rep.add("pass", "load", f"{cls.__name__} from {pkg.entry}")
        n_classes = brain.n_classes
        names = [str(c) for c in impl.classes]
        if pkg.manifest.get("classes") and list(pkg.manifest["classes"]) != names:
            rep.add("warn", "classes", "manifest classes differ from the class's own list; the class wins")
        if brain.macs <= 0:
            rep.add("fail", "cost", "macs must be a positive integer: the simulator charges it per frame on the "
                                    "device, so declare an honest worst case")
            return
        rep.add("pass", "cost", f"declares {brain.macs} MACs/frame (trusted - the simulator cannot measure Python)")
        rep.info.update({"input_shape": list(brain.input_shape), "n_classes": n_classes, "macs": brain.macs,
                         "frontend_macs": brain.frontend_macs, "feature_bytes": brain.feature_bytes,
                         "ram_bytes": getattr(impl, "ram_bytes", None)})
        third_party = sorted(m for m in imports - set(sys.stdlib_module_names) - {"numpy", "swarm"})
        if "swarm" in imports:
            portable = False
            rep.add("warn", "portability", "imports swarm.* - fine in the simulator, but the device payload carries "
                                           "only the runtime: drop the import to make it deployable")
        else:
            rep.add("pass", "portability", "no swarm imports - deployable to Linux companions (swarm.sdk export)")
        if third_party:
            rep.add("info", "dependencies", "the device must provide: " + ", ".join(third_party))

        def factory():
            return PythonBrainAdapter(instantiate(cls, 0, options), pkg.name)

        rep.info["engine"] = "python"
    rep.info["classes"] = names

    shape = tuple(rep.info["input_shape"])
    c, h, w = shape
    if c != 1:
        rep.add("fail", "input", f"input_shape {shape}: the simulator's cameras deliver 1 channel")
        return
    if h != w or not 8 <= h <= 512:
        rep.add("fail", "input", f"input_shape {shape}: frames are square, 8..512 px per side")
        return
    rep.add("pass", "input", f"{c}x{h}x{w} float32 in [0, 1]")
    if n_classes < 2:
        rep.add("fail", "classes", "need at least 2 classes (0 = nothing, then what you detect)")
        return
    if str(names[0]).lower() not in EMPTY_CLASS_NAMES:
        rep.add("warn", "classes", f"class 0 is {names[0]!r}; the swarm treats class 0 as 'nothing here' "
                                   "and reports every other class as a detection")
    else:
        rep.add("pass", "classes", f"{n_classes}: " + ", ".join(map(str, names)))

    ds = make_dataset(96, h, seed=4242)
    frames = [Frame(id=i, t_capture=0.0, src=0, image=ds.x[i], label=int(ds.y[i])) for i in range(len(ds.y))]
    try:
        brain = factory()
        t0 = time.perf_counter()
        outs = [brain.infer(f) for f in frames]
        host_ms = (time.perf_counter() - t0) / len(frames) * 1e3
        fresh = factory()
        again = [fresh.infer(f) for f in frames[:24]]
        repeat = [brain.infer(f) for f in frames[:24]]
    except Exception as exc:  # noqa: BLE001
        rep.add("fail", "inference", _where(exc, pkg.root))
        return
    bad = [(i, o) for i, o in enumerate(outs)
           if not (0 <= o[0] < n_classes) or not math.isfinite(o[1]) or not 0.0 <= o[1] <= 1.0]
    if bad:
        i, o = bad[0]
        rep.add("fail", "outputs", f"{len(bad)}/{len(outs)} frames gave invalid (class, confidence), e.g. frame {i}: {o}; "
                                   f"class must be 0..{n_classes - 1}, confidence a finite value in [0, 1]")
        return
    rep.add("pass", "outputs", f"{len(outs)} synthetic frames -> valid (class, confidence)")
    same = all(a[0] == b[0] and abs(a[1] - b[1]) < 1e-6 for a, b in zip(outs[:24], again)) and \
        all(a[0] == b[0] and abs(a[1] - b[1]) < 1e-6 for a, b in zip(outs[:24], repeat))
    if not same:
        rep.add("fail", "determinism", "the same frame gave different answers (re-run or fresh instance); "
                                       "seed any randomness from the constructor's seed")
        return
    rep.add("pass", "determinism", "same frame -> same answer, across instances")

    preds = np.array([o[0] for o in outs])
    y = ds.y
    rep.info["host_ms_per_frame"] = host_ms
    if [str(n).lower() for n in names] == list(CLASSES):
        acc = float((preds == y).mean())
        recall = float(((preds != 0) & (y != 0)).sum() / max(1, (y != 0).sum()))
        false_alarm = float(((preds != 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
        rep.info.update({"accuracy": acc, "object_recall": recall, "false_alarm_rate": false_alarm})
        level = "warn" if acc < 0.3 else "info"
        rep.add(level, "accuracy", f"{acc:.0%} on the synthetic dataset (chance = {1 / len(CLASSES):.0%}); "
                                   f"finds {recall:.0%} of objects, {false_alarm:.0%} false alarms")
    else:
        rep.add("info", "accuracy", "classes differ from the simulator's world (" + ", ".join(CLASSES)
                + ") - accuracy in simulation compares class indices")
    rep.add("warn" if host_ms > 20 else "info", "host speed",
            f"{host_ms:.3f} ms/frame on this PC" + (" - long simulations with this brain will be slow" if host_ms > 20 else ""))

    rows = device_table(rep.info["macs"], pkg.format, pkg.model_path, rep.info.get("ram_bytes"), portable)
    rep.info["devices"] = rows
    hosts = [r["device"] for r in rows if r["fits"] is not False and r["runtime"] in ("C++ engine", "python runtime")]
    if pkg.format == "swm":
        rep.add("pass", "deploy", "compiled network: runs in the C++ engine on " + (", ".join(hosts) or "no profile")
                + " (RAM checked under each device's cap)")
    elif portable:
        rep.add("info", "deploy", "python brain: Linux companions only (" + (", ".join(hosts) or "none") + "); "
                                  "MCU-class drones need a .swm network")


# ----------------------------------------------------------------------------
# tactics
# ----------------------------------------------------------------------------
def _run_metrics(sc, strategy: str) -> tuple[dict, dict]:
    from swarm.bench.runner import run_scenario

    res = run_scenario(sc, with_stress=False)
    by = res.by_strategy()
    return by[strategy], by[sc.baseline]


def _validate_tactic(pkg: Package, rep: Report) -> None:
    from swarm.bench import BrainSpec, Camera, Faults, Link, Scenario, Swarm
    from swarm.sim.brains import downscale_macs
    from swarm.sim.strategies import BUILTIN_STRATEGIES, SwarmNode

    from .api import Tactic

    _imports, broken = scan_imports(pkg.root)
    for b in broken:
        rep.add("fail", "syntax", b)
    if broken:
        return
    try:
        cls = pkg.load_class()
    except Exception as exc:  # noqa: BLE001
        rep.add("fail", "load", _where(exc, pkg.root))
        return
    if not (isinstance(cls, type) and issubclass(cls, SwarmNode)):
        rep.add("fail", "load", f"{pkg.entry} must subclass swarm.sdk.Tactic")
        return
    if not issubclass(cls, Tactic):
        rep.add("warn", "load", "subclass swarm.sdk.Tactic (not SwarmNode directly) for the documented helpers")
    if pkg.strategy in BUILTIN_STRATEGIES:
        rep.add("fail", "name", f"{pkg.strategy!r} is a built-in strategy")
        return
    if cls.on_capture is SwarmNode.on_capture:
        rep.add("fail", "hooks", "on_capture(frame) is not implemented - every tactic decides what happens to a frame")
        return
    try:
        strategy = pkg.register()
    except Exception as exc:  # noqa: BLE001
        rep.add("fail", "register", _where(exc, pkg.root))
        return
    rep.add("pass", "register", f"strategy {strategy!r} ({cls.__name__}, camera_mode={cls.camera_mode!r})")
    summary = getattr(cls, "summary", "") or pkg.manifest.get("description", "")
    rep.info.update({"strategy": strategy, "camera_mode": cls.camera_mode, "summary": summary})

    brain = BrainSpec(kind="oracle", macs=401_568, accuracy=0.87, input_size=32)
    cam = Camera(fps=10, size=128, p_object=0.3)
    base = dict(swarm=Swarm(size=5), camera=cam, brain=brain, link=Link(loss=0.01), duration=12.0,
                strategies=("local", strategy))
    runs: list[dict[str, Any]] = []

    def record(label: str, m: dict) -> None:
        runs.append({"run": label, "frames_processed": m["frames_processed"], "recall": m["recall"],
                     "kbps": m["kbps"], "busy_max": m["busy_max"],
                     "recover_s": None if math.isnan(m["recover_s"]) else m["recover_s"]})

    try:
        sc = Scenario(name="sdk_reference", **base)
        m, local = _run_metrics(sc, strategy)
        m2, _ = _run_metrics(sc, strategy)
    except Exception as exc:  # noqa: BLE001
        rep.add("fail", "smoke run", _where(exc, pkg.root))
        return
    record("reference swarm", m)
    record("  (local baseline)", local)
    if m["frames_processed"] == 0:
        rep.add("fail", "smoke run", "no frame was processed in 12 s on a healthy 5-drone swarm")
        return
    rep.add("pass", "smoke run", f"{m['frames_processed']} frames, recall {_pct(m['recall'])} "
                                 f"(local {_pct(local['recall'])}), {m['kbps']:.1f} kbit/s")
    if not math.isnan(local["recall"]) and m["recall"] < 0.5 * local["recall"]:
        rep.add("warn", "mission", "reports less than half of the objects the local baseline reports")
    diff = [k for k in m if k not in ("busy_by_node",) and not (m[k] == m2[k] or (m[k] != m[k] and m2[k] != m2[k]))]
    if diff:
        rep.add("fail", "determinism", f"two runs with the same seed differ in {diff[:5]} - use self.rng, never "
                                       "time or unseeded random")
        return
    rep.add("pass", "determinism", "same seed -> identical metrics")

    floor = m["frames_processed"] * (brain.macs + downscale_macs(cam.size ** 2))
    if m["macs_total"] + 1e-6 * floor < floor:
        rep.add("fail", "physics", f"{m['macs_total']} MACs for {m['frames_processed']} processed frames; every design "
                                   f"pays >= brain ({brain.macs}) + downscale ({cam.size ** 2}) per frame "
                                   "(use preprocess() and run_brain())")
        return
    rep.add("pass", "physics", f"{m['kmac_per_frame']:.0f} kMAC per processed frame >= brain + downscale "
                               f"({(brain.macs + cam.size ** 2) / 1e3:.0f} kMAC)")

    for label, faults in (("leader lost @4s", Faults(kill=(("leader", 4.0),))),
                          ("worker 2 lost @4s", Faults(kill=((2, 4.0),)))):
        try:
            mf, _ = _run_metrics(Scenario(name="sdk_faults", faults=faults, **base), strategy)
        except Exception as exc:  # noqa: BLE001
            rep.add("fail", label, _where(exc, pkg.root))
            return
        record(label, mf)
        if math.isinf(mf["recover_s"]):
            rep.add("fail", label, "results never reached a leader again - handle rebalance()/on_start() so the "
                                   "survivors keep working")
            return
        level = "warn" if mf["work_retained"] < 0.5 else "pass"
        rep.add(level, label, f"results flowing again after {mf['recover_s']:.2f} s, work retained "
                              f"{_pct(mf['work_retained'])} vs a fault-free twin")

    try:
        het = Scenario(name="sdk_s11", swarm=Swarm(nodes=("drone-a72", "drone-a53", "drone-a53", "mcu-m4", "mcu-m4",
                                                          "mcu-m4")),
                       camera=Camera(fps=30, size=128, p_object=0.3), link=Link(bps=6e6, latency_s=0.003, loss=0.01),
                       brain=BrainSpec(kind="oracle", macs=3_200_000, accuracy=0.9, input_size=32), duration=10.0,
                       strategies=("local", "striped", strategy))
        from swarm.bench.runner import run_scenario

        by = run_scenario(het, with_stress=False).by_strategy()
        for name, label in ((strategy, "s11 yardstick"), ("striped", "  (striped)")):
            record(label, by[name])
        mh, ms = by[strategy], by["striped"]
        rep.add("info", "s11 yardstick", f"mixed A72/A53/M4 swarm: busiest CPU {_pct(mh['busy_max'])} "
                                         f"(striped {_pct(ms['busy_max'])}), rejected {mh['frames_rejected']} "
                                         f"(striped {ms['frames_rejected']}), objects/s {mh['objects_reported_ps']:.2f} "
                                         f"(striped {ms['objects_reported_ps']:.2f})")
    except Exception as exc:  # noqa: BLE001
        rep.add("warn", "s11 yardstick", _where(exc, pkg.root))
    rep.info["conformance"] = runs
    rep.add("info", "deploy", "tactics are simulation models of coordination logic; the device runtime has its "
                              "own bounded load balancer (docs/RUNTIME.md) - a winning tactic is its specification")


# ----------------------------------------------------------------------------
# arena policies
# ----------------------------------------------------------------------------
POLICY_TESTS = {"recon": ("recon", {"duration": 150.0}), "defender": ("defence", {}), "attacker": ("offence", {})}


def _policy_devices(brain, decisions_per_s: float, fmt: str) -> list[dict[str, Any]]:
    """Per device: time per decision, the share of the CPU at the test's decision rate, and RAM."""
    rows = []
    ram = None
    if fmt == "npz":
        from swarm.arena.brains import ROWS

        ram = brain.size * 4 + ROWS[brain.role] * (brain.n_in + brain.hidden + brain.n_out) * 4
    for p in DEVICE_PROFILES.values():
        if p.name == "host":
            continue
        dev_s = p.compute_seconds(brain.macs)
        if fmt == "npz":
            runtime, fits = "C++ engine", ram <= p.ram_bytes
        else:
            ok = runs_python(p)
            runtime, fits = ("python runtime", None) if ok else ("needs Linux", False)
        rows.append({"device": p.name, "runtime": runtime, "ms_per_frame": dev_s * 1e3,
                     "max_fps": 1.0 / dev_s if dev_s > 0 else float("inf"), "ram_needed": ram,
                     "ram_cap": p.ram_bytes, "fits": fits, "cpu_load": dev_s * decisions_per_s})
    return rows


def _validate_policy(pkg: Package, rep: Report) -> None:
    from swarm.arena.run import run_test

    role = pkg.role
    if pkg.format == "python":
        mods, broken = scan_imports(pkg.root)
        if broken:
            for b in broken:
                rep.add("fail", "syntax", b)
            return
        extra = sorted(mods - {"numpy", "math", "swarm", "dataclasses", "typing", "__future__", "random",
                               "functools", "itertools", "collections"})
        if extra:
            rep.add("info", "imports", f"also imports {extra}: they must exist wherever the brain runs")
    try:
        brain = pkg.make_policy()
    except Exception as exc:  # noqa: BLE001 - user code
        rep.add("fail", "load", _where(exc, pkg.root))
        return
    rep.add("pass", "load", f"{brain.title}: {role} brain ({brain.kind}), {brain.macs} MACs per drone per decision")
    test, settings = POLICY_TESTS[role]
    runs = []
    for attempt in range(2):
        try:
            runs.append(run_test(test, settings, {role: brain}, episodes=4, seed=1, compare=attempt == 0))
        except Exception as exc:  # noqa: BLE001 - user code
            rep.add("fail", "act", _where(exc, pkg.root))
            return
    doc = runs[0]
    rep.add("pass", "act", f"{doc['title']} test: {doc['verdict']['label']} - {doc['verdict']['line']}")
    same = all(runs[0]["metrics"][k] == runs[1]["metrics"][k] for k in doc["columns"])
    rep.add("pass" if same else "warn", "determinism",
            "same seed, same result" if same else "two runs with the same seed differ - seed any randomness "
                                                  "from reset(batch, seed) so tests are repeatable")
    mine = next(r for r in doc["compare"] if r["selected"])
    rep.add("info", "ranking", f"rank {mine['rank']} of {len(doc['compare'])} against the built-in {role} brains "
                               "on the same situations (" + ", ".join(
                                   f"{r['brain']['title']} {r['score']:.2f}" for r in doc["compare"]) + ")")
    rep.info["test"] = {"test": test, "verdict": doc["verdict"], "compare": doc["compare"]}
    cfg = doc["settings"]
    devices = _policy_devices(brain, 1.0 / cfg["dt"], pkg.format)
    rep.info["devices"] = devices
    slow = [d["device"] for d in devices if d["fits"] is not False and d["cpu_load"] > 1.0]
    if slow:
        rep.add("warn", "compute", f"too slow to decide every {cfg['dt']} s on {slow}: it will decide less often there")
    if pkg.format == "python":
        rep.add("info", "deploy", "python policies run in the simulator and on Linux companions; train or "
                                  "port it to a mesh network (.npz -> .swm) to run on MCU-class drones")
    else:
        rep.add("pass", "deploy", "compiles to .swm for the C++ engine (python -m swarm.arena export)")


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
def validate(path: str | Path) -> Report:
    t0 = time.perf_counter()
    rep = Report(target=str(path))
    try:
        pkg = load_package(path)
    except (PackageError, OSError) as exc:
        rep.add("fail", "manifest", str(exc))
        rep.seconds = time.perf_counter() - t0
        return rep
    rep.name, rep.type, rep.format, rep.version = pkg.name, pkg.type, pkg.format, pkg.version
    rep.add("pass", "manifest", f"{pkg.type} package {pkg.label()}"
            + (f" ({pkg.format})" if pkg.type in ("perception", "policy") else "")
            + (f", role {pkg.role}" if pkg.type == "policy" else ""))
    try:
        if pkg.type == "perception":
            _validate_perception(pkg, rep)
        elif pkg.type == "policy":
            _validate_policy(pkg, rep)
        else:
            _validate_tactic(pkg, rep)
    except Exception as exc:  # noqa: BLE001 - a validator must report, not crash
        rep.add("fail", "internal", _where(exc, pkg.root))
    rep.seconds = time.perf_counter() - t0
    return rep
