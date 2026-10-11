"""Brain & tactic SDK: packages, validation, plugin tactics in the bench, safe installs, device export."""

from __future__ import annotations

import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from swarm.bench import BrainSpec, Camera, Faults, Link, Scenario, Swarm, run_scenario
from swarm.sdk import load_package, use_package, validate
from swarm.sdk.export import export_for_device
from swarm.sdk.install import install_archive, install_source, install_swm, pack, remove_package
from swarm.sdk.packages import PackageError, check_manifest
from swarm.sdk.templates import new_package
from swarm.sim.strategies import NODE_CLASSES, SwarmNode, register_strategy

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "brains"


def fails(report) -> list[str]:
    return [f"{f.check}: {f.detail}" for f in report.findings if f.level == "fail"]


def write_pkg(root: Path, manifest: dict, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def tactic_manifest(name: str, entry: str = "tactic.py:T") -> dict:
    return {"schema": "swarm.package/1", "name": name, "type": "tactic", "entry": entry, "strategy": name}


# ----------------------------------------------------------------------------- examples and templates
def test_example_python_brain_validates_and_is_accurate():
    rep = validate(EXAMPLES / "edge_detector")
    assert rep.ok, fails(rep)
    assert rep.info["accuracy"] > 0.9
    devices = {d["device"]: d for d in rep.info["devices"]}
    assert devices["drone-a53"]["runtime"] == "python runtime" and devices["mcu-m4"]["fits"] is False


def test_example_tactic_validates_and_beats_round_robin_on_mixed_swarm():
    rep = validate(EXAMPLES / "load_aware_stripe")
    assert rep.ok, fails(rep)
    checks = {f.check: f.level for f in rep.findings}
    assert checks["physics"] == "pass" and checks["determinism"] == "pass"
    runs = {r["run"]: r for r in rep.info["conformance"]}
    assert runs["s11 yardstick"]["busy_max"] < 0.2 < runs["  (striped)"]["busy_max"]


@pytest.mark.parametrize("kind", ["perception-python", "tactic"])
def test_templates_validate_out_of_the_box(tmp_path, kind):
    root = new_package(tmp_path, f"my_{kind.split('-')[0]}", kind)
    rep = validate(root)
    assert rep.ok, fails(rep)


def test_swm_template_fails_until_compiled_then_passes(tmp_path, model):
    root = new_package(tmp_path, "my_net", "perception-swm")
    rep = validate(root)
    assert not rep.ok and "model" in fails(rep)[0]
    from swarm.brain import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential

    rng = np.random.default_rng(0)
    net = Sequential([Conv2D(1, 4, 3, 1, 1, rng=rng), ReLU(), MaxPool2D(2), Flatten(), Dense(4 * 16 * 16, 5, rng=rng)],
                     input_shape=(1, 32, 32))
    net.save(root / "model.swm", quantize=True)
    rep = validate(root)
    assert rep.ok, fails(rep)
    assert rep.info["int8"] and rep.info["macs"] == net.macs()
    assert all(d["runtime"] == "C++ engine" for d in rep.info["devices"])


# ----------------------------------------------------------------------------- tactics in the bench
def test_plugin_tactic_runs_in_the_bench_and_survives_leader_loss():
    use_package(EXAMPLES / "load_aware_stripe")
    sc = Scenario(name="t_plugin", swarm=Swarm(size=4), camera=Camera(fps=10), link=Link(loss=0.0),
                  brain=BrainSpec(kind="oracle"), faults=Faults(kill=(("leader", 4.0),)), duration=12.0,
                  strategies=("local", "load_aware"))
    m = run_scenario(sc, with_stress=False).by_strategy()["load_aware"]
    assert m["leader_changes"] == 1 and m["recover_s"] <= 3.0
    assert m["work_retained"] > 0.8 and m["cameras_on"] == 1


def test_single_camera_plugins_get_the_striped_frame_rate():
    use_package(EXAMPLES / "load_aware_stripe")
    sc = Scenario(name="t_fps", swarm=Swarm(size=5), camera=Camera(fps=10), brain=BrainSpec(kind="oracle"),
                  duration=4.0, strategies=("local", "striped", "load_aware"))
    by = run_scenario(sc, with_stress=False).by_strategy()
    assert by["load_aware"]["frames_captured"] == by["striped"]["frames_captured"] > by["local"]["frames_captured"] * 0.9


def test_physics_check_rejects_a_tactic_that_skips_the_downscale(tmp_path):
    cheat = '''from swarm.sdk import Tactic

class T(Tactic):
    strategy = "cheat"
    def on_start(self):
        self.start_camera()
    def on_capture(self, frame):
        self.run_brain(self.brain.macs, frame, self._done)   # no preprocess(): skips the camera downscale
    def _done(self, frame, pred, conf):
        if self.useful(pred, conf):
            self.report_to_leader(frame, pred, conf)
'''
    rep = validate(write_pkg(tmp_path / "cheat", tactic_manifest("cheat"), {"tactic.py": cheat}))
    assert any(f.startswith("physics") for f in fails(rep)), fails(rep)


def test_determinism_check_rejects_unseeded_randomness(tmp_path):
    rnd = '''import random
from swarm.sdk import Tactic

class T(Tactic):
    strategy = "dice"
    camera_mode = "single"
    def on_start(self):
        self.rebalance(None, True)
    def rebalance(self, node_id, alive):
        self.start_camera() if self.is_leader else self.stop_camera()
    def on_capture(self, frame):
        self.preprocess(frame, self._go)
    def _go(self, frame):
        target = random.choice(self.live_members())          # global, unseeded: differs run to run
        if target == self.id:
            self.run_brain(self.brain.macs, frame, self._done)
        else:
            self.send_frame(target, frame)
    def on_frame(self, src, msg):
        self.run_brain(self.brain.macs, msg.payload, self._done)
    def _done(self, frame, pred, conf):
        if self.useful(pred, conf):
            self.report_to_leader(frame, pred, conf)
'''
    rep = validate(write_pkg(tmp_path / "dice", tactic_manifest("dice"), {"tactic.py": rnd}))
    assert any(f.startswith("determinism") for f in fails(rep)), fails(rep)


def test_register_strategy_guards_names():
    class Dummy(SwarmNode):
        def on_capture(self, frame):
            pass

    with pytest.raises(ValueError):
        register_strategy("striped", Dummy)
    register_strategy("t_dummy", Dummy, source="a")
    register_strategy("t_dummy", Dummy, source="a")  # same source may reload
    with pytest.raises(ValueError):
        register_strategy("t_dummy", Dummy, source="b")
    with pytest.raises(TypeError):
        register_strategy("t_other", object)
    assert NODE_CLASSES["t_dummy"] is Dummy


# ----------------------------------------------------------------------------- brain packages in scenarios
def test_python_brain_package_runs_in_a_scenario():
    sc = Scenario(name="t_pkg", swarm=Swarm(size=3), brain=BrainSpec(kind="package", model="examples/brains/edge_detector"),
                  duration=4.0, strategies=("local",))
    res = run_scenario(sc, with_stress=False)
    assert res.brain_kind == "package" and "edge_detector" in res.brain_desc
    assert res.by_strategy()["local"]["accuracy"] > 0.9


def test_auto_brain_kind_accepts_a_package_folder():
    sc = Scenario(name="t_auto", swarm=Swarm(size=2), brain=BrainSpec(kind="auto", model="examples/brains/edge_detector"),
                  duration=2.0, strategies=("local",))
    assert run_scenario(sc, with_stress=False).brain_kind == "package"


# ----------------------------------------------------------------------------- manifests and installs
def test_manifest_paths_must_stay_inside_the_package(tmp_path):
    base = {"schema": "swarm.package/1", "name": "x_pkg", "type": "perception", "format": "swm"}
    for bad in ("../model.swm", "/etc/model.swm", "C:/m.swm", "a\\b.swm", ""):
        with pytest.raises(PackageError):
            check_manifest({**base, "model": bad}, tmp_path)
    with pytest.raises(PackageError):
        check_manifest({**base, "model": "m.swm", "name": "Bad-Name"}, tmp_path)


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.mark.parametrize("evil", ["../escape.py", "/abs.py", "a/../../x.py", "C:/x.py", "pkg/CON.py"])
def test_archive_install_rejects_unsafe_names(tmp_path, evil):
    manifest = json.dumps(tactic_manifest("zipped")).encode()
    data = _zip({"manifest.json": manifest, "tactic.py": b"x = 1\n", evil: b"boom"})
    with pytest.raises(PackageError):
        install_archive(data, tmp_path)
    assert not (tmp_path.parent / "escape.py").exists()
    assert not any(p.name != ".gitkeep" and not p.name.startswith(".incoming") for p in tmp_path.iterdir())


def test_archive_install_rejects_links_and_bombs(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(tactic_manifest("linky")))
        info = zipfile.ZipInfo("tactic.py")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "target")
    with pytest.raises(PackageError, match="link"):
        install_archive(buf.getvalue(), tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(tactic_manifest("bomb")))
        zf.writestr("tactic.py", b"\0" * (8 << 20))  # 8 MiB of zeros deflate to a few KiB
    with pytest.raises(PackageError, match="compression"):
        install_archive(buf.getvalue(), tmp_path)


def test_pack_install_roundtrip_and_remove(tmp_path):
    zipped = pack(EXAMPLES / "edge_detector", tmp_path / "edge.zip")
    lib = tmp_path / "lib"
    pkg = install_archive(zipped.read_bytes(), lib)
    assert pkg.name == "edge_detector" and (lib / "edge_detector" / "brain.py").is_file()
    with pytest.raises(PackageError, match="already installed"):
        install_archive(zipped.read_bytes(), lib)
    assert install_archive(zipped.read_bytes(), lib, replace=True).name == "edge_detector"
    # a zip of the folder (one top-level directory) works too
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel, _ in load_package(EXAMPLES / "load_aware_stripe").files():
            zf.write(EXAMPLES / "load_aware_stripe" / rel, f"load_aware_stripe/{rel}")
    assert install_archive(buf.getvalue(), lib).type == "tactic"
    remove_package("edge_detector", lib)
    assert not (lib / "edge_detector").exists()
    with pytest.raises(PackageError):
        remove_package("../lib", lib)


def test_single_file_installs_without_running_the_code(tmp_path):
    src = b'"""Planner: round robin over everyone."""\nraise SystemExit("must not run at install")\n' \
          b"from swarm.sdk import Tactic\nclass Plan(Tactic):\n    strategy = 'my_plan'\n    def on_capture(self, f):\n        pass\n"
    pkg = install_source(src, "My Plan.py", tmp_path)
    assert pkg.type == "tactic" and pkg.strategy == "my_plan" and pkg.entry == "my_plan.py:Plan"
    brain = install_source(b"class Eye:\n    def classify(self, image):\n        return 0, 1.0\n", "eye.py", tmp_path)
    assert brain.type == "perception" and brain.format == "python"
    with pytest.raises(PackageError):
        install_source(b"x = 1\n", "nothing.py", tmp_path)
    bom = install_source(b"\xef\xbb\xbfclass Bom:\n    def classify(self, image):\n        return 0, 1.0\n", "bom.py", tmp_path)
    assert bom.name == "bom"


def test_swm_upload_becomes_a_package(tmp_path, swm_paths):
    pkg = install_swm(swm_paths["int8"].read_bytes(), "Tiny Net.swm", tmp_path)
    assert pkg.name == "tiny_net" and pkg.format == "swm"
    rep = validate(pkg.root)
    assert rep.ok, fails(rep)
    with pytest.raises(PackageError):
        install_swm(b"not a model", "x.swm", tmp_path)


# ----------------------------------------------------------------------------- device export
def test_python_brain_export_runs_through_the_device_runtime(tmp_path, monkeypatch):
    res = export_for_device(EXAMPLES / "edge_detector", tmp_path / "out")
    assert res["factory"] == "edge_detector_brain._device:make_handler"
    monkeypatch.syspath_prepend(str(tmp_path / "out"))
    from swarm.runtime.handlers import load_handler
    from swarm.train.dataset import render

    handler = load_handler(res["factory"], {"threshold": 0.5, "empty_class": 0})
    vehicle = (render(1, np.random.default_rng(3), 32)[0] * 255).astype(np.uint8).tobytes()
    out = handler(vehicle)
    assert out.useful and struct.unpack("!If", out.payload)[0] == 1
    empty = handler((render(0, np.random.default_rng(4), 32)[0] * 255).astype(np.uint8).tobytes())
    assert not empty.useful and empty.payload == b""
    with pytest.raises(ValueError):
        handler(b"short")
    sys.modules.pop("edge_detector_brain", None)
    sys.modules.pop("edge_detector_brain._device", None)
    sys.modules.pop("edge_detector_brain.brain", None)


def test_swm_export_and_tactic_refusal(tmp_path, swm_paths):
    pkg = install_swm(swm_paths["int8"].read_bytes(), "net.swm", tmp_path / "lib")
    res = export_for_device(pkg.root, tmp_path / "out")
    assert res["kind"] == "swm" and (tmp_path / "out" / "net.swm").is_file()
    assert "--model" in res["instructions"] and "swarm_core" in res["instructions"]
    with pytest.raises(PackageError, match="simulator only"):
        export_for_device(EXAMPLES / "load_aware_stripe", tmp_path / "out2")
