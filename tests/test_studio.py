"""3D studio: traces, scenario I/O, preset files, the local server's security gates and a full run job."""

from __future__ import annotations

import io
import json
import math
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import replace

import pytest

from swarm.bench import BrainSpec, Camera, Check, Faults, Link, Scenario, Swarm, run_scenario
from swarm.bench.cli import DEFAULT_DIR, load_file
from swarm.bench.scenario import to_plain
from swarm.studio import scenario_io, store
from swarm.studio.scenario_io import ScenarioError, normalize, normalize_theatre, render_preset, save_preset


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


# ----------------------------------------------------------------------------- traces
FAULTED = Scenario(name="t_trace", swarm=Swarm(size=4), camera=Camera(fps=10), link=Link(loss=0.05),
                   brain=BrainSpec(kind="oracle"), faults=Faults(kill=(("leader", 3.0),), revive=((0, 6.0),)),
                   duration=9.0)


def test_tracing_does_not_change_any_metric():
    plain = run_scenario(FAULTED, with_stress=False)
    traced = run_scenario(FAULTED, with_stress=False, trace=True, keep_frames=True)
    for a, b in zip(plain.strategies, traced.strategies):
        assert a.trace is None and b.trace is not None
        diff = [k for k in a.metrics if not same(a.metrics[k], b.metrics[k])]
        assert not diff, (a.strategy, diff)


def test_trace_reconciles_with_the_metrics():
    res = run_scenario(FAULTED, with_stress=False, trace=True, keep_frames=True)
    for r in res.strategies:
        d, m = r.trace.to_dict(), r.metrics
        assert len(d["capture"]) == m["frames_captured"]
        assert len(d["infer"]) == m["frames_processed"]
        assert sum(1 for row in d["learn"] if row[3] != 0) == m["detections_reported"]
        assert sum(row[7] + 24 for row in d["tx"]) + sum(row[6] + 24 for row in d["bcast"]) == m["bytes_sent"]
        assert sum(row[8] for row in d["tx"]) + sum(len(row[7]) for row in d["bcast"]) == m["messages_lost"]
        assert sum(row[4] for row in d["cpu"]) <= m["macs_total"]  # work cut short by the kill is booked but not drawn
        kill = next(row[0] for row in d["event"] if row[1] == "kill")
        victim = next(row[2] for row in d["event"] if row[1] == "kill")
        assert all(not (row[3] == victim and row[1] < kill < row[2]) for row in d["cpu"]), "work runs past a kill"
        assert [row[1:] for row in d["role"]][:1] == [[0, 1]]
        fids, images = r.trace.frame_arrays()
        assert len(fids) == m["frames_captured"] and images.shape[1:] == (32, 32) and images.dtype.name == "uint8"
        assert set(d["schema"]) >= {"tx", "bcast", "cpu", "capture", "infer", "learn", "event", "role", "camera"}
        json.dumps(d)


# ----------------------------------------------------------------------------- scenario I/O
def preset_plain(name: str) -> dict:
    files = {f.stem: f for f in DEFAULT_DIR.glob("s*.py")}
    [sc] = [s for s in load_file(files[name]) if s.name == name]
    return to_plain(sc)


@pytest.mark.parametrize("name", ["s02_leader_loss", "s07_satellites", "s11_heterogeneous", "s12_low_power_stripe"])
def test_presets_survive_normalize_render_and_reload(tmp_path, name):
    plain = normalize(preset_plain(name))
    theatre = {"terrain": "arctic", "formation": "ring", "route": "orbit", "altitude_m": 90}
    src = render_preset(plain, theatre, [])
    (tmp_path / "_bootstrap.py").write_text("", encoding="utf-8")
    path = tmp_path / f"{name}.py"
    path.write_text(src, encoding="utf-8")
    [sc] = load_file(path)
    assert normalize(to_plain(sc)) == plain
    mod_globals: dict = {}
    exec(compile(src.replace("import _bootstrap", "pass"), str(path), "exec"), mod_globals)
    assert mod_globals["THEATRE"] == normalize_theatre(theatre)


@pytest.mark.parametrize("patch, needle", [
    ({"name": "x; import os"}, "name"),
    ({"name": "Bad"}, "name"),
    ({"swarm": {"nodes": ["drone-a72", "no-such-board"]}}, "unknown device"),
    ({"swarm": {"nodes": ["drone-a53"] * 25}}, "swarm.nodes"),
    ({"faults": {"kill": [[9, 2.0]], "revive": []}}, "faults.kill"),
    ({"faults": {"kill": [["leader", 99.0]], "revive": []}}, "faults.kill"),
    ({"strategies": ["local", "teleport"]}, "unknown strategies"),
    ({"checks": [{"strategy": "*", "metric": "__class__", "op": ">", "value": 0}]}, "check metric"),
    ({"checks": [{"strategy": "*", "metric": "recall", "op": "import", "value": 0}]}, "operator"),
    ({"camera": {"fps": float("nan")}}, "camera.fps"),
    ({"description": "two\nlines"}, "description"),
    ({"brain": {"kind": "exec"}}, "brain.kind"),
])
def test_normalize_rejects_bad_scenarios(patch, needle):
    plain = preset_plain("s02_leader_loss")
    plain.update(patch)
    with pytest.raises(ScenarioError, match=needle):
        normalize(plain)


def test_plugin_strategies_need_their_package():
    plain = preset_plain("s01_reference")
    plain["strategies"] = ["local", "load_aware"]
    with pytest.raises(ScenarioError):
        normalize(plain)
    assert normalize(plain, {"load_aware"})["strategies"] == ["local", "load_aware"]


def test_save_preset_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(scenario_io, "STUDIO_DIR", tmp_path / "studio")
    monkeypatch.setattr(scenario_io, "BATTERY_DIR", tmp_path)
    plain = normalize(dict(preset_plain("s02_leader_loss"), name="op_test"))
    path = save_preset(plain, None, [], "studio")
    assert path.is_file() and (path.parent / "_bootstrap.py").is_file()
    with pytest.raises(ScenarioError, match="exists"):
        save_preset(plain, None, [], "studio")
    save_preset(plain, None, [], "studio", overwrite=True)
    (tmp_path / "studio" / "handmade.py").write_text("SCENARIO = None\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match="not created by the studio"):
        save_preset(dict(plain, name="handmade"), None, [], "studio", overwrite=True)
    with pytest.raises(ScenarioError, match="sNN"):
        save_preset(plain, None, [], "battery")
    with pytest.raises(ScenarioError, match="at least one check"):
        save_preset(dict(plain, name="s99_test", checks=[]), None, [], "battery")
    assert save_preset(dict(plain, name="s99_test"), None, [], "battery").name == "s99_test.py"


# ----------------------------------------------------------------------------- server
@pytest.fixture(scope="module")
def studio():
    from swarm.studio.server import Studio

    st = Studio(0)
    threading.Thread(target=st.serve, daemon=True).start()
    yield st
    st.stop()
    st.httpd.server_close()


def call(st, path, body=None, *, token=True, headers=None, method=None, raw=None):
    h = {"X-Studio-Token": st.token} if token else {}
    h.update(headers or {})
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(f"http://127.0.0.1:{st.port}{path}", data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_api_requires_the_session_token_and_a_local_origin(studio):
    assert call(studio, "/api/ping", token=False)[0] == 403
    assert call(studio, "/api/ping", headers={"X-Studio-Token": "guess"})[0] == 403
    assert call(studio, "/api/ping", headers={"Host": "attacker.example:80"})[0] == 421       # DNS rebinding
    assert call(studio, "/api/ping", headers={"Origin": "https://attacker.example"})[0] == 403
    assert call(studio, "/api/ping", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert call(studio, "/api/ping", method="OPTIONS")[0] == 403                               # no CORS preflight
    assert call(studio, "/api/ping")[0] == 200
    assert call(studio, "/api/ping", headers={"Origin": f"http://localhost:{studio.port}", "Host": f"localhost:{studio.port}"})[0] == 200


def test_static_files_are_confined_and_hardened(studio):
    code, headers, body = call(studio, "/", token=False)
    assert code == 200 and b"SWARM" in body and studio.token.encode() not in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"] and headers["X-Frame-Options"] == "DENY"
    for bad in ("/../pyproject.toml", "/js/../../server.py", "/%2e%2e/server.py", "/js/app.js%00.png"):
        assert call(studio, bad, token=False)[0] == 404, bad
    assert call(studio, "/js/app.js", token=False, method="POST")[0] in (403, 405)


def test_scenario_check_and_upload_validation(studio):
    plain = preset_plain("s01_reference")
    code, _, body = call(studio, "/api/scenario/check", {"scenario": plain})
    assert code == 200 and json.loads(body)["scenario"]["name"] == "s01_reference"
    code, _, body = call(studio, "/api/scenario/check", {"scenario": dict(plain, duration=-1)})
    assert code == 400 and "duration" in json.loads(body)["error"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("../escape.py", "x")
    code, _, body = call(studio, "/api/packages/upload?filename=evil.zip", raw=buf.getvalue(),
                         headers={"Content-Type": "application/octet-stream"})
    assert code == 400 and "unsafe" in json.loads(body)["error"]
    assert call(studio, "/api/packages/upload?filename=evil.exe", raw=b"MZ")[0] == 400
    code, _, body = call(studio, "/api/packages/export", {"path": "../../etc"})
    assert code == 400


def test_run_job_end_to_end(studio, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS", tmp_path / "runs")
    sc = replace(FAULTED, name="t_studio_run", duration=7.0, strategies=("local", "striped"),
                 checks=(Check("*", "frames_processed", ">", 0),))
    code, _, body = call(studio, "/api/runs", {"scenario": to_plain(sc), "theatre": {"route": "orbit"}})
    assert code == 200, body
    ids = json.loads(body)
    deadline = time.time() + 120
    while True:
        job = json.loads(call(studio, f"/api/jobs/{ids['job']}")[2])
        if job["state"] not in ("queued", "running") or time.time() > deadline:
            break
        time.sleep(0.3)
    assert job["state"] == "done", job
    code, headers, body = call(studio, f"/api/runs/{ids['run']}", headers={"Accept-Encoding": "identity"})
    run = json.loads(body)
    assert [s["name"] for s in run["strategies"]] == ["local", "striped"]
    assert run["theatre"]["route"] == "orbit" and run["checks"][0]["passed"]
    direct = run_scenario(sc, with_stress=False).by_strategy()
    for s in run["strategies"]:
        assert s["metrics"]["frames_processed"] == direct[s["name"]]["frames_processed"]
        assert s["trace"]["capture"] and s["frames"]["count"] == s["metrics"]["frames_captured"]
    fid = run["strategies"][1]["trace"]["capture"][3][2]
    frame = json.loads(call(studio, f"/api/runs/{ids['run']}/frame?s=1&fid={fid}")[2])
    assert frame["w"] == frame["h"] == 32
    assert any(r["id"] == ids["run"] for r in json.loads(call(studio, "/api/runs")[2])["runs"])
    assert call(studio, f"/api/runs/{ids['run']}", method="DELETE")[0] == 200
    assert call(studio, f"/api/runs/{ids['run']}")[0] == 400
