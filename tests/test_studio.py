"""The studio server: its security rules and the test / train / brains API end to end (real worker processes)."""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile

import pytest

from swarm.arena import BUILTINS

SMALL = {"defenders": 4, "attackers": 6, "width": 800.0, "depth": 700.0, "ceiling": 300.0, "duration": 40.0}
POLICY = b'''"""Holds the line."""
class HoldLine:
    role = "defender"
    def act(self, obs):
        return {"target": -1}
'''


@pytest.fixture(scope="module")
def studio(tmp_path_factory):
    from swarm.sdk import install, packages
    from swarm.studio import jobs, server, store

    out, lib = tmp_path_factory.mktemp("studio"), tmp_path_factory.mktemp("library")
    mp = pytest.MonkeyPatch()
    mp.setenv("SWARM_STUDIO_OUT", str(out))  # worker processes inherit these
    mp.setenv("SWARM_LIBRARY", str(lib))
    for mod, name, value in [(store, "OUT", out), (store, "RUNS", out / "runs"), (store, "TRAINING", out / "training"),
                             (store, "JOBS", out / "jobs"), (jobs, "JOBS", out / "jobs"),
                             (packages, "LIBRARY_DIRS", (lib, packages.LIBRARY_DIRS[1])), (install, "LIBRARY", lib),
                             (server, "LIBRARY", lib)]:
        mp.setattr(mod, name, value)
    st = server.Studio(0)
    threading.Thread(target=st.serve, daemon=True).start()
    st.lib = lib
    yield st
    st.stop()
    st.httpd.server_close()
    mp.undo()


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


def js(st, path, body=None, **kw):
    code, _, data = call(st, path, body, **kw)
    return code, json.loads(data) if data else None


def wait(st, job_id, timeout=240):
    deadline = time.time() + timeout
    while True:
        _, job = js(st, f"/api/jobs/{job_id}")
        if job["state"] not in ("queued", "running") or time.time() > deadline:
            return job
        time.sleep(0.3)


def test_api_requires_the_session_token_and_a_local_origin(studio):
    assert call(studio, "/api/ping", token=False)[0] == 403
    assert call(studio, "/api/ping", headers={"X-Studio-Token": "guess"})[0] == 403
    assert call(studio, "/api/ping", headers={"Host": "attacker.example:80"})[0] == 421       # DNS rebinding
    assert call(studio, "/api/ping", headers={"Origin": "https://attacker.example"})[0] == 403
    assert call(studio, "/api/ping", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert call(studio, "/api/ping", method="OPTIONS")[0] == 403                               # no CORS preflight
    assert call(studio, "/api/ping")[0] == 200
    assert call(studio, "/api/ping", headers={"Origin": f"http://localhost:{studio.port}",
                                              "Host": f"localhost:{studio.port}"})[0] == 200


def test_static_files_are_confined_and_hardened(studio):
    code, headers, body = call(studio, "/", token=False)
    assert code == 200 and b"SWARM" in body and studio.token.encode() not in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"] and headers["X-Frame-Options"] == "DENY"
    for bad in ("/../pyproject.toml", "/js/../../server.py", "/%2e%2e/server.py", "/js/app.js%00.png"):
        assert call(studio, bad, token=False)[0] == 404, bad
    assert call(studio, "/js/app.js", token=False, method="POST")[0] in (403, 405)
    for asset in ("/js/app.js", "/js/arena3d.js", "/js/chart.js", "/js/md.js", "/css/bench.css"):
        assert call(studio, asset, token=False)[0] == 200, asset


def test_catalog_describes_tests_settings_and_brains(studio):
    code, cat = js(studio, "/api/catalog")
    assert code == 200 and set(cat["tests"]) == {"recon", "defence", "offence"}
    assert cat["tests"]["offence"]["subject"] == "attacker" and cat["defaults"]["defender"] == "auction"
    ids = {b["id"] for b in cat["brains"]}
    assert {"auction", "decoy", "relay_sweep", "lib:mesh_defender", "lib:pincer_attack"} <= ids
    assert any(f["name"] == "generations" for f in cat["train"])


def test_bad_requests_are_explained(studio):
    for body, needle in [({"test": "chess"}, "test must be"), ({"test": "defence", "settings": {"attackers": 0}}, "attackers"),
                         ({"test": "defence", "brains": {"defender": "lib:nope"}}, "unknown defender brain"),
                         ({"test": "defence", "brains": {"defender": "rush"}}, "not a defender brain"),
                         ({"test": "recon", "episodes": 0}, "episodes")]:
        code, err = js(studio, "/api/test", body)
        assert code == 400 and needle in err["error"], (body, err)
    code, err = js(studio, "/api/train", {"train": {"population": 5}})
    assert code == 400 and "even" in err["error"]


def test_a_test_runs_end_to_end(studio):
    code, r = js(studio, "/api/test", {"test": "defence", "settings": SMALL, "brains": {"defender": "auction",
                                                                                         "attacker": "rush"},
                                       "episodes": 2, "seed": 4})
    assert code == 200
    job = wait(studio, r["job"])
    assert job["state"] == "done", job
    code, doc = js(studio, f"/api/runs/{job['result']['run']}", headers={"Accept-Encoding": "identity"})
    assert code == 200 and doc["test"] == "defence" and doc["episodes"] == 2 and doc["replay"]["frames"]
    walls = [n for n, c in BUILTINS.items() if c.role == "defender"]
    assert doc["brains"]["defender"]["id"] == "auction" and len(doc["compare"]) == len(walls)
    assert any(m["id"] == doc["id"] for m in js(studio, "/api/runs")[1]["runs"])
    assert call(studio, f"/api/runs/{doc['id']}", method="DELETE")[0] == 200
    assert call(studio, f"/api/runs/{doc['id']}")[0] == 400


def test_training_streams_generations_and_saves_brains(studio):
    body = {"train": {"generations": 2, "population": 4, "episodes": 1, "hidden": 4, "record_every": 1},
            "settings": SMALL}
    code, r = js(studio, "/api/train", body)
    assert code == 200, r
    job = wait(studio, r["job"])
    assert job["state"] == "done", job
    gens = [e for e in job["events"] if e["type"] == "generation"]
    assert [e["gen"] for e in gens] == [1, 2] and gens[-1]["match"] == "gen_0002.json"
    code, t = js(studio, f"/api/training/{r['training']}")
    assert t["status"]["state"] == "done" and len(t["history"]) == 2
    code, match = js(studio, f"/api/training/{r['training']}/matches/gen_0001.json")
    assert code == 200 and match["replay"]["kind"] == "clash"
    assert call(studio, f"/api/training/{r['training']}/matches/..%2fstatus.json")[0] == 400
    code, saved = js(studio, f"/api/training/{r['training']}/save", {"role": "defender", "name": "zz_wall"})
    assert code == 200 and saved["brain"]["id"] == "lib:zz_wall" and saved["brain"]["role"] == "defender"
    assert js(studio, f"/api/training/{r['training']}/save", {"role": "defender", "name": "zz_wall"})[0] == 409
    assert t["status"]["best"]["defender"]["gen"] in (1, 2) and t["status"]["workers"] >= 1
    code, best = js(studio, f"/api/training/{r['training']}/save", {"role": "defender", "which": "best", "name": "zz_best"})
    assert code == 200 and best["brain"]["id"] == "lib:zz_best"
    assert js(studio, f"/api/training/{r['training']}/save", {"role": "defender", "which": "oldest"})[0] == 400
    assert "lib:zz_wall" in {b["id"] for b in js(studio, "/api/brains")[1]["brains"]}
    assert call(studio, "/api/brains/zz_wall", method="DELETE")[0] == 200
    assert call(studio, "/api/brains/zz_best", method="DELETE")[0] == 200
    assert call(studio, "/api/brains/mesh_defender", method="DELETE")[0] == 400       # examples are read-only
    assert call(studio, f"/api/training/{r['training']}", method="DELETE")[0] == 200


def test_stop_keeps_what_was_learned(studio):
    code, r = js(studio, "/api/train", {"train": {"generations": 400, "population": 4, "episodes": 1, "hidden": 4},
                                        "settings": SMALL})
    assert code == 200
    deadline = time.time() + 120
    while not any(e["type"] == "generation" for e in js(studio, f"/api/jobs/{r['job']}")[1]["events"]):
        assert time.time() < deadline
        time.sleep(0.3)
    assert js(studio, f"/api/jobs/{r['job']}/stop", {})[0] == 200
    job = wait(studio, r["job"])
    assert job["state"] == "done" and job["result"]["stopped"] and job["result"]["generations"] < 400
    status = js(studio, f"/api/training/{r['training']}")[1]["status"]
    assert status["state"] == "stopped" and set(status["brains"]) == {"defender", "attacker"}


def test_loading_a_brain(studio):
    code, r = js(studio, "/api/brains/upload?filename=hold_line.py", raw=POLICY)
    assert code == 200 and r["brain"]["id"] == "lib:hold_line" and r["brain"]["role"] == "defender"
    job = wait(studio, r["job"])
    assert job["state"] == "done" and job["result"]["report"]["status"] in ("pass", "warn"), job
    assert "HoldLine" in js(studio, "/api/brains/hold_line/source")[1]["text"]
    code, headers, data = call(studio, "/api/brains/hold_line/download")
    assert code == 200 and "manifest.json" in zipfile.ZipFile(io.BytesIO(data)).namelist()
    code, r2 = js(studio, "/api/test", {"test": "defence", "settings": SMALL, "brains": {"defender": "lib:hold_line"},
                                        "episodes": 1, "compare": False})
    assert code == 200 and wait(studio, r2["job"])["state"] == "done"
    assert call(studio, "/api/brains/hold_line", method="DELETE")[0] == 200
    code, err = js(studio, "/api/brains/upload?filename=eye.py",
                   raw=b"class Eye:\n    def classify(self, image):\n        return 0, 1.0\n")
    assert code == 400 and "perception" in err["error"] and not (studio.lib / "eye").exists()
    assert js(studio, "/api/brains/upload?filename=evil.exe", raw=b"MZ")[0] == 400
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("../escape.py", "x")
    code, err = js(studio, "/api/brains/upload?filename=evil.zip", raw=buf.getvalue())
    assert code == 400 and "unsafe" in err["error"]


def test_manual_is_served(studio):
    code, docs = js(studio, "/api/docs")
    assert code == 200 and {"arena", "studio", "sdk"} <= {d["id"] for d in docs["docs"]}
    code, doc = js(studio, "/api/docs/arena")
    assert code == 200 and "RECON" in doc["markdown"]


def test_a_long_job_keeps_its_latest_events():
    from swarm.studio.jobs import MAX_EVENTS, Job

    job = Job("j", "train", "train")
    for i in range(MAX_EVENTS + 10):
        job.add_event({"gen": i})
    v = job.view(since=MAX_EVENTS + 5)
    assert [e["gen"] for e in v["events"]] == list(range(MAX_EVENTS + 5, MAX_EVENTS + 10)) and v["next"] == MAX_EVENTS + 10
    assert job.view(since=0)["events"][0]["gen"] == job.events_base   # a reader that fell behind gets what is kept
