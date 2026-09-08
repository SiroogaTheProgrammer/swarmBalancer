"""Minimal payload assembly + signed USB -> installed runtime, using loopback only."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cryptography", minversion="44")

from swarm.deploy import accept_enrollment, current_target, enroll, init_authority, init_device, install, launch, pack, request_fingerprint
from swarm.deploy.setup import prepare
from swarm.runtime.demo import demo

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.filterwarnings("ignore:Windows mode bits do not secure ACLs:UserWarning")


def test_prepared_payload_excludes_testing_toolkit(tmp_path):
    app = prepare(tmp_path / "application", config=ROOT / "deploy/raspberry_pi/node.example.json", target="python-any")
    assert {p.name for p in (app / "python/swarm").iterdir()} == {"__init__.py", "runtime", "robotics"}
    assert not list(app.rglob("*.pem")) and not list(app.rglob("*.pyc"))
    assert (app / "main.py").is_file() and (app / "node.json").is_file()
    with pytest.raises(ValueError, match="exists"):
        prepare(app, config=ROOT / "deploy/raspberry_pi/node.example.json", target="python-any")


def test_custom_brain_package_selected_without_executing_it(tmp_path):
    brain = tmp_path / "custom-brain"
    brain.mkdir()
    (brain / "selected.py").write_text("raise RuntimeError('must not be imported when packing')\n")
    app = prepare(tmp_path / "application", config=ROOT / "deploy/raspberry_pi/node.example.json", target="python-any", brain_dir=brain)
    assert (app / "python/selected.py").read_text().startswith("raise RuntimeError")
    (brain / "swarm").mkdir()
    (brain / "swarm/__init__.py").write_text("# not allowed")
    with pytest.raises(ValueError, match="namespace"):
        prepare(tmp_path / "rejected", config=ROOT / "deploy/raspberry_pi/node.example.json", target="python-any", brain_dir=brain)


def test_prepare_rejects_host_dll_for_pi_before_writing(tmp_path):
    model, dll = tmp_path / "brain.swm", tmp_path / "wrong.dll"
    model.write_bytes(b"SWM1")
    dll.write_bytes(b"MZ" + bytes(100))
    with pytest.raises(ValueError, match="Windows binaries"):
        prepare(tmp_path / "rejected", config=ROOT / "deploy/raspberry_pi/node.example.json",
                target="linux-aarch64", model=model, library=dll)
    assert not (tmp_path / "rejected").exists()


def test_native_model_is_selected_and_content_identified(tmp_path, swm_paths):
    from swarm.brain import native
    if not native.available():
        pytest.skip("native engine not built")
    app = prepare(tmp_path / "application", config=ROOT / "deploy/raspberry_pi/node.example.json", target=current_target(),
                  model=swm_paths["int8"], library=native.loaded_path())
    config = json.loads((app / "node.json").read_text())
    digest = hashlib.sha256(swm_paths["int8"].read_bytes()).hexdigest()
    assert config["brain"]["factory"] == "swarm.runtime.handlers:make_native_handler"
    assert config["node"]["workload_id"] == "swm-v1-" + digest


def test_prepared_runtime_crypto_install_and_check(tmp_path, capfd):
    authority = init_authority(tmp_path / "authority")
    identity = tmp_path / "identity"
    request = init_device(identity, node_id="pi-01")
    issued = enroll(request, authority_dir=authority, output_dir=tmp_path / "enrolled",
                    request_sha256=request_fingerprint(request))
    accept_enrollment(identity, enrollment_dir=issued, ca_certificate=authority / "ca.pem")
    config = json.loads((ROOT / "deploy/raspberry_pi/node.example.json").read_text())
    config["node"]["host"] = "127.0.0.1"
    config["node"]["port"] = 0
    local = tmp_path / "node.json"
    local.write_text(json.dumps(config))
    app = prepare(tmp_path / "application", config=local, target="python-any")
    bundle = pack(app, tmp_path / "release.swarmbundle", application_id="main-brain", version=1, node_id="pi-01",
                  target="python-any", entrypoint="main.py", signing_key=authority / "signing-key.pem", recipient=request)
    state = tmp_path / "installed"
    install(bundle, identity_dir=identity, trust_key=authority / "signing-public.pem", root=state)
    assert launch(state, identity_dir=identity, args=["--check"]) == 0
    assert '"valid": true' in capfd.readouterr().out
    assert launch(state, identity_dir=identity, args=["--run-seconds", "0.05"]) == 0
    assert '"hardware_enabled": false' in capfd.readouterr().out


def test_setup_entrypoint_help_works_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "setup_device.py"), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and "prepare" in result.stdout


def test_real_tls_demo_handles_lost_worker(tmp_path):
    import asyncio
    result = asyncio.run(demo(tmp_path))
    assert result["remote_jobs"] == 1
    assert result["local_jobs_after_failover"] == 1
    assert result["hardware_enabled"] is False