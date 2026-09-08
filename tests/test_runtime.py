"""Actual loopback TLS tests. No robot drivers, USB devices or external networks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("cryptography", minversion="44")

from swarm.deploy import accept_enrollment, certificate_fingerprint, enroll, init_authority, init_device, request_fingerprint
from swarm.runtime import ApplicationConfig, InferenceResult, JobError, NodeConfig, Peer, SwarmRuntime, TLSFiles
from swarm.runtime.handlers import NativeHandler, make_digest_handler
from swarm.runtime.protocol import HEADER, Kind, ProtocolError, read_packet, tls_context, validate_local_identity

pytestmark = pytest.mark.filterwarnings("ignore:Windows mode bits do not secure ACLs:UserWarning")


@pytest.fixture(scope="module")
def fleet(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime-fleet")
    authority = init_authority(root / "authority")
    configs = []
    for name in ("pi-a", "pi-b", "pi-c"):
        identity = root / name
        request = init_device(identity, node_id=name)
        issued = enroll(request, authority_dir=authority, output_dir=root / (name + "-enrolled"),
                        request_sha256=request_fingerprint(request))
        credential = accept_enrollment(identity, enrollment_dir=issued, ca_certificate=authority / "ca.pem")
        tls = TLSFiles(credential / "ca.pem", credential / "tls-cert.pem", identity / "tls-key.pem")
        configs.append(NodeConfig(name, "test-v1", tls, heartbeat_s=0.05, peer_timeout_s=0.5,
                                  connect_timeout_s=0.5, job_timeout_s=1.5, estimated_ms=10, port=0))
    return configs


def pair(fleet):
    # Reserve two different loopback port numbers until both are selected.
    with socket.socket() as a, socket.socket() as b:
        a.bind(("127.0.0.1", 0))
        b.bind(("127.0.0.1", 0))
        ports = a.getsockname()[1], b.getsockname()[1]
    left, right = (replace(fleet[i], port=ports[i]) for i in range(2))
    def peer(config):
        return Peer(config.node_id, "127.0.0.1", config.port, certificate_fingerprint(config.tls.certificate))
    return (replace(left, peers=(peer(right),), estimated_ms=1000),
            replace(right, peers=(peer(left),), estimated_ms=1))


def test_real_mutual_tls_remote_work_and_compact_completion(fleet):
    async def run():
        left, right = pair(fleet)
        def handler(payload):
            return InferenceResult(b"hidden not useful" if payload == b"empty" else hashlib.sha256(payload).digest(), payload != b"empty")
        async with SwarmRuntime(right, handler) as b, SwarmRuntime(left, handler) as a:
            await a.wait_for_peers(1, 3)
            assert a.online_peers == ("pi-b",) and b.online_peers == ("pi-a",)
            session = a._sessions["pi-b"]
            assert session.writer.get_extra_info("ssl_object").version() == "TLSv1.3"
            assert await a.submit(b"image data") == InferenceResult(hashlib.sha256(b"image data").digest())
            assert b.counters["local_jobs"] == 1 and a.counters["local_jobs"] == 0
            assert await a.submit(b"empty") == InferenceResult(b"", False)
            assert b.counters["suppressed_results"] == 1
            assert a.status()["preferred_leader"] == "pi-a"
            assert a.counters["application_bytes_sent"] < 2048
    asyncio.run(run())


def test_three_member_mesh_reserves_capacity_before_concurrent_sends(fleet):
    async def run():
        sockets = [socket.socket() for _ in fleet]
        try:
            for s in sockets:
                s.bind(("127.0.0.1", 0))
            ports = [s.getsockname()[1] for s in sockets]
        finally:
            for s in sockets:
                s.close()
        configs = []
        for i, cfg in enumerate(fleet):
            peers = tuple(Peer(other.node_id, "127.0.0.1", ports[j], certificate_fingerprint(other.tls.certificate))
                          for j, other in enumerate(fleet) if i != j)
            configs.append(replace(cfg, port=ports[i], peers=peers, estimated_ms=1000 if i == 0 else 5))
        release = threading.Event()
        def held(data):
            release.wait(2)
            return InferenceResult(data)
        async with AsyncExitStack() as stack:
            nodes = [await stack.enter_async_context(SwarmRuntime(cfg, held)) for cfg in configs]
            try:
                await asyncio.gather(*(node.wait_for_peers(2, 4) for node in nodes))
                jobs = [asyncio.create_task(nodes[0].submit(bytes([i]))) for i in range(6)]
                await asyncio.sleep(0.04)
                reservations = [s.reservations for s in nodes[0]._sessions.values()]
                assert sorted(reservations) == [3, 3]
                release.set()
                assert [r.payload for r in await asyncio.gather(*jobs)] == [bytes([i]) for i in range(6)]
                assert all(nodes[i].counters["local_jobs"] == 3 for i in (1, 2))
                assert nodes[0].counters["local_jobs"] == 0
            finally:
                release.set()
    asyncio.run(run())


def test_worker_disconnect_reassigns_owned_job(fleet):
    async def run():
        left, right = pair(fleet)
        entered, release = threading.Event(), threading.Event()
        def slow(payload):
            entered.set()
            release.wait(2)
            return InferenceResult(payload)
        async with SwarmRuntime(left, lambda p: InferenceResult(p)) as a:
            b = await SwarmRuntime(right, slow).start()
            try:
                await a.wait_for_peers(1, 3)
                task = asyncio.create_task(a.submit(b"idempotent frame"))
                assert await asyncio.to_thread(entered.wait, 1)
                await b.close()
                assert await task == InferenceResult(b"idempotent frame")
                assert a.counters["retried"] == 1 and a.counters["local_jobs"] == 1
                assert a.online_peers == ()
            finally:
                release.set()
                await b.close()
    asyncio.run(run())


@pytest.mark.parametrize("fault", ["wrong-pin", "wrong-workload", "unauthorized", "wrong-dns"])
def test_peer_identity_or_workload_mismatch_fails_closed(fleet, fault):
    async def run():
        left, right = pair(fleet)
        if fault == "wrong-pin":
            left = replace(left, peers=(replace(left.peers[0], fingerprint="0" * 64),))
        elif fault == "wrong-workload":
            right = replace(right, workload_id="other-model-v2")
        elif fault == "unauthorized":
            right = replace(right, peers=())
        else:
            left = replace(left, peers=(replace(left.peers[0], node_id="pi-wrong"),))
        async with SwarmRuntime(right, make_digest_handler({})), SwarmRuntime(left, make_digest_handler({})) as a:
            with pytest.raises(asyncio.TimeoutError):
                await a.wait_for_peers(1, 0.35)
            assert not a.online_peers
            assert (await a.submit(b"local fallback")).payload == hashlib.sha256(b"local fallback").digest()
    asyncio.run(run())


def test_deadlines_do_not_free_a_still_running_thread(fleet):
    async def run():
        entered, release = threading.Event(), threading.Event()
        def blocking(payload):
            entered.set()
            release.wait(2)
            return InferenceResult(payload)
        cfg = replace(fleet[0], queue_limit=1, retries=0)
        async with SwarmRuntime(cfg, blocking) as node:
            try:
                start = time.monotonic()
                with pytest.raises(JobError):
                    await node.submit(b"slow", timeout_s=0.05)
                assert entered.is_set() and time.monotonic() - start < 0.5
                assert len(node._work) == 1
                with pytest.raises(JobError, match="capacity"):
                    await node.submit(b"another")
                assert node.counters["local_jobs"] == 1
            finally:
                release.set()
    asyncio.run(run())


def test_active_and_completed_duplicate_jobs_are_not_reexecuted(fleet):
    async def run():
        calls = []
        async with SwarmRuntime(fleet[0], lambda data: (calls.append(data), InferenceResult(data))[1]) as node:
            key = b"a" * 16
            deadline = time.monotonic() + 1
            first = node._accept("pi-b", key, b"input", deadline)
            assert node._accept("pi-b", key, b"input", deadline) is first
            with pytest.raises(ProtocolError, match="different"):
                node._accept("pi-b", key, b"changed", deadline)
            assert await first == InferenceResult(b"input")
            assert await node._accept("pi-b", key, b"input", deadline) == InferenceResult(b"input")
            assert calls == [b"input"]
            with pytest.raises(ProtocolError):
                node._accept("pi-b", key, b"changed", deadline)
    asyncio.run(run())


@pytest.mark.parametrize("packet", [HEADER.pack(99, 0), HEADER.pack(Kind.JOB, 2**32 - 1),
                                    HEADER.pack(Kind.LOAD, 0), HEADER.pack(Kind.ERROR, 5000)])
def test_wire_lengths_checked_before_reading_body(packet):
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(packet)
        with pytest.raises(ProtocolError):
            await asyncio.wait_for(read_packet(reader, 1024), 0.1)
    asyncio.run(run())


def test_inbound_task_limit_reserved_before_coroutines_run(fleet, monkeypatch):
    async def run():
        left, _ = pair(fleet)
        node = SwarmRuntime(left, make_digest_handler({}))
        gate = asyncio.Event()
        spawned = []
        async def held(*args):
            spawned.append(True)
            await gate.wait()
        monkeypatch.setattr(node, "_serve", held)
        class Writer:
            closed = False
            def close(self):
                self.closed = True
        first, second = Writer(), Writer()
        node._incoming(None, first)
        node._incoming(None, second)
        assert node._inbound_connections == 1 and second.closed
        gate.set()
        await asyncio.gather(*list(node._tasks))
        assert spawned == [True] and node._inbound_connections == 0
        await node.close()
    asyncio.run(run())


def test_runtime_import_and_cli_do_not_require_site_packages():
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import swarm.runtime; "
            "from swarm.runtime.handlers import make_digest_handler; "
            "assert len(make_digest_handler({})(b'input').payload) == 32; "
            "assert not any(n.startswith(('numpy', 'swarm.brain', 'swarm.train', 'swarm.robotics', 'cryptography')) "
            "for n in sys.modules)")
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(root / "python")],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_no_plaintext_or_legacy_tls_context(fleet):
    context = tls_context(fleet[0], server=False)
    assert context.minimum_version == context.maximum_version == ssl.TLSVersion.TLSv1_3
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert not context.hostname_checks_common_name
    assert tls_context(fleet[0], server=True).verify_mode == ssl.CERT_REQUIRED


def test_local_certificate_identity_checked_before_opening_listener(fleet):
    wrong = replace(fleet[0], node_id="pi-wrong")
    with pytest.raises(ssl.SSLCertVerificationError):
        validate_local_identity(wrong, tls_context(wrong, server=True), tls_context(wrong, server=False))


@pytest.mark.parametrize("kwargs", [{"queue_limit": 0}, {"max_payload": 2**32}, {"heartbeat_s": float("nan")},
                                    {"estimated_ms": True}, {"peer_timeout_s": 0.01}, {"workload_id": "a b"}])
def test_invalid_runtime_config_rejected(fleet, kwargs):
    with pytest.raises(ValueError):
        replace(fleet[0], **kwargs)


def test_application_configuration_is_local_and_strict(tmp_path, fleet):
    cfg = fleet[0]
    data = {"schema_version": 1, "node": {"node_id": cfg.node_id, "workload_id": "test-v1",
            "tls": {k: str(v) for k, v in vars(cfg.tls).items()}},
            "brain": {"factory": "swarm.runtime.handlers:make_digest_handler", "options": {}}}
    path = tmp_path / "node.json"
    path.write_text(json.dumps(data))
    assert ApplicationConfig.load(path).node.node_id == cfg.node_id
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        ApplicationConfig.load(path)


def test_native_handler_uses_cpp_without_training_imports(swm_paths, frames):
    from swarm.brain import native
    if not native.available():
        pytest.skip("native library has not been built")
    path = swm_paths["int8"]
    handler = NativeHandler({"library": str(native.loaded_path()), "model": str(path),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "ram_cap_bytes": 200000,
                             "threshold": 0.0, "empty_class": 0})
    try:
        frame = (frames[0] * 255).astype("uint8")
        got = handler(frame.tobytes())
        with native.NativeModel(path) as model:
            expected = model.run(frame.astype("float32") / 255.0)
        best = int(expected.argmax())
        if best != 0:
            index, confidence = struct.unpack("!If", got.payload)
            assert index == best and confidence == pytest.approx(float(expected[best]), abs=1e-6)
        else:
            assert not got.useful and got.payload == b""
    finally:
        handler.close()