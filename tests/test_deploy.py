"""Deployment crypto, hostile inputs and CLI tests; temporary files, no hardware I/O."""

from __future__ import annotations

import hashlib
import io
import json
import os
import ssl
import stat
import struct
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cryptography", minversion="44")

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID

from swarm.deploy import (
    DeployError, FileRecord, Manifest, accept_enrollment, certificate_fingerprint,
    current_target, enroll, export_usb, init_authority, init_device, install, launch,
    pack, public_key_fingerprint, request_fingerprint, status, verify_bundle,
)
from swarm.deploy import _common, bundle as bundle_module, installer as installer_module
from swarm.deploy.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.filterwarnings("ignore:Windows mode bits do not secure ACLs:UserWarning")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _private(path: Path):
    # Never display private key contents, even in assertion failures.
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    root = tmp_path_factory.mktemp("deploy-identities")
    authority = init_authority(root / "authority")
    identity = root / "pi-01"
    request = init_device(identity, node_id="pi-01", ip_addresses=["192.0.2.10", "2001:db8::10"])
    other = root / "pi-02"
    init_device(other, node_id="pi-02")
    enrollment = enroll(request, authority_dir=authority, output_dir=root / "enrollment",
                        request_sha256=request_fingerprint(request), days=90)
    credentials = accept_enrollment(identity, enrollment_dir=enrollment, ca_certificate=authority / "ca.pem")
    return SimpleNamespace(root=root, authority=authority, identity=identity, other=other,
                           request=request, enrollment=enrollment, credentials=credentials,
                           signing=authority / "signing-key.pem", trust=authority / "signing-public.pem")


@pytest.fixture
def application(tmp_path):
    root = tmp_path / "application"
    (root / "python").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "main.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\nfrom selected_brain import infer\n"
        "print(json.dumps({'result': infer(Path('config.json').read_bytes()), 'args': sys.argv[1:], "
        "'node_id': os.environ.get('SWARM_NODE_ID'), 'cwd': str(Path.cwd()), "
        "'pythonpath': os.environ.get('PYTHONPATH'), 'identity': os.environ.get('SWARM_IDENTITY_DIR')}))\n",
        encoding="utf-8")
    (root / "python" / "selected_brain.py").write_text("def infer(data):\n    return 'local-inference-ok'\n", encoding="utf-8")
    (root / "config.json").write_text('{"brain":"selected-demo"}', encoding="utf-8")
    (root / "models" / "weights.swm").write_bytes(b"SWM1\x00model-data-not-executed")
    return root


def _pack(application, tmp_path, keys, *, name="update", **overrides):
    values = dict(application_id="demo", version=1, node_id="pi-01", target="python-any",
                  entrypoint="main.py", signing_key=keys.signing, recipient=keys.request)
    values.update(overrides)
    return pack(application, tmp_path / (name + ".swarmbundle"), **values)


def _verify(path, keys, **overrides):
    values = dict(identity_dir=keys.identity, trust_key=keys.trust)
    values.update(overrides)
    return verify_bundle(path, **values)


def _install(path, keys, root, **overrides):
    values = dict(identity_dir=keys.identity, trust_key=keys.trust, root=root)
    values.update(overrides)
    return install(path, **values)


def _save(path, raw):
    path.write_bytes(raw)
    return path


def _rewrite_header(raw, mutate, keys, *, resign=False):
    framing = len(bundle_module.MAGIC) + 4
    size = struct.unpack_from(">I", raw, len(bundle_module.MAGIC))[0]
    header = json.loads(raw[framing:framing + size])
    mutate(header)
    encoded = _common.canonical_json(header)
    message = bundle_module.MAGIC + struct.pack(">I", len(encoded)) + encoded + raw[framing + size:-64]
    signature = _private(keys.signing).sign(bundle_module.SIGN_CONTEXT + message) if resign else raw[-64:]
    return message + signature


def _manifest(records=None):
    return Manifest("demo", 1, "pi-01", "python-any", "main.py", tuple(records or [
        FileRecord("main.py", 5, _hash(b"pass\n"), False)]))


def _zip(entries=None, *, compression=zipfile.ZIP_STORED):
    entries = entries if entries is not None else [("main.py", b"pass\n", stat.S_IFREG | 0o400)]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", allowZip64=False) as archive:
        for name, data, mode in entries:
            item = zipfile.ZipInfo(name)
            item.create_system = 3
            item.external_attr = mode << 16
            item.compress_type = compression
            archive.writestr(item, data)
    return stream.getvalue()


def _forge(tmp_path, keys, *, manifest=None, archive=None, name="hostile"):
    public = serialization.load_pem_public_key((keys.identity / "encryption-public.pem").read_bytes())
    raw = bundle_module._seal_payload(manifest or _manifest(), _zip() if archive is None else archive,
                                      _private(keys.signing), public)
    return _save(tmp_path / (name + ".swarmbundle"), raw)


def test_crypto_roundtrip_install_status_and_launch(application, tmp_path, keys, capfd):
    packed = _pack(application, tmp_path, keys)
    verified = _verify(packed, keys, node_id="pi-01", application_id="demo")
    assert verified.manifest.version == 1
    assert verified.bundle_sha256 == _hash(packed.read_bytes())
    for record, data in zip(verified.manifest.files, verified.contents):
        assert data == (application / record.path).read_bytes()
    root = tmp_path / "state"
    assert status(root) is None and not root.exists()
    installed = _install(packed, keys, root)
    assert installed.release is not None and installed.release.parent == root / "releases"
    assert status(root) == installed
    state = json.loads((root / "current.json").read_bytes())
    assert state["version"] == 1 and state["release"] == installed.release.relative_to(root).as_posix()
    assert not (installed.release / "tls-key.pem").exists()
    assert not (installed.release / "encryption-key.pem").exists()
    assert launch(root, identity_dir=keys.identity, args=["literal;not-a-shell", "with spaces"]) == 0
    result = json.loads(capfd.readouterr().out.strip())
    assert result["result"] == "local-inference-ok"
    assert result["args"] == ["literal;not-a-shell", "with spaces"]
    assert Path(result["cwd"]) == installed.release
    assert result["pythonpath"] == str(installed.release / "python")
    assert result["identity"] == str(keys.identity)
    assert result["node_id"] == "pi-01"
    assert not list(installed.release.rglob("__pycache__"))
    if os.name == "posix":
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE((root / "current.json").stat().st_mode) == 0o600
        assert stat.S_IMODE(installed.release.stat().st_mode) == 0o500
        assert stat.S_IMODE((installed.release / "main.py").stat().st_mode) == 0o400


def test_enrollment_certificates_and_key_separation(keys):
    ca = x509.load_pem_x509_certificate((keys.enrollment / "ca.pem").read_bytes())
    certificate = x509.load_pem_x509_certificate((keys.credentials / "tls-cert.pem").read_bytes())
    certificate.verify_directly_issued_by(ca)
    csr = x509.load_pem_x509_csr((keys.identity / "tls.csr.pem").read_bytes())
    assert csr.is_signature_valid
    assert isinstance(_private(keys.identity / "tls-key.pem"), ec.EllipticCurvePrivateKey)
    assert isinstance(_private(keys.identity / "encryption-key.pem"), rsa.RSAPrivateKey)
    assert _private(keys.identity / "encryption-key.pem").key_size == 3072
    assert isinstance(_private(keys.signing), ed25519.Ed25519PrivateKey)
    now = datetime.now(timezone.utc)
    assert certificate.not_valid_before_utc <= now < certificate.not_valid_after_utc
    assert certificate.not_valid_after_utc - certificate.not_valid_before_utc <= timedelta(days=90)
    assert not certificate.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.path_length == 0
    assert set(certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value) == {
        ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH}
    san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.DNSName) == ["pi-01"]
    assert len(san.get_values_for_type(x509.IPAddress)) == 2
    assert certificate.public_key().public_numbers() == csr.public_key().public_numbers()
    assert certificate_fingerprint(keys.credentials / "tls-cert.pem") == certificate.fingerprint(hashes.SHA256()).hex()
    assert len(public_key_fingerprint(keys.trust)) == 64
    assert set(path.name for path in keys.enrollment.iterdir()) == {"ca.pem", "tls-cert.pem"}
    request = json.loads(keys.request.read_bytes())
    assert set(request) == {"schema", "node_id", "tls_csr_pem", "encryption_public_key_pem", "signature"}
    assert public_key_fingerprint(keys.identity / "encryption-public.pem") != public_key_fingerprint(keys.other / "encryption-public.pem")


def test_generated_credentials_support_tls13_mutual_authentication_in_memory(keys, tmp_path):
    other_request = keys.other / "enrollment-request.json"
    other_enrollment = enroll(other_request, authority_dir=keys.authority, output_dir=tmp_path / "other-certificate",
                              request_sha256=request_fingerprint(other_request))
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.minimum_version = ssl.TLSVersion.TLSv1_3
    server_context.maximum_version = ssl.TLSVersion.TLSv1_3
    server_context.verify_mode = ssl.CERT_REQUIRED
    server_context.load_verify_locations(cafile=str(keys.credentials / "ca.pem"))
    server_context.load_cert_chain(str(keys.credentials / "tls-cert.pem"), str(keys.identity / "tls-key.pem"))
    client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client_context.minimum_version = ssl.TLSVersion.TLSv1_3
    client_context.maximum_version = ssl.TLSVersion.TLSv1_3
    client_context.load_verify_locations(cafile=str(other_enrollment / "ca.pem"))
    client_context.load_cert_chain(str(other_enrollment / "tls-cert.pem"), str(keys.other / "tls-key.pem"))

    def handshake(hostname):
        client_in, client_out, server_in, server_out = (ssl.MemoryBIO() for _ in range(4))
        client = client_context.wrap_bio(client_in, client_out, server_hostname=hostname)
        server = server_context.wrap_bio(server_in, server_out, server_side=True)
        done = [False, False]
        for _ in range(20):
            for index, side in enumerate((client, server)):
                if not done[index]:
                    try:
                        side.do_handshake()
                        done[index] = True
                    except ssl.SSLWantReadError:
                        pass
            client_in.write(server_out.read())
            server_in.write(client_out.read())
            if all(done):
                return client, server
        pytest.fail("bounded in-memory TLS handshake did not finish")

    client, server = handshake("pi-01")
    assert client.version() == server.version() == "TLSv1.3"
    assert _hash(client.getpeercert(binary_form=True)) == certificate_fingerprint(keys.credentials / "tls-cert.pem")
    assert _hash(server.getpeercert(binary_form=True)) == certificate_fingerprint(other_enrollment / "tls-cert.pem")
    with pytest.raises(ssl.SSLCertVerificationError):
        handshake("wrong-node")


def test_identity_and_authority_never_overwritten(keys):
    before = {name: _hash((keys.identity / name).read_bytes()) for name in ("tls-key.pem", "encryption-key.pem")}
    with pytest.raises(DeployError, match="already exists"):
        init_device(keys.identity, node_id="pi-01")
    with pytest.raises(DeployError, match="already exists"):
        init_authority(keys.authority)
    with pytest.raises(DeployError, match="already exists"):
        accept_enrollment(keys.identity, enrollment_dir=keys.enrollment, ca_certificate=keys.authority / "ca.pem")
    assert before == {name: _hash((keys.identity / name).read_bytes()) for name in before}


@pytest.mark.parametrize("node_id", ["", "PI-01", "-pi", "pi.local", "../pi", "pi_1", "p" * 64, "pi\x00", 12, True])
def test_bad_node_ids_create_nothing(tmp_path, node_id):
    with pytest.raises(DeployError):
        init_device(tmp_path / "identity", node_id=node_id)
    assert not (tmp_path / "identity").exists()


@pytest.mark.parametrize("ips", [["bad-ip"], ["127.0.0.1", "127.0.0.1"], ["fe80::1%eth0"], "127.0.0.1"])
def test_bad_ip_sans_create_nothing(tmp_path, ips):
    with pytest.raises(DeployError):
        init_device(tmp_path / "identity", node_id="pi-01", ip_addresses=ips)
    assert not (tmp_path / "identity").exists()


@pytest.mark.parametrize("days", [0, -1, 91, True, "30"])
def test_enrollment_lifetime_bounds(keys, tmp_path, days):
    with pytest.raises(DeployError, match="lifetime"):
        enroll(keys.request, authority_dir=keys.authority, output_dir=tmp_path / "enrolled",
               request_sha256=request_fingerprint(keys.request), days=days)
    assert not (tmp_path / "enrolled").exists()


def test_enrollment_pinning_and_request_key_substitution_rejected(keys, tmp_path):
    data = json.loads(keys.request.read_bytes())
    data["encryption_public_key_pem"] = (keys.other / "encryption-public.pem").read_text(encoding="ascii")
    request = _save(tmp_path / "substituted.json", _common.canonical_json(data))
    with pytest.raises(DeployError, match="out-of-band"):
        enroll(request, authority_dir=keys.authority, output_dir=tmp_path / "rejected",
               request_sha256=request_fingerprint(keys.request))
    with pytest.raises(DeployError, match="proof"):
        enroll(request, authority_dir=keys.authority, output_dir=tmp_path / "rejected",
               request_sha256=request_fingerprint(request))
    assert not (tmp_path / "rejected").exists()


def test_accept_checks_pinned_ca_local_key_and_supports_new_renewal_directory(keys, tmp_path):
    wrong = init_authority(tmp_path / "wrong-ca")
    with pytest.raises(DeployError, match="pinned CA"):
        accept_enrollment(keys.identity, enrollment_dir=keys.enrollment, ca_certificate=wrong / "ca.pem",
                          credentials_dir=tmp_path / "bad")
    other_request = init_device(tmp_path / "same-name-new-key", node_id="pi-01")
    incoming = enroll(other_request, authority_dir=keys.authority, output_dir=tmp_path / "other-enrollment",
                      request_sha256=request_fingerprint(other_request))
    with pytest.raises(DeployError, match="local TLS key"):
        accept_enrollment(keys.identity, enrollment_dir=incoming, ca_certificate=keys.authority / "ca.pem",
                          credentials_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    renewed = accept_enrollment(keys.identity, enrollment_dir=keys.enrollment, ca_certificate=keys.authority / "ca.pem",
                                credentials_dir=tmp_path / "credentials-next")
    assert certificate_fingerprint(renewed / "tls-cert.pem") == certificate_fingerprint(keys.credentials / "tls-cert.pem")


@pytest.mark.parametrize("version", [0, -1, True, 1 << 63, "1"])
def test_positive_version_required(application, tmp_path, keys, version):
    with pytest.raises(DeployError, match="version"):
        _pack(application, tmp_path, keys, version=version)


def test_wrong_trust_node_recipient_application_and_target(application, tmp_path, keys):
    good = _pack(application, tmp_path, keys)
    other_public = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    wrong_trust = _save(tmp_path / "wrong-public.pem", other_public)
    with pytest.raises(DeployError, match="signature"):
        _verify(good, keys, trust_key=wrong_trust)
    with pytest.raises(DeployError, match="node_id"):
        _verify(good, keys, identity_dir=keys.other)
    with pytest.raises(DeployError, match="node_id"):
        _verify(good, keys, node_id="pi-02")
    with pytest.raises(DeployError, match="application_id"):
        _verify(good, keys, application_id="not-selected")
    wrong_device = _pack(application, tmp_path, keys, name="wrong-device", recipient=keys.other / "encryption-public.pem")
    with pytest.raises(DeployError, match="recipient"):
        _verify(wrong_device, keys)
    foreign = "linux-aarch64" if current_target() != "linux-aarch64" else "windows-aarch64"
    wrong_target = _pack(application, tmp_path, keys, name="wrong-target", target=foreign)
    with pytest.raises(DeployError, match="target mismatch"):
        _verify(wrong_target, keys)


def test_windows_arm64_is_not_linux_arm64(monkeypatch):
    monkeypatch.setattr(_common.sys, "platform", "win32")
    monkeypatch.setattr(_common.sysconfig, "get_platform", lambda: "win-arm64")
    assert current_target() == "windows-aarch64"
    with pytest.raises(DeployError, match="target mismatch"):
        _common.require_target("linux-aarch64")
    monkeypatch.setattr(_common.sysconfig, "get_platform", lambda: "win-amd64")
    assert current_target() == "windows-x86_64"


@pytest.mark.parametrize("part", ["header", "wrapped_key", "ciphertext", "signature"])
def test_unsigned_tampering_fails_before_metadata_or_decryption(application, tmp_path, keys, monkeypatch, part):
    path = _pack(application, tmp_path, keys)
    raw = bytearray(path.read_bytes())
    if part == "wrapped_key":
        mutated = _rewrite_header(bytes(raw), lambda h: h.update(wrapped_key=("A" if h["wrapped_key"][0] != "A" else "B")
                                                                 + h["wrapped_key"][1:]), keys)
    else:
        offset = {"header": len(bundle_module.MAGIC) + 4, "ciphertext": len(raw) - 65,
                  "signature": len(raw) - 1}[part]
        raw[offset] ^= 1
        mutated = bytes(raw)
    forged = _save(tmp_path / "tampered.swarmbundle", mutated)
    monkeypatch.setattr(bundle_module, "parse_json", lambda *a, **k: pytest.fail("unauthenticated header was parsed"))
    with pytest.raises(DeployError, match="signature"):
        _verify(forged, keys)


@pytest.mark.parametrize("part", ["nonce", "wrapped_key", "ciphertext"])
def test_even_resigned_corruption_fails_aead_or_key_unwrap(application, tmp_path, keys, part):
    path = _pack(application, tmp_path, keys)
    raw = path.read_bytes()
    if part == "ciphertext":
        message = bytearray(raw[:-64])
        message[-1] ^= 1
        changed = bytes(message) + _private(keys.signing).sign(bundle_module.SIGN_CONTEXT + bytes(message))
    else:
        changed = _rewrite_header(raw, lambda h: h.update({part: ("A" if h[part][0] != "A" else "B") + h[part][1:]}),
                                   keys, resign=True)
    with pytest.raises(DeployError, match="decryption/authentication"):
        _verify(_save(tmp_path / "resigned.swarmbundle", changed), keys)


def test_signature_context_and_noncanonical_json_rejected(application, tmp_path, keys):
    good = _pack(application, tmp_path, keys).read_bytes()
    without_context = good[:-64] + _private(keys.signing).sign(good[:-64])
    with pytest.raises(DeployError, match="signature"):
        _verify(_save(tmp_path / "no-context.swarmbundle", without_context), keys)
    framing = len(bundle_module.MAGIC) + 4
    size = struct.unpack_from(">I", good, len(bundle_module.MAGIC))[0]
    header = good[framing:framing + size]
    for bad in (b" " + header, b'{"schema":1,' + header[1:]):
        message = bundle_module.MAGIC + struct.pack(">I", len(bad)) + bad + good[framing + size:-64]
        signed = message + _private(keys.signing).sign(bundle_module.SIGN_CONTEXT + message)
        with pytest.raises(DeployError, match="JSON"):
            _verify(_save(tmp_path / "bad-json.swarmbundle", signed), keys)


@pytest.mark.parametrize("name", ["../main.py", "/main.py", "C:/main.py", "C:main.py", "a\\main.py",
                                    "a//main.py", ".git/main.py", "__pycache__/main.py", "NUL.py",
                                    "con/main.py", "com1.py", "a./main.py", "main.py\x00ignored",
                                    "a/../../main.py", "x" * 81 + ".py"])
def test_authenticated_manifest_unsafe_paths_rejected(keys, tmp_path, name):
    manifest = _manifest([FileRecord(name, 5, _hash(b"pass\n"), False)])
    path = _forge(tmp_path, keys, manifest=manifest)
    with pytest.raises(DeployError):
        _install(path, keys, tmp_path / "state")
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("paths", [["MAIN.py", "main.py"], ["main.py", "main.py"],
                                     ["a", "a/main.py", "main.py"], ["A/one.py", "a/two.py", "main.py"]])
def test_authenticated_manifest_duplicate_and_conflicting_paths(keys, tmp_path, paths):
    records = [FileRecord(name, 5, _hash(b"pass\n"), False) for name in sorted(paths)]
    with pytest.raises(DeployError):
        _verify(_forge(tmp_path, keys, manifest=_manifest(records)), keys)


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o400, stat.S_IFDIR | 0o500,
                                    stat.S_IFREG | 0o777, stat.S_IFREG | 0o4400, stat.S_IFREG | 0o600])
def test_zip_links_special_files_and_unsafe_modes_rejected(keys, tmp_path, mode):
    path = _forge(tmp_path, keys, archive=_zip([("main.py", b"pass\n", mode)]))
    with pytest.raises(DeployError, match="permissions"):
        _verify(path, keys)


@pytest.mark.parametrize("kind", ["missing", "extra", "renamed", "mutated", "duplicate", "traversal"])
def test_zip_must_exactly_match_manifest(keys, tmp_path, kind):
    entries = [("main.py", b"pass\n", stat.S_IFREG | 0o400)]
    if kind == "missing":
        entries = []
    elif kind == "extra":
        entries.append(("extra.py", b"pass\n", stat.S_IFREG | 0o400))
    elif kind == "renamed":
        entries[0] = ("MAIN.py", b"pass\n", stat.S_IFREG | 0o400)
    elif kind == "mutated":
        entries[0] = ("main.py", b"evil\n", stat.S_IFREG | 0o400)
    elif kind == "duplicate":
        entries.append(entries[0])
    else:
        entries[0] = ("../main.py", b"pass\n", stat.S_IFREG | 0o400)
    if kind == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive = _zip(entries)
    else:
        archive = _zip(entries)
    with pytest.raises(DeployError):
        _verify(_forge(tmp_path, keys, archive=archive), keys)


@pytest.mark.parametrize("kind", ["deflated", "too-many", "count-lie", "oversized", "overlap", "trailer"])
def test_zip_bombs_and_corrupt_structures_bounded_before_zipfile(keys, tmp_path, monkeypatch, kind):
    if kind in {"too-many", "count-lie"}:
        archive = bytearray(_zip([(f"file{i}.py", b"", stat.S_IFREG | 0o400)
                                   for i in range(_common.MAX_FILES + 1)]))
        if kind == "count-lie":
            struct.pack_into("<HH", archive, len(archive) - 22 + 8, 1, 1)
    elif kind == "deflated":
        archive = bytearray(_zip(compression=zipfile.ZIP_DEFLATED))
    else:
        archive = bytearray(_zip())
        central = archive.index(b"PK\x01\x02")
        if kind == "oversized":
            struct.pack_into("<II", archive, central + 20, _common.MAX_FILE_BYTES + 1, _common.MAX_FILE_BYTES + 1)
        elif kind == "overlap":
            struct.pack_into("<I", archive, central + 42, central)
        else:
            archive += b"unlisted trailer"
    path = _forge(tmp_path, keys, archive=bytes(archive))
    monkeypatch.setattr(bundle_module.zipfile, "ZipFile", lambda *a, **k: pytest.fail("unsafe ZIP reached ZipFile allocation"))
    with pytest.raises(DeployError):
        _verify(path, keys)


def test_local_zip_header_and_missing_hash_are_rejected(keys, tmp_path):
    archive = bytearray(_zip())
    archive[30] = ord("M")
    with pytest.raises(DeployError, match="local header"):
        _verify(_forge(tmp_path, keys, archive=bytes(archive)), keys)
    path = _forge(tmp_path, keys, name="missing-hash")
    changed = _rewrite_header(path.read_bytes(), lambda h: h["manifest"]["files"][0].pop("sha256"), keys, resign=True)
    with pytest.raises(DeployError, match="file record"):
        _verify(_save(tmp_path / "missing-record.swarmbundle", changed), keys)


@pytest.mark.parametrize("path", [".env", ".git/config", "__pycache__/cached.pyc", "secrets.json", "keys.pem",
                                   "node_modules/example.js", "venv/app.py", "identity/identity.json"])
def test_pack_refuses_common_secret_and_cache_paths(application, tmp_path, keys, path):
    file = application / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"not private material")
    with pytest.raises(DeployError, match="payload path"):
        _pack(application, tmp_path, keys)


def test_pack_refuses_private_material_even_under_innocent_name(application, tmp_path, keys):
    (application / "innocent.txt").write_bytes(b"-----BEGIN PRIVATE KEY-----\nFAKE TEST DATA\n-----END PRIVATE KEY-----")
    with pytest.raises(DeployError, match="private key material"):
        _pack(application, tmp_path, keys)


def _symlink(source: Path, destination: Path, *, directory=False):
    try:
        destination.symlink_to(source, target_is_directory=directory)
    except OSError:
        pytest.skip("OS does not grant symlink creation to this test account")


def test_pack_rejects_symlinks(application, tmp_path, keys):
    _symlink(application / "main.py", application / "linked.py")
    with pytest.raises(DeployError, match="links"):
        _pack(application, tmp_path, keys)


def test_pack_rejects_hardlinks(application, tmp_path, keys):
    try:
        os.link(application / "main.py", application / "hardlink.py")
    except OSError:
        pytest.skip("filesystem does not support hardlinks")
    with pytest.raises(DeployError, match="linked"):
        _pack(application, tmp_path, keys)


def test_identity_install_and_usb_paths_reject_symlinks(application, tmp_path, keys):
    existing = tmp_path / "real-directory"
    existing.mkdir(mode=0o700)
    linked = tmp_path / "linked-directory"
    _symlink(existing, linked, directory=True)
    with pytest.raises(DeployError, match="symlinks"):
        init_device(linked / "identity", node_id="pi-01")
    packed = _pack(application, tmp_path, keys)
    with pytest.raises(DeployError, match="symlinks"):
        _install(packed, keys, linked)
    with pytest.raises(DeployError, match="symlinks"):
        export_usb(packed, linked, trust_key=keys.trust)


def test_payload_source_and_bundle_bounds(application, tmp_path, keys, monkeypatch):
    good = _pack(application, tmp_path, keys)
    with monkeypatch.context() as scoped:
        scoped.setattr(bundle_module, "MAX_BUNDLE_BYTES", 128)
        with pytest.raises(DeployError, match="allowed size"):
            _verify(good, keys)
    with monkeypatch.context() as scoped:
        scoped.setattr(bundle_module, "MAX_FILE_BYTES", 2)
        with pytest.raises(DeployError, match="size limit"):
            _pack(application, tmp_path, keys, name="too-large-file")
    raw = bytearray(good.read_bytes())
    struct.pack_into(">I", raw, len(bundle_module.MAGIC), _common.MAX_HEADER_BYTES + 1)
    with pytest.raises(DeployError, match="header length"):
        _verify(_save(tmp_path / "large-header.swarmbundle", bytes(raw)), keys)


def test_native_binary_platform_guards(application, tmp_path, keys):
    (application / "engine.dll").write_bytes(b"MZ" + b"\x00" * 100)
    with pytest.raises(DeployError, match="Windows binaries"):
        _pack(application, tmp_path, keys, target="linux-aarch64")
    with pytest.raises(DeployError, match="python-any"):
        _pack(application, tmp_path, keys)
    (application / "engine.dll").unlink()
    elf = bytearray(64)
    elf[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", elf, 18, 62)  # x86_64 cannot be mislabeled as Pi aarch64.
    (application / "engine.so").write_bytes(elf)
    with pytest.raises(DeployError, match="architecture"):
        _pack(application, tmp_path, keys, target="linux-aarch64")


def test_replay_downgrade_and_app_switch_preserve_high_water(application, tmp_path, keys):
    root = tmp_path / "state"
    v2 = _pack(application, tmp_path, keys, name="v2", version=2)
    installed = _install(v2, keys, root)
    before = (root / "current.json").read_bytes()
    for path in (v2, _pack(application, tmp_path, keys, name="v1", version=1),
                 _pack(application, tmp_path, keys, name="other-app", application_id="other", version=2)):
        with pytest.raises(DeployError, match="replay/downgrade"):
            _install(path, keys, root)
        assert (root / "current.json").read_bytes() == before
        assert status(root) == installed
    v3 = _pack(application, tmp_path, keys, name="v3", application_id="other", version=3)
    assert _install(v3, keys, root).manifest.application_id == "other"
    assert installed.release.exists()


def test_dry_run_no_writes_and_no_reservation(application, tmp_path, keys):
    root = tmp_path / "not-created"
    packed = _pack(application, tmp_path, keys)
    verified = _install(packed, keys, root, dry_run=True)
    assert verified.dry_run and verified.release is None
    assert not root.exists()
    _install(packed, keys, root)
    before = (root / "current.json").read_bytes()
    with pytest.raises(DeployError, match="replay/downgrade"):
        _install(packed, keys, root, dry_run=True)
    higher = _pack(application, tmp_path, keys, name="higher", version=2)
    assert _install(higher, keys, root, dry_run=True).manifest.version == 2
    assert (root / "current.json").read_bytes() == before
    assert not list(root.glob(".stage-*"))


def test_crypto_failure_and_commit_failure_leave_previous_active_unchanged(application, tmp_path, keys, monkeypatch):
    root = tmp_path / "state"
    previous = _install(_pack(application, tmp_path, keys), keys, root)
    before = (root / "current.json").read_bytes()
    newer = _pack(application, tmp_path, keys, name="newer", version=2)
    mutated = bytearray(newer.read_bytes())
    mutated[-1] ^= 1
    with pytest.raises(DeployError):
        _install(_save(tmp_path / "broken.swarmbundle", mutated), keys, root)
    real_replace = installer_module.os.replace

    def fail_commit(src, dst):
        if Path(dst) == root / "current.json":
            raise OSError("injected atomic replacement failure")
        return real_replace(src, dst)

    monkeypatch.setattr(installer_module.os, "replace", fail_commit)
    with pytest.raises(OSError, match="injected"):
        _install(newer, keys, root)
    assert (root / "current.json").read_bytes() == before
    assert status(root) == previous
    assert not list(root.glob(".stage-*")) and not list(root.glob(".current-*.tmp"))


def test_failed_staging_and_lock_contention_preserve_previous(application, tmp_path, keys, monkeypatch):
    root = tmp_path / "state"
    previous = _install(_pack(application, tmp_path, keys), keys, root)
    newer = _pack(application, tmp_path, keys, name="newer", version=2)
    with installer_module._install_lock(root):
        with pytest.raises(DeployError, match="lock"):
            _install(newer, keys, root)
    assert status(root) == previous
    with monkeypatch.context() as scoped:
        scoped.setattr(installer_module.os, "rename", lambda *a, **k: (_ for _ in ()).throw(OSError("injected staging failure")))
        with pytest.raises(OSError, match="staging failure"):
            _install(newer, keys, root)
    assert status(root) == previous and not list(root.glob(".stage-*"))
    assert _install(newer, keys, root).manifest.version == 2


@pytest.mark.parametrize("kind", ["mutated-file", "extra-file", "missing-file", "pointer-traversal", "manifest", "node"])
def test_status_and_launch_fail_closed_on_local_corruption(application, tmp_path, keys, kind):
    root = tmp_path / "state"
    installed = _install(_pack(application, tmp_path, keys), keys, root)
    assert installed.release is not None
    if kind in {"mutated-file", "missing-file", "extra-file"}:
        os.chmod(installed.release, 0o700)
        file = installed.release / ("extra.py" if kind == "extra-file" else "main.py")
        if file.exists():
            os.chmod(file, 0o600)
        if kind == "missing-file":
            file.unlink()
        else:
            file.write_bytes(b"not approved")
            os.chmod(file, 0o400)
        os.chmod(installed.release, 0o500)
    elif kind == "manifest":
        file = installed.release / ".swarm-manifest.json"
        os.chmod(file, 0o600)
        file.write_bytes(b"{}")
        os.chmod(file, 0o400)
    else:
        file = root / "current.json"
        state = json.loads(file.read_bytes())
        state["release" if kind == "pointer-traversal" else "node_id"] = "../outside" if kind == "pointer-traversal" else "other-node"
        file.write_bytes(_common.canonical_json(state))
    with pytest.raises(DeployError):
        status(root)
    with pytest.raises(DeployError):
        launch(root)


def test_trust_and_identity_must_remain_outside_install_root(application, tmp_path, keys):
    packed = _pack(application, tmp_path, keys)
    with pytest.raises(DeployError, match="outside"):
        _install(packed, keys, keys.root)
    with pytest.raises(DeployError, match="outside"):
        _install(packed, keys, keys.authority)


def test_usb_export_is_only_verified_ciphertext_and_never_overwrites(application, tmp_path, keys):
    packed = _pack(application, tmp_path, keys)
    usb = tmp_path / "already-mounted-usb"
    usb.mkdir()
    existing = usb / "do-not-touch.txt"
    existing.write_bytes(b"operator's unrelated data")
    exported = export_usb(packed, usb, trust_key=keys.trust)
    assert exported.read_bytes() == packed.read_bytes()
    assert existing.read_bytes() == b"operator's unrelated data"
    before = _hash(exported.read_bytes())
    with pytest.raises(FileExistsError):
        export_usb(packed, usb, trust_key=keys.trust)
    assert _hash(exported.read_bytes()) == before
    with pytest.raises(FileNotFoundError):
        export_usb(packed, tmp_path / "not-mounted", trust_key=keys.trust)
    assert not (tmp_path / "not-mounted").exists()
    with pytest.raises(DeployError):
        export_usb(keys.identity / "tls-key.pem", usb, trust_key=keys.trust, filename="not-a-bundle.swarmbundle")
    for name in ("../escape.swarmbundle", "C:escape.swarmbundle", "keys.pem", "CON.swarmbundle"):
        with pytest.raises(DeployError):
            export_usb(packed, usb, trust_key=keys.trust, filename=name)
    assert sorted(path.name for path in usb.iterdir()) == ["do-not-touch.txt", packed.name]


def test_pack_never_overwrites_or_self_includes(application, tmp_path, keys):
    packed = _pack(application, tmp_path, keys)
    before = _hash(packed.read_bytes())
    with pytest.raises(DeployError, match="overwrite"):
        _pack(application, tmp_path, keys)
    assert _hash(packed.read_bytes()) == before
    with pytest.raises(DeployError, match="outside"):
        _pack(application, application, keys)


def test_launch_explicit_trusted_paths_and_no_shell(application, tmp_path, keys, monkeypatch):
    root = tmp_path / "state"
    selected = _install(_pack(application, tmp_path, keys), keys, root)
    trusted = tmp_path / "trusted-python"
    trusted.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(trusted))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(installer_module.subprocess, "run", fake_run)
    assert launch(root, args=["; no shell"], inherit_pythonpath=True) == 17
    command, kwargs = calls[0]
    assert command == [sys.executable, "-s", "-B", str(selected.release / "main.py"), "; no shell"]
    assert kwargs["shell"] is False and kwargs["cwd"] == selected.release
    assert kwargs["env"]["PYTHONPATH"] == os.pathsep.join([str(selected.release / "python"), str(trusted)])
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    assert launch(root, trusted_python_paths=[trusted]) == 17
    with pytest.raises(DeployError, match="absolute"):
        launch(root, trusted_python_paths=[Path("relative")])
    with pytest.raises(DeployError, match="arguments"):
        launch(root, args="not-a-list")
    with pytest.raises(DeployError, match="identity"):
        launch(root, identity_dir=keys.other)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions; Windows ACL ownership is an operator responsibility")
def test_posix_private_permissions_and_special_source_file(keys, application, tmp_path):
    for path in (keys.identity / "tls-key.pem", keys.identity / "encryption-key.pem", keys.signing, keys.authority / "ca-key.pem"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    copy = tmp_path / "key-dir"
    copy.mkdir(mode=0o700)
    key = copy / "signing-key.pem"
    key.write_bytes(keys.signing.read_bytes())
    os.chmod(key, 0o644)
    with pytest.raises(DeployError, match="private key|unsafe permissions"):
        _pack(application, tmp_path, keys, signing_key=key)
    os.mkfifo(application / "pipe")
    with pytest.raises(DeployError, match="regular"):
        _pack(application, tmp_path, keys)


def _cli(*args, expected=0):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run([sys.executable, "-m", "swarm.deploy", *(str(arg) for arg in args)],
                               cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    if "PRIVATE KEY-----" in completed.stdout or "PRIVATE KEY-----" in completed.stderr:
        pytest.fail("CLI exposed private key material", pytrace=False)
    assert completed.returncode == expected, completed.stderr
    return completed


def test_full_cli_temporary_pipeline_and_real_demo(tmp_path):
    authority = tmp_path / "authority"
    identity = tmp_path / "device"
    enrollment = tmp_path / "enrollment"
    root = tmp_path / "state"
    bundle = tmp_path / "pi-demo.swarmbundle"
    usb = tmp_path / "mounted"
    usb.mkdir()
    assert "init-device" in _cli("--help").stdout
    assert json.loads(_cli("target").stdout)["target"] == current_target()
    assert json.loads(_cli("status", "--root", root).stdout) == {"installed": False}
    _cli("init-authority", "--authority-dir", authority)
    public = json.loads(_cli("init-device", "--identity-dir", identity, "--node-id", "pi-demo").stdout)
    _cli("enroll", "--request", public["request"], "--request-sha256", public["request_sha256"],
         "--authority-dir", authority, "--output-dir", enrollment, "--days", 30)
    _cli("accept-enrollment", "--identity-dir", identity, "--enrollment-dir", enrollment,
         "--ca-cert", authority / "ca.pem")
    fingerprint = json.loads(_cli("fingerprint", "--certificate", enrollment / "tls-cert.pem").stdout)
    assert fingerprint["sha256"] == certificate_fingerprint(enrollment / "tls-cert.pem")
    _cli("pack", "--application-dir", ROOT / "deploy" / "raspberry_pi" / "demo_application",
         "--output", bundle, "--application-id", "pi-demo", "--version", 1, "--node-id", "pi-demo",
         "--target", "python-any", "--entrypoint", "main.py", "--signing-key", authority / "signing-key.pem",
         "--recipient", public["request"])
    _cli("export-usb", "--bundle", bundle, "--usb-dir", usb, "--trust-key", authority / "signing-public.pem")
    options = ["--bundle", usb / bundle.name, "--identity-dir", identity,
               "--trust-key", authority / "signing-public.pem"]
    assert json.loads(_cli("verify", *options).stdout)["verified"]
    assert json.loads(_cli("install", *options, "--root", root, "--dry-run").stdout)["dry_run"]
    assert not root.exists()
    _cli("install", *options, "--root", root)
    assert json.loads(_cli("status", "--root", root).stdout)["version"] == 1
    launched = json.loads(_cli("launch", "--root", root, "--identity-dir", identity, "--", "--message", "smoke").stdout)
    assert launched == {"application_id": "pi-demo", "node_id": "pi-demo", "hardware_enabled": False,
                        "result": _hash(b"smoke")}
    _cli("install", *options, "--root", root, expected=2)


def test_help_status_launch_imports_do_not_require_crypto_or_numpy(tmp_path):
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import swarm.deploy
from swarm.deploy import current_target, launch, status
assert not any(name == 'cryptography' or name.startswith('cryptography.') or name == 'numpy'
               for name in sys.modules)
assert status(sys.argv[2]) is None
from swarm.deploy.cli import main
raise SystemExit(main(['target']))
"""
    completed = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(ROOT / "python"),
                                str(tmp_path / "absent")], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["target"] == current_target()


def test_cli_bad_input_reports_error_without_traceback(tmp_path, capsys):
    assert cli_main(["init-device", "--identity-dir", str(tmp_path / "bad"), "--node-id", "../bad"]) == 2
    message = capsys.readouterr().err
    assert "deployment error" in message and "Traceback" not in message