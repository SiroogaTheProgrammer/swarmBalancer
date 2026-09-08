"""Authenticated, recipient-encrypted envelopes containing a restricted stored ZIP.

Framing: MAGIC | uint32be(header length) | canonical JSON | ciphertext+GCM tag |
64-byte Ed25519 signature. Signature verification precedes JSON interpretation,
RSA decryption, ZIP parsing, and all use of peer-supplied paths.
"""

from __future__ import annotations

import base64
import hmac
import io
import os
import stat
import struct
import zipfile
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ._common import (
    MAX_BUNDLE_BYTES, MAX_FILE_BYTES, MAX_FILES, MAX_HEADER_BYTES, MAX_PATH_BYTES,
    MAX_UNPACKED_BYTES, DeployError, FileRecord, Manifest, Pathish, VerifiedBundle,
    canonical_json, check_directory, digest, exact_keys, identifier, is_link,
    no_links, parse_json, positive_version, read_regular, relative_path,
    require_target, sha256_hex, target_name, write_new,
)
from .identities import (
    RSA_BITS, _key_fingerprint, _load_private, decode_base64, device_node_id,
    load_recipient, load_signing_public,
)

MAGIC = b"SWMDEP1\x00"
SIGN_CONTEXT = b"swarm.deploy/application-envelope/signature/v1\x00"
AAD_CONTEXT = b"swarm.deploy/application-envelope/aead/v1\x00"
WRAP_CONTEXT = b"swarm.deploy/application-envelope/key-wrap/v1\x00"
ALGORITHM = "Ed25519+AES-256-GCM+RSA-3072-OAEP-SHA256"
SIGNATURE_BYTES = 64
_CENTRAL = struct.Struct("<4s6H3I5H2I")
_LOCAL = struct.Struct("<4s5H3I2H")
_END = struct.Struct("<4s4H2IH")


def _oaep() -> padding.OAEP:
    return padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(),
                        label=WRAP_CONTEXT)


def _binary_target(name: str, data: bytes, target: str) -> None:
    """Catch common cross-OS/CPU mistakes; this is not a complete ABI analyzer."""
    suffix = Path(name).suffix.lower()
    native_suffix = suffix in {".dll", ".exe", ".pyd", ".so", ".dylib", ".a", ".o", ".elf"}
    if target == "python-any" and (native_suffix or data[:4] in {
            b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe"}
            or data.startswith((b"MZ", b"!<arch>\n"))):
        raise DeployError("python-any cannot contain native binaries or libraries")
    if suffix in {".dll", ".exe", ".pyd"} and not target.startswith("windows-"):
        raise DeployError("Windows binaries are not Raspberry Pi/Linux builds")
    if data.startswith(b"MZ"):
        if not target.startswith("windows-") or len(data) < 64:
            raise DeployError("PE binary does not match the target OS")
        offset = struct.unpack_from("<I", data, 60)[0]
        if offset > len(data) - 6 or data[offset:offset + 4] != b"PE\x00\x00":
            raise DeployError("invalid PE binary header")
        machine = struct.unpack_from("<H", data, offset + 4)[0]
        architecture = {0xAA64: "aarch64", 0x8664: "x86_64", 0x14C: "i686"}.get(machine)
        if target != "windows-" + str(architecture):
            raise DeployError("PE binary architecture does not match the target")
    elif data.startswith(b"\x7fELF"):
        if not target.startswith("linux-") or len(data) < 20 or data[5] != 1:
            raise DeployError("ELF binary does not match a supported Linux target")
        machine = struct.unpack_from("<H", data, 18)[0]
        expected = {
            "linux-aarch64": (183, 2), "linux-armv7l": (40, 1), "linux-armv6l": (40, 1),
            "linux-x86_64": (62, 2), "linux-i686": (3, 1),
        }.get(target)
        if (machine, data[4]) != expected:
            raise DeployError("ELF binary architecture does not match the target")
    elif data[:4] in {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe"}:
        if not target.startswith("darwin-") or len(data) < 8:
            raise DeployError("Mach-O binary does not match the target OS")
        architecture = {0x100000C: "aarch64", 0x1000007: "x86_64"}.get(
            struct.unpack_from("<I", data, 4)[0])
        if target != "darwin-" + str(architecture):
            raise DeployError("unsupported Mach-O architecture or fat binary")


def _payload_content(name: str, data: bytes, target: str) -> None:
    # This prevents common mistakes, not arbitrary secret discovery. The operator
    # must still review the deliberately selected application directory.
    if b"-----BEGIN " in data and b"PRIVATE KEY-----" in data:
        raise DeployError("private key material is forbidden in an application payload")
    if data.startswith(b"PuTTY-User-Key-File-"):
        raise DeployError("private key material is forbidden in an application payload")
    _binary_target(name, data, target)


def _source_files(directory: Pathish, target: str) -> tuple[tuple[FileRecord, ...], tuple[bytes, ...]]:
    root = check_directory(directory)
    pending = [root]
    entries_seen = 0
    total = 0
    selected: list[tuple[FileRecord, bytes]] = []
    while pending:
        folder = pending.pop()
        check_directory(folder)
        # scandir is streamed; a huge directory is not first loaded/sorted in RAM.
        with os.scandir(folder) as entries:
            for entry in entries:
                entries_seen += 1
                if entries_seen > MAX_FILES * 17:
                    raise DeployError("application directory contains too many entries")
                path = Path(entry.path)
                relative = relative_path(path.relative_to(root).as_posix())
                # Windows DirEntry's cached find-data has st_nlink=st_ino=0.
                # lstat queries the actual file so hard-link rejection remains real.
                info = path.lstat()
                if is_link(info) or info.st_mode & 0o7000:
                    raise DeployError("links and special permission bits are forbidden in payloads")
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise DeployError("only regular, non-linked payload files are allowed")
                if len(selected) >= MAX_FILES:
                    raise DeployError("payload has too many files")
                if info.st_size > MAX_FILE_BYTES or total + info.st_size > min(MAX_UNPACKED_BYTES, MAX_BUNDLE_BYTES):
                    raise DeployError("payload exceeds the file or total size limit")
                data = read_regular(path, MAX_FILE_BYTES)
                total += len(data)
                if total > min(MAX_UNPACKED_BYTES, MAX_BUNDLE_BYTES):
                    raise DeployError("payload grew beyond the total size limit during collection")
                _payload_content(relative, data, target)
                selected.append((FileRecord(relative, len(data), digest(data), bool(info.st_mode & 0o111)), data))
    selected.sort(key=lambda item: item[0].path)
    return tuple(item[0] for item in selected), tuple(item[1] for item in selected)


def _stored_zip(records: tuple[FileRecord, ...], contents: tuple[bytes, ...]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for record, data in zip(records, contents):
            item = zipfile.ZipInfo(record.path, date_time=(1980, 1, 1, 0, 0, 0))
            item.create_system = 3
            item.compress_type = zipfile.ZIP_STORED
            item.external_attr = (stat.S_IFREG | (0o500 if record.executable else 0o400)) << 16
            archive.writestr(item, data)
    return stream.getvalue()


def _seal_payload(manifest: Manifest, payload: bytes, signing: ed25519.Ed25519PrivateKey,
                  recipient: rsa.RSAPublicKey) -> bytes:
    if len(payload) > MAX_BUNDLE_BYTES:
        raise DeployError("stored ZIP exceeds the bundle size limit")
    key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    header = {"schema": 1, "algorithm": ALGORITHM, "manifest": manifest.as_dict(),
              "recipient_sha256": _key_fingerprint(recipient),
              "nonce": base64.b64encode(nonce).decode("ascii"),
              "wrapped_key": base64.b64encode(recipient.encrypt(key, _oaep())).decode("ascii"),
              "ciphertext_bytes": len(payload) + 16}
    encoded = canonical_json(header)
    if len(encoded) > MAX_HEADER_BYTES:
        raise DeployError("bundle header exceeds its size limit")
    prefix = MAGIC + struct.pack(">I", len(encoded)) + encoded
    if len(prefix) + len(payload) + 16 + SIGNATURE_BYTES > MAX_BUNDLE_BYTES:
        raise DeployError("encrypted bundle exceeds the size limit")
    ciphertext = AESGCM(key).encrypt(nonce, payload, AAD_CONTEXT + prefix)
    message = prefix + ciphertext
    return message + signing.sign(SIGN_CONTEXT + message)


def pack(application_dir: Pathish, output: Pathish, *, application_id: str, version: int,
         node_id: str, target: str, entrypoint: str, signing_key: Pathish,
         recipient: Pathish) -> Path:
    """Package only the explicitly selected clean directory. Never execute it.

    recipient is an approved RSA public PEM or the retained public enrollment
    request. Output is a new file; its parent must already exist.
    """
    application_id = identifier(application_id, "application_id")
    version = positive_version(version)
    node_id = identifier(node_id)
    target = target_name(target)
    entrypoint = relative_path(entrypoint)
    source = check_directory(application_dir)
    destination = no_links(output)
    if destination == source or source in destination.parents:
        raise DeployError("bundle output must be outside the application directory")
    if destination.exists():
        raise DeployError("bundle output already exists; refusing to overwrite it")
    signing = _load_private(signing_key, ed25519.Ed25519PrivateKey)
    encryption_public = load_recipient(recipient)
    records, contents = _source_files(source, target)
    manifest = Manifest.from_dict(Manifest(application_id, version, node_id, target,
                                           entrypoint, records).as_dict())
    return write_new(destination, _seal_payload(manifest, _stored_zip(records, contents),
                                                signing, encryption_public))


def _read_signed_envelope(bundle: Pathish, trust_key: Pathish) -> tuple[bytes, dict[str, Any], Manifest, int]:
    raw = read_regular(bundle, MAX_BUNDLE_BYTES, owned=False)
    framing = len(MAGIC) + 4
    if len(raw) < framing + 2 + 16 + SIGNATURE_BYTES or raw[:len(MAGIC)] != MAGIC:
        raise DeployError("invalid encrypted application envelope framing")
    header_length = struct.unpack_from(">I", raw, len(MAGIC))[0]
    if not 2 <= header_length <= MAX_HEADER_BYTES or framing + header_length + 16 + SIGNATURE_BYTES > len(raw):
        raise DeployError("invalid or excessive envelope header length")
    # Only fixed framing and global caps are examined before this trust gate.
    try:
        load_signing_public(trust_key).verify(raw[-SIGNATURE_BYTES:], SIGN_CONTEXT + raw[:-SIGNATURE_BYTES])
    except InvalidSignature as exc:
        raise DeployError("bundle signature verification failed") from exc
    offset = framing + header_length
    header = exact_keys(parse_json(raw[framing:offset], canonical=True), {
        "schema", "algorithm", "manifest", "recipient_sha256", "nonce", "wrapped_key", "ciphertext_bytes",
    }, "envelope header")
    if type(header["schema"]) is not int or header["schema"] != 1 or header["algorithm"] != ALGORITHM:
        raise DeployError("unsupported authenticated envelope format")
    if type(header["ciphertext_bytes"]) is not int or header["ciphertext_bytes"] != len(raw) - offset - SIGNATURE_BYTES:
        raise DeployError("authenticated ciphertext length does not match the envelope")
    sha256_hex(header["recipient_sha256"], "recipient SHA256")
    decode_base64(header["nonce"], 12)
    decode_base64(header["wrapped_key"], RSA_BITS // 8)
    manifest = Manifest.from_dict(header["manifest"])
    return raw, header, manifest, offset


def _preflight_zip(payload: bytes) -> int:
    """Bound the central directory BEFORE ZipFile allocates its ZipInfo list."""
    if len(payload) < _END.size:
        raise DeployError("truncated stored ZIP")
    signature, disk, cd_disk, disk_count, count, size, offset, comment = _END.unpack_from(payload, len(payload) - _END.size)
    if (signature != b"PK\x05\x06" or disk or cd_disk or comment or disk_count != count
            or not 1 <= count <= MAX_FILES or size > MAX_FILES * (_CENTRAL.size + MAX_PATH_BYTES)
            or offset + size != len(payload) - _END.size):
        raise DeployError("ZIP bounds, file count, comment, or multi-disk/ZIP64 structure is invalid")
    cursor = offset
    for _ in range(count):
        if cursor + _CENTRAL.size > offset + size:
            raise DeployError("truncated ZIP central directory")
        values = _CENTRAL.unpack_from(payload, cursor)
        (magic, made, needed, flags, method, _, _, _, compressed, unpacked, name_size,
         extra_size, comment_size, volume, _, _, local_offset) = values
        end = cursor + _CENTRAL.size + name_size
        if (magic != b"PK\x01\x02" or made >> 8 != 3 or needed > 20 or flags or method != zipfile.ZIP_STORED
                or extra_size or comment_size or volume or not 1 <= name_size <= MAX_PATH_BYTES
                or compressed != unpacked or unpacked > MAX_FILE_BYTES or end > offset + size
                or local_offset + _LOCAL.size > offset):
            raise DeployError("only bounded regular ZIP_STORED entries without extra fields are allowed")
        try:
            relative_path(payload[cursor + _CENTRAL.size:end].decode("ascii"))
        except UnicodeError as exc:
            raise DeployError("non-ASCII ZIP path") from exc
        cursor = end
    if cursor != offset + size:
        raise DeployError("ZIP file count disagrees with its central directory")
    return offset


def _unpack_payload(payload: bytes, manifest: Manifest) -> tuple[bytes, ...]:
    central_offset = _preflight_zip(payload)
    result: list[bytes] = []
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r", allowZip64=False) as archive:
            entries = archive.infolist()
            if len(entries) != len(manifest.files):
                raise DeployError("ZIP contains missing or extra files")
            cursor = 0
            for entry, record in zip(entries, manifest.files):
                expected_mode = stat.S_IFREG | (0o500 if record.executable else 0o400)
                if (entry.filename != record.path or entry.is_dir() or entry.create_system != 3
                        or entry.external_attr != expected_mode << 16 or entry.extra or entry.comment
                        or entry.flag_bits or entry.compress_type != zipfile.ZIP_STORED
                        or entry.file_size != record.size or entry.compress_size != record.size
                        or entry.header_offset != cursor):
                    raise DeployError("ZIP file names, types, permissions, sizes, or ordering disagree with the manifest")
                local = _LOCAL.unpack_from(payload, cursor)
                magic, needed, flags, method, _, _, crc, compressed, unpacked, name_size, extra_size = local
                name_end = cursor + _LOCAL.size + name_size
                if (magic != b"PK\x03\x04" or needed != entry.extract_version or flags or method != zipfile.ZIP_STORED
                        or extra_size or name_size != len(record.path) or crc != entry.CRC
                        or compressed != record.size or unpacked != record.size
                        or name_end + record.size > central_offset
                        or payload[cursor + _LOCAL.size:name_end] != record.path.encode("ascii")):
                    raise DeployError("ZIP local header, overlapping data, or path is invalid")
                with archive.open(entry, "r") as member:
                    data = member.read(record.size + 1)
                if len(data) != record.size or not hmac.compare_digest(digest(data), record.sha256):
                    raise DeployError("payload file hash or length mismatch")
                _payload_content(record.path, data, manifest.target)
                result.append(data)
                cursor = name_end + record.size
            if cursor != central_offset:
                raise DeployError("ZIP contains unlisted data or extra local files")
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, struct.error) as exc:
        raise DeployError("invalid restricted stored ZIP") from exc
    return tuple(result)


def verify_bundle(bundle: Pathish, *, identity_dir: Pathish, trust_key: Pathish,
                  node_id: str | None = None, application_id: str | None = None) -> VerifiedBundle:
    """Authenticate, check this host/device, decrypt, and validate all files in memory.

    No extraction, execution, dependency installation, or anti-rollback state change.
    """
    raw, header, manifest, offset = _read_signed_envelope(bundle, trust_key)
    actual_node = device_node_id(identity_dir)
    if node_id is not None and identifier(node_id) != actual_node:
        raise DeployError("requested node_id does not match the local bootstrap identity")
    if manifest.node_id != actual_node:
        raise DeployError("bundle node_id does not match this device")
    if application_id is not None and manifest.application_id != identifier(application_id, "application_id"):
        raise DeployError("bundle application_id does not match the selected application")
    require_target(manifest.target)
    key = _load_private(Path(identity_dir) / "encryption-key.pem", rsa.RSAPrivateKey)
    if not hmac.compare_digest(header["recipient_sha256"], _key_fingerprint(key.public_key())):
        raise DeployError("bundle encryption recipient does not match this device")
    try:
        aes_key = key.decrypt(decode_base64(header["wrapped_key"], RSA_BITS // 8), _oaep())
        if len(aes_key) != 32:
            raise DeployError("invalid wrapped AES key length")
        payload = AESGCM(aes_key).decrypt(decode_base64(header["nonce"], 12),
                                           raw[offset:-SIGNATURE_BYTES], AAD_CONTEXT + raw[:offset])
    except (ValueError, InvalidTag) as exc:
        raise DeployError("bundle decryption/authentication failed") from exc
    if len(payload) > MAX_BUNDLE_BYTES:
        raise DeployError("decrypted archive exceeds the size limit")
    contents = _unpack_payload(payload, manifest)
    return VerifiedBundle(manifest, contents, digest(raw))


def export_usb(bundle: Pathish, usb_dir: Pathish, *, trust_key: Pathish,
               filename: str | None = None) -> Path:
    """Copy a signature-verified encrypted envelope to a chosen existing directory.

    This is an O_EXCL file copy, not a USB uploader, mounter, flasher or ejector.
    The recipient/host need not be present on the exporting operator's machine.
    """
    directory = check_directory(usb_dir, owned=False)
    name = relative_path(filename if filename is not None else Path(bundle).name)
    if "/" in name or not name.endswith(".swarmbundle"):
        raise DeployError("USB filename must be a single safe .swarmbundle name")
    raw, _, _, _ = _read_signed_envelope(bundle, trust_key)
    return write_new(directory / name, raw)