"""Strict, bounded data and filesystem operations shared by deployment commands."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import struct
import sys
import sysconfig
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Pathish = str | os.PathLike[str]
MIB = 1024 * 1024
MAX_BUNDLE_BYTES = 64 * MIB
MAX_UNPACKED_BYTES = 128 * MIB
MAX_FILE_BYTES = 32 * MIB
MAX_FILES = 512
MAX_HEADER_BYTES = 256 * 1024
MAX_PATH_BYTES = 240
MAX_PUBLIC_BYTES = 32 * 1024
MAX_VERSION = (1 << 63) - 1
TARGETS = frozenset({
    "linux-aarch64", "linux-armv7l", "linux-armv6l", "linux-x86_64", "linux-i686",
    "windows-aarch64", "windows-x86_64", "windows-i686",
    "darwin-aarch64", "darwin-x86_64", "python-any",
})
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z", re.ASCII)
_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}\Z", re.ASCII)
_HEX = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}
_EXCLUDED = {
    "__pycache__", "__pypackages__", "node_modules", "venv", "env", "site-packages",
    "credentials", "identity", "authority", "secrets", "id_rsa", "id_ecdsa",
    "id_ed25519", "id_dsa", "credentials.json", "secrets.json",
}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".der", ".p8", ".ppk",
                    ".kdbx", ".jks", ".keystore", ".pyc", ".pyo"}


class DeployError(ValueError):
    """A deployment input, trust check, or local security precondition failed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def parse_json(data: bytes, *, canonical: bool = False) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DeployError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               DeployError("non-finite JSON value")))
        if canonical and canonical_json(value) != data:
            raise DeployError("JSON is not in canonical form")
        return value
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise DeployError("invalid or non-canonical JSON") from exc


def exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DeployError(f"invalid {label} fields")
    return value


def identifier(value: Any, label: str = "node_id") -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise DeployError(f"{label} must match [a-z0-9][a-z0-9-]{{0,62}}")
    return value


def positive_version(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_VERSION:
        raise DeployError("version must be a positive 63-bit integer")
    return value


def sha256_hex(value: Any, label: str = "SHA256 fingerprint") -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise DeployError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def target_name(value: Any) -> str:
    if not isinstance(value, str) or value not in TARGETS:
        raise DeployError("unsupported target; use the target command on the destination")
    return value


def current_target() -> str:
    """OS and *interpreter* architecture, never the CPU hidden beneath emulation."""
    if sys.platform == "win32":
        machine = {"win-amd64": "x86_64", "win-arm64": "aarch64",
                   "win32": "i686"}.get(sysconfig.get_platform(), "unknown")
        return "windows-" + machine
    machine = platform.machine().lower()
    machine = {"arm64": "aarch64", "amd64": "x86_64", "i386": "i686",
               "i486": "i686", "i586": "i686", "armv8l": "armv7l"}.get(machine, machine)
    # A 32-bit userspace on a 64-bit Pi kernel cannot load an aarch64 shared library.
    if struct.calcsize("P") == 4:
        machine = {"aarch64": "armv7l", "x86_64": "i686"}.get(machine, machine)
    return ("linux" if sys.platform.startswith("linux") else sys.platform) + "-" + machine


def require_target(target: str) -> None:
    if target != "python-any" and target != current_target():
        raise DeployError(f"target mismatch: bundle is {target}, interpreter is {current_target()}")


def relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_PATH_BYTES:
        raise DeployError("invalid or overlong payload path")
    parts = value.split("/")
    if len(parts) > 16:
        raise DeployError("payload path is too deep")
    for part in parts:
        lower = part.lower()
        if (not _COMPONENT.fullmatch(part) or part.endswith(".")
                or lower.split(".", 1)[0] in _RESERVED
                or lower in _EXCLUDED or Path(lower).suffix in _SECRET_SUFFIXES):
            raise DeployError("unsafe, reserved, cached, or credential-like payload path")
    return value


@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    sha256: str
    executable: bool

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256,
                "executable": self.executable}


def validate_records(records: tuple[FileRecord, ...]) -> None:
    if not 1 <= len(records) <= MAX_FILES:
        raise DeployError("payload file count is outside the allowed bounds")
    files: set[str] = set()
    directories: set[str] = set()
    spellings: dict[str, str] = {}
    total = 0
    for record in records:
        name = relative_path(record.path)
        folded = name.lower()
        if folded in files or folded in directories:
            raise DeployError("duplicate or conflicting case-insensitive payload paths")
        parts = name.split("/")
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            key = prefix.lower()
            if key in spellings and spellings[key] != prefix:
                raise DeployError("inconsistent case in payload path components")
            spellings[key] = prefix
            if length < len(parts):
                if key in files:
                    raise DeployError("payload file is also a parent directory")
                directories.add(key)
        files.add(folded)
        if type(record.size) is not int or not 0 <= record.size <= MAX_FILE_BYTES:
            raise DeployError("payload file size is outside the allowed bounds")
        if type(record.executable) is not bool:
            raise DeployError("executable must be a boolean")
        sha256_hex(record.sha256, "file SHA256")
        total += record.size
        if total > MAX_UNPACKED_BYTES:
            raise DeployError("total uncompressed payload exceeds the limit")
    if [r.path for r in records] != sorted(r.path for r in records):
        raise DeployError("manifest files must be sorted by path")


@dataclass(frozen=True)
class Manifest:
    application_id: str
    version: int
    node_id: str
    target: str
    entrypoint: str
    files: tuple[FileRecord, ...]

    @property
    def unpacked_bytes(self) -> int:
        return sum(item.size for item in self.files)

    def as_dict(self) -> dict[str, Any]:
        return {"application_id": self.application_id, "version": self.version,
                "node_id": self.node_id, "target": self.target, "entrypoint": self.entrypoint,
                "unpacked_bytes": self.unpacked_bytes,
                "files": [item.as_dict() for item in self.files]}

    @classmethod
    def from_dict(cls, value: Any) -> Manifest:
        data = exact_keys(value, {"application_id", "version", "node_id", "target",
                                  "entrypoint", "unpacked_bytes", "files"}, "manifest")
        if not isinstance(data["files"], list) or not 1 <= len(data["files"]) <= MAX_FILES:
            raise DeployError("payload file count is outside the allowed bounds")
        records = []
        for item in data["files"]:
            item = exact_keys(item, {"path", "size", "sha256", "executable"}, "file record")
            records.append(FileRecord(**item))
        files = tuple(records)
        validate_records(files)
        entrypoint = relative_path(data["entrypoint"])
        if not entrypoint.endswith(".py") or entrypoint not in {r.path for r in files}:
            raise DeployError("entrypoint must name a listed relative .py file")
        manifest = cls(identifier(data["application_id"], "application_id"),
                       positive_version(data["version"]), identifier(data["node_id"]),
                       target_name(data["target"]), entrypoint, files)
        if type(data["unpacked_bytes"]) is not int or data["unpacked_bytes"] != manifest.unpacked_bytes:
            raise DeployError("manifest uncompressed size does not match its files")
        return manifest


@dataclass(frozen=True)
class VerifiedBundle:
    manifest: Manifest
    contents: tuple[bytes, ...]
    bundle_sha256: str


@dataclass(frozen=True)
class Deployment:
    manifest: Manifest
    release: Path | None
    bundle_sha256: str
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"application_id": self.manifest.application_id, "version": self.manifest.version,
                "node_id": self.manifest.node_id, "target": self.manifest.target,
                "entrypoint": self.manifest.entrypoint,
                "release": str(self.release) if self.release else None,
                "bundle_sha256": self.bundle_sha256, "dry_run": self.dry_run}


def absolute(path: Pathish) -> Path:
    # Do not resolve(): it would erase the evidence of a symlink/junction.
    return Path(os.path.abspath(os.fspath(path)))


def is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def no_links(path: Pathish) -> Path:
    result = absolute(path)
    for part in (*reversed(result.parents), result):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if is_link(info):
            raise DeployError("symlinks and Windows reparse points are not allowed")
    return result


def _permissions(info: os.stat_result, *, private: bool, owned: bool) -> None:
    if os.name != "posix":
        return
    if owned and info.st_uid != os.geteuid():
        raise DeployError("deployment files must be owned by the current operator")
    if (private and info.st_mode & 0o077) or (owned and info.st_mode & 0o022):
        raise DeployError("unsafe permissions: private paths need owner-only access")
    if info.st_mode & (stat.S_ISUID | stat.S_ISGID):
        raise DeployError("setuid/setgid files and directories are not allowed")


def check_directory(path: Pathish, *, private: bool = False, owned: bool = True) -> Path:
    path = no_links(path)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise DeployError("expected a regular directory")
    _permissions(info, private=private, owned=owned)
    return path


def windows_acl_warning() -> None:
    if os.name == "nt":
        warnings.warn("Windows mode bits do not secure ACLs. Restrict the authority, identity, "
                      "trust and install directories to the operator using Windows ACLs; "
                      "use protected storage. No ACLs are changed automatically.",
                      UserWarning, stacklevel=2)


def private_directory(path: Pathish, *, new: bool = False) -> Path:
    path = no_links(path)
    missing = []
    parent = path
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    check_directory(parent, owned=False)
    # A writable, non-sticky ancestor lets another user replace our private child.
    if os.name == "posix":
        for ancestor in (parent, *parent.parents):
            info = ancestor.lstat()
            if info.st_mode & 0o022 and not (info.st_mode & stat.S_ISVTX):
                raise DeployError("private storage has a writable non-sticky ancestor")
    if new and not missing:
        raise DeployError("destination already exists; identities and outputs are never overwritten")
    for directory in reversed(missing):
        os.mkdir(directory, 0o700)
    return check_directory(path, private=True)


def _file_info(info: os.stat_result, *, private: bool, owned: bool) -> None:
    if is_link(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise DeployError("only regular, non-linked files are allowed")
    _permissions(info, private=private, owned=owned)


def read_regular(path: Pathish, limit: int, *, private: bool = False,
                 owned: bool = True) -> bytes:
    path = no_links(path)
    before = path.lstat()
    _file_info(before, private=private, owned=owned)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        _file_info(info, private=private, owned=owned)
        if (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino):
            raise DeployError("file changed while being opened")
        if not 0 <= info.st_size <= limit:
            raise DeployError("file exceeds its allowed size")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(info.st_size + 1)
        after = os.fstat(fd)
        if (len(data) != info.st_size or len(data) > limit
                or (after.st_size, after.st_mtime_ns) != (info.st_size, info.st_mtime_ns)):
            raise DeployError("file changed while being read or exceeds its allowed size")
        return data
    finally:
        os.close(fd)


def fsync_directory(path: Pathish) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_new(path: Pathish, data: bytes, *, mode: int = 0o600) -> Path:
    path = no_links(path)
    check_directory(path.parent, owned=False)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
    fsync_directory(path.parent)
    return path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def device_node_id(identity_dir: Pathish) -> str:
    directory = check_directory(identity_dir, private=True)
    data = exact_keys(parse_json(read_regular(directory / "identity.json", 1024, private=True),
                                 canonical=True), {"schema", "node_id"}, "device identity")
    if type(data["schema"]) is not int or data["schema"] != 1:
        raise DeployError("unsupported identity schema")
    return identifier(data["node_id"])