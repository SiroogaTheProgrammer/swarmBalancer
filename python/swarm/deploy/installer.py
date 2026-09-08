"""Local atomic installs and explicit launching; no services, shells or hardware I/O."""

from __future__ import annotations

import hmac
import os
import re
import stat
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Sequence

from ._common import (
    MAX_FILE_BYTES, MAX_FILES, MAX_HEADER_BYTES, Deployment, DeployError, Manifest, Pathish,
    _file_info, canonical_json, check_directory, device_node_id, digest, exact_keys,
    fsync_directory, identifier, is_link, no_links, parse_json, positive_version,
    private_directory, read_regular, require_target, sha256_hex, windows_acl_warning, write_new,
)

_RELEASE = re.compile(r"releases/[0-9a-f]{32}\Z", re.ASCII)
_MANIFEST_NAME = ".swarm-manifest.json"


@contextmanager
def _install_lock(root: Path) -> Generator[None, None, None]:
    """A persistent lock inode with an OS lock, automatically released on exit/crash."""
    path = no_links(root / ".install.lock")
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        _file_info(path.lstat(), private=True, owned=True)
        fd = os.open(path, flags)
    locked = False
    try:
        info = os.fstat(fd)
        _file_info(info, private=True, owned=True)
        if info.st_size > 1:
            raise DeployError("invalid install lock file")
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DeployError("another installation holds this root's lock") from exc
        locked = True
        if info.st_size == 0:
            os.write(fd, b"0")
            os.fsync(fd)
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _release_contents(release: Path, manifest: Manifest) -> None:
    expected = {record.path: record for record in manifest.files}
    directories = {"/".join(record.path.split("/")[:i])
                   for record in manifest.files for i in range(1, len(record.path.split("/")))}
    seen: set[str] = set()
    pending = [release]
    count = 0
    while pending:
        folder = pending.pop()
        check_directory(folder, private=True)
        if os.name == "posix" and stat.S_IMODE(folder.lstat().st_mode) != 0o500:
            raise DeployError("installed release directories must be owner-only read/execute")
        with os.scandir(folder) as entries:
            for entry in entries:
                count += 1
                if count > MAX_FILES * 17 + 1:
                    raise DeployError("installed release contains too many entries")
                path = Path(entry.path)
                name = path.relative_to(release).as_posix()
                # DirEntry cached metadata on Windows omits real link counts.
                info = path.lstat()
                if is_link(info):
                    raise DeployError("installed release contains a link or reparse point")
                if stat.S_ISDIR(info.st_mode):
                    if name not in directories:
                        raise DeployError("installed release contains an unlisted directory")
                    pending.append(path)
                    continue
                if name == _MANIFEST_NAME:
                    _file_info(info, private=True, owned=True)
                    if os.name == "posix" and stat.S_IMODE(info.st_mode) != 0o400:
                        raise DeployError("installed manifest must be owner-only read-only")
                    continue
                if name not in expected or name in seen:
                    raise DeployError("installed release contains an unlisted or duplicate file")
                record = expected[name]
                data = read_regular(path, MAX_FILE_BYTES, private=True)
                if len(data) != record.size or not hmac.compare_digest(digest(data), record.sha256):
                    raise DeployError("installed release file hash or length mismatch")
                if os.name == "posix" and stat.S_IMODE(info.st_mode) != (0o500 if record.executable else 0o400):
                    raise DeployError("installed file permissions disagree with the manifest")
                seen.add(name)
    if seen != set(expected):
        raise DeployError("installed release has missing files")


def status(root: Pathish, *, node_id: str | None = None,
           application_id: str | None = None) -> Deployment | None:
    """Read one atomic current pointer and check its complete immutable release.

    Does not create directories, import the application, or contact hardware/peers.
    """
    if node_id is not None:
        identifier(node_id)
    if application_id is not None:
        identifier(application_id, "application_id")
    root = no_links(root)
    if not root.exists():
        return None
    check_directory(root, private=True)
    current = no_links(root / "current.json")
    if not current.exists():
        return None
    state = exact_keys(parse_json(read_regular(current, 4096, private=True), canonical=True), {
        "schema", "application_id", "version", "node_id", "target", "entrypoint", "release",
        "manifest_sha256", "bundle_sha256",
    }, "current state")
    if (type(state["schema"]) is not int or state["schema"] != 1
            or not isinstance(state["release"], str) or not _RELEASE.fullmatch(state["release"])):
        raise DeployError("invalid current state schema or release pointer")
    positive_version(state["version"])
    sha256_hex(state["manifest_sha256"], "manifest SHA256")
    sha256_hex(state["bundle_sha256"], "bundle SHA256")
    check_directory(root / "releases", private=True)
    release = check_directory(root / state["release"], private=True)
    manifest_bytes = read_regular(release / _MANIFEST_NAME, MAX_HEADER_BYTES, private=True)
    if not hmac.compare_digest(digest(manifest_bytes), state["manifest_sha256"]):
        raise DeployError("installed manifest hash mismatch")
    manifest = Manifest.from_dict(parse_json(manifest_bytes, canonical=True))
    for field in ("application_id", "version", "node_id", "target", "entrypoint"):
        if state[field] != getattr(manifest, field):
            raise DeployError("current state disagrees with the release manifest")
    if node_id is not None and manifest.node_id != node_id:
        raise DeployError("installed node_id does not match the selected device")
    if application_id is not None and manifest.application_id != application_id:
        raise DeployError("installed application_id does not match the selected application")
    _release_contents(release, manifest)
    return Deployment(manifest, release, state["bundle_sha256"])


def _check_upgrade(previous: Deployment | None, manifest: Manifest) -> None:
    if previous is not None:
        if previous.manifest.node_id != manifest.node_id:
            raise DeployError("an install root cannot change device identity")
        # One high-water mark per root, including switches to another application.
        if manifest.version <= previous.manifest.version:
            raise DeployError("replay/downgrade refused: version must increase beyond the current version")


def _separate_credentials(root: Path, identity_dir: Pathish, trust_key: Pathish | None = None) -> None:
    for selected in (identity_dir, trust_key):
        if selected is None:
            continue
        path = no_links(selected)
        if path == root or root in path.parents:
            raise DeployError("bootstrap identity and trust key must be outside the installation root")


def _freeze_directory(directory: Path) -> None:
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                _freeze_directory(Path(entry.path))
    os.chmod(directory, 0o500)
    fsync_directory(directory)


def _remove_stage(directory: Path) -> None:
    """Remove only our uncommitted random staging tree, including read-only files."""
    os.chmod(directory, 0o700)
    with os.scandir(directory) as entries:
        for entry in entries:
            path = Path(entry.path)
            info = entry.stat(follow_symlinks=False)
            if not is_link(info) and stat.S_ISDIR(info.st_mode):
                _remove_stage(path)
            else:
                if not is_link(info):
                    os.chmod(path, 0o600)
                path.unlink()
    directory.rmdir()


def install(bundle: Pathish, *, identity_dir: Pathish, trust_key: Pathish, root: Pathish,
            node_id: str | None = None, application_id: str | None = None,
            dry_run: bool = False) -> Deployment:
    """Verify then prepare an immutable release and atomically commit current.json.

    dry_run also checks anti-rollback state but writes nothing and reserves nothing.
    Neither mode launches the application or installs dependencies.
    """
    from .bundle import verify_bundle

    root = no_links(root)
    _separate_credentials(root, identity_dir, trust_key)
    verified = verify_bundle(bundle, identity_dir=identity_dir, trust_key=trust_key,
                             node_id=node_id, application_id=application_id)
    if dry_run:
        _check_upgrade(status(root), verified.manifest)
        return Deployment(verified.manifest, None, verified.bundle_sha256, dry_run=True)
    windows_acl_warning()
    root = private_directory(root)
    with _install_lock(root):
        _check_upgrade(status(root), verified.manifest)
        releases = private_directory(root / "releases")
        token = uuid.uuid4().hex
        release = releases / token
        stage: Path | None = root / (".stage-" + token)
        if release.exists():
            raise DeployError("random release name collision; retry with a fresh name")
        os.mkdir(stage, 0o700)
        temporary: Path | None = None
        try:
            for record, data in zip(verified.manifest.files, verified.contents):
                path = stage / record.path
                private_directory(path.parent)
                # Create with final mode before fsync, so data and mode commit together.
                write_new(path, data, mode=0o500 if record.executable else 0o400)
            manifest_bytes = canonical_json(verified.manifest.as_dict())
            write_new(stage / _MANIFEST_NAME, manifest_bytes, mode=0o400)
            _freeze_directory(stage)
            os.rename(stage, release)
            stage = None
            fsync_directory(releases)
            # The version high-water mark and release selection live in ONE file.
            state = {"schema": 1, "application_id": verified.manifest.application_id,
                     "version": verified.manifest.version, "node_id": verified.manifest.node_id,
                     "target": verified.manifest.target, "entrypoint": verified.manifest.entrypoint,
                     "release": "releases/" + token, "manifest_sha256": digest(manifest_bytes),
                     "bundle_sha256": verified.bundle_sha256}
            temporary = write_new(root / (".current-" + uuid.uuid4().hex + ".tmp"), canonical_json(state))
            os.replace(temporary, root / "current.json")
            temporary = None
            try:
                fsync_directory(root)
            except OSError as exc:
                raise DeployError("current pointer committed, but directory sync failed; inspect status before retrying") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if stage is not None:
                _remove_stage(stage)
        return Deployment(verified.manifest, release, verified.bundle_sha256)


def launch(root: Pathish, *, node_id: str | None = None, application_id: str | None = None,
           identity_dir: Pathish | None = None, args: Sequence[str] = (),
           trusted_python_paths: Sequence[Pathish] = (), inherit_pythonpath: bool = False) -> int:
    """Explicitly execute the checked installed .py entrypoint with this interpreter.

    No shell. Child cwd is its immutable release. Only release/python and explicitly
    trusted extra paths enter PYTHONPATH; user-site imports and .pyc writes are off.
    The caller supervises the application; it is approved code, not a sandbox.
    """
    selected = status(root, node_id=node_id, application_id=application_id)
    if selected is None or selected.release is None:
        raise DeployError("no installed application")
    require_target(selected.manifest.target)
    if isinstance(args, (str, bytes)) or len(args) > 256 or any(
            not isinstance(arg, str) or "\x00" in arg or len(arg) > 8192 for arg in args):
        raise DeployError("application arguments must be a bounded sequence of strings")
    if isinstance(trusted_python_paths, (str, bytes)):
        raise DeployError("trusted Python paths must be a sequence")
    env = os.environ.copy()
    extra_paths = list(trusted_python_paths)
    if inherit_pythonpath and env.get("PYTHONPATH"):
        extra_paths.extend(env["PYTHONPATH"].split(os.pathsep))
    paths = [str(selected.release / "python")]
    for path in extra_paths:
        if not os.fspath(path) or not Path(path).is_absolute():
            raise DeployError("trusted Python search paths must be nonempty absolute directories")
        paths.append(str(check_directory(path)))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["SWARM_NODE_ID"] = selected.manifest.node_id
    env["SWARM_APPLICATION_ID"] = selected.manifest.application_id
    env["SWARM_RELEASE_DIR"] = str(selected.release)
    env.pop("SWARM_IDENTITY_DIR", None)
    if identity_dir is not None:
        _separate_credentials(no_links(root), identity_dir)
        identity = check_directory(identity_dir, private=True)
        if device_node_id(identity) != selected.manifest.node_id:
            raise DeployError("launch identity does not match the installed node_id")
        env["SWARM_IDENTITY_DIR"] = str(identity)
    command = [sys.executable, "-s", "-B", str(selected.release / selected.manifest.entrypoint), *args]
    return subprocess.run(command, cwd=selected.release, env=env, shell=False, check=False).returncode