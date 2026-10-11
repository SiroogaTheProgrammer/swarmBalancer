"""Installing packages into the library (``brains/``) from uploads: a ``.zip``, a bare ``.swm`` or one ``.py``.

Uploads are untrusted input until validated. Archive member names are checked before anything is written
(no absolute paths, ``..``, drive letters, links or reserved device names; bounded count, size and
compression ratio), files land in a scratch folder inside the library and are renamed into place under
the manifest's name only. Installing never runs package code - :func:`swarm.sdk.validate` does that.
"""

from __future__ import annotations

import ast
import io
import json
import re
import shutil
import uuid
import zipfile
from pathlib import Path

from .packages import MANIFEST, NAME_RE, REPO_ROOT, SCHEMA, Package, PackageError, load_package

LIBRARY = REPO_ROOT / "brains"
MAX_UPLOAD_BYTES = 128 << 20
MAX_UNPACKED_BYTES = 256 << 20
MAX_FILES = 400
MAX_RATIO = 200          # a member that inflates more than this (and to > 1 MiB) is treated as a zip bomb
JUNK = ("__pycache__", "__MACOSX", ".git", ".DS_Store", "Thumbs.db")
_BAD_CHARS = re.compile(r'[<>:"|?*\x00-\x1f\\]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def slug(text: str, fallback: str = "package") -> str:
    """A package name from a file name: lower-case identifier, 2-40 characters."""
    s = re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")
    if not s or not s[0].isalpha():
        s = f"{fallback}_{s}" if s else fallback
    return s[:40].rstrip("_") or fallback


def _check_member(name: str) -> list[str]:
    if name.startswith("/") or "\\" in name:
        raise PackageError(f"archive member {name!r}: absolute paths and backslashes are not allowed")
    parts = [p for p in name.split("/") if p != ""] if name.endswith("/") else name.split("/")
    for p in parts:
        if p in ("", ".", "..") or _BAD_CHARS.search(p) or len(p) > 120 or p.endswith((" ", ".")):
            raise PackageError(f"archive member {name!r}: unsafe path component {p!r}")
        if p.split(".")[0].upper() in _RESERVED:
            raise PackageError(f"archive member {name!r}: {p!r} is a reserved device name on Windows")
    return parts


def _place(staging: Path, library: Path, replace: bool) -> Package:
    pkg = load_package(staging)
    dest = library / pkg.name
    if dest.exists():
        if not replace:
            raise PackageError(f"a package named {pkg.name!r} is already installed (replace it explicitly)")
        shutil.rmtree(dest)
    staging.rename(dest)
    return load_package(dest)


def _staging(library: Path) -> Path:
    library.mkdir(parents=True, exist_ok=True)
    d = library / f".incoming-{uuid.uuid4().hex[:12]}"
    d.mkdir()
    return d


def install_archive(data: bytes, library: Path = LIBRARY, replace: bool = False) -> Package:
    if len(data) > MAX_UPLOAD_BYTES:
        raise PackageError(f"upload larger than {MAX_UPLOAD_BYTES >> 20} MiB")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise PackageError("not a zip archive") from None
    with zf:
        members: list[tuple[zipfile.ZipInfo, list[str]]] = []
        total = 0
        infos = zf.infolist()
        if len(infos) > MAX_FILES * 2:
            raise PackageError(f"archive has more than {MAX_FILES * 2} entries")
        for info in infos:
            parts = _check_member(info.filename)
            if info.is_dir() or any(p in JUNK for p in parts) or parts[-1].endswith(".pyc"):
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise PackageError(f"archive member {info.filename!r} is a link")
            if info.flag_bits & 0x1:
                raise PackageError("encrypted archives are not supported")
            total += info.file_size
            if total > MAX_UNPACKED_BYTES:
                raise PackageError(f"archive unpacks to more than {MAX_UNPACKED_BYTES >> 20} MiB")
            if info.file_size > (1 << 20) and info.file_size > MAX_RATIO * max(1, info.compress_size):
                raise PackageError(f"archive member {info.filename!r} has a suspicious compression ratio")
            members.append((info, parts))
        if len(members) > MAX_FILES:
            raise PackageError(f"archive has more than {MAX_FILES} files")
        if not any(parts == [MANIFEST] for _, parts in members):
            tops = {parts[0] for _, parts in members}
            if len(tops) == 1 and all(len(parts) > 1 for _, parts in members):
                members = [(info, parts[1:]) for info, parts in members]  # zip of a folder: strip it
        if not any(parts == [MANIFEST] for _, parts in members):
            raise PackageError(f"no {MANIFEST} at the archive root (or inside its single top folder)")
        staging = _staging(library)
        try:
            for info, parts in members:
                target = staging.joinpath(*parts)
                if staging not in target.parents:
                    raise PackageError(f"archive member {info.filename!r} escapes the package")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "xb") as dst:
                    remaining = info.file_size
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise PackageError(f"archive member {info.filename!r} is larger than declared")
                        dst.write(chunk)
            return _place(staging, library, replace)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def install_swm(data: bytes, filename: str, library: Path = LIBRARY, replace: bool = False) -> Package:
    """A bare compiled network becomes a perception package around it."""
    from swarm.brain.formats import MAGIC, read_swm

    if len(data) > MAX_UPLOAD_BYTES:
        raise PackageError(f"upload larger than {MAX_UPLOAD_BYTES >> 20} MiB")
    if not data.startswith(MAGIC):
        raise PackageError(f"{filename}: not a .swm model (magic {MAGIC!r} missing)")
    name = slug(Path(filename).stem, "brain")
    staging = _staging(library)
    try:
        (staging / "model.swm").write_bytes(data)
        try:
            _shape, layers = read_swm(staging / "model.swm")
        except (ValueError, KeyError) as exc:
            raise PackageError(f"{filename}: malformed .swm ({exc})") from None
        outs = [layer for layer in layers if layer["type"] == "dense"]
        n_out = outs[-1]["out"] if outs else 0
        manifest = {"schema": SCHEMA, "name": name, "version": "1.0.0", "type": "perception", "format": "swm",
                    "model": "model.swm", "title": Path(filename).stem,
                    "description": f"Uploaded compiled network {filename}."}
        if n_out == 5:
            manifest["classes"] = ["nothing", "vehicle", "building", "road", "aircraft"]
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return _place(staging, library, replace)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _class_role(node: ast.ClassDef) -> str | None:
    bases = [b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", "") for b in node.bases]
    if any(b in ("Tactic", "SwarmNode") for b in bases):
        return "tactic"
    if any(b == "PerceptionBrain" for b in bases) or any(isinstance(n, ast.FunctionDef) and n.name == "classify"
                                                          for n in node.body):
        return "perception"
    return None


def _literal_attr(node: ast.ClassDef, attr: str) -> str | None:
    for stmt in node.body:
        if isinstance(stmt, ast.Assign) and any(getattr(t, "id", None) == attr for t in stmt.targets):
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                return stmt.value.value
    return None


def install_source(data: bytes, filename: str, library: Path = LIBRARY, replace: bool = False) -> Package:
    """One ``.py`` file with a Tactic subclass or a class with ``classify()`` becomes a package.

    The file is parsed, not executed, to write the manifest.
    """
    if len(data) > (4 << 20):
        raise PackageError("a single-file package must be smaller than 4 MiB")
    try:
        text = data.decode("utf-8-sig")  # editors on Windows often save UTF-8 with a byte-order mark
        tree = ast.parse(text, filename=filename)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise PackageError(f"{filename}: not valid UTF-8 Python ({exc})") from None
    found = [(n, _class_role(n)) for n in tree.body if isinstance(n, ast.ClassDef)]
    found = [(n, role) for n, role in found if role]
    if not found:
        raise PackageError(f"{filename}: no Tactic subclass and no class with a classify() method")
    tactics = [n for n, role in found if role == "tactic"]
    node, role = (tactics[0], "tactic") if tactics else found[0]
    module = slug(Path(filename).stem, "module")
    name = slug(_literal_attr(node, "strategy") or module) if role == "tactic" else module
    if not NAME_RE.match(name):
        raise PackageError(f"cannot derive a package name from {filename!r}")
    doc = ast.get_docstring(tree) or ""
    manifest = {"schema": SCHEMA, "name": name, "version": "1.0.0", "type": role, "entry": f"{module}.py:{node.name}",
                "title": node.name, "description": doc.strip().splitlines()[0] if doc.strip() else f"Uploaded {filename}."}
    if role == "tactic":
        manifest["strategy"] = name
    else:
        manifest["format"] = "python"
    staging = _staging(library)
    try:
        (staging / f"{module}.py").write_bytes(data)
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return _place(staging, library, replace)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def install_upload(data: bytes, filename: str, library: Path = LIBRARY, replace: bool = False) -> Package:
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        return install_archive(data, library, replace)
    if suffix == ".swm":
        return install_swm(data, filename, library, replace)
    if suffix == ".py":
        return install_source(data, filename, library, replace)
    raise PackageError(f"{filename}: upload a package .zip, a compiled .swm network, or a single .py file")


def install_folder(src: Path, library: Path = LIBRARY, replace: bool = False) -> Package:
    """Copies a package folder into the library (no links followed)."""
    pkg = load_package(src)
    staging = _staging(library)
    try:
        shutil.rmtree(staging)
        shutil.copytree(pkg.root, staging, symlinks=False,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".DS_Store"))
        return _place(staging, library, replace)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def remove_package(name: str, library: Path = LIBRARY) -> Path:
    """Deletes an installed package - only ones inside the library folder."""
    if not NAME_RE.match(name or ""):
        raise PackageError(f"invalid package name {name!r}")
    target = (library / name).resolve()
    if target.parent != library.resolve() or not (target / MANIFEST).is_file():
        raise PackageError(f"{name!r} is not an installed package in {library}")
    shutil.rmtree(target)
    return target


def pack(path: str | Path, out: str | Path | None = None) -> Path:
    """Zips a package folder (manifest at the archive root) for upload."""
    pkg = load_package(path)
    out = Path(out) if out else Path.cwd() / f"{pkg.name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, _size in pkg.files():
            parts = rel.split("/")
            if any(p in JUNK for p in parts) or rel.endswith(".pyc"):
                continue
            zf.write(pkg.root / rel, rel)
    return out
