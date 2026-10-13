"""Brain and tactic packages: a folder (or a .zip of one) with ``manifest.json`` at its root.

::

    {
      "schema": "swarm.package/1",
      "name": "edge_detector",           lower-case identifier, unique in the library
      "version": "1.0.0",
      "type": "perception",              perception | tactic | policy
      "format": "python",                perception: python | swm;  policy: npz | python
      "role": "defender",                policy only: recon | defender | attacker (an arena test side)
      "entry": "brain.py:EdgeBrain",     python brains, policies and tactics: <file inside the package>:<class>
      "model": "model.swm",              swm brains: the compiled network; npz policies: the mesh weights
      "build": "train.py",               swm brains, optional: script that (re)compiles the model
      "classes": ["nothing", ...],       optional label names (swm files store only the class count)
      "options": {},                     optional constructor options of a python brain or policy
      "title": "...", "description": "...", "author": "..."
    }

Reading a manifest never runs package code; :meth:`Package.load_class` does (packages are trusted local
code, like any script you run - they are not sandboxed). Each package is imported under its own module
namespace, so two packages may both have a ``brain.py``; use relative imports between a package's files.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "manifest.json"
SCHEMA = "swarm.package/1"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
TYPES = ("perception", "tactic", "policy")
FORMATS = ("python", "swm")
POLICY_FORMATS = ("npz", "python")
POLICY_ROLES = ("recon", "defender", "attacker")
# your packages (SWARM_LIBRARY overrides, e.g. for tests), then the shipped examples
LIBRARY_DIRS = (Path(os.environ.get("SWARM_LIBRARY") or REPO_ROOT / "brains"), REPO_ROOT / "examples" / "brains")
MAX_MANIFEST_BYTES = 64 * 1024


class PackageError(ValueError):
    """A package is malformed (bad manifest, missing file, wrong class)."""


def _inside(root: Path, rel: str, what: str) -> Path:
    if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")) or ":" in rel or "\\" in rel:
        raise PackageError(f"{what} must be a relative path with '/' separators inside the package, got {rel!r}")
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise PackageError(f"{what} {rel!r} must stay inside the package")
    return root.joinpath(*parts)


def read_manifest(root: Path) -> dict:
    path = root / MANIFEST
    if not path.is_file():
        raise PackageError(f"{root}: no {MANIFEST} (a package is a folder with {MANIFEST} at its root)")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise PackageError(f"{path}: manifest larger than {MAX_MANIFEST_BYTES} bytes")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"{path}: not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise PackageError(f"{path}: the manifest must be a JSON object")
    return data


def check_manifest(m: dict, root: Path) -> None:
    """Raises :class:`PackageError` describing the first problem; does not import anything."""
    if m.get("schema") != SCHEMA:
        raise PackageError(f"manifest schema must be {SCHEMA!r}, got {m.get('schema')!r}")
    name = m.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise PackageError(f"name {name!r} must match {NAME_RE.pattern} (lower-case identifier, 2-40 chars)")
    if not isinstance(m.get("version", ""), str):
        raise PackageError("version must be a string such as '1.0.0'")
    kind = m.get("type")
    if kind not in TYPES:
        raise PackageError(f"type must be one of {TYPES}, got {kind!r}")
    for key in ("title", "description", "author"):
        if key in m and not isinstance(m[key], str):
            raise PackageError(f"{key} must be a string")
    if kind == "perception":
        fmt = m.get("format")
        if fmt not in FORMATS:
            raise PackageError(f"perception packages need format {FORMATS}, got {fmt!r}")
        if fmt == "swm":
            _inside(root, m.get("model", ""), "model")
            if "build" in m:
                script = m["build"]
                if not isinstance(script, str) or not script.endswith(".py") or \
                        not _inside(root, script, "build script").is_file():
                    raise PackageError(f"build must name a .py script inside the package, got {script!r}")
        classes = m.get("classes")
        if classes is not None and (not isinstance(classes, list) or len(classes) < 2
                                    or not all(isinstance(c, str) and c for c in classes)):
            raise PackageError("classes must be a list of at least two non-empty label strings")
        if "options" in m and not isinstance(m["options"], dict):
            raise PackageError("options must be a JSON object")
    if kind == "policy":
        if m.get("role") not in POLICY_ROLES:
            raise PackageError(f"policy packages need role {POLICY_ROLES}, got {m.get('role')!r}")
        fmt = m.get("format")
        if fmt not in POLICY_FORMATS:
            raise PackageError(f"policy packages need format {POLICY_FORMATS}, got {fmt!r}")
        if fmt == "npz":
            model = m.get("model", "")
            if not isinstance(model, str) or not model.endswith(".npz"):
                raise PackageError(f"model must name the .npz weights inside the package, got {model!r}")
            _inside(root, model, "model")
        if "options" in m and not isinstance(m["options"], dict):
            raise PackageError("options must be a JSON object")
        if "macs" in m and (isinstance(m["macs"], bool) or not isinstance(m["macs"], int) or m["macs"] < 0):
            raise PackageError("macs must be a whole number >= 0 (compute per drone per decision)")
        if "comms" in m and not isinstance(m["comms"], bool):
            raise PackageError("comms must be true or false")
    if kind == "tactic" or m.get("format") == "python":
        entry = m.get("entry")
        if not isinstance(entry, str) or entry.count(":") != 1:
            raise PackageError(f"entry must look like 'file.py:ClassName', got {entry!r}")
        file_part, cls_name = entry.split(":")
        if not file_part.endswith(".py") or not cls_name.isidentifier():
            raise PackageError(f"entry must look like 'file.py:ClassName', got {entry!r}")
        target = _inside(root, file_part, "entry file")
        if not target.is_file():
            raise PackageError(f"entry file {file_part!r} not found in the package")
        if not all(p.isidentifier() for p in file_part[:-3].split("/")):
            raise PackageError(f"entry file {file_part!r}: every path part must be a Python identifier")
    if kind == "tactic":
        strategy = m.get("strategy", name)
        if not isinstance(strategy, str) or not NAME_RE.match(strategy):
            raise PackageError(f"strategy {strategy!r} must match {NAME_RE.pattern}")


def _swm_is_int8(path: Path) -> bool:
    from swarm.brain.formats import read_swm

    return any(layer.get("dtype") == "i8" for layer in read_swm(path)[1])


@dataclass
class Package:
    root: Path
    manifest: dict

    # ------------------------------------------------------------ manifest views
    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "0"))

    @property
    def type(self) -> str:
        return self.manifest["type"]

    @property
    def format(self) -> str | None:
        return self.manifest.get("format") if self.type in ("perception", "policy") else "python"

    @property
    def role(self) -> str | None:
        return self.manifest.get("role") if self.type == "policy" else None

    @property
    def title(self) -> str:
        return self.manifest.get("title") or self.name

    @property
    def entry(self) -> str | None:
        return self.manifest.get("entry")

    @property
    def strategy(self) -> str | None:
        return self.manifest.get("strategy", self.name) if self.type == "tactic" else None

    @property
    def model_path(self) -> Path | None:
        if self.format not in ("swm", "npz"):
            return None
        return _inside(self.root, self.manifest["model"], "model")

    def label(self) -> str:
        return f"{self.name} v{self.version}"

    def files(self) -> list[tuple[str, int]]:
        return sorted((p.relative_to(self.root).as_posix(), p.stat().st_size) for p in self.root.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts)

    # ------------------------------------------------------------ static information (no package code runs)
    def summary(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": self.name, "version": self.version, "type": self.type, "format": self.format,
            "title": self.title, "description": self.manifest.get("description", ""),
            "author": self.manifest.get("author", ""), "entry": self.entry, "strategy": self.strategy,
            "root": str(self.root), "files": [{"path": p, "bytes": n} for p, n in self.files()],
        }
        try:
            info["root_rel"] = self.root.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            info["root_rel"] = None
        if self.format == "swm":
            path = self.model_path
            info["model"] = self.manifest.get("model")
            info["build"] = self.manifest.get("build")
            info["model_built"] = bool(path and path.is_file())
            if info["model_built"]:
                try:
                    info.update(swm_info(path))
                except (OSError, ValueError, KeyError) as exc:
                    info["model_error"] = str(exc)
            names = self.manifest.get("classes")
            if names:
                info["classes"] = names
        if self.type == "policy":
            info.update(self.policy_info())
        return info

    def policy_info(self) -> dict[str, Any]:
        """What the arena brain list shows for a policy package; reads the .npz header, runs no code."""
        info: dict[str, Any] = {
            "id": f"lib:{self.name}", "name": self.name, "role": self.role,
            "kind": "neural" if self.format == "npz" else "python", "title": self.title,
            "summary": self.manifest.get("description", ""), "version": self.version,
            "author": self.manifest.get("author", ""),
            "comms": bool(self.manifest.get("comms", self.format == "npz")),
            "macs": int(self.manifest.get("macs", 0) or 0), "package": str(self.root),
            "removable": self.root.parent.resolve() == LIBRARY_DIRS[0].resolve(),
        }
        if self.format == "npz":
            path = self.model_path
            if path is None or not path.is_file():
                info["error"] = f"weights {self.manifest.get('model')!r} missing"
                return info
            try:
                import numpy as np

                with np.load(path, allow_pickle=False) as z:
                    meta = json.loads(str(z["meta"]))
                n_in, hidden, n_out = int(meta["n_in"]), int(meta["hidden"]), int(meta["n_out"])
                from swarm.arena.brains import OBS, OLD_DEFENDER, OUTPUTS, ROWS

                if self.role == "defender" and (n_in, n_out) == OLD_DEFENDER:   # widened when it loads
                    n_in, n_out = OBS["defender"].FEATURES, OUTPUTS["defender"]
                rows = ROWS.get(self.role, 1)
                info.update({"macs": rows * (n_in * hidden + hidden * n_out), "hidden": hidden,
                             "params": n_in * hidden + hidden + hidden * n_out + n_out,
                             "trained": {k: meta[k] for k in ("generations", "trained_against", "seconds")
                                         if k in meta}})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                info["error"] = f"unreadable weights ({exc})"
        return info

    def make_policy(self):
        """The arena brain of a policy package (runs package code for python policies)."""
        if self.type != "policy":
            raise PackageError(f"{self.name} is a {self.type} package, not an arena policy")
        from swarm.arena.brains import BrainError, NeuralBrain, PythonBrain

        try:
            if self.format == "npz":
                path = self.model_path
                if path is None or not path.is_file():
                    raise PackageError(f"{self.name}: weights {self.manifest.get('model')!r} missing")
                brain = NeuralBrain.load(path, name=self.name)
                if brain.role != self.role:
                    raise PackageError(f"{self.name}: the weights are for a {brain.role}, the manifest says {self.role}")
                brain.title = self.title
                brain.summary = self.manifest.get("description") or brain.summary
            else:
                cls = self.load_class()
                options = dict(self.manifest.get("options") or {})
                brain = PythonBrain(cls(**options), self.role, self.name, self.title,
                                    self.manifest.get("description", ""))
        except BrainError as exc:
            raise PackageError(str(exc)) from None
        brain.lib_id = f"lib:{self.name}"
        return brain

    # ------------------------------------------------------------ code (trusted, runs the package)
    def module_name(self) -> str:
        tag = hashlib.sha1(str(self.root.resolve()).lower().encode("utf-8")).hexdigest()[:8]
        return f"swarm_pkg_{self.name}_{tag}"

    def load_class(self) -> type:
        if not self.entry:
            raise PackageError(f"{self.name}: no python entry (format {self.format!r})")
        file_part, cls_name = self.entry.split(":")
        pkg = self.module_name()
        if pkg not in sys.modules:
            ns = types.ModuleType(pkg)
            ns.__path__ = [str(self.root)]  # a synthetic package: the package folder is its import path
            ns.__package__ = pkg
            sys.modules[pkg] = ns
        importlib.invalidate_caches()
        module = importlib.import_module(f"{pkg}.{file_part[:-3].replace('/', '.')}")
        cls = getattr(module, cls_name, None)
        if not isinstance(cls, type):
            raise PackageError(f"{self.name}: {self.entry} does not define a class {cls_name!r}")
        return cls

    def make_brain(self, seed: int = 0, engine: str | None = None):
        """A fresh simulator brain (perception packages)."""
        return self.resolve_brain(seed, engine)[3]()

    def resolve_brain(self, seed: int = 0, engine: str | None = None) -> tuple[str, Path | None, str, Callable]:
        """Same contract as ``swarm.bench.runner.resolve_brain``: ``(kind, model_path, note, factory)``."""
        if self.type != "perception":
            raise PackageError(f"{self.name} is a {self.type} package, not a perception brain")
        if self.format == "swm":
            from swarm.brain import native
            from swarm.sim.brains import NativeBrain, NumpyBrain

            path = self.model_path
            if path is None or not path.is_file():
                raise PackageError(f"{self.name}: compiled model {self.manifest.get('model')!r} not built yet "
                                   "(run the package's train.py)")
            kind = engine or ("native" if native.available() else "numpy")
            note = f"package {self.label()} (.swm)"
            if kind == "native":
                if not native.available():
                    raise RuntimeError("engine 'native' requested but the C++ engine is not built (python dev.py build)")
                return kind, path, note, lambda: NativeBrain(path)
            if kind != "numpy":
                raise ValueError(f"unknown engine {kind!r} for a .swm package (native | numpy)")
            if not engine:
                note += " - C++ engine not built, numpy brain (RAM fit is estimated, not enforced)"
            int8 = _swm_is_int8(path)
            return kind, path, note, lambda: NumpyBrain(path, int8=int8)
        from .api import PythonBrainAdapter, instantiate

        cls = self.load_class()
        options = dict(self.manifest.get("options") or {})
        macs = getattr(cls, "macs", 0)
        note = f"python brain package {self.label()} - declared {macs} MACs/frame"
        return "package", None, note, lambda: PythonBrainAdapter(instantiate(cls, seed, options), self.name)

    def register(self) -> str:
        """Registers a tactic with the simulator; returns its strategy name."""
        if self.type != "tactic":
            raise PackageError(f"{self.name} is a {self.type} package, not a tactic")
        from swarm.sim.strategies import SwarmNode, register_strategy

        cls = self.load_class()
        if not issubclass(cls, SwarmNode):
            raise PackageError(f"{self.name}: {self.entry} must subclass swarm.sdk.Tactic")
        declared = getattr(cls, "strategy", "") or self.strategy
        if declared != self.strategy:
            raise PackageError(f"{self.name}: class says strategy {declared!r}, manifest says {self.strategy!r}")
        cls.strategy = self.strategy
        register_strategy(self.strategy, cls, source=str(self.root.resolve()))
        return self.strategy


def swm_info(path: Path) -> dict[str, Any]:
    """Shape / cost facts about a compiled model, read with numpy only."""
    import numpy as np

    from swarm.brain.model import Sequential

    model = Sequential.load(path)
    shapes = model._shapes
    split = next((i for i, layer in enumerate(model.layers) if type(layer).__name__ == "Flatten"), len(model.layers))
    frontend = sum(layer.macs(s) for layer, s in zip(model.layers[:split], shapes[:split]))
    int8 = _swm_is_int8(path)
    return {
        "input_shape": list(model.input_shape), "n_classes": int(model.output_shape[0]), "macs": int(model.macs()),
        "params": int(model.num_params()), "int8": int8, "model_bytes": path.stat().st_size,
        "weights_bytes": int(model.weights_bytes(int8)), "frontend_macs": int(frontend),
        "feature_bytes": int(np.prod(shapes[split])), "layers": len(model.layers),
    }


def resolve_path(path: str | Path) -> Path:
    """Relative paths are taken from the repository root (like ``BrainSpec.model``)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_package(path: str | Path) -> Package:
    root = resolve_path(path)
    if root.is_file() and root.name == MANIFEST:
        root = root.parent
    root = root.resolve()
    manifest = read_manifest(root)
    check_manifest(manifest, root)
    return Package(root, manifest)


def use_package(path: str | Path) -> Package:
    """Loads a package for use in scenarios: tactics are registered as strategies, brains are checked.

    Preset files call this before building their ``Scenario`` when they use a plugin tactic::

        use_package("examples/brains/load_aware_stripe")
    """
    pkg = load_package(path)
    if pkg.type == "tactic":
        pkg.register()
    return pkg


def discover_packages(dirs: tuple[Path, ...] | None = None) -> list[tuple[Path, Package | None, str]]:
    """Every ``<dir>/<pkg>/manifest.json``: ``(folder, package or None, error)``. Runs no package code."""
    found = []
    for d in LIBRARY_DIRS if dirs is None else dirs:
        if not d.is_dir():
            continue
        for child in sorted(d.iterdir()):
            if child.is_dir() and (child / MANIFEST).is_file():
                try:
                    found.append((child, load_package(child), ""))
                except PackageError as exc:
                    found.append((child, None, str(exc)))
    return found
