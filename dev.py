#!/usr/bin/env python
"""One command to build and run everything, whatever the shell's PATH looks like.

    python dev.py doctor      what this machine has / lacks, and the exact commands to fix it
    python dev.py build       C++ engine: native tools+tests, plus a DLL for this python.exe if it differs
    python dev.py test        ctest + pytest
    python dev.py train       train the tiny CNN -> models/*.swm (skipped if they exist; --force to retrain)
    python dev.py bench       the standardized scenario battery (reports in out/bench/)
    python dev.py all         doctor, build, test, train, bench

Standard library only, so it runs before numpy/pytest are installed (it will offer
to install them). Tools are located on PATH *or* where winget installs them, because
on Windows-on-ARM the shell that runs this often does not see them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

from swarm import _arch  # noqa: E402  (stdlib-only module)

WINGET = {
    "cmake": ("Kitware.CMake", "Kitware.CMake*/cmake-*/bin"),
    "ninja": ("Ninja-build.Ninja", "Ninja-build.Ninja*"),
    "llvm-mingw": ("MartinStorsjo.LLVM-MinGW.UCRT", "MartinStorsjo.LLVM-MinGW*/llvm-mingw-*/bin"),
}


def winget_cmd(pkg_id: str) -> str:
    return f"winget install --id {pkg_id} --scope user --silent --accept-package-agreements --accept-source-agreements"


# ----------------------------------------------------------------------------- tool discovery
def _winget_dirs(pattern: str) -> list[Path]:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return []
    return sorted(Path(base, "Microsoft", "WinGet", "Packages").glob(pattern), reverse=True) + \
        [Path(base, "Microsoft", "WinGet", "Links")]


def find_tool(exe: str, winget_key: str | None = None) -> Path | None:
    hit = shutil.which(exe)
    if hit:
        return Path(hit)
    if sys.platform == "win32" and winget_key:
        for d in _winget_dirs(WINGET[winget_key][1]):
            for name in (exe, exe + ".exe"):
                if (d / name).is_file():
                    return d / name
    return None


def compiler_for(arch: str) -> Path | None:
    if sys.platform != "win32":
        hit = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        return Path(hit) if hit else None
    triple = _arch.MINGW_TRIPLES.get(arch)
    return find_tool(f"{triple}-clang++", "llvm-mingw") if triple else None


def tool_env(*tools: Path | None) -> dict[str, str]:
    """PATH with the discovered tool directories in front, for child processes (cmake needs ninja + compiler)."""
    env = dict(os.environ)
    key = next((k for k in env if k.upper() == "PATH"), "PATH")  # Windows spells it "Path"
    dirs = [str(t.parent) for t in tools if t]
    env[key] = os.pathsep.join(dict.fromkeys(dirs + env.get(key, "").split(os.pathsep)))
    return env


def run(cmd: list[str | Path], env: dict | None = None, check: bool = True, cwd: Path = ROOT) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.call([str(c) for c in cmd], env=env, cwd=cwd)
    if check and rc != 0:
        sys.exit(f"command failed with exit code {rc}")
    return rc


def have_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------------- doctor
def doctor(quiet: bool = False) -> bool:
    ok = True
    machine, py = _arch.machine_arch(), _arch.python_arch()
    lines = [f"machine   : {sys.platform} {machine}",
             f"python    : {_arch.describe_python()}  [{sys.executable}]"]
    if sys.prefix != sys.base_prefix:
        lines[-1] += "  (venv)"
    cmake, ninja = find_tool("cmake", "cmake"), find_tool("ninja", "ninja")
    lines.append(f"cmake     : {cmake or 'MISSING  -> ' + winget_cmd(WINGET['cmake'][0])}")
    lines.append(f"ninja     : {ninja or 'MISSING  -> ' + winget_cmd(WINGET['ninja'][0])}")
    ok &= bool(cmake and ninja)
    for arch in dict.fromkeys([machine, py]):
        if not arch:
            continue
        cxx = compiler_for(arch)
        tag = "native tools/tests" if arch == machine else "DLL for this python"
        lines.append(f"c++ {arch:<5} : {cxx or 'MISSING  -> ' + winget_cmd(WINGET['llvm-mingw'][0])}  ({tag})")
        ok &= bool(cxx)
    for mod in ("numpy", "pytest"):
        lines.append(f"{mod:<10}: {'ok' if have_module(mod) else 'MISSING  -> ' + pip_hint()}")
    ok &= have_module("numpy")
    engine_ok = False
    if have_module("numpy"):
        from swarm.brain import native

        engine_ok = native.available()
        lines.append("engine    : " + native.diagnosis().replace("\n", "\n            "))
    models = sorted((ROOT / "models").glob("*.swm"))
    lines.append(f"models    : {', '.join(m.name for m in models) if models else 'none yet -> python dev.py train'}")
    if _arch.python_is_emulated():
        lines.append(f"note      : this python is an {py} build running emulated on an {machine} PC; ctypes can only "
                     f"load {py} DLLs, so `build` also cross-compiles one into build-{py}/ (automatic).")
    if not quiet:
        print("\n".join(lines))
        if not ok:
            print("\nsomething is missing - see above")
        elif engine_ok:
            print("\nall good - next: python dev.py test | train | bench")
        else:
            print("\ntoolchain OK - next: python dev.py build")
    return ok


def pip_hint() -> str:
    user = "" if sys.prefix != sys.base_prefix else " --user"
    return f"{Path(sys.executable).name} -m pip install{user} -e .[dev]"


def ensure_python_deps() -> None:
    missing = [m for m in ("numpy", "pytest") if not have_module(m)]
    if not missing:
        return
    print(f"python packages missing: {', '.join(missing)}")
    user = [] if sys.prefix != sys.base_prefix else ["--user"]
    run([sys.executable, "-m", "pip", "install", "--quiet", *user, "-e", ".[dev]"])


# ----------------------------------------------------------------------------- build
def cache_is_ours(bdir: Path, arch: str | None) -> bool:
    """True if ``bdir`` was configured by our toolchain file for ``arch`` and its compiler still exists.

    Anything else (a directory left behind by the CMake Tools extension with an MSVC-targeting clang,
    an old preset, a compiler that has since been uninstalled) would make cmake reuse a broken cache.
    """
    cache = bdir / "CMakeCache.txt"
    if not cache.is_file():
        return True  # nothing to inherit
    text = cache.read_text(errors="replace")
    cxx = next((l.split("=", 1)[1].strip() for l in text.splitlines() if l.startswith("CMAKE_CXX_COMPILER:")), "")
    if not cxx or not Path(cxx).is_file():
        return False
    if sys.platform != "win32":
        return True
    return f"SWARM_RESOLVED_ARCH:INTERNAL={arch}" in text


def build(clean: bool = False) -> None:
    cmake = find_tool("cmake", "cmake")
    ninja = find_tool("ninja", "ninja")
    if not cmake or not ninja:
        doctor()
        sys.exit("cmake/ninja missing - install with the winget commands above, then rerun")
    machine, py = _arch.machine_arch(), _arch.python_arch()
    # native build: tools, tests and (if python matches) the DLL
    targets = [(machine, ROOT / "build", ["-DSWARM_BUILD_TESTS=ON", "-DSWARM_BUILD_TOOLS=ON"])]
    if sys.platform == "win32" and py and py != machine:
        targets.append((py, ROOT / f"build-{py}", ["-DSWARM_BUILD_TESTS=OFF", "-DSWARM_BUILD_TOOLS=OFF",
                                                   "-DSWARM_NATIVE_ARCH=OFF"]))
    for arch, bdir, extra in targets:
        cxx = compiler_for(arch or "")
        if sys.platform == "win32" and not cxx:
            sys.exit(f"no llvm-mingw compiler for {arch}. Install: {winget_cmd(WINGET['llvm-mingw'][0])}")
        env = tool_env(cmake, ninja, cxx)
        if bdir.exists() and (clean or not cache_is_ours(bdir, arch)):
            print(f"removing {bdir.name}/ ({'--clean' if clean else 'stale or foreign CMake cache'})")
            shutil.rmtree(bdir)
        cfg = [cmake, "-S", ROOT, "-B", bdir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", *extra]
        if sys.platform == "win32":
            cfg.append(f"-DSWARM_TARGET_ARCH={arch}")
        print(f"\n== configure + build for {arch} -> {bdir.name}/")
        run(cfg, env)
        run([cmake, "--build", bdir], env)
    if have_module("numpy"):
        from swarm.brain import native

        print("\n" + native.diagnosis())
        if not native.available():
            sys.exit(1)


# ----------------------------------------------------------------------------- test / train / bench
def test(cpp: bool = True, py: bool = True) -> None:
    if cpp:
        ctest = find_tool("ctest", "cmake")
        if ctest and (ROOT / "build" / "CTestTestfile.cmake").is_file():
            run([ctest, "--test-dir", ROOT / "build", "--output-on-failure"])
        else:
            print("ctest: no native build with tests found (python dev.py build) - skipped")
    if py:
        ensure_python_deps()
        run([sys.executable, "-m", "pytest", "-q"])


def train(force: bool = False, epochs: int = 3) -> None:
    ensure_python_deps()
    models = ROOT / "models"
    if not force and (models / "tiny_cnn_int8.swm").is_file() and (models / "tiny_cnn_f32.swm").is_file():
        print("models/tiny_cnn_*.swm already exist (use --force to retrain)")
        return
    run([sys.executable, "-m", "swarm.train.train_tiny_cnn", "--epochs", str(epochs)])


def bench(args: list[str]) -> int:
    ensure_python_deps()
    if not (ROOT / "models" / "tiny_cnn_int8.swm").is_file():
        print("no trained model yet - scenarios will use the oracle brain (python dev.py train to change that)")
    return run([sys.executable, "-m", "swarm.bench", *args], check=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    b = sub.add_parser("build")
    b.add_argument("--clean", action="store_true", help="delete build directories first")
    t = sub.add_parser("test")
    t.add_argument("--no-cpp", action="store_true")
    t.add_argument("--no-python", action="store_true")
    tr = sub.add_parser("train")
    tr.add_argument("--force", action="store_true")
    tr.add_argument("--epochs", type=int, default=3)
    be = sub.add_parser("bench", help="extra arguments are passed to `python -m swarm.bench`")
    be.add_argument("rest", nargs=argparse.REMAINDER)
    a = sub.add_parser("all")
    a.add_argument("--clean", action="store_true")
    args = ap.parse_args(argv)

    os.chdir(ROOT)
    if args.cmd == "doctor":
        return 0 if doctor() else 1
    if args.cmd == "build":
        build(args.clean)
    elif args.cmd == "test":
        test(cpp=not args.no_cpp, py=not args.no_python)
    elif args.cmd == "train":
        train(args.force, args.epochs)
    elif args.cmd == "bench":
        return bench(args.rest)
    elif args.cmd == "all":
        if not doctor():
            return 1
        build(args.clean)
        ensure_python_deps()
        test()
        train()
        return bench([])
    return 0


if __name__ == "__main__":
    sys.exit(main())
