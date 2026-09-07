"""Architecture detection and the dev.py entry point (the things that broke on Windows-on-ARM)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from swarm import _arch
from swarm.brain import native

ROOT = Path(__file__).resolve().parents[1]


def test_arch_vocabulary_is_normalised():
    assert _arch.normalise("AMD64") == "x64" and _arch.normalise("aarch64") == "arm64"
    assert _arch.normalise("x86_64") == "x64" and _arch.normalise("ARM64") == "arm64"
    assert _arch.machine_arch() in {"arm64", "x64", "x86", "arm32"}
    assert _arch.python_arch() in {"arm64", "x64", "x86", "arm32"}


@pytest.mark.skipif(sys.platform != "win32", reason="PE headers are a Windows thing")
def test_dll_arch_reads_pe_header(tmp_path):
    assert _arch.dll_arch(tmp_path / "missing.dll") is None
    (tmp_path / "text.dll").write_text("not a dll")
    assert _arch.dll_arch(tmp_path / "text.dll") is None
    dlls = [p for p in ROOT.glob("build*/bin/swarm_brain.dll")]
    for p in dlls:
        assert _arch.dll_arch(p) in {"arm64", "x64", "x86"}
        assert p.parent.parent.name in ("build", f"build-{_arch.dll_arch(p)}", "build-python")


def test_native_loader_never_picks_a_foreign_arch_dll():
    if native.available():
        loaded = native.loaded_path()
        if sys.platform == "win32":
            assert _arch.dll_arch(loaded) == _arch.python_arch()
        assert "native engine OK" in native.diagnosis()
    else:
        assert "python dev.py build" in native.diagnosis()


def test_dev_doctor_runs_without_path():
    """dev.py must locate its tools even from a shell whose PATH has nothing useful on it."""
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PATH", "PYTHONPATH")}
    env["PATH"] = r"C:\Windows\System32;C:\Windows" if sys.platform == "win32" else "/usr/bin:/bin"
    r = subprocess.run([sys.executable, str(ROOT / "dev.py"), "doctor"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert "machine   :" in r.stdout and "python    :" in r.stdout and "cmake     :" in r.stdout, r.stdout + r.stderr
    assert "Traceback" not in r.stderr
