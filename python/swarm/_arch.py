"""Which CPU architecture is *really* in play - machine, Python interpreter, and compiled DLLs.

On an ARM64 Windows PC this is genuinely confusing: the Microsoft Store Python
is an x64 build that runs under emulation, ``platform.machine()`` still says
``ARM64``, ``PROCESSOR_ARCHITECTURE`` says ``AMD64`` inside emulated shells,
and ctypes can only load a DLL of the interpreter's own architecture. This
module answers the three questions from authoritative sources and uses one
vocabulary for all of them: ``"arm64"``, ``"x64"``, ``"x86"``, ``"arm32"``.

Standard library only - ``dev.py`` imports it before numpy is installed.
"""

from __future__ import annotations

import platform
import struct
import sys
import sysconfig
from pathlib import Path

# llvm-mingw target triples per architecture (prefix of the compiler executables)
MINGW_TRIPLES = {"arm64": "aarch64-w64-mingw32", "x64": "x86_64-w64-mingw32", "x86": "i686-w64-mingw32",
                 "arm32": "armv7-w64-mingw32"}

_NAMES = {"arm64": "arm64", "aarch64": "arm64", "amd64": "x64", "x86_64": "x64", "x64": "x64", "x86": "x86",
          "i386": "x86", "i686": "x86", "win32": "x86", "arm": "arm32", "armv7l": "arm32", "armv7": "arm32"}
_PE_MACHINES = {0x8664: "x64", 0xAA64: "arm64", 0x014C: "x86", 0x01C4: "arm32"}


def normalise(name: str | None) -> str | None:
    if not name:
        return None
    return _NAMES.get(name.strip().lower(), name.strip().lower())


def machine_arch() -> str | None:
    """The physical CPU architecture, seen through any emulation layer."""
    if sys.platform == "win32":
        try:
            import winreg

            # Written by the OS at install time; not subject to WOW64 redirection, unlike the
            # PROCESSOR_ARCHITECTURE environment variable of an emulated process.
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as k:
                return normalise(winreg.QueryValueEx(k, "PROCESSOR_ARCHITECTURE")[0])
        except OSError:
            pass
    return normalise(platform.machine())


def python_arch() -> str | None:
    """The architecture this interpreter was compiled for (what ctypes can load)."""
    if sys.platform == "win32":
        plat = sysconfig.get_platform()  # "win-amd64", "win-arm64", "win32" - the build target, not the CPU
        if plat.startswith("win-"):
            return normalise(plat[4:])
        if plat == "win32":
            return "x86"
    arch = normalise(platform.machine())
    if arch == "x64" and struct.calcsize("P") == 4:
        return "x86"
    return arch


def dll_arch(path: str | Path) -> str | None:
    """Architecture of a Windows PE file (.dll/.exe) from its header; None if not a PE file."""
    try:
        with open(path, "rb") as f:
            head = f.read(0x40)
            if len(head) < 0x40 or head[:2] != b"MZ":
                return None
            (pe_off,) = struct.unpack_from("<I", head, 0x3C)
            f.seek(pe_off)
            sig = f.read(6)
            if sig[:4] != b"PE\0\0":
                return None
            (machine,) = struct.unpack_from("<H", sig, 4)
            return _PE_MACHINES.get(machine, f"pe-0x{machine:04x}")
    except OSError:
        return None


def python_is_emulated() -> bool:
    m, p = machine_arch(), python_arch()
    return bool(m and p and m != p)


def describe_python() -> str:
    p, m = python_arch(), machine_arch()
    s = f"{sys.version.split()[0]} {p or '?'}"
    if python_is_emulated():
        s += f" (running emulated on an {m} machine)"
    return s
