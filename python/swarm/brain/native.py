"""ctypes bridge to the C++ engine (``swarm_brain`` shared library).

Everything degrades gracefully: if the library has not been built,
:func:`available` returns ``False`` and callers fall back to the numpy
engine. Build it with ``cmake --preset mingw-arm64 && cmake --build --preset mingw-arm64``
(or ``default`` on Linux/macOS); the loader looks in ``build*/bin``.
"""

from __future__ import annotations

import ctypes as C
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
_LIB_NAMES = {"win32": "swarm_brain.dll", "darwin": "libswarm_brain.dylib"}
LIB_NAME = _LIB_NAMES.get(sys.platform, "libswarm_brain.so")


def candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("SWARM_BRAIN_LIB")
    if env:
        paths.append(Path(env))
    for build_dir in sorted(REPO_ROOT.glob("build*")):
        paths.append(build_dir / "bin" / LIB_NAME)
        paths.append(build_dir / "cpp" / LIB_NAME)
    return paths


@lru_cache(maxsize=1)
def _load() -> C.CDLL | None:
    for p in candidate_paths():
        if not p.is_file():
            continue
        try:
            lib = C.CDLL(str(p))
        except OSError:
            # Typically an architecture mismatch (e.g. ARM64 DLL vs. an x64 python.exe
            # running emulated): keep looking, another build dir may have the right one.
            continue
        _declare(lib)
        return lib
    return None


def _declare(lib: C.CDLL) -> None:
    f32p, i8p, i32p = C.POINTER(C.c_float), C.POINTER(C.c_int8), C.POINTER(C.c_int32)
    lib.swm_build_info.restype = C.c_char_p
    lib.swm_last_error.restype = C.c_char_p
    lib.swm_load.restype = C.c_void_p
    lib.swm_load.argtypes = [C.c_char_p, C.c_size_t]
    lib.swm_free.argtypes = [C.c_void_p]
    lib.swm_input_shape.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int)]
    lib.swm_output_shape.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int)]
    lib.swm_num_layers.argtypes = [C.c_void_p]
    lib.swm_run.argtypes = [C.c_void_p, f32p, C.c_size_t, f32p, C.c_size_t]
    lib.swm_macs_per_run.restype = C.c_uint64
    lib.swm_macs_per_run.argtypes = [C.c_void_p]
    lib.swm_weights_bytes.restype = C.c_size_t
    lib.swm_weights_bytes.argtypes = [C.c_void_p]
    lib.swm_required_bytes.restype = C.c_size_t
    lib.swm_required_bytes.argtypes = [C.c_void_p]
    for name in ("swm_gemm_f32", "swm_gemm_f32_naive"):
        getattr(lib, name).argtypes = [C.c_int, C.c_int, C.c_int, f32p, f32p, f32p]
    for name in ("swm_gemm_i8", "swm_gemm_i8_naive"):
        getattr(lib, name).argtypes = [C.c_int, C.c_int, C.c_int, i8p, i8p, i32p]


def available() -> bool:
    return _load() is not None


def build_info() -> str:
    lib = _load()
    return lib.swm_build_info().decode() if lib else "native engine not built"


def _require() -> C.CDLL:
    lib = _load()
    if lib is None:
        raise RuntimeError(
            f"native engine not found or not loadable (looked for {LIB_NAME} in {[str(p) for p in candidate_paths()]}). "
            "Build it with: cmake --preset mingw-arm64 && cmake --build --preset mingw-arm64 "
            "(if python.exe is an x64 build on an ARM64 PC, also: cmake --preset mingw-x64 && cmake --build --preset mingw-x64)"
        )
    return lib


class NativeModel:
    """A ``.swm`` model running inside the C++ engine, optionally under a device RAM cap."""

    def __init__(self, path: str | Path, ram_cap_bytes: int = 0):
        self._lib = _require()
        self._h = self._lib.swm_load(str(path).encode(), int(ram_cap_bytes))
        if not self._h:
            raise MemoryError(self._lib.swm_last_error().decode())
        self.input_shape = self._shape(self._lib.swm_input_shape)
        self.output_shape = self._shape(self._lib.swm_output_shape)
        self.num_layers = self._lib.swm_num_layers(self._h)
        self.macs = self._lib.swm_macs_per_run(self._h)
        self.weights_bytes = self._lib.swm_weights_bytes(self._h)
        self.required_bytes = self._lib.swm_required_bytes(self._h)

    def _shape(self, fn) -> tuple[int, int, int]:
        c, h, w = C.c_int(), C.c_int(), C.c_int()
        fn(self._h, C.byref(c), C.byref(h), C.byref(w))
        return (c.value, h.value, w.value)

    def run(self, x: np.ndarray) -> np.ndarray:
        """Single frame ``(C,H,W)`` -> output vector."""
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        y = np.empty(int(np.prod(self.output_shape)), dtype=np.float32)
        rc = self._lib.swm_run(self._h, x.ctypes.data_as(C.POINTER(C.c_float)), x.size,
                               y.ctypes.data_as(C.POINTER(C.c_float)), y.size)
        if rc != 0:
            raise RuntimeError(self._lib.swm_last_error().decode())
        return y

    def run_batch(self, xs: np.ndarray) -> np.ndarray:
        return np.stack([self.run(x) for x in xs])

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.swm_free(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def gemm_f32(a: np.ndarray, b: np.ndarray, naive: bool = False) -> np.ndarray:
    lib = _require()
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = np.empty((M, N), dtype=np.float32)
    fn = lib.swm_gemm_f32_naive if naive else lib.swm_gemm_f32
    p = C.POINTER(C.c_float)
    fn(M, N, K, a.ctypes.data_as(p), b.ctypes.data_as(p), c.ctypes.data_as(p))
    return c


def gemm_i8(a: np.ndarray, b: np.ndarray, naive: bool = False) -> np.ndarray:
    lib = _require()
    a = np.ascontiguousarray(a, dtype=np.int8)
    b = np.ascontiguousarray(b, dtype=np.int8)
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = np.empty((M, N), dtype=np.int32)
    fn = lib.swm_gemm_i8_naive if naive else lib.swm_gemm_i8
    p8 = C.POINTER(C.c_int8)
    fn(M, N, K, a.ctypes.data_as(p8), b.ctypes.data_as(p8), c.ctypes.data_as(C.POINTER(C.c_int32)))
    return c
