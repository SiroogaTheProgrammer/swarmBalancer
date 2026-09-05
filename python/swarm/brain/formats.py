"""The ``.swm`` binary model format shared between Python and the C++ engine.

Little-endian, no alignment padding (see docs/MODEL_FORMAT.md)::

    char[4] magic = "SWM1"
    u32     n_layers
    u32     in_c, in_h, in_w
    repeat n_layers:
        u32 type      1=dense 2=conv2d 3=relu 4=maxpool2d 5=flatten 6=softmax
        u32 dtype     0=f32 1=i8          (only dense/conv2d may be i8)
        dense    : u32 in, u32 out, f32 w_scale, w[in*out] (f32|i8), f32 bias[out]
        conv2d   : u32 in_c, out_c, k, stride, pad, f32 w_scale, w[out_c*in_c*k*k] (f32|i8), f32 bias[out_c]
        maxpool2d: u32 k, u32 stride
        relu / flatten / softmax: (nothing)

Weight layouts: dense ``[in][out]``, conv ``[out_c][in_c*k*k]`` with the
``in_c*k*k`` index ordered ``c*k*k + ki*k + kj``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

MAGIC = b"SWM1"
LAYER_TYPE_IDS = {"dense": 1, "conv2d": 2, "relu": 3, "maxpool2d": 4, "flatten": 5, "softmax": 6}
LAYER_TYPE_NAMES = {v: k for k, v in LAYER_TYPE_IDS.items()}
DTYPE_IDS = {"f32": 0, "i8": 1}
DTYPE_NAMES = {v: k for k, v in DTYPE_IDS.items()}


def _u32(v: int) -> bytes:
    return struct.pack("<I", int(v))


def _f32(v: float) -> bytes:
    return struct.pack("<f", float(v))


def _weights_bytes(w: np.ndarray, dtype: str) -> bytes:
    if dtype == "i8":
        return np.ascontiguousarray(w, dtype=np.int8).tobytes()
    return np.ascontiguousarray(w, dtype="<f4").tobytes()


def encode_swm(input_shape: tuple[int, int, int], layers: list[dict]) -> bytes:
    """Serialises layer specs (as produced by ``Layer.spec``) into a ``.swm`` image."""
    out = bytearray(MAGIC)
    out += _u32(len(layers))
    for d in input_shape:
        out += _u32(d)
    for L in layers:
        t = L["type"]
        dtype = L.get("dtype", "f32")
        out += _u32(LAYER_TYPE_IDS[t]) + _u32(DTYPE_IDS[dtype])
        if t == "dense":
            out += _u32(L["in"]) + _u32(L["out"]) + _f32(L.get("w_scale", 1.0))
            out += _weights_bytes(L["w"], dtype)
            out += np.ascontiguousarray(L["b"], dtype="<f4").tobytes()
        elif t == "conv2d":
            out += _u32(L["in_c"]) + _u32(L["out_c"]) + _u32(L["k"]) + _u32(L["stride"]) + _u32(L["pad"])
            out += _f32(L.get("w_scale", 1.0))
            out += _weights_bytes(L["w"], dtype)
            out += np.ascontiguousarray(L["b"], dtype="<f4").tobytes()
        elif t == "maxpool2d":
            out += _u32(L["k"]) + _u32(L["stride"])
        elif t in ("relu", "flatten", "softmax"):
            pass
        else:
            raise ValueError(f"unknown layer type {t!r}")
    return bytes(out)


def write_swm(path: str | Path, input_shape: tuple[int, int, int], layers: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(encode_swm(input_shape, layers))


class _Reader:
    def __init__(self, f: BinaryIO):
        self.f = f

    def u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self._read(4))[0]

    def array(self, count: int, dtype: str) -> np.ndarray:
        np_dtype = np.int8 if dtype == "i8" else np.dtype("<f4")
        raw = self._read(count * np.dtype(np_dtype).itemsize)
        return np.frombuffer(raw, dtype=np_dtype).copy()

    def _read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated .swm file")
        return b


def read_swm(path: str | Path) -> tuple[tuple[int, int, int], list[dict]]:
    """Parses a ``.swm`` file back into ``(input_shape, layer_specs)``."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path}: not a .swm file")
        r = _Reader(f)
        n = r.u32()
        input_shape = (r.u32(), r.u32(), r.u32())
        layers: list[dict] = []
        for _ in range(n):
            t = LAYER_TYPE_NAMES[r.u32()]
            dtype = DTYPE_NAMES[r.u32()]
            L: dict = {"type": t, "dtype": dtype}
            if t == "dense":
                L["in"], L["out"], L["w_scale"] = r.u32(), r.u32(), r.f32()
                L["w"] = r.array(L["in"] * L["out"], dtype).reshape(L["in"], L["out"])
                L["b"] = r.array(L["out"], "f32")
            elif t == "conv2d":
                L["in_c"], L["out_c"], L["k"], L["stride"], L["pad"] = (r.u32() for _ in range(5))
                L["w_scale"] = r.f32()
                kk = L["in_c"] * L["k"] * L["k"]
                L["w"] = r.array(L["out_c"] * kk, dtype).reshape(L["out_c"], kk)
                L["b"] = r.array(L["out_c"], "f32")
            elif t == "maxpool2d":
                L["k"], L["stride"] = r.u32(), r.u32()
            layers.append(L)
        if f.read(1):
            raise ValueError("trailing bytes in .swm file")
    return input_shape, layers


def describe_swm(path: str | Path) -> str:
    """Human-readable summary: shapes, MACs, weight bytes per layer."""
    from .model import Sequential  # local import to avoid a cycle

    m = Sequential.load(path)
    return m.summary()
