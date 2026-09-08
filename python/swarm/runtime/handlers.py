"""Trusted-local brain factories. The network transports bytes, never factory names."""

from __future__ import annotations

import ctypes as C
import hashlib
import importlib
import inspect
import math
import struct
import threading
from pathlib import Path

from .node import InferenceResult


def load_handler(factory: str, options: dict):
    module, name = factory.split(":")
    make = getattr(importlib.import_module(module), name)
    if not callable(make) or inspect.iscoroutinefunction(make):
        raise ValueError("brain factory must be synchronous trusted local code")
    handler = make(dict(options))
    if not callable(handler) or inspect.iscoroutinefunction(handler):
        raise ValueError("brain factory must return a synchronous inference callable")
    inspect.signature(handler).bind(b"")
    return handler


def make_digest_handler(options: dict):
    """Hardware-free transport smoke test; NOT an image recognition brain."""
    if options:
        raise ValueError("digest test handler does not accept options")

    def infer(payload: bytes) -> InferenceResult:
        return InferenceResult(hashlib.sha256(payload).digest())

    return infer


class NativeHandler:
    """Use the C++ ABI without importing numpy/training. Input is uint8 CHW.

    Output is big-endian ``uint32 class, float32 confidence`` (8 bytes). Empty
    predictions send only a completion flag. One instance has ONE owning worker.
    The explicit model digest prevents accidentally serving different weights
    under the same workload ID. Library code is locally trusted, not sandboxed.
    """

    def __init__(self, options: dict):
        self._lock = threading.Lock()
        allowed = {"library", "model", "sha256", "ram_cap_bytes", "threshold", "empty_class"}
        if options.keys() - allowed or not {"library", "model", "sha256", "ram_cap_bytes"} <= options.keys():
            raise ValueError("native brain needs library, model, sha256 and ram_cap_bytes")
        model = Path(options["model"]).resolve(strict=True)
        library = Path(options["library"]).resolve(strict=True)
        if model.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("model exceeds the runtime's 64 MiB limit")
        if hashlib.sha256(model.read_bytes()).hexdigest() != options["sha256"]:
            raise ValueError("model digest does not match local configuration")
        cap = options["ram_cap_bytes"]
        if type(cap) is not int or not 0 < cap <= 1024 * 1024 * 1024:
            raise ValueError("ram_cap_bytes must be a positive integer <= 1 GiB")
        self.threshold = options.get("threshold", 0.5)
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be within [0, 1]")
        self.empty_class = options.get("empty_class", 0)
        if type(self.empty_class) is not int or self.empty_class < 0:
            raise ValueError("empty_class must be a nonnegative class index")
        self.lib = C.CDLL(str(library))
        pfloat = C.POINTER(C.c_float)
        self.lib.swm_load.argtypes = [C.c_char_p, C.c_size_t]
        self.lib.swm_load.restype = C.c_void_p
        self.lib.swm_free.argtypes = [C.c_void_p]
        self.lib.swm_free.restype = None
        for name in ("swm_input_shape", "swm_output_shape"):
            function = getattr(self.lib, name)
            function.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int)]
            function.restype = C.c_int
        self.lib.swm_run.argtypes = [C.c_void_p, pfloat, C.c_size_t, pfloat, C.c_size_t]
        self.lib.swm_run.restype = C.c_int
        self.handle = self.lib.swm_load(str(model).encode("utf-8"), cap)
        if not self.handle:
            raise ValueError("native model failed to load under the configured RAM cap")
        try:
            self.input_count = self._count(self.lib.swm_input_shape)
            self.output_count = self._count(self.lib.swm_output_shape)
            if not 1 <= self.input_count <= 1024 * 1024 or not 1 <= self.output_count <= 65536:
                raise ValueError("model input/output exceeds runtime limits")
            if self.empty_class >= self.output_count:
                raise ValueError("empty_class exceeds model output size")
            self._input = (C.c_float * self.input_count)()
            self._output = (C.c_float * self.output_count)()
        except BaseException:
            self.close()
            raise

    def _count(self, function):
        c, h, w = C.c_int(), C.c_int(), C.c_int()
        if function(self.handle, C.byref(c), C.byref(h), C.byref(w)) != 0:
            raise ValueError("invalid native model shape")
        return c.value * h.value * w.value

    def __call__(self, payload: bytes) -> InferenceResult:
        # Guard also protects an accidental second local caller or concurrent close.
        with self._lock:
            return self._infer(payload)

    def _infer(self, payload: bytes) -> InferenceResult:
        if not self.handle or len(payload) != self.input_count:
            raise ValueError("expected exactly one uint8 CHW model input")
        for i, value in enumerate(payload):
            self._input[i] = value / 255.0
        rc = self.lib.swm_run(self.handle, self._input, self.input_count, self._output, self.output_count)
        if rc != 0 or any(not math.isfinite(x) for x in self._output):
            raise ValueError("native inference failed")
        best = max(range(self.output_count), key=self._output.__getitem__)
        confidence = self._output[best]
        if not 0 <= confidence <= 1:
            raise ValueError("runtime native model must export final softmax probabilities")
        useful = best != self.empty_class and confidence >= self.threshold
        return InferenceResult(struct.pack("!If", best, confidence) if useful else b"", useful)

    def close(self):
        with self._lock:
            if getattr(self, "handle", None):
                self.lib.swm_free(self.handle)
                self.handle = None

    def __del__(self):
        self.close()


def make_native_handler(options: dict) -> NativeHandler:
    return NativeHandler(options)