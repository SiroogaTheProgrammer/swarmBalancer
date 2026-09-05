"""Brains as seen by the simulator: something with a cost (MACs) and a result.

Three implementations, all interchangeable:

* :class:`OracleBrain` - no model at all; returns the ground-truth label with a
  configurable accuracy. Fast, lets you sweep hypothetical model sizes
  (``macs``) and payloads without training anything.
* :class:`NumpyBrain`  - a real ``Sequential`` (from ``.swm``) run with numpy.
* :class:`NativeBrain` - the same ``.swm`` run inside the C++ engine.

A brain can be *split* at its ``Flatten`` layer into a convolutional frontend
(runs on the small drone, emits an int8 feature map) and a dense head (runs
on the leader). This is the "pre-process the image to shrink the payload"
option of the centralized strategy taken to its logical end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swarm.brain.model import Sequential
from swarm.brain.layers import Flatten


@dataclass
class Frame:
    id: int
    t_capture: float
    src: int                 # node whose camera took it
    image: np.ndarray        # brain input (C,H,W) float32
    label: int               # ground truth class (0 = nothing)


class Brain:
    input_shape: tuple[int, int, int]
    macs: int
    frontend_macs: int
    head_macs: int
    feature_bytes: int       # int8 feature map size at the split point
    n_classes: int

    @property
    def input_bytes(self) -> int:
        """Brain input as an 8-bit image (what a pre-processing drone would transmit)."""
        return int(np.prod(self.input_shape))

    def infer(self, frame: Frame) -> tuple[int, float]:
        """Returns ``(predicted_class, confidence)``."""
        raise NotImplementedError

    def describe(self) -> str:
        return (f"{type(self).__name__}: {self.macs} MACs/frame "
                f"(frontend {self.frontend_macs} + head {self.head_macs}), "
                f"input {self.input_bytes} B, features {self.feature_bytes} B")


class OracleBrain(Brain):
    def __init__(self, macs: int = 401_568, accuracy: float = 0.87, input_shape=(1, 32, 32), n_classes: int = 5,
                 frontend_fraction: float = 0.92, feature_bytes: int = 1024, seed: int = 0):
        self.macs = int(macs)
        self.accuracy = accuracy
        self.input_shape = tuple(input_shape)
        self.n_classes = n_classes
        self.frontend_macs = int(macs * frontend_fraction)
        self.head_macs = self.macs - self.frontend_macs
        self.feature_bytes = feature_bytes
        self.rng = np.random.default_rng(seed)

    def infer(self, frame: Frame) -> tuple[int, float]:
        if self.rng.random() < self.accuracy:
            return frame.label, float(self.rng.uniform(0.7, 1.0))
        wrong = [c for c in range(self.n_classes) if c != frame.label]
        return int(self.rng.choice(wrong)), float(self.rng.uniform(0.3, 0.7))


def _split_index(model: Sequential) -> int:
    for i, L in enumerate(model.layers):
        if isinstance(L, Flatten):
            return i
    return len(model.layers)


class NumpyBrain(Brain):
    def __init__(self, path: str | Path, int8: bool = False):
        self.model = Sequential.load(path)
        self.int8 = int8
        self.input_shape = self.model.input_shape
        self.n_classes = int(self.model.output_shape[0])
        self.macs = self.model.macs()
        k = _split_index(self.model)
        shapes = self.model._shapes
        self.frontend_macs = sum(L.macs(s) for L, s in zip(self.model.layers[:k], shapes[:k]))
        self.head_macs = self.macs - self.frontend_macs
        self.feature_bytes = int(np.prod(shapes[k]))

    def infer(self, frame: Frame) -> tuple[int, float]:
        p = self.model.predict_proba(frame.image[None], int8=self.int8)[0]
        c = int(p.argmax())
        return c, float(p[c])


class NativeBrain(Brain):
    def __init__(self, path: str | Path, ram_cap_bytes: int = 0):
        from swarm.brain import native

        self.native = native.NativeModel(path, ram_cap_bytes)
        # Use the numpy model only for the split bookkeeping (no compute).
        ref = NumpyBrain(path)
        self.input_shape = self.native.input_shape
        self.n_classes = int(np.prod(self.native.output_shape))
        self.macs = int(self.native.macs)
        self.frontend_macs, self.head_macs, self.feature_bytes = ref.frontend_macs, ref.head_macs, ref.feature_bytes

    def infer(self, frame: Frame) -> tuple[int, float]:
        p = self.native.run(frame.image)
        c = int(p.argmax())
        return c, float(p[c])


def make_brain(kind: str, model: str | Path | None = None, **kw) -> Brain:
    if kind == "oracle":
        return OracleBrain(**kw)
    if model is None:
        raise ValueError(f"--model is required for brain kind {kind!r}")
    if kind == "numpy":
        return NumpyBrain(model, int8="int8" in str(model))
    if kind == "native":
        return NativeBrain(model, kw.get("ram_cap_bytes", 0))
    raise ValueError(f"unknown brain kind {kind!r}")


def downscale_macs(cam_pixels: int) -> int:
    """Area-averaging a camera frame down to the brain input: ~1 MAC per source pixel."""
    return int(cam_pixels)
