"""Synthetic overhead imagery for image-recognition brains.

Real satellite / drone footage is not in the repo, so this generates small
greyscale "top-down" frames with a textured background and one of a few
object classes rendered at a random position, size and rotation, plus sensor
noise. It is deliberately simple - the point is to have a deterministic,
dependency-free stream of frames to train the brain on and to feed the swarm
simulator - but the classes are distinguishable only by shape, so a network
actually has to learn something.

Classes (``CLASSES``):
    0 nothing   - background only
    1 vehicle   - small filled rectangle (aspect ~2:1)
    2 building  - large square with a darker roof edge
    3 road      - long thin line crossing the frame
    4 aircraft  - cross / T shape

Frames are ``float32`` in ``[0, 1]`` with shape ``(1, size, size)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CLASSES = ["nothing", "vehicle", "building", "road", "aircraft"]
NUM_CLASSES = len(CLASSES)


def _background(rng: np.random.Generator, size: int) -> np.ndarray:
    # low-frequency blotches (fields / terrain) + fine noise
    coarse = rng.uniform(0.25, 0.6, size=(size // 8 + 1, size // 8 + 1)).astype(np.float32)
    bg = np.ascontiguousarray(np.kron(coarse, np.ones((8, 8), dtype=np.float32))[:size, :size])
    bg += rng.normal(0, 0.03, size=(size, size)).astype(np.float32)
    return bg


def _rot(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return points @ np.array([[c, -s], [s, c]], dtype=np.float32)


def _paint_polygon(img: np.ndarray, poly: np.ndarray, value: float) -> None:
    """Fills a convex polygon (Nx2, pixel coords) via half-plane tests on the pixel grid."""
    size = img.shape[0]
    ys, xs = np.mgrid[0:size, 0:size]
    pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1).astype(np.float32)
    inside = np.ones(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        edge = b - a
        rel = pts - a
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        inside &= cross >= 0
    img[inside.reshape(img.shape)] = value


def _rect(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float32)
    return _rot(corners, angle) + np.array([cx, cy], dtype=np.float32)


def render(cls: int, rng: np.random.Generator, size: int = 32) -> np.ndarray:
    img = _background(rng, size)
    cx, cy = rng.uniform(size * 0.3, size * 0.7, size=2)
    angle = rng.uniform(0, math.pi)
    bright = rng.uniform(0.75, 1.0)
    dark = rng.uniform(0.0, 0.15)

    if cls == 1:  # vehicle
        w = rng.uniform(size * 0.12, size * 0.2)
        _paint_polygon(img, _rect(cx, cy, w, w * 0.5, angle), bright)
    elif cls == 2:  # building
        w = rng.uniform(size * 0.3, size * 0.45)
        _paint_polygon(img, _rect(cx, cy, w, w * rng.uniform(0.8, 1.2), angle), dark)
        _paint_polygon(img, _rect(cx, cy, w * 0.75, w * 0.75, angle), bright * 0.8)
    elif cls == 3:  # road
        _paint_polygon(img, _rect(cx, cy, size * 1.6, rng.uniform(1.5, 3.0), angle), dark)
    elif cls == 4:  # aircraft: fuselage + wings
        L = rng.uniform(size * 0.35, size * 0.5)
        _paint_polygon(img, _rect(cx, cy, L, L * 0.15, angle), bright)
        _paint_polygon(img, _rect(cx, cy, L * 0.18, L * 0.8, angle), bright)

    img += rng.normal(0, 0.02, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)[None].astype(np.float32)


@dataclass
class Dataset:
    x: np.ndarray  # (N, 1, size, size) float32
    y: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return len(self.y)

    def batches(self, batch_size: int, rng: np.random.Generator):
        idx = rng.permutation(len(self))
        for i in range(0, len(idx), batch_size):
            j = idx[i : i + batch_size]
            yield self.x[j], self.y[j]


def make_dataset(n: int, size: int = 32, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, NUM_CLASSES, size=n)
    x = np.stack([render(int(c), rng, size) for c in y])
    return Dataset(x, y.astype(np.int64))


class FramePool:
    """Pre-rendered frames grouped by class; the simulator samples from it instead of rendering per frame."""

    def __init__(self, size: int = 32, per_class: int = 64, seed: int = 12345):
        rng = np.random.default_rng(seed)
        self.size = size
        self.frames = [np.stack([render(c, rng, size) for _ in range(per_class)]) for c in range(NUM_CLASSES)]

    def sample(self, cls: int, rng: np.random.Generator) -> np.ndarray:
        pool = self.frames[cls]
        return pool[int(rng.integers(0, len(pool)))]


class FrameStream:
    """Endless deterministic stream of ``(frame, label)`` for the simulator.

    ``p_object`` controls how often a frame contains something (the
    "useful result" rate that the striped strategy exploits). With a
    ``pool`` frames are drawn from pre-rendered images (fast); without one
    every frame is rendered fresh.
    """

    def __init__(self, size: int = 32, seed: int = 0, p_object: float = 0.3, pool: FramePool | None = None):
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.p_object = p_object
        self.pool = pool
        if pool is not None and pool.size != size:
            raise ValueError(f"pool renders {pool.size}px frames, stream wants {size}px")

    def next(self) -> tuple[np.ndarray, int]:
        if self.rng.random() < self.p_object:
            cls = int(self.rng.integers(1, NUM_CLASSES))
        else:
            cls = 0
        if self.pool is not None:
            return self.pool.sample(cls, self.rng), cls
        return render(cls, self.rng, self.size), cls
