"""Shared fixtures: a tiny trained-ish model exported to .swm in a temp dir."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from swarm.brain import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential  # noqa: E402
from swarm.brain import native  # noqa: E402


def tiny_model(seed: int = 0, size: int = 16) -> Sequential:
    rng = np.random.default_rng(seed)
    return Sequential(
        [
            Conv2D(1, 4, 3, 1, 1, rng=rng), ReLU(), MaxPool2D(2),
            Conv2D(4, 8, 3, 1, 1, rng=rng), ReLU(), MaxPool2D(2),
            Flatten(),
            Dense(8 * (size // 4) ** 2, 16, rng=rng), ReLU(),
            Dense(16, 5, rng=rng),
        ],
        input_shape=(1, size, size),
    )


@pytest.fixture(scope="session")
def model() -> Sequential:
    return tiny_model()


@pytest.fixture(scope="session")
def swm_paths(model, tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("models")
    return {"f32": model.save(d / "tiny_f32.swm"), "int8": model.save(d / "tiny_int8.swm", quantize=True)}


@pytest.fixture(scope="session")
def frames() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, size=(6, 1, 16, 16)).astype(np.float32)


needs_native = pytest.mark.skipif(not native.available(), reason="C++ engine not built (cmake --preset ... && cmake --build ...)")
