"""Defines, trains and compiles micro_cnn into model.swm - the file that flies.

    python examples/brains/micro_cnn/train.py      (or COMPILE in the studio's ARMORY)

A deliberately small network for microcontroller-class drones: 6 and 12 conv channels, a 24-unit
dense layer, int8 weights (~20 KB) and ~0.24 MMAC per 32x32 frame - about 3 ms on a Cortex-M4.
swarm.brain layers use the same maths as the C++ engine, so the exported model.swm is exactly what
runs on the drone; for a .swm brain, "compile" means "export".
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    import swarm  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`: find <repo>/python
    for parent in HERE.parents:
        if (parent / "python" / "swarm").is_dir():
            sys.path.insert(0, str(parent / "python"))
            break

import numpy as np  # noqa: E402

from swarm.brain import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential  # noqa: E402
from swarm.train.dataset import NUM_CLASSES, make_dataset  # noqa: E402
from swarm.train.train_tiny_cnn import accuracy, train  # noqa: E402

SIZE = 32
EPOCHS = 3


def build(seed=0):
    rng = np.random.default_rng(seed)
    return Sequential(
        [
            Conv2D(1, 6, k=3, stride=1, pad=1, rng=rng), ReLU(), MaxPool2D(2),
            Conv2D(6, 12, k=3, stride=1, pad=1, rng=rng), ReLU(), MaxPool2D(2),
            Flatten(),  # split point: central/features ships this tensor (int8) to the leader
            Dense(12 * (SIZE // 4) ** 2, 24, rng=rng), ReLU(),
            Dense(24, NUM_CLASSES, rng=rng),
        ],
        input_shape=(1, SIZE, SIZE),
    )


def main():
    train_ds = make_dataset(3000, SIZE, seed=1)
    test_ds = make_dataset(600, SIZE, seed=2)
    model = build()
    print(model.summary())
    train(model, train_ds, test_ds, epochs=EPOCHS, batch_size=32, lr=2e-3, seed=0)
    out = model.save(HERE / "model.swm", quantize=True)
    print(f"int8 test accuracy {accuracy(model, test_ds.x, test_ds.y, int8=True):.3f} -> {out}")


if __name__ == "__main__":
    main()
