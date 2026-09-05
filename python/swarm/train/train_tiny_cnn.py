"""Trains ``tiny_cnn`` - the satellite/drone image-recognition brain - and exports it.

    python -m swarm.train.train_tiny_cnn --epochs 3

Produces ``models/tiny_cnn_f32.swm`` and ``models/tiny_cnn_int8.swm`` (plus a
``.npz`` of the raw weights), then verifies both files load and agree with
numpy. If the C++ engine is built it also checks the native engine matches.

The architecture is intentionally minimal (~7k params, ~0.26 MMAC/frame at 32x32)
so it fits the "super minimalist" budget of a satellite OBC or a drone MCU;
see ``build_tiny_cnn`` to change it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from swarm.brain import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential, softmax_cross_entropy
from swarm.brain import native
from swarm.train.dataset import NUM_CLASSES, make_dataset

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def build_tiny_cnn(size: int = 32, width: int = 8, seed: int = 0) -> Sequential:
    rng = np.random.default_rng(seed)
    c1, c2 = width, width * 2
    flat = c2 * (size // 4) * (size // 4)
    return Sequential(
        [
            Conv2D(1, c1, k=3, stride=1, pad=1, rng=rng),
            ReLU(),
            MaxPool2D(2),
            Conv2D(c1, c2, k=3, stride=1, pad=1, rng=rng),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Dense(flat, 32, rng=rng),
            ReLU(),
            Dense(32, NUM_CLASSES, rng=rng),
        ],
        input_shape=(1, size, size),
    )


class Adam:
    def __init__(self, model: Sequential, lr: float = 2e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.model, self.lr, self.b1, self.b2, self.eps = model, lr, betas[0], betas[1], eps
        self.t = 0
        self.m = [np.zeros_like(p) for p, _ in model.parameters()]
        self.v = [np.zeros_like(p) for p, _ in model.parameters()]

    def step(self) -> None:
        self.t += 1
        for i, (p, g) in enumerate(self.model.parameters()):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1**self.t)
            vh = self.v[i] / (1 - self.b2**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def accuracy(model: Sequential, x: np.ndarray, y: np.ndarray, int8: bool = False, batch: int = 256) -> float:
    correct = 0
    for i in range(0, len(y), batch):
        correct += int((model.predict(x[i : i + batch], int8=int8) == y[i : i + batch]).sum())
    return correct / len(y)


def train(model: Sequential, train_ds, test_ds, epochs: int, batch_size: int, lr: float, seed: int, log=print) -> None:
    opt = Adam(model, lr=lr)
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        losses = []
        for xb, yb in train_ds.batches(batch_size, rng):
            logits = model.forward(xb)
            loss, dlogits = softmax_cross_entropy(logits, yb)
            model.backward(dlogits)
            opt.step()
            losses.append(loss)
        acc = accuracy(model, test_ds.x, test_ds.y)
        log(f"epoch {ep}/{epochs}  loss={np.mean(losses):.4f}  test_acc={acc:.3f}  ({time.perf_counter() - t0:.1f}s)")


def export(model: Sequential, name: str, out_dir: Path, log=print) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "f32": model.save(out_dir / f"{name}_f32.swm", quantize=False),
        "int8": model.save(out_dir / f"{name}_int8.swm", quantize=True),
    }
    np.savez(out_dir / f"{name}.npz", **{f"{i}.{k}": v for i, L in enumerate(model.layers) for k, v in L.params().items()})
    for k, p in paths.items():
        log(f"wrote {p}  ({p.stat().st_size} bytes)")
    return paths


def verify(model: Sequential, paths: dict[str, Path], x: np.ndarray, log=print) -> None:
    """Round-trips through the .swm files and, if available, the C++ engine."""
    ref = model.predict_proba(x[:8])
    for kind, p in paths.items():
        back = Sequential.load(p)
        got = back.predict_proba(x[:8], int8=(kind == "int8"))
        err = float(np.abs(got - ref).max())
        log(f"numpy reload {kind:<4}: max |dp| = {err:.4f}")
    if native.available():
        log(f"native engine: {native.build_info()}")
        for kind, p in paths.items():
            with native.NativeModel(p) as nm:
                got = nm.run_batch(x[:8])
            numpy_same_path = model.predict_proba(x[:8], int8=(kind == "int8"))
            err = float(np.abs(got - numpy_same_path).max())
            log(f"native {kind:<4}: max |dp| vs numpy = {err:.5f}  "
                f"(MACs={nm.macs}, weights={nm.weights_bytes} B, RAM={nm.required_bytes} B)")
    else:
        log("native engine not built - skipped C++ cross-check (cmake --preset mingw-arm64 && cmake --build --preset mingw-arm64)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--size", type=int, default=32, help="frame size (pixels)")
    ap.add_argument("--width", type=int, default=8, help="channels of the first conv (second has 2x)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="tiny_cnn")
    ap.add_argument("--out", type=Path, default=MODELS_DIR)
    args = ap.parse_args(argv)

    print("generating synthetic overhead imagery ...")
    train_ds = make_dataset(args.train_size, args.size, seed=args.seed)
    test_ds = make_dataset(args.test_size, args.size, seed=args.seed + 1)

    model = build_tiny_cnn(args.size, args.width, args.seed)
    print(model.summary())
    train(model, train_ds, test_ds, args.epochs, args.batch_size, args.lr, args.seed)
    print(f"int8 test accuracy: {accuracy(model, test_ds.x, test_ds.y, int8=True):.3f}")

    paths = export(model, args.name, args.out)
    verify(model, paths, test_ds.x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
