"""Train an image-recognition model and optionally export an int8 quantized variant.

Examples:
    python -m swarm.image_recognition.train_image_recognition --epochs 3
    python -m swarm.image_recognition.train_image_recognition --epochs 5 --quantize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zlib

import numpy as np

from swarm.brain import formats
from swarm.brain.model import Sequential
from swarm.train.dataset import make_dataset
from swarm.train.train_tiny_cnn import accuracy, build_tiny_cnn, train

MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "image_recognition"


def export_models(model: Sequential, name: str, out_dir: Path, quantize: bool) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    f32_path = out_dir / f"{name}_f32.swm"
    model.save(f32_path, quantize=False)
    out["f32"] = f32_path

    if quantize:
        int8_path = out_dir / f"{name}_int8.swm"
        model.save(int8_path, quantize=True)
        out["int8"] = int8_path

    return out


def _pack_layers(model: Sequential, out_path: Path, quantize: bool) -> Path:
    """Store layer params as compressed blobs plus JSON metadata."""
    specs = model.specs(quantize=quantize, with_softmax=True)
    metadata: dict[str, object] = {"input_shape": list(model.input_shape), "layers": []}
    arrays: dict[str, np.ndarray] = {}

    for li, layer in enumerate(specs):
        entry: dict[str, object] = {}
        for key, value in layer.items():
            if isinstance(value, np.ndarray):
                blob_key = f"layer_{li}_{key}"
                raw = np.ascontiguousarray(value).tobytes()
                comp = zlib.compress(raw, level=9)
                arrays[blob_key] = np.frombuffer(comp, dtype=np.uint8)
                entry[key] = {
                    "blob": blob_key,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            elif isinstance(value, (np.integer, np.floating)):
                entry[key] = value.item()
            else:
                entry[key] = value
        cast_layers = metadata["layers"]
        assert isinstance(cast_layers, list)
        cast_layers.append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=np.array(json.dumps(metadata)), **arrays)
    return out_path


def _unpack_layers_to_swm(bundle_path: Path, swm_path: Path) -> Path:
    """Rebuild a standard .swm model from compressed layer blobs."""
    with np.load(bundle_path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["meta"].item()))
        input_shape = tuple(int(v) for v in metadata["input_shape"])
        specs: list[dict] = []
        for entry in metadata["layers"]:
            layer: dict = {}
            for key, value in entry.items():
                if isinstance(value, dict) and {"blob", "dtype", "shape"} <= set(value):
                    blob = bytes(bundle[value["blob"]].tolist())
                    raw = zlib.decompress(blob)
                    arr = np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(tuple(value["shape"])).copy()
                    layer[key] = arr
                else:
                    layer[key] = value
            specs.append(layer)
    formats.write_swm(swm_path, input_shape, specs)
    return swm_path


def export_compressed_variants(
    model: Sequential,
    exported: dict[str, Path],
    name: str,
    out_dir: Path,
) -> dict[str, Path]:
    """Create compressed-layer bundles and decompressed .swm round-trips."""
    out: dict[str, Path] = {}
    for kind in exported:
        bundle = _pack_layers(model, out_dir / f"{name}_{kind}_layers.npz", quantize=(kind == "int8"))
        restored = _unpack_layers_to_swm(bundle, out_dir / f"{name}_{kind}_from_layers.swm")
        out[f"{kind}_layers"] = bundle
        out[f"{kind}_from_layers"] = restored
    return out


def verify_exports(model: Sequential, exported: dict[str, Path], x: np.ndarray) -> None:
    ref = model.predict_proba(x[:8])
    for kind, path in exported.items():
        loaded = Sequential.load(path)
        got = loaded.predict_proba(x[:8], int8=(kind == "int8"))
        err = float(np.abs(got - ref).max())
        print(f"verify {kind:<4}: max |dp| = {err:.5f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--size", type=int, default=32, help="frame size in pixels")
    parser.add_argument("--width", type=int, default=8, help="first conv width (second conv is 2x)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="image_recognition")
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    parser.add_argument("--quantize", action="store_true", help="also export int8 quantized weights")
    parser.add_argument(
        "--compress-layers",
        action="store_true",
        help="export compressed per-layer bundles and restore them back to .swm",
    )
    args = parser.parse_args(argv)

    print("building synthetic image-recognition dataset...")
    train_ds = make_dataset(args.train_size, args.size, seed=args.seed)
    test_ds = make_dataset(args.test_size, args.size, seed=args.seed + 1)

    model = build_tiny_cnn(args.size, args.width, args.seed)
    print(model.summary())

    train(model, train_ds, test_ds, args.epochs, args.batch_size, args.lr, args.seed)

    exported = export_models(model, args.name, args.out, args.quantize)
    verify_exports(model, exported, test_ds.x)

    extras: dict[str, Path] = {}
    if args.compress_layers:
        extras = export_compressed_variants(model, exported, args.name, args.out)

    print("\nmodel artifacts (size + accuracy):")
    for kind, path in exported.items():
        eval_model = Sequential.load(path)
        acc = accuracy(eval_model, test_ds.x, test_ds.y, int8=(kind == "int8"))
        print(f"- {kind:<16} {path.stat().st_size:>9} bytes   acc={acc:.3f}   {path}")
        layer_bundle = extras.get(f"{kind}_layers")
        restored = extras.get(f"{kind}_from_layers")
        if layer_bundle and restored:
            restored_model = Sequential.load(restored)
            restored_acc = accuracy(restored_model, test_ds.x, test_ds.y, int8=(kind == "int8"))
            print(f"  {kind + '_layers':<16} {layer_bundle.stat().st_size:>9} bytes   compressed bundle")
            print(f"  {kind + '_from_layers':<16} {restored.stat().st_size:>9} bytes   acc={restored_acc:.3f}   {restored}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
