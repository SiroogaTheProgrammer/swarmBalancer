"""Stress-test a brain "as if on a small device".

For a ``.swm`` file and a device profile this checks, on the host:

1. **RAM fit** - the C++ engine is asked to load the model under a hard
   arena cap equal to the device's RAM (``swm_load(path, ram_cap)``). This
   is exact: every byte the brain will ever touch (weights, activations,
   im2col/quantisation scratch) comes from that one arena, so if it loads
   here it fits there.
2. **Cycle budget** - the model's MAC count is converted to seconds on the
   device (``profile.compute_seconds``) and compared with the frame period at
   the requested fps -> max sustainable fps, deadline misses, CPU load.
3. **Host reality check** - the model is actually run for ``--frames``
   frames through the C++ engine (or numpy if it is not built) and the
   measured host time is reported next to the device estimate, along with
   the achieved GMAC/s.
4. **Accuracy under quantisation** - if the file is int8, its predictions
   are compared with the f32 sibling on the same frames.

    python -m swarm.stress.device_stress --model models/tiny_cnn_int8.swm --device mcu-m4 --fps 10
    python -m swarm.stress.device_stress --model models/tiny_cnn_int8.swm --sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swarm.brain import native
from swarm.brain.model import Sequential
from swarm.devices import DEVICE_PROFILES, DeviceProfile, get_profile
from swarm.train.dataset import make_dataset


@dataclass
class StressResult:
    device: str
    fits_ram: bool
    ram_needed: int
    ram_cap: int
    ram_error: str
    macs: int
    device_ms_per_frame: float
    max_fps: float
    cpu_load_at_fps: float
    deadline_misses: int
    host_ms_per_frame: float
    host_gmacs: float
    engine: str
    accuracy: float
    int8_agreement: float

    def row(self) -> str:
        fit = "FITS" if self.fits_ram else "OVERFLOW"
        agree = f"{self.int8_agreement*100:>5.1f}%" if self.int8_agreement == self.int8_agreement else "   n/a"
        return (f"{self.device:<12} {fit:<8} {self.ram_needed/1024:>8.1f}/{self.ram_cap/1024:<9.0f} "
                f"{self.device_ms_per_frame:>9.2f} {self.max_fps:>8.1f} {self.cpu_load_at_fps*100:>6.1f}% "
                f"{self.deadline_misses:>6} {self.host_ms_per_frame:>8.3f} {self.host_gmacs:>6.2f} "
                f"{self.accuracy*100:>5.1f}% {agree}")


HEADER = (f"{'device':<12} {'ram':<8} {'need/cap KiB':>18} {'dev ms/fr':>9} {'max fps':>8} {'load':>7} "
          f"{'misses':>6} {'host ms':>8} {'GMAC/s':>6} {'acc':>6} {'i8=f32':>6}")


def check_ram(model_path: Path, profile: DeviceProfile) -> tuple[bool, int, str]:
    """Loads under the device's RAM cap in the C++ engine; falls back to a numpy estimate."""
    if native.available():
        try:
            with native.NativeModel(model_path, ram_cap_bytes=profile.ram_bytes) as m:
                return True, m.required_bytes, ""
        except MemoryError as e:
            msg = str(e)
            needed = int(msg.split("needs ")[1].split(" ")[0]) if "needs " in msg else 0
            return False, needed, msg
    # numpy estimate: weights + 2 activation buffers + scratch (no engine available)
    m = Sequential.load(model_path)
    quant = "int8" in model_path.name
    acts = max(int(np.prod(s)) for s in m._shapes) * 4 * 2
    needed = m.weights_bytes(quant) + acts
    return needed <= profile.ram_bytes, needed, "" if needed <= profile.ram_bytes else "estimated overflow (engine not built)"


def run_frames(model_path: Path, x: np.ndarray, y: np.ndarray) -> tuple[float, str, np.ndarray]:
    """Runs every frame; returns (host seconds per frame, engine name, predictions)."""
    if native.available():
        with native.NativeModel(model_path) as m:
            m.run(x[0])  # warm-up
            t0 = time.perf_counter()
            preds = np.array([int(m.run(f).argmax()) for f in x])
            dt = (time.perf_counter() - t0) / len(x)
        return dt, f"native ({native.build_info()})", preds
    m = Sequential.load(model_path)
    int8 = "int8" in model_path.name
    t0 = time.perf_counter()
    preds = np.array([int(m.predict(f[None], int8=int8)[0]) for f in x])
    dt = (time.perf_counter() - t0) / len(x)
    return dt, "numpy", preds


def stress(model_path: Path, profile: DeviceProfile, fps: float, x: np.ndarray, y: np.ndarray,
           f32_preds: np.ndarray | None = None) -> StressResult:
    fits, needed, err = check_ram(model_path, profile)
    seq = Sequential.load(model_path)
    macs = seq.macs()
    dev_s = profile.compute_seconds(macs)
    period = 1.0 / fps
    max_fps = 1.0 / dev_s if dev_s > 0 else float("inf")
    load = dev_s / period

    # Deadline simulation: frames arrive every `period`; a single core serves them FIFO.
    misses = 0
    t_free = 0.0
    for i in range(len(x)):
        t_arrive = i * period
        t_start = max(t_arrive, t_free)
        t_free = t_start + dev_s
        if t_free > t_arrive + period:
            misses += 1

    host_s, engine, preds = run_frames(model_path, x, y)
    acc = float((preds == y).mean())
    agreement = float((preds == f32_preds).mean()) if f32_preds is not None else float("nan")
    return StressResult(profile.name, fits, needed, profile.ram_bytes, err, macs, dev_s * 1e3, max_fps, load, misses,
                        host_s * 1e3, macs / host_s / 1e9 if host_s > 0 else float("nan"), engine, acc, agreement)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help=".swm brain to test")
    ap.add_argument("--device", default="mcu-m4", help=f"device profile; known: {sorted(DEVICE_PROFILES)}")
    ap.add_argument("--sweep", action="store_true", help="test against every device profile")
    ap.add_argument("--fps", type=float, default=10.0, help="required frame rate on the device")
    ap.add_argument("--frames", type=int, default=200, help="frames to actually run on the host")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    if not args.model.is_file():
        print(f"{args.model} not found - train one first: python -m swarm.train.train_tiny_cnn", file=sys.stderr)
        return 2

    seq = Sequential.load(args.model)
    size = seq.input_shape[-1]
    ds = make_dataset(args.frames, size, seed=args.seed)

    f32_preds = None
    sibling = args.model.with_name(args.model.name.replace("int8", "f32"))
    if "int8" in args.model.name and sibling.is_file():
        _, _, f32_preds = run_frames(sibling, ds.x, ds.y)

    print(f"model : {args.model}  ({seq.macs()} MACs/frame, {seq.num_params()} params)")
    print(f"engine: {'native ' + native.build_info() if native.available() else 'numpy (C++ engine not built)'}")
    print(f"fps   : {args.fps}  frames: {args.frames}")
    print()
    print(HEADER)
    print("-" * len(HEADER))
    profiles = list(DEVICE_PROFILES.values()) if args.sweep else [get_profile(args.device)]
    worst_exit = 0
    for p in profiles:
        r = stress(args.model, p, args.fps, ds.x, ds.y, f32_preds)
        print(r.row())
        if not r.fits_ram or r.deadline_misses:
            worst_exit = 1
        if r.ram_error and not r.fits_ram:
            print(f"    -> {r.ram_error}")
    print()
    print("ram = can the C++ engine load the brain under a hard arena cap of the device's RAM;")
    print("dev ms/fr = estimated compute time per frame on that device; misses = frames that would miss the")
    print(f"{1000/args.fps:.0f} ms deadline; host ms = measured time per frame here; i8=f32 = prediction agreement "
          "with the f32 model.")
    return worst_exit


if __name__ == "__main__":
    raise SystemExit(main())
