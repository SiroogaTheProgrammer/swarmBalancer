"""Select, run, or delete trained image-recognition models."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import numpy as np

try:
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None

try:
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    cv2 = None

from swarm.brain.model import Sequential

MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "image_recognition"


def list_models(model_dir: Path = MODELS_DIR) -> list[Path]:
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("*.swm"))


def choose_model(model_dir: Path = MODELS_DIR) -> Path | None:
    models = list_models(model_dir)
    if not models:
        print("No models found in", model_dir)
        return None
    for i, path in enumerate(models, start=1):
        print(f"{i}. {path.name}  ({path.stat().st_size} bytes)")
    choice = input("Select model number to run, or press Enter to cancel: ").strip()
    if not choice:
        return None
    try:
        idx = int(choice) - 1
    except ValueError:
        print("Invalid choice.")
        return None
    if 0 <= idx < len(models):
        return models[idx]
    print("Choice out of range.")
    return None


def delete_model(model_dir: Path = MODELS_DIR) -> Path | None:
    models = list_models(model_dir)
    if not models:
        print("No models available to delete.")
        return None
    for i, path in enumerate(models, start=1):
        print(f"{i}. {path.name}")
    choice = input("Select model number to delete: ").strip()
    if not choice:
        return None
    try:
        idx = int(choice) - 1
    except ValueError:
        print("Invalid choice.")
        return None
    if 0 <= idx >= len(models):
        print("Choice out of range.")
        return None
    target = models[idx]
    target.unlink()
    print(f"Deleted {target.name}")
    return target


def _tk_photo_from_frame(frame: np.ndarray) -> tuple[object, str]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".png", rgb)
    if not ok:
        raise RuntimeError("could not encode frame for popup window")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return tk.PhotoImage(data=data), data


def run_selected(
    model_path: Path | None,
    *,
    camera_index: int = 0,
    resize_to: tuple[int, int] = (64, 64),
    display: bool = True,
    max_frames: int | None = None,
    force: bool = False,
) -> int:
    if model_path is None:
        return 0
    if cv2 is None:
        print("OpenCV is not installed. Install it with: python -m pip install opencv-python")
        return 1
    if display and tk is None:
        print("Tkinter is not available; falling back to console-only output.")
        display = False

    print(f"Running model: {model_path}")
    print(f"Using camera index {camera_index} (front camera default on most laptops).")
    if force:
        print("(forced run mode)")

    model = Sequential.load(model_path)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}. Try --camera-index 1 or --camera-index 0.")
        return 1

    root = None
    label_var = None
    image_label = None
    quit_requested = False
    if display:
        root = tk.Tk()
        root.title("Front Camera Detection")
        root.geometry("900x700")
        label_var = tk.StringVar(value="NN detection: waiting...")
        top_label = tk.Label(root, textvariable=label_var, font=("Arial", 18, "bold"), fg="lime", bg="black")
        top_label.pack(fill="x", pady=(10, 0))
        image_label = tk.Label(root, bg="black")
        image_label.pack(fill="both", expand=True, padx=10, pady=10)

        def stop_loop() -> None:
            nonlocal quit_requested
            quit_requested = True
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", stop_loop)

    last_label = "unknown"
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame read failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = x[None, None, :, :]

        logits = model.forward(x)
        pred = int(np.argmax(logits))
        label = f"class_{pred}"
        last_label = label
        print(f"frame {frames}: {label}", flush=True)

        if display and root is not None:
            try:
                display_frame = frame.copy()
                cv2.putText(display_frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                photo = tk.PhotoImage(data=base64.b64encode(cv2.imencode(".png", display_frame)[1]).decode("ascii"))
                image_label.configure(image=photo)
                image_label.image = photo
                label_var.set(f"NN detects: {label}")
                root.update()
                if quit_requested:
                    break
            except Exception:
                print("GUI popup unavailable on this system; continuing without preview.")
                display = False

        frames += 1
        if max_frames is not None and frames >= max_frames:
            break
        if quit_requested:
            break

    cap.release()
    if root is not None:
        try:
            root.destroy()
        except Exception:
            pass
    print(f"Session ended. Last prediction: {last_label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--run", action="store_true", help="interactively choose a model to run")
    parser.add_argument("--delete", action="store_true", help="interactively choose a model to delete")
    parser.add_argument("--list", action="store_true", help="list available models")
    parser.add_argument("--camera-index", type=int, default=0, help="camera index to use; default 0 is usually the front/integrated camera")
    parser.add_argument("--no-display", action="store_true", help="run model without showing a window")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames; useful for quick smoke tests")
    args = parser.parse_args(argv)

    if args.list:
        models = list_models(args.model_dir)
        if not models:
            print("No models found.")
            return 0
        for p in models:
            print(f"{p.name}  {p.stat().st_size} bytes")
        return 0

    if args.run:
        target = choose_model(args.model_dir)
        return run_selected(target, camera_index=args.camera_index, display=not args.no_display, max_frames=args.max_frames)

    if args.delete:
        delete_model(args.model_dir)
        return 0

    print("Usage: choose --run or --delete, or use --list to inspect model files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
