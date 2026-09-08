"""Workstation setup: select local brain code/model, assemble a minimal device payload, export USB.

The old training/sim/benchmark toolkit is deliberately NOT copied to a robot.
Credentials remain outside the application. Custom packages are trusted code,
not sandboxed. Nothing is installed, flashed, executed or armed by preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import queue
import shutil
import threading
from pathlib import Path

from ._common import (MAX_FILE_BYTES, DeployError, absolute, canonical_json, private_directory,
                      read_regular, relative_path, target_name, write_new)
from .bundle import _payload_content, _source_files, export_usb, pack


def prepare(output: Path, *, config: Path, target: str, brain_dir: Path | None = None,
            model: Path | None = None, library: Path | None = None, ram_cap_bytes: int = 8 * 1024 * 1024) -> Path:
    """Assemble runtime + SDK + selected plugin/model. Destination MUST be new.

    A custom ``brain_dir`` is the root of reviewed Python importable packages,
    copied under payload/python, not pip-installed. Vendor packages and native
    dependencies must already be prepared for the target by the operator.
    """
    from swarm.runtime.config import ApplicationConfig, read_json

    target_name(target)
    output = absolute(output)
    ApplicationConfig.load(config, expand_environment=False)  # No TLS keys are read on the workstation.
    data = read_json(config)
    source_root = Path(__file__).resolve().parents[1]
    selected: dict[str, bytes] = {}
    for package in ("runtime", "robotics"):
        if not (source_root / package / "__init__.py").is_file():
            raise DeployError("preparation needs the reviewed runtime and robotics SDK installed on the workstation")
        for path in sorted((source_root / package).glob("*.py")):
            selected[f"python/swarm/{package}/{path.name}"] = read_regular(path, MAX_FILE_BYTES)
    selected["python/swarm/__init__.py"] = b'"""Device-side swarm coordination and local robot interfaces."""\n'
    if brain_dir is not None:
        if absolute(brain_dir) == output or absolute(brain_dir) in output.parents:
            raise DeployError("output must be outside the selected brain directory")
        records, contents = _source_files(brain_dir, target)
        for record, content in zip(records, contents):
            if record.path.split("/", 1)[0].lower() == "swarm":
                raise DeployError("custom brains must not replace the swarm runtime namespace")
            selected[f"python/{record.path}"] = content
    if (model is None) != (library is None):
        raise DeployError("a native .swm model and its TARGET-built library must be selected together")
    if model is not None and library is not None:
        if target == "python-any":
            raise DeployError("native inference requires an exact OS/architecture target")
        weights = read_regular(model, MAX_FILE_BYTES)
        binary = read_regular(library, MAX_FILE_BYTES)
        if not weights.startswith(b"SWM1"):
            raise DeployError("selected model is not a SWM1 brain")
        relative_path(library.name)
        _payload_content(library.name, binary, target)
        digest = hashlib.sha256(weights).hexdigest()
        options = {"model": "models/brain.swm", "library": f"native/{library.name}",
                   "sha256": digest, "ram_cap_bytes": ram_cap_bytes, "threshold": 0.5, "empty_class": 0}
        if type(ram_cap_bytes) is not int or not 0 < ram_cap_bytes <= 1024 * 1024 * 1024:
            raise DeployError("ram_cap_bytes must be positive and <= 1 GiB")
        data["brain"] = {"factory": "swarm.runtime.handlers:make_native_handler", "options": options}
        data["node"]["workload_id"] = "swm-v1-" + digest
        selected["models/brain.swm"] = weights
        selected[f"native/{library.name}"] = binary
    selected["node.json"] = canonical_json(data)
    selected["main.py"] = (
        '"""Inference-only device application. No robot hardware is armed or imported."""\n'
        "from pathlib import Path\nfrom swarm.runtime.cli import main\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main(default_config=Path(__file__).parent / 'node.json'))\n"
    ).encode("utf-8")
    # Validate everything before creating a destination. Installer repeats validation after authentication.
    for name, content in selected.items():
        relative_path(name)
        _payload_content(name, content, target)
    root = private_directory(output, new=True)
    try:
        for name, content in selected.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            write_new(path, content)
        _source_files(root, target)  # Apply exactly the bundle's size/count/ABI rules to the assembled tree.
    except BaseException:
        shutil.rmtree(root)
        raise
    return root


def gui() -> int:
    """Optional stdlib desktop UI. Background packing never reads widget state from its thread."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise DeployError("desktop UI unavailable; use setup_device.py prepare or python -m swarm.deploy") from exc
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise DeployError("desktop UI unavailable; use setup_device.py prepare or python -m swarm.deploy") from exc
    root.title("swarmBalancer — device setup / USB transfer")
    root.minsize(880, 700)
    panel = ttk.Frame(root, padding=18)
    panel.pack(fill="both", expand=True)
    panel.columnconfigure(1, weight=1)
    ttk.Label(panel, text="Prepare a brain, sign and encrypt it for ONE device", font=("Segoe UI", 14, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
    ttk.Label(panel, text="No flashing, formatting, SSH or automatic launch. A booted Pi installs the verified USB file locally.").grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))
    values = {}
    row = 2

    def field(key, label, default="", browse=None):
        nonlocal row
        var = tk.StringVar(value=default)
        values[key] = var
        ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(panel, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
        if browse:
            def choose():
                if browse == "dir":
                    path = filedialog.askdirectory()
                elif browse == "save":
                    path = filedialog.asksaveasfilename(defaultextension=".swarmbundle")
                else:
                    path = filedialog.askopenfilename()
                if path:
                    var.set(path)
            ttk.Button(panel, text="Browse…", command=choose).grid(row=row, column=2, padx=(8, 0))
        row += 1

    field("config", "Local node configuration", browse="file")
    field("brain_dir", "Custom brain package root (optional)", browse="dir")
    field("model", "Selected .swm model (optional)", browse="file")
    field("library", "Target-built native library (with .swm)", browse="file")
    field("application", "NEW prepared application directory", browse="dir")
    field("target", "Exact target (Pi 64-bit default)", "linux-aarch64")
    field("node_id", "Recipient node ID", "pi-01")
    field("application_id", "Application ID", "main-brain")
    field("version", "Increasing release version", "1")
    field("signing_key", "Offline signing PRIVATE key path", browse="file")
    field("recipient", "Approved public enrollment request/key", browse="file")
    field("trust_key", "Independently trusted signing PUBLIC key", browse="file")
    field("output", "NEW encrypted bundle file", browse="save")
    field("usb", "Mounted USB folder (optional)", browse="dir")
    status_text = tk.StringVar(value="Preparation needs a new folder name. Browse its parent, then append a new name.")
    ttk.Label(panel, textvariable=status_text, wraplength=820).grid(row=row + 1, column=0, columnspan=3, sticky="w", pady=12)
    buttons = ttk.Frame(panel)
    buttons.grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
    busy = False
    completed: queue.SimpleQueue[str] = queue.SimpleQueue()

    def poll():
        nonlocal busy
        try:
            message = completed.get_nowait()
        except queue.Empty:
            pass
        else:
            busy = False
            status_text.set(message)
        root.after(100, poll)

    def close():
        if busy:
            messagebox.showinfo("Operation in progress", "Wait for the current local operation to finish before closing.")
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)

    def operate(action):
        nonlocal busy
        if busy:
            return
        fields = {key: var.get().strip() for key, var in values.items()}
        if action == "bundle" and not messagebox.askyesno(
                "Confirm selected recipient and trust", "Have you independently verified the recipient request/key and signing public key?\n"
                "Only the selected application is encrypted. Private keys must never be copied to USB."):
            return
        busy = True
        status_text.set("Working locally… no device is being executed or armed.")

        def worker():
            try:
                if action == "prepare":
                    result = prepare(Path(fields["application"]), config=Path(fields["config"]), target=fields["target"],
                                     brain_dir=Path(fields["brain_dir"]) if fields["brain_dir"] else None,
                                     model=Path(fields["model"]) if fields["model"] else None,
                                     library=Path(fields["library"]) if fields["library"] else None)
                    message = f"Prepared {result}. Review this directory, then sign/encrypt it. No credentials included."
                else:
                    result = pack(fields["application"], fields["output"], application_id=fields["application_id"],
                                  version=int(fields["version"]), node_id=fields["node_id"], target=fields["target"],
                                  entrypoint="main.py", signing_key=fields["signing_key"], recipient=fields["recipient"])
                    if fields["usb"]:
                        result = export_usb(result, fields["usb"], trust_key=fields["trust_key"])
                    message = f"Encrypted, signed package written to {result}. Installation on the Pi is still required."
            except Exception as exc:
                message = f"Operation refused: {type(exc).__name__}: {exc}"
            completed.put(message)

        threading.Thread(target=worker, daemon=True).start()

    ttk.Button(buttons, text="1. Prepare selected brain", command=lambda: operate("prepare")).pack(side="left", padx=(0, 10))
    ttk.Button(buttons, text="2. Sign, encrypt and export", command=lambda: operate("bundle")).pack(side="left")
    root.after(100, poll)
    root.mainloop()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", help="desktop application/USB file selector (default)")
    build = sub.add_parser("prepare", help="assemble a minimal runtime payload without copying the testing toolkit")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--target", default="linux-aarch64")
    build.add_argument("--brain-dir", type=Path)
    build.add_argument("--model", type=Path)
    build.add_argument("--library", type=Path)
    build.add_argument("--ram-cap-bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "gui"):
            return gui()
        result = prepare(args.output, config=args.config, target=args.target, brain_dir=args.brain_dir,
                         model=args.model, library=args.library, ram_cap_bytes=args.ram_cap_bytes)
        print(f"Prepared inference-only application: {result}")
        print("Review it, then use swarm.deploy pack / export-usb. No dependencies installed or hardware accessed.")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"setup failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())