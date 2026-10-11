"""``python -m swarm.sdk new``: package skeletons that validate out of the box.

* ``perception-python`` - a plain numpy class (no ``swarm`` import, so it can ship to a drone)
* ``perception-swm``    - ``train.py`` that defines, trains and compiles a network to ``model.swm``
* ``tactic``            - a ``Tactic`` that starts as the built-in striped design, every decision marked
"""

from __future__ import annotations

import json
from pathlib import Path

from .packages import NAME_RE, SCHEMA, PackageError

KINDS = ("perception-python", "perception-swm", "tactic")

PYTHON_BRAIN = '''"""__TITLE__ - a Python perception brain for swarmBalancer.

The simulator calls classify() on every frame a drone's camera takes; the device runtime calls it too
after `python -m swarm.sdk export`. It needs only numpy - keep `swarm` imports out of this file so it
stays deployable.
"""

import numpy as np


class __CLASS__:
    input_shape = (1, 32, 32)  # (channels, height, width) the camera frame is resized to
    classes = ("nothing", "vehicle", "building", "road", "aircraft")  # index 0 = nothing here
    macs = 4_096  # honest worst-case multiply-accumulates per frame; the simulator charges this per device

    def __init__(self, seed=0, options=None):
        options = dict(options or {})
        self.bright = float(options.get("bright", 0.72))  # background terrain never gets this bright...
        self.dark = float(options.get("dark", 0.17))      # ...or this dark

    def classify(self, image):
        """image: float32 array (1, H, W) in [0, 1]. Returns (class index, confidence 0..1)."""
        x = image[0]
        bright = int((x > self.bright).sum())
        dark = int((x < self.dark).sum())
        if max(bright, dark) < 4:
            return 0, 0.8
        if dark > bright:
            return 2, 0.6  # TODO: tell roads (long, thin) from buildings (square, dark rim)
        return 1, 0.6      # TODO: tell aircraft (cross) from vehicles (small rectangle)
'''

SWM_TRAIN = '''"""Defines, trains and compiles __TITLE__ into model.swm - the file that flies.

    python train.py

The network uses swarm.brain layers (numpy, the same maths as the C++ engine), is trained on the
synthetic overhead imagery, quantised to int8 and written as model.swm. The C++ engine runs exactly
this file on every target - Cortex-M MCUs included - so for a .swm brain "compile" means "export".
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

SIZE = 32      # input is SIZE x SIZE, one channel
EPOCHS = 2


def build(seed=0):
    rng = np.random.default_rng(seed)
    return Sequential(
        [
            Conv2D(1, 6, k=3, stride=1, pad=1, rng=rng), ReLU(), MaxPool2D(2),
            Conv2D(6, 12, k=3, stride=1, pad=1, rng=rng), ReLU(), MaxPool2D(2),
            Flatten(),  # the split point: central/features ships this tensor (int8) to the leader
            Dense(12 * (SIZE // 4) ** 2, 24, rng=rng), ReLU(),
            Dense(24, NUM_CLASSES, rng=rng),
        ],
        input_shape=(1, SIZE, SIZE),
    )


def main():
    train_ds = make_dataset(2500, SIZE, seed=1)
    test_ds = make_dataset(500, SIZE, seed=2)
    model = build()
    print(model.summary())
    train(model, train_ds, test_ds, epochs=EPOCHS, batch_size=32, lr=2e-3, seed=0)
    out = model.save(HERE / "model.swm", quantize=True)
    print(f"int8 test accuracy {accuracy(model, test_ds.x, test_ds.y, int8=True):.3f} -> {out}")


if __name__ == "__main__":
    main()
'''

TACTIC = '''"""__TITLE__ - a coordination tactic for swarmBalancer.

It starts as a copy of the built-in "striped" design: only the leader's camera runs, every frame goes
to one live member, and workers speak up only when they found something. Change _pick_worker() (or
anything else), then: python -m swarm.sdk validate <this folder>
"""

from swarm.sdk import Tactic


class __CLASS__(Tactic):
    strategy = "__NAME__"
    summary = "one camera (the leader's); frames round-robin over live members; workers report only detections"
    camera_mode = "single"  # one camera, fed at the scenario's striped fps

    def on_start(self):
        self._next = 0
        self._ensure_role()

    def rebalance(self, node_id, alive):
        self._ensure_role()  # whoever leads now runs the camera

    def _ensure_role(self):
        if self.is_leader:
            self.start_camera()
        else:
            self.stop_camera()

    def on_capture(self, frame):
        self.preprocess(frame, self._dispatch)  # pay the downscale on the camera drone

    def _dispatch(self, frame):
        if not self.is_leader:
            return  # demoted while pre-processing; the new leader's camera has taken over
        target = self._pick_worker()
        if target == self.id:
            self.run_brain(self.brain.macs, frame, self._done)
        else:
            self.send_frame(target, frame)

    def _pick_worker(self):
        members = self.live_members()
        target = members[self._next % len(members)]
        self._next += 1
        return target

    def on_frame(self, src, msg):
        self.run_brain(self.brain.macs, msg.payload, self._done)

    def _done(self, frame, pred, conf):
        if self.useful(pred, conf):
            self.report_to_leader(frame, pred, conf)
        elif self.is_leader:
            self.leader_learns(frame, pred, conf)
        # nothing found on a worker: stay silent and save the radio
'''

README = '''# __TITLE__

__KIND__ package for swarmBalancer, created with `python -m swarm.sdk new`.

```
python -m swarm.sdk validate __PATH__     # conformance checks (run before uploading)
python -m swarm.sdk pack __PATH__         # -> __NAME__.zip for the studio's ARMORY upload
```

See docs/SDK.md for the interface and the rules.
'''


def _camel(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_") if p) or "Brain"


def new_package(directory: Path, name: str, kind: str, title: str = "", author: str = "") -> Path:
    """Creates ``directory/name`` with a manifest and a working skeleton; refuses to overwrite."""
    if not NAME_RE.match(name):
        raise PackageError(f"name {name!r} must match {NAME_RE.pattern}")
    if kind not in KINDS:
        raise PackageError(f"kind must be one of {KINDS}")
    root = Path(directory) / name
    if root.exists():
        raise PackageError(f"{root} already exists")
    title = title or name.replace("_", " ").title()
    cls = _camel(name)
    manifest: dict = {"schema": SCHEMA, "name": name, "version": "0.1.0", "title": title, "author": author,
                      "description": ""}
    files: dict[str, str] = {}
    if kind == "perception-python":
        manifest.update(type="perception", format="python", entry=f"brain.py:{cls}",
                        description="Python perception brain (numpy only, deployable to Linux companions).")
        files["brain.py"] = PYTHON_BRAIN
    elif kind == "perception-swm":
        manifest.update(type="perception", format="swm", model="model.swm", build="train.py",
                        classes=["nothing", "vehicle", "building", "road", "aircraft"],
                        description="Compiled neural network; run train.py to build model.swm.")
        files["train.py"] = SWM_TRAIN
    else:
        manifest.update(type="tactic", entry=f"tactic.py:{cls}", strategy=name,
                        description="Coordination tactic (simulation).")
        files["tactic.py"] = TACTIC
    files["README.md"] = README.replace("__KIND__", kind)
    root.mkdir(parents=True)
    for rel, text in files.items():
        text = (text.replace("__TITLE__", title).replace("__CLASS__", cls).replace("__NAME__", name)
                .replace("__PATH__", root.as_posix()))
        (root / rel).write_text(text, encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root
