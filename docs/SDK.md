# Brain & Tactic SDK (`swarm.sdk`)

The SDK is the library for writing code the swarm runs: code you can **load into the test bench**
(`python -m swarm.studio`), run in **the simulator and the standardized battery**, and **ship to the
drones** (compiled networks).

| You write | It decides | Runs in the simulator | Runs on a drone |
|---|---|---|---|
| **Arena policy** (Python, or a trained mesh network `.npz`) | how each drone of a recon swarm, interceptor wall or attacking swarm moves, aims and talks - the RECON / DEFENCE / OFFENCE tests ([ARENA.md](ARENA.md)) | yes, in the arena and the test bench | mesh networks: **every device**, as `.swm` in the C++ engine; Python: Linux companions with your glue |
| **Perception brain, `.swm` format** | what one drone concludes from one camera frame | yes (C++ engine or numpy) | **every device it fits**, MCUs included: the C++ engine executes the same `.swm` |
| **Perception brain, Python format** | the same, in plain Python + numpy | yes | Linux companions (Pi-class) through the device runtime |
| **Tactic** | how the swarm shares the inference work: who looks, who thinks, who tells whom | yes, next to `local` / `central` / `striped` | no - a tactic is a model of coordination logic; see [deploying](#getting-brains-onto-drones) |

Everything is a **package**: a folder (or a `.zip` of it) with `manifest.json` at its root.

## Quick start

```powershell
python -m swarm.sdk new my_wall --type policy-defender     # -> brains/my_wall/policy.py (a working smart wall)
python -m swarm.sdk validate brains/my_wall                # runs a DEFENCE test, ranks it against the built-ins
python -m swarm.arena test defence --defender lib:my_wall  # or LOAD it in the studio's DEFENCE tab

python -m swarm.sdk new my_eye --type perception-python    # -> brains/my_eye/ (validates out of the box)
python -m swarm.sdk new my_net --type perception-swm       # -> brains/my_net/train.py compiles model.swm
python -m swarm.sdk new my_plan --type tactic              # -> brains/my_plan/tactic.py (a copy of striped)
python brains/my_net/train.py                              # "compile": train + export the .swm
python -m swarm.sdk pack brains/my_plan                    # -> my_plan.zip
```

In the test bench, **LOAD** (next to every brain list) installs an arena brain - a policy `.py`, trained
`.npz` weights or a package `.zip` - checks it and selects it; the **BRAINS** drawer checks, downloads,
shows and deletes them. Perception brains and tactics are used from the command line and the battery.
Worked examples live in [`examples/brains/`](../examples/brains):

| package | type | what it shows |
|---|---|---|
| `pincer_attack` | policy / python (attacker) | radio-coordinated two-flank attack under the terrain mask: breaks a zone defence (60 % through), not a wall that shares what it sees |
| `mesh_defender`, `mesh_attacker` | policy / npz | networks evolved against each other in the 3-D arena |
| `mesh_recon` | policy / npz (recon) | a recon swarm evolved for 3000 generations: beats the hand-written lawnmower overall - same coverage and contact, half the sensing energy |
| `edge_detector` | perception / python | a classical-CV brain (no network): 98 % on the synthetic world, ~4 operations per pixel, deployable to Linux companions |
| `micro_cnn` | perception / swm | a 20 KB int8 CNN for Cortex-M4 drones; `train.py` compiles it |
| `load_aware_stripe` | tactic | a stripe that deals frames by member speed - wins the s11 heterogeneous yardstick (busiest CPU 3 % vs 100 % for round-robin) |

## Packages

```
my_brain/
  manifest.json     required - what this is and how to load it
  brain.py          python brains: the class named in "entry"   (policies: policy.py by convention)
  model.swm         swm brains: the compiled network named in "model"
  brain.npz         npz policies: the trained mesh weights named in "model"
  train.py          swm brains, optional: the "build" script that produces model.swm
  README.md         optional
```

`manifest.json`:

| key | required | meaning |
|---|---|---|
| `schema` | yes | always `"swarm.package/1"` |
| `name` | yes | lower-case identifier, 2-40 characters (`[a-z][a-z0-9_]+`), unique in the library |
| `version` | no | text, e.g. `"1.0.0"` |
| `type` | yes | `"policy"`, `"perception"` or `"tactic"` |
| `format` | policy, perception | policy: `"python"` or `"npz"`; perception: `"python"` or `"swm"` |
| `role` | policy | `"recon"`, `"defender"` or `"attacker"` |
| `entry` | python, tactic | `"file.py:ClassName"` - a file inside the package and the class to load |
| `model` | swm, npz | path of the compiled network / trained weights inside the package, e.g. `"model.swm"` |
| `build` | no (swm) | script that (re)compiles the model |
| `classes` | no | label names; swm files store only the class count |
| `options` | no (python) | JSON object handed to the constructor |
| `strategy` | no (tactic) | the name scenarios use; defaults to `name` |
| `title`, `description`, `author` | no | shown in the brain lists |

Paths always use `/`, stay inside the package, and are never absolute. Reading a manifest never runs code.

Each package is imported under its own module namespace, so two packages may both have a `brain.py`.
Between files of one package use **relative imports** (`from .features import edges`).

## Arena policies

A policy flies one side of an arena test: a class with `role` and `act(obs)`, where `obs` holds what every
drone of that side senses and hears, for every episode at once (numpy arrays `[B, N, ...]`, positions in
3-D), and `act` returns the actions (`{"target": ...}` for defenders, `{"heading": ..., "speed": ...}` for
attackers, `{"direction": ..., "speed": ..., "camera": ...}` for recon; optionally `"altitude"` for
attackers and recon drones, `"station"` - where each interceptor waits, e.g. pickets far forward - for
defenders, and `"msg"` radio channels). Optional:
`reset(batch, seed)`, `title`, `summary`, `macs` (compute per decision, for the CPU figure), `comms`.
[ARENA.md](ARENA.md#writing-a-brain) lists every observation field. A trained mesh network (`.npz` from
`python -m swarm.arena train` or the studio's TRAIN tab) is a policy too: `format: "npz"`, `model:
"brain.npz"` - `python -m swarm.sdk install weights.npz` wraps one into a package.

## Perception brains

### The contract

* **Input** - one camera frame as the simulator delivers it: `float32` array, shape `(1, H, W)`
  (one channel, square), values in `[0, 1]`. Cameras are downscaled to your input size for you - that cost
  is charged to the swarm, not to the brain.
* **Output** - `(class_index, confidence)`. Class **0 means "nothing here"**; every other class is a
  detection, and a detection is *reported* when `confidence >= confidence_threshold` (scenario setting,
  default 0.5). Confidence must be a finite number in `[0, 1]`.
* **Cost** - `macs`: multiply-accumulates per frame. The simulator turns it into time on each device
  (`DeviceProfile.compute_seconds`) and energy; it is the brain's whole footprint in the simulation.
  `.swm` networks are measured; Python brains **declare** it, so declare an honest worst case.
* **Deterministic** - the same frame gives the same answer, in every instance. Seed randomness from the
  constructor's `seed`.

The simulated world contains five classes - `nothing, vehicle, building, road, aircraft`
(`swarm.train.dataset`). A brain with those five labels gets a meaningful accuracy; other label sets
still run (accuracy then compares class indices).

### Python format

```python
import numpy as np


class MyEye:
    input_shape = (1, 32, 32)                       # (channels, height, width)
    classes = ("nothing", "vehicle", "building", "road", "aircraft")
    macs = 4096                                     # declared cost per frame
    frontend_fraction = 0.75                        # optional: share of macs before the split point
    ram_bytes = 16384                               # optional: declared working memory

    def __init__(self, seed=0, options=None):       # both optional; options come from the manifest
        self.threshold = (options or {}).get("threshold", 0.7)

    def classify(self, image):                      # float32 (1, H, W) in [0, 1]
        bright = int((image[0] > self.threshold).sum())
        return (1, 0.8) if bright > 6 else (0, 0.9)
```

`swarm.sdk.PerceptionBrain` is an optional base class with these attributes, but a brain that should
also fly **must not import `swarm`**: the device payload carries only the runtime. The validator warns when
it finds such an import, and lists every third-party module the device would have to provide.

### `.swm` format - "compile" means export

A `.swm` file is a network the C++ engine runs as-is, on the PC and on the drone ([format](MODEL_FORMAT.md)).
Define it with the numpy layers in `swarm.brain` (same weight layouts and int8 arithmetic as the engine),
train it, and save it - the template's `train.py` does all three:

```python
from swarm.brain import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential
from swarm.train.dataset import make_dataset
from swarm.train.train_tiny_cnn import train

model = Sequential([Conv2D(1, 6, k=3, stride=1, pad=1), ReLU(), MaxPool2D(2),
                    Flatten(),                       # split point used by central/features
                    Dense(6 * 16 * 16, 5)], input_shape=(1, 32, 32))
train(model, make_dataset(2500, 32, seed=1), make_dataset(500, 32, seed=2), epochs=2, batch_size=32, lr=2e-3, seed=0)
model.save("model.swm", quantize=True)               # int8 weights: ~4x smaller, NEON int8 GEMM on the drone
```

Available layers: `Conv2D`, `Dense`, `ReLU`, `MaxPool2D` (k == stride), `Flatten`, final softmax (added
on export). The validator loads the file in the C++ engine **under each device's RAM cap** - that number is
the RAM the device must provide, not an estimate - and prints time per frame per device.

### How the simulator uses a brain

Each swarm member's camera produces frames at the scenario's fps; *where* the brain runs on them is the
tactic's decision (on the camera drone, on the leader, on a worker). Each run of the brain books `macs`
on that member's CPU (a single FIFO core; a full queue rejects work). `central/features` runs the part
before `Flatten` on the drone and ships the int8 feature map; Python brains approximate that split with
`frontend_fraction` and `feature_bytes`.

## Tactics

A tactic is the coordination logic of **one** swarm member; every member runs its own instance and they
talk only over the shared radio channel. Everything happens in simulated time, driven by events.

```python
from swarm.sdk import Tactic


class Relay(Tactic):
    strategy = "relay"                  # name used in scenarios (not a built-in name)
    summary = "every camera on; frames processed where they were taken; detections go to the leader"
    camera_mode = "all"                 # "single": one camera, fed at the scenario's striped fps

    def on_start(self):                 # boot, and again after coming back from a failure
        self.start_camera()

    def on_capture(self, frame):        # this member's camera took a frame
        self.preprocess(frame, self._think)        # pay the downscale (every design does)

    def _think(self, frame):
        self.run_brain(self.brain.macs, frame, self._done)

    def _done(self, frame, pred, conf):
        if self.useful(pred, conf):
            self.report_to_leader(frame, pred, conf)
        elif self.is_leader:
            self.leader_learns(frame, pred, conf)
```

### Hooks

| hook | called when |
|---|---|
| `on_start()` | the member boots, and again after a revive - decide whether its camera runs |
| `on_capture(frame)` | **required** - its camera produced a frame |
| `on_frame(src, msg)` | a message of kind `"frame"` arrived; any kind `k` you send goes to `on_k(src, msg)` |
| `on_result(src, msg)` | a detection report arrived (default: the leader learns it) |
| `rebalance(node_id, alive)` | a peer was declared dead (missed heartbeats) or came back |
| `on_killed()` / `on_revived()` | call `super()` if you override them |

Heartbeats, failure detection and leader election are done for you, identically for every design: a peer
silent for `missed_beats x hb_interval` is suspected, and every member deterministically agrees the leader
is the live member with the most compute.

### Helpers

| helper | does |
|---|---|
| `is_leader`, `leader_id`, `live_members()` | this member's view of the swarm |
| `peer_load(i)`, `node_speed(i)` | a member's queue fill from its last heartbeat; its MACs per second |
| `now`, `rng` | simulated time; a per-member seeded `random.Random` |
| `start_camera()`, `stop_camera()` | camera on / off |
| `preprocess(frame, then, *args)` | books the camera->brain-input downscale here, then calls `then(frame, *args)` |
| `run_brain(macs, frame, then, *args)` | books `macs` here, runs the brain, calls `then(frame, pred, conf, *args)` |
| `compute_time(macs, i)`, `cpu_free_at()`, `airtime(nbytes)` | cost estimates for scheduling decisions |
| `send(dst, kind, payload, size)`, `broadcast(kind, payload, size)` | radio; `size` bytes of airtime are charged |
| `send_frame(dst, frame)` | ships a downscaled frame (`input_bytes`) - the receiver's `on_frame` gets it |
| `useful(pred, conf)` | `pred != 0 and conf >= threshold` |
| `report_to_leader(frame, pred, conf)`, `leader_learns(frame, pred, conf)` | the mission's scorekeeping: the moment the leader knows |
| `self.brain`, `self.cfg`, `self.profile` | the brain's costs, the scenario config, this member's device |

### Rules

1. **Pay the physics.** Every frame is shrunk to the brain input somewhere (`preprocess`), and every
   processed frame runs the full brain (`run_brain(self.brain.macs, ...)`). The validator checks that the
   swarm spent at least *brain + downscale* MACs per processed frame - a tactic that skips costs the
   built-in designs pay is rejected.
2. **Move data only over the radio.** Never call another member's methods or read its state; use
   `send` / `broadcast` (and heartbeat-carried `peer_load`).
3. **Be deterministic.** Use `self.rng`, never wall-clock time or unseeded randomness. The validator runs
   everything twice.
4. **Survive.** After losing the leader or a worker, results must reach a leader again - handle
   `rebalance` and `on_start`.

### Walkthrough: the load-aware stripe

The built-in `striped` design deals frames round-robin, so on a mixed swarm the Cortex-M4s get the same
share as the Cortex-A72 and drown (scenario `s11_heterogeneous`). [`load_aware_stripe`](../examples/brains/load_aware_stripe/tactic.py)
keeps everything else and replaces one decision - who gets the next frame - with a smooth weighted
round-robin whose weights are `node_speed(m)`, skipping members whose heartbeat reports a nearly full queue.
Result on the full s11 preset (20 s): busiest CPU 3 % instead of 100 %, no rejected frames instead of 199,
95 % recall instead of 87 %, detection latency 0.3 ms instead of 8 ms, and about half the radio traffic -
because the A72 now takes most frames itself instead of shipping them to microcontrollers.

### Using a tactic in a preset

```python
from swarm.bench import Scenario, Swarm
from swarm.sdk import use_package

use_package("examples/brains/load_aware_stripe")      # registers strategy "load_aware"

SCENARIO = Scenario(name="s13_mixed_swarm", swarm=Swarm(nodes=("drone-a72", "drone-a53", "mcu-m4")),
                    strategies=("local", "striped", "load_aware"), checks=(...))
```

A brain package goes into `BrainSpec(kind="package", model="examples/brains/edge_detector")`
(`kind="auto"` also recognises a package folder).

## Validation

`python -m swarm.sdk validate PATH` (or CHECK in the studio's BRAINS drawer) prints a report of findings -
`PASS`, `INFO`, `WARN`, `FAIL` - and exits 1 if anything failed. Brains loaded in the studio are validated
automatically.

| perception brain | tactic |
|---|---|
| manifest, file paths, Python syntax | manifest, Python syntax |
| loads; required attributes; declared cost > 0 | subclasses `Tactic`; `on_capture` implemented; name free |
| input `(1, H, W)`, square, 8-512 px; >= 2 classes; class 0 is "nothing" | registers as a strategy |
| 96 synthetic frames give valid `(class, confidence)` | smoke run on the reference swarm |
| determinism (repeat and fresh instance) | determinism (two identical runs) |
| accuracy, object recall, false alarms on the synthetic world | physics: MACs per processed frame >= brain + downscale |
| host speed | leader lost at 4 s, worker lost at 4 s: results flow again |
| device table: ms/frame, RAM fit (C++ engine, under each cap), runtime | s11 yardstick vs striped (informational) |
| portability: `swarm` imports, third-party dependencies | |

An **arena policy** must load, return valid actions through a short test of its role (DEFENCE, OFFENCE or
RECON with default settings), and give the same result twice for the same seed. It is then ranked against
the built-in brains of its role on the same situations, and a device table shows the time per decision and
its share of each board at the test's decision rate. Python policies are marked as needing Linux; mesh
networks run in the C++ engine everywhere.

## Installing packages

`python -m swarm.sdk install FILE` (and the studio's LOAD, for arena brains) accepts:

* **`.zip`** - a package, with `manifest.json` at the archive root or inside one top folder
  (`python -m swarm.sdk pack` produces this);
* **`.npz`** - trained mesh weights; they are wrapped into a policy package (role from the weights);
* **`.swm`** - a compiled network; it is wrapped into a perception package;
* **`.py`** - one file with an arena policy (a class with `act()` and `role = "..."`), a `Tactic` subclass
  or a class with a `classify` method; the manifest is written from the file *without running it* (a
  tactic's `strategy = "..."` becomes its name).

Installed packages land in `brains/` (git-ignored except its README). Archives are checked before anything
is written: no absolute paths, `..`, links or reserved device names, bounded file count, size and compression
ratio. **Package code is trusted code**: validating and simulating it runs it on your PC with your user's
rights (in a separate worker process, not a sandbox) - only load code you would run as a script.

## Getting brains onto drones

`python -m swarm.sdk export PATH --out DIR` writes the device-side artefacts and a `DEPLOY.txt` with the
exact next steps. Nothing is signed, copied or executed - that stays the operator's explicit, audited flow
in [DEPLOYMENT.md](DEPLOYMENT.md).

* **Mesh policy (`.npz`)** -> the network as `.swm` (Dense -> ReLU -> Dense) for the C++ engine on any
  board, plus its input layout (every feature by name and scale) and how to decode the outputs. The glue
  that builds those inputs from real sensors and the radio, and turns the decision into autopilot set
  points, is per-airframe integration work; `swarm/arena` is its reference implementation.
* **Python policy** -> refused: it runs in the simulator; train a mesh network for the same role to fly.
* **`.swm` brain** -> the network file. On a Linux companion it runs in the C++ engine through the
  runtime's native handler (`setup_device.py prepare --model ... --library <target-built swarm_brain.so>`);
  on a microcontroller you link `swarm_core` (`cpp/`) with the board toolchain and load the same file
  under the board's RAM cap. This is the path for MCU-class drones.
* **Python brain** -> a `--brain-dir` folder: your package as module `<name>_brain` plus a generated
  `_device.py` whose `make_handler(options)` is the runtime factory (`"<name>_brain._device:make_handler"`).
  It feeds your `classify` the uint8 CHW observation scaled to `[0, 1]` and answers like the native handler
  (8 bytes: class, confidence - or empty when nothing useful). The device needs Python and numpy.
* **Tactics** are not exported. The device runtime ([RUNTIME.md](RUNTIME.md)) has its own bounded load
  balancer with authenticated peers; a tactic that wins in simulation is the specification for changing it,
  not code that flies.

## Command line

```
python -m swarm.sdk new NAME --type {policy-defender,policy-attacker,policy-recon,perception-python,perception-swm,tactic} [--dir brains] [--title T]
python -m swarm.sdk validate PATH [--json REPORT.json]
python -m swarm.sdk pack PATH [-o FILE.zip]
python -m swarm.sdk install FILE.zip|FILE.npz|FILE.swm|FILE.py|FOLDER [--replace]
python -m swarm.sdk export PATH --out NEW_DIR [--target linux-aarch64]
python -m swarm.sdk list
```

## Python API

```python
from swarm.sdk import load_package, use_package, validate, discover_packages, Tactic, PerceptionBrain

pkg = load_package("examples/brains/micro_cnn")    # reads the manifest only
brain = pkg.make_brain(seed=0)                      # a simulator Brain (runs package code)
policy = load_package("examples/brains/pincer_attack").make_policy()   # an arena brain
use_package("examples/brains/load_aware_stripe")    # registers a tactic's strategy
report = validate("brains/my_plan")                 # .ok, .status, .findings, .info, .render_text()
```

## Troubleshooting

* *"unknown strategies ['my_plan']"* - load the package before building the `Scenario` (`use_package`).
* *"strategy 'x' is already provided by ..."* - two packages declare the same strategy name; rename one.
* *physics FAIL* - a frame was processed without `preprocess` (downscale) or with fewer than `brain.macs`.
* *determinism FAIL* - randomness not from `self.rng` / the constructor seed, or iteration over a `set`.
* *"imports swarm.*"* warning - fine for simulation; drop the import to deploy the brain.
* *swm "not built"* - run the package's `train.py`.
* *"built for a different ... observation layout"* - mesh weights from an older version of the arena (for
  example the 2-D one); train them again (the inputs a network was trained on cannot change under it).
* *a policy's `act()` "is missing" / "has shape" / "NaN"* - the actions are checked at every decision;
  return arrays shaped `[episodes, drones]` (scalars are broadcast) with finite values.
