# Architecture

## Goals

1. Design brains (tiny neural nets) for constrained swarm members and **know** whether they fit: RAM, cycles, deadline.
2. Design **how the swarm shares work** and measure it: radio load, latency, drops, per-node busy time.
3. Watch the swarm **rebalance** when a member disappears or returns.
4. Do all of it on the development PC, deterministically, with the *same* C++ code that would ship on the device.

## Building on Windows-on-ARM (`dev.py`, `cmake/toolchain-auto.cmake`, `swarm/_arch.py`)

Three facts make an ARM64 Windows PC hostile to "just run cmake": the MSVC-targeting clang from LLVM.org cannot link without the Windows SDK; winget-installed tools are not on every shell's PATH; the Store Python is an x64 build running emulated, so `PROCESSOR_ARCHITECTURE` lies and ctypes needs an x64 DLL. The fix is in three places:

* `cmake/toolchain-auto.cmake` is included **before `project()`** (the point at which CMake locks in a compiler). It reads the real architecture from the registry, resolves `SWARM_TARGET_ARCH` (`native|arm64|x64|x86|python`), finds `<triple>-clang++` on PATH or under `%LOCALAPPDATA%\Microsoft\WinGet\Packages`, finds ninja the same way, and records `SWARM_RESOLVED_ARCH` in the cache so a later configure cannot silently switch architecture inside the same build directory. It stands aside if the user set `CMAKE_CXX_COMPILER`, `CC/CXX` or a toolchain file.
* `swarm/_arch.py` (stdlib only) answers *machine*, *python* and *DLL* architecture from authoritative sources (registry, `sysconfig.get_platform()`, the PE header) in one vocabulary. `native.py` uses it to try only DLLs that can load and to explain mismatches (`native.diagnosis()`). It also finds the repo-local native interpreter (`native_python()`, set up by `python dev.py native-python`) that arena training re-launches itself on when the current one is emulated - numpy runs about twice as fast there.
* `dev.py` builds `build/` for the real CPU (tools, tests, native DLL) and `build-<pyarch>/` when the interpreter differs, replaces build directories whose cache came from another toolchain, and runs ctest/pytest/train/bench - all with tools located the same way, so it works from a shell that has nothing on PATH.

## Brain kernels (`cpp/src/matmul.cpp`)

Everything reduces to `C[M,N] = A[M,K] * B[K,N]`:

* **Dense**: `x[1,in] * W[in,out]` (weights stored `[in][out]` so the wide dimension is N).
* **Conv2D**: im2col turns the input into `cols[in_c*k*k, oh*ow]`, then `W[out_c, in_c*k*k] * cols`. N = output pixels, again wide.

The fast kernel is a 4x16 register-tiled micro-kernel with K blocked at 256 (the B panel a tile streams through is 16 KB and stays in L1). The f32 version is plain C++ that clang/gcc auto-vectorise (NEON, SSE, AVX - same source). The int8 version uses NEON `smlal` intrinsics (`vmlal_n_s16`) because the auto-vectoriser produced scalar gathers (~10x slower); a portable fallback exists for non-NEON targets. Accumulation is int32.

`bench_matmul` prints the throughput on this machine; `test_matmul` checks every tile/edge/K-block combination against the naive loop.

## Arena and RAM cap (`arena.hpp`, `model.cpp`)

`Model::load` makes two passes: first it parses the file image and *sums* the bytes every weight, both activation ping-pong buffers, and the worst-case scratch (im2col matrix, quantised input, int32 accumulators) will need; if a `ram_cap_bytes` is given and the total exceeds it, loading fails with both numbers in the message. Only then is one `Arena` of exactly that size allocated and everything placed. Consequently `required_bytes()` is the RAM a device must provide - not an estimate.

## `.swm` format

See [MODEL_FORMAT.md](MODEL_FORMAT.md). Written by `python/swarm/brain/formats.py`, read by `cpp/src/model.cpp` (bounds-checked; truncated/garbage files are rejected). int8 layers store per-tensor symmetric weights + one f32 scale; activations are quantised dynamically per tensor at run time, so no calibration data is needed.

## Python brain (`python/swarm/brain`)

The numpy layers mirror the C++ layers 1:1 (same weight layouts, same im2col row order) and add `backward`. `Sequential.forward_int8` reproduces the engine's int8 arithmetic so Python and C++ agree to float rounding - the tests assert `atol=1e-5`. `native.py` loads the shared library via ctypes from any `build*/bin`, skipping DLLs of the wrong architecture.

## Simulator (`python/swarm/sim`)

Discrete-event, simulated time. Costs come from **device profiles**:

* compute: `seconds = MACs / (MHz * macs_per_cycle)`; each node is a single FIFO core with a bounded queue (`--queue-limit`); overflow = rejected frame.
* radio: one **shared** channel - messages are serialised back-to-back at `bps`, plus latency, plus independent loss. Control messages (heartbeat, result, cmd) are a priority class by default (`--no-qos` disables it and lets a saturated link starve the heartbeats - a real failure mode).

The **content** of an inference (which class the brain predicts) can come from an `OracleBrain` (ground truth at a configurable accuracy, lets you sweep hypothetical model sizes), a `NumpyBrain`, or a `NativeBrain` (the C++ engine). Only the *duration* is modelled from the device profile.

### Membership

Heartbeats (8 B) are broadcast every `hb_interval`. A peer silent for `hb_interval * missed_beats` is suspected -> `on_member_change(id, alive=False)`. When it is heard again -> `on_member_change(id, True)`. The leader is `max(live members, key=(priority, -id))`; every node evaluates that on its own view, so all views converge on the same leader within one timeout without a vote. A revived node resets its table (trusts everyone until proven otherwise).

### Strategies (`strategies.py`)

All strategies subclass `SwarmNode` (camera, brain, membership, common metrics) and override `on_capture`, `on_frame`/`on_features` (bulk data in), `on_result` (small results in) and `rebalance`.

| | camera | who computes | radio per frame | weak point |
|---|---|---|---|---|
| `local` | all | each node, its own frames | results only (16 B) | N brains for N frames; every drone must carry a capable CPU |
| `central/none` | all | leader | raw frame (cam^2 B) | link saturates; leader is a single point of load |
| `central/downscale` | all | leader | brain input (1 KB) | leader compute |
| `central/features` | all | conv frontend on drone, dense head on leader | int8 features (1 KB) | drones need the conv budget |
| `striped` | leader only | member `i mod n` | 1 KB out, 16 B back only if useful | one viewpoint; frames to a dead worker until detected |

`striped` with `--striped-fps drones*fps` processes as many frames as the all-cameras designs from a single camera, spreading the brain over the swarm - that is the "faster way to process high-quality data" hypothesis this repo exists to test.

### Metrics

`Metrics.summary()` reports frames captured/processed/dropped/rejected, detections vs detections the leader learnt about, capture->leader latency p50/p95, bytes and kbps, channel utilisation (>1.0 = oversubscribed), MACs total and per node, busy fraction per node, leader changes, failures detected, rebalances, energy (from the profile's nJ/MAC and nJ/bit).

## Stress harness (`python/swarm/stress/device_stress.py`)

For a `.swm` and a device profile: (1) load in the C++ engine under `ram_cap = profile.ram_bytes`; (2) MACs -> seconds -> max fps, load, FIFO deadline misses at the requested fps; (3) actually run N frames on the host and report ms/frame and GMAC/s; (4) if int8, agreement with the f32 sibling. `--sweep` covers every profile; exit code 1 if anything overflows or misses deadlines (CI-friendly).

## Scenario presets and the battery (`python/swarm/bench`, `scenarios/`)

A `Scenario` (frozen dataclasses: `Swarm`, `Camera`, `Link`, `BrainSpec`, `Faults`, `Stress`, `Check`) is the unit of a standardized test. `run_scenario` simulates every strategy - **the baseline (`local`) always first** - with the scenario's faults; when there are faults it also runs a fault-free *twin* of every strategy, so survivability is reported as *work retained* / *recall retained* relative to an undisturbed swarm. Each strategy gets a fresh brain instance so results do not depend on run order. Survivability timings come from the simulator's event timeline (`Simulator.event`: kill / revive / suspect / readmit / leader): time to first suspicion, time to a new leader (if the victim led), and time until the leader learns a result again.

Reports (`bench/report.py`) are metric x strategy tables; the baseline column is absolute, every other cell is *value + ratio*. `[+]`/`[-]` mark the good direction. The battery scoreboard names the winner per category, but only among designs that still deliver >= 80 % of the baseline's objects/s - a design that drops most frames would otherwise "win" least compute and least radio.

Preset files in `scenarios/` are plain Python (`SCENARIO = Scenario(...)`, or `SCENARIOS = [...]` for A/B variants) with an `if __name__ == "__main__": main(SCENARIO)` footer, so a preset is both the definition and the executable; `_bootstrap.py` makes `swarm` importable without installation. `python -m swarm.bench` discovers `scenarios/*.py` (skipping `_*`), runs them, writes `out/bench/<name>.{md,json}` and `suite.md`, and exits 1 if any `Check` fails.

## Plugin tactics and brain packages

* **Plugin tactics**: `sim.strategies.register_strategy(name, cls)` adds a `SwarmNode` subclass to `NODE_CLASSES` (built-in names are reserved; a name cannot be claimed by two sources). `Scenario` validation accepts registered names (`bench.scenario.known_strategies`), and `SwarmNode.camera_mode` (`"all"` / `"single"`) replaces the hard-coded "striped gets the striped fps" rule, so one-camera plugins are fed like `striped`.
* **Brain packages**: `BrainSpec(kind="package", model=<package folder>)` (or `kind="auto"` with a folder) resolves through `swarm.sdk.packages.Package.resolve_brain` - a `.swm` package runs in the C++ engine or numpy like any model, a Python package through `PythonBrainAdapter` at its declared cost.

## SDK (`python/swarm/sdk`)

Packages are folders with `manifest.json` (`packages.py`); reading one never runs code, loading imports it under a private module namespace per package. Three types: `policy` (an arena brain: a Python class with `act(obs)`, or trained mesh weights `.npz`), `perception` and `tactic`. `api.py` defines the `Tactic` base (a `SwarmNode` with documented helpers) and the optional `PerceptionBrain` base. `validate.py` is the conformance gate: policies run a short test of their role (valid actions, determinism, ranking against the built-ins, device table), perception brains are run on synthetic frames (validity, determinism, accuracy, device fit with the RAM cap enforced by the C++ engine), tactics are simulated (determinism, the *physics* invariant - at least brain + downscale MACs per processed frame -, leader and worker loss, the s11 yardstick). `install.py` installs uploads with bounded, path-checked archive extraction; `export.py` produces the device artefacts (`.swm` for the C++ engine - perception networks and mesh policies alike -, or a runtime factory around a numpy brain). Manual: [SDK.md](SDK.md).

## Arena (`python/swarm/arena`)

The drone test bench's simulators, separate from the inference-sharing simulator above: a time-stepped, **vectorised** world in which every array has a leading batch axis of independent episodes, so a test plays all its episodes - or training plays a whole population of brain variants - in one numpy pass.

* `clash.py` (DEFENCE / OFFENCE): 3-D point-mass interceptors and attackers with separate speed and climb limits, in an airspace box with a terrain mask (low flyers are seen only up close, and fly slower). A two-layer line of guard stations that re-spaces itself as interceptors are spent; a defender brain may post interceptors elsewhere (guard posts, up to `guard_depth` out). Lead pursuit re-aimed every physics sub-step (decisions every `dt`, flight in steps of at most 0.5 s), intercepts by closest approach within a step (no tunnelling), limited shots, sensors of a fixed range, and a lossy mesh radio. Both sides radio their sightings (a drone knows what any friend it hears can see; with `relay` > 1 tracks are repeated hop by hop, aged one decision per hop); interceptors radio time-stamped claims (target + ETA, repeated once by every receiver); attackers radio "chased"; both carry learned message channels.
* `recon.py` (RECON): a shared coverage grid that keeps the best detail each cell was seen in (camera footprint grows with altitude, detail falls with it), hidden targets that each need some detail to be recognised, reports that reach the operator only through the lead drone's multi-hop radio chain (store and forward), and loads from `DeviceProfile` (detector MACs per frame, brain MACs, bytes, nJ per MAC/bit).
* `brains.py`: the built-in tactics; `NeuralBrain` (Dense-ReLU-Dense, weights per episode for training, `.npz` and `.swm` export; defenders score option rows with shared weights); `PythonBrain` (checks what user code returns; pickles as its library id, so worker processes reload it); `MixedBrain` (different brains for different episodes of a batch, each at its own decision pace); the registry over built-ins and library packages.
* `run.py`: a test = the brains under test plus every built-in alternative on identical situations; report = cards, verdict, comparison, replay.
* `es.py` / `train.py`: OpenAI-ES (antithetic, centred ranks, Adam). One generation = both sides' variants, sparring partners from a hall of fame, the skill test against the built-ins and head-to-head on fixed situations; the best weights on the skill test are kept beside the latest.
* `parallel.py`: a generation split into chunks on a pool of worker processes (`spawn`; each watches its parent and exits when it dies).
* `core.Noise`: tests draw their random numbers from one generator per batch (reproducible from the seed). Training gives each episode a key instead: its situation, radio losses and hit rolls are hashes of (key, what, when, who), so an episode plays out the same in any batch - which is what lets a generation be split across cores without changing a result - and brain variants sharing a key share their luck (common random numbers).

A brain too slow for its board decides less often (`core.decide_every`, per brain in a mixed batch); CPU load and radio bytes come only from device profiles and message sizes, never wall-clock time. With a seed, runs are deterministic - for training, on any number of cores. Manual: [ARENA.md](ARENA.md).

## Studio (`python/swarm/studio`)

A standard-library `ThreadingHTTPServer` bound to 127.0.0.1 serves a WebGL single-page app (`static/`, no third-party code) and a JSON API guarded by a per-launch token plus Host/Origin checks. The server never runs brain code: `jobs.py` starts `python -m swarm.studio.worker` per job (test, train, validate) with a timeout, cancellation and - for training - a graceful stop. Workers stream progress and structured events (one per training generation) on stdout. Test reports go to `out/studio/runs/`, and training runs (curves, recorded matches, weights) to `out/studio/training/` (`store.py`; `SWARM_STUDIO_OUT` and `SWARM_LIBRARY` redirect output and library, e.g. for tests). The page builds its forms from the settings schema (`arena.config.schema`) and replays episodes with `arena3d.js`. Guide: [STUDIO.md](STUDIO.md).

## Extending

* **New layer**: add `LayerType` + `plan_layer/scratch_needs/forward` in `cpp/src/nn.cpp`, the parser case in `model.cpp`, the numpy class with `spec()` in `layers.py`, the encoder/decoder case in `formats.py`, a case in `Sequential.load`, and a test in `cpp/tests/test_nn.cpp`.
* **New device**: add a `DeviceProfile` in `devices/profiles.py`. Calibrate `macs_per_cycle` by running `bench_matmul` on the real board. Tag Linux-class companions `"linux"` (they can host Python brains).
* **New strategy**: subclass `SwarmNode`, register in `NODE_CLASSES`, add to `COMPARE_SET` in `sim/run.py` - or ship it as an SDK tactic package (`python -m swarm.sdk new NAME --type tactic`) without touching the simulator.
* **New perception brain**: an SDK package (`.swm` network or numpy class); `python -m swarm.sdk validate` checks it against every device profile.
* **New arena brain**: an SDK policy package (`python -m swarm.sdk new NAME --type policy-defender`), or train one (`python -m swarm.arena train`). A new built-in goes into `arena/brains.py` (`BUILTINS`); the tests then compare every brain of that role against it automatically.
* **New arena observation feature**: append it in the obs class's `features()` (and `FEATURE_NAMES` in `brains.py`). Saved mesh networks are then refused with "different observation layout" - retrain them and the shipped examples, or widen them on load with zero weights for the new inputs and outputs (`_widen` in `NeuralBrain.load`, as for the defender's `role` input and guard-post outputs) so they decide exactly as before.
* **New sensor / member type** (satellite downlink, buggy, aircraft): a member is just a `DeviceProfile` + a `SwarmNode` subclass; satellites mostly differ in radio (`cubesat-obc`: 9600 bps, 10 ms) and in having no peers within range - model a ground station as the leader.
* **Real hardware**: build `swarm_core` with the `target-generic` preset (or your board's toolchain file); `run_model` needs only the C++ standard library.
