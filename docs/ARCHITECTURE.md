# Architecture

## Goals

1. Design brains (tiny neural nets) for constrained swarm members and **know** whether they fit: RAM, cycles, deadline.
2. Design **how the swarm shares work** and measure it: radio load, latency, drops, per-node busy time.
3. Watch the swarm **rebalance** when a member disappears or returns.
4. Do all of it on the development PC, deterministically, with the *same* C++ code that would ship on the device.

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

## Extending

* **New layer**: add `LayerType` + `plan_layer/scratch_needs/forward` in `cpp/src/nn.cpp`, the parser case in `model.cpp`, the numpy class with `spec()` in `layers.py`, the encoder/decoder case in `formats.py`, a case in `Sequential.load`, and a test in `cpp/tests/test_nn.cpp`.
* **New device**: add a `DeviceProfile` in `devices/profiles.py`. Calibrate `macs_per_cycle` by running `bench_matmul` on the real board.
* **New strategy**: subclass `SwarmNode`, register in `NODE_CLASSES`, add to `COMPARE_SET` in `sim/run.py`.
* **New sensor / member type** (satellite downlink, buggy, aircraft): a member is just a `DeviceProfile` + a `SwarmNode` subclass; satellites mostly differ in radio (`cubesat-obc`: 9600 bps, 10 ms) and in having no peers within range - model a ground station as the leader.
* **Real hardware**: build `swarm_core` with the `target-generic` preset (or your board's toolchain file); `run_model` needs only the C++ standard library.
