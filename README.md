# swarmBalancer

A workbench for designing **AI swarms that share their brains**: satellites, drones (and later buggies, small aircraft, ...) that redistribute the work of processing large / high-quality sensor data across the swarm instead of each member being an independent thinker.

Everything here is designed to be **built and stress-tested on this PC first**: the brain runs in a C++ engine with a hard RAM cap that emulates a small device, the swarm runs in a deterministic simulator with a shared radio channel, and members can be killed / revived to watch the swarm rebalance itself.

| Layer | Language | What |
|---|---|---|
| **Brain kernels** | C++17 | Register-tiled f32 GEMM (auto-vectorised) + NEON int8 GEMM, im2col conv, dense, relu, maxpool, softmax. 30+ GMAC/s on the host, ~15-46x a naive loop. |
| **Brain engine** | C++17 | Loads `.swm` models into one bump-allocated arena -> the exact RAM a device needs is known and can be capped. C ABI for Python/ctypes and any RTOS. |
| **Brain design + training** | Python/numpy | Same layer set with backward passes; trains and exports `.swm` (f32 + int8). No ML framework needed. |
| **Swarm simulator** | Python | Discrete-event: devices (MACs -> seconds), shared channel (bps/latency/loss), heartbeats, failure detection, deterministic leader election, three workload strategies. |
| **Stress harness** | Python + C++ | Checks a brain against device profiles (Cortex-M4 drone MCU ... CubeSat OBC): RAM fit under a hard cap, cycle budget, deadline misses, int8 vs f32 agreement. |

## Quick start

```powershell
# 1. C++ engine (Windows ARM64 with llvm-mingw; use --preset default on Linux/macOS or with another compiler)
cmake --preset mingw-arm64
cmake --build --preset mingw-arm64
ctest --preset mingw-arm64
.\build\bin\bench_matmul.exe          # kernel throughput on this machine

# If your python.exe is an x64 build running emulated on an ARM64 PC (Store Python), ctypes needs an x64 DLL too:
cmake --preset mingw-x64; cmake --build --preset mingw-x64

# 2. Python
python -m pip install -e .[dev]       # or just: set PYTHONPATH=python and pip install numpy pytest
python -m pytest

# 3. Train the satellite/drone image-recognition brain (synthetic overhead imagery, ~20 s)
python -m swarm.train.train_tiny_cnn --epochs 3
#    -> models/tiny_cnn_f32.swm, models/tiny_cnn_int8.swm (+ cross-check against the C++ engine)

# 4. Stress it "as if on a small device"
python -m swarm.stress.device_stress --model models/tiny_cnn_int8.swm --sweep
.\build\bin\run_model.exe models\tiny_cnn_int8.swm --ram-cap 196608 --repeat 1000

# 5. Simulate the swarm: compare strategies, kill the leader at t=10 s
python -m swarm.sim.run --compare --drones 5 --fps 10 --duration 30 --kill leader@10
python -m swarm.sim.run --strategy striped --kill 2@8 --revive 2@14 -v        # event log
python -m swarm.sim.run --compare --brain native --model models/tiny_cnn_int8.swm  # real brain in the C++ engine

# 6. Standardized tests: run a scenario preset, or the whole battery (reports in out/bench/)
python scenarios/s02_leader_loss.py --log
python -m swarm.bench
```

VS Code: `Terminal > Run Task` has entries for all of the above; `Run and Debug` has launch configs for the Python entry points and for the preset file open in the editor.

## Scenario presets and the standardized battery

`scenarios/*.py` are runnable preset files. Each describes one setup - who is in the swarm, what the cameras see, the radio link, the brain, who fails and when - plus pass/fail *checks*. Running one simulates **every strategy on that setup, always including the `local` baseline (every drone is an independent thinker)**, and prints each strategy as *value + ratio vs baseline*: mission (frames/s, objects reported, recall, detection latency), cost (compute, busiest CPU, radio, energy, cameras) and, when there are faults, survivability (time to detect, time to re-elect, time until results flow again, work and recall retained vs a fault-free twin run). If the brain is a real `.swm`, the device stress test (RAM cap in the C++ engine, cycle budget) runs for the devices in the swarm too.

```
python scenarios/s04_mcu_swarm.py            # one preset; -h for --brain, --fast, --strategies, --log, --duration ...
python -m swarm.bench                        # the battery: every preset, scoreboard at the end, exit 1 if a check fails
python -m swarm.bench --list                 # what is in the battery
python -m swarm.bench s02 s08 --no-stress    # a subset by name prefix
```

| preset | question it answers |
|---|---|
| `s01_reference` | With everything working, what does each design cost to deliver the same detections as 5 independent thinkers? |
| `s02_leader_loss` | The command drone dies: how fast is it noticed, re-elected around, and how much of the mission survives? |
| `s03_worker_churn` | Workers drop out and rejoin: is re-integration clean, what is lost in the detection windows? |
| `s04_mcu_swarm` | Six Cortex-M4s and a 6.4 MMAC brain: when nobody can compute for everybody, which design keeps up? |
| `s05_narrow_link` | 100 kbps long-range link: which designs are radio-bound? |
| `s06_lossy_link_*` | 15 % message loss: does the failure detector cry wolf, and how should its timeout be tuned? |
| `s07_satellites` | A payload sat and three CubeSats on 9.6 kbps: classify on orbit or ship imagery? |
| `s08_scale_*` | 3, 6, 12 drones: how do compute, radio and latency scale with N per design? |
| `s09_*_scene` | Busy vs empty scene: how much of the stripe's radio advantage depends on content? |
| `s10_no_qos` | Control messages queue behind frames: is a control-plane priority class a must? |
| `s11_heterogeneous` | Mixed A72/A53/M4 swarm: naive round-robin striping overloads the MCUs - the yardstick for a load-aware stripe. |
| `s12_low_power_stripe` | One camera at normal fps, four drones resting: compute/energy saved vs observations given up. |

Add a test by copying [scenarios/_template.py](scenarios/_template.py) to `scenarios/sNN_name.py`; it is picked up automatically. Checks (`Check(strategy, metric, op, value, ratio=...)`) make the battery a regression test: `pytest` runs every preset in a shortened form, `SWARM_FULL_BENCH=1 pytest` runs the full battery with its checks.

## The two drone designs (and a baseline)

```
local      every drone thinks for itself, reports detections           <- baseline ("independent thinkers")
central    every camera streams to the command drone, which computes for all and broadcasts decisions
           --preprocess none       raw camera frames            (cam_size^2 bytes)
           --preprocess downscale  resize on the drone          (brain input bytes)
           --preprocess features   run the conv frontend on the drone, ship int8 features,
                                   leader only runs the dense head (split inference)
striped    only the front drone's camera is on; frame i goes to live member i mod n;
           a worker answers only when it found something useful, otherwise stays silent
```

Typical result (5 drones, 128x128 camera, 10 fps, 6 Mbps shared link, leader killed at t=10 s):

```
strategy             proc   drop    acc det rep/all   p50ms   p95ms     kbps   chan B/frame   GMAC   busy    max ldr rbl
local                1300      1  87.5%   390/467       0.1     2.2      5.5   0.1%      16   0.52   0.1%   0.1%   1   4
central/none         1239     62  86.5%   452/452      45.8    89.5   4443.4  74.1%   13449   0.52   0.1%   0.2%   1   4
central/downscale    1240     61  87.6%   443/443       4.9     7.7    298.0   5.0%     901   0.52   0.1%   0.2%   1   4
central/features     1240     61  87.3%   439/439       4.9     7.7    298.0   5.0%     901   0.54   0.1%   0.1%   1   4
striped               287      3  88.9%    97/111       5.5     5.6     65.7   1.1%     858   0.12   0.0%   0.0%   1   4
```

Raw streaming eats 74% of the link (and 266% with a 1 Mbps radio, dropping half the frames); pre-processing cuts it 15x; striping uses one camera and a fifth of the compute. Swap `--leader mcu-m7 --worker mcu-m4 --brain-macs 6400000` to see compute rather than radio become the bottleneck, and `--no-qos` to see heartbeats starve on a saturated link.

## Fault tolerance

Every node broadcasts an 8-byte heartbeat every `--hb-interval` seconds; a peer silent for `--missed-beats` intervals is *suspected*, the node's rotation / leader is recomputed, and `rebalances` is incremented. The leader is the live node with the highest priority (compute power, ties -> lowest id) - the same rule on every node's own view, so no election messages are needed and all nodes converge within one timeout. Frames sent to a dead node before detection are the price (visible as `drop`); tune `--hb-interval` / `--missed-beats` to trade detection speed against false suspicions.

## Layout

```
cpp/                 C++ brain: include/swarm/*.hpp, src/*.cpp, tools/ (bench_matmul, run_model), tests/
python/swarm/
  brain/             numpy layers (fwd+bwd), Sequential, .swm format, ctypes bridge to the C++ engine
  train/             synthetic overhead-imagery dataset, tiny CNN trainer
  devices/           device profiles (MHz, MACs/cycle, RAM, radio)
  sim/               simulator core, channel, membership, brains, strategies, CLI
  stress/            device stress harness
  bench/             scenario presets: dataclasses, runner (twin run, survivability, checks), reports, battery CLI
scenarios/           the standardized battery (s01..s12) + _template.py; each file is runnable
tests/               pytest suite (C++ engine tests are skipped if it is not built)
models/              trained .swm files (git-ignored; regenerate with the trainer)
out/bench/           reports written by the battery (git-ignored)
docs/                ARCHITECTURE.md, MODEL_FORMAT.md
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design details and how to extend (new layer, new device, new strategy).
