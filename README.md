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

Prerequisites (all user-scope winget packages, no admin rights and no Visual Studio needed; on Linux/macOS any C++17 compiler + cmake + ninja):

```powershell
winget install --id Kitware.CMake --scope user --silent --accept-package-agreements --accept-source-agreements
winget install --id Ninja-build.Ninja --scope user --silent --accept-package-agreements --accept-source-agreements
winget install --id MartinStorsjo.LLVM-MinGW.UCRT --scope user --silent --accept-package-agreements --accept-source-agreements
```

Then everything goes through one script. It finds the tools itself (on PATH **or** in the winget package directory - shells on Windows-on-ARM often do not see them), reads the real CPU architecture, and builds a DLL that matches *your* python.exe:

```powershell
python dev.py doctor     # what this machine has / lacks, and the exact command to fix each item
python dev.py build      # C++ engine: tools + tests for the real CPU, plus a DLL for this python if it differs
python dev.py test       # ctest + pytest (installs numpy/pytest into the current interpreter if missing)
python dev.py train      # the satellite/drone image-recognition brain -> models/tiny_cnn_{f32,int8}.swm  (~20 s)
python dev.py bench      # the standardized scenario battery, reports in out/bench/
python dev.py all        # all of the above in order
```

<details><summary>Why a script, and what it does on Windows-on-ARM</summary>

Three things go wrong on an ARM64 Windows PC when CMake is left to its own devices, and they all look like "cmake is broken":

1. CMake picks `C:\Program Files\LLVM\clang++` (the MSVC-targeting build) and fails to link with *could not open kernel32.lib* - it needs the Windows SDK / Visual Studio. [cmake/toolchain-auto.cmake](cmake/toolchain-auto.cmake) instead selects llvm-mingw (`aarch64-w64-mingw32-clang++`), which is self-contained.
2. Tools installed by winget are not on the PATH of every shell (and often not of VS Code). The toolchain file and `dev.py` look in `%LOCALAPPDATA%\Microsoft\WinGet\Packages` as well.
3. The Microsoft Store Python is an **x64** build running under emulation (`sysconfig.get_platform()` says `win-amd64` even though the CPU is ARM64). ctypes can only load a DLL of the interpreter's own architecture, so `dev.py build` cross-compiles a second, x64 `swarm_brain.dll` into `build-x64/`; `python -c "from swarm.brain import native; print(native.diagnosis())"` tells you which DLL is in use and why. A native ARM64 Python (python.org installer) avoids the second build and runs numpy natively.

Plain CMake still works if you prefer it: `cmake --preset default` (auto toolchain for the real CPU), `--preset mingw-arm64`, `--preset mingw-x64`, `--preset python-dll` (DLL for whatever `python` on PATH is), then `cmake --build --preset <name>` and `ctest --preset mingw-arm64`. `-DSWARM_TARGET_ARCH=native|arm64|x64|x86|python` chooses the CPU for a bare `cmake -S . -B build -G Ninja`.
</details>

Manual equivalents of the steps above:

```powershell
# C++ engine by hand
cmake --preset default; cmake --build --preset default; ctest --preset default
.\build\bin\bench_matmul.exe          # kernel throughput on this machine

# Python
python -m pip install -e .[dev]       # or just: set PYTHONPATH=python and pip install numpy pytest
python -m pytest

# Train the satellite/drone image-recognition brain (synthetic overhead imagery, ~20 s)
python -m swarm.train.train_tiny_cnn --epochs 3
#    -> models/tiny_cnn_f32.swm, models/tiny_cnn_int8.swm (+ cross-check against the C++ engine)

# Stress it "as if on a small device"
python -m swarm.stress.device_stress --model models/tiny_cnn_int8.swm --sweep
.\build\bin\run_model.exe models\tiny_cnn_int8.swm --ram-cap 196608 --repeat 1000

# Simulate the swarm: compare strategies, kill the leader at t=10 s
python -m swarm.sim.run --compare --drones 5 --fps 10 --duration 30 --kill leader@10
python -m swarm.sim.run --strategy striped --kill 2@8 --revive 2@14 -v        # event log
python -m swarm.sim.run --compare --brain native --model models/tiny_cnn_int8.swm  # real brain in the C++ engine

# Standardized tests: run a scenario preset, or the whole battery (reports in out/bench/)
python scenarios/s02_leader_loss.py --log
python -m swarm.bench
```

VS Code: `Terminal > Run Task` has `dev: build C++ engine` (default build task), `dev: test`, `dev: all`, plus the trainer, simulator, stress and bench entries; `Run and Debug` has launch configs for the Python entry points and for the preset file open in the editor. The CMake Tools extension is configured to use the presets (kit scanning would pick the wrong clang).

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
dev.py               one entry point: doctor | build | test | train | bench | all (finds the toolchain itself)
cmake/               toolchain-auto.cmake: picks llvm-mingw + ninja + target CPU on Windows before project()
cpp/                 C++ brain: include/swarm/*.hpp, src/*.cpp, tools/ (bench_matmul, run_model), tests/
python/swarm/
  _arch.py           real machine / python / DLL architecture (stdlib only)
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
