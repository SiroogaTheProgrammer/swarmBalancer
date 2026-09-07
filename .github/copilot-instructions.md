# Copilot instructions for swarmBalancer

Workbench for AI swarms (satellites, drones, ...) that redistribute compute. Two languages by design:

- **C++17** (`cpp/`): anything that runs on a device - GEMM kernels, layers, `.swm` loader, C ABI. No dependencies beyond the standard library; must build for Cortex-M/A targets, so no exceptions-for-control-flow in hot paths, no heap in `Model::run`, all memory from `swarm::Arena`.
- **Python** (`python/swarm/`, numpy only): brain design + training, `.swm` export, simulation, stress harness. No ML framework.

## Build / test

- Everything through `python dev.py {doctor,build,test,train,bench,all}` - it finds cmake/ninja/llvm-mingw on PATH *or* in `%LOCALAPPDATA%\Microsoft\WinGet\Packages` (shells here often lack them), builds `build/` for the real CPU (ARM64) and `build-x64/` for the Store Python, which is an x64 build running emulated. `python -c "from swarm.brain import native; print(native.diagnosis())"` explains DLL/interpreter architecture problems.
- Plain CMake: `cmake --preset default|mingw-arm64|mingw-x64|python-dll` then `cmake --build --preset <p>`; `cmake/toolchain-auto.cmake` (included before `project()`) chooses the compiler from `SWARM_TARGET_ARCH` and refuses to switch architecture inside an existing build dir. Never hardcode compiler names in presets - they must work without PATH.
- Python: `python -m pytest` (uses `pythonpath = ["python"]` from `pyproject.toml`). Native tests skip if the engine is not built. `swarm/_arch.py` must stay stdlib-only (dev.py imports it before numpy exists).
- Train: `python -m swarm.train.train_tiny_cnn`. Sim: `python -m swarm.sim.run --compare`. Stress: `python -m swarm.stress.device_stress --model models/tiny_cnn_int8.swm --sweep`.
- Standardized battery: `python -m swarm.bench` (every `scenarios/*.py`; exit 1 if a check fails; reports in `out/bench/`). One preset: `python scenarios/s02_leader_loss.py --log`. `SWARM_FULL_BENCH=1 python -m pytest tests/test_bench.py` runs the battery under pytest.

## Invariants to keep

- numpy layers and C++ layers must stay bit-compatible: same weight layouts (`Dense [in][out]`, `Conv [out_c][in_c*k*k]`), same im2col row order, same dynamic int8 quantisation. `tests/test_native_bridge.py` asserts `atol=1e-5`.
- Any `.swm` change: update `formats.py`, `model.cpp`, `docs/MODEL_FORMAT.md`, `cpp/tests/test_nn.cpp` together; bump the magic if incompatible.
- Simulator costs come only from `DeviceProfile` (`compute_seconds`, channel `bps`); never from wall-clock time. Runs must be deterministic for a given `--seed`.
- Keep the stress harness honest: RAM fit is decided by the C++ arena under a hard cap, not by a Python estimate (the estimate is only a fallback when the engine is not built).
- Every scenario report compares against the `local` baseline (independent thinkers); `swarm.bench.runner.run_scenario` always simulates it first. Fault scenarios also run a fault-free twin so survivability is "retained vs undisturbed", not raw counts.
- All strategies must pay the same physics: e.g. every design downsamples camera->brain input somewhere (`downscale_macs`); a new strategy must not skip costs the others pay.

## Style

- Small, dependency-free, readable over clever. Comments explain *why* (device constraints, trade-offs), not what.
- New strategy -> subclass `SwarmNode` in `sim/strategies.py`, register in `NODE_CLASSES` and `COMPARE_SET`, add it to `ALL_STRATEGIES`/`STRATEGY_NOTES` in `bench/scenario.py`, add a test in `tests/test_sim.py` that kills a node, and run `python -m swarm.bench` (s11 is the yardstick for a load-aware stripe).
- New standardized test -> copy `scenarios/_template.py` to `scenarios/sNN_name.py` (name == file stem), give it a description and `Check`s; it is discovered automatically. New metric -> add to `bench/runner.collect` and a row in `bench/report.py`.
