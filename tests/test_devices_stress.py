"""Device profiles and the stress harness."""

import subprocess
import sys
from pathlib import Path

import pytest

from swarm.brain import native
from swarm.devices import DEVICE_PROFILES, get_profile
from swarm.stress.device_stress import stress
from swarm.train.dataset import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_profiles_are_sane():
    for p in DEVICE_PROFILES.values():
        assert p.cpu_mhz > 0 and p.macs_per_cycle > 0 and p.ram_bytes > 0 and p.radio_bps > 0
    m4 = get_profile("mcu-m4")
    assert m4.compute_seconds(84_000_000) == pytest.approx(1.0)
    assert m4.tx_seconds(250_000 // 8) == pytest.approx(1.0)
    with pytest.raises(KeyError):
        get_profile("nope")


def test_stress_reports_fit_and_deadlines(swm_paths):
    x, y = make_dataset(20, 16, seed=3).x, make_dataset(20, 16, seed=3).y
    r = stress(swm_paths["int8"], get_profile("mcu-m4"), fps=10.0, x=x, y=y)
    assert r.fits_ram and r.deadline_misses == 0 and r.max_fps > 10
    assert r.host_ms_per_frame > 0 and 0 <= r.accuracy <= 1
    # the tiny test brain takes ~0.35 ms on a Cortex-M4; a 0.1 ms period is impossible -> misses
    r2 = stress(swm_paths["int8"], get_profile("mcu-m4"), fps=10_000.0, x=x, y=y)
    assert r2.deadline_misses > 0 and r2.cpu_load_at_fps > 1.0


@pytest.mark.skipif(not native.available(), reason="C++ engine not built")
def test_stress_detects_ram_overflow(swm_paths, model):
    from swarm.devices.profiles import DeviceProfile

    tiny = DeviceProfile("tiny", "", 100, 1.0, ram_bytes=1024, radio_bps=1e3, radio_latency_s=0.0)
    x, y = make_dataset(5, 16, seed=3).x, make_dataset(5, 16, seed=3).y
    r = stress(swm_paths["f32"], tiny, fps=1.0, x=x, y=y)
    assert not r.fits_ram and "device cap" in r.ram_error and r.ram_needed > 1024


def test_cli_entrypoints_run(swm_paths):
    env_py = sys.executable
    cmds = [
        [env_py, "-m", "swarm.sim.run", "--compare", "--drones", "3", "--duration", "3", "--kill", "1@1"],
        [env_py, "-m", "swarm.stress.device_stress", "--model", str(swm_paths["int8"]), "--device", "mcu-m7",
         "--frames", "10"],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "python")})
        assert r.returncode == 0, r.stdout + r.stderr
