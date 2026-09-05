"""Scenario presets and the benchmark battery.

Every preset must load, run (shortened, oracle brain) and render; the baseline must always be
present and first. The full battery with its checks is what ``python -m swarm.bench`` runs; it
is also available here behind ``SWARM_FULL_BENCH=1`` because it takes about a minute.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from swarm.bench import (BrainSpec, Camera, Check, Faults, Link, Scenario, Swarm, render_markdown, render_text,
                         run_scenario, to_json)
from swarm.bench.cli import DEFAULT_DIR, discover, load_file, load_scenarios
from swarm.bench.report import dump_json, render_scoreboard

ROOT = Path(__file__).resolve().parents[1]
PRESETS = discover(DEFAULT_DIR)
SCENARIOS = [sc for f in PRESETS for sc in load_file(f)]


def test_presets_exist_and_have_unique_names_matching_files():
    assert len(PRESETS) >= 10
    names = [sc.name for sc in SCENARIOS]
    assert len(names) == len(set(names)), "scenario names must be unique"
    for f in PRESETS:
        for sc in load_file(f):
            assert sc.name.startswith(f.stem.split("_")[0]), f"{sc.name} should carry the {f.stem} prefix"
            assert sc.description and sc.checks, f"{sc.name}: every standardized test needs a description and checks"


def test_template_is_a_valid_scenario():
    [sc] = load_file(DEFAULT_DIR / "_template.py")
    assert sc.name == "sNN_my_test" and sc.faults


def test_selectors_match_by_prefix_and_path():
    picked = load_scenarios(DEFAULT_DIR, ["s02"])
    assert [sc.name for _, sc in picked] == ["s02_leader_loss"]
    picked = load_scenarios(DEFAULT_DIR, [str(DEFAULT_DIR / "s06_lossy_link.py")])
    assert {sc.name for _, sc in picked} == {"s06_lossy_link_hb3", "s06_lossy_link_hb5"}
    with pytest.raises(SystemExit):
        load_scenarios(DEFAULT_DIR, ["does_not_exist"])


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_every_preset_runs_and_renders(sc):
    res = run_scenario(sc, brain_kind="oracle", duration=min(sc.duration, 8.0), with_stress=False)
    assert res.strategies[0].strategy == sc.baseline, "the baseline is always simulated first"
    assert {r.strategy for r in res.strategies} == set(sc.strategies) | {sc.baseline}
    for r in res.strategies:
        assert r.metrics["frames_captured"] > 0
        for key in ("fps_processed", "recall", "latency_p50_ms", "gmac_ps", "kbps", "channel_util", "busy_leader",
                    "work_retained", "detect_s"):
            assert key in r.metrics
        assert (r.twin_metrics is not None) == bool(sc.faults)
    assert len(res.checks) >= len(sc.checks)
    text = render_text(res, show_log=True)
    assert sc.name in text and "baseline = " in text and "[+]" in text
    md = render_markdown(res)
    assert f"## {sc.name}" in md and "| metric |" in md
    json.loads(dump_json(res))  # NaN/inf must be sanitised
    to_json(res)


def test_survivability_metrics_and_twin_run():
    sc = Scenario(name="t_leader_loss", swarm=Swarm(size=4), camera=Camera(fps=10), link=Link(loss=0.0),
                  brain=BrainSpec(kind="oracle"), faults=Faults(kill=(("leader", 5.0),)), duration=15.0,
                  strategies=("local", "striped"))
    res = run_scenario(sc, with_stress=False)
    by = res.by_strategy()
    for name in ("local", "striped"):
        m = by[name]
        assert 1.0 <= m["detect_s"] <= 2.0 and 1.0 <= m["reelect_s"] <= 2.0 and m["recover_s"] <= 3.0
        assert m["leader_changes"] == 1
        assert 0.5 < m["work_retained"] < 1.0
    # the stripe moves its camera to the new leader; independent thinkers lose a camera for good
    assert by["striped"]["work_retained"] > by["local"]["work_retained"]
    assert by["local"]["work_retained"] < 0.95


def test_checks_evaluate_ratio_and_wildcards():
    sc = Scenario(name="t_checks", swarm=Swarm(size=3), brain=BrainSpec(kind="oracle"), duration=5.0,
                  strategies=("local", "striped"),
                  checks=(Check("*", "frames_processed", ">", 0),
                          Check("striped", "cameras_on", "<=", 0.5, ratio=True),
                          Check("striped", "recall", ">=", 2.0, note="impossible: must fail")))
    res = run_scenario(sc, with_stress=False)
    assert [c.strategy for c in res.checks] == ["local", "striped", "striped", "striped"]
    assert [c.passed for c in res.checks] == [True, True, True, False]
    assert not res.checks_passed()
    assert "FAIL" in render_text(res)


def test_bad_check_metric_is_reported():
    sc = Scenario(name="t_bad", brain=BrainSpec(kind="oracle"), duration=2.0, strategies=("local",),
                  checks=(Check("local", "no_such_metric", ">", 0),))
    with pytest.raises(KeyError, match="no_such_metric"):
        run_scenario(sc, with_stress=False)


def test_scenario_validation():
    with pytest.raises(ValueError):
        Scenario(name="x", strategies=("teleport",))
    with pytest.raises(ValueError):
        Scenario(name="x", duration=0)


def test_scoreboard_renders_for_many_results():
    results = [run_scenario(sc, brain_kind="oracle", duration=4.0, with_stress=False) for sc in SCENARIOS[:3]]
    board = render_scoreboard(results)
    assert "scoreboard" in board and all(r.scenario.name in board for r in results)


def test_preset_file_runs_directly(tmp_path):
    """`python scenarios/s01_reference.py` must work from any cwd without pip install."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, str(DEFAULT_DIR / "s01_reference.py"), "--brain", "oracle", "--duration", "3",
                        "--no-stress", "--out", str(tmp_path)], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert r.returncode in (0, 1), r.stderr  # 1 = a check failed on the shortened run, still a valid run
    assert "=== s01_reference ===" in r.stdout
    assert (tmp_path / "s01_reference.md").is_file() and (tmp_path / "s01_reference.json").is_file()


@pytest.mark.skipif(not os.environ.get("SWARM_FULL_BENCH"), reason="set SWARM_FULL_BENCH=1 to run the full battery (~1 min)")
def test_full_battery_checks_pass(tmp_path):
    r = subprocess.run([sys.executable, "-m", "swarm.bench", "--out", str(tmp_path)], cwd=ROOT, capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
    assert r.returncode == 0, "\n".join(l for l in r.stdout.splitlines() if l.startswith("FAIL")) + r.stderr
