"""Scenario presets and the standardized benchmark battery.

A *scenario* is a frozen description of a swarm setup (who is in it, what the
cameras see, the radio link, the brain, which members fail and when). Running
a scenario simulates **every strategy** on that setup, always including the
``local`` baseline where every drone is an independent thinker, and reports
each strategy as *value + ratio vs. baseline* for speed, compute, radio,
mission recall and survivability. If the brain is a real ``.swm`` file, the
device stress test (RAM cap, cycle budget) is run for the devices in the swarm
too.

Preset files live in ``scenarios/`` at the repo root: each one is a runnable
Python file (``python scenarios/s02_leader_loss.py``) that defines a
``SCENARIO`` (or a ``SCENARIOS`` list). ``python -m swarm.bench`` runs the
whole battery and prints a scoreboard; copy ``scenarios/_template.py`` to add
a new standardized test.
"""

from .report import render_markdown, render_scoreboard, render_text, to_json
from .runner import ScenarioResult, StrategyResult, run_scenario
from .scenario import ALL_STRATEGIES, BrainSpec, Camera, Check, Faults, Link, Scenario, Stress, Swarm

__all__ = [
    "ALL_STRATEGIES", "BrainSpec", "Camera", "Check", "Faults", "Link", "Scenario", "Stress", "Swarm",
    "ScenarioResult", "StrategyResult", "run_scenario",
    "render_markdown", "render_scoreboard", "render_text", "to_json",
    "main",
]


def main(scenario: "Scenario | None" = None, argv=None) -> int:
    """Entry point used by preset files (``main(SCENARIO)``) and by ``python -m swarm.bench``."""
    from .cli import main as _main

    return _main(scenario=scenario, argv=argv)
