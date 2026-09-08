"""Runnable without installation or vendor libraries; refuses hardware-enabled profiles."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Keep the checkout demo runnable without importing training, numpy or a real runtime.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from swarm.robotics import HardwareGuard, NullAdapter, RobotProfile, load_adapter  # noqa: E402
from examples.robots.local_inference import conservative_rover_policy, run_once  # noqa: E402


@dataclass
class DemoResult:
    payload: bytes
    useful: bool


class DemoNode:
    async def submit(self, observation: bytes) -> DemoResult:
        # In-memory inference stand-in, NOT a network endpoint or actuator decoder.
        if observation != b"demo-camera-frame":
            raise ValueError("unexpected demonstration observation")
        return DemoResult(payload=b"clear", useful=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("profile.json"))
    parser.add_argument("--arm", action="store_true", help="explicit LOCAL arm for in-memory dry run only")
    args = parser.parse_args()
    profile = RobotProfile.load(args.config)
    if not profile.dry_run:
        parser.error("this demonstration refuses dry_run=false; no adapter has been loaded")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    adapter = load_adapter(profile)
    assert type(adapter) is NullAdapter
    with HardwareGuard(adapter, profile) as guard:
        if args.arm:
            adapter.queue_observation(b"demo-camera-frame")
            guard.arm(local=True)  # Only the explicit local CLI choice can reach this line.
            asyncio.run(run_once(DemoNode(), guard, conservative_rover_policy, sequence=0))
            guard.disarm("dry-run demonstration finished")
        else:
            print("DISARMED dry run: no commands. --arm explicitly enables an in-memory demonstration.")
    print(f"DRY RUN ONLY: {len(adapter.writes)} recorded writes; closed={adapter.closed}; no hardware I/O")


if __name__ == "__main__":
    main()