"""Caller-owned inference integration; this module never arms or executes peer commands."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import monotonic
from typing import Protocol

from swarm.robotics import HardwareGuard


class Result(Protocol):
    payload: bytes
    useful: bool


class InferenceNode(Protocol):
    async def submit(self, observation: bytes) -> Result: ...


def conservative_rover_policy(payload: bytes) -> Mapping[str, float]:
    """A tiny LOCAL allowlist, not a decoder for peer-provided actuator mappings.

    Only a test rover should use these sample values. A useful inference is not
    a safety certificate; real deployments need independent local interlocks.
    """
    if payload == b"clear":
        return {"left": 0.2, "right": 0.2}
    if payload == b"obstacle":
        return {"left": 0.0, "right": 0.0}
    raise ValueError("unrecognized inference label")


async def run_once(node: InferenceNode, guard: HardwareGuard,
                   local_policy: Callable[[bytes], Mapping[str, float]], sequence: int) -> bool:
    """Use an ALREADY locally armed guard; waiting cannot extend an old deadline."""
    try:
        observation = guard.observe()
        if observation is None:
            guard.disarm("no local observation")
            return False
        result = await node.submit(observation)
        if not isinstance(result.payload, bytes) or type(result.useful) is not bool:
            raise ValueError("inference must return payload: bytes and useful: bool")
        if not result.useful:
            guard.disarm("no useful inference")
            return False
        outputs = local_policy(result.payload)
        guard.command(sequence, outputs, expires_at=monotonic() + min(0.1, guard.profile.watchdog_s))
        return True
    except Exception:
        # Transport errors, invalid results AND local policy errors all fail stopped.
        # The surrounding guard context also cleans up on cancellation/BaseException.
        guard.on_link_loss()
        raise