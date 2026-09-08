"""Stdlib-only, local robot integration. Not a controller or a network command API."""

from .adapters import AdapterError, NullAdapter, RobotAdapter, load_adapter, make_null_adapter
from .guard import GuardError, HardwareGuard
from .profile import Attachment, MotorConfig, RobotProfile, map_motor_outputs

__all__ = [
    "AdapterError", "Attachment", "GuardError", "HardwareGuard", "MotorConfig",
    "NullAdapter", "RobotAdapter", "RobotProfile", "load_adapter", "make_null_adapter",
    "map_motor_outputs",
]