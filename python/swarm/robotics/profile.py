"""Strict, dependency-free local robot configuration. No hardware discovery."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

DEFAULT_ADAPTER = "swarm.robotics:make_null_adapter"
MAX_WATCHDOG_S = 60.0
CONNECTION_INTERFACES = frozenset({"virtual", "gpio", "i2c", "spi", "serial", "usb", "can"})


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, not a bool or string")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _object(value: object, allowed: set[str], name: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    return value


def _required(data: dict, names: set[str], context: str) -> None:
    if names - data.keys():
        raise ValueError(f"{context} missing fields: {sorted(names - data.keys())}")


def _json_copy(value: object) -> object:
    """Copy only JSON values; reject Python objects and nonfinite driver settings."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        _finite(value, "adapter_config value")
        return value
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_copy(item) for key, item in value.items()}
    raise ValueError("adapter_config must contain only JSON values with string object keys")


def _factory_path(value: object) -> str:
    path = _text(value, "adapter")
    module, separator, factory = path.partition(":")
    if not separator or not factory.isidentifier() or not all(part.isidentifier() for part in module.split(".")):
        raise ValueError("adapter must be a trusted local 'module:factory' import path")
    return path


@dataclass(frozen=True)
class MotorConfig:
    """Explicit channel and signed driver-unit limits; inversion never clamps."""

    id: str
    channel: int | str
    inverted: bool = False
    min: float = -1.0
    max: float = 1.0

    def __post_init__(self) -> None:
        _text(self.id, "motor.id")
        if isinstance(self.channel, str):
            _text(self.channel, "motor.channel")
        elif type(self.channel) is not int or self.channel < 0:
            raise ValueError("motor.channel must be an explicit nonnegative int or nonempty string")
        if type(self.inverted) is not bool:
            raise ValueError("motor.inverted must be a bool")
        lower, upper = _finite(self.min, "motor.min"), _finite(self.max, "motor.max")
        if lower >= upper:
            raise ValueError("motor.min must be less than motor.max")
        object.__setattr__(self, "min", lower)
        object.__setattr__(self, "max", upper)

    def map_value(self, value: float) -> float:
        """Validate logical AND post-inversion bounds, then return driver units."""
        logical = _finite(value, f"output {self.id}")
        physical = -logical if self.inverted else logical
        if not self.min <= logical <= self.max or not self.min <= physical <= self.max:
            raise ValueError(f"output {self.id} exceeds [{self.min}, {self.max}] before or after inversion")
        return physical


@dataclass(frozen=True)
class Attachment:
    """Descriptive only: endpoints are never opened by the SDK."""

    id: str
    type: str
    connection: Mapping[str, str]

    def __post_init__(self) -> None:
        _text(self.id, "attachment.id")
        _text(self.type, "attachment.type")
        if not isinstance(self.connection, Mapping):
            raise ValueError("attachment.connection must be an object")
        connection = _object(dict(self.connection), {"interface", "endpoint"}, "connection")
        _required(connection, {"interface", "endpoint"}, "connection")
        interface = _text(connection["interface"], "connection.interface")
        if interface not in CONNECTION_INTERFACES:
            raise ValueError(f"unknown connection.interface: {interface}")
        _text(connection["endpoint"], "connection.endpoint")
        object.__setattr__(self, "connection", MappingProxyType(connection))


@dataclass(frozen=True)
class RobotProfile:
    """Safety fields are immutable; adapter_config is a detached JSON dictionary."""

    adapter: str = DEFAULT_ADAPTER
    adapter_config: dict = field(default_factory=dict)
    motors: tuple[MotorConfig, ...] = ()
    attachments: tuple[Attachment, ...] = ()
    required_capabilities: frozenset[str] = frozenset()
    dry_run: bool = True
    watchdog_s: float = 0.5
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only schema_version 1 is supported")
        _factory_path(self.adapter)
        if type(self.dry_run) is not bool:
            raise ValueError("dry_run must be a bool")
        watchdog = _finite(self.watchdog_s, "watchdog_s")
        if not 0 < watchdog <= MAX_WATCHDOG_S:
            raise ValueError(f"watchdog_s must be > 0 and <= {MAX_WATCHDOG_S}")
        object.__setattr__(self, "watchdog_s", watchdog)
        if not isinstance(self.adapter_config, dict):
            raise ValueError("adapter_config must be a dict")
        try:
            object.__setattr__(self, "adapter_config", _json_copy(self.adapter_config))
        except RecursionError as exc:
            raise ValueError("adapter_config must be acyclic JSON") from exc
        if not isinstance(self.motors, tuple) or any(not isinstance(m, MotorConfig) for m in self.motors):
            raise ValueError("motors must be a tuple of MotorConfig instances")
        if not isinstance(self.attachments, tuple) or any(not isinstance(a, Attachment) for a in self.attachments):
            raise ValueError("attachments must be a tuple of Attachment instances")
        if not isinstance(self.required_capabilities, frozenset):
            raise ValueError("required_capabilities must be a frozenset of strings")
        for capability in self.required_capabilities:
            _text(capability, "required capability")
        ids = [item.id for item in (*self.motors, *self.attachments)]
        if len(ids) != len(set(ids)):
            raise ValueError("motor and attachment ids must be unique, including across both lists")
        # Also reject the obvious ambiguous alias, channel 1 versus channel "1".
        channels = [str(m.channel) for m in self.motors]
        if len(channels) != len(set(channels)):
            raise ValueError("motor channels must be unique")

    @classmethod
    def from_dict(cls, data: dict) -> RobotProfile:
        data = _object(data, {
            "schema_version", "adapter", "adapter_config", "motors", "attachments",
            "required_capabilities", "dry_run", "watchdog_s",
        }, "profile")
        motors = data.get("motors", [])
        attachments = data.get("attachments", [])
        required = data.get("required_capabilities", [])
        if not all(isinstance(value, list) for value in (motors, attachments, required)):
            raise ValueError("motors, attachments and required_capabilities must be lists")
        parsed_motors = []
        for motor in motors:
            motor = _object(motor, {"id", "channel", "inverted", "min", "max"}, "motor")
            _required(motor, {"id", "channel"}, "motor")
            parsed_motors.append(MotorConfig(**motor))
        parsed_attachments = []
        for attachment in attachments:
            attachment = _object(attachment, {"id", "type", "connection"}, "attachment")
            _required(attachment, {"id", "type", "connection"}, "attachment")
            parsed_attachments.append(Attachment(**attachment))
        for capability in required:
            _text(capability, "required capability")
        if len(required) != len(set(required)):
            raise ValueError("required_capabilities contains duplicates")
        return cls(**{
            **data, "motors": tuple(parsed_motors), "attachments": tuple(parsed_attachments),
            "required_capabilities": frozenset(required),
        })

    @classmethod
    def load(cls, path: Path) -> RobotProfile:
        def unique_object(pairs: list[tuple[str, object]]) -> dict:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"nonfinite JSON constant: {value}")

        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
        return cls.from_dict(data)


def _checked_outputs(profile: RobotProfile, outputs: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(outputs, Mapping):
        raise ValueError("outputs must be a mapping of motor ids to finite numbers")
    snapshot = dict(outputs)
    ids = {motor.id for motor in profile.motors}
    if not ids or snapshot.keys() != ids:
        raise ValueError("outputs must contain every configured motor exactly once and no unknown actuators")
    result = {}
    for motor in profile.motors:
        value = _finite(snapshot[motor.id], f"output {motor.id}")
        motor.map_value(value)
        result[motor.id] = value
    return result


def map_motor_outputs(profile: RobotProfile, outputs: Mapping[str, float]) -> dict[int | str, float]:
    """Map a complete logical snapshot to explicit channels, inverting exactly once."""
    checked = _checked_outputs(profile, outputs)
    return {motor.channel: motor.map_value(checked[motor.id]) for motor in profile.motors}