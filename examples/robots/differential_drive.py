"""Example ground rover driver bridge. Not suitable for aircraft or thrusters."""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Mapping
from typing import Protocol

from swarm.robotics import AdapterError, NullAdapter, RobotAdapter, RobotProfile, map_motor_outputs


class DriveDriver(Protocol):
    """External implementations must start inactive and bound every I/O timeout."""

    def read_observation(self) -> bytes | None: ...
    def write(self, channels: Mapping[int | str, float]) -> None: ...
    def stop(self, reason: str) -> None: ...
    def close(self) -> None: ...


class FakeDriver:
    """Injected test driver; never opens any physical connection."""

    def __init__(self) -> None:
        self.writes: list[dict[int | str, float]] = []
        self.stops: list[str] = []
        self.closed = False

    def read_observation(self) -> bytes | None:
        if self.closed:
            raise RuntimeError("fake driver is closed")
        return b"fake-camera-frame"

    def write(self, channels: Mapping[int | str, float]) -> None:
        if self.closed:
            raise RuntimeError("fake driver is closed")
        self.writes.append(dict(channels))

    def stop(self, reason: str) -> None:
        self.stops.append(reason)

    def close(self) -> None:
        self.closed = True


def _validate_drive_profile(profile: RobotProfile) -> None:
    if {motor.id for motor in profile.motors} != {"left", "right"}:
        raise ValueError("differential drive requires exactly the motor ids 'left' and 'right'")
    if any(motor.min < -1.0 or motor.max > 1.0 for motor in profile.motors):
        raise ValueError("differential-drive limits must be normalized within [-1, 1]")
    if profile.adapter_config.keys() - {"driver_factory", "driver_config"}:
        raise ValueError("unknown differential-drive adapter_config fields")
    if not isinstance(profile.adapter_config.get("driver_config", {}), dict):
        raise ValueError("driver_config must be a dict")


class DifferentialDriveAdapter(RobotAdapter):
    """Wrap an explicitly injected driver; map logical ids to configured channels."""

    def __init__(self, profile: RobotProfile, driver: DriveDriver) -> None:
        if profile.dry_run:
            raise ValueError("dry_run must use NullAdapter; do not construct a physical driver")
        _validate_drive_profile(profile)
        for method, args in (("read_observation", ()), ("write", ({},)),
                             ("stop", ("validation",)), ("close", ())):
            callback = getattr(driver, method, None)
            if (not callable(callback) or inspect.iscoroutinefunction(callback)
                    or inspect.isgeneratorfunction(callback) or inspect.isasyncgenfunction(callback)):
                raise ValueError(f"driver.{method} must be synchronous and callable")
            inspect.signature(callback).bind(*args)
        self.profile, self.driver = profile, driver

    def capabilities(self) -> frozenset[str]:
        return frozenset({"motors", "observe", "differential_drive"})

    def observe(self) -> bytes | None:
        return self.driver.read_observation()

    def apply(self, outputs: Mapping[str, float]) -> None:
        # One batch lets an external driver implement atomic wheel updates if supported.
        if self.driver.write(map_motor_outputs(self.profile, outputs)) is not None:
            raise TypeError("driver.write() must return None")

    def safe_stop(self, reason: str) -> None:
        # Ground-platform policy belongs in the driver; never generalize this to aircraft.
        if self.driver.stop(reason) is not None:
            raise TypeError("driver.stop() must return None")

    def close(self) -> None:
        if self.driver.close() is not None:
            raise TypeError("driver.close() must return None")


def make_fake_driver(config: dict) -> FakeDriver:
    if config:
        raise ValueError("the fake driver accepts an empty driver_config only")
    return FakeDriver()


def make_adapter(profile: RobotProfile) -> RobotAdapter:
    """Opt-in module factory. Vendor packages are imported only here, never above."""
    if profile.dry_run:
        return NullAdapter(profile, suppressed_adapter=profile.adapter)
    _validate_drive_profile(profile)
    path = profile.adapter_config.get("driver_factory")
    factory = make_fake_driver
    if path is not None:
        if not isinstance(path, str):
            raise ValueError("driver_factory must be a trusted local 'module:factory' string")
        module, separator, name = path.partition(":")
        if not separator or not name.isidentifier() or not all(part.isidentifier() for part in module.split(".")):
            raise ValueError("invalid driver_factory path")
        factory = getattr(importlib.import_module(module), name)
    if (not callable(factory) or inspect.iscoroutinefunction(factory)
            or inspect.isgeneratorfunction(factory) or inspect.isasyncgenfunction(factory)):
        raise ValueError("driver factory must be synchronous and callable")
    config = profile.adapter_config.get("driver_config", {})
    inspect.signature(factory).bind(config)
    driver = factory(config)
    try:
        return DifferentialDriveAdapter(profile, driver)
    except Exception as exc:
        # Factories that raise before returning a driver must clean up themselves.
        for name, args in (("stop", ("invalid driver",)), ("close", ())):
            try:
                callback = getattr(driver, name)
                if not inspect.iscoroutinefunction(callback):
                    callback(*args)
            except Exception:
                logging.getLogger(__name__).exception("Failed driver cleanup: %s", name)
        raise AdapterError("invalid differential-drive driver") from exc