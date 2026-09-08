"""Focused robot SDK tests: fake drivers only, never GPIO, serial or CMake."""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
import threading
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from swarm.robotics import (
    AdapterError, Attachment, GuardError, HardwareGuard, MotorConfig, NullAdapter,
    RobotAdapter, RobotProfile, load_adapter, map_motor_outputs,
)
from swarm.robotics import adapters as adapter_module
from examples.robots import differential_drive, local_inference

ROOT = Path(__file__).resolve().parents[1]


def profile(**overrides) -> RobotProfile:
    # Hardware opt-in is ONLY for these injected, in-memory test doubles.
    data = {
        "dry_run": False,
        "watchdog_s": 0.5,
        "motors": [
            {"id": "left", "channel": 2, "min": -0.5, "max": 0.5},
            {"id": "right", "channel": "right-channel", "inverted": True, "min": -0.5, "max": 0.5},
        ],
        "required_capabilities": ["motors", "observe"],
    }
    return RobotProfile.from_dict({**data, **overrides})


class Clock:
    def __init__(self, now: float = 100.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class RecordingAdapter(RobotAdapter):
    def __init__(self):
        self.writes = []
        self.stops = []
        self.close_calls = 0
        self.capability_calls = 0
        self.observations = 0
        self.supported = frozenset({"motors", "observe"})
        self.payload = b"observation"
        self.fail_apply = self.fail_observe = self.fail_stop = self.fail_close = False

    def capabilities(self) -> frozenset[str]:
        self.capability_calls += 1
        return self.supported

    def observe(self) -> bytes | None:
        self.observations += 1
        if self.fail_observe:
            raise OSError("sensor failed")
        return self.payload

    def apply(self, outputs: Mapping[str, float]) -> None:
        self.writes.append(dict(outputs))
        if self.fail_apply:
            raise OSError("driver write failed")

    def safe_stop(self, reason: str) -> None:
        self.stops.append(reason)
        if self.fail_stop:
            raise OSError("driver stop failed")

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise OSError("driver close failed")


def test_profile_defaults_and_null_adapter():
    p = RobotProfile.from_dict({})
    assert p.dry_run is True and 0 < p.watchdog_s <= 60
    assert p.motors == p.attachments == ()
    assert isinstance(p.adapter_config, dict)
    adapter = load_adapter(p)
    assert type(adapter) is NullAdapter
    assert adapter.observe() is None
    adapter.queue_observation(b"frame")
    assert adapter.observe() == b"frame"
    assert adapter.observe() is None
    with pytest.raises(ValueError):
        adapter.queue_observation(bytearray(b"not bytes"))
    adapter.close()
    with pytest.raises(AdapterError):
        adapter.apply({})


@pytest.mark.parametrize("data", [
    [], {"armed": True}, {"schema_version": 2}, {"schema_version": True},
    {"dry_run": "false"}, {"dry_run": 0}, {"watchdog_s": 0}, {"watchdog_s": -1},
    {"watchdog_s": 61}, {"watchdog_s": True}, {"watchdog_s": "0.1"},
    {"watchdog_s": float("nan")}, {"watchdog_s": float("inf")},
    {"adapter": "vendor"}, {"adapter": "vendor:make:again"}, {"adapter": ".vendor:make"},
    {"adapter": "vendor:make()"}, {"adapter": " vendor:make"},
    {"motors": "left"}, {"attachments": {}}, {"required_capabilities": "motors"},
    {"required_capabilities": ["motors", "motors"]}, {"required_capabilities": [1]},
    {"required_capabilities": [""]}, {"adapter_config": []},
    {"adapter_config": {"bad": [float("nan")]}}, {"adapter_config": {"bad": float("inf")}},
    {"adapter_config": {"bad": object()}}, {"adapter_config": {1: "bad key"}},
    {"adapter_config": {"bad": {1, 2}}},
])
def test_rejects_invalid_profile_schema(data):
    with pytest.raises(ValueError):
        RobotProfile.from_dict(data)


@pytest.mark.parametrize("motor", [
    {"id": "left"}, {"channel": 1}, {"id": "", "channel": 1},
    {"id": "left", "channel": True}, {"id": "left", "channel": -1},
    {"id": "left", "channel": ""}, {"id": "left", "channel": 1.0},
    {"id": "left", "channel": None}, {"id": "left", "channel": []},
    {"id": "left", "channel": 1, "inverted": 1},
    {"id": "left", "channel": 1, "min": float("nan")},
    {"id": "left", "channel": 1, "max": float("inf")},
    {"id": "left", "channel": 1, "min": True},
    {"id": "left", "channel": 1, "min": 0, "max": 0},
    {"id": "left", "channel": 1, "min": 2, "max": 1},
    {"id": "left", "channel": 1, "gpio_pin": 3},
])
def test_rejects_invalid_motor_schema(motor):
    with pytest.raises(ValueError):
        RobotProfile.from_dict({"motors": [motor]})


@pytest.mark.parametrize("attachment", [
    {"id": "camera", "type": "camera"},
    {"id": "camera", "type": 1, "connection": {"interface": "virtual", "endpoint": "front"}},
    {"id": "camera", "type": "camera", "connection": "usb0"},
    {"id": "camera", "type": "camera", "connection": {"interface": "unknown", "endpoint": "front"}},
    {"id": "camera", "type": "camera", "connection": {"interface": 1, "endpoint": "front"}},
    {"id": "camera", "type": "camera", "connection": {"interface": "usb"}},
    {"id": "camera", "type": "camera", "connection": {"interface": "usb", "endpoint": ""}},
    {"id": "camera", "type": "camera", "connection": {"interface": "usb", "endpoint": "front", "port": 1}},
])
def test_rejects_invalid_attachment_schema(attachment):
    with pytest.raises(ValueError):
        RobotProfile.from_dict({"attachments": [attachment]})


def test_duplicate_ids_channels_and_cross_namespace_ids():
    with pytest.raises(ValueError, match="ids"):
        profile(motors=[{"id": "same", "channel": 1}, {"id": "same", "channel": 2}])
    for alias in (1, "1"):
        with pytest.raises(ValueError, match="channels"):
            profile(motors=[{"id": "a", "channel": 1}, {"id": "b", "channel": alias}])
    attachment = {"id": "camera", "type": "camera", "connection": {"interface": "usb", "endpoint": "front"}}
    with pytest.raises(ValueError, match="ids"):
        profile(attachments=[attachment, attachment])
    with pytest.raises(ValueError, match="ids"):
        profile(attachments=[{**attachment, "id": "left"}])


def test_config_is_detached_and_safety_fields_immutable():
    settings = {"nested": [{"baud": 9600}]}
    p = profile(adapter_config=settings)
    settings["nested"][0]["baud"] = 0
    assert p.adapter_config["nested"][0]["baud"] == 9600
    with pytest.raises(FrozenInstanceError):
        p.dry_run = False
    with pytest.raises(FrozenInstanceError):
        p.motors[0].max = 100
    attachment = Attachment("camera", "camera", {"interface": "virtual", "endpoint": "front"})
    with pytest.raises(TypeError):
        attachment.connection["interface"] = "gpio"
    with pytest.raises(ValueError):
        RobotProfile(watchdog_s=float("nan"))
    with pytest.raises(ValueError):
        MotorConfig("left", True)


def test_profile_json_load_and_duplicate_keys(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text('{"dry_run": true, "watchdog_s": 0.25}', encoding="utf-8")
    assert RobotProfile.load(path).watchdog_s == 0.25
    for content in ('{"dry_run":true,"dry_run":false}', '{"watchdog_s":NaN}',
                    '{"adapter_config":{"a":1,"a":2}}', '[{}]', '{"watchdog_s":1e999}'):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            RobotProfile.load(path)


def test_mapping_checks_logical_and_physical_limits():
    p = profile()
    assert map_motor_outputs(p, {"left": 0.3, "right": 0.2}) == {2: 0.3, "right-channel": -0.2}
    assert map_motor_outputs(p, {"left": -0.5, "right": 0.5}) == {2: -0.5, "right-channel": -0.5}
    asymmetric = MotorConfig("servo", "servo-channel", inverted=True, min=-0.2, max=0.8)
    with pytest.raises(ValueError, match="after inversion"):
        asymmetric.map_value(0.5)
    with pytest.raises(ValueError):
        map_motor_outputs(p, {"left": 0.1})


def install_plugin(monkeypatch, factory):
    module = ModuleType("robotics_test_plugin")
    module.make = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return "robotics_test_plugin:make"


def test_dry_run_never_imports_or_calls_selected_driver(monkeypatch):
    def forbidden(*args):
        raise AssertionError("driver import must not happen")

    monkeypatch.setattr(adapter_module.importlib, "import_module", forbidden)
    p = profile(dry_run=True, adapter="missing_robot_vendor:make")
    adapter = load_adapter(p)
    assert type(adapter) is NullAdapter
    assert adapter.suppressed_adapter == p.adapter
    with HardwareGuard(adapter, p, monitor=False, clock=Clock()) as guard:
        guard.arm(local=True)
        guard.command(0, {"left": 0.1, "right": 0.2}, 100.2)
    assert adapter.writes == [{"left": 0.1, "right": 0.2}] and adapter.closed


def test_guard_suppresses_direct_adapter_and_null_subclass_in_dry_run():
    class PretendNull(NullAdapter):
        def apply(self, outputs):
            raise AssertionError("overridden null method must not run")

    for supplied in (RecordingAdapter(), PretendNull(profile())):
        with HardwareGuard(supplied, profile(dry_run=True), monitor=False, clock=Clock()) as guard:
            assert type(guard.adapter) is NullAdapter and guard.adapter is not supplied
            assert guard.adapter.suppressed_adapter
            guard.arm(local=True)
            guard.command(0, {"left": 0.1, "right": 0.1}, 100.2)
        if isinstance(supplied, RecordingAdapter):
            assert supplied.capability_calls == supplied.close_calls == 0
            assert not supplied.writes and not supplied.stops


def test_trusted_plugin_factory_gets_profile(monkeypatch):
    received = []
    adapter = RecordingAdapter()

    def make(p):
        received.append(p)
        return adapter

    p = profile(adapter=install_plugin(monkeypatch, make))
    assert load_adapter(p) is adapter
    assert received == [p] and adapter.capability_calls == 1


@pytest.mark.parametrize("capabilities", [set(), {"motors", "observe"}, frozenset({1}), frozenset({"motors"})])
def test_invalid_capabilities_stop_and_close_plugin(monkeypatch, capabilities):
    adapter = RecordingAdapter()
    adapter.supported = capabilities
    path = install_plugin(monkeypatch, lambda p: adapter)
    with pytest.raises(AdapterError, match="capabilit"):
        load_adapter(profile(adapter=path))
    assert adapter.stops and adapter.close_calls == 1


def test_null_does_not_claim_real_hardware_capabilities():
    with pytest.raises(AdapterError, match="capabilit"):
        load_adapter(profile(dry_run=True, required_capabilities=["flight_autopilot"]))


def test_invalid_api_signature_and_async_callback(monkeypatch):
    async def async_observe():
        return b"frame"

    for name, callback in (("apply", lambda: None), ("observe", async_observe), ("apply", None)):
        adapter = RecordingAdapter()
        monkeypatch.setattr(adapter, name, callback)
        path = install_plugin(monkeypatch, lambda p: adapter)
        with pytest.raises(AdapterError):
            load_adapter(profile(adapter=path))
        assert adapter.stops and adapter.close_calls == 1


def test_bad_factory_and_missing_dependency_fail_without_install(monkeypatch):
    for factory in (lambda: RecordingAdapter(), lambda p: object(), "not callable"):
        path = install_plugin(monkeypatch, factory)
        with pytest.raises(AdapterError):
            load_adapter(profile(adapter=path))
    with pytest.raises(AdapterError) as error:
        load_adapter(profile(adapter="robotics_vendor_that_does_not_exist_90871:make"))
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_local_arm_manual_disarm_and_close():
    adapter, clock = RecordingAdapter(), Clock()
    guard = HardwareGuard(adapter, profile(), monitor=False, clock=clock)
    assert not guard.armed and adapter.stops == ["initially disarmed"]
    with guard:
        assert guard._thread is None
        guard.arm(local=True)
        outputs = {"left": 0.2, "right": 0.1}
        guard.command(0, outputs, 100.2)
        outputs["left"] = 99
        assert adapter.writes == [{"left": 0.2, "right": 0.1}]
        assert guard.observe() == b"observation"
        guard.disarm("operator stopped")
        assert not guard.armed and guard.fault_reason is None
        guard.arm(local=True)
        guard.command(1, {"left": 0.0, "right": 0.0}, 100.2)
    assert guard.closed and not guard.armed and adapter.close_calls == 1
    guard.close()
    assert adapter.close_calls == 1
    with pytest.raises(GuardError, match="closed"):
        guard.arm(local=True)


@pytest.mark.parametrize("authorization", [None, False, 1, "local"])
def test_arming_requires_explicit_local_boolean(authorization):
    adapter = RecordingAdapter()
    with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
        with pytest.raises(GuardError, match="local=True"):
            if authorization is None:
                guard.arm()
            else:
                guard.arm(local=authorization)
        assert not guard.armed and guard.fault_reason
    assert not adapter.writes


def test_disarmed_command_latches_stop_and_cannot_rearm():
    adapter = RecordingAdapter()
    with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
        with pytest.raises(GuardError, match="disarmed"):
            guard.command(0, {"left": 0.1, "right": 0.1}, 100.2)
        with pytest.raises(GuardError, match="latched"):
            guard.arm(local=True)
        assert not guard.poll(101)
        guard.on_link_loss()
        assert len(adapter.stops) == 2  # Initial safe state + one latched fault, not a stop loop.
    assert not adapter.writes


@pytest.mark.parametrize("sequence,outputs,deadline", [
    (-1, {"left": 0.0, "right": 0.0}, 100.2),
    (True, {"left": 0.0, "right": 0.0}, 100.2),
    (1.0, {"left": 0.0, "right": 0.0}, 100.2),
    (0, {"left": 0.0, "right": 0.0}, 100.0),
    (0, {"left": 0.0, "right": 0.0}, 99.0),
    (0, {"left": 0.0, "right": 0.0}, 101.0),
    (0, {"left": 0.0, "right": 0.0}, float("nan")),
    (0, {"left": 0.0, "right": 0.0}, float("inf")),
    (0, {"left": 0.0, "right": 0.0}, True),
    (0, {"left": 0.0, "right": 0.0}, "100.2"),
    (0, {"left": float("nan"), "right": 0.0}, 100.2),
    (0, {"left": float("inf"), "right": 0.0}, 100.2),
    (0, {"left": float("-inf"), "right": 0.0}, 100.2),
    (0, {"left": 0.6, "right": 0.0}, 100.2),
    (0, {"left": -0.6, "right": 0.0}, 100.2),
    (0, {"left": True, "right": 0.0}, 100.2),
    (0, {"left": "0.1", "right": 0.0}, 100.2),
    (0, {"left": 10 ** 1000, "right": 0.0}, 100.2),
    (0, {"left": 0.0, "right": 0.0, "unknown": 0.0}, 100.2),
    (0, {"left": 0.0}, 100.2), (0, {}, 100.2), (0, [0, 0], 100.2),
])
def test_invalid_commands_fail_closed(sequence, outputs, deadline):
    adapter = RecordingAdapter()
    with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
        guard.arm(local=True)
        with pytest.raises(GuardError):
            guard.command(sequence, outputs, deadline)
        assert not guard.armed and guard.fault_reason
        assert not adapter.writes and len(adapter.stops) == 2


@pytest.mark.parametrize("manual_rearm", [False, True])
def test_replay_rejected_even_across_manual_disarm(manual_rearm):
    adapter = RecordingAdapter()
    with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
        guard.arm(local=True)
        guard.command(7, {"left": 0.1, "right": 0.1}, 100.2)
        if manual_rearm:
            guard.disarm()
            guard.arm(local=True)
        with pytest.raises(GuardError, match="sequence"):
            guard.command(7, {"left": 0.1, "right": 0.1}, 100.2)
        assert len(adapter.writes) == 1 and not guard.armed


@pytest.mark.parametrize("send_command,expiry", [(False, 100.5), (True, 100.1)])
def test_deterministic_watchdog_including_initial_arm(send_command, expiry):
    adapter, clock = RecordingAdapter(), Clock()
    with HardwareGuard(adapter, profile(), monitor=False, clock=clock) as guard:
        guard.arm(local=True)
        if send_command:
            guard.command(0, {"left": 0.2, "right": 0.2}, expiry)
        assert guard.poll(expiry - 0.001)
        assert not guard.poll(expiry)
        assert guard.fault_reason and "deadline" in guard.fault_reason
        assert not guard.poll(expiry + 1)
        assert len(adapter.stops) == 2
        with pytest.raises(GuardError, match="latched"):
            guard.arm(local=True)


def test_late_command_cannot_refresh_expired_watchdog():
    adapter, clock = RecordingAdapter(), Clock()
    with HardwareGuard(adapter, profile(), monitor=False, clock=clock) as guard:
        guard.arm(local=True)
        clock.now = 100.5
        with pytest.raises(GuardError, match="latched"):
            guard.command(0, {"left": 0.1, "right": 0.1}, 100.7)
        assert not adapter.writes


def test_late_disarm_does_not_erase_a_watchdog_fault():
    clock = Clock()
    with HardwareGuard(RecordingAdapter(), profile(), monitor=False, clock=clock) as guard:
        guard.arm(local=True)
        clock.now = 101.0
        guard.disarm()
        assert guard.fault_reason is not None
        with pytest.raises(GuardError, match="latched"):
            guard.arm(local=True)


@pytest.mark.parametrize("previous_command", [False, True])
def test_rechecks_expiry_after_output_validation_before_any_write(previous_command):
    adapter, clock = RecordingAdapter(), Clock()

    class DelayedOutputs(Mapping):
        def __iter__(self):
            clock.now = 100.25
            return iter(("left", "right"))

        def __len__(self):
            return 2

        def __getitem__(self, key):
            return 0.1

    with HardwareGuard(adapter, profile(), monitor=False, clock=clock) as guard:
        guard.arm(local=True)
        if previous_command:
            guard.command(0, {"left": 0.0, "right": 0.0}, 100.1)
        before = len(adapter.writes)
        with pytest.raises(GuardError):
            guard.command(1, DelayedOutputs(), 100.5 if previous_command else 100.2)
        assert len(adapter.writes) == before


@pytest.mark.parametrize("now", [99.0, float("nan"), float("inf"), True])
def test_invalid_monotonic_time_stops(now):
    with HardwareGuard(RecordingAdapter(), profile(), monitor=False, clock=Clock()) as guard:
        guard.arm(local=True)
        with pytest.raises(GuardError):
            guard.poll(now)
        assert not guard.armed and guard.fault_reason


def test_link_loss_and_repeated_arm_are_not_heartbeats():
    for operation in (lambda g: g.on_link_loss(), lambda g: g.arm(local=True)):
        adapter = RecordingAdapter()
        with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
            guard.arm(local=True)
            try:
                operation(guard)
            except GuardError:
                pass
            assert not guard.armed and guard.fault_reason
            with pytest.raises(GuardError):
                guard.command(0, {"left": 0.0, "right": 0.0}, 100.2)


@pytest.mark.parametrize("method", ["apply", "observe"])
def test_adapter_errors_stop_and_close(method):
    adapter = RecordingAdapter()
    setattr(adapter, "fail_" + method, True)
    with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
        guard.arm(local=True)
        with pytest.raises(AdapterError) as error:
            if method == "apply":
                guard.command(0, {"left": 0.1, "right": 0.1}, 100.2)
            else:
                guard.observe()
        assert isinstance(error.value.__cause__, OSError)
        assert guard.fault_reason and not guard.armed and len(adapter.stops) == 2
    assert adapter.close_calls == 1


@pytest.mark.parametrize("payload", [bytearray(b"frame"), "frame", {"left": 1}])
def test_observe_interface_rejects_nonbytes(payload):
    adapter = RecordingAdapter()
    adapter.payload = payload
    with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
        with pytest.raises(AdapterError):
            guard.observe()
        assert guard.fault_reason


def test_apply_return_contract_and_read_only_snapshot(monkeypatch):
    adapter = RecordingAdapter()

    def bad_apply(outputs):
        with pytest.raises(TypeError):
            outputs["left"] = 1.0
        return "not None"

    monkeypatch.setattr(adapter, "apply", bad_apply)
    with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
        guard.arm(local=True)
        with pytest.raises(AdapterError):
            guard.command(0, {"left": 0.1, "right": 0.1}, 100.2)
        assert guard.fault_reason


def test_callback_overrun_is_stopped_on_return(monkeypatch):
    adapter, clock = RecordingAdapter(), Clock()
    monkeypatch.setattr(adapter, "apply", lambda outputs: setattr(clock, "now", 100.3))
    with HardwareGuard(adapter, profile(), monitor=False, clock=clock) as guard:
        guard.arm(local=True)
        with pytest.raises(GuardError, match="latched"):
            guard.command(0, {"left": 0.1, "right": 0.1}, 100.2)
        assert not guard.armed


def test_failed_stop_does_not_mask_driver_error():
    adapter = RecordingAdapter()
    with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
        adapter.fail_apply = adapter.fail_stop = True
        guard.arm(local=True)
        with pytest.raises(AdapterError) as error:
            guard.command(0, {"left": 0.1, "right": 0.1}, 100.2)
        assert str(error.value.__cause__) == "driver write failed"
        assert isinstance(guard.stop_error, OSError) and not guard.armed
        assert not guard.poll(200)
    assert adapter.close_calls == 1


def test_initial_stop_failure_closes_adapter():
    adapter = RecordingAdapter()
    adapter.fail_stop = True
    with pytest.raises(AdapterError):
        HardwareGuard(adapter, profile(), monitor=False)
    assert adapter.close_calls == 1


def test_close_failure_is_reported_and_context_preserves_original_exception():
    adapter = RecordingAdapter()
    guard = HardwareGuard(adapter, profile(), monitor=False, clock=Clock())
    guard.arm(local=True)
    adapter.fail_close = True
    with pytest.raises(AdapterError):
        guard.close()
    assert guard.closed and not guard.armed and guard.fault_reason
    adapter = RecordingAdapter()
    with pytest.raises(LookupError, match="local policy broke"):
        with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
            guard.arm(local=True)
            adapter.fail_close = True
            raise LookupError("local policy broke")
    assert guard.closed and not guard.armed and adapter.stops


def test_optional_daemon_watchdog_starts_only_in_context_and_joins():
    adapter = RecordingAdapter()
    stopped = threading.Event()
    original_stop = adapter.safe_stop

    def stop(reason):
        original_stop(reason)
        stopped.set()

    adapter.safe_stop = stop
    guard = HardwareGuard(adapter, profile(watchdog_s=0.05))
    assert guard._thread is None
    stopped.clear()
    with guard:
        assert guard._thread is not None and guard._thread.daemon
        guard.arm(local=True)
        assert stopped.wait(2.0), "watchdog did not request a stop"
        assert not guard.armed and guard.fault_reason
    assert guard.closed and not guard._thread.is_alive()


def test_callbacks_serialized_during_concurrent_link_loss():
    entered, release, requested, stopped = (threading.Event() for _ in range(4))
    failures = []

    class BlockingAdapter(RecordingAdapter):
        active = False

        def apply(self, outputs):
            self.active = True
            entered.set()
            try:
                assert release.wait(2.0)
                super().apply(outputs)
            finally:
                self.active = False

        def safe_stop(self, reason):
            assert not self.active, "safe_stop overlapped apply"
            super().safe_stop(reason)
            stopped.set()

    adapter = BlockingAdapter()
    with HardwareGuard(adapter, profile(), monitor=False, clock=Clock()) as guard:
        stopped.clear()
        guard.arm(local=True)

        def command():
            try:
                guard.command(0, {"left": 0.1, "right": 0.1}, 100.2)
            except Exception as exc:
                failures.append(exc)

        def link_loss():
            requested.set()
            guard.on_link_loss()

        writer = threading.Thread(target=command)
        stopper = threading.Thread(target=link_loss)
        writer.start()
        try:
            assert entered.wait(2.0)
            stopper.start()
            assert requested.wait(2.0)
            assert not stopped.wait(0.02)
        finally:
            release.set()
            writer.join(2.0)
            if stopper.ident is not None:
                stopper.join(2.0)
        assert not failures and not writer.is_alive() and not stopper.is_alive()
        assert stopped.is_set() and not guard.armed


def test_differential_drive_injected_driver_mapping_and_stop_close():
    driver = differential_drive.FakeDriver()
    p = profile()
    adapter = differential_drive.DifferentialDriveAdapter(p, driver)
    with HardwareGuard(adapter, p, monitor=False, clock=Clock()) as guard:
        guard.arm(local=True)
        assert guard.observe() == b"fake-camera-frame"
        guard.command(0, {"left": 0.3, "right": 0.2}, 100.2)
        assert driver.writes == [{2: 0.3, "right-channel": -0.2}]
        guard.on_link_loss()
    assert driver.stops and driver.closed


def test_differential_drive_fake_driver_error_stops():
    class BrokenDriver(differential_drive.FakeDriver):
        def write(self, channels):
            raise OSError("disconnected controller")

    driver, p = BrokenDriver(), profile()
    with HardwareGuard(differential_drive.DifferentialDriveAdapter(p, driver), p,
                       monitor=False, clock=Clock()) as guard:
        guard.arm(local=True)
        with pytest.raises(AdapterError):
            guard.command(0, {"left": 0.1, "right": 0.1}, 100.2)
        assert not guard.armed and len(driver.stops) == 2
    assert driver.closed


def test_differential_drive_module_factory_is_opt_in(monkeypatch):
    driver = differential_drive.FakeDriver()
    calls = []

    def make(config):
        calls.append(config)
        return driver

    path = install_plugin(monkeypatch, make)
    p = profile(adapter_config={"driver_factory": path, "driver_config": {"baud": 9600}})
    adapter = differential_drive.make_adapter(p)
    assert adapter.driver is driver and calls == [{"baud": 9600}]
    adapter.close()
    calls.clear()
    dry = profile(dry_run=True, adapter_config={"driver_factory": path})
    assert type(differential_drive.make_adapter(dry)) is NullAdapter
    assert not calls


@pytest.mark.parametrize("overrides", [
    {"motors": [{"id": "one", "channel": "one"}]},
    {"motors": [{"id": "left", "channel": 2, "max": 2}, {"id": "right", "channel": 3}]},
    {"adapter_config": {"unexpected": 1}}, {"adapter_config": {"driver_config": []}},
])
def test_differential_drive_rejects_bad_configuration(overrides):
    with pytest.raises(ValueError):
        differential_drive.make_adapter(profile(**overrides))


def test_async_integration_keeps_bytes_and_local_policy_separate(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(local_inference, "monotonic", clock)
    received = []

    class Node:
        async def submit(self, observation):
            received.append(observation)
            return SimpleNamespace(payload=b"clear", useful=True)

    p = profile(dry_run=True)
    adapter = load_adapter(p)
    adapter.queue_observation(b"local-camera")
    with HardwareGuard(adapter, p, monitor=False, clock=clock) as guard:
        guard.arm(local=True)
        assert asyncio.run(local_inference.run_once(Node(), guard, local_inference.conservative_rover_policy, 0))
        assert received == [b"local-camera"]
        assert adapter.writes == [{"left": 0.2, "right": 0.2}]


@pytest.mark.parametrize("payload,useful", [
    (b'{"left":1,"right":1}', True), (b"clear", 1), ({"left": 1}, True),
])
def test_async_integration_rejects_remote_commands_and_bad_results(monkeypatch, payload, useful):
    clock = Clock()
    monkeypatch.setattr(local_inference, "monotonic", clock)

    class Node:
        async def submit(self, observation):
            return SimpleNamespace(payload=payload, useful=useful)

    adapter = RecordingAdapter()
    with HardwareGuard(adapter, profile(), monitor=False, clock=clock) as guard:
        guard.arm(local=True)
        with pytest.raises(ValueError):
            asyncio.run(local_inference.run_once(Node(), guard, local_inference.conservative_rover_policy, 0))
        assert not guard.armed and guard.fault_reason and not adapter.writes


def test_async_integration_never_arms_and_nonuseful_disarms(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(local_inference, "monotonic", clock)

    class Node:
        useful = True

        async def submit(self, observation):
            return SimpleNamespace(payload=b"clear", useful=self.useful)

    adapter, node = RecordingAdapter(), Node()
    with HardwareGuard(adapter, profile(), monitor=False, clock=clock) as guard:
        with pytest.raises(GuardError, match="disarmed"):
            asyncio.run(local_inference.run_once(node, guard, local_inference.conservative_rover_policy, 0))
        assert not adapter.writes
    with HardwareGuard(RecordingAdapter(), profile(), monitor=False, clock=clock) as guard:
        guard.arm(local=True)
        node.useful = False
        assert not asyncio.run(local_inference.run_once(node, guard, local_inference.conservative_rover_policy, 0))
        assert not guard.armed


@pytest.mark.parametrize("failure", ["exception", "late", "cancellation"])
def test_async_transport_failure_latency_and_cancellation_failstop(monkeypatch, failure):
    clock = Clock()
    monkeypatch.setattr(local_inference, "monotonic", clock)

    class Node:
        async def submit(self, observation):
            if failure == "exception":
                raise OSError("transport lost")
            if failure == "cancellation":
                raise asyncio.CancelledError()
            clock.now = 101.0
            return SimpleNamespace(payload=b"clear", useful=True)

    expected = {"exception": OSError, "late": GuardError, "cancellation": asyncio.CancelledError}[failure]
    adapter = RecordingAdapter()
    with pytest.raises(expected):
        with HardwareGuard(adapter, profile(), monitor=False, clock=clock) as guard:
            guard.arm(local=True)
            asyncio.run(local_inference.run_once(Node(), guard, local_inference.conservative_rover_policy, 0))
    assert guard.closed and not guard.armed and guard.fault_reason
    assert adapter.stops and not adapter.writes and adapter.close_calls == 1


def test_sdk_import_is_stdlib_only_and_python310_syntax():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import swarm.robotics; "
        "assert not any(n == 'numpy' or n.startswith(('swarm.brain', 'swarm.train', 'swarm.runtime')) "
        "for n in sys.modules); "
        "assert type(swarm.robotics.load_adapter(swarm.robotics.RobotProfile.from_dict({}))) "
        "is swarm.robotics.NullAdapter"
    )
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(ROOT / "python")],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    for folder in (ROOT / "python/swarm/robotics", ROOT / "examples/robots"):
        for path in folder.glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))


@pytest.mark.parametrize("arm", [False, True])
def test_dry_run_cli_runs_without_site_packages_or_hardware(arm):
    args = [sys.executable, "-I", "-S", str(ROOT / "examples/robots/dry_run.py")]
    result = subprocess.run(args + (["--arm"] if arm else []), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{int(arm)} recorded writes; closed=True; no hardware I/O" in result.stdout


def test_dry_run_cli_refuses_hardware_profile_before_import(tmp_path):
    path = tmp_path / "enabled.json"
    path.write_text(json.dumps({"dry_run": False, "adapter": "missing_vendor:make"}), encoding="utf-8")
    result = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "examples/robots/dry_run.py"),
                             "--config", str(path), "--arm"], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "refuses dry_run=false; no adapter has been loaded" in result.stderr