"""Trusted-local plugin boundary. Importing this module never imports a driver."""

from __future__ import annotations

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping

from .profile import DEFAULT_ADAPTER, RobotProfile, _text

_LOG = logging.getLogger(__name__)


class AdapterError(RuntimeError):
    """A plugin could not be loaded, violated its contract, or failed a callback."""


class RobotAdapter(ABC):
    """Synchronous, bounded callbacks. Construction MUST leave hardware inactive."""

    @abstractmethod
    def capabilities(self) -> frozenset[str]:
        """Advertise supported local interfaces, not peer permissions."""

    @abstractmethod
    def observe(self) -> bytes | None:
        """Return pure inference input, or None when there is no observation."""

    @abstractmethod
    def apply(self, outputs: Mapping[str, float]) -> None:
        """Consume a complete logical motor snapshot; map channels in the driver."""

    @abstractmethod
    def safe_stop(self, reason: str) -> None:
        """Request the PLATFORM-SPECIFIC safe state; tolerate repeated calls."""

    @abstractmethod
    def close(self) -> None:
        """Release local resources safely; tolerate repeated calls."""


class NullAdapter(RobotAdapter):
    """In-memory dry run only: no pins, ports, dynamic drivers, or hardware I/O."""

    def __init__(self, profile: RobotProfile, *, suppressed_adapter: str | None = None) -> None:
        self.profile = profile
        self.suppressed_adapter = suppressed_adapter
        self.writes: list[dict[str, float]] = []
        self.stops: list[str] = []
        self.closed = False
        self._observations: deque[bytes] = deque()
        if suppressed_adapter:
            _LOG.warning("DRY RUN: suppressed %s; using in-memory NullAdapter", suppressed_adapter)

    def capabilities(self) -> frozenset[str]:
        return frozenset({"motors", "observe", "dry_run"})

    def queue_observation(self, payload: bytes) -> None:
        if self.closed or not isinstance(payload, bytes):
            raise ValueError("queue_observation requires bytes and an open NullAdapter")
        self._observations.append(payload)

    def observe(self) -> bytes | None:
        if self.closed:
            raise AdapterError("NullAdapter is closed")
        return self._observations.popleft() if self._observations else None

    def apply(self, outputs: Mapping[str, float]) -> None:
        if self.closed:
            raise AdapterError("NullAdapter is closed")
        snapshot = dict(outputs)
        self.writes.append(snapshot)
        _LOG.info("DRY RUN motor snapshot: %s", snapshot)

    def safe_stop(self, reason: str) -> None:
        self.stops.append(reason)
        _LOG.info("DRY RUN safe_stop: %s", reason)

    def close(self) -> None:
        self.closed = True
        self._observations.clear()


def make_null_adapter(profile: RobotProfile) -> RobotAdapter:
    return NullAdapter(profile)


def _sync_callable(callback: object, args: tuple, name: str) -> None:
    if (not callable(callback) or inspect.iscoroutinefunction(callback)
            or inspect.isasyncgenfunction(callback) or inspect.isgeneratorfunction(callback)):
        raise AdapterError(f"{name} must be a synchronous callable")
    try:
        inspect.signature(callback).bind(*args)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{name} has an incompatible or uninspectable signature") from exc


def _validate_adapter(adapter: RobotAdapter, profile: RobotProfile) -> None:
    if not isinstance(adapter, RobotAdapter):
        raise AdapterError("factory must return a RobotAdapter instance")
    for name, args in (("capabilities", ()), ("observe", ()), ("apply", ({},)),
                       ("safe_stop", ("validation",)), ("close", ())):
        _sync_callable(getattr(adapter, name, None), args, f"adapter.{name}")
    try:
        capabilities = adapter.capabilities()
        if not isinstance(capabilities, frozenset):
            raise ValueError("capabilities() must return frozenset[str]")
        for capability in capabilities:
            _text(capability, "advertised capability")
        missing = profile.required_capabilities - capabilities
        if missing:
            raise ValueError(f"adapter lacks required capabilities: {sorted(missing)}")
    except Exception as exc:
        raise AdapterError(f"invalid adapter capabilities: {exc}") from exc


def _cleanup_failed_adapter(adapter: object, reason: str) -> None:
    for name, args in (("safe_stop", (reason,)), ("close", ())):
        try:
            callback = getattr(adapter, name, None)
            _sync_callable(callback, args, name)
            callback(*args)
        except Exception:
            _LOG.exception("Adapter cleanup failed: %s", name)


def load_adapter(profile: RobotProfile) -> RobotAdapter:
    """Load only explicitly trusted local Python code; never install dependencies.

    Dry runs do not even import the requested module, much less call its factory.
    Null capabilities are honest simulated interfaces, not copied from a plugin.
    """
    if not isinstance(profile, RobotProfile):
        raise TypeError("profile must be a RobotProfile")
    adapter = None
    try:
        if profile.dry_run or profile.adapter == DEFAULT_ADAPTER:
            suppressed = profile.adapter if profile.adapter != DEFAULT_ADAPTER else None
            adapter = NullAdapter(profile, suppressed_adapter=suppressed)
        else:
            module, factory_name = profile.adapter.split(":")
            factory = getattr(importlib.import_module(module), factory_name)
            _sync_callable(factory, (profile,), "adapter factory")
            adapter = factory(profile)
        _validate_adapter(adapter, profile)
        return adapter
    except Exception as exc:
        if adapter is not None:
            _cleanup_failed_adapter(adapter, "adapter validation failed")
        raise AdapterError(f"cannot load {profile.adapter}: {exc}") from exc