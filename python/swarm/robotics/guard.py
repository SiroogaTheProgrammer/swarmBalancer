"""Local actuator gate, not a flight controller or an independent safety system."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from types import MappingProxyType

from .adapters import AdapterError, NullAdapter, RobotAdapter, _cleanup_failed_adapter, _validate_adapter
from .profile import RobotProfile, _checked_outputs, _finite, _text

_LOG = logging.getLogger(__name__)


class GuardError(RuntimeError):
    """A local operation was rejected. Faults remain latched for this guard."""


class HardwareGuard:
    """Serialize ALL callbacks; explicitly arm locally; expire complete snapshots.

    Production clocks/deadlines use time.monotonic(). A supplied clock and poll(now)
    are test hooks, not a way for a peer to supply time or authorize commands.
    """

    def __init__(self, adapter: RobotAdapter, profile: RobotProfile, *, monitor: bool = True,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(profile, RobotProfile) or type(monitor) is not bool or not callable(clock):
            raise TypeError("expected RobotProfile, bool monitor and callable clock")
        # Even a caller-supplied subclass of NullAdapter could perform hardware I/O.
        # Do not call or close a suppressed instance: it remains the caller's resource.
        if profile.dry_run and type(adapter) is not NullAdapter:
            adapter = NullAdapter(profile, suppressed_adapter=f"{type(adapter).__module__}:{type(adapter).__name__}")
        self._adapter, self._profile = adapter, profile
        self._clock, self._monitor = clock, monitor
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._armed = self._closed = self._entered = False
        self._fault_reason: str | None = None
        self._stop_error: Exception | None = None
        self._last_sequence = -1
        self._last_now: float | None = None
        self._deadline: float | None = None
        self._needs_stop = True
        with self._lock:
            try:
                _validate_adapter(adapter, profile)
                error = self._stop_locked("initially disarmed")
                if error:
                    raise AdapterError("initial safe_stop failed") from error
            except Exception:
                _cleanup_failed_adapter(adapter, "guard initialization failed")
                self._closed = True
                raise

    @property
    def adapter(self) -> RobotAdapter:
        """Effective adapter, possibly substituted. Never bypass this guard to apply."""
        return self._adapter

    @property
    def profile(self) -> RobotProfile:
        return self._profile

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def fault_reason(self) -> str | None:
        with self._lock:
            return self._fault_reason

    @property
    def stop_error(self) -> Exception | None:
        """A failed stop request means physical safety is NOT confirmed."""
        with self._lock:
            return self._stop_error

    def _stop_locked(self, reason: str, *, fault: bool = False) -> Exception | None:
        if fault and self._fault_reason is not None:
            return self._stop_error
        self._armed = False
        self._deadline = None
        if fault:
            self._fault_reason = reason
            self._done.set()
        if not self._needs_stop and not fault:
            return None
        self._needs_stop = False
        try:
            if self._adapter.safe_stop(reason) is not None:
                raise TypeError("safe_stop() must return None")
        except Exception as exc:
            self._stop_error = exc
            self._fault_reason = self._fault_reason or f"safe_stop failed: {reason}"
            self._done.set()
            _LOG.exception("Robot safe_stop failed; independent safety is required")
            return exc
        return None

    def _reject_locked(self, reason: str) -> None:
        self._stop_locked(reason, fault=True)
        raise GuardError(reason)

    def _healthy_locked(self) -> None:
        if self._closed:
            raise GuardError("guard is closed")
        if self._fault_reason is not None:
            raise GuardError(f"guard fault is latched: {self._fault_reason}")

    def _now_locked(self, now: float | None = None) -> float:
        try:
            value = _finite(self._clock() if now is None else now, "monotonic time")
        except Exception as exc:
            self._stop_locked("invalid monotonic clock", fault=True)
            raise GuardError("invalid monotonic clock") from exc
        if self._last_now is not None and value < self._last_now:
            self._reject_locked("monotonic time moved backwards")
        self._last_now = value
        return value

    def _poll_locked(self, now: float) -> bool:
        if self._armed and self._deadline is not None and now >= self._deadline:
            self._stop_locked("watchdog/command deadline expired", fault=True)
        return self._armed

    def arm(self, *, local: bool = False) -> None:
        """Local operator authorization only. No config, inference, or peer arming."""
        with self._lock:
            self._healthy_locked()
            if local is not True:
                self._reject_locked("arming requires explicit local=True authorization")
            now = self._now_locked()
            self._poll_locked(now)
            self._healthy_locked()
            if self._armed:
                self._reject_locked("already armed; arm is not a watchdog heartbeat")
            self._deadline = now + self._profile.watchdog_s
            if self._deadline <= now:
                self._reject_locked("clock cannot represent watchdog interval")
            self._armed = self._needs_stop = True

    def disarm(self, reason: str = "local disarm") -> None:
        with self._lock:
            if self._closed or self._fault_reason is not None:
                return
            # Manual disarm must not erase an already-expired lease and enable rearm.
            self._poll_locked(self._now_locked())
            if self._fault_reason is not None:
                return
            try:
                _text(reason, "disarm reason")
            except ValueError as exc:
                self._reject_locked(str(exc))
            error = self._stop_locked(reason)
            if error:
                raise AdapterError("safe_stop failed during disarm") from error

    def command(self, sequence: int, outputs: Mapping[str, float], expires_at: float) -> None:
        with self._lock:
            self._healthy_locked()
            now = self._now_locked()
            # A late packet must not revive an expired command before poll gets a turn.
            self._poll_locked(now)
            self._healthy_locked()
            if not self._armed:
                self._reject_locked("command rejected while disarmed")
            try:
                if type(sequence) is not int or sequence < 0 or sequence <= self._last_sequence:
                    raise ValueError("sequence must be a strictly increasing nonnegative integer")
                deadline = _finite(expires_at, "expires_at")
                if not now < deadline <= now + self._profile.watchdog_s:
                    raise ValueError("expires_at must be fresh local monotonic time within watchdog_s")
                snapshot = _checked_outputs(self._profile, outputs)
            except Exception as exc:
                self._stop_locked(f"invalid command: {exc}", fault=True)
                raise GuardError(f"invalid command: {exc}") from exc
            # Mapping iteration can take time. Check BOTH the previous lease and
            # this command's TTL again immediately before committing a hardware write.
            now = self._now_locked()
            self._poll_locked(now)
            self._healthy_locked()
            if not self._armed or sequence <= self._last_sequence or now >= deadline:
                self._reject_locked("command became disarmed, replayed or expired during validation")
            self._last_sequence, self._deadline = sequence, deadline
            try:
                if self._adapter.apply(MappingProxyType(snapshot)) is not None:
                    raise TypeError("apply() must return None")
            except Exception as exc:
                self._stop_locked("adapter.apply failed", fault=True)
                raise AdapterError("adapter.apply failed") from exc
            self._poll_locked(self._now_locked())
            self._healthy_locked()

    def observe(self) -> bytes | None:
        """Use this wrapper, rather than adapter.observe(), to serialize sensor I/O."""
        with self._lock:
            self._healthy_locked()
            self._poll_locked(self._now_locked())
            self._healthy_locked()
            try:
                payload = self._adapter.observe()
                if payload is not None and not isinstance(payload, bytes):
                    raise TypeError("observe() must return bytes or None")
            except Exception as exc:
                self._stop_locked("adapter.observe failed", fault=True)
                raise AdapterError("adapter.observe failed") from exc
            self._poll_locked(self._now_locked())
            self._healthy_locked()
            return payload

    def poll(self, now: float | None = None) -> bool:
        """Return whether still armed; expiry stops once and permanently latches."""
        with self._lock:
            if self._closed or self._fault_reason is not None:
                return False
            return self._poll_locked(self._now_locked(now))

    def on_link_loss(self) -> None:
        with self._lock:
            if not self._closed:
                self._stop_locked("link loss", fault=True)

    def _watch(self) -> None:
        interval = min(0.05, self._profile.watchdog_s / 4)
        while not self._done.wait(interval):
            try:
                self.poll()
            except Exception:
                with self._lock:
                    self._stop_locked("watchdog monitor failed", fault=True)
                return

    def __enter__(self) -> HardwareGuard:
        with self._lock:
            self._healthy_locked()
            if self._entered or self._armed:
                self._reject_locked("enter a guard context once, before local arming")
            self._entered = True
            if self._monitor:
                self._thread = threading.Thread(target=self._watch, name="swarm-robot-watchdog", daemon=True)
                try:
                    self._thread.start()
                except Exception:
                    self._stop_locked("watchdog monitor could not start", fault=True)
                    _cleanup_failed_adapter(self._adapter, "watchdog startup failed")
                    self._closed = True
                    raise
        return self

    def close(self) -> None:
        self._done.set()
        failure = None
        with self._lock:
            if self._closed:
                return
            if self._fault_reason is None:
                failure = self._stop_locked("guard closed")
            try:
                if self._adapter.close() is not None:
                    raise TypeError("close() must return None")
            except Exception as exc:
                self._stop_locked("adapter.close failed", fault=True)
                failure = exc
            finally:
                self._closed, self._armed, self._deadline = True, False, None
        # Never join under the callback lock: the monitor may be waiting for it.
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        if failure:
            raise AdapterError("adapter stop/close failed") from failure

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            with self._lock:
                if not self._closed:
                    self._stop_locked(f"local application exception: {type(exc).__name__}", fault=True)
        try:
            self.close()
        except AdapterError:
            if exc is None:
                raise
            _LOG.exception("Cleanup failed; preserving the original local application exception")
        return False