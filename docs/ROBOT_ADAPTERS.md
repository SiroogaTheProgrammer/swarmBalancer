# Local robot adapters — safety boundary and SDK

This small Python 3.10+ SDK connects a **local application** to a locally installed
driver. It is **not flight control, an autopilot, an emergency-stop system, or a
network actuator-command service**. Importing it uses only the Python standard
library; it does not import numpy, training, inference engines, or vendor drivers.
The existing simulator and C++ engine remain the testing toolkit. The separate
[live inference runtime](RUNTIME.md) and [USB deployment workflow](DEPLOYMENT.md)
use this SDK only through an explicitly written local application.

## Safety requirements before connecting hardware

- Default configuration is `dry_run: true`, with a no-I/O `NullAdapter`. Nothing
  discovers hardware, chooses GPIO pins, opens serial ports, or installs packages.
- Factory/import paths and configuration are **trusted local executable code and
  operator-owned files**, not data from peers. Python plugins are not sandboxed.
  A plugin can bypass this SDK; review, pin and secure its package and configuration.
- Construction must leave outputs inactive. Drivers must implement bounded,
  synchronous callbacks, idempotent stopping/closing, and safe cleanup if their
  factory raises. Do not energize a motor from an import, constructor, sensor read,
  `capabilities()`, or resource cleanup. Only guarded `apply()` may request motion.
- `safe_stop(reason)` means a **platform-specific safe-state request**, not a
  universal zero-output command. A ground rover might brake or coast. An aircraft
  must request a validated **controlled hold/land through its autopilot**, not
  blindly switch off motors in flight. Satellite attitude control and thrusters
  need their own independently reviewed safe-mode authority and interlocks.
  **Do not use the differential-drive example for drones, aircraft or thrusters.**
- The watchdog is software, not hard real-time or independent safety. A deadlocked
  driver, held callback lock, blocked interpreter/GIL, OS scheduling delay, power
  loss or process crash can prevent it from running. A request to stop is not
  evidence that the hardware stopped. Inspect `guard.stop_error` and arrange
  independent feedback; failed stopping requires external intervention.
- Use a **hardware watchdog and physical emergency stop**, independent of Python,
  plus platform-appropriate limits and local interlocks. Establish those protections
  first. Test with actuators disconnected or wheels lifted in a controlled area.

## Stable public API

Import from `swarm.robotics`:

| API | Contract |
| --- | --- |
| `RobotProfile.from_dict(data)` / `RobotProfile.load(Path(...))` | Strict versioned local configuration; invalid profiles raise `ValueError`. File errors propagate normally. |
| `RobotAdapter` | ABC: `capabilities() -> frozenset[str]`, `observe() -> bytes \| None`, `apply(outputs: Mapping[str, float]) -> None`, `safe_stop(reason: str) -> None`, `close() -> None`. |
| `load_adapter(profile) -> RobotAdapter` | Import the explicitly selected `module:factory` using `importlib`; call `factory(profile)`. Verify subclass, synchronous callable signatures, capability value types and required capabilities. Raise `AdapterError` on failure; best-effort stop/close returned invalid plugins. |
| `NullAdapter(profile)` / `make_null_adapter(profile)` | Record copied `writes`, `stops`, and `closed` entirely in memory. `queue_observation(bytes)` supplies a test observation. `observe()` otherwise returns `None`. |
| `HardwareGuard(adapter, profile, *, monitor=True, clock=time.monotonic)` | Start disarmed and request the initial platform-specific safe state. Own and serialize effective-adapter callbacks. Use as a context manager. |
| `guard.arm(local=True)` | Explicit **local operator authorization**, never called by a peer or inference handler. Merely entering a context does not arm. Repeated arm is not a keepalive. |
| `guard.command(sequence, outputs, expires_at)` | Validate a complete logical motor snapshot and local monotonic deadline before applying it. Raises `GuardError` on rejection, `AdapterError` on callback failure. |
| `guard.observe()` | Serialized observation callback; accepts only bytes or `None`. Observation does not refresh the watchdog. |
| `guard.poll(now=None) -> bool` | Check expiry deterministically; return whether still armed. Expiry requests a stop and latches the fault. Explicit `now` and custom clocks are **local test hooks**, not remote inputs. |
| `guard.disarm(reason="local disarm")` | Stop without clearing faults or sequence history. A healthy manual disarm may be followed by another explicit local arm. |
| `guard.on_link_loss()` | Immediately request safe stop and permanently disarm/fault this guard. Wire local transport-health notifications to this method. |
| `guard.close()` | Disarm, request stop if needed, close resources and stop/join the monitor. Idempotent. A close failure is reported; context cleanup preserves an already-raised application exception. |
| `guard.armed`, `closed`, `fault_reason`, `stop_error` | Read-only state. `guard.adapter` exposes the **effective** adapter for diagnostics; do not bypass the guard to call hardware methods. `guard.profile` is the validated profile. |
| `MotorConfig`, `Attachment`, `map_motor_outputs(profile, outputs)` | Validated descriptors and the shared logical-id → explicit-channel/inversion helper for external drivers. |

Any invalid command, disarmed command, replay, callback failure, link loss or
watchdog expiry **latches a fault**. Polling/receiving more results does not rearm
or repeatedly issue the same fault stop. There is no fault-reset/network-arm API.
Close a faulted guard, investigate locally, and only construct/authorize a new one
after the platform is independently safe. Keep one owner/guard per physical driver;
never reuse a closed or still-owned adapter in a second guard.

The sequence is a nonnegative Python `int` (not `bool`), strictly increasing for the
entire guard lifetime, including manual disarm/rearm. Generate it **locally**, not
from received payloads. `expires_at` is finite local `time.monotonic()` time in
`(now, now + watchdog_s]`; wall-clock timestamps and peer clocks are not accepted
as authority. Equality with the deadline is expired. A command's shorter TTL wins
over the watchdog. Initial arming also expires after `watchdog_s` without a command.
Late commands cannot revive an expired earlier command, even if no poll ran yet.
An expired deadline after a blocking callback is detected as soon as it returns.

The daemon monitor starts **only on context entry**, uses `threading.Event.wait`,
and is stopped on fault/close. `monitor=False` is for deterministic tests or an
application that reliably calls `poll()` itself. Outside a context no daemon is
started. All adapter callbacks, including reads, stops and closing, share a lock;
callbacks must not reenter their guard or perform unbounded I/O. A lock cannot
make a blocking driver safe: the independent hardware watchdog remains required.

## Configuration schema (version 1)

The fully runnable example configuration is [examples/robots/profile.json](../examples/robots/profile.json).
Its left/right wheel names are **fake channel labels, not suggested pin numbers**.
The camera/bumper are descriptors for virtual attachments; nothing opens them.

- Root keys: `schema_version` (default `1`), `adapter` (default
  `swarm.robotics:make_null_adapter`), `adapter_config` (default `{}`), `motors` and
  `attachments` (default `[]`), `required_capabilities` (default `[]`), `dry_run`
  (default `true`), and `watchdog_s` (default `0.5`, finite and `0 < value <= 60`).
  No arming flag is part of the profile. Unknown keys/versions and duplicate JSON
  object keys are rejected, not ignored.
- Each motor has required `id` and **explicit** `channel` (nonnegative integer or
  nonempty string), `inverted` (boolean, default `false`), `min` and `max` (finite
  driver-unit numbers, defaults `-1` and `1`, with `min < max`). IDs are strings;
  channel names and integer indices have no universal hardware meaning.
- Every command supplies **all configured motor IDs**, once each, and no others.
  Empty/partial snapshots are rejected: otherwise refreshing one wheel could keep
  another wheel's old value alive indefinitely. Values must be finite numbers,
  not booleans or numeric strings. There is no implicit clipping or scaling.
- The guard validates logical values; a driver calls `map_motor_outputs()` exactly
  once to replace IDs with channels and apply `inverted`. Limits must hold **both
  before and after sign inversion**. Symmetric limits are recommended for reversed
  motors; an asymmetric inverted motor may reject some otherwise logical in-range
  values. This avoids mapping into an unsafe physical value. The differential-drive
  example further restricts limits to normalized `[-1, 1]`.
- Each attachment has `id`, a descriptive `type` string, and `connection` with
  exactly `interface` and `endpoint`. Interfaces are `virtual`, `gpio`, `i2c`, `spi`,
  `serial`, `usb`, or `can`; endpoint is an explicit nonempty string interpreted only
  by the external driver. Put vendor-specific bus/address settings in
  `adapter_config`, not unknown attachment fields. Descriptors do not grant capabilities.
- Motor/attachment IDs are globally unique. Motor channels are unique, also rejecting
  the ambiguous pair `1` and `"1"`. Device-specific aliases, bus conflicts and physical
  pin restrictions must additionally be checked by the external driver.
- `required_capabilities` contains unique nonempty strings. Advertised capabilities
  must actually be a `frozenset[str]`; missing requirements are errors. The null
  implementation honestly advertises only `motors`, `observe`, `dry_run` (simulated
  interfaces), not arbitrary camera/autopilot capabilities.
- `adapter_config` is a detached dictionary of JSON-compatible values, with finite
  numbers and string object keys. Only the plugin knows its inner schema and must
  reject unsupported keys/settings. Profile safety fields and descriptors are
  immutable; treat driver settings as fixed after configuration validation.

### Dry-run enforcement

`load_adapter()` substitutes the bundled null implementation **before importing
the selected plugin or calling its factory** whenever `dry_run` is true. Missing
vendor libraries are irrelevant in that mode. The substitution logs a `DRY RUN`
warning and records `NullAdapter.suppressed_adapter`. Real capability requirements
that the null adapter cannot advertise still fail safely; dry runs do not certify
hardware capability. Even with `dry_run=false`, the default adapter remains null.

The guard also substitutes a null adapter for directly supplied real instances
or null subclasses when `dry_run=true`, without invoking **any** of their methods.
The caller still owns such a suppressed, already-created instance; the guard cannot
undo its constructor's side effects or safely assume that closing it is inert.
Always call `load_adapter(profile)` **before** constructing anything hardware-facing.
Do not select hardware by importing and constructing a driver ahead of the dry-run
check. Dry-run enforcement is not a sandbox against malicious same-process code.

## Runnable, hardware-disabled examples

[examples/robots/dry_run.py](../examples/robots/dry_run.py) runs directly from a checkout
without installation. Run it normally for an explicitly disarmed demonstration,
or supply its local `--arm` flag for one **in-memory** command. It refuses any
configuration with `dry_run=false` before loading an adapter. No sockets, GPIO,
serial device, vendor package, or inference model are involved.

[examples/robots/differential_drive.py](../examples/robots/differential_drive.py) defines
`DifferentialDriveAdapter(profile, driver)` and a `FakeDriver`. The driver implements
`read_observation()`, batched `write(channel_values)`, platform-specific `stop(reason)`
and `close()`. Motor IDs are `left` and `right`; their physical channels and inversion
come only from configuration. `make_adapter(profile)` optionally resolves a trusted
local `adapter_config.driver_factory` with signature `factory(driver_config)`. It
uses a fake driver if none is chosen. Imports of external drivers occur inside the
factory only, never at module import. The plugin validates its own configuration
before calling a driver factory. A batched write does not promise atomicity unless
the external controller really supports it.

## Pure inference networking; local application policy only

Runtime networking distributes **observation bytes** and returns an
`InferenceResult(payload: bytes, useful: bool)`. It must never execute actuator
commands from peers. The robotics SDK has no runtime dependency and exposes no
network listener, peer-controlled arming, deserialization-to-actuator path, or
remote factory/configuration loader. Do not carry `guard.command()` arguments,
profile files, driver import paths, sequence numbers or deadlines over the wire.

[examples/robots/local_inference.py](../examples/robots/local_inference.py) demonstrates
the explicit caller-owned integration. Its `run_once()` deliberately **does not
arm**. The containing local application creates a guard context and may separately
call `guard.arm(local=True)` **only after a real local operator/interlock decision**.
No useful result, reconnection, application startup, or profile flag is permission
to arm. The core flow is:

```python
# In an async local application; guard was explicitly authorized locally elsewhere.
# local_policy is locally installed/reviewed code, not code or actuator JSON from peers.
try:
    observation = guard.observe()
    if observation is None:
        guard.disarm("no observation")
    else:
        result = await node.submit(observation)
        if not isinstance(result.payload, bytes) or type(result.useful) is not bool:
            raise ValueError("invalid inference result")
        if result.useful:
            outputs = local_policy(result.payload)
            guard.command(sequence, outputs,
                          expires_at=time.monotonic() + min(0.1, guard.profile.watchdog_s))
            sequence += 1  # Local counter, never a peer's sequence.
        else:
            guard.disarm("no useful inference")
except Exception:
    guard.on_link_loss()  # Latch a safe state for transport AND policy errors.
    raise
# An outer 'with HardwareGuard(...) as guard:' also stops/closes on cancellation.
```

The example policy recognizes only `b"clear"` and `b"obstacle"`, mapping those labels
to locally chosen bounded rover values. It **does not** parse remote motor mappings
or execute remote code. `useful=True` is an inference optimization hint, not a safety
assessment or proof that a scene is clear. Real policies must validate model output,
apply local safety interlocks and separately enforce acceptable observation age.
Awaiting a slow inference does not extend a running actuator deadline; if it expires,
a subsequently received result cannot rearm or renew it. Configure the control loop
and independent watchdog for realistic latency, not just the demonstration numbers.

## Shipping a driver outside this toolkit

Create a separately reviewed Python package exporting `your_robot_package:make_adapter`.
Implement `RobotAdapter` and all five methods; advertise only capabilities actually
provided. Inside the factory, validate `profile.adapter_config` **before** opening
devices, import optional vendor packages there, and construct an inactive driver.
Use `map_motor_outputs(profile, outputs)` in `apply()`; never apply inversion twice.
If the factory fails after opening anything, it must stop/close those resources
itself because no adapter was returned to the loader. Missing dependencies are
reported, not automatically fetched. Install/package exact approved dependencies
as a separate operator-controlled deployment step.

For a C++ driver, wrap a narrow C ABI with stdlib `ctypes` in that same external
package: load only an explicitly configured trusted absolute library path **inside
the factory**, set every `argtypes` and `restype`, check status codes, and manage
handle ownership, integer widths, calling convention and library/interpreter
architecture. Do not use this toolkit's inference ABI to control motors. Provide
bounded write/stop/close functions in the hardware library and keep exception
handling on the correct side of the ABI. A C function can hang too; Python cannot
turn it into a hard real-time or independent watchdog.

The standalone payload can contain just this SDK's
[python/swarm/robotics](../python/swarm/robotics/__init__.py), an otherwise lightweight
`swarm` namespace, your local application, the external driver package/shared
library, and reviewed configuration. The SDK uses relative imports and does not
need training/simulation/model code. Alternatively use this distribution with an
operator-prepared environment; the repository's overall package metadata includes
numpy, but **this SDK itself does not need or import it**. Do not change the toolkit's
global initialization to auto-import a driver. Ship only the runtime pieces and
licenses/dependencies you have explicitly audited; testing this SDK does not certify
a physical platform.

## Verification

[tests/test_robotics.py](../tests/test_robotics.py) covers strict schemas, duplicate
keys/IDs/channels, finite limits and inputs, mapping/inversion, plugin contracts,
dry-run factory suppression, explicit arming, replay, TTL/watchdog, latched link
loss, serialization, failing drivers, cleanup, byte-only inference integration and
the dry-run CLI. It tests deterministic clocks as well as a bounded Event-based
daemon check. Isolated subprocess checks disable site packages to verify the SDK
and example do not need numpy or a hardware library. No CMake build, full-suite
execution, vendor installation, or physical hardware testing is part of this SDK.