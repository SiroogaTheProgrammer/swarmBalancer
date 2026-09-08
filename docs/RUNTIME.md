# Device runtime — encrypted, load-balanced inference

This is the **on-device component**, separate from the existing testing toolkit.
It runs on Python 3.10+ / OpenSSL with TLS 1.3, including Raspberry Pi OS. It does
not require numpy, training, a simulator, or a vendor motor library. Optional
inference kernels run through the existing C++ C ABI. Bare microcontrollers are
not Python deployment targets; the portable C++ kernels remain usable separately.

## Boundaries

| Component | Responsibility |
| --- | --- |
| [SwarmRuntime](../python/swarm/runtime/node.py) | Authenticated peers, bounded work, load-aware scheduling, expiry/retries and completion |
| [Local brain factory](../python/swarm/runtime/handlers.py) | Pure, idempotent observation bytes → inference result |
| [Robot SDK](ROBOT_ADAPTERS.md) | Local attachment/motor descriptions, trusted driver factory, explicit arming and safety guard |
| [Device setup](../setup_device.py) | Select runtime/config/brain code or model and prepare a minimal payload |
| [Deployment](DEPLOYMENT.md) | Enroll identities; authenticate/encrypt USB packages; install atomically; launch explicitly |
| Testing toolkit | Existing C++ tests, numpy training, discrete-event scenarios, stress tests, benchmark battery |

No protocol message imports a module, downloads a model, changes configuration,
installs software, launches a process, arms hardware, or controls a motor. Peers
only submit observations and return inference bytes. A locally signed/configured
brain is executable trusted code, **not sandboxed**. The local application must
validate results before its own policy uses them. Neither `useful=True` nor a TLS
identity proves a physical scene is safe.

## Try the real encrypted transport on this PC

Install the optional deployment/test dependencies into the chosen environment:

```sh
python -m pip install -e ".[dev,deploy]"
python dev.py runtime-demo
```

The demo creates unique temporary identities, starts two loopback TLS listeners,
offloads a SHA256 smoke job, closes the worker, and completes a subsequent job
locally. It prints public counters, not credentials, then closes listeners and
removes temporary files. **SHA256 is a transport smoke handler, not an AI model.**
This checks actual sockets/TLS rather than simulator time; it accesses no robot.

## Configure each device

Start with [node.example.json](../deploy/raspberry_pi/node.example.json). The example
has no peers: it can serve local inference after enrollment but does not discover
or trust other nodes. Configure both directions of every desired peer link. Example
peer entry (replace the fingerprint with the separately approved DER leaf digest):

```json
{
  "node_id": "pi-02",
  "host": "192.0.2.12",
  "port": 7443,
  "fingerprint": "REPLACE_WITH_64_LOWERCASE_HEX_CERTIFICATE_SHA256"
}
```

Documentation IPs must be replaced with real private network addresses. The peer
ID is the certificate's exact DNS SAN; it is independent of its IP. The lower
lexicographic node ID initiates each connection, eliminating simultaneous dial
duplicates. Each configured link is persistent and bidirectional; full mesh is
simple for small swarms. Only **directly connected** peers participate in a node's
scheduler. There is no multi-hop routing or automatic mesh-radio setup.

Configure:

- `node_id`, `workload_id`: the identity and exact input/model/output contract.
  Every peer on a link must have the same workload ID. For custom plugins, bind
  it to the approved code, model and preprocessing/output version. The preparer
  binds the native SWM handler's ID to the model SHA256 and handler contract version.
- `tls.ca`, `tls.certificate`, `tls.private_key`: local, operator-provisioned files.
  The example uses `${SWARM_IDENTITY_DIR}` supplied by the deployment launcher.
  Private files never enter application bundles. Changing approved certificate
  pins or renewing credentials requires a deliberate local config update/restart.
- `host`, `port`: explicit listening address; library default is loopback. The Pi
  example deliberately uses `0.0.0.0`. Restrict the port using the OS/network firewall.
- `queue_limit` (1–64): total accepted jobs, **including the running job**. One
  inference thread per runtime. `max_submissions` (1–256) bounds input owners waiting
  for replies. `max_payload` defaults to 64 KiB, maximum 1 MiB, for input/output.
- `heartbeat_s`, `peer_timeout_s`: periodic load announcements and read timeout.
  Timeout must exceed twice the heartbeat period. `connect_timeout_s` bounds TLS
  negotiation; failed links retry with backoff capped at five seconds.
- `job_timeout_s` (up to 300 seconds), `retries` (0–8): owner deadline and retry
  budget. Different nodes should use compatible TTL policies; use the same value
  unless there is a deliberate reason to refuse longer remote jobs.
- `estimated_ms`: initial service-time estimate; actual completed inference times
  update it with an exponential moving average. The scheduler uses estimated finish
  time, advertised load and locally reserved capacity, not round-robin alone.
- `cache_entries` (0–256): short-lived completed-result deduplication, default 128.
  Worst-case result memory is roughly `cache_entries * max_payload`, plus queued
  inputs/results, TLS buffers and model memory. Set smaller limits on small devices.

Unknown config fields, duplicate JSON keys, invalid IDs/limits and nonfinite values
are errors. Relative TLS paths are relative to the config. Relative brain assets
in the worker CLI are also relative to the config's directory.

## Run a worker or a local input-owning application

After provisioning identities as described in [DEPLOYMENT.md](DEPLOYMENT.md):

```sh
python -m swarm.runtime --config node.json --check
python -m swarm.runtime --config node.json
```

`--check` validates schema, TLS support and the local certificate/key/CA, expiry,
usage and exact node-ID SAN with an in-memory mutual handshake. It neither imports
the selected brain nor opens a network listener. It does not test peer reachability.

The worker CLI is intentionally inference-only. An input-owning local application
uses the same instance to capture/preprocess observations and submit jobs:

```python
from swarm.runtime import ApplicationConfig, SwarmRuntime
from swarm.runtime.handlers import load_handler

app = ApplicationConfig.load(config_path)
handler = load_handler(app.factory, app.options)
async with SwarmRuntime(app.node, handler) as node:
    result = await node.submit(preprocessed_observation_bytes, timeout_s=1.0)
    # Validate locally; result is NOT an actuator command or permission to arm.
```

The factory is a trusted, locally installed `module:factory(options_dict)` returning
a synchronous `handler(bytes) -> InferenceResult(payload: bytes, useful: bool)`.
Missing dependencies fail; no automatic fetching. Factories and handlers must not
have hardware side effects. A plugin cannot be reliably sandboxed inside Python.
Do not share one mutable handler between runtimes. The native handler serializes
its own calls/close defensively, but a blocked C call still needs process supervision.

The optional native factory takes explicit `model`, `library`, `sha256`, and positive
`ram_cap_bytes`; `threshold` and `empty_class` are optional. Input is **uint8 CHW**
already resized to the model shape. C++ receives normalized float32. Output is
eight bytes, network-byte-order `uint32 class_index, float32 confidence`, only when
the predicted class is not empty and exceeds threshold. Export a final softmax.
The C++ arena cap covers inference model buffers, **not** the whole Python process.
No distributed layer-splitting or uncompressed camera capture is claimed here;
custom local factories can implement reviewed preprocessing/inference contracts.

## Failure and retry semantics

- The node that owns the observation schedules it; no single leader must stay up
  to perform inference. `leader_id` / `preferred_leader` is a diagnostic preference
  among directly connected nodes, **not consensus, a lease, or motor authority**.
  A partition may have multiple preferences; a local application must not use
  this as exclusive ownership of shared actuators or flight roles.
- Disconnects reject affected pending replies. Busy/error/timeout outcomes can
  reassign to another available worker or local compute within the original
  deadline. The budget is divided across the permitted attempts; retries are not
  unlimited and can reduce the time available to an individual slow job.
- `JobError` means no timely result is available. There is no durable observation
  journal: an owner process crash loses its outstanding jobs. Remaining members
  can keep submitting theirs. Applications own any persistence/recovery policy.
- Jobs use random 128-bit IDs. A worker deduplicates active/recent `(owner, ID)`
  pairs and rejects reuse with different bytes. This is bounded, in-memory,
  **not exactly-once across workers/restarts**. All handlers must be pure/idempotent.
- Expired waiting jobs are skipped, late results are discarded. Timeout does not
  kill an executing thread or pretend its capacity is free. A hung plugin can
  keep its worker and Python process alive even after async shutdown. Use bounded
  native code and independently supervised processes for untrusted/possibly hung
  workloads; do not reuse/close a running native handle from another owner.
- Cached/replayed results apply only to identical bytes within the cache lifetime;
  application observation-age limits remain local. No remote wall clock controls
  motor deadlines. The local submission timer bounds the result's usefulness even
  when remote clocks differ.

## Bandwidth and cryptography, honestly

TLS 1.3 mutual certificate validation plus pinned SHA256 DER leaf identities and
exact SANs are mandatory. No plaintext, TLS 1.2 fallback, wildcard/common-name
identity fallback or automatic TOFU. CA issuance alone does not authorize a peer.
Established connections rotate at least hourly so certificates are rechecked.
Removing a pin takes effect through the explicit config/restart workflow. Sessions
use the OpenSSL TLS 1.3 AEAD/key exchange, not a custom stream cipher; the optional
deployment library supplies certificate tooling, not an alternate live encryption.

Records use a 5-byte type/length header; load announcements are 8 bytes, jobs add
20 bytes (ID/relative TTL), results add 17 bytes (ID/useful flag). A non-useful
inference sends an **empty completion**, not full output. Silence would be
indistinguishable from a lost job and cause needless retries.

These are application sizes, **not on-air sizes**. TLS records, handshakes, TCP/IP,
radio framing, retransmissions and keepalives cost additional bandwidth. Public
counters explicitly say `application_bytes_*`. Payloads are bounded raw bytes, no
base64, pickle, JSON image expansion or compression bombs. Preprocess large camera
images locally. No fragmentation/multiplexed priorities or free control channel is
implemented: TCP has head-of-line blocking, and a heartbeat cannot overtake a frame
already sent. On a 9.6 kbps link even 1 KiB takes about 0.85 s before overhead;
64 KiB takes about 55 s. Reduce payload/rate and increase timeouts accordingly.

TLS does not stop RF jamming, traffic analysis, bandwidth exhaustion, a compromised
authorized worker returning false results, a stolen/root-readable key, or malicious
approved brain code. This implementation is not independently security audited or
flight-certified. See [SECURITY.md](SECURITY.md) for bootstrap and storage limits.

## Connect motors and attachments only through local policy

Use [ROBOT_ADAPTERS.md](ROBOT_ADAPTERS.md) and
[local_inference.py](../examples/robots/local_inference.py). `RobotProfile` describes
motors (logical IDs, explicit channels, inversion and bounds), attachments
(GPIO/I²C/SPI/serial/USB/CAN endpoints) and required capabilities. The library itself
does not guess pins or vendor packages. An external factory provides the driver.

Start in dry run, physically isolate actuators, validate an independent watchdog
and platform emergency handling, and require explicit local arming. Loss of a link
required by the local safety policy, or invalid inference, must latch a local
safe-state request; late/reconnected peers
never rearm it. For aircraft, safe state means a reviewed hold/land through a real
autopilot, not zeroing motors in flight. No real motor-control library is bundled.

## Verification

[test_runtime.py](../tests/test_runtime.py) exercises real loopback mutual TLS,
two/three-member links, capacity reservation, wrong pins/SANs/workloads, disconnect
retry, TTL, deduplication, length bounds and the native C++ inference bridge.
[test_device_setup.py](../tests/test_device_setup.py) installs and launches the
prepared runtime through authenticated encrypted bundles. Tests do not validate
Raspberry Pi RF links, GPIO, USB media reliability, power-loss durability or vehicle
control. Those remain operator commissioning tasks on the actual hardware.