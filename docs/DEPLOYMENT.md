# Secure, operator-driven USB deployment

This is a **runnable application deployment pipeline**, not a disk image writer or
a remote-control protocol. It runs on an ordinary Raspberry Pi **already booted
into Raspberry Pi OS**, using a mounted USB mass-storage drive to transfer public
enrollment material and a signed, recipient-encrypted application bundle. A bare
USB power cable does **not** upload an application to an ordinary Pi. Nothing here
formats, mounts, flashes, SSHs, changes accounts/ACLs, installs dependencies, starts
services, discovers hardware, or arms actuators automatically.

The existing training, simulation, benchmarking, C++ inference engine, and
[local robotics SDK](ROBOT_ADAPTERS.md) remain tools for designing and testing the
application you explicitly select. Deployment does not replace them or assume
every application uses the tiny CNN. Read [SECURITY.md](SECURITY.md) before using
real identities or hardware. Software tests are not a physical Raspberry Pi test.

## 1. Prepare the operator and Pi environments

- Raspberry Pi OS with **Python 3.10 or later**, a working trusted clock, and an
  operator-owned local account. Older images with Python 3.9 need a newer OS or a
  separately reviewed Python environment. TLS 1.3 also requires a suitable OpenSSL
  in that interpreter. Prefer a 64-bit image on capable Pis.
- `cryptography>=44` is needed for authority creation, enrollment, packing,
  verification and installation. It is **not** imported by deployment `status`,
  `launch` or `target`, or required by the stdlib runtime itself. Numpy and model/
  driver dependencies are separate application choices.
- Bootstrap the reviewed deployment source and interpreter using a trusted
  installation route **before** trusting updates. Put identities, trust anchors,
  the install root and application writable data **outside the repository and
  outside each other**, on protected local storage. On Linux use operator-owned
  0700 directories. Protect Windows directories with operator-only ACLs: chmod
  mode bits do not implement Windows ACL security, and commands warn accordingly.
- Use local ext4 (or another filesystem with POSIX permissions, reliable file
  locking, atomic rename and fsync) for identity/install state. FAT/exFAT is suitable
  for encrypted **transport**, not for private keys or installed release state.
  Network shares, removable install roots and filesystems without these semantics
  are not supported durability/security choices.

One source-checkout setup on the Pi, run deliberately as its unprivileged account
after Python/venv support is available:

```sh
umask 077
python3 -m venv /home/swarm/deploy-venv
/home/swarm/deploy-venv/bin/python -m pip install 'cryptography>=44'
export PYTHONPATH=/home/swarm/toolkit/python
/home/swarm/deploy-venv/bin/python -m swarm.deploy --help
/home/swarm/deploy-venv/bin/python -m swarm.deploy target
```

Here `/home/swarm/toolkit` is the explicitly reviewed checkout. This source route
does not require installing the toolkit's numpy dependency. Alternatively install
the reviewed distribution into the venv, with its selected optional deployment
extra when available. Do not run pip from a payload or add it to an install hook.
The example systemd unit assumes the deployment package is **installed in the
venv**; if using a source checkout instead, an operator must set an appropriate
trusted absolute `PYTHONPATH` for that unit.

### Offline Python and cryptography

Use a trusted machine with the **same OS, Python minor version, ABI and target
architecture** to prepare an audited wheelhouse, or deliberately choose pip's
cross-download platform/ABI tags. Download **binary wheels only** and check that
all dependencies have wheels. A Windows ARM64/x64 wheel is not a Linux aarch64
wheel. Availability varies for 32-bit Pi OS and Python versions; if a wheel is
missing, build/review it on a matching trusted build machine, not during install.

```sh
python3 -m pip download --only-binary=:all: --dest /home/operator/wheelhouse 'cryptography>=44'
/home/swarm/deploy-venv/bin/python -m pip install --no-index --only-binary=:all: --find-links /media/swarm/TRANSFER/wheelhouse 'cryptography>=44'
```

For production, pin exact approved versions and hashes in an operator-maintained
requirements file and use pip's `--require-hashes`. Authenticate the wheelhouse,
deployment source and interpreter out of band: **bootstrap cannot authenticate
itself**. They can travel on USB after review, but are not application-envelope
contents or automatically installed by this subsystem. No complete OS image,
secure-boot configuration or disk encryption is supplied.

## 2. Create the offline fleet authority once

On the trusted operator workstation, with an approved Python environment and
reviewed checkout on its import path:

```sh
python -m swarm.deploy init-authority --authority-dir /home/operator/swarm-authority --name lab-fleet
python -m swarm.deploy fingerprint --public-key /home/operator/swarm-authority/signing-public.pem
python -m swarm.deploy fingerprint --certificate /home/operator/swarm-authority/ca.pem
```

The destination must **not already exist**. The result is:

```text
swarm-authority/              (0700 on POSIX; keep offline/protected)
  signing-key.pem             Ed25519 PRIVATE deployment-signing key, 0600
  signing-public.pem          public deployment trust anchor
  ca-key.pem                  P-256 PRIVATE fleet-CA key, 0600
  ca.pem                      public self-signed fleet CA
```

Provision `signing-public.pem` and `ca.pem` on each Pi into a protected trust
directory, for example `/home/swarm/trust/`. Authenticate their fingerprints using
a **separate trusted channel** (direct trusted console, authenticated provisioning
session, or independently recorded values). Transporting a public key and a hash
beside the update does not establish trust. Never pass a signing key taken from
the USB update to `--trust-key`. Keep CA and signing private keys off every Pi and
off transfer media. The CLI prints public paths/fingerprints, not PEM key contents.

On Windows use Windows paths, for example an existing protected directory under
the operator's home, and the explicitly selected Python interpreter. For this
workspace the selected interpreter is the Store Python 3.13, an x64 interpreter
on an ARM64 machine; `target` reports **windows-x86_64**, not a Pi target.

## 3. Bootstrap each device locally, then enroll it

Run **on the Pi**, not on the authority machine. Each node must have its own
identity and simple node ID matching `[a-z0-9][a-z0-9-]{0,62}`. Example:

```sh
/home/swarm/deploy-venv/bin/python -m swarm.deploy init-device --identity-dir /home/swarm/identity --node-id pi-01 --ip 192.0.2.10
```

`--ip` is optional and repeatable (maximum 16); use the device's actual intended
IP SANs, not the documentation address above. The unique keys are generated with
the target OS randomness. The directory is new, 0700, and contains:

```text
identity/
  identity.json              public schema/node ID, kept locally owner-only
  tls-key.pem                PRIVATE P-256 TLS key; stays on this Pi
  tls.csr.pem                public TLS certificate request
  encryption-key.pem         PRIVATE RSA-3072 bundle-decryption key; stays on this Pi
  encryption-public.pem      public bundle recipient key
  enrollment-request.json    public CSR + recipient key + TLS-key possession proof
```

The CLI prints the exact enrollment-request SHA256. Record it from the trusted Pi
console and send that value to the operator **independently of the USB**. Copy only
the public request (optionally its separate public CSR/recipient PEM) onto the
already mounted USB. Do **not** recursively copy the identity directory. The
enrollment request binds the recipient key and CSR together with a P-256 proof;
the out-of-band digest binds that request to the intended physical device.

On the authority workstation, after checking the node and the independent digest:

```sh
python -m swarm.deploy enroll --authority-dir /home/operator/swarm-authority --request /media/operator/TRANSFER/enrollment-request.json --request-sha256 REPLACE_WITH_64_LOWERCASE_HEX_FROM_PI --output-dir /home/operator/enrolled-pi-01 --days 30
python -m swarm.deploy fingerprint --certificate /home/operator/enrolled-pi-01/tls-cert.pem
```

Replace the digest placeholder; it is not a key or secret. The new output directory
contains **only** the public `tls-cert.pem` and `ca.pem`. Save an approved local
copy of the public request for future `pack --recipient`; do not reuse an unchecked
replacement from a subsequent USB. Keep an operator inventory of node ID, request/
recipient fingerprint and current authorized leaf certificate fingerprint.

Certificates are signed by the common fleet CA, use P-256, have `DNS:pi-01` as their
only DNS SAN (plus the requested IP SANs), and both **serverAuth and clientAuth**
EKUs. A leaf's total validity interval is at most 90 days; `--days` accepts 1–90,
default 30. The start is backdated up to five minutes and the end is bounded by
the CA's validity. Check the Pi's trusted wall clock before TLS use.

Copy the two public enrollment output files to the Pi using mounted USB. Accept
them against the CA **already pinned separately**:

```sh
/home/swarm/deploy-venv/bin/python -m swarm.deploy accept-enrollment --identity-dir /home/swarm/identity --enrollment-dir /media/swarm/TRANSFER/enrolled-pi-01 --ca-cert /home/swarm/trust/ca.pem
/home/swarm/deploy-venv/bin/python -m swarm.deploy fingerprint --certificate /home/swarm/identity/credentials/tls-cert.pem
```

This creates `identity/credentials/{tls-cert.pem,ca.pem}` without replacing any
existing file. It checks the CA pin, leaf signature, validity/usage/SAN, and the
match to the Pi's **existing local TLS private key**. Do not use the returned USB
CA as the trust argument. Renewal uses a fresh enrollment output directory and a
fresh `accept-enrollment --credentials-dir /home/swarm/credentials-next`; update
local runtime config and peer fingerprint pins deliberately. Keys are not silently
regenerated. There is no automatic expiry renewal or same-ID authorization service.

### Runtime integration contract

Supply these operator-controlled paths to the runtime's TLS configuration:

| Purpose | Example path/value |
| --- | --- |
| Common fleet CA | `/home/swarm/identity/credentials/ca.pem` |
| Local public leaf certificate | `/home/swarm/identity/credentials/tls-cert.pem` |
| Local private TLS key | `/home/swarm/identity/tls-key.pem` |
| Expected DNS identity | `pi-01` (exact DNS SAN, independent of IP connection address) |
| Authorized peers | separately approved mapping of node ID to SHA256 **DER leaf certificate** fingerprint |

`certificate_fingerprint()` and the certificate CLI command produce 64 lowercase
hex characters over DER, **not** over PEM text or a public-key SPKI. The common CA
authenticates issuance; it does **not** make every fleet certificate an authorized
peer. Pin leaf fingerprints separately and configure runtime TLS 1.3, mutual
certificate validation, exact node-ID SAN checking and the desired authorization
map. Renewal changes the leaf fingerprint. Deployment does not select peers,
start TLS listeners or implement load balancing; it supplies these verified inputs.

The operator can extract/share each device's public certificate directly from
the enrollment output's `tls-cert.pem`, or from the Pi's credentials directory.
**Never extract/copy the private TLS key.** Install/verify can work using the local
encryption identity before enrollment, but a TLS application is not ready until
its leaf/CA and peer authorization are provisioned.

## 4. Choose a brain/application and pack it

### Desktop selector and minimal payload builder

From the development checkout, install the optional extra and start the setup
program. It has two explicit operations: **prepare selected brain** and **sign,
encrypt and export**. Nothing is run on a device by pressing either button.

```sh
python -m pip install -e ".[deploy]"
python setup_device.py
```

Select an edited copy of [node.example.json](../deploy/raspberry_pi/node.example.json),
a **new** output directory, target and one of:

- A clean custom Python package root, with the config's `brain.factory` set to
  its reviewed `module:factory`. This copies code without importing it. Use a
  matching `workload_id` on peers; dependencies are prepared separately, not fetched.
- An exported SWM model **and** a compatible target-built C++ shared library.
  The builder selects the native factory, copies both assets, sets the model digest
  and a native workload ID. Defaults are empty class 0, threshold 0.5 and an 8 MiB
  inference arena. The CLI exposes `--ram-cap-bytes`; review config before signing.
- Neither: the configured local factory is retained (the example uses the
  built-in hardware-free checksum handler, not an AI brain).

GUI path pickers select existing directories; append a **new child name** for
preparation. For signing, an already prepared/reviewed application directory is
valid. Choose a new bundle filename, application ID, increasing version, recipient
node ID, offline signing-key path, independently trusted public signing key and
approved public enrollment request. Select the existing mounted USB folder only
if a copy is wanted. No private key is read into an on-screen text field or copied
to the payload; only its local path is selected. Tkinter is optional; on a
headless system use the CLI.

Equivalent preparation without the GUI:

```sh
python setup_device.py prepare --config /home/operator/node-pi-01.json --brain-dir /home/operator/reviewed-brain-packages --output /home/operator/prepared-pi-01 --target linux-aarch64
```

For the repository's C++ inference engine, build **on Raspberry Pi OS or a proper
Linux cross toolchain**, not with the Windows MinGW preset:

```sh
cmake -S . -B build-pi -G Ninja -DCMAKE_BUILD_TYPE=Release -DSWARM_NATIVE_ARCH=OFF -DSWARM_BUILD_TESTS=OFF
cmake --build build-pi
```

The current CMake output is `build-pi/bin/swarm_brain.so`. Copy the approved Pi-built
library to the operator workstation, then select it together with the trained
model (the setup program checks recognizable ELF/PE architecture headers):

```sh
python setup_device.py prepare --config /home/operator/node-pi-01.json --model models/tiny_cnn_int8.swm --library /home/operator/pi-build/swarm_brain.so --output /home/operator/prepared-pi-01 --target linux-aarch64 --ram-cap-bytes 8388608
```

The new payload contains `main.py`, `node.json`, `python/swarm/runtime`,
`python/swarm/robotics`, the lightweight namespace, and the selected plugin/assets.
It excludes the simulator, numpy training, benchmarks and device credentials.
The default entrypoint is an **inference-only worker**, never an automatically
armed robot application. A custom local control application must deliberately
integrate the [robot SDK](ROBOT_ADAPTERS.md) and [runtime API](RUNTIME.md).
Use the prepared directory as `--application-dir` in the following `pack` step.

### Pack any deliberately selected application

Select a **clean, deliberately assembled application directory**, not the entire
checkout, venv, home directory or build tree. Everything within it is selected;
unsafe/credential/cache paths are rejected rather than silently omitted. It can
contain reviewed Python source, target-built C/C++ binaries/shared libraries and
source, your chosen model/config, and the required runtime/robotics modules. No
entrypoint, build script, dependency installer or model is executed while packing.

The included [hardware-disabled application](../deploy/raspberry_pi/demo_application/main.py)
is a working checksum smoke test, **not an AI model or motor policy**. Its separate
[local brain module](../deploy/raspberry_pi/demo_application/python/demo_brain.py) and
[configuration](../deploy/raspberry_pi/demo_application/application.json) demonstrate
selection. Pack it directly from the repository on the operator machine:

```sh
python -m swarm.deploy pack --application-dir deploy/raspberry_pi/demo_application --output /home/operator/out/pi-01-v1.swarmbundle --application-id demo --version 1 --node-id pi-01 --target linux-aarch64 --entrypoint main.py --signing-key /home/operator/swarm-authority/signing-key.pem --recipient /home/operator/approved-pi-01/enrollment-request.json
```

Pre-create the output parent; the output file itself must not exist and must be
outside the application directory. `--recipient` also accepts the device's
approved RSA public `encryption-public.pem`. Review its association with `--node-id`;
the target will refuse any mismatch to its local identity/key.

- Run `target` on the **destination interpreter**. Supported tags are
  `linux-aarch64`, `linux-armv7l`, `linux-armv6l`, `linux-x86_64`, `linux-i686`,
  `windows-aarch64`, `windows-x86_64`, `windows-i686`, `darwin-aarch64`,
  `darwin-x86_64`, and `python-any`.
- The lab smoke payload can use `--target python-any` for a same-machine Windows
  test. This tag rejects recognizable native libraries/binaries; it is not an
  architecture bypass for a real Pi build. A Windows ARM64 DLL is **not** a Linux
  ARM64 build. PE/ELF/common Mach-O headers catch obvious OS/CPU mismatches; they do
  not establish libc, instruction-set, driver or dependent-library compatibility.
- The entrypoint must be a listed **relative `.py` file**. A Python launcher can
  deliberately invoke a reviewed target executable or load an explicit C ABI;
  deployment itself never constructs arbitrary shell commands. POSIX execute bits
  from selected source files are recorded and normalized to owner-only permissions.
  When packaging on Windows, verify intended executable bits on a matching trusted
  build host; DLL/shared-library loading does not require Unix execute bits.
- File paths use portable ASCII components, no hidden components, backslashes,
  drive paths, traversal, reserved DOS names, duplicate/case-alias names, links,
  hardlinks, special files, credential-like names or caches. Each component is at
  most 80 characters; paths at most 240 characters and 16 components.
- Bounds: **64 MiB whole envelope**, **256 KiB header**, **512 files**, **32 MiB per
  file**, **128 MiB declared total uncompressed**, ZIP_STORED only. Because the ZIP
  is stored, the 64 MiB envelope cap is normally stricter than 128 MiB. Verification
  holds bounded buffers in memory; peak memory is several times bundle size. Use
  smaller bundles on low-RAM Pis. Compression and ZIP64 are intentionally absent.

For a real inference application, choose the exported model, target-compiled
native engine if used, and a Python entrypoint wrapping your reviewed runtime and
configuration. Package only the subset needed on-device. The `python/` directory
under the selected release is put on `PYTHONPATH`; include a compatible lightweight
`swarm/__init__.py` when packaging runtime/robotics there. Do not ship a second,
unexpected package that shadows part of the intended `swarm` namespace. A parent
payload builder described above prepares these directories; `pack` never infers a brain choice.

Hardware libraries use the [documented RobotAdapter and local factory interface](ROBOT_ADAPTERS.md#stable-public-api).
Keep driver dependencies/pins, bus/channel mapping and safety policy local and
reviewed. For C++ hardware drivers use the separately documented narrow C ABI;
the inference engine ABI is not a motor API. Start with `dry_run=true` and null
hardware. A signed deployment, service start, TLS peer or useful inference result
is **never** permission to arm. No deployment command touches actuators.

## 5. Copy to mounted USB; install explicitly on the Pi

The operator selects an existing mounted mass-storage directory:

```sh
python -m swarm.deploy export-usb --bundle /home/operator/out/pi-01-v1.swarmbundle --usb-dir /media/operator/TRANSFER --trust-key /home/operator/swarm-authority/signing-public.pem
```

On Windows a matching example uses `--usb-dir E:/`. Check the actual mount/drive
yourself. The tool does not enumerate disks or certify that this directory is a
mount. It verifies the signature/schema before copying **only encrypted envelope
bytes** to a new `.swarmbundle` file. An existing filename is an error; use a new
`--filename pi-01-v1-copy.swarmbundle` if needed. Unrelated media files are untouched.
Safely eject using the OS yourself; the tool does not eject or unmount.

On the Pi, after the operator mounts the USB:

```sh
/home/swarm/deploy-venv/bin/python -m swarm.deploy verify --bundle /media/swarm/TRANSFER/pi-01-v1.swarmbundle --identity-dir /home/swarm/identity --trust-key /home/swarm/trust/signing-public.pem --node-id pi-01 --application-id demo
/home/swarm/deploy-venv/bin/python -m swarm.deploy install --bundle /media/swarm/TRANSFER/pi-01-v1.swarmbundle --identity-dir /home/swarm/identity --trust-key /home/swarm/trust/signing-public.pem --root /home/swarm/application-state --dry-run
/home/swarm/deploy-venv/bin/python -m swarm.deploy install --bundle /media/swarm/TRANSFER/pi-01-v1.swarmbundle --identity-dir /home/swarm/identity --trust-key /home/swarm/trust/signing-public.pem --root /home/swarm/application-state --node-id pi-01 --application-id demo
/home/swarm/deploy-venv/bin/python -m swarm.deploy status --root /home/swarm/application-state
```

`verify` authenticates/decrypts and checks all files, but has no install-root history
and therefore does **not** check rollback. `install --dry-run` additionally checks
the existing high-water mark and writes nothing. A dry run reserves nothing; a
subsequent real install must still acquire the root lock and recheck history.

Successful installation produces:

```text
application-state/                 private local directory, 0700
  .install.lock                    persistent OS-locked inode, 0600
  current.json                     atomic version + release selection, 0600
  releases/
    <local-random-32-hex>/          read-only release, 0500 on POSIX
      .swarm-manifest.json         authenticated installed file metadata, 0400
      main.py                     selected application, 0400 (0500 if executable)
      application.json
      python/demo_brain.py
```

Application ID, positive version, node ID, target, entrypoint, complete file list,
lengths, execute policy and SHA256 hashes are authenticated. Installation uses no
peer-controlled release directory name. It stages under the chosen root, fsyncs
files and directories, and atomically replaces **one** current-state file holding
both the selected release and the version high-water mark. **No launch occurs.**

Versions must strictly increase across the **whole root**, including a switch to
another application ID. An existing root cannot switch node identity. Use separate
operator-selected roots for independently versioned applications; choose which to
launch locally. Old releases are retained, not automatically selected on failure.
Failed verification/staging/commit leaves the previous current pointer unchanged.
If syncing after the pointer replacement fails, the CLI explicitly reports commit
uncertainty: inspect `status`, do not blindly roll back or reset history.

An interrupted install can leave an unreferenced random staging/release directory;
there is no automatic destructive garbage collection. Inspect locally and remove
only verified orphaned data during maintenance. The OS lock releases on process
exit/crash; do not delete/replace the persistent lock inode while installers run.
Do not erase or restore an older `current.json` to bypass rollback protection.

## 6. Launch deliberately, then optionally supervise

```sh
/home/swarm/deploy-venv/bin/python -m swarm.deploy launch --root /home/swarm/application-state --node-id pi-01 --application-id demo --identity-dir /home/swarm/identity -- --message hello-pi
```

The launcher checks state and **all installed hashes/files**, enforces the target,
then uses `sys.executable -s -B <release-entrypoint>` with a list of arguments,
**no shell**, and release cwd. Child output is ordinary application output; do not
write secrets in your application logs. Exit status is the child exit status.
`PYTHONPATH` starts with `<release>/python`. Extra paths require repeatable explicit
`--python-path ABSOLUTE_DIRECTORY`, or `--inherit-pythonpath` as an explicit trust
decision. User-site imports and bytecode writes are disabled; the approved venv's
installed packages and normal environment remain trusted, not sandboxed.

The child receives `SWARM_NODE_ID`, `SWARM_APPLICATION_ID`, `SWARM_RELEASE_DIR`, and
optionally `SWARM_IDENTITY_DIR`. Credentials are not copied into the release. The
application must use the supplied identity path plus its separately reviewed local
TLS/peer config. Use an external writable directory for logs, queues or snapshots;
the release is read-only. Updating the current pointer does not restart a running
process; supervision/restart is an explicit operator step.

The [example systemd unit](../deploy/raspberry_pi/swarm-application.service) is a
template for an existing `swarm` account with reviewed absolute paths and a
pre-created private writable data directory. It uses no elevated capabilities,
read-only identity/releases, private devices, no-new-privileges and
`KillMode=control-group`, so the Python launcher **and its child** belong to the
same supervised unit. It hides the bundle decryption key from that unit. It
intentionally exposes no hardware devices. Copying, enabling or starting the unit,
or granting any real device access, is a separate local administrative decision;
no command here does it. Review systemd-version compatibility and application
resource limits before use.

### Optional external route: USB Ethernet and scp

A compatible Pi/model/OS can be configured **by the operator** for USB Ethernet,
or connected through a normal USB Ethernet adapter. This is not universal gadget
support and is not enabled by this package. After independently verifying the
SSH host key and account, ordinary operator-run `scp` may transfer the same
encrypted bundle or the **public** enrollment files:

```sh
scp /home/operator/out/pi-01-v1.swarmbundle swarm@pi-01:/home/swarm/incoming/
```

Then run the identical local `install` command on the Pi, using the incoming
file path. Never scp the authority keys or device private keys. No custom USB
network upload protocol or remote installer is introduced.

## Python API (exact public call signatures)

Import from `swarm.deploy`. `Pathish = str | os.PathLike[str]`. Validation/trust
failures raise `DeployError` (`ValueError`); filesystem failures remain `OSError`
subclasses. The CLI reports these on stderr and returns 2, without key material.

```python
init_authority(directory: Pathish, *, name: str = "swarm-fleet") -> Path
init_device(directory: Pathish, *, node_id: str, ip_addresses: Iterable[str] = ()) -> Path
enroll(request: Pathish, *, authority_dir: Pathish, output_dir: Pathish,
       request_sha256: str, days: int = 30) -> Path
accept_enrollment(identity_dir: Pathish, *, enrollment_dir: Pathish,
                  ca_certificate: Pathish, credentials_dir: Pathish | None = None) -> Path
certificate_fingerprint(certificate: Pathish) -> str
public_key_fingerprint(public_key: Pathish) -> str
request_fingerprint(request: Pathish) -> str
current_target() -> str

pack(application_dir: Pathish, output: Pathish, *, application_id: str, version: int,
     node_id: str, target: str, entrypoint: str, signing_key: Pathish,
     recipient: Pathish) -> Path
verify_bundle(bundle: Pathish, *, identity_dir: Pathish, trust_key: Pathish,
              node_id: str | None = None, application_id: str | None = None) -> VerifiedBundle
export_usb(bundle: Pathish, usb_dir: Pathish, *, trust_key: Pathish,
           filename: str | None = None) -> Path
install(bundle: Pathish, *, identity_dir: Pathish, trust_key: Pathish, root: Pathish,
        node_id: str | None = None, application_id: str | None = None,
        dry_run: bool = False) -> Deployment
status(root: Pathish, *, node_id: str | None = None,
       application_id: str | None = None) -> Deployment | None
launch(root: Pathish, *, node_id: str | None = None, application_id: str | None = None,
       identity_dir: Pathish | None = None, args: Sequence[str] = (),
       trusted_python_paths: Sequence[Pathish] = (), inherit_pythonpath: bool = False) -> int
```

- `init_device` returns the public request path; other identity operations return
  their new output directory. There is no shared default/demo private identity.
  Tests call these same functions in temporary directories for a real crypto lab.
- `FileRecord(path, size, sha256, executable)` and
  `Manifest(application_id, version, node_id, target, entrypoint, files)` are frozen
  dataclasses. `Manifest.from_dict()` strictly validates a manifest;
  `manifest.as_dict()` includes `unpacked_bytes` and complete file records.
- `VerifiedBundle(manifest, contents, bundle_sha256)` contains validated bytes in
  manifest order and the whole envelope digest; verification does not extract it.
- `Deployment(manifest, release, bundle_sha256, dry_run=False)` describes the
  selected/verified release; a dry run has `release=None`. `as_dict()` is safe
  public status metadata. Public import is lazy: non-crypto APIs remain stdlib-only.

## Focused verification and integration

[tests/test_deploy.py](../tests/test_deploy.py) exercises real crypto enrollment,
pack/verify/install/status/launch, an end-to-end CLI smoke application, foreign
targets/recipients/trust, tampering at each layer, hostile ZIPs and paths, version
replay, atomic-commit and staging failure injection, lock contention, USB collision
protection and no-site-package imports. Generated leaf credentials also complete a
stdlib TLS 1.3 mutual-authentication handshake using in-memory BIOs, including a
negative DNS-name check; no network listener is started. POSIX permission tests are platform-gated;
symlink tests run only where the test account can create links. No CMake/full battery
or physical hardware execution is required by these tests.

Verified on 2026-09-07 with this workspace's selected Store Python 3.13.14 and
cryptography 50.0.1: **107 passed, 3 skipped**. The skips are two symlink
tests requiring Windows symlink privileges and one POSIX-only permissions test.
All ten owned Python files also passed a Python 3.10 grammar check; this is not a
claim of testing a Python 3.10 interpreter or Raspberry Pi OS itself.

The optional `deploy` extra, desktop [setup program](../setup_device.py),
`dev.py deploy` / `dev.py setup` shortcuts and [live runtime](RUNTIME.md) are wired
into the repository. [test_device_setup.py](../tests/test_device_setup.py) exercises
prepared runtime → encrypted bundle → verified install → hardware-disabled launch.
The existing toolkit commands and baseline-comparison scenarios remain separate.