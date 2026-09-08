# Deployment security model and limits

This document describes the `swarm.deploy` application-envelope and local-install
boundary. It does not claim the entire toolkit, a Pi OS image, a driver, a model,
or a running robot is secure or safe. See [DEPLOYMENT.md](DEPLOYMENT.md) for concrete
commands and [ROBOT_ADAPTERS.md](ROBOT_ADAPTERS.md) for the hardware safety boundary.
There is no claim of an “unbreakable” system or physical Raspberry Pi testing.

## Trust boundaries

Trusted: the operator, independently provisioned signing public key and fleet CA,
the bootstrap interpreter/deployment package/cryptography build, approved signing
workstation and storage, local owner-controlled install/identity directories,
the OS/kernel, and application code that the operator approves for execution.

Untrusted: USB content, incoming bundle bytes and metadata, removable-media file
names, unauthenticated enrollment transfers, peers not individually authorized,
and any build/source directory not yet explicitly reviewed for inclusion.

Protect identity/install/trust paths and **all their parent directories** against
replacement by another local account. The implementation rejects symlinks (including
Windows reparse points), hardlinked files, unsafe POSIX ownership/permissions and
nonregular files, uses exclusive creation and no-follow opens where available,
and stages only within the private root. It is not a race-safe sandbox against a
malicious process with the **same UID**, Windows ACL rights, root, or an already
compromised OS. Do not let an untrusted process write these directories while
verification, installation or launching is in progress.

## Key separation and bootstrap

| Key | Location | Purpose |
| --- | --- | --- |
| Ed25519 signing private key | offline protected operator authority only | approve application envelope bytes |
| P-256 CA private key | offline protected operator authority only | issue fleet TLS identities |
| P-256 TLS private key | generated and retained on each destination | runtime mutual TLS |
| RSA-3072 decryption private key | generated and retained on each destination | unwrap that device's application key |
| AES-256 payload key | fresh random in-memory key per pack | encrypt one application archive |

The authority and identity directories must be new; no identity is overwritten,
silently regenerated, shared among devices or included as demo credentials. Private
files use `O_CREAT|O_EXCL` and 0600 on POSIX; new directories use 0700. Public
identity/authority outputs are initially owner-only too. POSIX permission checks
are enforced; Windows emits an explicit ACL-responsibility warning because mode
bits cannot enforce equivalent access. Windows ACL setup is a manual prerequisite.

Keys are PKCS#8 PEM **unencrypted at rest**, with no passphrase automation. Filesystem
permissions are access control, not encryption. Use operator-managed encrypted
storage, protected backups and secure bootstrap/boot where appropriate. Do not put
private files in source control, logs, raw USB transfers or application directories.
No deployment command prints private material; an arbitrary launched application
can print anything it can read, so its logging remains an operator responsibility.

The public enrollment request includes a signed CSR and RSA recipient public key
bound by a P-256 signature under the CSR key. `enroll` **requires** a SHA256 of the
exact request obtained out of band. A digest copied on the same untrusted USB is
not a device-authentication check. The enrolling operator approves the request's
physical device/node association and retains an approved public recipient copy.

Returned enrollment material is only a public leaf certificate and CA certificate.
`accept-enrollment` verifies it against a **separately provisioned** CA, the exact
local TLS key, node DNS SAN, leaf constraints/EKUs and a validity interval no longer
than 90 days. Public keys/certificates transported with updates are never treated
as new trust anchors. Trust/signing-key rotation is a separate authenticated local
provisioning operation, not a payload feature.

## Application envelope version 1

Standard maintained primitives come from `cryptography>=44`; this is an application
envelope using established ciphers, not a newly invented cipher or USB protocol.

```text
8 bytes   magic: ASCII SWMDEP1 followed by NUL
4 bytes   unsigned big-endian canonical-JSON header length
N bytes   canonical ASCII JSON header
C bytes   AES-256-GCM ciphertext including its 16-byte tag
64 bytes  Ed25519 signature
```

The header has exactly `schema`, `algorithm`, `manifest`, `recipient_sha256`,
`nonce`, `wrapped_key`, and `ciphertext_bytes`. Algorithm is fixed to
`Ed25519+AES-256-GCM+RSA-3072-OAEP-SHA256`; algorithm negotiation is not implemented.
Canonical JSON uses sorted keys, no whitespace, ASCII escaping, no NaN/infinity,
and no duplicate keys. Unrecognized fields/schema versions fail closed.

1. Generate a fresh random 32-byte AES key and 12-byte nonce for each pack.
2. Wrap the AES key using RSA-3072 OAEP with SHA256 and MGF1-SHA256, label
   `swarm.deploy/application-envelope/key-wrap/v1` followed by NUL.
3. Encrypt the restricted ZIP using AES-GCM. Associated data is
   `swarm.deploy/application-envelope/aead/v1` followed by NUL, then **the exact
   magic, length prefix and header bytes**.
4. Sign `swarm.deploy/application-envelope/signature/v1` followed by NUL, then
   **the exact magic, length prefix, header and ciphertext/tag bytes** using Ed25519.

Before signature verification, only fixed framing, total input length and a bounded
header-length field are examined. No JSON metadata is interpreted, no recipient
key is used, no payload path is chosen, and no decryption/extraction occurs until
the Ed25519 signature has verified against the independently trusted public key.
Then the canonical header/schema/manifest, node, recipient and **local OS/interpreter
target** are checked, followed by RSA unwrap, GCM authentication and full archive/
file validation. A re-signed header change still fails GCM unless encryption is
also correctly recomputed by an authorized packer.

The manifest authenticates application ID, monotonically positive version, node
ID, target, relative Python entrypoint, complete ordered file list, sizes,
normalized executable policy, total length and each file's SHA256. The recipient
identifier is SHA256 of public RSA SubjectPublicKeyInfo DER. A bundle is bound to
one device and does not grant decryption to other fleet members.

**Metadata is public**: filenames, sizes, hashes, application/node identifiers,
version and recipient fingerprint are signed but not encrypted. File contents are
encrypted. An attacker can delete, truncate, delay or replace USB files, observe
metadata, or deny service, but cannot approve a changed executable without the
signing key. There is no forward secrecy for archived envelopes: later compromise
of a device RSA key permits decrypting previously captured bundles for that device.
Python does not promise reliable zeroization of key/plaintext buffers.

## Bounded archive and file handling

- Maximum whole envelope 64 MiB; header 256 KiB; 512 files; 32 MiB per file; total
  declared uncompressed 128 MiB. Actual ZIP_STORED content must also fit the envelope.
  Bounded buffers still consume several times bundle size; low-RAM operators must
  choose smaller payloads. There is no streaming unbounded `read_all` or compression.
- ZIP_STORED only, no encryption inside ZIP, ZIP64, multi-disk files, data descriptors,
  archive/member comments, extra fields, symlink/directory/special-file entries,
  group/world access, setuid/setgid/sticky files or user-selected extraction modes.
- Before constructing `ZipFile`, a bounded structural scan checks the end record,
  central-directory extent, actual entry count/lengths, method and sizes. It rejects
  a false small declared count hiding many real entries. Local headers, filenames,
  sizes, CRCs, contiguous nonoverlapping offsets and the absence of hidden/trailing
  entries must match the central directory and authenticated manifest.
- No `extractall`, pickle or object deserialization, arbitrary file-permission
  restoration, install hooks, subprocesses or dynamic imports during pack/verify/
  install. Every regular file must be listed exactly once and match length/SHA256.
- Paths reject absolute/drive/UNC forms, backslashes, traversal, NULs, hidden
  components, reserved DOS devices, trailing dots, ASCII case aliases, conflicting
  file/directory prefixes and excess length/depth. Empty hidden cache directories
  are rejected too. There is no OS-dependent archive path normalization shortcut.
- Common credential/cache paths/extensions and PEM/private-key markers are refused
  during pack and authenticated content verification. **This is not a complete
  secret scanner**: a password/token/model-embedded secret with an innocuous name
  cannot be reliably detected. Review the explicitly selected payload; never select
  a whole home directory, repository or virtual environment as a convenience.
- PE/ELF/common Mach-O checks reject recognizable foreign binaries, including a
  Windows ARM64 library mislabeled as a Pi build. They are not static analysis,
  malware detection, ABI certification or evidence of compatible dependencies.

## Installation atomicity and anti-rollback

An OS file lock serializes installs per root and releases on process death. It
uses a persistent inode; do not unlink it while another installer might hold it.
Authenticated files are written exclusively into a locally random staging tree,
made owner-only read-only, synced, and renamed to a random immutable release name.
`os.replace` commits **one** current-state JSON file that contains both the version
high-water mark and release path, plus manifest/bundle hashes. The installer never
updates the current pointer and high-water mark as separate operations.

Versions must increase across the entire root, even when application ID changes.
An existing root cannot change device identity. Invalid signatures, keys, targets,
hashes or archives and failed staging/commit do not change the previous pointer.
No fallback to an old release occurs automatically. Status/launch validate the
entire current release and fail closed on missing/extra/mutated files or state
disagreement. A failed directory fsync **after** replacement is reported as commit
uncertainty, not a claimed rollback. Use reliable local POSIX storage and inspect
status after a power failure. Windows file flushing/atomic replacement is exercised
by tests, but POSIX directory-fsync durability is not available on Windows.

This is local software anti-rollback, **not** a TPM/hardware monotonic counter. An
attacker with storage/root access can restore an older complete state/OS image or
delete history. A local owner can deliberately chmod/write “immutable” files or
reset the root. Protect backups and high-water history; do not claim that this
design prevents physical snapshot rollback. Recovery/garbage collection is an
explicit operator procedure; unreferenced crash remnants are never auto-executed.

## Runtime, authorization and approved-code limits

The fleet CA is common, but authorization requires independently pinned **SHA256
DER leaf certificate** fingerprints mapped to exact node DNS SANs. The
[live runtime](RUNTIME.md) enforces TLS 1.3 mutual certificate validation and those
pins, and checks its own identity before listening. Deployment provides
the P-256 server/client-EKU credentials and public fingerprint helpers; it does not
implement peer enrollment over the network, automatically trust every CA-issued
peer, or configure runtime policy. Certificate expiry needs a trusted clock and
renewal needs an explicit local credential/pin update. There is no automated CRL/
OCSP service; remove compromised leaf pins promptly in every affected peer's
operator-controlled authorization map.

Installing a valid signature means **the signer approved these bytes**, not that
the code/model/driver is correct, benign or safe. `launch` executes that approved
Python application with the selected interpreter, no shell, release cwd and
explicit trusted import paths. It is **not a Python sandbox**. Same-user approved
code can access TLS credentials and other data allowed by its OS rights, call
subprocesses, use the network or bypass a library-level safety guard. Malicious
approved code, a compromised signing workstation, dependency supply-chain attack,
root compromise, DMA or physical storage access are outside the envelope's
protection. Minimize libraries/privileges; review payloads and the bootstrap chain.

The example systemd unit is a conservative, hardware-disabled starting point,
with read-only release/identity, the RSA decryption key hidden, no capabilities,
private devices and cgroup-wide shutdown. It does not make arbitrary Python code
safe. Neither TLS, bundle signing, an application restart nor an inference result
can authorize motion. Physical robots require the SDK's explicit local policy,
independent interlocks/watchdogs and platform-appropriate emergency handling.

## Verification claims

The focused [deployment tests](../tests/test_deploy.py) run actual maintained crypto
primitives and exercise adversarial signed/unsigned inputs, the complete CLI,
public enrollment, USB collisions, archive bounds, replay/downgrade, lock contention,
failure injection and local release integrity. Generated certificates complete a
TLS 1.3 mutual-authentication handshake over stdlib in-memory BIOs, with a negative
DNS-SAN check and DER leaf-fingerprint comparisons. Tests never import an actuator
driver or access hardware. Windows tests cannot certify POSIX permissions/fsync,
real Pi instruction sets, USB reliability, power-loss behavior, secure boot,
systemd/device policy or a live TLS topology. Those need operator verification on
the intended hardware plus an independent security review before production use.

The separate [runtime tests](../tests/test_runtime.py) additionally exercise a real
two/three-peer **loopback** TLS topology and worker-loss recovery on the development
PC. They do not certify radio behavior or physical devices. Runtime packets have
bounded lengths, no compression/pickle, pinned workload identities and bounded
worker/request/reply/cache state. TLS handshakes still consume resources before
post-handshake pin checks; firewall/rate controls and OS resource limits are needed
against flooding. Authorized peers can lie about load or inference, not execute
protocol-supplied code. Local signed plugins remain privileged by their OS account.