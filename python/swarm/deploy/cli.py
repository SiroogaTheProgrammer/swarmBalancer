"""Operator CLI. Prints public metadata only; no implicit execution or provisioning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ._common import TARGETS, DeployError, current_target, device_node_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m swarm.deploy",
        description="Offline signed/encrypted application transfer for operator-owned devices. "
                    "Never mounts, flashes, SSHs, installs dependencies, starts services or arms hardware.")
    commands = parser.add_subparsers(dest="command", required=True)

    authority = commands.add_parser("init-authority", help="create a NEW offline signing authority and fleet CA")
    authority.add_argument("--authority-dir", required=True, type=Path)
    authority.add_argument("--name", default="swarm-fleet")

    device = commands.add_parser("init-device", help="run ON the target; generate NEW private keys locally")
    device.add_argument("--identity-dir", required=True, type=Path)
    device.add_argument("--node-id", required=True)
    device.add_argument("--ip", action="append", default=[], help="optional IP SAN; repeat at most 16 times")

    enroll = commands.add_parser("enroll", help="approve an out-of-band-pinned public enrollment request")
    enroll.add_argument("--request", required=True, type=Path)
    enroll.add_argument("--request-sha256", required=True, help="exact request SHA256 obtained independently of USB")
    enroll.add_argument("--authority-dir", required=True, type=Path)
    enroll.add_argument("--output-dir", required=True, type=Path)
    enroll.add_argument("--days", type=int, default=30, help="certificate lifetime, 1..90 days (default 30)")

    accept = commands.add_parser("accept-enrollment", help="validate returned certs using an independently trusted CA")
    accept.add_argument("--identity-dir", required=True, type=Path)
    accept.add_argument("--enrollment-dir", required=True, type=Path)
    accept.add_argument("--ca-cert", required=True, type=Path, help="independently provisioned fleet CA, NOT USB trust-on-first-use")
    accept.add_argument("--credentials-dir", type=Path, help="NEW directory for renewal; default identity-dir/credentials")

    pack = commands.add_parser("pack", help="sign/encrypt an explicitly selected, reviewed application directory")
    pack.add_argument("--application-dir", required=True, type=Path)
    pack.add_argument("--output", required=True, type=Path)
    pack.add_argument("--application-id", required=True)
    pack.add_argument("--version", required=True, type=int)
    pack.add_argument("--node-id", required=True)
    pack.add_argument("--target", required=True, choices=sorted(TARGETS))
    pack.add_argument("--entrypoint", required=True, help="listed relative .py entrypoint")
    pack.add_argument("--signing-key", required=True, type=Path)
    pack.add_argument("--recipient", required=True, type=Path,
                      help="approved public encryption PEM or retained public enrollment request")

    export = commands.add_parser("export-usb", help="copy a verified encrypted bundle into a chosen existing mounted directory")
    export.add_argument("--bundle", required=True, type=Path)
    export.add_argument("--usb-dir", required=True, type=Path)
    export.add_argument("--trust-key", required=True, type=Path)
    export.add_argument("--filename", help="new safe .swarmbundle basename; never overwrites")

    for command in ("verify", "install"):
        sub = commands.add_parser(command, help=("verify without extracting" if command == "verify"
                                                 else "verify and atomically prepare a release; never launch"))
        sub.add_argument("--bundle", required=True, type=Path)
        sub.add_argument("--identity-dir", required=True, type=Path)
        sub.add_argument("--trust-key", required=True, type=Path,
                         help="independently provisioned Ed25519 public key, never a key from the update")
        sub.add_argument("--node-id", help="optional extra assertion against the local identity")
        sub.add_argument("--application-id", help="optional extra application selection pin")
        if command == "install":
            sub.add_argument("--root", required=True, type=Path)
            sub.add_argument("--dry-run", action="store_true", help="check crypto and anti-rollback; write nothing")

    for command in ("status", "launch"):
        sub = commands.add_parser(command, help=("check active release integrity without executing" if command == "status"
                                                 else "explicitly execute the selected installed .py application"))
        sub.add_argument("--root", required=True, type=Path)
        sub.add_argument("--node-id")
        sub.add_argument("--application-id")
        if command == "launch":
            sub.add_argument("--identity-dir", type=Path, help="external identity exposed as SWARM_IDENTITY_DIR, not copied")
            sub.add_argument("--python-path", action="append", default=[], type=Path,
                             help="explicit trusted absolute import directory; repeat as needed")
            sub.add_argument("--inherit-pythonpath", action="store_true",
                             help="also trust all absolute directories in the operator's PYTHONPATH")
            sub.add_argument("args", nargs=argparse.REMAINDER, help="application arguments after --")

    fingerprint = commands.add_parser("fingerprint", help="print a public certificate, key or request SHA256")
    source = fingerprint.add_mutually_exclusive_group(required=True)
    source.add_argument("--certificate", type=Path, help="SHA256 of certificate DER (runtime leaf pin)")
    source.add_argument("--public-key", type=Path, help="SHA256 of public SubjectPublicKeyInfo DER")
    source.add_argument("--request", type=Path, help="SHA256 of exact public enrollment request bytes")
    commands.add_parser("target", help="report this OS/interpreter's native target tag")
    return parser


def _print(value) -> None:
    print(json.dumps(value, sort_keys=True, indent=2))


def _execute(args: argparse.Namespace) -> int:
    command = args.command
    if command == "target":
        _print({"target": current_target()})
    elif command in {"init-authority", "init-device", "enroll", "accept-enrollment", "fingerprint"}:
        from . import identities

        if command == "init-authority":
            directory = identities.init_authority(args.authority_dir, name=args.name)
            _print({"authority_dir": str(directory),
                    "signing_public_sha256": identities.public_key_fingerprint(directory / "signing-public.pem"),
                    "ca_sha256": identities.certificate_fingerprint(directory / "ca.pem")})
        elif command == "init-device":
            request = identities.init_device(args.identity_dir, node_id=args.node_id, ip_addresses=args.ip)
            _print({"node_id": args.node_id, "request": str(request),
                    "request_sha256": identities.request_fingerprint(request),
                    "encryption_public_sha256": identities.public_key_fingerprint(request.parent / "encryption-public.pem")})
        elif command == "enroll":
            directory = identities.enroll(args.request, authority_dir=args.authority_dir, output_dir=args.output_dir,
                                          request_sha256=args.request_sha256, days=args.days)
            _print({"enrollment_dir": str(directory),
                    "certificate_sha256": identities.certificate_fingerprint(directory / "tls-cert.pem"),
                    "ca_sha256": identities.certificate_fingerprint(directory / "ca.pem")})
        elif command == "accept-enrollment":
            directory = identities.accept_enrollment(args.identity_dir, enrollment_dir=args.enrollment_dir,
                                                     ca_certificate=args.ca_cert, credentials_dir=args.credentials_dir)
            _print({"credentials_dir": str(directory), "node_id": device_node_id(args.identity_dir),
                    "certificate_sha256": identities.certificate_fingerprint(directory / "tls-cert.pem")})
        else:
            if args.certificate:
                value = identities.certificate_fingerprint(args.certificate)
                kind = "certificate-DER"
            elif args.public_key:
                value = identities.public_key_fingerprint(args.public_key)
                kind = "public-SPKI-DER"
            else:
                value = identities.request_fingerprint(args.request)
                kind = "enrollment-request-bytes"
            _print({"kind": kind, "sha256": value})
    elif command in {"pack", "export-usb", "verify"}:
        from . import bundle

        if command == "pack":
            path = bundle.pack(args.application_dir, args.output, application_id=args.application_id,
                               version=args.version, node_id=args.node_id, target=args.target,
                               entrypoint=args.entrypoint, signing_key=args.signing_key, recipient=args.recipient)
            _print({"bundle": str(path)})
        elif command == "export-usb":
            path = bundle.export_usb(args.bundle, args.usb_dir, trust_key=args.trust_key, filename=args.filename)
            _print({"exported": str(path), "note": "No mount, formatting or eject was performed."})
        else:
            verified = bundle.verify_bundle(args.bundle, identity_dir=args.identity_dir, trust_key=args.trust_key,
                                             node_id=args.node_id, application_id=args.application_id)
            _print({"verified": True, "bundle_sha256": verified.bundle_sha256,
                    "manifest": verified.manifest.as_dict(), "anti_rollback_checked": False})
    else:
        from . import installer

        if command == "install":
            result = installer.install(args.bundle, identity_dir=args.identity_dir, trust_key=args.trust_key,
                                       root=args.root, node_id=args.node_id, application_id=args.application_id,
                                       dry_run=args.dry_run)
            _print(result.as_dict())
        elif command == "status":
            selected = installer.status(args.root, node_id=args.node_id, application_id=args.application_id)
            _print({"installed": selected is not None, **(selected.as_dict() if selected else {})})
        else:
            forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
            return installer.launch(args.root, node_id=args.node_id, application_id=args.application_id,
                                    identity_dir=args.identity_dir, args=forwarded,
                                    trusted_python_paths=args.python_path, inherit_pythonpath=args.inherit_pythonpath)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _execute(args)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "cryptography" or exc.name.startswith("cryptography.")):
            print("deployment crypto commands require the optional cryptography>=44 dependency; "
                  "prepare the approved deployment environment first", file=sys.stderr)
            return 2
        raise
    except (DeployError, OSError) as exc:
        print(f"deployment error: {exc}", file=sys.stderr)
        return 2