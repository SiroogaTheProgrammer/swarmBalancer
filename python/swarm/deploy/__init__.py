"""Operator-driven, signed and encrypted application deployment (optional crypto extra).

Importing this namespace does not import cryptography or any runtime/hardware code.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ._common import DeployError, Deployment, FileRecord, Manifest, VerifiedBundle, current_target

if TYPE_CHECKING:
    from .bundle import export_usb, pack, verify_bundle
    from .identities import (
        accept_enrollment, certificate_fingerprint, enroll, init_authority, init_device,
        public_key_fingerprint, request_fingerprint,
    )
    from .installer import install, launch, status

_MODULES = {
    "init_authority": "identities", "init_device": "identities", "enroll": "identities",
    "accept_enrollment": "identities", "certificate_fingerprint": "identities",
    "public_key_fingerprint": "identities", "request_fingerprint": "identities",
    "pack": "bundle", "verify_bundle": "bundle", "export_usb": "bundle",
    "install": "installer", "status": "installer", "launch": "installer",
}
__all__ = ["DeployError", "Deployment", "FileRecord", "Manifest", "VerifiedBundle",
           "current_target", *_MODULES]


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value