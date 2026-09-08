"""Offline authority and on-target identity bootstrap. Private keys never travel."""

from __future__ import annotations

import base64
import binascii
import hmac
import ipaddress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from cryptography import x509
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from ._common import (
    MAX_PUBLIC_BYTES, DeployError, Pathish, canonical_json, check_directory, device_node_id, digest,
    exact_keys, fsync_directory, identifier, parse_json, private_directory,
    read_regular, sha256_hex, windows_acl_warning, write_new,
)

REQUEST_CONTEXT = b"swarm.deploy/enrollment-request/v1\x00"
RSA_BITS = 3072


def _public_bytes(key: Any) -> bytes:
    return key.public_bytes(serialization.Encoding.PEM,
                            serialization.PublicFormat.SubjectPublicKeyInfo)


def _private_bytes(key: Any) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _key_fingerprint(key: Any) -> str:
    return digest(key.public_bytes(serialization.Encoding.DER,
                                   serialization.PublicFormat.SubjectPublicKeyInfo))


def _load_private(path: Pathish, kind: type):
    windows_acl_warning()
    check_directory(Path(path).absolute().parent, private=True)
    try:
        key = serialization.load_pem_private_key(read_regular(path, MAX_PUBLIC_BYTES, private=True),
                                                  password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise DeployError("invalid, encrypted, or inaccessible private key") from exc
    if not isinstance(key, kind):
        raise DeployError("private key has the wrong algorithm")
    if isinstance(key, ec.EllipticCurvePrivateKey) and not isinstance(key.curve, ec.SECP256R1):
        raise DeployError("TLS and CA keys must use ECDSA P-256")
    if isinstance(key, rsa.RSAPrivateKey) and key.key_size != RSA_BITS:
        raise DeployError("device encryption keys must use RSA-3072")
    return key


def _load_public(data: bytes, kind: type):
    if len(data) > MAX_PUBLIC_BYTES or b"PRIVATE KEY" in data:
        raise DeployError("expected a bounded public key, not private key material")
    try:
        key = serialization.load_pem_public_key(data)
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise DeployError("invalid public key") from exc
    if not isinstance(key, kind):
        raise DeployError("public key has the wrong algorithm")
    if isinstance(key, rsa.RSAPublicKey):
        if key.key_size != RSA_BITS or key.public_numbers().e != 65537:
            raise DeployError("recipient must use RSA-3072 with exponent 65537")
    return key


def load_signing_public(path: Pathish) -> ed25519.Ed25519PublicKey:
    return _load_public(read_regular(path, MAX_PUBLIC_BYTES), ed25519.Ed25519PublicKey)


def _certificate(data: bytes) -> x509.Certificate:
    if len(data) > MAX_PUBLIC_BYTES or data.count(b"-----BEGIN CERTIFICATE-----") != 1:
        raise DeployError("expected exactly one PEM certificate")
    try:
        certificate = x509.load_pem_x509_certificate(data)
    except (ValueError, UnsupportedAlgorithm) as exc:
        raise DeployError("invalid PEM certificate") from exc
    if certificate.public_bytes(serialization.Encoding.PEM).strip() != data.strip():
        raise DeployError("certificate contains unexpected trailing or non-PEM data")
    return certificate


def certificate_fingerprint(certificate: Pathish) -> str:
    """Lowercase SHA256 of the DER leaf (or CA), suitable for runtime pins."""
    return _certificate(read_regular(certificate, MAX_PUBLIC_BYTES)).fingerprint(hashes.SHA256()).hex()


def public_key_fingerprint(public_key: Pathish) -> str:
    """SHA256 of public SubjectPublicKeyInfo DER, not of the PEM text."""
    data = read_regular(public_key, MAX_PUBLIC_BYTES)
    if b"PRIVATE KEY" in data:
        raise DeployError("fingerprint expects a public key file")
    try:
        key = serialization.load_pem_public_key(data)
    except (ValueError, UnsupportedAlgorithm) as exc:
        raise DeployError("invalid public key") from exc
    return _key_fingerprint(key)


def request_fingerprint(request: Pathish) -> str:
    """SHA256 of the exact public enrollment request bytes; compare out of band."""
    return digest(read_regular(request, MAX_PUBLIC_BYTES, owned=False))


def _key_usage(*, ca: bool = False) -> x509.KeyUsage:
    return x509.KeyUsage(digital_signature=not ca, content_commitment=False,
                         key_encipherment=False, data_encipherment=False, key_agreement=False,
                         key_cert_sign=ca, crl_sign=ca, encipher_only=False, decipher_only=False)


def init_authority(directory: Pathish, *, name: str = "swarm-fleet") -> Path:
    """Create a new offline signing authority and fleet CA; never replace files."""
    identifier(name, "authority name")
    windows_acl_warning()
    directory = private_directory(directory, new=True)
    signing = ed25519.Ed25519PrivateKey.generate()
    ca_key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
                   .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
                   .not_valid_before(now - timedelta(minutes=5))
                   .not_valid_after(now + timedelta(days=3650))
                   .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                   .add_extension(_key_usage(ca=True), critical=True)
                   .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
                                  critical=False)
                   .sign(ca_key, hashes.SHA256()))
    write_new(directory / "signing-key.pem", _private_bytes(signing))
    write_new(directory / "signing-public.pem", _public_bytes(signing.public_key()))
    write_new(directory / "ca-key.pem", _private_bytes(ca_key))
    write_new(directory / "ca.pem", certificate.public_bytes(serialization.Encoding.PEM))
    fsync_directory(directory)
    return directory


def _ip_addresses(values: Iterable[str]) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    if isinstance(values, (str, bytes)):
        raise DeployError("IP addresses must be a sequence")
    result = []
    for value in values:
        if len(result) >= 16 or not isinstance(value, str) or "%" in value:
            raise DeployError("invalid or excessive IP SANs")
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise DeployError("invalid IP SAN") from exc
        if address in result:
            raise DeployError("duplicate IP SAN")
        result.append(address)
    return tuple(result)


def init_device(directory: Pathish, *, node_id: str,
                ip_addresses: Iterable[str] = ()) -> Path:
    """Run on the destination: generate unique local TLS and encryption keys.

    Return the public enrollment request path. Its digest must be carried to the
    enrolling operator through a separately authenticated channel.
    """
    node_id = identifier(node_id)
    ips = _ip_addresses(ip_addresses)
    windows_acl_warning()
    directory = private_directory(directory, new=True)
    tls_key = ec.generate_private_key(ec.SECP256R1())
    encryption_key = rsa.generate_private_key(public_exponent=65537, key_size=RSA_BITS)
    san = x509.SubjectAlternativeName([x509.DNSName(node_id), *(x509.IPAddress(ip) for ip in ips)])
    csr = (x509.CertificateSigningRequestBuilder()
           .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)]))
           .add_extension(san, critical=False).sign(tls_key, hashes.SHA256()))
    csr_pem = csr.public_bytes(serialization.Encoding.PEM)
    encryption_pem = _public_bytes(encryption_key.public_key())
    body = {"schema": 1, "node_id": node_id, "tls_csr_pem": csr_pem.decode("ascii"),
            "encryption_public_key_pem": encryption_pem.decode("ascii")}
    proof = tls_key.sign(REQUEST_CONTEXT + canonical_json(body), ec.ECDSA(hashes.SHA256()))
    request = {**body, "signature": base64.b64encode(proof).decode("ascii")}
    write_new(directory / "identity.json", canonical_json({"schema": 1, "node_id": node_id}))
    write_new(directory / "tls-key.pem", _private_bytes(tls_key))
    write_new(directory / "tls.csr.pem", csr_pem)
    write_new(directory / "encryption-key.pem", _private_bytes(encryption_key))
    write_new(directory / "encryption-public.pem", encryption_pem)
    result = write_new(directory / "enrollment-request.json", canonical_json(request))
    fsync_directory(directory)
    return result


def decode_base64(value: Any, size: int | None = None) -> bytes:
    if not isinstance(value, str) or len(value) > MAX_PUBLIC_BYTES:
        raise DeployError("invalid base64 field")
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise DeployError("invalid base64 field") from exc
    if base64.b64encode(data).decode("ascii") != value or (size is not None and len(data) != size):
        raise DeployError("invalid base64 field size or encoding")
    return data


def _request(data: bytes) -> tuple[str, x509.CertificateSigningRequest, rsa.RSAPublicKey]:
    body = exact_keys(parse_json(data, canonical=True), {"schema", "node_id", "tls_csr_pem",
                       "encryption_public_key_pem", "signature"}, "enrollment request").copy()
    if type(body["schema"]) is not int or body["schema"] != 1:
        raise DeployError("unsupported enrollment request schema")
    node_id = identifier(body["node_id"])
    signature = decode_base64(body.pop("signature"))
    try:
        csr = x509.load_pem_x509_csr(body["tls_csr_pem"].encode("ascii"))
        recipient = _load_public(body["encryption_public_key_pem"].encode("ascii"), rsa.RSAPublicKey)
        key = csr.public_key()
        if (not isinstance(key, ec.EllipticCurvePublicKey) or not isinstance(key.curve, ec.SECP256R1)
                or not csr.is_signature_valid):
            raise DeployError("CSR must prove possession of an ECDSA P-256 TLS key")
    except (ValueError, TypeError, AttributeError, UnicodeError, UnsupportedAlgorithm) as exc:
        raise DeployError("invalid enrollment public material") from exc
    if csr.subject != x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)]):
        raise DeployError("CSR subject does not match node_id")
    try:
        extensions = list(csr.extensions)
        if len(extensions) != 1 or not isinstance(extensions[0].value, x509.SubjectAlternativeName):
            raise DeployError("CSR must contain only its node DNS SAN and optional IP SANs")
        _validate_san(extensions[0].value, node_id)
        key.verify(signature, REQUEST_CONTEXT + canonical_json(body), ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError, x509.DuplicateExtension, UnsupportedAlgorithm) as exc:
        raise DeployError("invalid CSR SANs or enrollment request proof") from exc
    return node_id, csr, recipient


def _validate_san(san: x509.SubjectAlternativeName, node_id: str) -> None:
    names = list(san)
    if (not 1 <= len(names) <= 17 or san.get_values_for_type(x509.DNSName) != [node_id]
            or any(not isinstance(item, (x509.DNSName, x509.IPAddress)) for item in names)):
        raise DeployError("certificate SAN must be the exact node DNS name plus optional IPs")
    ips = san.get_values_for_type(x509.IPAddress)
    if len(set(ips)) != len(ips):
        raise DeployError("duplicate IP SANs")


def load_recipient(recipient: Pathish) -> rsa.RSAPublicKey:
    """Accept a reviewed public RSA PEM or the retained, approved public request."""
    data = read_regular(recipient, MAX_PUBLIC_BYTES)
    if data.startswith(b"{"):
        return _request(data)[2]
    return _load_public(data, rsa.RSAPublicKey)


def _validate_ca(ca: x509.Certificate) -> None:
    now = datetime.now(timezone.utc)
    try:
        basic = ca.extensions.get_extension_for_class(x509.BasicConstraints).value
        usage = ca.extensions.get_extension_for_class(x509.KeyUsage).value
        if not basic.ca or basic.path_length != 0 or not usage.key_cert_sign:
            raise DeployError("invalid fleet CA constraints")
        key = ca.public_key()
        if not isinstance(key, ec.EllipticCurvePublicKey) or not isinstance(key.curve, ec.SECP256R1):
            raise DeployError("fleet CA must use ECDSA P-256")
        ca.verify_directly_issued_by(ca)
    except (ValueError, TypeError, InvalidSignature, x509.ExtensionNotFound,
            x509.DuplicateExtension, UnsupportedAlgorithm) as exc:
        raise DeployError("invalid self-signed fleet CA") from exc
    if not ca.not_valid_before_utc <= now < ca.not_valid_after_utc:
        raise DeployError("fleet CA is not currently valid; check the local clock")


def enroll(request: Pathish, *, authority_dir: Pathish, output_dir: Pathish,
           request_sha256: str, days: int = 30) -> Path:
    """Approve a pinned public request and write only a leaf certificate and CA.

    The caller must obtain request_sha256 independently of the USB carrying the
    request. A CA certificate alone does not authorize a runtime peer.
    """
    if type(days) is not int or not 1 <= days <= 90:
        raise DeployError("certificate lifetime must be between 1 and 90 days")
    expected = sha256_hex(request_sha256, "out-of-band enrollment request SHA256")
    raw = read_regular(request, MAX_PUBLIC_BYTES, owned=False)
    if not hmac.compare_digest(digest(raw), expected):
        raise DeployError("enrollment request does not match the out-of-band fingerprint")
    node_id, csr, _ = _request(raw)
    authority = check_directory(authority_dir, private=True)
    ca_key = _load_private(authority / "ca-key.pem", ec.EllipticCurvePrivateKey)
    ca = _certificate(read_regular(authority / "ca.pem", MAX_PUBLIC_BYTES))
    _validate_ca(ca)
    if _key_fingerprint(ca_key.public_key()) != _key_fingerprint(ca.public_key()):
        raise DeployError("CA key does not match its certificate")
    now = datetime.now(timezone.utc)
    before = max(now - timedelta(minutes=5), ca.not_valid_before_utc)
    after = min(before + timedelta(days=days), ca.not_valid_after_utc)
    certificate = (x509.CertificateBuilder()
                   .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)]))
                   .issuer_name(ca.subject).public_key(csr.public_key())
                   .serial_number(x509.random_serial_number())
                   .not_valid_before(before).not_valid_after(after)
                   .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                   .add_extension(_key_usage(), critical=True)
                   .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH,
                                                         ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
                   .add_extension(csr.extensions.get_extension_for_class(x509.SubjectAlternativeName).value,
                                  critical=False)
                   .add_extension(x509.SubjectKeyIdentifier.from_public_key(csr.public_key()), critical=False)
                   .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.public_key()),
                                  critical=False)
                   .sign(ca_key, hashes.SHA256()))
    output = private_directory(output_dir, new=True)
    write_new(output / "tls-cert.pem", certificate.public_bytes(serialization.Encoding.PEM))
    write_new(output / "ca.pem", ca.public_bytes(serialization.Encoding.PEM))
    return output


def accept_enrollment(identity_dir: Pathish, *, enrollment_dir: Pathish,
                      ca_certificate: Pathish, credentials_dir: Pathish | None = None) -> Path:
    """Validate returned public certificates against an independently pinned CA.

    Create credentials/ once, or a fresh explicitly named credentials directory for
    renewal. No private key or existing certificate is ever replaced.
    """
    node_id = device_node_id(identity_dir)
    identity = check_directory(identity_dir, private=True)
    incoming = check_directory(enrollment_dir, owned=False)
    pinned = _certificate(read_regular(ca_certificate, MAX_PUBLIC_BYTES))
    ca = _certificate(read_regular(incoming / "ca.pem", MAX_PUBLIC_BYTES, owned=False))
    if not hmac.compare_digest(pinned.fingerprint(hashes.SHA256()), ca.fingerprint(hashes.SHA256())):
        raise DeployError("returned CA does not match the independently pinned CA")
    _validate_ca(ca)
    leaf = _certificate(read_regular(incoming / "tls-cert.pem", MAX_PUBLIC_BYTES, owned=False))
    tls_key = _load_private(identity / "tls-key.pem", ec.EllipticCurvePrivateKey)
    try:
        leaf.verify_directly_issued_by(ca)
        basic = leaf.extensions.get_extension_for_class(x509.BasicConstraints).value
        usage = leaf.extensions.get_extension_for_class(x509.KeyUsage).value
        eku = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        _validate_san(leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value, node_id)
    except (ValueError, TypeError, InvalidSignature, x509.ExtensionNotFound,
            x509.DuplicateExtension, UnsupportedAlgorithm) as exc:
        raise DeployError("returned TLS certificate failed verification") from exc
    if (basic.ca or not usage.digital_signature or usage.key_cert_sign or usage.crl_sign
            or usage.key_agreement or usage.key_encipherment or usage.data_encipherment
            or usage.content_commitment or set(eku) != {ExtendedKeyUsageOID.SERVER_AUTH,
                                                       ExtendedKeyUsageOID.CLIENT_AUTH}
            or len(eku) != 2):
        raise DeployError("returned TLS certificate has invalid constraints or EKUs")
    if leaf.subject != x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)]):
        raise DeployError("returned TLS certificate subject does not match node_id")
    if _key_fingerprint(leaf.public_key()) != _key_fingerprint(tls_key.public_key()):
        raise DeployError("returned TLS certificate does not match the device's local TLS key")
    now = datetime.now(timezone.utc)
    if (not leaf.not_valid_before_utc <= now < leaf.not_valid_after_utc
            or leaf.not_valid_after_utc - leaf.not_valid_before_utc > timedelta(days=90)
            or leaf.not_valid_before_utc < ca.not_valid_before_utc
            or leaf.not_valid_after_utc > ca.not_valid_after_utc):
        raise DeployError("returned TLS certificate is expired, not yet valid, or exceeds 90 days")
    destination = private_directory(credentials_dir if credentials_dir is not None
                                    else identity / "credentials", new=True)
    write_new(destination / "tls-cert.pem", leaf.public_bytes(serialization.Encoding.PEM))
    write_new(destination / "ca.pem", ca.public_bytes(serialization.Encoding.PEM))
    return destination