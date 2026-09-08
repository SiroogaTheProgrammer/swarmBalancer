"""Bounded binary records *inside TLS*. No compression, pickle or executable messages."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import ssl
import stat
import struct
from enum import IntEnum

from .config import NodeConfig, Peer

HEADER = struct.Struct("!BI")
JOB_PREFIX = struct.Struct("!16sI")
LOAD = struct.Struct("!HHI")  # accepted work, capacity, estimated microseconds/job
VERSION = 1


class Kind(IntEnum):
    HELLO = 1
    LOAD = 2
    JOB = 3
    RESULT = 4
    ERROR = 5


class ProtocolError(ConnectionError):
    pass


async def read_packet(reader: asyncio.StreamReader, max_payload: int) -> tuple[Kind, bytes]:
    kind, size = HEADER.unpack(await reader.readexactly(HEADER.size))
    try:
        kind = Kind(kind)
    except ValueError as exc:
        raise ProtocolError("unknown message type") from exc
    # Check lengths before allocating/reading attacker-controlled bodies.
    limits = {Kind.HELLO: 256, Kind.LOAD: LOAD.size, Kind.JOB: max_payload + JOB_PREFIX.size,
              Kind.RESULT: max_payload + 17, Kind.ERROR: 17}
    if size > limits[kind]:
        raise ProtocolError("message exceeds configured limit")
    if kind in (Kind.LOAD, Kind.ERROR) and size != limits[kind]:
        raise ProtocolError("invalid fixed-size record")
    return kind, await reader.readexactly(size)


def tls_context(config: NodeConfig, *, server: bool) -> ssl.SSLContext:
    if not ssl.HAS_TLSv1_3:
        raise RuntimeError("TLS 1.3 support is required; there is no plaintext/TLS 1.2 fallback")
    if os.name == "posix":
        key = config.tls.private_key.lstat()
        if not stat.S_ISREG(key.st_mode) or key.st_mode & 0o077:
            raise ValueError("TLS private key must be a regular owner-only file (0600 or 0400)")
    purpose = ssl.Purpose.CLIENT_AUTH if server else ssl.Purpose.SERVER_AUTH
    context = ssl.create_default_context(purpose, cafile=str(config.tls.ca))
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    if not server:
        context.check_hostname = True
        context.hostname_checks_common_name = False
    else:
        context.num_tickets = 0
    context.options |= ssl.OP_NO_COMPRESSION
    context.load_cert_chain(str(config.tls.certificate), str(config.tls.private_key))
    return context


def validate_local_identity(config: NodeConfig, server: ssl.SSLContext, client: ssl.SSLContext) -> None:
    """Validate the local cert's CA, expiry, usage and SAN before listening, using TLS BIOs.

    ssl exposes peer certificate parsing but no public standalone X.509 decoder.
    A tiny in-memory mutual handshake avoids an extra runtime crypto dependency
    or a network listener just to check our own operator-provisioned identity.
    """
    client_in, client_out, server_in, server_out = (ssl.MemoryBIO() for _ in range(4))
    sockets = (client.wrap_bio(client_in, client_out, server_hostname=config.node_id),
               server.wrap_bio(server_in, server_out, server_side=True))
    done = [False, False]
    for _ in range(16):
        for i, sock in enumerate(sockets):
            if not done[i]:
                try:
                    sock.do_handshake()
                    done[i] = True
                except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                    pass
        for source, destination in ((client_out, server_in), (server_out, client_in)):
            if source.pending:
                destination.write(source.read())
        if all(done):
            names = [value for kind, value in sockets[0].getpeercert().get("subjectAltName", ()) if kind == "DNS"]
            if names != [config.node_id]:
                raise ValueError("local TLS certificate DNS SAN does not match node_id exactly")
            return
    raise ValueError("local TLS identity validation did not complete")


def authorized_peer(writer: asyncio.StreamWriter, config: NodeConfig, expected: Peer | None) -> Peer:
    tls = writer.get_extra_info("ssl_object")
    if tls is None or tls.version() != "TLSv1.3":
        raise ProtocolError("mutual TLS 1.3 is mandatory")
    fingerprint = hashlib.sha256(tls.getpeercert(binary_form=True)).hexdigest()
    peer = next((p for p in config.peers if hmac.compare_digest(p.fingerprint, fingerprint)), None)
    if peer is None or (expected is not None and peer != expected):
        raise ProtocolError("certificate is not an authorized peer")
    names = [value for kind, value in tls.getpeercert().get("subjectAltName", ()) if kind == "DNS"]
    if names != [peer.node_id]:
        raise ProtocolError("peer certificate must have the exact configured node DNS identity")
    return peer