"""Strict operator-owned configuration. No discovery, peer-supplied code, or secrets in logs."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", value):
        raise ValueError("node_id must contain 1..63 lowercase letters, digits or hyphens")
    return value


def positive(value: object, name: str, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= upper:
        raise ValueError(f"{name} must be > 0 and <= {upper}")
    return number


def integer(value: object, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def read_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def bad(value):
        raise ValueError(f"nonfinite JSON: {value}")

    if path.stat().st_size > 256 * 1024:
        raise ValueError("configuration exceeds 256 KiB")
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=bad)
    if not isinstance(data, dict):
        raise ValueError("configuration must be a JSON object")
    return data


@dataclass(frozen=True)
class Peer:
    node_id: str
    host: str
    port: int
    fingerprint: str

    def __post_init__(self):
        identifier(self.node_id)
        if not isinstance(self.host, str) or not self.host.strip() or len(self.host) > 253:
            raise ValueError("peer host must be a nonempty hostname or IP address")
        integer(self.port, "peer port", 1, 65535)
        if not isinstance(self.fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", self.fingerprint):
            raise ValueError("peer fingerprint must be SHA256 DER certificate digest (64 lowercase hex)")


@dataclass(frozen=True)
class TLSFiles:
    ca: Path
    certificate: Path
    private_key: Path


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    workload_id: str
    tls: TLSFiles
    peers: tuple[Peer, ...] = ()
    host: str = "127.0.0.1"  # Exposing a listener requires an explicit local setting.
    port: int = 7443
    queue_limit: int = 4  # Includes the running inference, not just waiting jobs.
    max_submissions: int = 32
    max_payload: int = 65536
    heartbeat_s: float = 1.0
    peer_timeout_s: float = 5.0
    connect_timeout_s: float = 3.0
    job_timeout_s: float = 2.0
    retries: int = 2
    estimated_ms: float = 10.0
    cache_entries: int = 128

    def __post_init__(self):
        identifier(self.node_id)
        if not isinstance(self.workload_id, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,128}", self.workload_id):
            raise ValueError("workload_id must be 1..128 ASCII identifier characters")
        if not isinstance(self.tls, TLSFiles):
            raise ValueError("tls must be TLSFiles")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a hostname or IP address")
        integer(self.port, "port", 0, 65535)
        for name, lower, upper in (("queue_limit", 1, 64), ("max_submissions", 1, 256),
                                   ("max_payload", 1, 1024 * 1024), ("retries", 0, 8),
                                   ("cache_entries", 0, 256)):
            integer(getattr(self, name), name, lower, upper)
        for name in ("heartbeat_s", "peer_timeout_s", "connect_timeout_s", "job_timeout_s"):
            positive(getattr(self, name), name, 300.0)
        positive(self.estimated_ms, "estimated_ms", 300000.0)
        if self.peer_timeout_s <= self.heartbeat_s * 2:
            raise ValueError("peer_timeout_s must exceed twice heartbeat_s")
        if not isinstance(self.peers, tuple) or len(self.peers) > 32 or any(not isinstance(p, Peer) for p in self.peers):
            raise ValueError("peers must be a tuple of at most 32 Peer objects")
        ids = [self.node_id, *(p.node_id for p in self.peers)]
        pins = [p.fingerprint for p in self.peers]
        if len(ids) != len(set(ids)) or len(pins) != len(set(pins)):
            raise ValueError("node IDs and authorized peer certificate pins must be unique")


@dataclass(frozen=True)
class ApplicationConfig:
    node: NodeConfig
    factory: str
    options: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, *, expand_environment: bool = True) -> ApplicationConfig:
        path = Path(path).resolve()
        data = read_json(path)
        if data.keys() - {"schema_version", "node", "brain"} or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("expected schema_version 1, node and brain configuration")
        node = dict(data["node"])
        tls = node.pop("tls")
        if not isinstance(tls, dict) or tls.keys() != {"ca", "certificate", "private_key"}:
            raise ValueError("tls requires ca, certificate and private_key paths")

        def local_path(value):
            if not isinstance(value, str) or not value:
                raise ValueError("TLS paths must be nonempty strings")
            value = os.path.expandvars(value) if expand_environment else value
            if expand_environment and "$" in value:
                raise ValueError("unresolved TLS environment variable; provide SWARM_IDENTITY_DIR when launching")
            resolved = Path(value).expanduser()
            return resolved if resolved.is_absolute() else path.parent / resolved

        node["tls"] = TLSFiles(**{key: local_path(value) for key, value in tls.items()})
        peers = node.pop("peers", [])
        if not isinstance(peers, list):
            raise ValueError("peers must be a list")
        node["peers"] = tuple(Peer(**peer) for peer in peers)
        brain = data["brain"]
        if not isinstance(brain, dict) or brain.keys() - {"factory", "options"}:
            raise ValueError("brain requires factory and optional options")
        factory = brain.get("factory")
        if not isinstance(factory, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", factory):
            raise ValueError("brain factory must name trusted local module:factory code")
        options = brain.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("brain options must be a JSON object")
        return cls(NodeConfig(**node), factory, options)