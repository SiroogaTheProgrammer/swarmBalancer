"""Hardware-free, real TLS 1.3 demo with temporary identities and a lost worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from .config import NodeConfig, Peer, TLSFiles
from .handlers import make_digest_handler
from .node import SwarmRuntime


async def demo(root: Path) -> dict:
    from swarm.deploy import accept_enrollment, certificate_fingerprint, enroll, init_authority, init_device, request_fingerprint

    authority = init_authority(root / "authority")
    identities = []
    for node_id in ("demo-a", "demo-b"):
        identity = root / node_id
        request = init_device(identity, node_id=node_id)
        issued = enroll(request, authority_dir=authority, output_dir=root / (node_id + "-enrolled"),
                        request_sha256=request_fingerprint(request))
        certs = accept_enrollment(identity, enrollment_dir=issued, ca_certificate=authority / "ca.pem")
        identities.append(TLSFiles(certs / "ca.pem", certs / "tls-cert.pem", identity / "tls-key.pem"))
    left_tls, right_tls = identities
    # Only the lower node ID dials, so the listening-only worker needs the caller's pin, not its ephemeral port.
    worker_config = NodeConfig("demo-b", "digest-demo-v1", right_tls, port=0, estimated_ms=1,
                               peers=(Peer("demo-a", "127.0.0.1", 1, certificate_fingerprint(left_tls.certificate)),))
    worker = await SwarmRuntime(worker_config, make_digest_handler({})).start()
    try:
        owner_config = NodeConfig("demo-a", "digest-demo-v1", left_tls, port=0, estimated_ms=1000,
                                  peers=(Peer("demo-b", "127.0.0.1", worker.port, certificate_fingerprint(right_tls.certificate)),))
        async with SwarmRuntime(owner_config, make_digest_handler({})) as owner:
            await owner.wait_for_peers(1)
            result = await owner.submit(b"local observation before worker loss")
            assert result.payload == hashlib.sha256(b"local observation before worker loss").digest()
            remote_jobs = worker.counters["local_jobs"]
            await worker.close()
            recovered = await owner.submit(b"local observation after worker loss")
            assert recovered.payload == hashlib.sha256(b"local observation after worker loss").digest()
            return {"tls": "1.3 mutual + pinned identities", "remote_jobs": remote_jobs,
                    "local_jobs_after_failover": owner.counters["local_jobs"],
                    "hardware_enabled": False, "status": owner.status()}
    finally:
        await worker.close()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="swarm-tls-demo-") as temporary:
        print(json.dumps(asyncio.run(demo(Path(temporary))), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())