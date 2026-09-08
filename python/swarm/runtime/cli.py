"""Run or inspect a device-side worker. Does not load robot drivers or arm outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from .config import ApplicationConfig
from .handlers import load_handler
from .node import SwarmRuntime
from .protocol import tls_context, validate_local_identity


async def _run(app: ApplicationConfig, args):
    handler = load_handler(app.factory, app.options)
    async with SwarmRuntime(app.node, handler) as node:
        print(json.dumps({"listening": app.node.host, "port": node.port, "node_id": app.node.node_id,
                          "workload_id": app.node.workload_id, "hardware_enabled": False}), flush=True)
        if args.wait_peers:
            await node.wait_for_peers(args.wait_peers)
        if args.submit_file:
            if args.submit_file.stat().st_size > app.node.max_payload:
                raise ValueError("input exceeds max_payload; preprocess locally first")
            result = await node.submit(args.submit_file.read_bytes())
            if args.output:
                with args.output.open("xb") as output:
                    output.write(result.payload)
            print(json.dumps({"useful": result.useful, "result_bytes": len(result.payload), "status": node.status()}))
        elif args.run_seconds is not None:
            if args.run_seconds <= 0:
                raise ValueError("run-seconds must be positive")
            await asyncio.sleep(args.run_seconds)
            print(json.dumps(node.status()), flush=True)
        else:
            await asyncio.Event().wait()


def main(argv=None, *, default_config: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--check", action="store_true", help="validate local config and TLS files WITHOUT importing the brain")
    parser.add_argument("--run-seconds", type=float, help="bounded smoke run instead of a long-running worker")
    parser.add_argument("--wait-peers", type=int, default=0)
    parser.add_argument("--submit-file", type=Path, help="submit one local inference input (never actuator commands)")
    parser.add_argument("--output", type=Path, help="write returned bytes to a new file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        app = ApplicationConfig.load(args.config)
        if args.check:
            validate_local_identity(app.node, tls_context(app.node, server=True), tls_context(app.node, server=False))
            print(json.dumps({"valid": True, "node_id": app.node.node_id, "tls": "1.3 mutual + pinned peers",
                              "hardware_enabled": False}))
            return 0
        # Factory's relative asset paths are intentionally rooted at the trusted application config.
        os.chdir(args.config.resolve().parent)
        asyncio.run(_run(app, args))
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"runtime failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())