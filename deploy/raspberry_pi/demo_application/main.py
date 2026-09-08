"""Runnable deployment smoke application: stdlib, local bytes, no hardware/network."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from demo_brain import infer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", default="deployment smoke test")
    args = parser.parse_args()
    config = json.loads((Path(__file__).parent / "application.json").read_text(encoding="utf-8"))
    if config != {"schema_version": 1, "hardware_enabled": False}:
        raise ValueError("this demonstration only accepts its hardware-disabled configuration")
    print(json.dumps({"application_id": os.environ.get("SWARM_APPLICATION_ID"),
                      "node_id": os.environ.get("SWARM_NODE_ID"), "hardware_enabled": False,
                      "result": infer(args.message.encode("utf-8")).decode("ascii")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())