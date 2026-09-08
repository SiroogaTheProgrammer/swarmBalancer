"""Desktop/CLI device setup entry point; existing dev.py commands remain the testing toolkit."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

if __name__ == "__main__":
    try:
        from swarm.deploy.setup import main
    except ModuleNotFoundError as exc:
        if exc.name != "cryptography":
            raise
        raise SystemExit("Device packaging requires the optional deployment dependency; install .[deploy] in this Python environment.")
    raise SystemExit(main())