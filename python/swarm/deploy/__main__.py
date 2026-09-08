"""Entry point for ``python -m swarm.deploy``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())