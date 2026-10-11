"""Imported by every studio preset so `python scenarios/studio/<preset>.py` works without installation."""

import sys
from pathlib import Path

_PY = str(Path(__file__).resolve().parents[2] / "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)
