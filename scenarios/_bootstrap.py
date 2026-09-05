"""Imported by every preset so ``python scenarios/<preset>.py`` works without ``pip install -e .``.

It only puts ``<repo>/python`` on ``sys.path``; harmless if the package is already installed.
"""

import sys
from pathlib import Path

_PY = str(Path(__file__).resolve().parents[1] / "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)
