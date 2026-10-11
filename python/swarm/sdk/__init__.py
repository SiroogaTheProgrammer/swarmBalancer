"""swarm.sdk - write brains and tactics the simulator (and the 3D studio) can load, and ship brains to drones.

Two kinds of package, one format: a folder (or a .zip of it) with ``manifest.json`` at its root.

* **perception brain** - what one drone concludes from one camera frame
    - format ``swm``: a compiled network (``.swm``). The C++ engine runs it on every target, MCUs included.
    - format ``python``: a class with ``classify(image)``. Runs in the simulator and on Linux companions.
* **tactic** - how the swarm shares the work (a :class:`Tactic`), like the built-in local/central/striped.

    python -m swarm.sdk new my_brain --type perception-python     # skeleton that validates out of the box
    python -m swarm.sdk validate brains/my_brain                    # conformance checks
    python -m swarm.sdk pack brains/my_brain                        # -> my_brain.zip for the studio upload
    python -m swarm.sdk export brains/my_brain --out deploy/my_brain  # device artefacts + instructions

docs/SDK.md is the manual.
"""

from .api import CLASSES, Frame, Message, PerceptionBrain, Tactic, downscale_macs
from .packages import (LIBRARY_DIRS, Package, PackageError, discover_packages, load_package, swm_info,
                       use_package)
from .validate import Report, validate

__all__ = [
    "CLASSES", "Frame", "Message", "PerceptionBrain", "Tactic", "downscale_macs",
    "LIBRARY_DIRS", "Package", "PackageError", "discover_packages", "load_package", "swm_info", "use_package",
    "Report", "validate",
]
