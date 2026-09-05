"""The brain: minimalist neural networks for constrained devices.

* :mod:`swarm.brain.layers`  - numpy reference layers (forward + backward, used for training)
* :mod:`swarm.brain.model`   - ``Sequential`` container, int8 quantisation, ``.swm`` export
* :mod:`swarm.brain.formats` - the ``.swm`` binary format shared with the C++ engine
* :mod:`swarm.brain.native`  - ctypes bridge to the C++ engine (``swarm_brain`` shared library)
"""

from .layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU, softmax_cross_entropy
from .model import Sequential

__all__ = ["Conv2D", "Dense", "Flatten", "MaxPool2D", "ReLU", "Sequential", "softmax_cross_entropy"]
