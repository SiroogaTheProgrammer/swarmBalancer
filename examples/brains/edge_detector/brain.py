"""Edge detector - a classical computer-vision perception brain (no neural network).

It thresholds the frame for pixels brighter or darker than any terrain, then tells the five classes of the
simulator's world apart by how many such pixels there are and how far they spread:

    road      many dark pixels spanning the whole frame (a long thin line)
    building  many dark pixels in a compact square (the dark roof rim)
    aircraft  many bright pixels in a cross shape (fuselage + wings)
    vehicle   a few bright pixels close together (a small rectangle)
    nothing   neither

About 4 operations per pixel - cheap enough for any drone. Plain numpy and no `swarm` import, so
`python -m swarm.sdk export` can ship it to a Linux companion computer unchanged.
"""

import numpy as np


class EdgeDetector:
    input_shape = (1, 32, 32)
    classes = ("nothing", "vehicle", "building", "road", "aircraft")
    macs = 4 * 32 * 32  # threshold x2, bounding box x2 per pixel
    frontend_fraction = 0.75
    ram_bytes = 4 * 32 * 32  # two boolean masks + the frame

    def __init__(self, seed=0, options=None):
        options = dict(options or {})
        self.bright = float(options.get("bright", 0.72))  # terrain tops out near 0.6 (+ noise)
        self.dark = float(options.get("dark", 0.17))      # ...and bottoms out near 0.25
        self.span = float(options.get("road_span", 0.82))  # dark extent (fraction of the frame) of a road

    @staticmethod
    def _extent(mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return 0
        return int(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1)

    def classify(self, image):
        x = image[0]
        size = x.shape[0]
        bright = x > self.bright
        dark = x < self.dark
        nb, nd = int(bright.sum()), int(dark.sum())
        if nd >= 20 and nd >= nb // 2:
            span = self._extent(dark) / size
            if span >= self.span:
                return 3, min(0.95, 0.6 + (span - self.span) * 2.0)
            return 2, min(0.95, 0.6 + (self.span - span))
        if nb >= 4:
            spread = self._extent(bright) / size
            if nb >= 28 or spread >= 0.28:
                return 4, min(0.95, 0.55 + nb / 120.0)
            return 1, min(0.95, 0.6 + (28 - nb) / 60.0)
        return 0, 0.9 if max(nb, nd) == 0 else 0.7
