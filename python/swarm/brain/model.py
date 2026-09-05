"""``Sequential``: an ordered list of layers with training, int8 evaluation and ``.swm`` export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import formats
from .layers import Conv2D, Dense, Flatten, Layer, MaxPool2D, ReLU, Shape, softmax


class Sequential:
    def __init__(self, layers: list[Layer], input_shape: tuple[int, int, int]):
        self.layers = layers
        self.input_shape = tuple(int(d) for d in input_shape)
        self._shapes = self._plan()

    # ------------------------------------------------------------------ shapes
    def _plan(self) -> list[Shape]:
        shapes: list[Shape] = [self.input_shape]
        cur: Shape = self.input_shape
        for L in self.layers:
            cur = L.output_shape(cur)
            shapes.append(cur)
        return shapes

    @property
    def output_shape(self) -> Shape:
        return self._shapes[-1]

    def macs(self) -> int:
        return sum(L.macs(s) for L, s in zip(self.layers, self._shapes))

    def num_params(self) -> int:
        return sum(int(p.size) for L in self.layers for p in L.params().values())

    def weights_bytes(self, quantize: bool = False) -> int:
        total = 0
        for L in self.layers:
            for name, p in L.params().items():
                total += int(p.size) * (1 if (quantize and name == "W") else 4)
        return total

    # ------------------------------------------------------------------ compute
    def forward(self, x: np.ndarray) -> np.ndarray:
        for L in self.layers:
            x = L.forward(x)
        return x

    __call__ = forward

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for L in reversed(self.layers):
            dy = L.backward(dy)
        return dy

    def forward_int8(self, x: np.ndarray) -> np.ndarray:
        """numpy reference of the C++ int8 path (weights int8, activations quantised per sample)."""
        for L in self.layers:
            x = L.forward_int8(x)  # type: ignore[attr-defined]
        return x

    def predict_proba(self, x: np.ndarray, int8: bool = False) -> np.ndarray:
        logits = self.forward_int8(x) if int8 else self.forward(x)
        return softmax(logits)

    def predict(self, x: np.ndarray, int8: bool = False) -> np.ndarray:
        return self.predict_proba(x, int8).argmax(axis=1)

    def parameters(self):
        """Yields ``(param, grad)`` pairs for optimisers."""
        for L in self.layers:
            p, g = L.params(), L.grads()
            for name in p:
                yield p[name], g[name]

    # ------------------------------------------------------------------ export / import
    def specs(self, quantize: bool = False, with_softmax: bool = True) -> list[dict]:
        specs = [L.spec(s, quantize) for L, s in zip(self.layers, self._shapes)]
        if with_softmax:
            specs.append({"type": "softmax"})
        return specs

    def save(self, path: str | Path, quantize: bool = False, with_softmax: bool = True) -> Path:
        """Writes a ``.swm`` file the C++ engine can load. ``quantize`` stores int8 weights."""
        path = Path(path)
        formats.write_swm(path, self.input_shape, self.specs(quantize, with_softmax))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Sequential":
        """Loads a ``.swm`` (int8 weights are dequantised; a trailing softmax is dropped)."""
        input_shape, specs = formats.read_swm(path)
        layers: list[Layer] = []
        for s in specs:
            t = s["type"]
            if t == "dense":
                L = Dense(s["in"], s["out"])
                L.W[...] = s["w"].astype(np.float32) * (s["w_scale"] if s["dtype"] == "i8" else 1.0)
                L.b[...] = s["b"]
            elif t == "conv2d":
                L = Conv2D(s["in_c"], s["out_c"], s["k"], s["stride"], s["pad"])
                w = s["w"].astype(np.float32) * (s["w_scale"] if s["dtype"] == "i8" else 1.0)
                L.W[...] = w.reshape(L.W.shape)
                L.b[...] = s["b"]
            elif t == "relu":
                L = ReLU()
            elif t == "maxpool2d":
                if s["k"] != s["stride"]:
                    raise ValueError("numpy MaxPool2D only supports k == stride")
                L = MaxPool2D(s["k"])
            elif t == "flatten":
                L = Flatten()
            elif t == "softmax":
                continue
            else:
                raise ValueError(f"unknown layer {t}")
            layers.append(L)
        return cls(layers, input_shape)

    # ------------------------------------------------------------------ info
    def summary(self) -> str:
        rows = [f"input {'x'.join(map(str, self.input_shape))}"]
        for L, s_in, s_out in zip(self.layers, self._shapes, self._shapes[1:]):
            n_p = sum(int(p.size) for p in L.params().values())
            rows.append(f"  {type(L).__name__:<10} -> {'x'.join(map(str, s_out)):<12} params={n_p:<7} macs={L.macs(s_in)}")
        rows.append(f"total params={self.num_params()}  macs/frame={self.macs()}  "
                    f"weights f32={self.weights_bytes()} B  int8={self.weights_bytes(True)} B")
        return "\n".join(rows)
