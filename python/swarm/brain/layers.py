"""numpy reference implementation of the brain's layer set.

Every layer here has an exact counterpart in ``cpp/src/nn.cpp``; the numpy
version is used for training (it has ``backward``) and as the oracle the C++
engine is tested against. Tensors are NCHW; dense inputs are ``(B, n)``.
Weight layouts match the ``.swm`` format so export is a plain copy:

* ``Dense.W``  is ``[in, out]``  (inference is ``x[1,in] @ W[in,out]``)
* ``Conv2D.W`` is ``[out_c, in_c, k, k]`` -> flattened to ``[out_c, in_c*k*k]``
"""

from __future__ import annotations

import math

import numpy as np

Shape = tuple[int, ...]


def conv_out(size: int, k: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - k) // stride + 1


def im2col(x: np.ndarray, k: int, stride: int, pad: int) -> tuple[np.ndarray, int, int]:
    """``x[B,C,H,W]`` -> ``cols[B, C*k*k, oh*ow]``; row index is ``c*k*k + ki*k + kj`` (same as C++)."""
    B, C, H, W = x.shape
    oh, ow = conv_out(H, k, stride, pad), conv_out(W, k, stride, pad)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    cols = np.empty((B, C, k, k, oh, ow), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = xp[:, :, i : i + stride * oh : stride, j : j + stride * ow : stride]
    return cols.reshape(B, C * k * k, oh * ow), oh, ow


def col2im(cols: np.ndarray, x_shape: Shape, k: int, stride: int, pad: int, oh: int, ow: int) -> np.ndarray:
    B, C, H, W = x_shape
    cols = cols.reshape(B, C, k, k, oh, ow)
    xp = np.zeros((B, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for i in range(k):
        for j in range(k):
            xp[:, :, i : i + stride * oh : stride, j : j + stride * ow : stride] += cols[:, :, i, j]
    return xp[:, :, pad : pad + H, pad : pad + W] if pad else xp


def quantize_symmetric(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric int8 quantisation: ``w ~= q * scale``."""
    amax = float(np.abs(w).max())
    scale = amax / 127.0 if amax > 0 else 1.0
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale


def quantize_dynamic(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample dynamic activation quantisation (what the C++ engine does at run time).

    ``x[B, ...]`` -> ``(q[B, ...] int8, scale[B])``.
    """
    flat = x.reshape(x.shape[0], -1)
    amax = np.abs(flat).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(flat / scale[:, None]), -127, 127).astype(np.int8)
    return q.reshape(x.shape), scale


class Layer:
    """Base class. ``forward`` caches what ``backward`` needs."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict[str, np.ndarray]:
        return {}

    def grads(self) -> dict[str, np.ndarray]:
        return {}

    def output_shape(self, in_shape: Shape) -> Shape:
        return in_shape

    def macs(self, in_shape: Shape) -> int:
        return 0

    def spec(self, in_shape: Shape, quantize: bool = False) -> dict:
        """Layer description consumed by :mod:`swarm.brain.formats`."""
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.W = (rng.standard_normal((n_in, n_out)) * math.sqrt(2.0 / n_in)).astype(np.float32)
        self.b = np.zeros(n_out, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._x is not None
        self.dW[...] = self._x.T @ dy
        self.db[...] = dy.sum(axis=0)
        return dy @ self.W.T

    def forward_int8(self, x: np.ndarray) -> np.ndarray:
        """Reference for the C++ int8 path: dynamic per-sample activation quantisation."""
        Wq, w_scale = quantize_symmetric(self.W)
        xq, a_scale = quantize_dynamic(x)
        acc = xq.astype(np.int32) @ Wq.astype(np.int32)
        return acc.astype(np.float32) * (a_scale * w_scale)[:, None] + self.b

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}

    def output_shape(self, in_shape):
        n_in = int(np.prod(in_shape))
        if n_in != self.W.shape[0]:
            raise ValueError(f"Dense expects {self.W.shape[0]} inputs, got shape {in_shape}")
        return (self.W.shape[1],)

    def macs(self, in_shape):
        return int(self.W.size)

    def spec(self, in_shape, quantize=False):
        s = {"type": "dense", "in": self.W.shape[0], "out": self.W.shape[1], "b": self.b}
        if quantize:
            s["w"], s["w_scale"] = quantize_symmetric(self.W)
            s["dtype"] = "i8"
        else:
            s["w"], s["w_scale"], s["dtype"] = self.W, 1.0, "f32"
        return s


class Conv2D(Layer):
    def __init__(self, in_c: int, out_c: int, k: int = 3, stride: int = 1, pad: int = 1,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        fan_in = in_c * k * k
        self.W = (rng.standard_normal((out_c, in_c, k, k)) * math.sqrt(2.0 / fan_in)).astype(np.float32)
        self.b = np.zeros(out_c, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.k, self.stride, self.pad = k, stride, pad
        self._cache: tuple | None = None

    @property
    def in_c(self) -> int:
        return self.W.shape[1]

    @property
    def out_c(self) -> int:
        return self.W.shape[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        cols, oh, ow = im2col(x, self.k, self.stride, self.pad)
        Wm = self.W.reshape(self.out_c, -1)
        y = Wm @ cols + self.b[:, None]  # [B, out_c, oh*ow]
        self._cache = (x.shape, cols, oh, ow)
        return y.reshape(B, self.out_c, oh, ow)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        x_shape, cols, oh, ow = self._cache
        B = dy.shape[0]
        dyf = dy.reshape(B, self.out_c, -1)
        self.dW[...] = np.einsum("bon,bkn->ok", dyf, cols).reshape(self.W.shape)
        self.db[...] = dyf.sum(axis=(0, 2))
        Wm = self.W.reshape(self.out_c, -1)
        dcols = Wm.T @ dyf  # [B, K, N]
        return col2im(dcols, x_shape, self.k, self.stride, self.pad, oh, ow)

    def forward_int8(self, x: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        Wq, w_scale = quantize_symmetric(self.W)
        xq, a_scale = quantize_dynamic(x)
        cols, oh, ow = im2col(xq, self.k, self.stride, self.pad)
        acc = Wq.reshape(self.out_c, -1).astype(np.int32) @ cols.astype(np.int32)
        y = acc.astype(np.float32) * (a_scale * w_scale)[:, None, None] + self.b[:, None]
        return y.reshape(B, self.out_c, oh, ow)

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}

    def output_shape(self, in_shape):
        c, h, w = in_shape
        if c != self.in_c:
            raise ValueError(f"Conv2D expects {self.in_c} channels, got {c}")
        return (self.out_c, conv_out(h, self.k, self.stride, self.pad), conv_out(w, self.k, self.stride, self.pad))

    def macs(self, in_shape):
        _, oh, ow = self.output_shape(in_shape)
        return self.out_c * oh * ow * self.in_c * self.k * self.k

    def spec(self, in_shape, quantize=False):
        s = {"type": "conv2d", "in_c": self.in_c, "out_c": self.out_c, "k": self.k, "stride": self.stride,
             "pad": self.pad, "b": self.b}
        Wm = self.W.reshape(self.out_c, -1)
        if quantize:
            s["w"], s["w_scale"] = quantize_symmetric(Wm)
            s["dtype"] = "i8"
        else:
            s["w"], s["w_scale"], s["dtype"] = Wm, 1.0, "f32"
        return s


class ReLU(Layer):
    def __init__(self):
        self._mask: np.ndarray | None = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dy):
        return dy * self._mask

    def forward_int8(self, x):
        return self.forward(x)

    def spec(self, in_shape, quantize=False):
        return {"type": "relu"}


class MaxPool2D(Layer):
    """Non-overlapping max pooling (``k == stride``, input divisible by ``k``)."""

    def __init__(self, k: int = 2):
        self.k = k
        self._cache: tuple | None = None

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.k
        if H % k or W % k:
            raise ValueError(f"MaxPool2D({k}) needs input divisible by {k}, got {H}x{W}")
        xr = x.reshape(B, C, H // k, k, W // k, k)
        y = xr.max(axis=(3, 5))
        self._cache = (xr, y)
        return y

    def backward(self, dy):
        xr, y = self._cache
        mask = xr == y[:, :, :, None, :, None]
        # Distribute the gradient to the (first) arg-max of each window.
        B, C, oh, k, ow, _ = xr.shape
        flat = mask.transpose(0, 1, 2, 4, 3, 5).reshape(B, C, oh, ow, k * k)
        first = np.zeros_like(flat)
        idx = flat.argmax(axis=-1)
        np.put_along_axis(first, idx[..., None], True, axis=-1)
        dxr = first.reshape(B, C, oh, ow, k, k).transpose(0, 1, 2, 4, 3, 5) * dy[:, :, :, None, :, None]
        return dxr.reshape(B, C, oh * k, ow * k)

    def forward_int8(self, x):
        return self.forward(x)

    def output_shape(self, in_shape):
        c, h, w = in_shape
        return (c, h // self.k, w // self.k)

    def spec(self, in_shape, quantize=False):
        return {"type": "maxpool2d", "k": self.k, "stride": self.k}


class Flatten(Layer):
    def __init__(self):
        self._shape: Shape | None = None

    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dy):
        return dy.reshape(self._shape)

    def forward_int8(self, x):
        return self.forward(x)

    def output_shape(self, in_shape):
        return (int(np.prod(in_shape)),)

    def spec(self, in_shape, quantize=False):
        return {"type": "flatten"}


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy and its gradient w.r.t. the logits."""
    p = softmax(logits)
    B = logits.shape[0]
    loss = -float(np.log(p[np.arange(B), labels] + 1e-12).mean())
    dlogits = p
    dlogits[np.arange(B), labels] -= 1.0
    return loss, dlogits / B
