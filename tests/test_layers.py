"""numpy layer maths: gradients agree with finite differences; int8 path tracks f32."""

import numpy as np

from swarm.brain import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential, softmax_cross_entropy
from swarm.brain.layers import im2col, col2im


def numerical_grad(f, p, eps=1e-3, n=12):
    g = np.zeros_like(p)
    flat = p.reshape(-1)
    gflat = g.reshape(-1)
    for i in np.linspace(0, flat.size - 1, n).astype(int):
        old = flat[i]
        flat[i] = old + eps
        lp = f()
        flat[i] = old - eps
        lm = f()
        flat[i] = old
        gflat[i] = (lp - lm) / (2 * eps)
    return g


def test_param_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    m = Sequential([Conv2D(1, 3, rng=rng), ReLU(), MaxPool2D(2), Conv2D(3, 4, rng=rng), ReLU(), MaxPool2D(2),
                    Flatten(), Dense(16, 5, rng=rng)], (1, 8, 8))
    for L in m.layers:
        for p in L.params().values():
            p[...] = p.astype(np.float64)
    x = rng.standard_normal((4, 1, 8, 8))
    y = np.array([0, 1, 2, 3])

    def loss():
        return softmax_cross_entropy(m.forward(x), y)[0]

    _, d = softmax_cross_entropy(m.forward(x), y)
    m.backward(d)
    for L in m.layers:
        for name, p in L.params().items():
            ng = numerical_grad(loss, p)
            ag = L.grads()[name]
            mask = ng != 0
            assert np.allclose(ag[mask], ng[mask], rtol=1e-2, atol=1e-4), f"{type(L).__name__}.{name}"


def test_input_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    m = Sequential([Conv2D(1, 2, rng=rng), ReLU(), MaxPool2D(2), Flatten(), Dense(8, 3, rng=rng)], (1, 4, 4))
    x = rng.standard_normal((2, 1, 4, 4))
    y = np.array([0, 2])
    _, d = softmax_cross_entropy(m.forward(x), y)
    dx = m.backward(d)
    eps = 1e-4
    for idx in [(0, 0, 0, 0), (1, 0, 2, 3), (0, 0, 3, 1)]:
        old = x[idx]
        x[idx] = old + eps
        lp = softmax_cross_entropy(m.forward(x), y)[0]
        x[idx] = old - eps
        lm = softmax_cross_entropy(m.forward(x), y)[0]
        x[idx] = old
        assert abs(dx[idx] - (lp - lm) / (2 * eps)) < 1e-3


def test_im2col_roundtrip_counts_overlaps():
    x = np.arange(2 * 1 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4)
    cols, oh, ow = im2col(x, 3, 1, 1)
    assert cols.shape == (2, 9, 16) and (oh, ow) == (4, 4)
    back = col2im(np.ones_like(cols), x.shape, 3, 1, 1, oh, ow)
    # each interior pixel is covered by 9 windows, corners by 4
    assert back[0, 0, 1, 1] == 9 and back[0, 0, 0, 0] == 4


def test_int8_forward_tracks_f32(model, frames):
    p32 = model.predict_proba(frames)
    p8 = model.predict_proba(frames, int8=True)
    assert np.abs(p32 - p8).max() < 0.05
    assert (p32.argmax(1) == p8.argmax(1)).mean() >= 5 / 6


def test_shapes_and_macs(model):
    assert model.output_shape == (5,)
    assert model.macs() == 4 * 16 * 16 * 9 + 8 * 8 * 8 * 36 + 128 * 16 + 16 * 5
    assert model.weights_bytes(True) < model.weights_bytes() / 3
