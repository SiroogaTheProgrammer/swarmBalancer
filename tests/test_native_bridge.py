""".swm round trip and the Python <-> C++ engine agreement."""

import numpy as np
import pytest

from swarm.brain import Sequential, native
from swarm.brain.formats import read_swm

from conftest import needs_native


def test_swm_roundtrip_preserves_predictions(model, swm_paths, frames):
    ref = model.predict_proba(frames)
    back = Sequential.load(swm_paths["f32"])
    assert np.allclose(back.predict_proba(frames), ref, atol=1e-6)
    back8 = Sequential.load(swm_paths["int8"])
    assert np.abs(back8.predict_proba(frames) - ref).max() < 0.1


def test_swm_layout(swm_paths):
    shape, layers = read_swm(swm_paths["int8"])
    assert shape == (1, 16, 16)
    assert [L["type"] for L in layers][-1] == "softmax"
    conv = layers[0]
    assert conv["dtype"] == "i8" and conv["w"].dtype == np.int8 and conv["w"].shape == (4, 9)
    assert abs(conv["w"]).max() == 127


@needs_native
def test_native_matches_numpy_f32(model, swm_paths, frames):
    with native.NativeModel(swm_paths["f32"]) as nm:
        assert nm.input_shape == (1, 16, 16) and nm.output_shape == (5, 1, 1)
        assert nm.macs == model.macs()
        got = nm.run_batch(frames)
    assert np.allclose(got, model.predict_proba(frames), atol=1e-5)


@needs_native
def test_native_matches_numpy_int8(model, swm_paths, frames):
    with native.NativeModel(swm_paths["int8"]) as nm:
        got = nm.run_batch(frames)
        assert nm.weights_bytes == model.weights_bytes(True)
    assert np.allclose(got, model.predict_proba(frames, int8=True), atol=1e-5)


@needs_native
def test_native_ram_cap_enforced(swm_paths):
    with native.NativeModel(swm_paths["f32"]) as nm:
        need = nm.required_bytes
    with pytest.raises(MemoryError, match="device cap"):
        native.NativeModel(swm_paths["f32"], ram_cap_bytes=need - 1)
    with native.NativeModel(swm_paths["f32"], ram_cap_bytes=need) as ok:
        assert ok.required_bytes == need


@needs_native
def test_native_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.swm"
    bad.write_bytes(b"SWM1" + b"\xff" * 64)
    with pytest.raises(MemoryError):
        native.NativeModel(bad)


@needs_native
def test_native_gemm_kernels_match_numpy():
    rng = np.random.default_rng(3)
    for M, N, K in [(1, 5, 1024), (7, 33, 70), (64, 64, 300)]:
        a = rng.standard_normal((M, K)).astype(np.float32)
        b = rng.standard_normal((K, N)).astype(np.float32)
        assert np.allclose(native.gemm_f32(a, b), a @ b, rtol=1e-4, atol=1e-3)
        a8 = rng.integers(-128, 128, (M, K), dtype=np.int8)
        b8 = rng.integers(-128, 128, (K, N), dtype=np.int8)
        assert np.array_equal(native.gemm_i8(a8, b8), a8.astype(np.int32) @ b8.astype(np.int32))
