"""Training on the synthetic dataset actually learns."""

import numpy as np

from swarm.train.dataset import NUM_CLASSES, FramePool, FrameStream, make_dataset
from swarm.train.train_tiny_cnn import accuracy, build_tiny_cnn, train


def test_dataset_classes_are_visually_distinct():
    ds = make_dataset(150, 32, seed=0)
    assert ds.x.shape == (150, 1, 32, 32) and ds.x.dtype == np.float32
    assert set(np.unique(ds.y)) <= set(range(NUM_CLASSES))
    means = [ds.x[ds.y == c].mean() for c in range(NUM_CLASSES)]
    assert max(means) - min(means) > 0.02, "objects must change the image (regression: painting into a copy)"


def test_frame_stream_object_rate_and_pool():
    pool = FramePool(32, per_class=4, seed=1)
    s = FrameStream(32, seed=2, p_object=0.5, pool=pool)
    labels = [s.next()[1] for _ in range(400)]
    rate = np.mean([l != 0 for l in labels])
    assert 0.4 < rate < 0.6


def test_tiny_cnn_learns_quickly():
    train_ds = make_dataset(1200, 32, seed=0)
    test_ds = make_dataset(300, 32, seed=1)
    m = build_tiny_cnn(32, width=8, seed=0)
    before = accuracy(m, test_ds.x, test_ds.y)
    train(m, train_ds, test_ds, epochs=3, batch_size=32, lr=2e-3, seed=0, log=lambda *_: None)
    after = accuracy(m, test_ds.x, test_ds.y)
    assert after > max(0.5, before + 0.2), (before, after)  # chance is 0.2
