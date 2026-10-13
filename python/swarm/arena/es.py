"""Evolution strategies (OpenAI-ES): antithetic noise, centred ranks, Adam.

No gradients through the simulator are needed - each generation tries ``pairs`` random directions both
ways, ranks the results and steps the weights towards the better ones. Robust for small networks and
noisy, non-differentiable scores such as "attackers that got through".
"""

from __future__ import annotations

import numpy as np


def centred_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks scaled to [-0.5, 0.5]: only the order of the scores matters, not their scale."""
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(1, len(x) - 1) - 0.5


class ES:
    def __init__(self, theta: np.ndarray, sigma: float = 0.1, lr: float = 0.05, pairs: int = 8, seed: int = 0,
                 l2: float = 0.002):
        self.theta = np.asarray(theta, dtype=float).copy()
        self.sigma, self.lr, self.pairs, self.l2 = float(sigma), float(lr), int(pairs), float(l2)
        self.rng = np.random.default_rng(seed)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.t = 0
        self.eps: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        """``2 * pairs`` weight vectors to try: theta + sigma * eps for each eps, then theta - sigma * eps."""
        self.eps = self.rng.standard_normal((self.pairs, self.theta.size))
        return np.concatenate([self.theta + self.sigma * self.eps, self.theta - self.sigma * self.eps])

    def tell(self, fitness: np.ndarray) -> float:
        """Scores of the vectors from :meth:`ask` (higher is better); returns the step size taken."""
        if self.eps is None:
            raise RuntimeError("tell() without ask()")
        r = centred_ranks(np.asarray(fitness, dtype=float))
        g = (r[:self.pairs] - r[self.pairs:]) @ self.eps / (2 * self.pairs * self.sigma) - self.l2 * self.theta
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        step = self.lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)
        self.theta += step
        self.eps = None
        return float(np.sqrt((step * step).sum()))
