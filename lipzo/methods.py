"""Projected zeroth-order SGD on a Euclidean ball, vectorised over independent runs."""

import numpy as np

from .estimators import random_directions, two_point


def project_ball(x, radius):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x * np.minimum(1.0, radius / np.maximum(norm, 1e-300))


def zo_sgd(oracle, x0, gamma, step, n_iter, radius, rng, tail=0.5):
    """Projected two-point ZO-SGD with tail averaging.

    x0 has shape (..., d); every leading index is an independent run sharing
    the oracle. `gamma` and `step` may be arrays broadcastable to x0[..., :1].
    Returns the average of the last `tail` fraction of iterates.
    """
    x = np.array(x0, dtype=float)
    start = int((1.0 - tail) * n_iter)
    avg = np.zeros_like(x)
    for k in range(n_iter):
        e = random_directions(rng, x.shape[:-1], x.shape[-1])
        g = two_point(oracle, x, gamma, e)
        x = project_ball(x - step * g, radius)
        if k >= start:
            avg += x
    return avg / (n_iter - start)
