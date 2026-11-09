"""Deterministic noise models delta(x) for a zeroth-order oracle f(x) + delta(x).

All profiles act along a fixed unit direction u, so delta(x) = phi(<u, x>).
This keeps the worst case of the analysis (the noise is aligned with the
objective) while letting the exact bias be computed by 1D quadrature.
"""

import numpy as np


def step_profile(z, amp):
    """Bounded adversarial step: phi(z) = amp * sign(z). Jump 2*amp, not Lipschitz."""
    return amp * np.sign(z)


def ramp_profile(z, amp, slope):
    """Lipschitz ramp: phi(z) = amp * clip(slope * z / amp, -1, 1).

    |phi| <= amp and phi is slope-Lipschitz; slope -> inf recovers the step.
    """
    if np.isinf(slope):
        return step_profile(z, amp)
    return amp * np.clip(slope * z / amp, -1.0, 1.0)


class AlignedNoise:
    """delta(x) = phi(<u, x>) for a 1D profile phi."""

    def __init__(self, profile, u, **params):
        self.profile = profile
        self.u = np.asarray(u, dtype=float)
        self.params = params

    def __call__(self, x):
        return self.profile(x @ self.u, **self.params)


def lipschitz_envelope(values, grid, slope):
    """Inf-convolution delta_L(x) = min_y [delta(y) + slope * |x - y|] on a 1D grid.

    For noise with |delta(x) - delta(y)| <= delta0 + slope * |x - y| it returns a
    slope-Lipschitz function with 0 <= delta - delta_L <= delta0 (Lemma 1 of the notes).
    """
    dist = np.abs(grid[:, None] - grid[None, :])
    return np.min(values[None, :] + slope * dist, axis=1)
