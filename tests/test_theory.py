"""Numerical checks of the statements in notes/lipschitz_noise.tex."""

from functools import partial

import numpy as np
import pytest

from lipzo.estimators import (
    aligned_noise_bias,
    expected_abs_coordinate,
    random_directions,
    two_point,
)
from lipzo.noise import lipschitz_envelope, ramp_profile, step_profile


def test_two_point_unbiased_for_linear():
    """For linear f the two-point estimator is unbiased for grad f = a."""
    rng = np.random.default_rng(0)
    d, n = 8, 400_000
    a = rng.standard_normal(d)
    e = random_directions(rng, (n,), d)
    g = two_point(lambda y: y @ a, np.zeros((n, d)), 0.3, e)
    assert np.allclose(g.mean(axis=0), a, atol=5 * np.sqrt(d / n) * np.linalg.norm(a))


@pytest.mark.parametrize("d", [5, 20, 100])
def test_second_moment_bound(d):
    """E||g||^2 <= 2 d M^2 for an M-Lipschitz f (Poincare on the sphere)."""
    rng = np.random.default_rng(1)
    n = 200_000
    e = random_directions(rng, (n,), d)
    f = lambda y: np.abs(y).sum(axis=-1)  # M = sqrt(d)
    x = np.zeros((n, d))
    x[:, 0] = 0.05
    g = two_point(f, x, 0.1, e)
    assert (g * g).sum(axis=1).mean() <= 2 * d * d


@pytest.mark.parametrize("d", [3, 10, 50, 200])
def test_step_bias_matches_formula(d):
    """Bounded step noise: bias at the jump is (d * amp / gamma) * E|t| ~ sqrt(2d/pi) amp / gamma."""
    amp, gamma = 0.01, 0.1
    b = aligned_noise_bias(partial(step_profile, amp=amp), 0.0, gamma, d)
    assert b == pytest.approx(d * amp / gamma * expected_abs_coordinate(d), rel=1e-4)


@pytest.mark.parametrize("d", [3, 10, 50, 200])
def test_ramp_bias_equals_slope(d):
    """Lipschitz ramp wider than the probe: bias equals the slope M exactly, in every dimension."""
    amp, slope, gamma = 1.0, 0.2, 0.1  # ramp half-width amp/slope = 5 > gamma
    b = aligned_noise_bias(partial(ramp_profile, amp=amp, slope=slope), 0.3, gamma, d)
    assert b == pytest.approx(slope, rel=1e-6)


@pytest.mark.parametrize("d", [3, 50])
def test_ramp_bias_never_exceeds_slope(d):
    """Lemma 2: |bias| <= M for any M-Lipschitz noise, at any point and any gamma."""
    for slope in [0.05, 0.5, 5.0]:
        for s in np.linspace(-0.3, 0.3, 13):
            b = aligned_noise_bias(partial(ramp_profile, amp=0.01, slope=slope), s, 0.05, d)
            assert abs(b) <= slope * (1 + 1e-6)


def test_lipschitz_envelope_decomposition():
    """Lemma 1: delta_L is M-Lipschitz and 0 <= delta - delta_L <= delta0."""
    rng = np.random.default_rng(2)
    grid = np.linspace(-1, 1, 801)
    slope, delta0 = 0.7, 0.05
    smooth = 0.3 * np.sin(2 * grid) + 0.2 * grid  # Lipschitz constant <= 0.8
    smooth = slope / 0.8 * smooth
    delta = smooth + rng.uniform(0, delta0, grid.size)  # |delta(x)-delta(y)| <= delta0 + slope|x-y|
    env = lipschitz_envelope(delta, grid, slope)
    gap = delta - env
    assert gap.min() >= -1e-12 and gap.max() <= delta0 + 1e-12
    lip = np.abs(np.diff(env)) / np.diff(grid)
    assert lip.max() <= slope * (1 + 1e-9)
