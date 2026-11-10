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


def test_ramp_belongs_to_both_classes():
    """Remark after the corollary: a ramp of height amp and slope M lies in N(M, 0) and in N(0, 2 amp)."""
    amp, slope = 0.3, 2.0
    z = np.linspace(-1, 1, 2001)
    phi = ramp_profile(z, amp, slope)
    diff = np.abs(phi[:, None] - phi[None, :])
    dist = np.abs(z[:, None] - z[None, :])
    assert (diff <= slope * dist + 1e-12).all()
    assert (diff <= 2 * amp + 1e-12).all()


def test_lower_bound_oracle_is_blind():
    """Theorem 2: f_plus + delta_plus and f_minus + delta_minus both answer 0 everywhere."""
    rng = np.random.default_rng(3)
    d, a = 7, 0.4
    u = random_directions(rng, (), d)
    y = rng.standard_normal((1000, d)) * 5
    for sign in (+1, -1):
        f = sign * a * (y @ u)
        delta = -f
        assert np.allclose(f + delta, 0.0)


@pytest.mark.parametrize("d", [5, 40])
def test_mixed_noise_bias_bound(d):
    """Lemma 2: noise = M-Lipschitz part + jump of size delta0 gives bias <= M + sqrt(d) delta0 / (2 gamma)."""
    slope, delta0, gamma = 0.05, 1e-3, 0.05
    phi = lambda z: slope * z + delta0 * (z > 0)
    bound = slope + np.sqrt(d) * delta0 / (2 * gamma)
    worst = max(abs(aligned_noise_bias(phi, s, gamma, d)) for s in np.linspace(-2 * gamma, 2 * gamma, 41))
    assert worst <= bound * (1 + 1e-6)


@pytest.mark.parametrize("d", [5, 30])
def test_mixed_noise_second_moment(d):
    """Lemma 2: E||g||^2 <= 4 d (M_f + M)^2 + d^2 delta0^2 / (2 gamma^2)."""
    rng = np.random.default_rng(4)
    n, gamma, m_f, slope, delta0 = 200_000, 0.05, 1.0, 0.1, 2e-3
    u = np.zeros(d)
    u[0] = 1.0
    c = 0.3 * random_directions(rng, (), d)
    oracle = lambda y: m_f * np.linalg.norm(y - c, axis=-1) + slope * (y @ u) + delta0 * ((y @ u) > 0)
    e = random_directions(rng, (n,), d)
    g = two_point(oracle, np.zeros((n, d)), gamma, e)
    second = (g * g).sum(axis=1).mean()
    assert second <= 4 * d * (m_f + slope) ** 2 + d**2 * delta0**2 / (2 * gamma**2)


def test_theorem1_small_run():
    """Theorem 1 with its own constants reaches accuracy eps (d = 3, eps = 0.3)."""
    from lipzo.methods import project_ball

    rng = np.random.default_rng(5)
    d, eps, m_f, D = 3, 0.3, 1.0, 2.0
    M, delta0 = eps / (6 * D), eps**2 / (9 * np.sqrt(d) * m_f * D)
    gamma = eps / (3 * m_f)
    n_iter = int(np.ceil(45 * d * (m_f + M) ** 2 * D**2 / eps**2))
    step = D / (np.sqrt(5 * d) * (m_f + M) * np.sqrt(n_iter))
    c = np.array([0.3, -0.2, 0.1])
    u = np.array([1.0, 0.0, 0.0])
    oracle = lambda y: m_f * np.linalg.norm(y - c, axis=-1) + M * (y @ u) + delta0 * ((y @ u) > 0)
    x = np.zeros((4, d))
    avg = np.zeros_like(x)
    for _ in range(n_iter):
        avg += x
        e = random_directions(rng, (4,), d)
        x = project_ball(x - step * two_point(oracle, x, gamma, e), 1.0)
    err = m_f * np.linalg.norm(avg / n_iter - c, axis=-1)
    assert (err <= eps).all()
