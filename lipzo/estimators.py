"""Two-point randomized gradient estimator and exact sphere integrals."""

from math import lgamma, log, pi

import numpy as np


def random_directions(rng, shape, d):
    """Uniform directions on the unit sphere S^{d-1}; output shape (*shape, d)."""
    e = rng.standard_normal((*shape, d))
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


def two_point(oracle, x, gamma, e):
    """g = d / (2 gamma) * (F(x + gamma e) - F(x - gamma e)) e, vectorised over leading axes."""
    d = x.shape[-1]
    diff = oracle(x + gamma * e) - oracle(x - gamma * e)
    return (d / (2.0 * gamma)) * diff[..., None] * e


def sphere_coordinate_density(t, d):
    """Density of t = <e, u> for e ~ Unif(S^{d-1}): c_d (1 - t^2)^{(d-3)/2} on (-1, 1)."""
    log_c = lgamma(d / 2) - lgamma((d - 1) / 2) - 0.5 * log(pi)
    if d == 3:
        return np.full_like(t, np.exp(log_c))
    with np.errstate(divide="ignore"):
        log_density = log_c + 0.5 * (d - 3) * np.log1p(-t * t)
    return np.exp(log_density)


def expected_abs_coordinate(d):
    """E|<e, u>| = Gamma(d/2) / (sqrt(pi) Gamma((d+1)/2))."""
    return np.exp(lgamma(d / 2) - lgamma((d + 1) / 2) - 0.5 * log(pi))


def aligned_noise_bias(phi, s, gamma, d, n_grid=400_001):
    """Exact mean of the noise part of the two-point estimator for delta(x) = phi(<u, x>).

    At a point with <u, x> = s the mean is b(s) u with
    b(s) = d / (2 gamma) * E[(phi(s + gamma t) - phi(s - gamma t)) t],  t = <e, u>,
    computed by the trapezoid rule over the exact density of t (no Monte Carlo error
    beyond quadrature; components orthogonal to u vanish by symmetry).
    """
    t = np.linspace(-1.0, 1.0, n_grid)
    w = sphere_coordinate_density(t, d)
    integrand = (phi(s + gamma * t) - phi(s - gamma * t)) * t * w
    return d / (2.0 * gamma) * np.trapezoid(integrand, t)
