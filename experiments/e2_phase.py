"""E2. Phase transition (slide 29 of the Shenzhen talk) with a Lipschitz adversary.

Objective f(x) = -g0 <u, x> on the unit ball, g0 = 2 eps / R, minimiser x* = u.
Oracle f(x) + phi(<u, x>) with phi a ramp of height amp and slope M (M = inf: step).
The noise builds a wall across the hyperplane <u, x> = 0 that the method must cross.
Projected two-point ZO-SGD from x0 = -0.6 u; success means f(x_bar) - f* <= eps.

Prediction: the wall stops the method iff the largest noise bias exceeds g0, and the bias
is at most min(M, ~sqrt(2d/pi) amp / gamma) (Lemma 2). So Lipschitz noise with M < g0 is
harmless at any amplitude, while a step blocks already at amp ~ g0 gamma / sqrt(d).
"""

from functools import partial

import numpy as np
from common import plt, save

from lipzo.estimators import aligned_noise_bias, random_directions, two_point
from lipzo.methods import project_ball

D, EPS, R = 20, 0.1, 1.0
G0 = 2 * EPS / R
GAMMA = 0.05
N_ITER, SEEDS, STEP = 3000, 16, 0.01
AMPS = np.logspace(-4, 0, 13)
SLOPES = np.concatenate([np.logspace(-2, 1.3, 12), [np.inf]])


def predicted_bias(amp, slope):
    phi = lambda z: amp * (np.sign(z) if np.isinf(slope) else np.clip(slope * z / amp, -1, 1))
    s_grid = np.linspace(-1.5 * GAMMA, 1.5 * GAMMA, 31)
    return max(aligned_noise_bias(phi, s, GAMMA, D, n_grid=40_001) for s in s_grid)


def run():
    rng = np.random.default_rng(0)
    A, M = np.meshgrid(AMPS, SLOPES, indexing="ij")
    amp = np.repeat(A.ravel(), SEEDS)[:, None]
    slope = np.repeat(M.ravel(), SEEDS)[:, None]

    def oracle(y):
        z = y[..., 0]
        wall = np.where(np.isinf(slope[:, 0]), np.sign(z),
                        np.clip(np.where(np.isinf(slope[:, 0]), 0, slope[:, 0]) * z / amp[:, 0], -1, 1))
        return -G0 * z + amp[:, 0] * wall

    x = np.zeros((amp.shape[0], D))
    x[:, 0] = -0.6
    avg, start = np.zeros_like(x), N_ITER // 2
    for k in range(N_ITER):
        e = random_directions(rng, (x.shape[0],), D)
        x = project_ball(x - STEP * two_point(oracle, x, GAMMA, e), R)
        if k >= start:
            avg += x
    avg /= N_ITER - start
    err = G0 * (R - avg[:, 0])
    success = (err <= EPS).reshape(len(AMPS), len(SLOPES), SEEDS).mean(axis=2)
    return success


def main():
    success = run()
    pred = np.array([[predicted_bias(a, m) for m in SLOPES] for a in AMPS])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    xs = np.arange(len(SLOPES))
    im = ax.imshow(1 - success, origin="lower", aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.contour(xs, np.arange(len(AMPS)), pred / G0, levels=[1.0], colors="tab:blue", linewidths=1.5)
    ax.set_xticks(xs, [f"{m:.2g}" if np.isfinite(m) else "∞\n(ступ.)" for m in SLOPES], fontsize=7, rotation=45)
    ax.set_yticks(np.arange(len(AMPS)), [f"{a:.0e}" for a in AMPS], fontsize=7)
    ax.axvline(np.searchsorted(SLOPES, G0) - 0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"наклон шума $M$ (пунктир: $M=g_0=2\varepsilon/R$)")
    ax.set_ylabel(r"амплитуда шума $\Delta$")
    ax.set_title(rf"Доля неудач ZO-SGD, $d={D}$, $\gamma={GAMMA}$; синяя линия: смещение $=g_0$")
    fig.colorbar(im, ax=ax, label="доля запусков без точности $\\varepsilon$")
    save(fig, "e2_phase", {"amps": AMPS.tolist(), "slopes": [str(m) for m in SLOPES],
                          "success": success.tolist(), "predicted_bias_over_g0": (pred / G0).tolist(),
                          "d": D, "eps": EPS, "gamma": GAMMA, "n_iter": N_ITER, "seeds": SEEDS})
    agree = ((pred / G0 < 1) == (success > 0.5)).mean()
    print(f"agreement of the prediction 'bias < g0' with 'success > 1/2': {agree:.3f}")


if __name__ == "__main__":
    main()
