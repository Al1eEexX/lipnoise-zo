"""E1. Bias of the two-point estimator: bounded step noise vs Lipschitz ramp noise.

Exact 1D quadrature (no Monte Carlo). Prediction (Lemma 2 and survey eq. (13)):
step noise of height 2*amp gives bias (d amp / gamma) E|t| ~ sqrt(2d/pi) amp / gamma,
a Lipschitz ramp with slope M gives bias <= M, independent of d and gamma.
"""

from functools import partial

import numpy as np
from common import plt, save

from lipzo.estimators import aligned_noise_bias, expected_abs_coordinate
from lipzo.noise import ramp_profile, step_profile

AMP, GAMMA = 1e-3, 0.05
DIMS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
SLOPES = [0.01, 0.03, 0.1, 0.3]


def max_bias(phi, d):
    s_grid = np.linspace(-2 * GAMMA, 2 * GAMMA, 81)
    return max(abs(aligned_noise_bias(phi, s, GAMMA, d)) for s in s_grid)


def main():
    step = [max_bias(partial(step_profile, amp=AMP), d) for d in DIMS]
    ramps = {M: [max_bias(partial(ramp_profile, amp=AMP, slope=M), d) for d in DIMS] for M in SLOPES}
    theory_step = [d * AMP / GAMMA * expected_abs_coordinate(d) for d in DIMS]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(DIMS, step, "o-", color="black", label=r"ступенька $2\Delta$ (ограниченный шум)")
    ax.loglog(DIMS, theory_step, ":", color="black", label=r"теория $\frac{d\Delta}{\gamma}\mathbb{E}|t|\approx\sqrt{2d/\pi}\,\Delta/\gamma$")
    for M, color in zip(SLOPES, plt.cm.viridis(np.linspace(0.15, 0.85, len(SLOPES)))):
        ax.loglog(DIMS, ramps[M], "s-", color=color, ms=4, label=rf"липшицев шум, $M={M}$")
        ax.axhline(M, color=color, ls="--", lw=0.8)
    ax.set_xlabel("размерность $d$")
    ax.set_ylabel(r"$\max_x\|\mathbb{E}\,[\text{шумовая часть } g]\|$")
    ax.set_title(rf"Смещение оценки: $\Delta={AMP}$, $\gamma={GAMMA}$")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "e1_bias", {"dims": DIMS, "amp": AMP, "gamma": GAMMA, "step": step,
                         "theory_step": theory_step, "ramps": {str(k): v for k, v in ramps.items()}})
    print("step bias d=2..1024:", np.round(step, 4))
    for M in SLOPES:
        print(f"ramp M={M}: max bias / M =", np.round(np.array(ramps[M]) / M, 4))


if __name__ == "__main__":
    main()
