"""E5. The rounding dead zone: how its threshold in gamma scales with the dimension.

f(x) = 1 + ||x - c||_1 / sqrt(d) (||grad f|| = 1) evaluated in fp16. Two values
f(x +- gamma e) differ by about 2 gamma <grad f, e> ~ 2 gamma / sqrt(d); once this is below
the rounding step q = ulp(f) both values round to the same number and the estimate is zero.
Heuristic prediction: the method stalls for gamma below gamma_crit ~ sqrt(d) q / M_f.
We measure gamma_crit(d) as the smallest gamma at which ZO-SGD halves the initial error.
"""

import numpy as np
from common import plt, save

from lipzo.methods import zo_sgd

DIMS = [4, 8, 16, 32, 64, 128]
GAMMAS = np.logspace(-4.5, -1.0, 71)
N_ITER, SEEDS, RADIUS = 3000, 6, 2.0
ULP = 2.0**-10  # spacing of fp16 numbers in [1, 2)


def fp16_oracle(c):
    scale = np.float16(1.0 / np.sqrt(c.shape[-1]))
    c16 = c.astype(np.float16)

    def f(y):
        return (np.float16(1) + np.abs(y.astype(np.float16) - c16).sum(-1, dtype=np.float16) * scale).astype(float)

    return f


def main():
    rng = np.random.default_rng(0)
    crit, curves = [], {}
    for d in DIMS:
        c = rng.standard_normal(d)
        c *= 0.6 / np.linalg.norm(c)
        exact = lambda y, c=c, d=d: np.abs(y - c).sum(-1) / np.sqrt(d)
        err0 = exact(np.zeros(d))
        gam = np.repeat(GAMMAS, SEEDS)[:, None]
        step = RADIUS / (np.sqrt(2 * d) * np.sqrt(N_ITER))
        x_bar = zo_sgd(fp16_oracle(c), np.zeros((gam.shape[0], d)), gam, step, N_ITER, RADIUS, rng)
        ratio = (exact(x_bar) / err0).reshape(len(GAMMAS), SEEDS).mean(axis=1)
        curves[d] = ratio.tolist()
        ok = np.nonzero(ratio < 0.5)[0]
        g_crit = float(GAMMAS[ok[0]]) if ok.size else float("nan")
        crit.append(g_crit)
        print(f"d={d:3d}: gamma_crit = {g_crit:.2e},  gamma_crit*M_f/(sqrt(d) ulp) = {g_crit/(np.sqrt(d)*ULP):.2f}")

    slope = np.polyfit(np.log(DIMS), np.log(crit), 1)[0]
    print(f"fitted slope of log gamma_crit vs log d: {slope:.2f} (prediction 0.5)")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    shades = plt.cm.Blues(np.linspace(0.4, 0.95, len(DIMS)))
    for d, col in zip(DIMS, shades):
        ax.semilogx(GAMMAS, curves[d], "o-", ms=3, color=col, label=f"$d={d}$")
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.set_xlabel(r"радиус сглаживания $\gamma$")
    ax.set_ylabel("ошибка / начальная ошибка")
    ax.set_title("fp16: где метод начинает двигаться", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.loglog(DIMS, crit, "o-", color="#2a78d6", label=f"измерено (наклон {slope:.2f})")
    const = np.exp(np.mean(np.log(crit) - 0.5 * np.log(DIMS)))  # best constant for the sqrt(d) law
    ref = const * np.sqrt(np.array(DIMS))
    ax.loglog(DIMS, ref, "--", color="#6b6a64", label=rf"${const/ULP:.2f}\,\sqrt{{d}}\,\mathrm{{ulp}}/M_f$")
    ax.set_xlabel("размерность $d$")
    ax.set_ylabel(r"порог $\gamma_{\mathrm{crit}}$")
    ax.set_title("Порог мёртвой зоны растёт как $\\sqrt{d}$", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "e5_dead_zone", {"dims": DIMS, "gammas": GAMMAS.tolist(), "gamma_crit": crit, "slope": float(slope),
                              "error_ratio": {str(k): v for k, v in curves.items()}, "ulp": ULP, "n_iter": N_ITER})


if __name__ == "__main__":
    main()
