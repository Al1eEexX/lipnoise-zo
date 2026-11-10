"""E4. Theorems 1 and 2 checked numerically.

(a) Theorem 1 with its own constants: gamma = eps/(3 M_f), G^2 = 5 d (M_f + M)^2,
    N = ceil(45 d (M_f + M)^2 D^2 / eps^2), h = D / (G sqrt N), noise at the largest
    levels the theorem allows (M = eps/(6D), Delta0 = eps^2/(9 sqrt(d) M_f D)).
    Prediction: E f(x_bar) - f* <= eps for every d and every instance.
(b) Thresholds: the Lipschitz threshold does not move with d, the bounded one moves as 1/sqrt(d).
    Linear objective f(x) = -g0 <u, x> on the unit ball; noise either the linear "counter-slope"
    M <u, x> (Lipschitz, any amplitude) or a step Delta * sign(<u, x>) (bounded).
"""

import numpy as np
from common import plt, save

from lipzo.estimators import random_directions, two_point
from lipzo.methods import project_ball

RHO = 1.0
D = 2 * RHO


def run_sgd(oracle, x0, gamma, step, n_iter, rng):
    """Projected two-point ZO-SGD on the ball, plain averaging of x_0..x_{N-1}."""
    x = x0.copy()
    avg = np.zeros_like(x)
    for _ in range(n_iter):
        avg += x
        e = random_directions(rng, x.shape[:-1], x.shape[-1])
        x = project_ball(x - step * two_point(oracle, x, gamma, e), RHO)
    return avg / n_iter


def theorem1(rng):
    eps, m_f, seeds = 0.2, 1.0, 8
    dims = [5, 10, 20, 40]
    m_noise = eps / (6 * D)
    rows = []
    for d in dims:
        delta0 = eps**2 / (9 * np.sqrt(d) * m_f * D)
        gamma = eps / (3 * m_f)
        g2 = 5 * d * (m_f + m_noise) ** 2
        n_iter = int(np.ceil(45 * d * (m_f + m_noise) ** 2 * D**2 / eps**2))
        step = D / (np.sqrt(g2) * np.sqrt(n_iter))
        u = np.zeros(d)
        u[0] = 1.0
        c = 0.5 * random_directions(rng, (), d)
        for inst, slope in [("крутая, a=M_f", m_f), ("пологая, a=3ε/D", 3 * eps / D)]:
            f = lambda y, a=slope: a * np.linalg.norm(y - c, axis=-1)
            for noise_name, with_noise in [("без шума", False), ("шум на пределе теоремы", True)]:
                def oracle(y, f=f, with_noise=with_noise):
                    val = f(y)
                    if with_noise:
                        z = y @ u
                        val = val + m_noise * z + delta0 * (z > 0)  # Lipschitz part + jump of size delta0
                    return val
                x0 = np.zeros((seeds, d))
                x0[:, 0] = -0.9
                x_bar = run_sgd(oracle, x0, gamma, step, n_iter, rng)
                err = f(x_bar)  # f* = 0 at c (inside the ball)
                rows.append({"d": d, "instance": inst, "noise": noise_name, "N": n_iter,
                             "mean_err_over_eps": float(err.mean() / eps), "max_err_over_eps": float(err.max() / eps)})
                print(f"d={d:3d} N={n_iter:7d} {inst:16s} {noise_name:24s} err/eps mean={err.mean()/eps:.3f} max={err.max()/eps:.3f}")
    return rows


def thresholds(rng):
    eps, gamma, n_iter, seeds = 0.1, 0.05, 4000, 8
    g0 = 2 * eps / RHO
    dims = [5, 20, 80]
    m_grid = g0 * np.linspace(0.0, 2.0, 21)
    a_grid = np.logspace(-4, -0.5, 22)
    lip, step_noise = {}, {}
    for d in dims:
        step = 0.01
        x0 = np.zeros((seeds, d))
        x0[:, 0] = -0.6
        res = []
        for m in m_grid:
            oracle = lambda y, m=m: -g0 * y[..., 0] + m * y[..., 0]
            x_bar = run_sgd(oracle, x0, gamma, step, n_iter, rng)
            res.append(float((g0 * (RHO - x_bar[:, 0])).mean()))
        lip[d] = res
        res = []
        for a in a_grid:
            oracle = lambda y, a=a: -g0 * y[..., 0] + a * np.sign(y[..., 0])
            x_bar = run_sgd(oracle, x0, gamma, step, n_iter, rng)
            res.append(float((g0 * (RHO - x_bar[:, 0])).mean()))
        step_noise[d] = res
        print(f"d={d}: done")
    return eps, g0, gamma, dims, m_grid, a_grid, lip, step_noise


def main():
    rng = np.random.default_rng(0)
    rows = theorem1(rng)
    eps, g0, gamma, dims, m_grid, a_grid, lip, step_noise = thresholds(rng)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    labels = sorted({(r["instance"], r["noise"]) for r in rows})
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for (inst, noise), col in zip(labels, colors):
        sel = [r for r in rows if r["instance"] == inst and r["noise"] == noise]
        ax.plot([r["d"] for r in sel], [r["max_err_over_eps"] for r in sel], "o-", color=col, label=f"{inst}; {noise}")
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.text(5, 1.05, "граница теоремы: ошибка = ε", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("размерность $d$")
    ax.set_ylabel(r"худшая по запускам ошибка / $\varepsilon$")
    ax.set_title("(a) Теорема 1 с её константами", fontsize=10)
    ax.legend(fontsize=6.5, loc="center right")

    ax = axes[1]
    shades = ["#86b6ef", "#2a78d6", "#184f95"]
    for d, col in zip(dims, shades):
        ax.plot(m_grid / g0, np.array(lip[d]) / eps, "o-", ms=3, color=col, label=f"$d={d}$")
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.axvline(1.0, color="#6b6a64", ls=":", lw=1)
    ax.set_xlabel(r"наклон шума $M/g_0$")
    ax.set_ylabel(r"ошибка / $\varepsilon$")
    ax.set_title("(b) Липшицев шум: порог не зависит от $d$", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    for d, col in zip(dims, shades):
        ax.semilogx(a_grid * np.sqrt(d), np.array(step_noise[d]) / eps, "o-", ms=3, color=col, label=f"$d={d}$")
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xlabel(r"амплитуда ступеньки $\times\sqrt{d}$")
    ax.set_ylabel(r"ошибка / $\varepsilon$")
    ax.set_title(r"(c) Ограниченный шум: порог $\propto1/\sqrt{d}$", fontsize=10)
    ax.legend(fontsize=8)

    save(fig, "e4_theorems", {"theorem1": rows, "eps_thresholds": eps, "g0": g0, "gamma": gamma, "dims": dims,
                              "m_over_g0": (m_grid / g0).tolist(), "lipschitz_err": {str(k): v for k, v in lip.items()},
                              "step_amp": a_grid.tolist(), "step_err": {str(k): v for k, v in step_noise.items()}})


if __name__ == "__main__":
    main()
