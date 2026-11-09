"""Методичка, §4: градиентный спуск, тяжёлый шарик и метод Нестерова на простых функциях.

Запуск: python3 tutorial/ch4_acceleration.py  -> картинки в tutorial/figures/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
COLOR = {"gd": "#2a78d6", "heavy": "#eb6834", "nesterov": "#1baf7a"}
NAME = {"gd": "градиентный спуск", "heavy": "тяжёлый шарик", "nesterov": "Нестеров"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})


def iterate(method, grad, x0, L, mu=0.0, n=100, bias=0.0, step=None):
    """Траектория метода; mu > 0 включает варианты для сильно выпуклых функций.

    step задаёт шаг градиентного спуска (по умолчанию 1/L)."""
    x_prev = x = np.array(x0, float)
    path = [x.copy()]
    for k in range(1, n + 1):
        if method == "gd":
            x_new = x - (step or 1 / L) * (grad(x) + bias)
        elif method == "nesterov":
            beta = (k - 1) / (k + 2) if mu == 0 else (np.sqrt(L / mu) - 1) / (np.sqrt(L / mu) + 1)
            y = x + beta * (x - x_prev)
            x_new = y - grad(y) / L
        else:
            h = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
            b = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
            x_new = x - h * grad(x) + b * (x - x_prev)
        x_prev, x = x, x_new
        path.append(x.copy())
    return np.array(path)


def ravine(kappa):
    a = np.array([1.0, kappa])
    return (lambda x: 0.5 * (a * x * x).sum(-1)), (lambda x: a * x)


def fig_trajectories():
    kappa = 20
    f, g = ravine(kappa)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    xs, ys = np.meshgrid(np.linspace(-0.2, 1.1, 300), np.linspace(-0.35, 0.35, 200))
    ax.contour(xs, ys, 0.5 * (xs**2 + kappa * ys**2), levels=12, colors="#b8b8b0", linewidths=0.6)
    # спуск с лучшим постоянным шагом 2/(L+mu): по крутому направлению он перескакивает дно
    p = iterate("gd", g, [1.0, 0.3], L=kappa, n=15, step=2 / (kappa + 1))
    ax.plot(p[:, 0], p[:, 1], "-o", color=COLOR["gd"], ms=3, lw=1.5, label=r"спуск, шаг $2/(L+\mu)$, 15 шагов")
    p = iterate("nesterov", g, [1.0, 0.3], L=kappa, mu=1.0, n=15)
    ax.plot(p[:, 0], p[:, 1], "-o", color=COLOR["nesterov"], ms=3, lw=1.5, label="Нестеров, 15 шагов")
    ax.plot(0, 0, "*", color="black", ms=10)
    ax.set_title(r"Овраг $f=\frac{1}{2}(x_1^2+20x_2^2)$: траектории за 15 шагов")
    ax.set_xlabel("$x_1$ (пологое направление)")
    ax.set_ylabel("$x_2$ (крутое)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG / "ch4_trajectories.png")
    plt.close(fig)


def fig_error_curves():
    kappa = 100
    f, g = ravine(kappa)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ["gd", "heavy", "nesterov"]:
        p = iterate(m, g, [1.0, 1.0], L=kappa, mu=1.0, n=700)
        err = f(p)
        ax.semilogy(err, color=COLOR[m], label=NAME[m])
    ax.set_ylim(1e-12, 1e3)
    ax.set_xlabel("номер шага $k$")
    ax.set_ylabel(r"$f(x_k)-f^*$ (логарифмическая шкала)")
    ax.set_title(r"Овраг, $\kappa=100$: линейная сходимость = прямая на лог-шкале")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ch4_error_curves.png")
    plt.close(fig)


def steps_to(method, kappa, tol=1e-6):
    f, g = ravine(kappa)
    p = iterate(method, g, [1.0, 1.0], L=kappa, mu=1.0, n=200_000)
    return int(np.argmax(f(p) <= tol))


def fig_kappa_scaling():
    kappas = np.array([3, 10, 30, 100, 300, 1000, 3000])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ["gd", "heavy", "nesterov"]:
        ax.loglog(kappas, [steps_to(m, k) for k in kappas], "o-", color=COLOR[m], ms=5, label=NAME[m])
    ax.loglog(kappas, 10 * kappas, ":", color="#6b6a64", lw=1)
    ax.loglog(kappas, 12 * np.sqrt(kappas), "--", color="#6b6a64", lw=1)
    ax.text(kappas[-1], 10 * kappas[-1], r"  $\propto\kappa$", va="center", fontsize=9)
    ax.text(kappas[-1], 12 * np.sqrt(kappas[-1]), r"  $\propto\sqrt{\kappa}$", va="center", fontsize=9)
    ax.set_xlabel(r"число обусловленности $\kappa=L/\mu$")
    ax.set_ylabel(r"шагов до $f-f^*\leq 10^{-6}$")
    ax.set_title(r"Сложность на овраге: спуск $\sim\kappa$, ускоренные $\sim\sqrt{\kappa}$")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "ch4_kappa_scaling.png")
    plt.close(fig)


def fig_guarantee_vs_reality():
    d = 200
    lam = 1 / np.arange(1, d + 1) ** 2
    f = lambda x: 0.5 * (lam * x * x).sum(-1)
    g = lambda x: lam * x
    x0, R2, n = np.ones(d), float(d), 3000
    k = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ["gd", "nesterov"]:
        ax.loglog(k, f(iterate(m, g, x0, L=1.0, n=n))[1:], color=COLOR[m], label=f"{NAME[m]}: на деле")
    ax.loglog(k, R2 / (2 * k), ":", color=COLOR["gd"], lw=1.5, label=r"гарантия спуска $LR^2/(2N)$")
    ax.loglog(k, 2 * R2 / (k + 1) ** 2, ":", color=COLOR["nesterov"], lw=1.5, label=r"гарантия Нестерова $2LR^2/(N+1)^2$")
    ax.set_xlabel("номер шага $N$")
    ax.set_ylabel(r"$f(x_N)-f^*$")
    ax.set_title(r"Гарантия (пунктир) и реальная ошибка (сплошные)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "ch4_guarantee.png")
    plt.close(fig)


def fig_bias_floor():
    f = lambda x: 0.5 * (x * x).sum(-1)
    g = lambda x: x
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for B, shade in zip([1e-1, 1e-2, 1e-3], ["#184f95", "#2a78d6", "#86b6ef"]):
        p = iterate("gd", g, [1.0], L=2.0, n=60, bias=B)
        ax.semilogy(f(p), color=shade, label=f"смещение $B={B:g}$")
        ax.axhline(B**2 / 2, color=shade, ls=":", lw=1)
    ax.semilogy(f(iterate("gd", g, [1.0], L=2.0, n=60)), color="#6b6a64", lw=1.2, label="без смещения")
    ax.set_ylim(1e-12, 1)
    ax.set_xlabel("номер шага $k$")
    ax.set_ylabel(r"$f(x_k)-f^*$")
    ax.set_title(r"$f=\frac{1}{2}x^2$, оракул $x+B$: ошибка упирается в пол $B^2/2$ (пунктир)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ch4_bias_floor.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_trajectories()
    fig_error_curves()
    fig_kappa_scaling()
    fig_guarantee_vs_reality()
    fig_bias_floor()
    print("готово:", sorted(p.name for p in FIG.glob("ch4_*.png")))
