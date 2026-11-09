"""E3. Real finite-precision noise: f evaluated in fp16 / bf16 / fp32 / fp64.

f(x) = 1 + ||x - c||_1 / sqrt(d) on the ball of radius 2 (M_f = 1, f* = 1); the offset 1
makes the rounding error of order ulp(1), as for a loss of order one.
bf16 is emulated by rounding the fp32 result to 8 mantissa bits (round to nearest even).

The adversarial bounded-noise model predicts an accuracy floor
  M_f gamma + D sqrt(d) Delta0 / (2 gamma),  Delta0 = ulp(f) / 2,
so small gamma should fail. We measure how far real rounding is from this worst case, and
compare with unbiased stochastic rounding (the random-last-bit trick of arXiv:1411.4218, Sec. 4).
"""

import numpy as np
from common import plt, save

from lipzo.methods import zo_sgd

D_DIM, RADIUS, N_ITER, SEEDS = 10, 2.0, 20_000, 6
GAMMAS = np.logspace(-5, -0.5, 10)
PRECISIONS = {"fp64": 2.0**-52, "fp32": 2.0**-23, "bf16": 2.0**-7, "fp16": 2.0**-10,
              "fp16+dither": 2.0**-10}


def to_bf16(a):
    bits = np.asarray(a, dtype=np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def stochastic_round(v, mantissa_bits, rng):
    """Round v to the floating-point grid with the given mantissa, up or down at random (unbiased)."""
    q = 2.0 ** (np.floor(np.log2(np.abs(v))) - mantissa_bits)
    return (np.floor(v / q + rng.random(v.shape))) * q


def make_oracle(precision, c, rng=None):
    scale = 1.0 / np.sqrt(c.shape[-1])

    def f(y):
        if precision == "fp64":
            return 1.0 + np.abs(y - c).sum(-1) * scale
        if precision == "fp16+dither":
            return stochastic_round(1.0 + np.abs(y - c).sum(-1) * scale, 10, rng)
        if precision == "fp32":
            return (np.float32(1) + np.abs(y.astype(np.float32) - c.astype(np.float32)).sum(-1) * np.float32(scale)).astype(float)
        if precision == "fp16":
            y16, c16 = y.astype(np.float16), c.astype(np.float16)
            return (np.float16(1) + np.abs(y16 - c16).sum(-1, dtype=np.float16) * np.float16(scale)).astype(float)
        val = np.float32(1) + np.abs(to_bf16(y) - to_bf16(c)).sum(-1) * np.float32(scale)
        return to_bf16(val).astype(float)

    return f


def main():
    rng = np.random.default_rng(0)
    c = rng.standard_normal(D_DIM)
    c *= 0.8 / np.linalg.norm(c)
    exact = lambda y: 1.0 + np.abs(y - c).sum(-1) / np.sqrt(D_DIM)
    step = RADIUS / (np.sqrt(2 * D_DIM) * np.sqrt(N_ITER))
    gam = np.repeat(GAMMAS, SEEDS)[:, None]
    errors = {}
    for name in PRECISIONS:
        x0 = np.zeros((gam.shape[0], D_DIM))
        x_bar = zo_sgd(make_oracle(name, c, rng), x0, gam, step, N_ITER, RADIUS, rng)
        errors[name] = (exact(x_bar) - 1.0).reshape(len(GAMMAS), SEEDS).mean(axis=1)
        print(name, np.round(errors[name], 4))

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = dict(zip(PRECISIONS, ["black", "tab:blue", "tab:orange", "tab:red", "tab:green"]))
    for name, eps_mach in PRECISIONS.items():
        ax.loglog(GAMMAS, errors[name], "o-", color=colors[name], ms=4, label=f"{name}: измерено")
        if name in ("fp32", "bf16", "fp16"):
            delta0 = eps_mach  # ulp(f)/2 for f in [1, 2)
            bound = GAMMAS + 2 * RADIUS * np.sqrt(D_DIM) * delta0 / (2 * GAMMAS)
            ax.loglog(GAMMAS, bound, "--", color=colors[name], lw=0.9, label=f"{name}: худший случай")
    ax.set_ylim(1e-5, 1e2)
    ax.set_xlabel(r"радиус сглаживания $\gamma$")
    ax.set_ylabel(r"$f(\bar x_N)-f^*$")
    ax.set_title(rf"Округление в разной точности, $d={D_DIM}$, $N={N_ITER}$")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "e3_rounding", {"gammas": GAMMAS.tolist(), "errors": {k: v.tolist() for k, v in errors.items()},
                             "d": D_DIM, "n_iter": N_ITER, "seeds": SEEDS})


if __name__ == "__main__":
    main()
