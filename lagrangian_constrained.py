#!/usr/bin/env python3
"""Augmented Lagrangian maximization of CR_hat subject to PSNR/SSIM constraints."""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

MetricFn = Callable[[float], float]


def lagrangian(
    e,
    lam1: float,
    lam2: float,
    cr_hat: MetricFn,
    psnr_hat: MetricFn,
    ssim_hat: MetricFn,
    psnr_min: float,
    ssim_min: float,
    rho: float = 0.0,
):
    g1 = psnr_hat(e) - psnr_min
    g2 = ssim_hat(e) - ssim_min
    L = cr_hat(e) + lam1 * g1 + lam2 * g2
    if rho > 0:
        viol1 = np.maximum(0.0, -np.asarray(g1))
        viol2 = np.maximum(0.0, -np.asarray(g2))
        L = L - 0.5 * rho * (viol1 ** 2 + viol2 ** 2)
    return L


def inner_argmax(
    lam1: float,
    lam2: float,
    cr_hat: MetricFn,
    psnr_hat: MetricFn,
    ssim_hat: MetricFn,
    psnr_min: float,
    ssim_min: float,
    e_lo: float,
    e_hi: float,
    rho: float = 0.0,
    n_grid: int = 4001,
    refine: bool = True,
) -> Tuple[float, float]:
    x_grid = np.linspace(math.log10(e_lo), math.log10(e_hi), n_grid)
    e_grid = np.power(10.0, x_grid)
    L_grid = lagrangian(
        e_grid, lam1, lam2, cr_hat, psnr_hat, ssim_hat,
        psnr_min, ssim_min, rho=rho,
    )
    i = int(np.argmax(L_grid))
    e_star, L_star = float(e_grid[i]), float(L_grid[i])
    if refine and 0 < i < n_grid - 1:
        x_lo = x_grid[max(0, i - 2)]
        x_hi = x_grid[min(n_grid - 1, i + 2)]
        xs = np.linspace(x_lo, x_hi, 401)
        es = np.power(10.0, xs)
        Ls = lagrangian(
            es, lam1, lam2, cr_hat, psnr_hat, ssim_hat,
            psnr_min, ssim_min, rho=rho,
        )
        j = int(np.argmax(Ls))
        e_star, L_star = float(es[j]), float(Ls[j])
    return e_star, L_star


def dual_ascent(
    cr_hat: MetricFn,
    psnr_hat: MetricFn,
    ssim_hat: MetricFn,
    psnr_min: float,
    ssim_min: float,
    e_lo: float,
    e_hi: float,
    method: str = "auglag",
    T: int = 80,
    tol_g: float = 1e-6,
    eta1: float = 10.0,
    eta2: float = 1.0e4,
    rho0: float = 1.0e4,
    rho_max: float = 1.0e12,
    rho_grow: float = 4.0,
    verbose: bool = True,
):
    if method not in ("auglag", "plain"):
        raise ValueError("method must be 'auglag' or 'plain'")

    lam1, lam2 = 0.0, 0.0
    rho = rho0 if method == "auglag" else 0.0
    best = None
    history = []
    prev_e = None
    stalled = 0

    for t in range(T):
        e_t, _ = inner_argmax(
            lam1, lam2, cr_hat, psnr_hat, ssim_hat,
            psnr_min, ssim_min, e_lo, e_hi, rho=rho,
        )
        psnr_t = float(psnr_hat(e_t))
        ssim_t = float(ssim_hat(e_t))
        cr_t = float(cr_hat(e_t))
        g1 = psnr_t - psnr_min
        g2 = ssim_t - ssim_min
        feasible = (g1 >= -tol_g) and (g2 >= -tol_g)

        history.append({
            "t": t, "lam1": lam1, "lam2": lam2, "rho": rho,
            "e": e_t, "cr": cr_t, "psnr": psnr_t, "ssim": ssim_t,
            "g_psnr": g1, "g_ssim": g2, "feasible": feasible,
        })
        if feasible and (best is None or cr_t > best["cr"]):
            best = dict(history[-1])

        if method == "auglag":
            lam1 = max(0.0, lam1 - rho * g1)
            lam2 = max(0.0, lam2 - rho * g2)
            if prev_e is not None and abs(math.log10(e_t) - math.log10(prev_e)) < 1e-3:
                stalled += 1
                if stalled >= 5 and rho < rho_max:
                    rho = min(rho_max, rho * rho_grow)
                    stalled = 0
            else:
                stalled = 0
        else:
            step = 1.0 / math.sqrt(1.0 + t)
            lam1 = max(0.0, lam1 - eta1 * step * g1)
            lam2 = max(0.0, lam2 - eta2 * step * g2)

        if verbose and (t < 5 or t % 10 == 0 or t == T - 1):
            print(
                "  t={:3d}  rho={:.2g}  lam=({:9.3f},{:11.3f})  e={:.4g}  "
                "CR={:9.2f}  PSNR={:7.3f} (g={:+.3f})  SSIM={:.4f} (g={:+.4f})  "
                "feas={}".format(
                    t, rho, lam1, lam2, e_t, cr_t, psnr_t, g1, ssim_t, g2, feasible,
                )
            )
        prev_e = e_t

    return best, history


def penalty_objective(
    e,
    mu: float,
    cr_hat: MetricFn,
    psnr_hat: MetricFn,
    ssim_hat: MetricFn,
    psnr_min: float,
    ssim_min: float,
):
    """Quadratic-penalty objective to MINIMIZE: -CR_hat + mu*(viol^2).

    Vectorized over `e` (scalar or ndarray). Only one-sided violations are
    penalized: viol_PSNR = max(0, psnr_min - PSNR_hat(e)), similarly for SSIM.
    """
    cr = np.asarray(cr_hat(e), dtype=float)
    v1 = np.maximum(0.0, psnr_min - np.asarray(psnr_hat(e), dtype=float))
    v2 = np.maximum(0.0, ssim_min - np.asarray(ssim_hat(e), dtype=float))
    return -cr + mu * (v1 ** 2 + v2 ** 2)


def penalty_minimize(
    cr_hat: MetricFn,
    psnr_hat: MetricFn,
    ssim_hat: MetricFn,
    psnr_min: float,
    ssim_min: float,
    e_lo: float,
    e_hi: float,
    mu0: float = 1.0,
    mu_grow: float = 10.0,
    mu_max: float = 1.0e12,
    max_rounds: int = 13,
    feas_tol: float = 1e-6,
    x_stable_tol: float = 1e-7,
    verbose: bool = True,
):
    """Quadratic-penalty method for  max CR_hat  s.t. PSNR/SSIM constraints.

    Solves  min_x  -CR_hat(e) + mu*(viol_PSNR^2 + viol_SSIM^2)  on x = log10(e)
    with scipy.optimize.minimize_scalar (bounded Brent). mu is increased
    geometrically each round; as mu grows the minimizer is pulled from the
    high-CR infeasible region back onto the constraint boundary. No QoI grid
    and no dense surrogate grid are used — only the 1-D continuous solve.

    Returns (best, history):
      - history: one record per mu-round (mu, e, cr, psnr, ssim, slacks, feas).
      - best: max-CR feasible iterate if any feasible were found, otherwise the
        least-violating iterate (the boundary limit point).
    """
    from scipy.optimize import minimize_scalar

    x_lo, x_hi = math.log10(e_lo), math.log10(e_hi)

    def obj(x: float, mu: float) -> float:
        e = 10.0 ** x
        return float(penalty_objective(
            e, mu, cr_hat, psnr_hat, ssim_hat, psnr_min, ssim_min,
        ))

    history = []
    best = None
    mu = mu0
    prev_x = None
    for k in range(max_rounds):
        res = minimize_scalar(
            obj, bounds=(x_lo, x_hi), method="bounded",
            args=(mu,), options={"xatol": 1e-10},
        )
        x = float(res.x)
        e = 10.0 ** x
        psnr = float(psnr_hat(e))
        ssim = float(ssim_hat(e))
        cr = float(cr_hat(e))
        g1 = psnr - psnr_min
        g2 = ssim - ssim_min
        v1 = max(0.0, -g1)
        v2 = max(0.0, -g2)
        feasible = (g1 >= -feas_tol) and (g2 >= -feas_tol)
        rec = {
            "k": k, "mu": mu, "e": e, "cr": cr, "psnr": psnr, "ssim": ssim,
            "g_psnr": g1, "g_ssim": g2, "viol_psnr": v1, "viol_ssim": v2,
            "feasible": feasible, "method": "lagrangian_penalty",
        }
        history.append(rec)
        if feasible and (best is None or cr > best["cr"]):
            best = dict(rec)

        if verbose and (k < 3 or k % 2 == 0 or k == max_rounds - 1):
            print(
                "  [penalty] k={:2d}  mu={:.2g}  e={:.4g}  CR={:9.2f}  "
                "PSNR={:7.3f} (g={:+.3f})  SSIM={:.4f} (g={:+.4f})  feas={}".format(
                    k, mu, e, cr, psnr, g1, ssim, g2, feasible,
                )
            )

        if prev_x is not None and abs(x - prev_x) < x_stable_tol and k >= 2:
            break
        prev_x = x
        if mu >= mu_max:
            break
        mu = min(mu_max, mu * mu_grow)

    if best is None and history:
        best = dict(min(history, key=lambda r: r["viol_psnr"] + r["viol_ssim"]))
    return best, history


def find_max_e_for(
    constraint_fn: MetricFn,
    target_min: float,
    e_lo: float,
    e_hi: float,
    n: int = 200001,
) -> Optional[float]:
    x = np.linspace(math.log10(e_lo), math.log10(e_hi), n)
    e = np.power(10.0, x)
    vals = np.asarray(constraint_fn(e))
    mask = vals >= target_min
    if not np.any(mask):
        return None
    return float(e[mask][-1])
