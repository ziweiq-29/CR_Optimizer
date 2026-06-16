#!/usr/bin/env python3
"""
Plot Phase-2 penalty-method history using the same solver as three_phase_optimize.py.

Change defaults or run_phase1()/run_phase2() in three_phase_optimize.py — plots follow.

Example:
    cd /anvil/projects/x-cis240669/optimizer
    python plot_phase2_lagrangian.py \\
      --csv /anvil/projects/x-cis240669/Hurricane/results/CLOUDf01_sz3_sweep.csv \\
      --psnr-min 80 --ssim-min 0.9 \\
      --out /anvil/projects/x-cis240669/Hurricane/results/three_phase_CLOUDf01/phase2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from lagrangian_constrained import penalty_objective  # noqa: E402
from three_phase_optimize import (  # noqa: E402
    DEFAULT_PLOT_GRID_N,
    add_phase2_args,
    plot_overview,
    run_phase1,
    run_phase2,
)


def _e_grid(e_lo: float, e_hi: float, n: int) -> np.ndarray:
    return np.power(10.0, np.linspace(math.log10(e_lo), math.log10(e_hi), n))


def history_to_dataframe(history: list[dict]) -> pd.DataFrame:
    rows = []
    for h in history:
        rows.append({
            "k": h["k"],
            "mu": h["mu"],
            "error_bound": h["e"],
            "log10_e": math.log10(h["e"]),
            "cr_hat": h["cr"],
            "psnr_hat": h["psnr"],
            "ssim_hat": h["ssim"],
            "g_psnr": h["g_psnr"],
            "g_ssim": h["g_ssim"],
            "viol_psnr": h.get("viol_psnr", max(0.0, -h["g_psnr"])),
            "viol_ssim": h.get("viol_ssim", max(0.0, -h["g_ssim"])),
            "feasible": bool(h["feasible"]),
        })
    return pd.DataFrame(rows)


def write_summary(p2_result: dict, psnr_min: float, ssim_min: float, out_path: str) -> None:
    history = p2_result["history"]
    phase2 = p2_result["phase2"]
    lines = [
        "Phase 2: quadratic-penalty method (three_phase_optimize.run_phase2)",
        "=" * 60,
        "min -CR_hat(e) + mu*(viol_PSNR^2 + viol_SSIM^2)   on x = log10(e)",
        "constraints:  PSNR_hat >= {:.4g},  SSIM_hat >= {:.4g}".format(
            psnr_min, ssim_min),
        "e range: [{:g}, {:g}]".format(p2_result["e_lo"], p2_result["e_hi"]),
        "mu-rounds: {}".format(len(history)),
        "",
    ]
    if phase2:
        lines += [
            "Selected Phase-2 model (method={}):".format(phase2.get("method", "?")),
            "  mu     = {:.6g}".format(phase2.get("mu", float("nan"))),
            "  e      = {:.6g}".format(phase2["e"]),
            "  cr     = {:.6f}".format(phase2["cr"]),
            "  psnr   = {:.6f}  (g={:+.6g})".format(
                phase2.get("psnr", float("nan")), phase2.get("g_psnr", float("nan"))),
            "  ssim   = {:.6f}  (g={:+.6g})".format(
                phase2.get("ssim", float("nan")), phase2.get("g_ssim", float("nan"))),
            "  feasible = {}".format(phase2.get("feasible", "?")),
        ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _select_L_snapshots(history: list[dict], max_curves: int = 8) -> list[int]:
    if not history:
        return []
    n = len(history)
    if n <= max_curves:
        return list(range(n))
    idx = {0, n - 1}
    step = max(1, (n - 1) // (max_curves - 2))
    for i in range(0, n, step):
        idx.add(i)
    return sorted(idx)[:max_curves]


def plot_history_traces(history, psnr_min, ssim_min, out_path) -> None:
    df = history_to_dataframe(history)
    k = df["k"].values
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    ax = axes[0, 0]
    ax.plot(k, df["mu"], "k.-", ms=5)
    ax.set_yscale("log")
    ax.set_xlabel("penalty round k")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("Penalty weight (geometric growth)")
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[0, 1]
    ax.plot(k, df["error_bound"], "o-", ms=4, color="C2")
    ax.set_yscale("log")
    ax.set_xlabel("penalty round k")
    ax.set_ylabel("e*(mu)")
    ax.set_title("Penalty minimizer e*")
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[0, 2]
    ax.plot(k, df["cr_hat"], "o-", ms=4, color="C0")
    ax.set_xlabel("penalty round k")
    ax.set_ylabel("CR_hat(e*)")
    ax.set_title("Objective at minimizer")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(k, df["viol_psnr"], "o-", ms=4, label="PSNR violation")
    ax.plot(k, df["viol_ssim"], "s-", ms=4, label="SSIM violation")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel("penalty round k")
    ax.set_ylabel("violation  max(0, min - val)")
    ax.set_title("Constraint violations -> 0")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.plot(k, df["g_psnr"], "o-", ms=4, label="PSNR slack")
    ax.plot(k, df["g_ssim"], "s-", ms=4, label="SSIM slack")
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("penalty round k")
    ax.set_ylabel("constraint slack  (val - min)")
    ax.set_title("Constraint slacks")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    colors = np.where(df["feasible"], "C2", "C3")
    ax.scatter(k, df["cr_hat"], c=colors, s=45, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("penalty round k")
    ax.set_ylabel("CR_hat(e*)")
    ax.set_title("Feasibility (green=feasible)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Phase 2 penalty-method history  (PSNR>={:.2g}, SSIM>={:.3g})".format(
            psnr_min, ssim_min),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_penalty_objective(
    bundle, history, psnr_min, ssim_min, e_lo, e_hi, n_grid, out_path,
) -> None:
    es = _e_grid(e_lo, e_hi, n_grid)
    log_e = np.log10(es)
    snap = _select_L_snapshots(history)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(snap)))
    for j, ki in enumerate(snap):
        h = history[ki]
        P = penalty_objective(
            es, h["mu"], bundle.cr, bundle.psnr, bundle.ssim,
            psnr_min, ssim_min,
        )
        ax.plot(log_e, P, color=cmap[j], lw=1.8,
                label="k={}  mu={:.2g}".format(ki, h["mu"]))
        ax.scatter(
            [math.log10(h["e"])],
            [float(penalty_objective(
                h["e"], h["mu"], bundle.cr, bundle.psnr, bundle.ssim,
                psnr_min, ssim_min,
            ))],
            color=cmap[j], s=55, zorder=5, edgecolors="k", linewidths=0.4,
        )
    ax.set_xlabel(r"$\log_{10}(e)$")
    ax.set_ylabel(r"$P_\mu(e) = -\mathrm{CR}+\mu\,\mathrm{viol}^2$")
    ax.set_title("Penalty objective (minimized) as mu grows")
    ax.legend(loc="best", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_primal_trajectory(
    bundle, history, e_lo, e_hi, n_grid, phase2, out_path,
) -> None:
    es = _e_grid(e_lo, e_hi, n_grid)
    log_e = np.log10(es)
    cr_bg = bundle.cr(es)
    df = history_to_dataframe(history)
    x, y, k = df["log10_e"].values, df["cr_hat"].values, df["k"].values

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(log_e, cr_bg, color="0.85", lw=2, label="background CR_hat(e)")
    sc = ax.scatter(x, y, c=k, cmap="plasma", s=55, edgecolors="k", linewidths=0.4)
    ax.plot(x, y, "k-", lw=0.8, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="penalty round k")

    if phase2:
        ax.scatter(
            [math.log10(phase2["e"])], [phase2["cr"]],
            c="red", marker="*", s=220, zorder=6, edgecolors="k", linewidths=0.6,
            label="Phase-2 e*={:.3g}  CR={:.2f}".format(phase2["e"], phase2["cr"]),
        )
    ax.set_xlabel(r"$\log_{10}(e)$")
    ax.set_ylabel("CR_hat")
    ax.set_title("Penalty minimizer path on CR_hat surrogate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_phase2_plots(
    bundle,
    df,
    p2_result: dict,
    psnr_min: float,
    ssim_min: float,
    out_dir: str,
    *,
    n_grid: int = DEFAULT_PLOT_GRID_N,
    csv_path: str = "",
    degree: int = 0,
    iters: int = 0,
    grid_opt_n: int = 0,
    baseline_df=None,
) -> str:
    """
    Write all Phase-2 Lagrangian history plots (called from three_phase_optimize or CLI).
    Returns output directory path.
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    history = p2_result["history"]
    phase2 = p2_result["phase2"]
    if not history or phase2 is None:
        raise ValueError("p2_result must contain non-empty history and a phase2")

    write_summary(p2_result, psnr_min, ssim_min,
                  os.path.join(out_dir, "phase2_summary.txt"))
    history_to_dataframe(history).to_csv(
        os.path.join(out_dir, "phase2_penalty_history.csv"), index=False,
    )
    with open(os.path.join(out_dir, "phase2_penalty_history.json"), "w") as f:
        json.dump({
            "source_csv": csv_path,
            "degree": degree,
            "iters": iters,
            "grid_opt_n": grid_opt_n,
            "constraints": {"psnr_min": psnr_min, "ssim_min": ssim_min},
            **{k: p2_result.get(k) for k in (
                "phase2", "penalty_opt", "e_lo", "e_hi",
            )},
            "history": history,
        }, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else x)

    plot_history_traces(
        history, psnr_min, ssim_min,
        os.path.join(out_dir, "phase2_history_traces.png"),
    )
    plot_penalty_objective(
        bundle, history, psnr_min, ssim_min,
        p2_result["e_lo"], p2_result["e_hi"], n_grid,
        os.path.join(out_dir, "phase2_penalty_objective.png"),
    )
    plot_primal_trajectory(
        bundle, history, p2_result["e_lo"], p2_result["e_hi"],
        n_grid, phase2, os.path.join(out_dir, "phase2_primal_trajectory.png"),
    )
    from three_phase_optimize import grid_truth_optimum
    grid_best = grid_truth_optimum(df, psnr_min, ssim_min)
    plot_overview(
        bundle, psnr_min, ssim_min, phase2, grid_best,
        os.path.join(out_dir, "phase2_feasible_cr.png"),
        p2_result["e_lo"], p2_result["e_hi"],
        baseline_df=baseline_df,
    )
    return out_dir


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_phase2_args(ap)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-grid", type=int, default=DEFAULT_PLOT_GRID_N,
                    help="grid points for penalty-objective / trajectory plots")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    bundle, df = run_phase1(args.csv, degree=args.degree)
    p2_result = run_phase2(
        bundle, args.psnr_min, args.ssim_min,
        iters=args.iters, grid_n=args.grid_opt_n, verbose=not args.quiet,
    )
    history = p2_result["history"]
    phase2 = p2_result["phase2"]
    if not history:
        print("ERROR: empty penalty-method history.", file=sys.stderr)
        sys.exit(1)
    if phase2 is None:
        print("ERROR: Phase 2 found no model.", file=sys.stderr)
        sys.exit(1)

    generate_phase2_plots(
        bundle, df, p2_result,
        args.psnr_min, args.ssim_min, out_dir,
        n_grid=args.n_grid,
        csv_path=args.csv,
        degree=args.degree,
        iters=args.iters,
        grid_opt_n=args.grid_opt_n,
    )

    print("Phase 2 (via three_phase_optimize.run_phase1 + run_phase2)")
    print("  method = {}".format(phase2.get("method")))
    print("  e*     = {:.6g}  CR_hat = {:.4f}".format(phase2["e"], phase2["cr"]))
    print("\nWrote to {}:".format(out_dir))
    for name in sorted(os.listdir(out_dir)):
        print("  -", name)


if __name__ == "__main__":
    main()
