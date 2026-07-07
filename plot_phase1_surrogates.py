#!/usr/bin/env python3
"""
Plot Phase-1 surrogate curves using the same fit as three_phase_optimize.py.

Change defaults or run_phase1() in three_phase_optimize.py — plots follow automatically.

Example:
    cd /anvil/projects/x-cis240669/optimizer
    python plot_phase1_surrogates.py \\
      --csv /anvil/projects/x-cis240669/Hurricane/results/CLOUDf01/CLOUDf01_sz3_sweep.csv \\
      --out /anvil/projects/x-cis240669/Hurricane/results/CLOUDf01/three_phase_CLOUDf01/phase1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from surrogate_lasso import LassoSurrogate, PiecewiseLassoSurrogate, SurrogateBundle  # noqa: E402
from three_phase_optimize import (  # noqa: E402
    DEFAULT_DEGREE,
    DEFAULT_PLOT_GRID_N,
    add_model_training_args,
    run_phase1,
)


def surrogate_to_dict(s: LassoSurrogate | PiecewiseLassoSurrogate) -> dict:
    out = {
        "name": s.name,
        "transform": s.transform,
        "n_points": s.n_points,
        "alpha": s.alpha,
        "r2": s.r2,
        "rmse": s.rmse,
        "formula": s.formula_str(),
    }
    if isinstance(s, LassoSurrogate):
        out["intercept_raw"] = s.intercept_raw
        out["coefs_raw"] = [float(c) for c in s.coefs_raw]
        out["feature_names"] = list(s.feature_names)
    else:
        out["piecewise"] = True
        out["n_segments"] = s.n_segments
        out["breakpoints"] = list(s.breakpoints)
        out["bic"] = s.bic
        out["method"] = s.method
        out["segments"] = [
            {
                "e_lo": elo,
                "e_hi": ehi,
                "surrogate": surrogate_to_dict(seg),
            }
            for elo, ehi, seg in s.segments
        ]
    return out


def bundle_to_json(bundle: SurrogateBundle) -> dict:
    return {
        "source_csv": bundle.csv_path,
        "e_lo": bundle.e_lo,
        "e_hi": bundle.e_hi,
        "surrogates": {
            "psnr": surrogate_to_dict(bundle.psnr),
            "ssim": surrogate_to_dict(bundle.ssim),
            "cr": surrogate_to_dict(bundle.cr),
        },
    }


def write_fitted_points_csv(df: pd.DataFrame, bundle: SurrogateBundle, out_path: str) -> None:
    e = df["error_bound"].values.astype(float)
    rows = []
    for i, eb in enumerate(e):
        rows.append({
            "error_bound": float(eb),
            "log10_e": float(math.log10(eb)),
            "psnr_measured": float(df["psnr"].iloc[i]),
            "psnr_hat": float(bundle.psnr(eb)),
            "psnr_residual": float(df["psnr"].iloc[i] - bundle.psnr(eb)),
            "ssim_measured": float(df["ssim"].iloc[i]),
            "ssim_hat": float(bundle.ssim(eb)),
            "ssim_residual": float(df["ssim"].iloc[i] - bundle.ssim(eb)),
            "cr_measured": float(df["compression_ratio"].iloc[i]),
            "cr_hat": float(bundle.cr(eb)),
            "cr_residual": float(df["compression_ratio"].iloc[i] - bundle.cr(eb)),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _log_e_grid(e_lo: float, e_hi: float, n: int) -> np.ndarray:
    return np.linspace(math.log10(e_lo), math.log10(e_hi), n)


def _format_equation(surrogate, width: int = 92) -> str:
    """Wrap the fitted surrogate formula for display under a plot."""
    lines = []
    for raw in surrogate.formula_str().splitlines():
        lines.append(textwrap.fill(
            raw, width=width,
            subsequent_indent="        ", break_long_words=False,
        ))
    return "\n".join(lines)


def _plot_surrogate_curve(ax, surrogate, e_lo, e_hi, n_grid, *, color="r-", label="Lasso surrogate"):
    if isinstance(surrogate, PiecewiseLassoSurrogate):
        for i, (seg_lo, seg_hi, seg) in enumerate(surrogate.segments):
            xg = _log_e_grid(seg_lo, seg_hi, max(32, n_grid // max(1, surrogate.n_segments)))
            eg = np.power(10.0, xg)
            ax.plot(
                xg, seg(eg), color="C3" if i else "r", lw=2,
                label=label if i == 0 else None,
            )
        for bp in surrogate.breakpoints:
            ax.axvline(math.log10(bp), color="0.45", ls="--", lw=1, zorder=2)
    else:
        xg = _log_e_grid(e_lo, e_hi, n_grid)
        ax.plot(xg, surrogate(np.power(10.0, xg)), color, lw=2, label=label)


def plot_metric_fit(
    df, surrogate, y_col, y_label, out_path, e_lo, e_hi, n_grid, degree,
    baseline_df=None,
) -> None:
    e = df["error_bound"].values.astype(float)
    y = df[y_col].values.astype(float)
    x = np.log10(e)

    fig, ax = plt.subplots(figsize=(8, 6))
    if baseline_df is not None and len(baseline_df):
        be = baseline_df["error_bound"].values.astype(float)
        by = baseline_df[y_col].values.astype(float)
        ax.scatter(
            np.log10(be), by, s=8, c="0.6", alpha=0.35, zorder=1,
            label="measured baseline ({} pts)".format(len(baseline_df)),
        )
    ax.scatter(x, y, color="C0", edgecolor="k", s=70, zorder=3, label="training data")
    _plot_surrogate_curve(ax, surrogate, e_lo, e_hi, n_grid)
    title_extra = ""
    if isinstance(surrogate, PiecewiseLassoSurrogate):
        title_extra = ", segments={}".format(surrogate.n_segments)
    ax.set_xlabel(r"$\log_{10}(\mathrm{error\_bound})$")
    ax.set_ylabel(y_label)
    ax.set_title(
        "{} surrogate (deg={}, transform={}, alpha={:.3g}{})\n"
        "R2={:.4f}, RMSE={:.4g}, N={}".format(
            surrogate.name, degree, surrogate.transform,
            surrogate.alpha, title_extra,
            surrogate.r2, surrogate.rmse, surrogate.n_points,
        )
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.text(
        0.5, 0.015, _format_equation(surrogate),
        ha="center", va="bottom", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.7", alpha=0.9),
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_fit_transformed(
    df, surrogate, y_col, out_path, e_lo, e_hi, n_grid, degree,
    baseline_df=None,
) -> None:
    """Plot the first-line (fit-space) polynomial: log10(CR) / logit(SSIM) vs log10(e)."""
    e = df["error_bound"].values.astype(float)
    y = surrogate.to_fit_space(df[y_col].values.astype(float))
    x = np.log10(e)
    xg = _log_e_grid(e_lo, e_hi, n_grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    if baseline_df is not None and len(baseline_df):
        be = baseline_df["error_bound"].values.astype(float)
        bz = surrogate.to_fit_space(baseline_df[y_col].values.astype(float))
        ax.scatter(np.log10(be), bz, s=8, c="0.6", alpha=0.35, zorder=1,
                   label="measured baseline ({} pts)".format(len(baseline_df)))
    ax.scatter(x, y, color="C0", edgecolor="k", s=70, zorder=3, label="training data")
    if isinstance(surrogate, PiecewiseLassoSurrogate):
        for i, (seg_lo, seg_hi, seg) in enumerate(surrogate.segments):
            sxg = _log_e_grid(seg_lo, seg_hi, max(32, n_grid // max(1, surrogate.n_segments)))
            seg_e = np.power(10.0, sxg)
            ax.plot(sxg, seg.predict_fit_space(seg_e), "r-", lw=2,
                    label="polynomial fit (fit space)" if i == 0 else None)
        for bp in surrogate.breakpoints:
            ax.axvline(math.log10(bp), color="0.45", ls="--", lw=1, zorder=2)
    else:
        ax.plot(xg, surrogate.predict_fit_space(np.power(10.0, xg)), "r-", lw=2,
                label="polynomial fit (fit space)")
    zlabel = surrogate.fit_space_label()
    ax.set_xlabel(r"$\log_{10}(\mathrm{error\_bound})$")
    ax.set_ylabel(zlabel)
    ax.set_title(
        "{} fit-space model (deg={}, transform={}, alpha={:.3g})\n"
        "R2={:.4f}, RMSE={:.4g}, N={}".format(
            surrogate.name, degree, surrogate.transform,
            surrogate.alpha, surrogate.r2, surrogate.rmse, surrogate.n_points,
        )
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.text(
        0.5, 0.015, textwrap.fill(surrogate.formula_str().splitlines()[0], width=92,
                                  subsequent_indent="        ", break_long_words=False),
        ha="center", va="bottom", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.7", alpha=0.9),
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(df, bundle, out_path) -> None:
    specs = [
        ("psnr", bundle.psnr, "PSNR (dB)"),
        ("ssim", bundle.ssim, "SSIM"),
        ("compression_ratio", bundle.cr, "compression ratio"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for row, (col, surr, label) in enumerate(specs):
        e = df["error_bound"].values.astype(float)
        y = df[col].values.astype(float)
        y_hat = surr(e)
        res = y - y_hat
        x = np.log10(e)

        ax = axes[row, 0]
        ax.scatter(y_hat, res, color="C0", edgecolor="k", s=55)
        ax.axhline(0, color="red", lw=1)
        ax.set_xlabel("fitted {}".format(label))
        ax.set_ylabel("residual")
        ax.set_title("{} residuals vs fitted".format(surr.name))
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        ax.scatter(x, res, color="C0", edgecolor="k", s=55)
        ax.axhline(0, color="red", lw=1)
        ax.set_xlabel(r"$\log_{10}(e)$")
        ax.set_ylabel("residual")
        ax.set_title(
            "{} vs log10(e)  (max|res|={:.4g})".format(
                surr.name, float(np.max(np.abs(res))),
            )
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 1 surrogate residual diagnostics", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview(df, bundle, out_path, n_grid, degree, baseline_df=None) -> None:
    e_lo, e_hi = bundle.e_lo, bundle.e_hi
    xg = _log_e_grid(e_lo, e_hi, n_grid)
    eg = np.power(10.0, xg)
    specs = [
        ("psnr", bundle.psnr, "PSNR (dB)", "C0"),
        ("ssim", bundle.ssim, "SSIM", "C1"),
        ("compression_ratio", bundle.cr, "CR", "C2"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, surr, ylab, color) in zip(axes, specs):
        e = df["error_bound"].values.astype(float)
        y = df[col].values.astype(float)
        if baseline_df is not None and len(baseline_df):
            be = baseline_df["error_bound"].values.astype(float)
            by = baseline_df[col].values.astype(float)
            ax.scatter(np.log10(be), by, s=6, c="0.6", alpha=0.3, zorder=1,
                       label="measured baseline")
        ax.scatter(np.log10(e), y, color=color, edgecolor="k", s=55, zorder=3,
                   label="training data")
        if isinstance(surr, PiecewiseLassoSurrogate):
            _plot_surrogate_curve(ax, surr, e_lo, e_hi, n_grid, color="k-", label="surrogate")
        else:
            ax.plot(xg, surr(eg), "k-", lw=2, label="surrogate")
        ax.set_xlabel(r"$\log_{10}(e)$")
        ax.set_ylabel(ylab)
        ax.set_title("{}\nR2={:.4f}  RMSE={:.4g}".format(surr.name, surr.r2, surr.rmse))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Phase 1 Lasso surrogates (degree={}, source: {})".format(
            degree, os.path.basename(bundle.csv_path),
        ),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_phase1_plots(
    bundle,
    df,
    out_dir: str,
    degree: int = DEFAULT_DEGREE,
    n_grid: int = DEFAULT_PLOT_GRID_N,
    baseline_df=None,
) -> str:
    """
    Write all Phase-1 plot artifacts (called from three_phase_optimize or CLI).
    If baseline_df is given, its measured points are overlaid as ground truth.
    Returns output directory path.
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "phase1_surrogates.json"), "w") as f:
        json.dump(bundle_to_json(bundle), f, indent=2)
    write_fitted_points_csv(
        df, bundle, os.path.join(out_dir, "phase1_fitted_points.csv"),
    )
    plot_metric_fit(
        df, bundle.psnr, "psnr", "PSNR (dB)",
        os.path.join(out_dir, "phase1_psnr.png"),
        bundle.e_lo, bundle.e_hi, n_grid, degree, baseline_df=baseline_df,
    )
    plot_metric_fit(
        df, bundle.ssim, "ssim", "SSIM",
        os.path.join(out_dir, "phase1_ssim.png"),
        bundle.e_lo, bundle.e_hi, n_grid, degree, baseline_df=baseline_df,
    )
    plot_metric_fit(
        df, bundle.cr, "compression_ratio", "compression ratio",
        os.path.join(out_dir, "phase1_cr.png"),
        bundle.e_lo, bundle.e_hi, n_grid, degree, baseline_df=baseline_df,
    )
    # First-line (fit-space) plots. PSNR uses transform="none", so its
    # fit-space curve equals the real-metric curve.
    for surr, col in (
        (bundle.psnr, "psnr"),
        (bundle.ssim, "ssim"),
        (bundle.cr, "compression_ratio"),
    ):
        plot_metric_fit_transformed(
            df, surr, col,
            os.path.join(out_dir, "phase1_{}_fitspace.png".format(surr.name.lower())),
            bundle.e_lo, bundle.e_hi, n_grid, degree, baseline_df=baseline_df,
        )
    plot_residuals(df, bundle, os.path.join(out_dir, "phase1_residuals.png"))
    plot_overview(
        df, bundle, os.path.join(out_dir, "phase1_surrogates_overview.png"),
        n_grid, degree, baseline_df=baseline_df,
    )
    return out_dir


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_model_training_args(ap)
    ap.add_argument("--csv", required=True, help="Phase-1 training sweep CSV")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-grid", type=int, default=DEFAULT_PLOT_GRID_N,
                    help="points for smooth curve plots")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    bundle, df = run_phase1(args.csv, degree=args.degree, report_dir=out_dir)
    generate_phase1_plots(bundle, df, out_dir, degree=args.degree, n_grid=args.n_grid)

    print("Phase 1 (via three_phase_optimize.run_phase1)")
    print("  CSV    : {}".format(args.csv))
    print("  degree : {}".format(args.degree))
    print("  e range: [{:g}, {:g}]  N={}".format(bundle.e_lo, bundle.e_hi, len(df)))
    for s in (bundle.psnr, bundle.ssim, bundle.cr):
        print("  {:4s}  R2={:.4f}  RMSE={:.4g}  alpha={:.4g}".format(
            s.name, s.r2, s.rmse, s.alpha))
    print("\nWrote to {}:".format(out_dir))
    for name in sorted(os.listdir(out_dir)):
        print("  -", name)


if __name__ == "__main__":
    main()
