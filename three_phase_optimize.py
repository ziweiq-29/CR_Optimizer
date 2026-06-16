#!/usr/bin/env python3
"""
Three-phase MODEL for constrained CR optimization.

All three phases are the model (surrogate-only, no pressio at inference):

    maximize    compression_ratio(e)
    subject to  PSNR(e) >= PSNR_min,  SSIM(e) >= SSIM_min

Phase 1 — Fit Lasso surrogates from a pressio sweep CSV (training data).
Phase 2 — Augmented Lagrangian on surrogates → coarse e*.
Phase 3 — Wise local search near e* (default: 20-pt training-curve interp).
    Optional --phase3-pressio: run N real pressio measurements in the window.

With --input, runs a 10k-point pressio baseline sweep (oracle ground truth).
--input alone does NOT enable Phase-3 pressio; use --phase3-pressio for that.

Optional --verify-pressio: one extra pressio at model e* (debug only).

By default also writes Phase-1/2 diagnostic plots under {out}/phase1 and {out}/phase2
(same parameters as plot_phase1_surrogates.py / plot_phase2_lagrangian.py).
Use --skip-phase-plots to disable.

Example (same command as before; baseline auto when --input is set):

    python three_phase_optimize.py \\
      --csv .../CLOUDf01_sz3_sweep.csv \\
      --input .../CLOUDf01.bin \\
      --dims 500 500 100 --psnr-min 80 --ssim-min 0.9 \\
      --out .../three_phase_CLOUDf01
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running as script from optimizer/ or repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from lagrangian_constrained import (  # noqa: E402
    penalty_minimize,
)
from pressio_baseline_sweep import (  # noqa: E402
    DEFAULT_BASELINE_N,
    run_baseline_sweep,
    run_pressio_on_ebs,
)
from pressio_run import DEFAULT_PRESSIO, run_pressio  # noqa: E402
from surrogate_lasso import fit_surrogates_from_csv, write_surrogate_report  # noqa: E402
from validation_metrics import (  # noqa: E402
    format_mape_block,
    measured_at_e,
    mape_model_vs_oracle,
    pressio_oracle_grid,
)

# Shared defaults — plot_phase1_surrogates.py / plot_phase2_lagrangian.py import these.
DEFAULT_PSNR_MIN = 80.0
DEFAULT_SSIM_MIN = 0.9
DEFAULT_DEGREE = 4
DEFAULT_LAGRANGIAN_ITERS = 80
DEFAULT_SURROGATE_GRID_N = 40001
DEFAULT_PLOT_GRID_N = 800


def surrogate_feasible_maximum(
    cr_hat,
    psnr_hat,
    ssim_hat,
    psnr_min: float,
    ssim_min: float,
    e_lo: float,
    e_hi: float,
    n: int = DEFAULT_SURROGATE_GRID_N,
):
    """Max CR_hat on a log grid over the surrogate-feasible set."""
    x = np.linspace(math.log10(e_lo), math.log10(e_hi), n)
    es = np.power(10.0, x)
    feas = (psnr_hat(es) >= psnr_min) & (ssim_hat(es) >= ssim_min)
    if not np.any(feas):
        return None
    cr_v = np.asarray(cr_hat(es), dtype=float)
    cr_v[~feas] = -np.inf
    i = int(np.argmax(cr_v))
    return {
        "e": float(es[i]),
        "cr": float(cr_v[i]),
        "psnr": float(psnr_hat(es[i])),
        "ssim": float(ssim_hat(es[i])),
        "method": "surrogate_grid",
    }


def pick_phase2_e_star(best_lagrangian, grid_opt, e_direct, cr_hat, psnr_hat, ssim_hat):
    """Prefer surrogate grid max; fall back to Lagrangian / monotone cap."""
    if grid_opt is not None:
        return grid_opt
    candidates = []
    if best_lagrangian is not None and best_lagrangian.get("feasible", True):
        candidates.append({**best_lagrangian, "method": "auglag"})
    if e_direct is not None:
        candidates.append({
            "e": e_direct,
            "cr": float(cr_hat(e_direct)),
            "psnr": float(psnr_hat(e_direct)),
            "ssim": float(ssim_hat(e_direct)),
            "method": "monotone_cap",
        })
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["cr"])


def run_phase1(
    csv_path: str,
    degree: int = DEFAULT_DEGREE,
    report_dir: str | None = None,
    df=None,
):
    """Phase 1: fit Lasso surrogates."""
    from surrogate_lasso import fit_surrogates_from_dataframe, load_sweep_csv

    if df is None:
        df = load_sweep_csv(csv_path)
    bundle = fit_surrogates_from_dataframe(
        df, degree=degree, source_label=csv_path,
    )
    if report_dir is not None:
        write_surrogate_report(bundle, report_dir)
    return bundle, df


def run_phase2(
    bundle,
    psnr_min: float = DEFAULT_PSNR_MIN,
    ssim_min: float = DEFAULT_SSIM_MIN,
    iters: int = DEFAULT_LAGRANGIAN_ITERS,
    grid_n: int = DEFAULT_SURROGATE_GRID_N,
    verbose: bool = False,
):
    """Phase 2: quadratic-penalty solve (method = "lagrangian_penalty").

    Minimizes  -CR_hat(e) + mu*(viol_PSNR^2 + viol_SSIM^2)  on x = log10(e)
    via scipy.optimize.minimize_scalar, increasing mu until the minimizer
    settles on the constraint boundary. No QoI grid and no dense surrogate
    grid are used — purely the 1-D continuous solve.

    `iters`/`grid_n` are accepted for call-signature compatibility but unused.
    """
    e_lo, e_hi = bundle.e_lo, bundle.e_hi
    penalty_opt, history = penalty_minimize(
        bundle.cr, bundle.psnr, bundle.ssim,
        psnr_min, ssim_min, e_lo, e_hi, verbose=verbose,
    )
    phase2 = (
        {**penalty_opt, "method": "lagrangian_penalty"}
        if penalty_opt is not None
        else None
    )
    return {
        "phase2": phase2,
        "penalty_opt": penalty_opt,
        "grid_opt": None,
        "e_direct": None,
        "history": history,
        "e_lo": e_lo,
        "e_hi": e_hi,
    }


def log_phase1_fit(bundle) -> None:
    for s in (bundle.psnr, bundle.ssim, bundle.cr):
        print("  {}  R2={:.4f}  RMSE={:.4g}  alpha={:.4g}".format(
            s.name, s.r2, s.rmse, s.alpha))


def log_phase2_result(p2: dict, bundle, e_lo: float, e_hi: float):
    """Print Phase-2 summary; return phase2 dict or exit if infeasible."""
    phase2 = p2["phase2"]
    history = p2.get("history") or []
    if phase2 is None:
        print("  [ERROR] Phase 2 penalty solve found no optimum in "
              "[{:g}, {:g}].".format(e_lo, e_hi))
        sys.exit(1)
    print("  Phase-2 optimum (penalty, method={}, rounds={}):".format(
        phase2.get("method", "?"), len(history)))
    print("    e*={:.6g}  CR_hat={:.4f}  PSNR_hat={:.3f}  SSIM_hat={:.5f}".format(
        phase2["e"], phase2["cr"],
        phase2.get("psnr", float("nan")),
        phase2.get("ssim", float("nan")),
    ))
    if not phase2.get("feasible", True):
        print("    [WARN] penalty optimum marginally infeasible "
              "(g_psnr={:+.4g}, g_ssim={:+.4g}); Phase 3 will refine.".format(
                  phase2.get("g_psnr", float("nan")),
                  phase2.get("g_ssim", float("nan"))))
    print()
    return phase2


def add_model_training_args(ap: argparse.ArgumentParser) -> None:
    """CLI flags shared with plot_phase1 / plot_phase2 scripts."""
    ap.add_argument("--csv", required=True, help="Phase-1 training sweep CSV")
    ap.add_argument("--psnr-min", type=float, default=DEFAULT_PSNR_MIN)
    ap.add_argument("--ssim-min", type=float, default=DEFAULT_SSIM_MIN)
    ap.add_argument("--degree", type=int, default=DEFAULT_DEGREE,
                    help="Lasso polynomial degree in log10(e)")


def add_phase2_args(ap: argparse.ArgumentParser) -> None:
    add_model_training_args(ap)
    ap.add_argument("--iters", type=int, default=DEFAULT_LAGRANGIAN_ITERS,
                    help="augmented Lagrangian iterations")
    ap.add_argument("--grid-opt-n", type=int, default=DEFAULT_SURROGATE_GRID_N,
                    help="grid size for surrogate feasible maximum")


def logspace_candidates_near_e_star(
    e_star: float, n: int, log_span: float, e_lo: float, e_hi: float,
) -> list[float]:
    """Pure log-spaced candidates around e* (for pressio Phase-3 sweep)."""
    if n <= 1:
        return [float(np.clip(e_star, e_lo, e_hi))]
    offsets = np.linspace(-log_span, log_span, n)
    es = [float(np.clip(e_star * (10.0 ** d), e_lo, e_hi)) for d in offsets]
    return sorted(set(es))


def candidates_near_e_star(
    e_star: float, n: int, log_span: float, e_lo: float, e_hi: float, df=None,
):
    """Log-spaced points around e*, plus any sweep grid points in the window."""
    if n <= 1:
        es = [float(np.clip(e_star, e_lo, e_hi))]
    else:
        offsets = np.linspace(-log_span, log_span, n)
        es = [float(np.clip(e_star * (10.0 ** d), e_lo, e_hi)) for d in offsets]
        es.append(float(np.clip(e_star, e_lo, e_hi)))
    if df is not None:
        lo, hi = min(es), max(es)
        for e in df["error_bound"].values:
            e = float(e)
            if lo <= e <= hi:
                es.append(e)
    return sorted(set(es))


# def sample_near_e_star(e_star: float, n: int, log_span: float, e_lo: float, e_hi: float):
#     """Log-uniform offsets around e_star, clipped to [e_lo, e_hi]."""
#     if n <= 1:
#         return [float(np.clip(e_star, e_lo, e_hi))]
#     offsets = np.linspace(-log_span, log_span, n)
#     es = [float(np.clip(e_star * (10.0 ** d), e_lo, e_hi)) for d in offsets]
#     es.append(float(np.clip(e_star, e_lo, e_hi)))
#     return sorted(set(es))


def grid_truth_optimum(df, psnr_min: float, ssim_min: float):
    """Best CR on the training grid among feasible points (reference)."""
    mask = (df["psnr"] >= psnr_min) & (df["ssim"] >= ssim_min)
    if not mask.any():
        return None
    row = df.loc[mask].sort_values("compression_ratio", ascending=False).iloc[0]
    return {
        "e": float(row["error_bound"]),
        "cr": float(row["compression_ratio"]),
        "psnr": float(row["psnr"]),
        "ssim": float(row["ssim"]),
    }


def phase3_wise_search(
    bundle,
    df,
    e_candidates,
    psnr_min: float,
    ssim_min: float,
    tol_g: float = 1e-6,
):
    """
    Wise local search near Phase-2 e*.

    Surrogate guides the search window; final pick uses the Phase-1 sweep
    calibration curve (log-interp, no new pressio) for feasibility + CR.
    """
    rows = []
    best = None
    for eb in e_candidates:
        cr_h = float(bundle.cr(eb))
        psnr_h = float(bundle.psnr(eb))
        ssim_h = float(bundle.ssim(eb))
        meas = measured_at_e(df, eb)
        feas_s = (psnr_h >= psnr_min - tol_g) and (ssim_h >= ssim_min - tol_g)
        feas_m = (meas["psnr"] >= psnr_min - tol_g) and (meas["ssim"] >= ssim_min - tol_g)
        row = {
            "error_bound": eb,
            "compression_ratio_hat": cr_h,
            "psnr_hat": psnr_h,
            "ssim_hat": ssim_h,
            "compression_ratio": meas["cr"],
            "psnr": meas["psnr"],
            "ssim": meas["ssim"],
            "feasible_surrogate": feas_s,
            "feasible_measured": feas_m,
        }
        rows.append(row)
        print(
            "  e={:.6g}  CR={:.4f} (hat {:.4f})  PSNR={:.3f} (hat {:.3f})  "
            "feas_meas={}".format(
                eb, meas["cr"], cr_h, meas["psnr"], psnr_h, feas_m,
            )
        )
        if feas_m and (best is None or meas["cr"] > best["compression_ratio"]):
            best = {
                "error_bound": eb,
                "compression_ratio": meas["cr"],
                "psnr": meas["psnr"],
                "ssim": meas["ssim"],
                "compression_ratio_hat": cr_h,
                "psnr_hat": psnr_h,
                "ssim_hat": ssim_h,
                "feasible": True,
                "calibration": "train_interp",
            }
    return best, rows


def phase3_pressio_wise_search(
    bundle,
    pressio_rows: list[dict],
    psnr_min: float,
    ssim_min: float,
    tol_g: float = 1e-6,
):
    """Pick best feasible point from real pressio measurements near e*."""
    rows = []
    best = None
    for pr in pressio_rows:
        eb = float(pr["error_bound"])
        cr_h = float(bundle.cr(eb))
        psnr_h = float(bundle.psnr(eb))
        ssim_h = float(bundle.ssim(eb))
        cr = float(pr["compression_ratio"])
        psnr = float(pr["psnr"])
        ssim = float(pr["ssim"])
        feas_m = (psnr >= psnr_min - tol_g) and (ssim >= ssim_min - tol_g)
        row = {
            "error_bound": eb,
            "compression_ratio_hat": cr_h,
            "psnr_hat": psnr_h,
            "ssim_hat": ssim_h,
            "compression_ratio": cr,
            "psnr": psnr,
            "ssim": ssim,
            "feasible_surrogate": (
                psnr_h >= psnr_min - tol_g and ssim_h >= ssim_min - tol_g
            ),
            "feasible_measured": feas_m,
        }
        rows.append(row)
        print(
            "  e={:.6g}  CR={:.4f} (hat {:.4f})  PSNR={:.3f} (hat {:.3f})  "
            "feas_pressio={}".format(
                eb, cr, cr_h, psnr, psnr_h, feas_m,
            )
        )
        if feas_m and (best is None or cr > best["compression_ratio"]):
            best = {
                "error_bound": eb,
                "compression_ratio": cr,
                "psnr": psnr,
                "ssim": ssim,
                "compression_ratio_hat": cr_h,
                "psnr_hat": psnr_h,
                "ssim_hat": ssim_h,
                "feasible": True,
                "calibration": "pressio",
            }
    return best, rows


# def phase3_surrogate_refine(
#     bundle,
#     e_candidates,
#     psnr_min: float,
#     ssim_min: float,
#     tol_g: float = 1e-6,
# ):
#     """Wise local search: max CR_hat among surrogate-feasible candidates."""
#     rows = []
#     best = None
#     for eb in e_candidates:
#         cr = float(bundle.cr(eb))
#         psnr = float(bundle.psnr(eb))
#         ssim = float(bundle.ssim(eb))
#         feas = (psnr >= psnr_min - tol_g) and (ssim >= ssim_min - tol_g)
#         row = {
#             "error_bound": eb,
#             "compression_ratio": cr,
#             "psnr": psnr,
#             "ssim": ssim,
#             "feasible": feas,
#         }
#         rows.append(row)
#         print(
#             "  e={:.6g}  CR_hat={:.4f}  PSNR_hat={:.3f}  SSIM_hat={:.5f}  "
#             "feasible={}".format(eb, cr, psnr, ssim, feas)
#         )
#         if feas and (best is None or cr > best["compression_ratio"]):
#             best = dict(row)
#     return best, rows


def phase3_pressio_verify(
    pressio_bin,
    input_path,
    e_candidates,
    compressor,
    dims,
    dtype,
    hdf5_field,
    psnr_min,
    ssim_min,
    tol_g=1e-6,
):
    rows = []
    best = None
    for eb in e_candidates:
        metrics, elapsed = run_pressio(
            pressio_bin, input_path, eb,
            compressor=compressor, dims=dims, dtype=dtype,
            hdf5_field=hdf5_field,
        )
        if metrics is None:
            continue
        cr = metrics.get("compression_ratio")
        psnr = metrics.get("psnr")
        ssim = metrics.get("ssim")
        row = {
            "error_bound": eb,
            "compression_ratio": cr,
            "psnr": psnr,
            "ssim": ssim,
            "wall_time_sec": elapsed,
        }
        rows.append(row)
        if cr is None or psnr is None or ssim is None:
            print("  e={:.6g}  incomplete metrics".format(eb))
            continue
        g1, g2 = psnr - psnr_min, ssim - ssim_min
        feas = (g1 >= -tol_g) and (g2 >= -tol_g)
        print(
            "  e={:.6g}  CR={:.4f}  PSNR={:.3f}  SSIM={:.5f}  feasible={}".format(
                eb, cr, psnr, ssim, feas,
            )
        )
        if feas and (best is None or cr > best["compression_ratio"]):
            best = dict(row)
            best["feasible"] = True
    return best, rows


def plot_overview(bundle, psnr_min, ssim_min, phase2_best, grid_best, out_path,
                  e_lo, e_hi, baseline_df=None):
    es = np.power(10.0, np.linspace(math.log10(e_lo), math.log10(e_hi), 800))
    cr_v = bundle.cr(es)
    feas = (bundle.psnr(es) >= psnr_min) & (bundle.ssim(es) >= ssim_min)

    fig, ax = plt.subplots(figsize=(9, 6))
    if baseline_df is not None and len(baseline_df):
        be = baseline_df["error_bound"].values.astype(float)
        bcr = baseline_df["compression_ratio"].values.astype(float)
        ax.scatter(
            np.log10(be), bcr, s=8, c="0.6", alpha=0.35, zorder=1,
            label="measured baseline ({} pts)".format(len(baseline_df)),
        )
    ax.plot(np.log10(es), cr_v, "b-", lw=2, label="CR_hat(e)")
    ax.fill_between(
        np.log10(es), 0, cr_v.max() * 1.05, where=feas,
        color="green", alpha=0.12, label="feasible (surrogate)",
    )
    if grid_best:
        ax.scatter(
            [math.log10(grid_best["e"])], [grid_best["cr"]],
            c="purple", s=90, marker="s", zorder=4,
            label="grid truth e*={:.3g}".format(grid_best["e"]),
        )
    if phase2_best:
        ax.scatter(
            [math.log10(phase2_best["e"])], [phase2_best["cr"]],
            c="red", s=140, marker="*", zorder=5,
            label="Phase-2 e*={:.3g}".format(phase2_best["e"]),
        )
    ax.set_xlabel(r"$\log_{10}(e)$")
    ax.set_ylabel("CR_hat")
    ax.set_title(
        "max CR s.t. PSNR>={:.2g}, SSIM>={:.3g}".format(psnr_min, ssim_min)
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_text_atomic(path: str, content: str) -> None:
    """Write text atomically and flush to disk (helps on NFS/shared filesystems)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_model_training_args(ap)
    ap.add_argument(
        "--baseline-csv", default=None,
        help="oracle baseline CSV (default: {out}/baseline_Npts.csv)",
    )
    ap.add_argument(
        "--baseline-n", type=int, default=None,
        help="pressio baseline sample count (default: 10000 if --input, else 0)",
    )
    ap.add_argument(
        "--baseline-jobs", type=int, default=None,
        help="parallel pressio workers for baseline (default: min(32, cpu_count))",
    )
    ap.add_argument(
        "--skip-baseline-sweep", action="store_true",
        help="do not run pressio; use existing --baseline-csv only",
    )
    ap.add_argument("--input", default=None,
                    help="dataset path for 10k baseline pressio sweep (oracle)")
    ap.add_argument("--pressio", default=DEFAULT_PRESSIO)
    ap.add_argument("--compressor", default="sz3")
    ap.add_argument("--dims", nargs=3, type=int, default=[500, 500, 100])
    ap.add_argument("--dtype", default="float")
    ap.add_argument("--hdf5-field", default=None)
    ap.add_argument("--iters", type=int, default=DEFAULT_LAGRANGIAN_ITERS)
    ap.add_argument("--grid-opt-n", type=int, default=DEFAULT_SURROGATE_GRID_N)
    ap.add_argument("--plot-grid-n", type=int, default=DEFAULT_PLOT_GRID_N,
                    help="grid points for phase1/phase2 plot curves")
    ap.add_argument("--skip-phase-plots", action="store_true",
                    help="do not run plot_phase1_surrogates / plot_phase2_lagrangian")
    ap.add_argument("--phase3-n", type=int, default=9,
                    help="log-spaced samples around Phase-2 e* in Phase 3")
    ap.add_argument("--phase3-span", type=float, default=0.25,
                    help="half-width in log10(e) for Phase-3 window")
    ap.add_argument("--phase3-pressio", action="store_true",
                    help="Phase 3: run real pressio at each candidate (requires --input)")
    ap.add_argument("--phase3-jobs", type=int, default=None,
                    help="parallel pressio workers for --phase3-pressio (default: min(32, cpu))")
    ap.add_argument("--skip-phase3", action="store_true")
    ap.add_argument("--verify-pressio", action="store_true",
                    help="optional: run one pressio at final model e* (not part of model)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    dims = tuple(args.dims)
    input_base = os.path.basename(args.input) if args.input else "n/a"

    if args.baseline_n is None:
        args.baseline_n = DEFAULT_BASELINE_N if args.input else 0
    if args.baseline_jobs is None:
        args.baseline_jobs = max(1, min(32, os.cpu_count() or 1))
    if args.phase3_jobs is None:
        args.phase3_jobs = max(1, min(32, os.cpu_count() or 1))
    if args.phase3_pressio and not args.input:
        print("ERROR: --phase3-pressio requires --input (dataset for pressio runs).",
              file=sys.stderr)
        sys.exit(2)

    from surrogate_lasso import load_sweep_csv

    df_train = load_sweep_csv(args.csv)
    eb_lo = float(df_train["error_bound"].min())
    eb_hi = float(df_train["error_bound"].max())

    baseline_csv = args.baseline_csv
    if args.baseline_n > 0:
        if not args.input:
            print("ERROR: --baseline-n requires --input for pressio runs.",
                  file=sys.stderr)
            sys.exit(2)
        if baseline_csv is None:
            baseline_csv = os.path.join(
                out_dir, "baseline_{}pts.csv".format(args.baseline_n))
        args.baseline_csv = baseline_csv
    elif baseline_csv is None:
        # No fresh pressio sweep requested and no explicit --baseline-csv:
        # reuse the densest existing baseline_*pts.csv in out_dir as the
        # measured oracle, so MAPE is computed against it (not training df).
        existing = glob.glob(os.path.join(out_dir, "baseline_*pts.csv"))

        def _baseline_pts(path: str) -> int:
            m = re.search(r"baseline_(\d+)pts\.csv$", os.path.basename(path))
            return int(m.group(1)) if m else 0

        existing = [p for p in existing if _baseline_pts(p) > 0]
        if existing:
            baseline_csv = max(existing, key=_baseline_pts)
            args.baseline_csv = baseline_csv
            print("Oracle      : reusing existing baseline {} ({} pts)".format(
                baseline_csv, _baseline_pts(baseline_csv)))

    print("=" * 72)
    print("Three-phase MODEL (Phase 1 + 2 + 3, surrogate-only)")
    print("=" * 72)
    print("CSV input   : {}".format(args.csv))
    if args.input:
        print("Dataset     : {}".format(args.input))
    if args.baseline_n > 0:
        print("Baseline    : {} pressio samples -> {}".format(
            args.baseline_n, baseline_csv))
        print("  jobs={}  resume={}".format(
            args.baseline_jobs, not args.skip_baseline_sweep))
    print("Constraints : PSNR >= {:.4g} dB,  SSIM >= {:.4g}".format(
        args.psnr_min, args.ssim_min))
    print("Phase 3     : {}".format(
        "pressio ({} samples, span={:.3g}, jobs={})".format(
            args.phase3_n, args.phase3_span, args.phase3_jobs)
        if args.phase3_pressio
        else "training-curve interp ({} candidates, span={:.3g})".format(
            args.phase3_n, args.phase3_span)))
    print("Output dir  : {}".format(out_dir))
    print()

    if args.baseline_n > 0 and args.skip_baseline_sweep:
        if not os.path.isfile(baseline_csv):
            print("ERROR: --skip-baseline-sweep but missing: {}".format(
                baseline_csv), file=sys.stderr)
            sys.exit(2)
    elif args.baseline_n > 0:
        run_baseline_sweep(
            pressio_bin=args.pressio,
            input_path=args.input,
            out_csv=baseline_csv,
            n=args.baseline_n,
            eb_lo=eb_lo,
            eb_hi=eb_hi,
            dims=dims,
            dtype=args.dtype,
            compressor=args.compressor,
            jobs=args.baseline_jobs,
            resume=True,
        )
        print()

    # Measured baseline (oracle / ground truth) loaded once for plots + MAPE.
    df_baseline = None
    if baseline_csv and os.path.isfile(baseline_csv):
        df_baseline = load_sweep_csv(baseline_csv)

    # ----- Phase 1 -----
    print("### Phase 1: Lasso surrogates from sweep CSV ###")
    phase1_dir = os.path.join(out_dir, "phase1")
    report_dir = phase1_dir if not args.skip_phase_plots else out_dir
    bundle, df = run_phase1(
        args.csv, degree=args.degree, report_dir=report_dir, df=df_train,
    )
    e_lo, e_hi = bundle.e_lo, bundle.e_hi
    log_phase1_fit(bundle)
    if not args.skip_phase_plots:
        from plot_phase1_surrogates import generate_phase1_plots
        generate_phase1_plots(
            bundle, df, phase1_dir,
            degree=args.degree, n_grid=args.plot_grid_n,
            baseline_df=df_baseline,
        )
        print("  Phase-1 plots -> {}".format(phase1_dir))
    print()

    grid_best = grid_truth_optimum(df, args.psnr_min, args.ssim_min)
    if grid_best:
        print("  Grid reference (best feasible on training points):")
        print("    e={:.6g}  CR={:.4f}  PSNR={:.3f}  SSIM={:.5f}".format(
            grid_best["e"], grid_best["cr"], grid_best["psnr"], grid_best["ssim"]))
    else:
        print("  [WARN] No feasible point on training grid for given constraints.")
    print()

    # ----- Phase 2 -----
    print("### Phase 2: quadratic-penalty solve on surrogates ###")
    p2 = run_phase2(
        bundle, args.psnr_min, args.ssim_min,
        iters=args.iters, grid_n=args.grid_opt_n, verbose=True,
    )
    phase2 = log_phase2_result(p2, bundle, e_lo, e_hi)
    if not args.skip_phase_plots:
        from plot_phase2_lagrangian import generate_phase2_plots
        phase2_dir = os.path.join(out_dir, "phase2")
        generate_phase2_plots(
            bundle, df, p2,
            args.psnr_min, args.ssim_min, phase2_dir,
            n_grid=args.plot_grid_n,
            csv_path=args.csv,
            degree=args.degree,
            iters=args.iters,
            grid_opt_n=args.grid_opt_n,
            baseline_df=df_baseline,
        )
        print("  Phase-2 plots -> {}".format(phase2_dir))

    e_star = phase2["e"]
    if args.phase3_pressio:
        e_candidates = logspace_candidates_near_e_star(
            e_star, args.phase3_n, args.phase3_span, e_lo, e_hi,
        )
    else:
        e_candidates = candidates_near_e_star(
            e_star, args.phase3_n, args.phase3_span, e_lo, e_hi, df=df,
        )

    # ----- Phase 3 (wise search near e*) -----
    phase3_best = None
    phase3_rows = []
    if args.skip_phase3:
        print("### Phase 3: skipped (--skip-phase3) ###")
        phase3_best = {
            "error_bound": phase2["e"],
            "compression_ratio": phase2["cr"],
            "psnr": phase2.get("psnr"),
            "ssim": phase2.get("ssim"),
            "feasible": True,
        }
    elif args.phase3_pressio:
        print("### Phase 3: pressio wise search near e*={:.6g} ###".format(e_star))
        print("  {} log-spaced pressio runs in span={:.3g}  jobs={}".format(
            len(e_candidates), args.phase3_span, args.phase3_jobs))
        print("  e range: [{:.6g}, {:.6g}]".format(
            min(e_candidates), max(e_candidates)))
        print()
        pressio_rows = run_pressio_on_ebs(
            pressio_bin=args.pressio,
            input_path=args.input,
            ebs=e_candidates,
            dims=dims,
            dtype=args.dtype,
            compressor=args.compressor,
            jobs=args.phase3_jobs,
            verbose=True,
        )
        if not pressio_rows:
            print("  [ERROR] Phase 3 pressio sweep returned no successful runs.")
            sys.exit(1)
        print()
        phase3_best, phase3_rows = phase3_pressio_wise_search(
            bundle, pressio_rows, args.psnr_min, args.ssim_min,
        )
        print()
        if phase3_best:
            print("  MODEL FINAL (Phase 3, measured by pressio):")
            print("    e*={:.6g}  CR={:.4f}  PSNR={:.3f}  SSIM={:.5f}".format(
                phase3_best["error_bound"],
                phase3_best["compression_ratio"],
                phase3_best["psnr"],
                phase3_best["ssim"],
            ))
        else:
            print("  [WARN] No feasible pressio point in Phase-3 window.")
    else:
        print("### Phase 3: wise search near e*={:.6g} (training-curve interp) ###".format(e_star))
        print("  Candidates ({} points, incl. sweep grid in window):".format(
            len(e_candidates)))
        for eb in e_candidates:
            print("    {:.6g}".format(eb))
        print()
        phase3_best, phase3_rows = phase3_wise_search(
            bundle, df, e_candidates, args.psnr_min, args.ssim_min,
        )
        print()
        if phase3_best:
            print("  MODEL FINAL (Phase 3, measured on training sweep curve):")
            print("    e*={:.6g}  CR={:.4f}  PSNR={:.3f}  SSIM={:.5f}".format(
                phase3_best["error_bound"],
                phase3_best["compression_ratio"],
                phase3_best["psnr"],
                phase3_best["ssim"],
            ))
        else:
            print("  [WARN] No feasible surrogate point in Phase-3 window.")

    # ----- Optional pressio verify (not part of model) -----
    verify_row = None
    if args.verify_pressio:
        if not args.input:
            print("[ERROR] --verify-pressio requires --input", file=sys.stderr)
            sys.exit(2)
        if phase3_best is None:
            print("[WARN] skip pressio verify: no Phase-3 model optimum")
        else:
            eb = phase3_best["error_bound"]
            print("\n### Optional pressio verify at model e*={:.6g} ###".format(eb))
            metrics, elapsed = run_pressio(
                args.pressio, args.input, eb,
                compressor=args.compressor, dims=dims, dtype=args.dtype,
                hdf5_field=args.hdf5_field,
            )
            if metrics:
                verify_row = {
                    "error_bound": eb,
                    "compression_ratio": metrics.get("compression_ratio"),
                    "psnr": metrics.get("psnr"),
                    "ssim": metrics.get("ssim"),
                    "wall_time_sec": elapsed,
                }
                print("  measured CR={} PSNR={} SSIM={}".format(
                    verify_row["compression_ratio"],
                    verify_row["psnr"], verify_row["ssim"]))

    # ----- MAPE vs pressio oracle (dense baseline CSV if provided) -----
    df_oracle = df_baseline if df_baseline is not None else df
    oracle = pressio_oracle_grid(df_oracle, args.psnr_min, args.ssim_min)
    mape_block = []
    if oracle and phase3_best:
        e_model = phase3_best["error_bound"]
        if phase3_best.get("calibration") == "pressio":
            model_eval = {
                "e": e_model,
                "cr": phase3_best["compression_ratio"],
                "psnr": phase3_best["psnr"],
                "ssim": phase3_best["ssim"],
            }
        elif args.baseline_csv:
            meas_o = measured_at_e(df_oracle, e_model)
            model_eval = {
                "e": e_model,
                "cr": meas_o["cr"],
                "psnr": meas_o["psnr"],
                "ssim": meas_o["ssim"],
            }
        else:
            model_eval = {
                "e": e_model,
                "cr": phase3_best["compression_ratio"],
                "psnr": phase3_best["psnr"],
                "ssim": phase3_best["ssim"],
            }
        mape = mape_model_vs_oracle(oracle, model_eval)
        mape_block = format_mape_block(
            oracle, model_eval, mape, n_grid=len(df_oracle),
        )
        print("\n### Validation MAPE (model vs pressio oracle) ###")
        for line in mape_block:
            print(line)

    # ----- Summary files -----
    summary_path = os.path.join(out_dir, "summary.txt")
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_lines = [
        "Three-phase MODEL output",
        "=" * 72,
        "(Phase 1 + 2 + 3 are the model; Phase 3 below is MODEL FINAL)",
        "generated_at : {}".format(generated_at),
        "input_file : {}".format(input_base),
        "sweep_csv  : {}".format(args.csv),
    ]
    if args.baseline_csv:
        summary_lines.append("baseline_csv : {}".format(args.baseline_csv))
    summary_lines += [
        "PSNR_min   : {:.6g}".format(args.psnr_min),
        "SSIM_min   : {:.6g}".format(args.ssim_min),
        "e_range    : [{:g}, {:g}]".format(e_lo, e_hi),
        "",
        "Phase 1 — surrogates (Lasso, degree={})".format(args.degree),
    ]
    for s in (bundle.psnr, bundle.ssim, bundle.cr):
        summary_lines.append("  {} R2={:.6f} RMSE={:.6g}".format(s.name, s.r2, s.rmse))
    summary_lines.append("")

    if grid_best:
        summary_lines.append("Training-grid reference (best feasible on sweep points):")
        for k, v in grid_best.items():
            summary_lines.append("  {:8s} = {}".format(k, v))
        summary_lines.append("")

    summary_lines.append("Phase 2 — quadratic-penalty solve (scipy minimize_scalar):")
    summary_lines.append("  selected_method = {}".format(phase2.get("method", "?")))
    for k, v in phase2.items():
        if k != "method":
            summary_lines.append("  {:10s} = {}".format(k, v))
    summary_lines.append("")

    summary_lines.append("MODEL FINAL — Phase 3 wise search (sweep-calibrated):")
    if phase3_best:
        for k, v in phase3_best.items():
            summary_lines.append("  {:18s} = {}".format(k, v))
    else:
        summary_lines.append("  (none feasible or skipped)")

    if verify_row:
        summary_lines.append("")
        summary_lines.append("Optional pressio verify (not part of model):")
        for k, v in verify_row.items():
            summary_lines.append("  {:18s} = {}".format(k, v))

    if mape_block:
        summary_lines.append("")
        summary_lines.extend(mape_block)

    _write_text_atomic(summary_path, "\n".join(summary_lines) + "\n")
    print("\nUpdated {}  (PSNR_min={:.6g}, SSIM_min={:.6g}, at {})".format(
        summary_path, args.psnr_min, args.ssim_min, generated_at))

    if phase3_rows:
        p3_csv = os.path.join(out_dir, "phase3_runs.csv")
        with open(p3_csv, "w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(phase3_rows[0].keys()))
            w.writeheader()
            w.writerows(phase3_rows)

    plot_overview(
        bundle, args.psnr_min, args.ssim_min, phase2, grid_best,
        os.path.join(out_dir, "feasible_surrogate.png"), e_lo, e_hi,
    )

    print("\nWrote results to: {}".format(out_dir))
    for n in sorted(os.listdir(out_dir)):
        print("  -", n)


if __name__ == "__main__":
    main()
