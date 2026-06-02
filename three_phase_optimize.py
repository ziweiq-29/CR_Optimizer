#!/usr/bin/env python3
"""
Three-phase MODEL for constrained CR optimization.

All three phases are the model (surrogate-only, no pressio at inference):

    maximize    compression_ratio(e)
    subject to  PSNR(e) >= PSNR_min,  SSIM(e) >= SSIM_min

Phase 1 — Fit Lasso surrogates from a pressio sweep CSV (training data).
Phase 2 — Augmented Lagrangian on surrogates → coarse e*.
Phase 3 — Wise local surrogate search near e* → MODEL FINAL output.

Validation is separate: compare model final vs ORACLE (pressio brute-force
baseline on the sweep grid) using validate_three_phase.py.

Optional --verify-pressio: one pressio run at model e* for debugging only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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
    dual_ascent,
    find_max_e_for,
)
from pressio_run import DEFAULT_PRESSIO, run_pressio  # noqa: E402
from surrogate_lasso import fit_surrogates_from_csv, write_surrogate_report  # noqa: E402
from validation_metrics import (  # noqa: E402
    format_mape_block,
    measured_at_e,
    mape_model_vs_oracle,
    pressio_oracle_grid,
)


def surrogate_feasible_maximum(
    cr_hat,
    psnr_hat,
    ssim_hat,
    psnr_min: float,
    ssim_min: float,
    e_lo: float,
    e_hi: float,
    n: int = 40001,
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


def sample_near_e_star(e_star: float, n: int, log_span: float, e_lo: float, e_hi: float):
    """Log-uniform offsets around e_star, clipped to [e_lo, e_hi]."""
    if n <= 1:
        return [float(np.clip(e_star, e_lo, e_hi))]
    offsets = np.linspace(-log_span, log_span, n)
    es = [float(np.clip(e_star * (10.0 ** d), e_lo, e_hi)) for d in offsets]
    es.append(float(np.clip(e_star, e_lo, e_hi)))
    return sorted(set(es))


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
            }
    return best, rows


def phase3_surrogate_refine(
    bundle,
    e_candidates,
    psnr_min: float,
    ssim_min: float,
    tol_g: float = 1e-6,
):
    """Wise local search: max CR_hat among surrogate-feasible candidates."""
    rows = []
    best = None
    for eb in e_candidates:
        cr = float(bundle.cr(eb))
        psnr = float(bundle.psnr(eb))
        ssim = float(bundle.ssim(eb))
        feas = (psnr >= psnr_min - tol_g) and (ssim >= ssim_min - tol_g)
        row = {
            "error_bound": eb,
            "compression_ratio": cr,
            "psnr": psnr,
            "ssim": ssim,
            "feasible": feas,
        }
        rows.append(row)
        print(
            "  e={:.6g}  CR_hat={:.4f}  PSNR_hat={:.3f}  SSIM_hat={:.5f}  "
            "feasible={}".format(eb, cr, psnr, ssim, feas)
        )
        if feas and (best is None or cr > best["compression_ratio"]):
            best = dict(row)
    return best, rows


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
                  e_lo, e_hi):
    es = np.power(10.0, np.linspace(math.log10(e_lo), math.log10(e_hi), 800))
    cr_v = bundle.cr(es)
    feas = (bundle.psnr(es) >= psnr_min) & (bundle.ssim(es) >= ssim_min)

    fig, ax = plt.subplots(figsize=(9, 6))
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", required=True, help="Phase-1 pressio sweep CSV")
    ap.add_argument("--input", default=None,
                    help="dataset path (only needed with --verify-pressio)")
    ap.add_argument("--pressio", default=DEFAULT_PRESSIO)
    ap.add_argument("--compressor", default="sz3")
    ap.add_argument("--dims", nargs=3, type=int, default=[500, 500, 100])
    ap.add_argument("--dtype", default="float")
    ap.add_argument("--hdf5-field", default=None)
    ap.add_argument("--psnr-min", type=float, default=80.0)
    ap.add_argument("--ssim-min", type=float, default=0.9)
    ap.add_argument("--degree", type=int, default=4, help="Lasso poly degree")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--phase3-n", type=int, default=9,
                    help="samples in log-space around Phase-2 e*")
    ap.add_argument("--phase3-span", type=float, default=0.25,
                    help="half-width in log10(e) for Phase-3 local search")
    ap.add_argument("--skip-phase3", action="store_true")
    ap.add_argument("--verify-pressio", action="store_true",
                    help="optional: run one pressio at final model e* (not part of model)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    dims = tuple(args.dims)
    input_base = os.path.basename(args.input) if args.input else "n/a"

    print("=" * 72)
    print("Three-phase MODEL (Phase 1 + 2 + 3, surrogate-only)")
    print("=" * 72)
    print("CSV input   : {}".format(args.csv))
    if args.input:
        print("Dataset     : {}".format(args.input))
    print("Constraints : PSNR >= {:.4g} dB,  SSIM >= {:.4g}".format(
        args.psnr_min, args.ssim_min))
    print("Output dir  : {}".format(out_dir))
    print()

    # ----- Phase 1 -----
    print("### Phase 1: Lasso surrogates from sweep CSV ###")
    bundle = fit_surrogates_from_csv(args.csv, degree=args.degree)
    e_lo, e_hi = bundle.e_lo, bundle.e_hi
    write_surrogate_report(bundle, out_dir)
    for s in (bundle.psnr, bundle.ssim, bundle.cr):
        print("  {}  R2={:.4f}  RMSE={:.4g}  alpha={:.4g}".format(
            s.name, s.r2, s.rmse, s.alpha))
    print()

    from surrogate_lasso import load_sweep_csv
    df = load_sweep_csv(args.csv)
    grid_best = grid_truth_optimum(df, args.psnr_min, args.ssim_min)
    if grid_best:
        print("  Grid reference (best feasible on training points):")
        print("    e={:.6g}  CR={:.4f}  PSNR={:.3f}  SSIM={:.5f}".format(
            grid_best["e"], grid_best["cr"], grid_best["psnr"], grid_best["ssim"]))
    else:
        print("  [WARN] No feasible point on training grid for given constraints.")
    print()

    # ----- Phase 2 -----
    print("### Phase 2: Augmented Lagrangian on surrogates ###")
    e_psnr_cap = find_max_e_for(bundle.psnr, args.psnr_min, e_lo, e_hi)
    e_ssim_cap = find_max_e_for(bundle.ssim, args.ssim_min, e_lo, e_hi)
    if e_psnr_cap is None or e_ssim_cap is None:
        e_direct = None
        print("  Surrogate feasible set empty in [{:g}, {:g}]".format(e_lo, e_hi))
    else:
        e_direct = min(e_psnr_cap, e_ssim_cap)
        print("  Direct cap (monotone surrogates): e*={:.6g}  CR_hat={:.4f}".format(
            e_direct, float(bundle.cr(e_direct))))

    best2, history = dual_ascent(
        bundle.cr, bundle.psnr, bundle.ssim,
        args.psnr_min, args.ssim_min, e_lo, e_hi,
        method="auglag", T=args.iters, verbose=True,
    )
    grid_opt = surrogate_feasible_maximum(
        bundle.cr, bundle.psnr, bundle.ssim,
        args.psnr_min, args.ssim_min, e_lo, e_hi,
    )
    phase2 = pick_phase2_e_star(
        best2, grid_opt, e_direct, bundle.cr, bundle.psnr, bundle.ssim,
    )
    print()
    if phase2 is None:
        print("  [ERROR] Phase 2 found no feasible surrogate optimum.")
        sys.exit(1)
    print("  Phase-2 optimum (surrogate, method={}):".format(phase2.get("method", "?")))
    print("    e*={:.6g}  CR_hat={:.4f}  PSNR_hat={:.3f}  SSIM_hat={:.5f}".format(
        phase2["e"], phase2["cr"],
        phase2.get("psnr", float("nan")),
        phase2.get("ssim", float("nan")),
    ))
    if best2 is not None and phase2.get("method") != "auglag":
        print("  (Augmented Lagrangian best feasible: e={:.6g} CR_hat={:.4f})".format(
            best2["e"], best2["cr"]))
    print()

    e_star = phase2["e"]
    e_candidates = candidates_near_e_star(
        e_star, args.phase3_n, args.phase3_span, e_lo, e_hi, df=df,
    )

    # ----- Phase 3 (model: surrogate wise local search) -----
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
    else:
        print("### Phase 3: wise search near e*={:.6g} (sweep-calibrated) ###".format(e_star))
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
            print("  MODEL FINAL (Phase 3, measured on sweep curve):")
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

    # ----- MAPE vs pressio oracle -----
    oracle = pressio_oracle_grid(df, args.psnr_min, args.ssim_min)
    mape_block = []
    if oracle and phase3_best:
        model_eval = {
            "e": phase3_best["error_bound"],
            "cr": phase3_best["compression_ratio"],
            "psnr": phase3_best["psnr"],
            "ssim": phase3_best["ssim"],
        }
        mape = mape_model_vs_oracle(oracle, model_eval)
        mape_block = format_mape_block(oracle, model_eval, mape)
        print("\n### Validation MAPE (model vs pressio oracle) ###")
        for line in mape_block:
            print(line)

    # ----- Summary files -----
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Three-phase MODEL output\n")
        f.write("=" * 72 + "\n")
        f.write("(Phase 1 + 2 + 3 are the model; Phase 3 below is MODEL FINAL)\n")
        f.write("input_file : {}\n".format(input_base))
        f.write("sweep_csv  : {}\n".format(args.csv))
        f.write("PSNR_min   : {:.6g}\n".format(args.psnr_min))
        f.write("SSIM_min   : {:.6g}\n".format(args.ssim_min))
        f.write("e_range    : [{:g}, {:g}]\n\n".format(e_lo, e_hi))

        f.write("Phase 1 — surrogates (Lasso, degree={})\n".format(args.degree))
        for s in (bundle.psnr, bundle.ssim, bundle.cr):
            f.write("  {} R2={:.6f} RMSE={:.6g}\n".format(s.name, s.r2, s.rmse))
        f.write("\n")

        if grid_best:
            f.write("Training-grid reference (best feasible on sweep points):\n")
            for k, v in grid_best.items():
                f.write("  {:8s} = {}\n".format(k, v))
            f.write("\n")

        f.write("Phase 2 — Lagrangian + surrogate grid optimum:\n")
        f.write("  selected_method = {}\n".format(phase2.get("method", "?")))
        for k, v in phase2.items():
            if k != "method":
                f.write("  {:10s} = {}\n".format(k, v))
        if best2 is not None:
            f.write("\n  Augmented Lagrangian best feasible iterate:\n")
            for k, v in best2.items():
                f.write("    {:10s} = {}\n".format(k, v))
        f.write("\n")

        if e_direct is not None:
            f.write("Direct monotone-cap e* = {:.6g}\n\n".format(e_direct))

        f.write("MODEL FINAL — Phase 3 wise search (sweep-calibrated):\n")
        if phase3_best:
            for k, v in phase3_best.items():
                f.write("  {:18s} = {}\n".format(k, v))
        else:
            f.write("  (none feasible or skipped)\n")

        if verify_row:
            f.write("\nOptional pressio verify (not part of model):\n")
            for k, v in verify_row.items():
                f.write("  {:18s} = {}\n".format(k, v))

        if mape_block:
            f.write("\n")
            for line in mape_block:
                f.write(line + "\n")

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
