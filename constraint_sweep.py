#!/usr/bin/env python3
"""
Sweep multiple PSNR/SSIM constraint pairs through the three-phase model.

Phase 1 (surrogate fit) runs once; Phase 2 + 3 re-run for each constraint pair.
Results are written to a dedicated CSV + text report (not summary.txt).

Edit CONSTRAINTS below, or pass pairs on the command line / via a file.

Example:
    cd /anvil/projects/x-cis240669/optimizer
    python constraint_sweep.py \\
      --csv /anvil/projects/x-cis240669/Hurricane/results/CLOUDf01_sz3_sweep.csv \\
      --baseline-csv /anvil/projects/x-cis240669/Hurricane/results/three_phase_CLOUDf01/baseline_10000pts.csv \\
      --out /anvil/projects/x-cis240669/Hurricane/results/three_phase_CLOUDf01/constraint_sweep

    # override constraints on CLI (pairs: psnr ssim psnr ssim ...)
    python constraint_sweep.py --csv ... --baseline-csv ... --out ... \\
      --constraints 80 0.9 85 0.95 85 0.99
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from surrogate_lasso import load_sweep_csv  # noqa: E402
from three_phase_optimize import (  # noqa: E402
    DEFAULT_DEGREE,
    candidates_near_e_star,
    run_phase1,
    run_phase2,
)
from validation_metrics import (  # noqa: E402
    measured_at_e,
    mape_model_vs_oracle,
    pressio_oracle_grid,
)

# ---------------------------------------------------------------------------
# Manual constraint list — edit these (psnr_min_dB, ssim_min) pairs as needed.
# ---------------------------------------------------------------------------
CONSTRAINTS: list[tuple[float, float]] = [
    (80.0, 0.90),
    (85.0, 0.95),
    (90.0, 0.99),
]


def _parse_constraint_pairs(flat: list[float]) -> list[tuple[float, float]]:
    if len(flat) % 2 != 0:
        raise ValueError(
            "--constraints expects an even number of values (psnr ssim pairs)")
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]


def _load_constraints_file(path: str) -> list[tuple[float, float]]:
    pairs = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError("{}:{}: need psnr_min and ssim_min".format(
                    path, lineno))
            pairs.append((float(parts[0]), float(parts[1])))
    if not pairs:
        raise ValueError("no constraints in {}".format(path))
    return pairs


def _find_baseline_csv(out_dir: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    existing = glob.glob(os.path.join(out_dir, "baseline_*pts.csv"))

    def _pts(p: str) -> int:
        m = re.search(r"baseline_(\d+)pts\.csv$", os.path.basename(p))
        return int(m.group(1)) if m else 0

    existing = [p for p in existing if _pts(p) > 0]
    return max(existing, key=_pts) if existing else None


def _phase3_search(
    bundle,
    df_train,
    e_candidates,
    psnr_min: float,
    ssim_min: float,
    tol_g: float = 1e-6,
):
    """Phase 3 wise search (same logic as three_phase_optimize, no printing)."""
    best = None
    for eb in e_candidates:
        cr_h = float(bundle.cr(eb))
        psnr_h = float(bundle.psnr(eb))
        ssim_h = float(bundle.ssim(eb))
        meas = measured_at_e(df_train, eb)
        feas_m = (
            meas["psnr"] >= psnr_min - tol_g
            and meas["ssim"] >= ssim_min - tol_g
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
    return best


CSV_COLUMNS = [
    "psnr_min",
    "ssim_min",
    "e_range",
    "model_error_bound",
    "model_cr",
    "model_psnr",
    "model_ssim",
    "baseline_error_bound",
    "baseline_cr",
    "baseline_psnr",
    "baseline_ssim",
    "mape_error_bound_pct",
    "mape_cr_pct",
    "mape_psnr_pct",
    "mape_ssim_pct",
]


def _run_one(
    bundle,
    df_train,
    df_baseline,
    psnr_min: float,
    ssim_min: float,
    *,
    phase3_n: int,
    phase3_span: float,
) -> dict:
    """Run Phase 2 + 3 for one constraint pair; return a CSV row dict."""
    row = {
        "psnr_min": psnr_min,
        "ssim_min": ssim_min,
        "e_range": "[{:g}, {:g}]".format(bundle.e_lo, bundle.e_hi),
        "status": "ok",
    }

    p2 = run_phase2(bundle, psnr_min, ssim_min, verbose=False)
    phase2 = p2["phase2"]
    if phase2 is None:
        row["status"] = "phase2_infeasible"
        return row

    e_candidates = candidates_near_e_star(
        phase2["e"], phase3_n, phase3_span,
        bundle.e_lo, bundle.e_hi, df=df_train,
    )
    phase3 = _phase3_search(
        bundle, df_train, e_candidates, psnr_min, ssim_min,
    )
    if phase3 is None:
        row["status"] = "phase3_infeasible"
        return row

    row.update({
        "model_error_bound": phase3["error_bound"],
        "model_cr": phase3["compression_ratio"],
        "model_psnr": phase3["psnr"],
        "model_ssim": phase3["ssim"],
    })

    if df_baseline is not None:
        oracle = pressio_oracle_grid(df_baseline, psnr_min, ssim_min)
        if oracle is None:
            row["status"] = "oracle_infeasible"
            return row

        model_eval = measured_at_e(df_baseline, phase3["error_bound"])
        mape = mape_model_vs_oracle(oracle, model_eval)
        row.update({
            "baseline_error_bound": oracle["e"],
            "baseline_cr": oracle["cr"],
            "baseline_psnr": oracle["psnr"],
            "baseline_ssim": oracle["ssim"],
            "mape_error_bound_pct": mape["e"],
            "mape_cr_pct": mape["cr"],
            "mape_psnr_pct": mape["psnr"],
            "mape_ssim_pct": mape["ssim"],
        })

    return row


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_report(path: str, rows: list[dict], meta: dict) -> None:
    lines = [
        "Constraint sweep — three-phase model",
        "=" * 72,
        "generated_at : {}".format(meta["generated_at"]),
        "sweep_csv    : {}".format(meta["csv"]),
        "baseline_csv : {}".format(meta.get("baseline_csv", "n/a")),
        "n_constraints: {}".format(len(rows)),
        "",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            "[{:02d}] PSNR_min={:.6g}  SSIM_min={:.6g}  status={}".format(
                i, r["psnr_min"], r["ssim_min"], r.get("status", "?")))
        if r.get("status") != "ok":
            lines.append("")
            continue
        lines += [
            "  e_range: {}".format(r.get("e_range", "")),
            "  MODEL:",
            "    error_bound={:.6g}  CR={:.4f}  PSNR={:.3f}  SSIM={:.5f}".format(
                r["model_error_bound"], r["model_cr"],
                r["model_psnr"], r["model_ssim"]),
        ]
        if "baseline_cr" in r:
            lines += [
                "  BASELINE (oracle):",
                "    error_bound={:.6g}  CR={:.4f}  PSNR={:.3f}  SSIM={:.5f}".format(
                    r["baseline_error_bound"], r["baseline_cr"],
                    r["baseline_psnr"], r["baseline_ssim"]),
                "  MAPE (%):",
                "    error_bound={:.3f}  CR={:.3f}  PSNR={:.3f}  SSIM={:.3f}".format(
                    r["mape_error_bound_pct"], r["mape_cr_pct"],
                    r["mape_psnr_pct"], r["mape_ssim_pct"]),
            ]
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", required=True, help="Phase-1 training sweep CSV")
    ap.add_argument("--baseline-csv", default=None,
                    help="pressio oracle baseline (10k); auto-detect in --out parent")
    ap.add_argument("--out", required=True,
                    help="directory for sweep results (constraint_sweep.csv)")
    ap.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    ap.add_argument("--phase3-n", type=int, default=9)
    ap.add_argument("--phase3-span", type=float, default=0.25)
    ap.add_argument(
        "--constraints", nargs="*", type=float, metavar=("PSNR", "SSIM"),
        help="constraint pairs: psnr1 ssim1 psnr2 ssim2 ... (overrides script list)",
    )
    ap.add_argument(
        "--constraints-file", default=None,
        help="file with one 'psnr_min ssim_min' pair per line (# comments ok)",
    )
    args = ap.parse_args()

    if args.constraints:
        pairs = _parse_constraint_pairs(args.constraints)
    elif args.constraints_file:
        pairs = _load_constraints_file(args.constraints_file)
    else:
        pairs = list(CONSTRAINTS)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    baseline_csv = _find_baseline_csv(
        os.path.dirname(out_dir) if out_dir.endswith("constraint_sweep") else out_dir,
        args.baseline_csv,
    )
    if baseline_csv is None:
        baseline_csv = _find_baseline_csv(out_dir, args.baseline_csv)
    if args.baseline_csv:
        baseline_csv = args.baseline_csv

    df_train = load_sweep_csv(args.csv)
    df_baseline = load_sweep_csv(baseline_csv) if baseline_csv else None

    print("=" * 72)
    print("Constraint sweep (three-phase model)")
    print("=" * 72)
    print("Training CSV : {}".format(args.csv))
    print("Baseline CSV : {}".format(baseline_csv or "(none — MAPE skipped)"))
    print("Output dir   : {}".format(out_dir))
    print("Constraints  : {} pair(s)".format(len(pairs)))
    for psnr, ssim in pairs:
        print("  PSNR >= {:.6g},  SSIM >= {:.6g}".format(psnr, ssim))
    print()

    print("### Phase 1: fit surrogates (once) ###")
    bundle, _ = run_phase1(args.csv, degree=args.degree, df=df_train)
    for s in (bundle.psnr, bundle.ssim, bundle.cr):
        print("  {}  R2={:.4f}  RMSE={:.4g}".format(s.name, s.r2, s.rmse))
    print()

    rows = []
    for i, (psnr_min, ssim_min) in enumerate(pairs, 1):
        print("### [{}/{}] PSNR>={:.6g}, SSIM>={:.6g} ###".format(
            i, len(pairs), psnr_min, ssim_min))
        row = _run_one(
            bundle, df_train, df_baseline,
            psnr_min, ssim_min,
            phase3_n=args.phase3_n,
            phase3_span=args.phase3_span,
        )
        rows.append(row)
        if row.get("status") == "ok" and "model_cr" in row:
            print("  MODEL: e={:.6g}  CR={:.4f}  PSNR={:.3f}  SSIM={:.5f}".format(
                row["model_error_bound"], row["model_cr"],
                row["model_psnr"], row["model_ssim"]))
            if "mape_cr_pct" in row:
                print("  MAPE: e={:.3f}%  CR={:.3f}%  PSNR={:.3f}%  SSIM={:.3f}%".format(
                    row["mape_error_bound_pct"], row["mape_cr_pct"],
                    row["mape_psnr_pct"], row["mape_ssim_pct"]))
        else:
            print("  status: {}".format(row.get("status")))
        print()

    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "generated_at": generated_at,
        "csv": args.csv,
        "baseline_csv": baseline_csv,
    }

    csv_path = os.path.join(out_dir, "constraint_sweep.csv")
    txt_path = os.path.join(out_dir, "constraint_sweep.txt")
    _write_csv(csv_path, rows)
    _write_report(txt_path, rows, meta)

    print("Wrote:")
    print("  {}".format(csv_path))
    print("  {}".format(txt_path))


if __name__ == "__main__":
    main()
