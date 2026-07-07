#!/usr/bin/env python3
"""Validation: MODEL vs pressio oracle baseline (MAPE)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from surrogate_lasso import load_sweep_csv  # noqa: E402
from validation_metrics import (  # noqa: E402
    format_mape_block,
    measured_at_e,
    mape_model_vs_oracle,
    pressio_oracle_grid,
)


def load_model_final(summary_path: str) -> dict:
    out = {}
    in_p3 = False
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Phase 3") or line.startswith("MODEL FINAL"):
                in_p3 = True
                continue
            if in_p3 and (
                line.startswith("Optional pressio")
                or line.startswith("Validation")
            ):
                break
            if in_p3 and " = " in line:
                k, v = line.split(" = ", 1)
                k = k.strip()
                if k in ("feasible", "feasible_surrogate", "feasible_measured"):
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    if "compression_ratio" not in out and "error_bound" not in out:
        raise ValueError("No MODEL FINAL in: {}".format(summary_path))
    return {
        "e": float(out.get("error_bound", out.get("e"))),
        "cr": float(out["compression_ratio"]),
        "psnr": float(out.get("psnr", float("nan"))),
        "ssim": float(out.get("ssim", float("nan"))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline-csv", "--csv", dest="baseline_csv", required=True,
        help="10k (or full) pressio baseline sweep CSV for oracle",
    )
    ap.add_argument(
        "--train-csv", default=None,
        help="optional training sweep; model e is re-evaluated on baseline if omitted",
    )
    ap.add_argument("--summary", "--empirical-summary", dest="summary", required=True)
    ap.add_argument("--psnr-min", type=float, default=80.0)
    ap.add_argument("--ssim-min", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not os.path.isfile(args.summary):
        print("ERROR: summary not found: {}".format(args.summary), file=sys.stderr)
        sys.exit(2)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    df_baseline = load_sweep_csv(args.baseline_csv)
    oracle = pressio_oracle_grid(df_baseline, args.psnr_min, args.ssim_min)
    if oracle is None:
        print("Oracle infeasible.", file=sys.stderr)
        sys.exit(1)

    model = load_model_final(args.summary)
    # Compare model e on the same measured baseline curve (log-interp on dense grid).
    meas = measured_at_e(df_baseline, model["e"])
    model_eval = {
        "e": model["e"],
        "cr": meas["cr"],
        "psnr": meas["psnr"],
        "ssim": meas["ssim"],
    }
    mape = mape_model_vs_oracle(oracle, model_eval)
    lines = [
        "Validation: MODEL vs BASELINE (pressio oracle)",
        "=" * 60,
        "Baseline CSV : {}".format(args.baseline_csv),
        "Constraints    : PSNR >= {:.4g}, SSIM >= {:.4g}".format(
            args.psnr_min, args.ssim_min),
        "",
    ] + format_mape_block(oracle, model_eval, mape, n_grid=len(df_baseline))

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(out_dir, "validation_report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(out_dir, "validation_summary.json"), "w") as f:
        json.dump({
            "baseline_csv": args.baseline_csv,
            "oracle": oracle,
            "model_summary": model,
            "model_eval_on_baseline": model_eval,
            "mape_pct": mape,
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["error_bound", "CR", "PSNR", "SSIM"]
    ax.bar(labels, [mape["e"], mape["cr"], mape["psnr"], mape["ssim"]], color="steelblue")
    ax.set_ylabel("MAPE vs oracle (%)")
    ax.set_title("MODEL vs pressio oracle")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "final_vs_oracle_mape.png"), dpi=150)
    plt.close(fig)
    print("\nWrote:", out_dir)


if __name__ == "__main__":
    main()
