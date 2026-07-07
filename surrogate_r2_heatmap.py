#!/usr/bin/env python3
"""
R² heatmaps over Lasso hyperparameters (alpha × degree).

Cases:
  same_field       — one fixed 20-pt train + 10k oracle (optional bootstrap)
  cloud_timesteps  — same field (CLOUD), multiple timesteps f01..fN;
                     per timestep: 20-pt train + own 10k oracle;
                     aggregate mean / p05 / p95 / var of R² across timesteps
  fields           — different fields at one timestep (default f01);
                     per field: 20-pt train + own 10k oracle;
                     aggregate mean / p05 / p95 / var of R² across fields

Example (same field):
    python surrogate_r2_heatmap.py --case same_field \\
      --train-csv ../Hurricane/results/CLOUDf01/CLOUDf01_sz3_sweep.csv \\
      --baseline-csv ../Hurricane/results/CLOUDf01/three_phase_CLOUDf01/baseline_10000pts.csv \\
      --degrees 1,2,3,4,5,6,7,8 --n-alphas 40 \\
      --out ../Hurricane/results/CLOUDf01/r2_heatmaps

Example (CLOUD timesteps f01-f24):
    python surrogate_r2_heatmap.py --case cloud_timesteps \\
      --results-dir ../Hurricane/results/CLOUDf01 \\
      --sweeps-dir ../Hurricane/results/CLOUDf01/sweeps \\
      --timesteps 1-24 \\
      --degrees 1,2,3,4,5,6,7,8 --n-alphas 40 \\
      --out ../Hurricane/results/CLOUDf01/r2_heatmaps_cloud

Example (different fields at f01):
    python surrogate_r2_heatmap.py --case fields \\
      --results-dir ../Hurricane/results/CLOUDf01 \\
      --sweeps-dir ../Hurricane/results/CLOUDf01/sweeps \\
      --fields CLOUD,P,PRECIP,QCLOUD,QGRAUP --timestep 1 \\
      --degrees 1,2,3,4,5,6,7,8 --n-alphas 40 \\
      --out ../Hurricane/results/CLOUDf01/r2_heatmaps_fields

Prerequisite sweeps per field/timestep (single runner, choose field via --field):
    cd ../Hurricane
    python run_cloudf01_pressio_sweep.py --field PRECIPf01 --input PRECIPf01.bin --preset train \\
      --jobs 32 --resume
    python run_cloudf01_pressio_sweep.py --field PRECIPf01 --input PRECIPf01.bin --preset baseline \\
      --jobs 32 --resume
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from surrogate_lasso import (  # noqa: E402
    default_alpha_grid,
    fit_surrogates_from_dataframe,
    load_sweep_csv,
    oracle_mean_r2,
)
from hurricane_paths import (  # noqa: E402
    DEFAULT_FIELD,
    baseline_csv as hp_baseline_csv,
    three_phase_out,
    train_sweep_csv,
)

# Fallback paths for CLOUDf01 when using standard sweeps layout.
CLOUDF01_TRAIN_ALT = train_sweep_csv(DEFAULT_FIELD)
CLOUDF01_BASELINE_ALT = os.path.join(
    three_phase_out(DEFAULT_FIELD), "baseline_10000pts.csv",
)


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_timesteps(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def stem(prefix: str, idx: int) -> str:
    return "{}f{:02d}".format(prefix, idx)


def bootstrap_sample(df, n: int, rng: np.random.Generator):
    idx = rng.integers(0, len(df), size=n)
    return df.iloc[idx].reset_index(drop=True)


def resolve_timestep_pair(
    results_dir: str, sweeps_dir: str, name: str,
) -> tuple[str, str]:
    """Return (train_csv, baseline_csv) for one timestep stem e.g. CLOUDf01."""
    train_candidates = [
        os.path.join(results_dir, "{}_sz3_sweep.csv".format(name)),
        os.path.join(sweeps_dir, "{}_sz3_sweep.csv".format(name)),
        os.path.join(sweeps_dir, "{}_train20.csv".format(name)),
    ]
    baseline_candidates = [
        os.path.join(sweeps_dir, "{}_baseline10k.csv".format(name)),
        os.path.join(results_dir, "{}_baseline10k.csv".format(name)),
        os.path.join(sweeps_dir, "{}_baseline10000pts.csv".format(name)),
    ]
    if name == "CLOUDf01":
        train_candidates = [CLOUDF01_TRAIN_ALT] + train_candidates
        baseline_candidates = [
            CLOUDF01_BASELINE_ALT,
            hp_baseline_csv(DEFAULT_FIELD),
        ] + baseline_candidates

    train_csv = next((p for p in train_candidates if os.path.isfile(p)), None)
    baseline_csv = next((p for p in baseline_candidates if os.path.isfile(p)), None)
    if train_csv is None:
        raise FileNotFoundError(
            "No train CSV for {} (checked {})".format(name, results_dir))
    if baseline_csv is None:
        raise FileNotFoundError(
            "No baseline CSV for {} (checked {})".format(name, sweeps_dir))
    return train_csv, baseline_csv


def load_cloud_timestep_replicates(
    results_dir: str,
    sweeps_dir: str,
    field_prefix: str,
    timestep_indices: list[int],
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """List of (label, df_train, df_baseline) per timestep."""
    out = []
    for idx in timestep_indices:
        name = stem(field_prefix, idx)
        train_csv, baseline_csv = resolve_timestep_pair(
            results_dir, sweeps_dir, name,
        )
        df_train = load_sweep_csv(train_csv)
        df_baseline = load_sweep_csv(baseline_csv)
        out.append((name, df_train, df_baseline))
    return out


def parse_field_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def discover_fields(hurricane_dir: str, timestep: int, n: int) -> list[str]:
    """First *n* field prefixes (sorted) that have {prefix}f{tt}.bin."""
    suffix = "f{:02d}.bin".format(timestep)
    prefixes = sorted(
        fname[: -len(suffix)]
        for fname in os.listdir(hurricane_dir)
        if fname.endswith(suffix)
        and os.path.isfile(os.path.join(hurricane_dir, fname))
    )
    return prefixes[:n]


def load_field_replicates(
    results_dir: str,
    sweeps_dir: str,
    field_prefixes: list[str],
    timestep: int,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """List of (label, df_train, df_baseline) per field at one timestep.

    Reuses resolve_timestep_pair: the {STEM} = {field}f{tt} naming is shared
    between the timesteps case and the fields case.
    """
    out = []
    for prefix in field_prefixes:
        name = stem(prefix, timestep)
        train_csv, baseline_csv = resolve_timestep_pair(
            results_dir, sweeps_dir, name,
        )
        df_train = load_sweep_csv(train_csv)
        df_baseline = load_sweep_csv(baseline_csv)
        out.append((name, df_train, df_baseline))
    return out


def run_replicate_grid(
    replicates: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    degrees: list[int],
    alphas: np.ndarray,
    *,
    bootstrap_per_replicate: int = 1,
    seed: int = 0,
    case_label: str = "replicates",
):
    """For each (degree, alpha), fit per replicate → R² on its oracle → stats."""
    n_rep = len(replicates)
    n_deg = len(degrees)
    n_alp = len(alphas)
    n_cells = n_deg * n_alp
    cell = 0
    rng = np.random.default_rng(seed)

    mean_grid = np.full((n_deg, n_alp), np.nan)
    p05_grid = np.full((n_deg, n_alp), np.nan)
    p95_grid = np.full((n_deg, n_alp), np.nan)
    var_grid = np.full((n_deg, n_alp), np.nan)

    t0 = time.time()
    for i, deg in enumerate(degrees):
        for j, alpha in enumerate(alphas):
            cell += 1
            r2_all = []
            for label, df_train, df_baseline in replicates:
                n_train = len(df_train)
                for _ in range(bootstrap_per_replicate):
                    df_fit = (
                        df_train if bootstrap_per_replicate == 1
                        else bootstrap_sample(df_train, n_train, rng)
                    )
                    try:
                        bundle = fit_surrogates_from_dataframe(
                            df_fit, degree=deg, alpha=float(alpha),
                            source_label=label,
                        )
                        r2_all.append(oracle_mean_r2(bundle, df_baseline))
                    except Exception:
                        r2_all.append(np.nan)

            valid = np.array([x for x in r2_all if np.isfinite(x)], dtype=float)
            if len(valid):
                mean_grid[i, j] = float(np.mean(valid))
                p05_grid[i, j] = float(np.percentile(valid, 5))
                p95_grid[i, j] = float(np.percentile(valid, 95))
                var_grid[i, j] = (
                    float(np.var(valid, ddof=1)) if len(valid) > 1 else 0.0
                )
            if cell % max(1, n_cells // 20) == 0 or cell == n_cells:
                elapsed = time.time() - t0
                print("  [{:>14s}] grid {}/{}  deg={} alpha={:.3g}  "
                      "mean_R2={:.4f}  n_rep={}  [{:.0f}s]".format(
                          case_label, cell, n_cells, deg, alpha,
                          mean_grid[i, j] if np.isfinite(mean_grid[i, j]) else float("nan"),
                          len(valid), elapsed,
                      ))
    return {
        "mean": mean_grid,
        "p05": p05_grid,
        "p95": p95_grid,
        "var": var_grid,
    }


def plot_heatmap(
    grid: np.ndarray,
    degrees: list[int],
    alphas: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: str,
) -> None:
    n_deg, n_alp = grid.shape
    fig_w = max(16.0, 0.5 * n_alp)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    log_alphas = np.log10(alphas)
    if n_alp > 1:
        dx = (log_alphas[-1] - log_alphas[0]) / (n_alp - 1) / 2.0
    else:
        dx = 0.5
    x_edges = np.linspace(log_alphas[0] - dx, log_alphas[-1] + dx, n_alp + 1)
    y_edges = np.arange(n_deg + 1) - 0.5
    im = ax.pcolormesh(x_edges, y_edges, grid, shading="flat", cmap="viridis")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(-0.5, n_deg - 0.5)
    ax.set_yticks(range(len(degrees)))
    ax.set_yticklabels([str(d) for d in degrees])
    ax.set_xlabel("alpha")
    ax.set_ylabel("Lasso polynomial degree")
    ax.set_title(title)

    ax.set_xticks(log_alphas)
    tick_fs = 6 if n_alp >= 30 else 8
    ax.set_xticklabels(
        ["{:.3g}".format(a) for a in alphas],
        rotation=90, ha="center", va="top", fontsize=tick_fs,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_grid_csv(path: str, case_name: str, degrees, alphas, grids: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "stat", "degree", "alpha", "r2_value"])
        for stat, grid in grids.items():
            for i, deg in enumerate(degrees):
                for j, alpha in enumerate(alphas):
                    w.writerow([
                        case_name, stat, deg, alpha,
                        grid[i, j] if np.isfinite(grid[i, j]) else "",
                    ])


def write_summary(path: str, lines: list[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_case_heatmaps(
    grids: dict,
    degrees: list[int],
    alphas: np.ndarray,
    case_dir: str,
    case_title: str,
    *,
    single_r2: bool = False,
) -> list[str]:
    os.makedirs(case_dir, exist_ok=True)
    written = []
    if single_r2:
        out_png = os.path.join(case_dir, "r2.png")
        plot_heatmap(
            grids["mean"], degrees, alphas,
            "{}: oracle R²".format(case_title),
            "R²", out_png,
        )
        written.append(out_png)
    else:
        specs = [
            ("mean", "mean oracle R²", "mean R²", "mean_r2.png"),
            ("p05", "5th percentile oracle R²", "5th pct R²", "p05_r2.png"),
            ("p95", "95th percentile oracle R²", "95th pct R²", "p95_r2.png"),
            ("var", "variance of oracle R²", "var(R²)", "var_r2.png"),
        ]
        for stat, subtitle, label, fname in specs:
            if stat not in grids:
                continue
            out_png = os.path.join(case_dir, fname)
            plot_heatmap(
                grids[stat], degrees, alphas,
                "{}: {}".format(case_title, subtitle),
                label, out_png,
            )
            written.append(out_png)
    return written


def run_same_field(args, degrees, alphas):
    df_train = load_sweep_csv(args.train_csv)
    df_baseline = load_sweep_csv(args.baseline_csv)
    replicates = [("same_field", df_train, df_baseline)]

    print("Same-field R² heatmap scan")
    print("  train     : {} ({} pts)".format(args.train_csv, len(df_train)))
    print("  baseline  : {} ({} pts)".format(args.baseline_csv, len(df_baseline)))
    print("  degrees   : {}".format(degrees))
    print("  alphas    : {} values [{:.3g}, {:.3g}]".format(
        len(alphas), alphas.min(), alphas.max()))
    if args.n_bootstrap > 1:
        print("  bootstrap : {} per cell".format(args.n_bootstrap))
    else:
        print("  mode      : fixed 20-pt train")
    print()

    grids = run_replicate_grid(
        replicates, degrees, alphas,
        bootstrap_per_replicate=max(1, args.n_bootstrap),
        seed=args.seed,
        case_label="same_field",
    )
    case_dir = os.path.join(args.out, "same_field")
    single = args.n_bootstrap == 1
    written = plot_case_heatmaps(
        grids, degrees, alphas, case_dir, "Same-field", single_r2=single,
    )
    summary = [
        "Same-field R² heatmaps (oracle evaluation)",
        "=" * 60,
        "train_csv    : {}".format(args.train_csv),
        "baseline_csv : {}".format(args.baseline_csv),
        "n_bootstrap  : {}".format(args.n_bootstrap),
        "degrees      : {}".format(degrees),
        "n_alphas     : {}".format(len(alphas)),
    ]
    return grids, "same_field", written, summary


def run_cloud_timesteps(args, degrees, alphas):
    indices = parse_timesteps(args.timesteps)
    replicates = load_cloud_timestep_replicates(
        args.results_dir, args.sweeps_dir, args.field_prefix, indices,
    )

    print("CLOUD timestep R² heatmap scan")
    print("  results_dir: {}".format(args.results_dir))
    print("  sweeps_dir : {}".format(args.sweeps_dir))
    print("  timesteps  : {} ({})".format(indices, args.field_prefix))
    print("  replicates : {}".format(len(replicates)))
    for label, df_t, df_b in replicates:
        print("    {}  train={}  baseline={}".format(label, len(df_t), len(df_b)))
    print("  degrees    : {}".format(degrees))
    print("  alphas     : {} values [{:.3g}, {:.3g}]".format(
        len(alphas), alphas.min(), alphas.max()))
    print()

    grids = run_replicate_grid(
        replicates, degrees, alphas,
        bootstrap_per_replicate=1,
        seed=args.seed,
        case_label="cloud_ts",
    )
    case_dir = os.path.join(args.out, "cloud_timesteps")
    written = plot_case_heatmaps(
        grids, degrees, alphas, case_dir, "CLOUD timesteps",
        single_r2=False,
    )
    summary = [
        "CLOUD timestep R² heatmaps (oracle per timestep)",
        "=" * 60,
        "results_dir  : {}".format(args.results_dir),
        "sweeps_dir   : {}".format(args.sweeps_dir),
        "field_prefix : {}".format(args.field_prefix),
        "timesteps    : {}".format(indices),
        "n_replicates : {}".format(len(replicates)),
        "degrees      : {}".format(degrees),
        "n_alphas     : {}".format(len(alphas)),
        "",
        "Per timestep: fit on 20-pt train, R² on that file's 10k oracle.",
        "Heatmap stats: mean / p05 / p95 / var across timesteps.",
    ]
    return grids, "cloud_timesteps", written, summary


def run_fields(args, degrees, alphas):
    if args.fields:
        fields = parse_field_list(args.fields)
    else:
        fields = discover_fields(args.hurricane_dir, args.timestep, args.n_fields)
    replicates = load_field_replicates(
        args.results_dir, args.sweeps_dir, fields, args.timestep,
    )

    print("Multi-field R² heatmap scan")
    print("  hurricane  : {}".format(args.hurricane_dir))
    print("  results_dir: {}".format(args.results_dir))
    print("  sweeps_dir : {}".format(args.sweeps_dir))
    print("  timestep   : f{:02d}".format(args.timestep))
    print("  fields     : {}".format(fields))
    print("  replicates : {}".format(len(replicates)))
    for label, df_t, df_b in replicates:
        print("    {}  train={}  baseline={}".format(label, len(df_t), len(df_b)))
    print("  degrees    : {}".format(degrees))
    print("  alphas     : {} values [{:.3g}, {:.3g}]".format(
        len(alphas), alphas.min(), alphas.max()))
    print()

    grids = run_replicate_grid(
        replicates, degrees, alphas,
        bootstrap_per_replicate=1,
        seed=args.seed,
        case_label="fields",
    )
    case_dir = os.path.join(args.out, "fields")
    written = plot_case_heatmaps(
        grids, degrees, alphas, case_dir,
        "Hurricane fields (f{:02d})".format(args.timestep),
        single_r2=False,
    )
    summary = [
        "Multi-field R² heatmaps (oracle per field, timestep f{:02d})".format(
            args.timestep),
        "=" * 60,
        "hurricane_dir: {}".format(args.hurricane_dir),
        "results_dir  : {}".format(args.results_dir),
        "sweeps_dir   : {}".format(args.sweeps_dir),
        "fields       : {}".format(fields),
        "n_replicates : {}".format(len(replicates)),
        "degrees      : {}".format(degrees),
        "n_alphas     : {}".format(len(alphas)),
        "",
        "Per field: fit on 20-pt train, R² on that field's 10k oracle.",
        "Heatmap stats: mean / p05 / p95 / var across fields.",
    ]
    return grids, "fields", written, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--case", choices=["same_field", "cloud_timesteps", "fields"],
        default="same_field",
    )
    ap.add_argument("--train-csv", default=None,
                    help="20-pt train CSV (same_field case)")
    ap.add_argument("--baseline-csv", default=None,
                    help="10k oracle CSV (same_field case)")
    ap.add_argument("--results-dir", default=None,
                    help="train CSVs: {STEM}_sz3_sweep.csv (cloud_timesteps/fields)")
    ap.add_argument("--sweeps-dir", default=None,
                    help="oracle CSVs: {STEM}_baseline10k.csv (cloud_timesteps/fields)")
    ap.add_argument("--field-prefix", default="CLOUD")
    ap.add_argument("--timesteps", default="1-24",
                    help="e.g. 1-24 for CLOUDf01..f24 (cloud_timesteps)")
    ap.add_argument("--hurricane-dir", default=None,
                    help="dir with {FIELD}f{tt}.bin (fields case)")
    ap.add_argument("--fields", default=None,
                    help="comma-separated prefixes, e.g. CLOUD,P,PRECIP (fields)")
    ap.add_argument("--n-fields", type=int, default=5,
                    help="fields case: first N fields when --fields omitted")
    ap.add_argument("--timestep", type=int, default=1,
                    help="fields case: single timestep index (f01, f02, ...)")
    ap.add_argument("--degrees", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--n-alphas", type=int, default=40)
    ap.add_argument("--alpha-min", type=float, default=None)
    ap.add_argument("--alpha-max", type=float, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=1,
                    help="same_field only: bootstrap replicates per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    degrees = parse_int_list(args.degrees)
    if not degrees:
        print("ERROR: --degrees is empty", file=sys.stderr)
        sys.exit(2)

    hurricane_dir = os.path.join(os.path.dirname(_SCRIPT_DIR), "Hurricane")
    hurricane_results = os.path.join(hurricane_dir, "results", "CLOUDf01")

    if args.case == "same_field":
        if not args.train_csv or not args.baseline_csv:
            print("ERROR: same_field requires --train-csv and --baseline-csv",
                  file=sys.stderr)
            sys.exit(2)
        ref_train = load_sweep_csv(args.train_csv)
    elif args.case == "cloud_timesteps":
        if args.results_dir is None:
            args.results_dir = hurricane_results
        if args.sweeps_dir is None:
            args.sweeps_dir = os.path.join(hurricane_results, "sweeps")
        indices = parse_timesteps(args.timesteps)
        train_csv, _ = resolve_timestep_pair(
            args.results_dir, args.sweeps_dir,
            stem(args.field_prefix, indices[0]),
        )
        ref_train = load_sweep_csv(train_csv)
    else:  # fields
        if args.hurricane_dir is None:
            args.hurricane_dir = hurricane_dir
        if args.results_dir is None:
            args.results_dir = hurricane_results
        if args.sweeps_dir is None:
            args.sweeps_dir = os.path.join(hurricane_results, "sweeps")
        if args.fields:
            ref_fields = parse_field_list(args.fields)
        else:
            ref_fields = discover_fields(
                args.hurricane_dir, args.timestep, args.n_fields)
        if not ref_fields:
            print("ERROR: no fields found under {}".format(args.hurricane_dir),
                  file=sys.stderr)
            sys.exit(2)
        train_csv, _ = resolve_timestep_pair(
            args.results_dir, args.sweeps_dir,
            stem(ref_fields[0], args.timestep),
        )
        ref_train = load_sweep_csv(train_csv)

    if args.alpha_min is not None and args.alpha_max is not None:
        alphas = np.logspace(
            np.log10(args.alpha_min), np.log10(args.alpha_max), args.n_alphas,
        )
    else:
        ref_deg = degrees[len(degrees) // 2]
        alphas = default_alpha_grid(ref_train, degree=ref_deg, n_alphas=args.n_alphas)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    if args.case == "same_field":
        grids, case_name, written, summary = run_same_field(args, degrees, alphas)
    elif args.case == "cloud_timesteps":
        grids, case_name, written, summary = run_cloud_timesteps(args, degrees, alphas)
    else:
        grids, case_name, written, summary = run_fields(args, degrees, alphas)

    write_grid_csv(os.path.join(out_dir, "heatmap_grid.csv"), case_name, degrees, alphas, grids)
    write_summary(os.path.join(out_dir, "summary.txt"), summary)

    print()
    print("Wrote:")
    for p in written:
        print("  {}".format(p))
    print("  {}/heatmap_grid.csv".format(out_dir))
    print("  {}/summary.txt".format(out_dir))


if __name__ == "__main__":
    main()
