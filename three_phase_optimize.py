#!/usr/bin/env python3
"""
Three-phase MODEL for constrained CR optimization.

All three phases are the model (surrogate-only, no pressio at inference):

    maximize    compression_ratio(e)
    subject to  PSNR(e) >= PSNR_min,  SSIM(e) >= SSIM_min

Phase 1 — Fit Lasso surrogates from a pressio sweep CSV (training data).
Phase 2 — Augmented Lagrangian on surrogates → coarse e*.
Phase 3 — Random local search near e* with real pressio measurements.

Phase 0 — training-point pressio sweep (auto when --input is set and CSV missing).
  Timed wall-clock for all training error_bounds; writes sweeps/{field}_train{N}.csv.
  Use --skip-train-sweep to skip pressio when that CSV already exists.
Timing reports phase0/1/2/3 and with vs without Phase 3.

With --input, runs a 10k-point pressio baseline sweep (oracle ground truth).
Phase 3 requires --input (always runs real pressio near Phase-2 e*).

Optional --verify-pressio: one extra pressio at model e* (debug only).

By default also writes Phase-1/2 diagnostic plots under {out}/phase1 and {out}/phase2
(same parameters as plot_phase1_surrogates.py / plot_phase2_lagrangian.py).
Use --skip-phase-plots to disable.

Example (fixed hyperparameters from heatmap):

    python three_phase_optimize.py \\
      --field CLOUDf01 --train-points 48 \\
      --input .../CLOUDf01.bin \\
      --baseline-csv .../CLOUDf01_baseline10k.csv --skip-baseline-sweep \\
      --dims 500 500 100 --psnr-min 80 --ssim-min 0.9 \\
      --degree 4 --alpha 0.01 \\
      --out .../three_phase_CLOUDf01

CR surrogate segmentation (optional; default is global single segment):

    # auto: scan training data for K-1 breakpoints (exactly K segments, min RSS)
    python three_phase_optimize.py ... --cr-segmentation auto --cr-segments-max 2
    # manual breakpoints
    python three_phase_optimize.py ... --cr-segmentation fixed --cr-breakpoints 0.01
    # legacy flags still work: --piecewise-cr (auto), --cr-breakpoints (fixed)

Training point count (writes CLOUDf01_train{N}.csv under --sweeps-dir):

    python three_phase_optimize.py \\
      --train-points 48 \\
      --input .../CLOUDf01.bin --baseline-csv .../CLOUDf01_baseline10k.csv \\
      --skip-baseline-sweep --degree 2 --alpha 0.00139 \\
      --out .../three_phase_train48

Tradeoff sweep (train × phase3 grid; then plot with train_phase3_tradeoff.py).
No --csv needed: each cell uses sweeps/{field}_train{N}.csv; missing CSVs are
generated via Phase-0 pressio unless --skip-train-csvs.

    python three_phase_optimize.py --tradeoff-sweep \\
      --train-points 12 24 48 --phase3-points 12 24 48 \\
      --input .../CLOUDf01.bin --baseline-csv .../CLOUDf01_baseline10k.csv \\
      --skip-baseline-sweep --degree 2 --alpha 0.00139 \\
      --out Hurricane/results/CLOUDf01/tradeoff_sweep

Resume skips existing train CSVs (same row count) and grid cells with matching
run_config.json. Use --no-resume to force re-run.

Default paths are per-field under Hurricane/results/{field}/ (default field: CLOUDf01).

CLOUDf04 example:

    python three_phase_optimize.py --field CLOUDf04 \\
      --train-points 48 --phase3-points 24 --skip-baseline-sweep \\
      --degree 2 --alpha 0.00139 \\
      --out Hurricane/results/CLOUDf04/three_phase_CLOUDf04
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running as script from optimizer/ or repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from lagrangian_constrained import (  # noqa: E402
    penalty_minimize,
)
from pressio_baseline_sweep import (  # noqa: E402
    DEFAULT_BASELINE_N,
    eb_key,
    error_bounds_logspace,
    run_baseline_sweep,
    run_pressio_on_ebs,
    write_csv,
)
from hurricane_paths import (  # noqa: E402
    DEFAULT_COMPRESSOR,
    DEFAULT_FIELD,
    baseline_csv as hp_baseline_csv,
    ensure_field_layout,
    field_stem_from_input,
    input_bin as hp_input_bin,
    normalize_compressor,
    normalize_field_stem,
    sweeps_dir as hp_sweeps_dir,
)
from pressio_run import DEFAULT_PRESSIO, run_pressio  # noqa: E402
from surrogate_lasso import fit_surrogates_from_csv, write_surrogate_report  # noqa: E402
from validation_metrics import (  # noqa: E402
    format_mape_block,
    format_oracle_header,
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
DEFAULT_EB_LO = 1e-6
DEFAULT_EB_HI = 1e-1

# Tradeoff sweep defaults (train × phase3 grid; per-field under results/{stem}/)
DEFAULT_INPUT = hp_input_bin(DEFAULT_FIELD)
DEFAULT_BASELINE_CSV = hp_baseline_csv(DEFAULT_FIELD)
DEFAULT_SWEEPS_DIR = hp_sweeps_dir(DEFAULT_FIELD)
DEFAULT_TRAIN_POINTS = [12, 24, 48, 96, 192]
DEFAULT_PHASE3_POINTS = [12, 24, 48, 96, 192]
TRAIN_TIMING_CSV = "train_sweep_timings.csv"
CELL_CONFIG_JSON = "run_config.json"
SWEEP_CONFIG_JSON = "tradeoff_run_config.json"
CELL_DIR_RE = re.compile(r"^train(\d+)_phase3(\d+)$")


def _phase0_sidecar_path(csv_path: str) -> str:
    return csv_path + ".phase0_sec"


def _read_phase0_sidecar(csv_path: str) -> float:
    path = _phase0_sidecar_path(csv_path)
    if not os.path.isfile(path):
        return float("nan")
    try:
        with open(path) as f:
            return float(f.read().strip())
    except ValueError:
        return float("nan")


def _write_phase0_sidecar(csv_path: str, sec: float) -> None:
    path = _phase0_sidecar_path(csv_path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("{:.6f}\n".format(sec))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _train_csv_path(sweeps_dir: str, n_train: int, stem: str) -> str:
    return os.path.join(sweeps_dir, "{}_train{}.csv".format(stem, n_train))


def _resolve_field_args(args, *, set_default_baseline: bool = False) -> str:
    """Set field-dependent defaults from --field and/or --input."""
    stem = getattr(args, "field", None) or DEFAULT_FIELD
    if args.input:
        stem = field_stem_from_input(args.input, default=stem)
    stem = normalize_field_stem(stem)
    args.field = stem
    compressor = normalize_compressor(
        getattr(args, "compressor", None) or DEFAULT_COMPRESSOR
    )
    args.compressor = compressor
    ensure_field_layout(stem, compressor=compressor)
    if args.input is None:
        args.input = hp_input_bin(stem)
    if args.sweeps_dir is None:
        args.sweeps_dir = hp_sweeps_dir(stem, compressor=compressor)
    if args.baseline_csv is None and (
        set_default_baseline or getattr(args, "skip_baseline_sweep", False)
    ):
        n = getattr(args, "baseline_n", None) or 10000
        args.baseline_csv = hp_baseline_csv(
            stem, compressor=compressor, n=int(n) if n else 10000,
        )
    return stem


def _cell_dir(out_dir: str, n_train: int, n_phase3: int) -> str:
    return os.path.join(out_dir, "train{}_phase3{}".format(n_train, n_phase3))


def _make_run_config(
    *,
    degree: int,
    alpha: float,
    psnr_min: float,
    ssim_min: float,
    phase3_span: float,
    cr_segmentation: str = "none",
    piecewise_cr: bool = False,
    cr_breakpoints: list[float] | None = None,
) -> dict:
    cfg = {
        "degree": int(degree),
        "alpha": float(alpha),
        "psnr_min": float(psnr_min),
        "ssim_min": float(ssim_min),
        "phase3_span": float(phase3_span),
        "cr_segmentation": cr_segmentation,
        "piecewise_cr": bool(piecewise_cr),
    }
    if cr_breakpoints:
        cfg["cr_breakpoints"] = [float(b) for b in cr_breakpoints]
    return cfg


def _float_eq(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-12 + 1e-9 * max(abs(float(a)), abs(float(b)))


def _cr_segmentation_from_config(cfg: dict) -> str:
    """Normalize stored run config (supports legacy piecewise_cr-only JSON)."""
    if cfg.get("cr_segmentation"):
        return str(cfg["cr_segmentation"])
    if cfg.get("cr_breakpoints"):
        return "fixed"
    if cfg.get("piecewise_cr"):
        return "auto"
    return "none"


def _configs_match(stored: dict, current: dict) -> bool:
    for key in ("degree", "alpha", "psnr_min", "ssim_min"):
        if key not in stored:
            return False
        if key == "degree":
            if int(stored[key]) != int(current[key]):
                return False
        elif not _float_eq(stored[key], current[key]):
            return False
    if "phase3_span" in stored:
        if not _float_eq(stored["phase3_span"], current["phase3_span"]):
            return False
    if _cr_segmentation_from_config(stored) != _cr_segmentation_from_config(current):
        return False
    if "cr_breakpoints" in stored or "cr_breakpoints" in current:
        sbp = stored.get("cr_breakpoints")
        cbp = current.get("cr_breakpoints")
        if sbp is None or cbp is None:
            if sbp != cbp:
                return False
        elif len(sbp) != len(cbp) or any(
            not _float_eq(a, b) for a, b in zip(sbp, cbp)
        ):
            return False
    return True


def _parse_config_from_summary(summary_path: str) -> dict | None:
    if not os.path.isfile(summary_path):
        return None
    text = open(summary_path).read()
    config: dict = {}
    m = re.search(r"degree=(\d+),\s*alpha=([\d.eE+-]+)", text)
    if m:
        config["degree"] = int(m.group(1))
        config["alpha"] = float(m.group(2))
    m2 = re.search(r"PSNR_min\s*:\s*([\d.eE+-]+)", text)
    m3 = re.search(r"SSIM_min\s*:\s*([\d.eE+-]+)", text)
    if m2:
        config["psnr_min"] = float(m2.group(1))
    if m3:
        config["ssim_min"] = float(m3.group(1))
    if {"degree", "alpha", "psnr_min", "ssim_min"} - set(config.keys()):
        return None
    return config


def _load_cell_config(cell: str) -> dict | None:
    path = os.path.join(cell, CELL_CONFIG_JSON)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return _parse_config_from_summary(os.path.join(cell, "summary.txt"))


def _write_cell_config(cell: str, config: dict) -> None:
    with open(os.path.join(cell, CELL_CONFIG_JSON), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def _write_sweep_config(
    out_dir: str,
    config: dict,
    train_points: list[int],
    phase3_points: list[int],
) -> None:
    payload = {
        **config,
        "train_points": train_points,
        "phase3_points": phase3_points,
    }
    with open(os.path.join(out_dir, SWEEP_CONFIG_JSON), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _should_skip_cell(cell: str, config: dict, resume: bool) -> bool:
    if not resume:
        return False
    if not os.path.isfile(os.path.join(cell, "summary.txt")):
        return False
    stored = _load_cell_config(cell)
    if stored is None:
        return False
    return _configs_match(stored, config)


def _load_phase0_timings(path: str) -> dict[int, float]:
    if not os.path.isfile(path):
        return {}
    out: dict[int, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["n_train"])
            out[n] = float(row["phase0_sec"])
    return out


def _save_phase0_timings(
    path: str,
    timings: dict[int, float],
    csv_paths: dict[int, str],
) -> None:
    rows = []
    for n in sorted(timings):
        rows.append({
            "n_train": n,
            "phase0_sec": "{:.6f}".format(timings[n]),
            "csv_path": csv_paths.get(n, ""),
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["n_train", "phase0_sec", "csv_path", "generated_at"],
        )
        w.writeheader()
        w.writerows(rows)


def _resolve_tradeoff_point_lists(args) -> tuple[list[int], list[int]]:
    if args.train_points is not None or args.points is not None:
        train_points = sorted(set(
            args.train_points if args.train_points is not None else args.points
        ))
    else:
        train_points = list(DEFAULT_TRAIN_POINTS)
    if args.phase3_points is not None or args.points is not None:
        phase3_points = sorted(set(
            args.phase3_points if args.phase3_points is not None else args.points
        ))
    else:
        phase3_points = list(DEFAULT_PHASE3_POINTS)
    return train_points, phase3_points


def _resolve_single_phase3_n(args) -> int:
    """Phase-3 pressio sample count for a single (non-tradeoff) run."""
    if args.phase3_points is not None:
        if len(args.phase3_points) > 1:
            print("ERROR: multiple --phase3-points requires --tradeoff-sweep.",
                  file=sys.stderr)
            sys.exit(2)
        return int(args.phase3_points[0])
    return 9


def gen_train_csvs(
    train_points: list[int],
    *,
    stem: str,
    sweeps_dir: str,
    input_path: str,
    train_jobs: int,
    out_dir: str,
    resume: bool,
    pressio_bin: str = DEFAULT_PRESSIO,
    dims: tuple[int, int, int],
    dtype: str,
    compressor: str,
) -> dict[int, float]:
    """Run pressio at every training point count; return phase0_sec map."""
    from surrogate_lasso import load_sweep_csv

    os.makedirs(sweeps_dir, exist_ok=True)
    timing_path = os.path.join(out_dir, TRAIN_TIMING_CSV)
    existing = _load_phase0_timings(timing_path)
    timings: dict[int, float] = dict(existing)
    csv_paths: dict[int, str] = {}

    for n in train_points:
        csv_path = _train_csv_path(sweeps_dir, n, stem)
        csv_paths[n] = csv_path
        if resume and os.path.isfile(csv_path):
            df = load_sweep_csv(csv_path)
            if len(df) == n:
                sidecar = _read_phase0_sidecar(csv_path)
                if not math.isnan(sidecar):
                    timings[n] = sidecar
                elif n in existing:
                    timings[n] = existing[n]
                print("[Phase0] skip (resume) n_train={}  {} rows -> {}".format(
                    n, len(df), csv_path))
                continue
        ebs = error_bounds_logspace(n, DEFAULT_EB_LO, DEFAULT_EB_HI)
        print("[Phase0] n_train={}  {} pressio runs -> {}".format(
            n, len(ebs), csv_path))
        print("  jobs={}".format(train_jobs))
        t0 = time.perf_counter()
        rows = run_pressio_on_ebs(
            pressio_bin=pressio_bin,
            input_path=input_path,
            ebs=ebs,
            dims=dims,
            dtype=dtype,
            compressor=compressor,
            jobs=train_jobs,
            verbose=True,
        )
        elapsed = time.perf_counter() - t0
        if not rows:
            print("ERROR: Phase 0 returned no successful pressio runs for n={}".format(
                n), file=sys.stderr)
            sys.exit(1)
        rows_by_key = {eb_key(float(r["error_bound"])): r for r in rows}
        write_csv(csv_path, rows_by_key, ebs)
        timings[n] = elapsed
        _write_phase0_sidecar(csv_path, elapsed)
        print("  phase0_sec = {:.2f}".format(elapsed))

    _save_phase0_timings(timing_path, timings, csv_paths)
    print("Wrote {}".format(timing_path))
    return timings


def _run_single_cell(
    *,
    stem: str,
    n_train: int,
    n_phase3: int,
    train_csv: str,
    baseline_csv: str,
    input_path: str,
    out_cell: str,
    compressor: str,
    sweeps_dir: str,
    degree: int,
    alpha: float,
    psnr_min: float,
    ssim_min: float,
    phase3_span: float,
    phase3_jobs: int,
    cr_segmentation: str,
    cr_breakpoints: list[float] | None,
    dims: tuple[int, int, int],
) -> int:
    os.makedirs(out_cell, exist_ok=True)
    script = os.path.join(_SCRIPT_DIR, "three_phase_optimize.py")
    cmd = [
        sys.executable, script,
        "--train-points", str(n_train),
        "--baseline-csv", baseline_csv,
        "--skip-baseline-sweep",
        "--skip-train-sweep",
        "--field", stem,
        "--compressor", compressor,
        "--sweeps-dir", sweeps_dir,
        "--input", input_path,
        "--dims", str(dims[0]), str(dims[1]), str(dims[2]),
        "--degree", str(degree),
        "--alpha", str(alpha),
        "--psnr-min", str(psnr_min),
        "--ssim-min", str(ssim_min),
        "--phase3-points", str(n_phase3),
        "--phase3-span", str(phase3_span),
        "--phase3-jobs", str(phase3_jobs),
        "--skip-phase-plots",
        "--out", out_cell,
        "--cr-segmentation", cr_segmentation,
    ]
    if cr_breakpoints:
        cmd += ["--cr-breakpoints"] + [str(b) for b in cr_breakpoints]
    print("\n" + "=" * 72)
    print("train={}  phase3={}  ->  {}".format(n_train, n_phase3, out_cell))
    print(" ".join(cmd))
    print("=" * 72)
    proc = subprocess.run(cmd, cwd=_SCRIPT_DIR)
    return proc.returncode


def run_tradeoff_grid(
    train_points: list[int],
    phase3_points: list[int],
    *,
    stem: str,
    compressor: str,
    sweeps_dir: str,
    baseline_csv: str,
    input_path: str,
    out_dir: str,
    phase3_jobs: int,
    resume: bool,
    run_config: dict,
    dims: tuple[int, int, int],
) -> None:
    total = len(train_points) * len(phase3_points)
    done = 0
    for n_train in train_points:
        train_csv = _train_csv_path(sweeps_dir, n_train, stem)
        if not os.path.isfile(train_csv):
            print("ERROR: missing train CSV: {}  (run without --skip-train-csvs)".format(
                train_csv), file=sys.stderr)
            sys.exit(2)
        for n_phase3 in phase3_points:
            done += 1
            cell = _cell_dir(out_dir, n_train, n_phase3)
            if _should_skip_cell(cell, run_config, resume):
                cfg_path = os.path.join(cell, CELL_CONFIG_JSON)
                if not os.path.isfile(cfg_path):
                    _write_cell_config(cell, run_config)
                print("[{}/{}] skip (resume, same config) train={} phase3={}".format(
                    done, total, n_train, n_phase3))
                continue
            if resume and os.path.isfile(os.path.join(cell, "summary.txt")):
                print("[{}/{}] re-run (config changed) train={} phase3={}".format(
                    done, total, n_train, n_phase3))
            rc = _run_single_cell(
                stem=stem,
                n_train=n_train,
                n_phase3=n_phase3,
                train_csv=train_csv,
                baseline_csv=baseline_csv,
                input_path=input_path,
                out_cell=cell,
                compressor=compressor,
                sweeps_dir=sweeps_dir,
                degree=int(run_config["degree"]),
                alpha=float(run_config["alpha"]),
                psnr_min=float(run_config["psnr_min"]),
                ssim_min=float(run_config["ssim_min"]),
                phase3_span=float(run_config["phase3_span"]),
                phase3_jobs=phase3_jobs,
                cr_segmentation=_cr_segmentation_from_config(run_config),
                cr_breakpoints=run_config.get("cr_breakpoints"),
                dims=dims,
            )
            if rc != 0:
                print("WARNING: three_phase_optimize exited {} for train={} phase3={}".format(
                    rc, n_train, n_phase3), file=sys.stderr)
            else:
                _write_cell_config(cell, run_config)


def run_tradeoff_sweep(args) -> None:
    """Train × phase3 grid: Phase0 CSVs + one three-phase run per cell."""
    stem = _resolve_field_args(args, set_default_baseline=True)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    train_points, phase3_points = _resolve_tradeoff_point_lists(args)
    dims = tuple(args.dims)

    if args.skip_train_sweep:
        print(
            "NOTE: --skip-train-sweep ignored with --tradeoff-sweep; use "
            "--skip-train-csvs to skip Phase-0 train CSV generation.",
            file=sys.stderr,
        )
    if not args.skip_train_csvs and not args.input:
        print(
            "ERROR: --tradeoff-sweep needs --input to generate missing train CSVs "
            "(or pass --skip-train-csvs if sweeps/ CSVs already exist).",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.alpha is None:
        print("ERROR: --tradeoff-sweep requires fixed --alpha", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.baseline_csv):
        print("ERROR: baseline oracle missing: {}".format(args.baseline_csv),
              file=sys.stderr)
        sys.exit(2)

    cr_seg, cr_bps = resolve_cr_segmentation(
        cr_segmentation=getattr(args, "cr_segmentation", None),
        piecewise_cr=bool(args.piecewise_cr),
        cr_breakpoints=args.cr_breakpoints,
    )
    run_config = _make_run_config(
        degree=args.degree,
        alpha=float(args.alpha),
        psnr_min=args.psnr_min,
        ssim_min=args.ssim_min,
        phase3_span=args.phase3_span,
        cr_segmentation=cr_seg,
        piecewise_cr=(cr_seg == "auto"),
        cr_breakpoints=cr_bps,
    )
    _write_sweep_config(out_dir, run_config, train_points, phase3_points)

    print("=" * 72)
    print("Tradeoff sweep (train × phase3)")
    print("=" * 72)
    print("Field       : {}".format(stem))
    print("Surrogate: degree={}  alpha={}".format(args.degree, args.alpha))
    print("Constraints: PSNR>={}  SSIM>={}".format(args.psnr_min, args.ssim_min))
    print("Phase3 span: {}".format(args.phase3_span))
    _print_cr_segmentation(cr_seg, cr_bps, cr_segments_max=args.cr_segments_max)
    print("Train points : {}".format(train_points))
    print("Phase3 points: {}".format(phase3_points))
    print("Grid size    : {} cells".format(len(train_points) * len(phase3_points)))
    print("Resume       : {}".format(args.resume))
    print("Output       : {}".format(out_dir))
    print()

    if not args.skip_train_csvs:
        gen_train_csvs(
            train_points,
            stem=stem,
            sweeps_dir=os.path.abspath(args.sweeps_dir),
            input_path=os.path.abspath(args.input),
            train_jobs=args.train_sweep_jobs,
            out_dir=out_dir,
            resume=args.resume,
            pressio_bin=args.pressio,
            dims=dims,
            dtype=args.dtype,
            compressor=args.compressor,
        )

    run_tradeoff_grid(
        train_points,
        phase3_points,
        stem=stem,
        compressor=args.compressor,
        sweeps_dir=os.path.abspath(args.sweeps_dir),
        baseline_csv=os.path.abspath(args.baseline_csv),
        input_path=os.path.abspath(args.input),
        out_dir=out_dir,
        phase3_jobs=args.phase3_jobs,
        resume=args.resume,
        run_config=run_config,
        dims=dims,
    )
    print("\nTradeoff sweep finished. Plot with:")
    print("  python3 train_phase3_tradeoff.py --out {}".format(out_dir))


def _training_ebs(csv_path: str, train_sweep_n: int) -> list[float]:
    from surrogate_lasso import load_sweep_csv

    if os.path.isfile(csv_path):
        df = load_sweep_csv(csv_path)
        return [float(x) for x in df["error_bound"].values]
    if train_sweep_n > 0:
        return error_bounds_logspace(train_sweep_n, DEFAULT_EB_LO, DEFAULT_EB_HI)
    raise RuntimeError(
        "training CSV missing: {} — pass --train-sweep-n to create it".format(csv_path)
    )


def run_phase0_train_sweep(args, dims: tuple[int, ...], *, resume: bool = True) -> float:
    """Run pressio at every training error_bound; return wall-clock seconds."""
    from surrogate_lasso import load_sweep_csv

    csv_path = os.path.abspath(args.train_csv)
    ebs = _training_ebs(csv_path, args.train_sweep_n)
    if resume and os.path.isfile(csv_path):
        df = load_sweep_csv(csv_path)
        if len(df) == len(ebs):
            sidecar = _read_phase0_sidecar(csv_path)
            if not math.isnan(sidecar):
                print("Phase 0 — skip (resume, {} rows in {})".format(
                    len(df), csv_path))
                print("  using {:.4f}s from sidecar".format(sidecar))
                print()
                return sidecar
    print("=" * 72)
    print("Phase 0 — training pressio sweep ({} points -> {})".format(
        len(ebs), csv_path))
    print("  input : {}".format(args.input))
    print("  jobs  : {}".format(args.train_sweep_jobs))
    print("=" * 72)
    _t_phase0 = time.perf_counter()
    rows = run_pressio_on_ebs(
        pressio_bin=args.pressio,
        input_path=args.input,
        ebs=ebs,
        dims=dims,
        dtype=args.dtype,
        compressor=args.compressor,
        jobs=args.train_sweep_jobs,
        verbose=True,
    )
    phase0_sec = time.perf_counter() - _t_phase0
    if not rows:
        print("ERROR: Phase 0 returned no successful pressio runs.", file=sys.stderr)
        sys.exit(1)
    rows_by_key = {eb_key(float(r["error_bound"])): r for r in rows}
    write_csv(csv_path, rows_by_key, ebs)
    _write_phase0_sidecar(csv_path, phase0_sec)
    print("  [time] Phase 0 (train sweep): {:.4f}s".format(phase0_sec))
    print()
    return phase0_sec


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
    alpha: float | None = None,
    report_dir: str | None = None,
    df=None,
    *,
    cr_segmentation: str | None = None,
    piecewise_cr: bool = False,
    cr_segments_max: int = 5,
    cr_breakpoints: list[float] | None = None,
):
    """Phase 1: fit Lasso surrogates."""
    from surrogate_lasso import fit_surrogates_from_dataframe, load_sweep_csv

    if df is None:
        df = load_sweep_csv(csv_path)
    cr_seg, cr_bps = resolve_cr_segmentation(
        cr_segmentation=cr_segmentation,
        piecewise_cr=piecewise_cr,
        cr_breakpoints=cr_breakpoints,
    )
    bundle = fit_surrogates_from_dataframe(
        df, degree=degree, alpha=alpha, source_label=csv_path,
        cr_segmentation=cr_seg,
        cr_segments_max=cr_segments_max,
        cr_breakpoints=cr_bps,
    )
    if report_dir is not None:
        write_surrogate_report(bundle, report_dir)
    return bundle, df


def resolve_search_e_range(
    bundle,
    df_baseline,
    *,
    baseline_csv: str | None = None,
) -> tuple[float, float, str]:
    """Phase 2/3 search bounds: baseline min/max when loaded, else train CSV."""
    if df_baseline is not None and len(df_baseline) > 0:
        lo = float(df_baseline["error_bound"].min())
        hi = float(df_baseline["error_bound"].max())
        source = baseline_csv if baseline_csv else "baseline"
        return lo, hi, source
    return float(bundle.e_lo), float(bundle.e_hi), "train"


def run_phase2(
    bundle,
    psnr_min: float = DEFAULT_PSNR_MIN,
    ssim_min: float = DEFAULT_SSIM_MIN,
    iters: int = DEFAULT_LAGRANGIAN_ITERS,
    grid_n: int = DEFAULT_SURROGATE_GRID_N,
    verbose: bool = False,
    e_lo: float | None = None,
    e_hi: float | None = None,
):
    """Phase 2: quadratic-penalty solve (method = "lagrangian_penalty").

    Minimizes  -CR_hat(e) + mu*(viol_PSNR^2 + viol_SSIM^2)  on x = log10(e)
    via scipy.optimize.minimize_scalar, increasing mu until the minimizer
    settles on the constraint boundary. No QoI grid and no dense surrogate
    grid are used — purely the 1-D continuous solve.

    `iters`/`grid_n` are accepted for call-signature compatibility but unused.
    Optional ``e_lo``/``e_hi`` override the search interval (default: train CSV).
    """
    if e_lo is None:
        e_lo = bundle.e_lo
    if e_hi is None:
        e_hi = bundle.e_hi
    e_lo, e_hi = float(e_lo), float(e_hi)
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
    from surrogate_lasso import PiecewiseLassoSurrogate

    for s in (bundle.psnr, bundle.ssim, bundle.cr):
        if isinstance(s, PiecewiseLassoSurrogate):
            print("  {}  R2={:.4f}  RMSE={:.4g}  segments={}  BIC={:.4g}  ({})".format(
                s.name, s.r2, s.rmse, s.n_segments, s.bic, s.method))
            if s.breakpoints:
                print("    breakpoints: {}".format(
                    ", ".join("{:.6g}".format(b) for b in s.breakpoints)))
        else:
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
    ap.add_argument("--psnr-min", type=float, default=DEFAULT_PSNR_MIN)
    ap.add_argument("--ssim-min", type=float, default=DEFAULT_SSIM_MIN)
    ap.add_argument("--degree", type=int, default=DEFAULT_DEGREE,
                    help="Lasso polynomial degree in log10(e)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="fixed Lasso alpha (default: None = LOO-CV via LassoCV)")


def add_piecewise_cr_args(ap: argparse.ArgumentParser) -> None:
    """Optional piecewise CR surrogate (opt-in; default is global single segment)."""
    ap.add_argument(
        "--cr-segmentation",
        choices=("none", "auto", "fixed"),
        default=None,
        help="CR surrogate: none=global Lasso, auto=FMR-EM+BIC, fixed=--cr-breakpoints. "
             "If omitted, inferred from --piecewise-cr / --cr-breakpoints.",
    )
    ap.add_argument(
        "--piecewise-cr", action="store_true",
        help="shortcut for --cr-segmentation auto (legacy)",
    )
    ap.add_argument(
        "--cr-segments-max", type=int, default=2,
        help="exact number of CR segments when --cr-segmentation auto "
             "(scan breakpoints on training data; default: %(default)s)",
    )
    ap.add_argument(
        "--cr-breakpoints", nargs="+", type=float, default=None,
        help="error_bound breakpoints for --cr-segmentation fixed (legacy: alone implies fixed)",
    )


def resolve_cr_segmentation(
    *,
    cr_segmentation: str | None = None,
    piecewise_cr: bool = False,
    cr_breakpoints: list[float] | None = None,
) -> tuple[str, list[float] | None]:
    """Resolve CR piecewise mode and breakpoints from CLI / run config."""
    bps = [float(b) for b in cr_breakpoints] if cr_breakpoints else None
    if cr_segmentation is not None:
        seg = cr_segmentation
    elif bps:
        seg = "fixed"
    elif piecewise_cr:
        seg = "auto"
    else:
        seg = "none"

    if seg == "fixed" and not bps:
        raise SystemExit(
            "--cr-segmentation fixed requires one or more --cr-breakpoints")
    if seg == "auto" and bps:
        print(
            "WARNING: --cr-breakpoints ignored with --cr-segmentation auto",
            file=sys.stderr,
        )
        bps = None
    if seg == "none" and bps:
        print(
            "WARNING: --cr-breakpoints ignored unless --cr-segmentation fixed",
            file=sys.stderr,
        )
        bps = None
    return seg, bps


def _print_cr_segmentation(
    seg: str,
    bps: list[float] | None,
    *,
    cr_segments_max: int = 5,
) -> None:
    if seg == "fixed":
        print("CR surrogate: piecewise fixed (breakpoints: {})".format(
            ", ".join("{:.6g}".format(b) for b in (bps or []))))
    elif seg == "auto":
        print("CR surrogate: piecewise auto (scan breakpoints, K={})".format(
            cr_segments_max))
    else:
        print("CR surrogate: global (default)")


def _resolve_cr_segmentation(args) -> tuple[str, int, list[float] | None]:
    seg, bps = resolve_cr_segmentation(
        cr_segmentation=getattr(args, "cr_segmentation", None),
        piecewise_cr=bool(getattr(args, "piecewise_cr", False)),
        cr_breakpoints=getattr(args, "cr_breakpoints", None),
    )
    return seg, int(getattr(args, "cr_segments_max", 5)), bps


def _resolve_single_train_csv(args, stem: str, sweeps_dir: str) -> str:
    """Training sweep path for a single (non-tradeoff) run."""
    if not args.train_points:
        print("ERROR: --train-points N required (uses sweeps/{field}_trainN.csv).",
              file=sys.stderr)
        sys.exit(2)
    if len(args.train_points) > 1:
        print("ERROR: multiple --train-points requires --tradeoff-sweep.",
              file=sys.stderr)
        sys.exit(2)
    n_train = int(args.train_points[0])
    if args.train_sweep_n <= 0:
        args.train_sweep_n = n_train
    return _train_csv_path(sweeps_dir, n_train, stem)


def add_phase2_args(ap: argparse.ArgumentParser) -> None:
    add_model_training_args(ap)
    ap.add_argument("--iters", type=int, default=DEFAULT_LAGRANGIAN_ITERS,
                    help="augmented Lagrangian iterations")
    ap.add_argument("--grid-opt-n", type=int, default=DEFAULT_SURROGATE_GRID_N,
                    help="grid size for surrogate feasible maximum")


def random_candidates_near_e_star(
    e_star: float,
    n: int,
    log_span: float,
    e_lo: float,
    e_hi: float,
    seed: int | None = None,
) -> list[float]:
    """Random candidates around e* in log10-space, clipped to [e_lo, e_hi]."""
    e_center = float(np.clip(e_star, e_lo, e_hi))
    if n <= 1:
        return [e_center]

    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-log_span, log_span, max(0, n - 1))
    es = [float(np.clip(e_center * (10.0 ** d), e_lo, e_hi)) for d in offsets]
    # Always include center so we do not miss Phase-2 optimum.
    es.append(e_center)
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


def _model_eval_phase2(phase2: dict, df_oracle, df_train) -> dict:
    """Phase-2 optimum evaluated on oracle (or training) curve for MAPE."""
    e = float(phase2["e"])
    df = df_oracle if df_oracle is not None else df_train
    if df is not None:
        m = measured_at_e(df, e)
        return {"e": e, "cr": m["cr"], "psnr": m["psnr"], "ssim": m["ssim"]}
    return {
        "e": e,
        "cr": float(phase2["cr"]),
        "psnr": float(phase2.get("psnr", float("nan"))),
        "ssim": float(phase2.get("ssim", float("nan"))),
    }


def _model_eval_phase3(
    phase3_best: dict,
    df_oracle,
    *,
    use_pressio_meas: bool,
) -> dict:
    """Phase-3 MODEL FINAL for MAPE vs oracle."""
    e = float(phase3_best["error_bound"])
    if use_pressio_meas and phase3_best.get("calibration") == "pressio":
        return {
            "e": e,
            "cr": float(phase3_best["compression_ratio"]),
            "psnr": float(phase3_best["psnr"]),
            "ssim": float(phase3_best["ssim"]),
        }
    if df_oracle is not None:
        m = measured_at_e(df_oracle, e)
        return {"e": e, "cr": m["cr"], "psnr": m["psnr"], "ssim": m["ssim"]}
    return {
        "e": e,
        "cr": float(phase3_best["compression_ratio"]),
        "psnr": float(phase3_best.get("psnr", float("nan"))),
        "ssim": float(phase3_best.get("ssim", float("nan"))),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_model_training_args(ap)
    add_piecewise_cr_args(ap)
    ap.add_argument(
        "--baseline-csv", default=None,
        help="oracle baseline CSV; also sets Phase 2/3 search range to "
             "baseline error_bound min/max (surrogate still fit on train CSV)",
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
    ap.add_argument(
        "--train-sweep-n", type=int, default=0,
        help="Phase 0: N log-spaced points when train CSV does not exist yet",
    )
    ap.add_argument(
        "--train-sweep-jobs", type=int, default=None,
        help="parallel pressio workers for Phase-0 train sweep (default: min(32, cpu))",
    )
    ap.add_argument(
        "--skip-train-sweep", action="store_true",
        help="single run: skip Phase-0 pressio if train CSV exists (auto-runs Phase-0 "
             "when CSV missing); ignored with --tradeoff-sweep",
    )
    ap.add_argument(
        "--phase0-sec", type=float, default=None,
        help="override Phase-0 seconds when --skip-train-sweep (default: read sidecar)",
    )
    ap.add_argument("--input", default=None,
                    help="dataset path for pressio sweeps (Phase 0 train / baseline / Phase 3)")
    ap.add_argument(
        "--field", default=DEFAULT_FIELD,
        help="field stem e.g. CLOUDf01, CLOUDf04 (default paths under results/{field}/)",
    )
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
    ap.add_argument(
        "--phase3-points", nargs="+", type=int, default=None,
        help="Phase-3 pressio sample count (single run: one value, default 9; "
             "tradeoff: list of counts per grid cell)",
    )
    ap.add_argument("--phase3-span", type=float, default=0.25,
                    help="half-width in log10(e) for Phase-3 window")
    ap.add_argument("--phase3-jobs", type=int, default=None,
                    help="parallel pressio workers for Phase-3 pressio (default: min(32, cpu))")
    ap.add_argument("--phase3-random-seed", type=int, default=None,
                    help="optional RNG seed for random Phase-3 e candidates")
    ap.add_argument(
        "--phase3-pressio", action="store_true",
        help="deprecated no-op (Phase 3 always runs real pressio)",
    )
    ap.add_argument("--skip-phase3", action="store_true")
    ap.add_argument("--verify-pressio", action="store_true",
                    help="optional: run one pressio at final model e* (not part of model)")
    ap.add_argument(
        "--train-points", nargs="+", type=int, default=None,
        help="training point count (single run: one value; tradeoff: list; "
             "uses sweeps/{field}_train{N}.csv)",
    )
    ap.add_argument(
        "--points", nargs="+", type=int, default=None,
        help="set both --train-points and --phase3-points (tradeoff sweep only)",
    )
    ap.add_argument("--sweeps-dir", default=None,
                    help="directory for {field}_train{N}.csv (default: results/{field}/sweeps/)")
    ap.add_argument(
        "--tradeoff-sweep", action="store_true",
        help="run train×phase3 grid (use train_phase3_tradeoff.py to plot)",
    )
    ap.add_argument(
        "--skip-train-csvs", action="store_true",
        help="tradeoff: do not run Phase0 pressio (require existing train CSVs)",
    )
    ap.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="re-run Phase0 CSVs and grid cells even if outputs exist",
    )
    ap.set_defaults(resume=True)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    if args.tradeoff_sweep:
        _resolve_field_args(args, set_default_baseline=True)
        if args.train_sweep_jobs is None:
            args.train_sweep_jobs = max(1, min(96, os.cpu_count() or 1))
        if args.phase3_jobs is None:
            args.phase3_jobs = max(1, min(96, os.cpu_count() or 1))
        run_tradeoff_sweep(args)
        return

    stem = _resolve_field_args(args)
    phase3_n = _resolve_single_phase3_n(args)
    sweeps_dir = os.path.abspath(args.sweeps_dir)
    args.train_csv = _resolve_single_train_csv(args, stem, sweeps_dir)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    dims = tuple(args.dims)
    input_base = os.path.basename(args.input) if args.input else "n/a"

    if args.baseline_n is None:
        args.baseline_n = DEFAULT_BASELINE_N if args.input else 0
    if args.baseline_jobs is None:
        args.baseline_jobs = max(1, min(32, os.cpu_count() or 1))
    if args.train_sweep_jobs is None:
        args.train_sweep_jobs = max(1, min(32, os.cpu_count() or 1))
    if args.phase3_jobs is None:
        args.phase3_jobs = max(1, min(32, os.cpu_count() or 1))
    if not args.skip_phase3 and not args.input:
        print("ERROR: Phase 3 now always runs real pressio and requires --input.",
              file=sys.stderr)
        sys.exit(2)

    from surrogate_lasso import load_sweep_csv

    phase_times: dict[str, float | str] = {}
    phase0_sec = float("nan")
    csv_path = os.path.abspath(args.train_csv)
    run_phase0 = bool(args.input) and (
        not args.skip_train_sweep or not os.path.isfile(csv_path)
    )
    if args.skip_train_sweep and not os.path.isfile(csv_path):
        if args.input:
            print(
                "NOTE: training CSV missing ({}); running Phase-0 pressio.".format(
                    csv_path),
                file=sys.stderr,
            )
        else:
            print("ERROR: --skip-train-sweep but missing: {}".format(csv_path),
                  file=sys.stderr)
            sys.exit(2)

    if run_phase0:
        if not os.path.isfile(csv_path) and args.train_sweep_n <= 0:
            print("ERROR: training CSV missing; set --train-sweep-n or use --input.",
                  file=sys.stderr)
            sys.exit(2)
        phase0_sec = run_phase0_train_sweep(args, dims, resume=args.resume)
    elif args.skip_train_sweep:
        if not os.path.isfile(csv_path):
            print("ERROR: --skip-train-sweep but missing: {}".format(csv_path),
                  file=sys.stderr)
            sys.exit(2)
        if args.phase0_sec is not None:
            phase0_sec = float(args.phase0_sec)
        else:
            phase0_sec = _read_phase0_sidecar(csv_path)
            if math.isnan(phase0_sec):
                print("NOTE: Phase 0 skipped; no {} — phase0_sec=n/a".format(
                    _phase0_sidecar_path(csv_path)))
    elif args.phase0_sec is not None:
        phase0_sec = float(args.phase0_sec)

    phase_times["phase0_sec"] = phase0_sec

    df_train = load_sweep_csv(args.train_csv)
    eb_lo = float(df_train["error_bound"].min())
    eb_hi = float(df_train["error_bound"].max())

    baseline_csv = args.baseline_csv
    if args.baseline_n > 0:
        if not args.input:
            print("ERROR: --baseline-n requires --input for pressio runs.",
                  file=sys.stderr)
            sys.exit(2)
        if baseline_csv is None:
            baseline_csv = hp_baseline_csv(
                args.field, compressor=args.compressor, n=args.baseline_n,
            )
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
    print("Three-phase MODEL (Phase 0 optional + Phase 1 + 2 + 3)")
    print("=" * 72)
    print("Train CSV   : {}".format(args.train_csv))
    if args.input and not args.skip_train_sweep:
        print("Phase 0     : auto (re-run {} training pressio, timed)".format(
            len(df_train)))
    elif run_phase0:
        print("Phase 0     : auto (generated missing training CSV, timed)")
    elif args.skip_train_sweep:
        sidecar = _phase0_sidecar_path(csv_path)
        if not math.isnan(phase0_sec):
            print("Phase 0     : skipped (using {:.4f}s from {})".format(
                phase0_sec, sidecar if os.path.isfile(sidecar) else "--phase0-sec"))
        else:
            print("Phase 0     : skipped (--skip-train-sweep, no timing recorded)")
    elif args.phase0_sec is not None:
        print("Phase 0     : recorded {:.4f}s (--phase0-sec)".format(args.phase0_sec))
    if args.input:
        print("Dataset     : {}".format(args.input))
    print("Compressor  : {}".format(args.compressor))
    if args.baseline_n > 0:
        print("Baseline    : {} pressio samples -> {}".format(
            args.baseline_n, baseline_csv))
        print("  jobs={}  resume={}".format(
            args.baseline_jobs, not args.skip_baseline_sweep))
    print("Constraints : PSNR >= {:.4g} dB,  SSIM >= {:.4g}".format(
        args.psnr_min, args.ssim_min))
    print("Surrogate   : degree={}  alpha={}".format(
        args.degree,
        args.alpha if args.alpha is not None else "LassoCV (auto)"))
    cr_seg, cr_kmax, cr_bps = _resolve_cr_segmentation(args)
    _print_cr_segmentation(cr_seg, cr_bps, cr_segments_max=cr_kmax)
    phase3_desc = "pressio ({} random samples, span={:.3g}, jobs={})".format(
        phase3_n, args.phase3_span, args.phase3_jobs
    )
    if args.phase3_random_seed is not None:
        phase3_desc += ", seed={}".format(args.phase3_random_seed)
    print("Phase 3     : {}".format(phase3_desc))
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

    # Wall-clock timing of model phases (plots/IO excluded).

    # ----- Phase 1 -----
    print("### Phase 1: Lasso surrogates from sweep CSV ###")
    phase1_dir = os.path.join(out_dir, "phase1")
    report_dir = phase1_dir if not args.skip_phase_plots else out_dir
    _t_phase = time.perf_counter()
    bundle, df = run_phase1(
        args.train_csv, degree=args.degree, alpha=args.alpha,
        report_dir=report_dir, df=df_train,
        cr_segmentation=cr_seg,
        cr_segments_max=cr_kmax,
        cr_breakpoints=cr_bps,
    )
    phase_times["phase1_sec"] = time.perf_counter() - _t_phase
    print("  [time] Phase 1: {:.4f}s".format(phase_times["phase1_sec"]))
    fit_e_lo, fit_e_hi = bundle.e_lo, bundle.e_hi
    search_e_lo, search_e_hi, search_src = resolve_search_e_range(
        bundle, df_baseline, baseline_csv=baseline_csv,
    )
    if (search_e_lo, search_e_hi) != (fit_e_lo, fit_e_hi):
        print("  Surrogate fit range (train): [{:g}, {:g}]".format(
            fit_e_lo, fit_e_hi))
        print("  Phase 2/3 search range ({}): [{:g}, {:g}]".format(
            search_src, search_e_lo, search_e_hi))
    else:
        print("  e range (train + search): [{:g}, {:g}]".format(
            search_e_lo, search_e_hi))
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
    _t_phase = time.perf_counter()
    p2 = run_phase2(
        bundle, args.psnr_min, args.ssim_min,
        iters=args.iters, grid_n=args.grid_opt_n, verbose=True,
        e_lo=search_e_lo, e_hi=search_e_hi,
    )
    phase_times["phase2_sec"] = time.perf_counter() - _t_phase
    print("  [time] Phase 2: {:.4f}s".format(phase_times["phase2_sec"]))
    phase2 = log_phase2_result(p2, bundle, search_e_lo, search_e_hi)
    if not args.skip_phase_plots:
        from plot_phase2_lagrangian import generate_phase2_plots
        phase2_dir = os.path.join(out_dir, "phase2")
        generate_phase2_plots(
            bundle, df, p2,
            args.psnr_min, args.ssim_min, phase2_dir,
            n_grid=args.plot_grid_n,
            csv_path=args.train_csv,
            degree=args.degree,
            iters=args.iters,
            grid_opt_n=args.grid_opt_n,
            baseline_df=df_baseline,
        )
        print("  Phase-2 plots -> {}".format(phase2_dir))

    _t_phase = time.perf_counter()
    e_star = phase2["e"]
    e_candidates = random_candidates_near_e_star(
        e_star,
        phase3_n,
        args.phase3_span,
        search_e_lo,
        search_e_hi,
        seed=args.phase3_random_seed,
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
    else:
        print("### Phase 3: pressio wise search near e*={:.6g} ###".format(e_star))
        print("  {} random pressio runs in span={:.3g}  jobs={}".format(
            len(e_candidates), args.phase3_span, args.phase3_jobs))
        if args.phase3_random_seed is not None:
            print("  random seed: {}".format(args.phase3_random_seed))
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

    phase_times["phase3_sec"] = time.perf_counter() - _t_phase
    phase_times["phase3_mode"] = (
        "skipped" if args.skip_phase3
        else "pressio"
    )
    print("  [time] Phase 3 ({}): {:.4f}s".format(
        phase_times["phase3_mode"], phase_times["phase3_sec"]))

    total_phase_sec = (
        phase_times["phase1_sec"] + phase_times["phase2_sec"]
        + phase_times["phase3_sec"]
    )
    phases12_sec = (
        float(phase_times["phase1_sec"]) + float(phase_times["phase2_sec"])
    )
    phase_times["phases12_sec"] = phases12_sec
    phase_times["phases123_sec"] = total_phase_sec
    phase_times["total_sec"] = total_phase_sec
    all_phases_sec = float("nan")
    if not math.isnan(float(phase_times["phase0_sec"])):
        all_phases_sec = float(phase_times["phase0_sec"]) + total_phase_sec
    phase_times["all_phases_sec"] = all_phases_sec

    print("\n### Phase timing summary ###")
    p0 = float(phase_times["phase0_sec"])
    if not math.isnan(p0):
        print("  Phase 0 (train sweep)     : {:.4f}s".format(p0))
    else:
        print("  Phase 0 (train sweep)     : n/a")
    print("  Phase 1 (surrogate fit)   : {:.4f}s".format(phase_times["phase1_sec"]))
    print("  Phase 2 (penalty solve)   : {:.4f}s".format(phase_times["phase2_sec"]))
    print("  Phase 3 ({:>7s} search) : {:.4f}s".format(
        phase_times["phase3_mode"], phase_times["phase3_sec"]))
    print("  Total without Phase 3     : {:.4f}s  (phases 1+2)".format(phases12_sec))
    print("  Total with Phase 3        : {:.4f}s  (phases 1+2+3)".format(total_phase_sec))
    if not math.isnan(all_phases_sec):
        print("  Total all phases          : {:.4f}s  (phase0 + phases 1+2+3)".format(
            all_phases_sec))

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
    mape_block_p2: list[str] = []
    mape_block_p3: list[str] = []
    if oracle and phase2 is not None:
        model_p2 = _model_eval_phase2(phase2, df_oracle, df)
        mape_p2 = mape_model_vs_oracle(oracle, model_p2)
        mape_block_p2 = format_mape_block(
            oracle, model_p2, mape_p2,
            title="Phase 2 vs oracle MAPE (without Phase 3 search)",
            model_label="MODEL at Phase-2 e* (on oracle curve)",
        )
        print("\n### Phase 2 vs oracle MAPE (without Phase 3) ###")
        for line in format_oracle_header(oracle, n_grid=len(df_oracle)):
            print(line)
        for line in mape_block_p2:
            print(line)
    if oracle and phase3_best:
        use_pressio = bool(phase3_best.get("calibration") == "pressio")
        model_p3 = _model_eval_phase3(
            phase3_best, df_oracle, use_pressio_meas=use_pressio,
        )
        mape_p3 = mape_model_vs_oracle(oracle, model_p3)
        mape_block_p3 = format_mape_block(
            oracle, model_p3, mape_p3,
            title="Phase 3 vs oracle MAPE (with Phase 3 search, MODEL FINAL)",
            model_label="MODEL FINAL at Phase-3 e*",
        )
        print("\n### Phase 3 vs oracle MAPE (MODEL FINAL) ###")
        if not mape_block_p2:
            for line in format_oracle_header(oracle, n_grid=len(df_oracle)):
                print(line)
        for line in mape_block_p3:
            print(line)
    # Legacy single block alias for tools expecting "Validation — MODEL vs BASELINE"
    mape_block = mape_block_p3

    # ----- Summary files -----
    summary_path = os.path.join(out_dir, "summary.txt")
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_lines = [
        "Three-phase MODEL output",
        "=" * 72,
        "(Phase 1 + 2 + 3 are the model; Phase 3 below is MODEL FINAL)",
        "generated_at : {}".format(generated_at),
        "input_file : {}".format(input_base),
        "sweep_csv  : {}".format(args.train_csv),
    ]
    if args.baseline_csv:
        summary_lines.append("baseline_csv : {}".format(args.baseline_csv))
    summary_lines += [
        "PSNR_min   : {:.6g}".format(args.psnr_min),
        "SSIM_min   : {:.6g}".format(args.ssim_min),
        "e_range_train  : [{:g}, {:g}]".format(fit_e_lo, fit_e_hi),
        "e_range_search : [{:g}, {:g}]".format(search_e_lo, search_e_hi),
        "",
        "Phase 1 — surrogates (Lasso, degree={}, alpha={})".format(
            args.degree,
            args.alpha if args.alpha is not None else "LassoCV"),
    ]
    from surrogate_lasso import PiecewiseLassoSurrogate
    summary_lines.append("  cr_segmentation : {}".format(cr_seg))
    if cr_seg == "auto":
        summary_lines.append("  cr_segments     : {}".format(cr_kmax))
    if cr_bps:
        summary_lines.append("  cr_breakpoints  : {}".format(
            ", ".join("{:.6g}".format(b) for b in cr_bps)))
    elif cr_seg == "auto" and isinstance(bundle.cr, PiecewiseLassoSurrogate):
        if bundle.cr.breakpoints:
            summary_lines.append("  cr_breakpoints  : {}".format(
                ", ".join("{:.6g}".format(b) for b in bundle.cr.breakpoints)))
    for s in (bundle.psnr, bundle.ssim, bundle.cr):
        if isinstance(s, PiecewiseLassoSurrogate):
            summary_lines.append(
                "  {} R2={:.6f} RMSE={:.6g} segments={} BIC={:.6g}".format(
                    s.name, s.r2, s.rmse, s.n_segments, s.bic))
        else:
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

    summary_lines.append("")
    summary_lines.append("Phase timing (wall-clock, plots/IO excluded):")
    p0 = float(phase_times["phase0_sec"])
    if not math.isnan(p0):
        summary_lines.append("  Phase 0 (train sweep)   = {:.4f} s".format(p0))
    else:
        summary_lines.append("  Phase 0 (train sweep)   = n/a")
    summary_lines.append("  Phase 1 (surrogate fit) = {:.4f} s".format(
        phase_times["phase1_sec"]))
    summary_lines.append("  Phase 2 (penalty solve) = {:.4f} s".format(
        phase_times["phase2_sec"]))
    summary_lines.append("  Phase 3 ({} search) = {:.4f} s".format(
        phase_times["phase3_mode"], phase_times["phase3_sec"]))
    summary_lines.append("  Total without Phase 3   = {:.4f} s  (phases 1+2)".format(
        phase_times["phases12_sec"]))
    summary_lines.append("  Total with Phase 3      = {:.4f} s  (phases 1+2+3)".format(
        phase_times["phases123_sec"]))
    if not math.isnan(float(phase_times["all_phases_sec"])):
        summary_lines.append("  Total all phases        = {:.4f} s  (phase0 + 1+2+3)".format(
            phase_times["all_phases_sec"]))

    if verify_row:
        summary_lines.append("")
        summary_lines.append("Optional pressio verify (not part of model):")
        for k, v in verify_row.items():
            summary_lines.append("  {:18s} = {}".format(k, v))

    if mape_block_p2 or mape_block_p3:
        summary_lines.append("")
        if mape_block_p2 and mape_block_p3:
            summary_lines.extend(format_oracle_header(oracle, n_grid=len(df_oracle)))
            summary_lines.append("")
        if mape_block_p2:
            summary_lines.extend(mape_block_p2)
            summary_lines.append("")
        if mape_block_p3:
            summary_lines.extend(mape_block_p3)
    elif mape_block:
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

    timing_csv = os.path.join(out_dir, "phase_timings.csv")
    with open(timing_csv, "w", newline="") as tf:
        w = csv.writer(tf)
        w.writerow(["input_file", "phase", "mode", "seconds"])
        if not math.isnan(p0):
            w.writerow([input_base, "phase0", "train_sweep", "{:.6f}".format(p0)])
        w.writerow([input_base, "phase1", "fit", "{:.6f}".format(phase_times["phase1_sec"])])
        w.writerow([input_base, "phase2", "penalty_solve", "{:.6f}".format(phase_times["phase2_sec"])])
        w.writerow([input_base, "phase3", phase_times["phase3_mode"], "{:.6f}".format(phase_times["phase3_sec"])])
        w.writerow([input_base, "total", "without_phase3", "{:.6f}".format(phase_times["phases12_sec"])])
        w.writerow([input_base, "total", "with_phase3", "{:.6f}".format(phase_times["phases123_sec"])])
        w.writerow([input_base, "total", "3_phases", "{:.6f}".format(phase_times["phases123_sec"])])
        if not math.isnan(float(phase_times["all_phases_sec"])):
            w.writerow([input_base, "total", "all_phases", "{:.6f}".format(
                phase_times["all_phases_sec"])])

    plot_overview(
        bundle, args.psnr_min, args.ssim_min, phase2, grid_best,
        os.path.join(out_dir, "feasible_surrogate.png"), fit_e_lo, fit_e_hi,
    )

    print("\nWrote results to: {}".format(out_dir))
    for n in sorted(os.listdir(out_dir)):
        print("  -", n)


if __name__ == "__main__":
    main()
