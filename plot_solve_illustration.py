#!/usr/bin/env python3
"""
"Solve" illustration plot for the three-phase optimizer (collaborator's Fig. 2 style).

For one optimizer run / tradeoff cell it draws, over candidate error_bound:
  - blue  : true 10k-oracle curve (measured ground truth)
  - green : model surrogate prediction (Phase-1 Lasso, refit identically)
  - orange: training pressio points  (real compression + QoI evaluations)
  - purple: Phase-3 pressio points   (real compression + QoI evaluations;
            open marker = infeasible measured)
  - green dashed verticals : Phase-3 refinement window around Phase-2 e*
  - gold solid vertical    : final chosen e*
  - blue diamond           : 10k baseline oracle solution (best feasible CR on 10k grid)
  - black horizontal       : constraint target (PSNR_min / SSIM_min)

Two stacked panels share the x-axis: a QoI panel (PSNR, with the target line)
on top and the maximized CR on the bottom (add SSIM with --metrics).

Field-driven: --field selects which field's results/baseline to read via
hurricane_paths, and output goes to a per-field gallery folder, so CLOUDf01 and
CLOUDf04 never mix.

Phase-1 CR surrogate refit uses the same piecewise settings recorded by each
cell's three_phase_optimize.py run (summary.txt + cell run_config.json). The
plot script does not re-scan breakpoints.

Examples:
    cd /anvil/projects/x-cis240669/optimizer
    python3 plot_solve_illustration.py --field CLOUDf04 --top-n 3
    python3 plot_solve_illustration.py --field CLOUDf01 --top-n 3
    python3 plot_solve_illustration.py --field CLOUDf04 --headline  # optional single run
    python3 plot_solve_illustration.py --field CLOUDf04 \\
      --cell /anvil/projects/x-cis240669/Hurricane/results/CLOUDf04/tradeoff_sweep/train24_phase3120
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_PRESSIO_VIEW_LIB = os.path.join(
    _REPO_ROOT, "libpressio-env", ".spack-env", "view", "lib",
)


def _configure_runtime_env() -> None:
    """Match other plot scripts: avoid spack numpy on PYTHONPATH; set pressio libs."""
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONPATH", None)
    user = os.environ.get("USER", "solveplot")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-{}".format(user))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if os.path.isdir(_PRESSIO_VIEW_LIB):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in ld.split(":") if p]
        if _PRESSIO_VIEW_LIB not in parts:
            os.environ["LD_LIBRARY_PATH"] = _PRESSIO_VIEW_LIB + (
                ":" + ld if ld else ""
            )


_configure_runtime_env()

import argparse
import csv
import json
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from surrogate_lasso import PiecewiseLassoSurrogate  # noqa: E402

from hurricane_paths import (  # noqa: E402
    DEFAULT_FIELD,
    baseline_csv as hp_baseline_csv,
    normalize_compressor,
    normalize_field_stem,
    results_dir as hp_results_dir,
    three_phase_out as hp_three_phase_out,
    tradeoff_out as hp_tradeoff_out,
    train_csv as hp_train_csv,
    train_sweep_csv as hp_train_sweep_csv,
)
from three_phase_optimize import run_phase1  # noqa: E402
from validation_metrics import mape_model_vs_oracle, pressio_oracle_grid  # noqa: E402

# Color scheme (mirrors the collaborator's figure legend)
C_ORACLE = "#1f77b4"   # blue  : true curve
C_MODEL = "#2ca02c"    # green : surrogate fit + refinement window
C_TRAIN = "#ff7f0e"    # orange: initial / training pressio points
C_PHASE3 = "#9467bd"   # purple: refinement pressio points
C_FINAL = "#d4a017"    # gold  : final solution

CELL_RE = re.compile(r"^train(\d+)_phase3(\d+)$")

# metric -> (measured column, surrogate attr, axis label, threshold key or None)
METRIC_SPECS = {
    "psnr": ("psnr", "psnr", "PSNR (dB)", "psnr_min"),
    "ssim": ("ssim", "ssim", "SSIM", "ssim_min"),
    "cr": ("compression_ratio", "cr", "Compression Ratio", None),
}


# ---------------------------------------------------------------------------
# summary.txt parsing
# ---------------------------------------------------------------------------
def _search(pattern, text, *, group=1, flags=0, cast=float):
    m = re.search(pattern, text, flags)
    if not m:
        return None
    try:
        return cast(m.group(group))
    except (TypeError, ValueError):
        return None


def parse_summary(cell_dir: str) -> dict:
    path = os.path.join(cell_dir, "summary.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError("no summary.txt in {}".format(cell_dir))
    text = open(path).read()

    info: dict = {
        "cell_dir": cell_dir,
        "sweep_csv": _search(r"sweep_csv\s*:\s*(\S+)", text, cast=str),
        "baseline_csv": _search(r"baseline_csv\s*:\s*(\S+)", text, cast=str),
        "psnr_min": _search(r"PSNR_min\s*:\s*([\d.eE+\-]+)", text),
        "ssim_min": _search(r"SSIM_min\s*:\s*([\d.eE+\-]+)", text),
        "degree": _search(r"degree=(\d+)", text, cast=int),
        "alpha": _search(r"alpha=([\d.eE+\-]+)", text),
        "cr_segmentation": _search(r"cr_segmentation\s*:\s*(\w+)", text, cast=str),
        "cr_segments": _search(r"cr_segments(?:_max)?\s*:\s*(\d+)", text, cast=int),
    }

    bps_raw = _search(r"cr_breakpoints\s*:\s*([\d.eE+\-,\s]+)", text, cast=str)
    info["cr_breakpoints"] = None
    if bps_raw:
        vals = [float(x) for x in re.split(r"[,\s]+", bps_raw.strip()) if x]
        info["cr_breakpoints"] = vals or None

    # Phase 2 e* (penalty solve block only; avoid header "Phase 2" / training-grid e)
    idx2 = text.find("Phase 2 — quadratic-penalty")
    if idx2 < 0:
        idx2 = text.find("Phase 2 —")
    idxf = text.find("MODEL FINAL — Phase 3")
    if idxf < 0:
        idxf = text.find("MODEL FINAL —")
    phase2_txt = text[idx2:idxf] if (idx2 >= 0 and idxf > idx2) else ""
    info["phase2_e"] = _search(r"^\s*e\s*=\s*([\d.eE+\-]+)", phase2_txt, flags=re.M)

    # MODEL FINAL block (stop before the timing section)
    idx_t = text.find("Phase timing")
    final_txt = text[idxf:idx_t] if (idxf >= 0 and idx_t > idxf) else text[idxf:]
    info["final_e"] = _search(r"error_bound\s*=\s*([\d.eE+\-]+)", final_txt)
    info["final_cr"] = _search(r"compression_ratio\s*=\s*([\d.eE+\-]+)", final_txt)
    info["final_psnr"] = _search(r"\bpsnr\s*=\s*([\d.eE+\-]+)", final_txt)
    info["final_ssim"] = _search(r"\bssim\s*=\s*([\d.eE+\-]+)", final_txt)

    m = CELL_RE.match(os.path.basename(cell_dir.rstrip("/")))
    info["n_train"] = int(m.group(1)) if m else None
    info["n_phase3"] = int(m.group(2)) if m else None
    return info


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def _load_csv(path: str):
    if path and os.path.isfile(path):
        return pd.read_csv(path)
    return None


def _load_cell_run_config(cell_dir: str) -> dict | None:
    """Per-cell run_config.json written by three_phase_optimize tradeoff cells."""
    path = os.path.join(cell_dir, "run_config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _segmentation_from_run_config(cfg: dict) -> str:
    """Same rules as three_phase_optimize._cr_segmentation_from_config."""
    if cfg.get("cr_segmentation"):
        return str(cfg["cr_segmentation"])
    if cfg.get("cr_breakpoints"):
        return "fixed"
    if cfg.get("piecewise_cr"):
        return "auto"
    return "none"


def _resolve_cr_fit_kwargs(info: dict) -> dict:
    """Refit CR surrogate exactly as three_phase_optimize recorded for this cell.

    Breakpoints and segment count come from summary.txt (authoritative after a
    run). Cell run_config.json fills in only missing segmentation metadata.
    Never re-runs auto RSS scan and never reads parent tradeoff_run_config.json.
    """
    cell_dir = info["cell_dir"]
    seg = info.get("cr_segmentation")
    bps = list(info["cr_breakpoints"]) if info.get("cr_breakpoints") else None
    n_seg = info.get("cr_segments")

    cfg = _load_cell_run_config(cell_dir)
    if cfg:
        if not seg:
            seg = _segmentation_from_run_config(cfg)
        if not bps and cfg.get("cr_breakpoints"):
            bps = [float(b) for b in cfg["cr_breakpoints"]]

    if not seg:
        seg = "fixed" if bps else "none"

    if seg == "none":
        return {
            "cr_segmentation": "none",
            "_optimize_cr": {"mode": "none", "segments": 1, "breakpoints": []},
        }

    if seg == "fixed":
        if not bps:
            raise ValueError(
                "{}: cr_segmentation=fixed but cr_breakpoints missing in "
                "summary.txt / run_config.json".format(cell_dir))
        return {
            "cr_segmentation": "fixed",
            "cr_breakpoints": bps,
            "_optimize_cr": {"mode": "fixed", "segments": len(bps) + 1, "breakpoints": bps},
        }

    if seg == "auto":
        if not bps:
            raise ValueError(
                "{}: cr_segmentation=auto but cr_breakpoints missing in summary.txt "
                "(re-run three_phase_optimize; plot does not re-scan)".format(cell_dir))
        return {
            # Optimize already scanned; refit at recorded breakpoints (no second scan).
            "cr_segmentation": "fixed",
            "cr_breakpoints": bps,
            "_optimize_cr": {
                "mode": "auto",
                "segments": n_seg or (len(bps) + 1),
                "breakpoints": bps,
            },
        }

    raise ValueError(
        "{}: unknown cr_segmentation {!r}".format(cell_dir, seg))


def _resolve_train_csv(info: dict, field: str, compressor: str = "sz3") -> str:
    """Find the training sweep CSV, tolerating stale paths in old summaries."""
    candidates = [info.get("sweep_csv")]
    n_train = info.get("n_train")
    if n_train:
        candidates.append(hp_train_csv(field, n_train, compressor=compressor))
    candidates.append(hp_train_sweep_csv(field, compressor=compressor))
    # also try a same-named file inside the cell dir
    if info.get("sweep_csv"):
        candidates.append(os.path.join(info["cell_dir"],
                                       os.path.basename(info["sweep_csv"])))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "training sweep CSV missing (tried: {})".format(
            ", ".join(str(c) for c in candidates if c)))


def load_cell_data(
    info: dict, *, field: str, span: float, compressor: str = "sz3",
) -> dict:
    sweep_csv = _resolve_train_csv(info, field, compressor=compressor)

    baseline = info.get("baseline_csv")
    if not baseline or not os.path.isfile(baseline):
        baseline = hp_baseline_csv(field, compressor=compressor)
    df_oracle = _load_csv(baseline)
    df_train = pd.read_csv(sweep_csv)
    df_phase3 = _load_csv(os.path.join(info["cell_dir"], "phase3_runs.csv"))

    cr_kw = _resolve_cr_fit_kwargs(info)
    opt_cr = cr_kw.pop("_optimize_cr", {})
    bundle, _ = run_phase1(
        sweep_csv,
        degree=info["degree"] or 2,
        alpha=info["alpha"],
        **cr_kw,
    )

    # Refinement window around Phase-2 e* (fall back to phase-3 spread)
    e2 = info.get("phase2_e")
    lo_w = hi_w = None
    if e2:
        lo_w = max(bundle.e_lo, e2 / (10.0 ** span))
        hi_w = min(bundle.e_hi, e2 * (10.0 ** span))
    elif df_phase3 is not None and len(df_phase3):
        lo_w = float(df_phase3["error_bound"].min())
        hi_w = float(df_phase3["error_bound"].max())

    return {
        "bundle": bundle,
        "df_oracle": df_oracle,
        "df_train": df_train,
        "df_phase3": df_phase3,
        "window": (lo_w, hi_w),
        "optimize_cr": opt_cr,
        "oracle": _oracle_solution(info, df_oracle),
    }


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _ylim_from(values, pad=0.08):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1.0
    m = (hi - lo) * pad
    return lo - m, hi + m


def _plot_surrogate_curve(ax, surr, xlim_log, *, color, label, zorder, n_grid=400):
    """Draw surrogate; piecewise CR is evaluated per segment (not one global grid)."""
    e_lo = 10.0 ** xlim_log[0]
    e_hi = 10.0 ** xlim_log[1]
    if isinstance(surr, PiecewiseLassoSurrogate):
        n_seg = max(1, surr.n_segments)
        n_per = max(32, n_grid // n_seg)
        for i, (seg_lo, seg_hi, seg) in enumerate(surr.segments):
            slo = max(float(seg_lo), e_lo)
            shi = min(float(seg_hi), e_hi)
            if slo >= shi:
                continue
            xg = np.linspace(math.log10(slo), math.log10(shi), n_per)
            ax.plot(
                xg, seg(np.power(10.0, xg)), color=color, lw=2.2,
                label=label if i == 0 else None, zorder=zorder,
            )
        for bp in surr.breakpoints:
            lbp = math.log10(bp)
            if xlim_log[0] <= lbp <= xlim_log[1]:
                ax.axvline(
                    lbp, color=color, ls=":", lw=1.0, alpha=0.65, zorder=zorder - 1,
                    label="CR segment break" if bp == surr.breakpoints[0] else None,
                )
        xg = np.linspace(xlim_log[0], xlim_log[1], n_grid)
        return np.asarray(surr(np.power(10.0, xg)), dtype=float)
    xg = np.linspace(xlim_log[0], xlim_log[1], n_grid)
    yg = np.asarray(surr(np.power(10.0, xg)), dtype=float)
    ax.plot(xg, yg, color=color, lw=2.2, label=label, zorder=zorder)
    return yg


def _final_solution(info: dict, data: dict) -> dict:
    """MODEL FINAL measured values — same source as three_phase_optimize MAPE block."""
    out = {
        "final_e": info.get("final_e"),
        "final_cr": info.get("final_cr"),
        "final_psnr": info.get("final_psnr"),
        "final_ssim": info.get("final_ssim"),
    }
    df_p = data.get("df_phase3")
    fe = out["final_e"]
    if fe is None or df_p is None or not len(df_p):
        return out
    row = df_p.iloc[(df_p["error_bound"] - fe).abs().argsort()[0]]
    if abs(float(row["error_bound"]) - fe) / max(fe, 1e-30) > 1e-6:
        return out
    p3 = {
        "final_e": float(row["error_bound"]),
        "final_cr": float(row["compression_ratio"]),
        "final_psnr": float(row["psnr"]),
        "final_ssim": float(row["ssim"]),
    }
    for key in ("final_cr", "final_psnr", "final_ssim"):
        a, b = out.get(key), p3.get(key)
        if a is not None and b is not None and abs(a - b) > max(1e-6, 1e-4 * abs(b)):
            print(
                "  WARN {}: summary={} phase3_runs={}".format(key, a, b),
                file=sys.stderr,
            )
    return p3


def _oracle_solution(info: dict, df_oracle) -> dict | None:
    """Best feasible point on the 10k baseline sweep (MAPE oracle)."""
    if df_oracle is None or not len(df_oracle):
        return None
    psnr_min = info.get("psnr_min")
    ssim_min = info.get("ssim_min")
    if psnr_min is None or ssim_min is None:
        return None
    oracle = pressio_oracle_grid(df_oracle, float(psnr_min), float(ssim_min))
    if oracle is None:
        return None
    return {
        "oracle_e": oracle["e"],
        "oracle_cr": oracle["cr"],
        "oracle_psnr": oracle["psnr"],
        "oracle_ssim": oracle["ssim"],
    }


def _model_mape_vs_oracle(final: dict, oracle: dict | None) -> dict | None:
    """MAPE (%) of MODEL FINAL vs 10k oracle — same as three_phase_optimize Phase 3 block."""
    if not oracle or not final.get("final_e"):
        return None
    model = {
        "e": final["final_e"],
        "cr": final.get("final_cr"),
        "psnr": final.get("final_psnr"),
        "ssim": final.get("final_ssim"),
    }
    if any(model[k] is None for k in ("cr", "psnr", "ssim")):
        return None
    o = {
        "e": oracle["oracle_e"],
        "cr": oracle["oracle_cr"],
        "psnr": oracle["oracle_psnr"],
        "ssim": oracle["oracle_ssim"],
    }
    return mape_model_vs_oracle(o, model)


def _mape_annotation_text(mape: dict | None) -> str | None:
    if not mape:
        return None
    lines = []
    for key, label in (("cr", "CR"), ("psnr", "PSNR"), ("ssim", "SSIM")):
        v = mape.get(key)
        if v is not None and np.isfinite(v):
            lines.append("{} {:6.2f}%".format(label, v))
    if not lines:
        return None
    return "MAPE vs 10k oracle:\n" + "\n".join(lines)


def _plot_panel(ax, metric, info, data, *, xlim_log, zoom):
    meas_col, surr_attr, ylabel, thr_key = METRIC_SPECS[metric]
    bundle = data["bundle"]
    surr = getattr(bundle, surr_attr)

    e_lo = 10.0 ** xlim_log[0]
    e_hi = 10.0 ** xlim_log[1]
    yg = _plot_surrogate_curve(
        ax, surr, xlim_log, color=C_MODEL, label="model surrogate", zorder=3,
    )

    y_for_lim = list(yg)

    # blue: true oracle curve
    df_o = data["df_oracle"]
    if df_o is not None and meas_col in df_o:
        sel = (df_o["error_bound"] >= e_lo) & (df_o["error_bound"] <= e_hi)
        dfo = df_o[sel].sort_values("error_bound")
        if len(dfo):
            ax.plot(np.log10(dfo["error_bound"]), dfo[meas_col],
                    color=C_ORACLE, lw=1.4, alpha=0.85,
                    label="true (10k oracle)", zorder=2)
            if zoom:
                y_for_lim += list(dfo[meas_col].values)

    # green: model surrogate prediction (done above via _plot_surrogate_curve)

    # orange: training pressio points
    df_t = data["df_train"]
    if df_t is not None and meas_col in df_t:
        ax.scatter(np.log10(df_t["error_bound"]), df_t[meas_col],
                   s=55, c=C_TRAIN, edgecolor="k", linewidth=0.5, zorder=4,
                   label="training pressio ({})".format(len(df_t)))
        if zoom:
            sel = (df_t["error_bound"] >= e_lo) & (df_t["error_bound"] <= e_hi)
            y_for_lim += list(df_t[sel][meas_col].values)

    # purple: Phase-3 pressio points (open marker = infeasible measured)
    df_p = data["df_phase3"]
    if df_p is not None and meas_col in df_p:
        feas = df_p.get("feasible_measured")
        if feas is not None:
            fe = df_p[feas.astype(str).str.lower() == "true"]
            inf = df_p[feas.astype(str).str.lower() != "true"]
        else:
            fe, inf = df_p, df_p.iloc[0:0]
        if len(fe):
            ax.scatter(np.log10(fe["error_bound"]), fe[meas_col],
                       s=42, c=C_PHASE3, edgecolor="k", linewidth=0.4, zorder=5,
                       label="Phase-3 pressio (feasible)")
        if len(inf):
            ax.scatter(np.log10(inf["error_bound"]), inf[meas_col],
                       s=42, facecolors="none", edgecolors=C_PHASE3,
                       linewidth=1.0, zorder=5, label="Phase-3 pressio (infeasible)")
        y_for_lim += list(df_p[meas_col].values)

    # green dashed: refinement window
    lo_w, hi_w = data["window"]
    for b, lab in ((lo_w, "Phase-3 window"), (hi_w, None)):
        if b and e_lo <= b <= e_hi:
            ax.axvline(math.log10(b), color=C_MODEL, ls="--", lw=1.2, alpha=0.8,
                       label=lab, zorder=2)

    final = data.get("final") or _final_solution(info, data)
    oracle = data.get("oracle")

    # blue dashed: 10k oracle e*
    oe = oracle.get("oracle_e") if oracle else None
    if oe and e_lo <= oe <= e_hi:
        ax.axvline(
            math.log10(oe), color=C_ORACLE, ls="--", lw=1.8, alpha=0.9,
            label="10k oracle e*", zorder=5,
        )

    # gold solid: final e*
    fe_star = final.get("final_e")
    if fe_star and e_lo <= fe_star <= e_hi:
        ax.axvline(math.log10(fe_star), color=C_FINAL, lw=2.2,
                   label="final e*", zorder=6)

    # black horizontal: constraint target
    thr = info.get(thr_key) if thr_key else None
    if thr is not None:
        ax.axhline(thr, color="k", lw=1.2, zorder=2,
                   label="{} = {:g}".format(thr_key.replace("_", " "), thr))
        y_for_lim.append(thr)

    # gold star: final measured solution on this panel
    final_val = {
        "psnr": final.get("final_psnr"),
        "ssim": final.get("final_ssim"),
        "compression_ratio": final.get("final_cr"),
    }.get(meas_col)
    if fe_star and final_val is not None:
        ax.scatter([math.log10(fe_star)], [final_val], marker="*", s=240,
                   c=C_FINAL, edgecolor="k", linewidth=0.6, zorder=7,
                   label="final solution")
        y_for_lim.append(final_val)

    # blue diamond: 10k oracle solution on this panel
    oracle_val = {
        "psnr": oracle.get("oracle_psnr") if oracle else None,
        "ssim": oracle.get("oracle_ssim") if oracle else None,
        "compression_ratio": oracle.get("oracle_cr") if oracle else None,
    }.get(meas_col)
    if oe and oracle_val is not None:
        ax.scatter(
            [math.log10(oe)], [oracle_val], marker="D", s=200,
            facecolors="none", edgecolors=C_ORACLE, linewidth=2.0, zorder=8,
            label="10k oracle solution",
        )
        y_for_lim.append(oracle_val)

    mape_txt = _mape_annotation_text(data.get("mape"))
    if mape_txt:
        ax.text(
            0.02, 0.98, mape_txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.65"),
            zorder=10,
        )

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(*xlim_log)
    if zoom:
        yl = _ylim_from(y_for_lim)
        if yl:
            ax.set_ylim(*yl)


def make_figure(info: dict, data: dict, metrics: list[str], *, field: str,
                zoom: bool) -> plt.Figure:
    bundle = data["bundle"]
    if zoom:
        lo_w, hi_w = data["window"]
        center = info.get("phase2_e") or info.get("final_e")
        if lo_w and hi_w:
            xlim_log = (math.log10(lo_w) - 0.35, math.log10(hi_w) + 0.35)
        elif center:
            xlim_log = (math.log10(center) - 0.6, math.log10(center) + 0.6)
        else:
            xlim_log = (math.log10(bundle.e_lo), math.log10(bundle.e_hi))
    else:
        xlim_log = (math.log10(bundle.e_lo), math.log10(bundle.e_hi))

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        _plot_panel(ax, metric, info, data, xlim_log=xlim_log, zoom=zoom)
    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel(r"candidate $\log_{10}(\mathrm{error\_bound})$")

    cell = os.path.basename(info["cell_dir"].rstrip("/"))
    bits = [field, cell]
    if data.get("final", {}).get("final_e"):
        bits.append("final e*={:.3e}".format(data["final"]["final_e"]))
    if data.get("final", {}).get("final_cr"):
        bits.append("CR={:.1f}".format(data["final"]["final_cr"]))
    suffix = " (zoom)" if zoom else ""
    axes[0].set_title("Solve illustration — " + "  ".join(bits) + suffix)
    fig.tight_layout()
    return fig


def _cell_tag(info: dict) -> str:
    cell = os.path.basename(info["cell_dir"].rstrip("/"))
    if info.get("n_train") and info.get("n_phase3"):
        return "train{}_phase3{}".format(info["n_train"], info["n_phase3"])
    if cell.startswith("three_phase_"):
        return "headline"
    return cell


def render_cell(cell_dir: str, *, field: str, metrics: list[str], span: float,
                out_dir: str, compressor: str = "sz3") -> list[str]:
    info = parse_summary(cell_dir)
    data = load_cell_data(info, field=field, span=span, compressor=compressor)
    data["final"] = _final_solution(info, data)
    data["mape"] = _model_mape_vs_oracle(data["final"], data.get("oracle"))
    cr = data["bundle"].cr
    opt_cr = data.get("optimize_cr") or {}
    seg = opt_cr.get("mode") or info.get("cr_segmentation") or "none"
    if isinstance(cr, PiecewiseLassoSurrogate):
        opt_bps = opt_cr.get("breakpoints") or info.get("cr_breakpoints") or []
        print("  CR surrogate: {} piecewise, segments={}, breakpoints={} (from optimize run)".format(
            seg, opt_cr.get("segments") or cr.n_segments,
            ", ".join("{:.6g}".format(b) for b in opt_bps) or "(none)",
        ))
    else:
        print("  CR surrogate: global (single segment)")
    fin = data["final"]
    if fin.get("final_e"):
        print("  MODEL FINAL: e={:.6e}  CR={:.4f}  PSNR={:.4f}  SSIM={:.6f}".format(
            fin["final_e"], fin.get("final_cr") or float("nan"),
            fin.get("final_psnr") or float("nan"), fin.get("final_ssim") or float("nan")))
    ora = data.get("oracle")
    if ora and ora.get("oracle_e"):
        print("  10k ORACLE: e={:.6e}  CR={:.4f}  PSNR={:.4f}  SSIM={:.6f}".format(
            ora["oracle_e"], ora.get("oracle_cr") or float("nan"),
            ora.get("oracle_psnr") or float("nan"), ora.get("oracle_ssim") or float("nan")))
    mape = data.get("mape")
    if mape:
        print("  MAPE (%):  CR={:.3f}  PSNR={:.3f}  SSIM={:.3f}  median={:.3f}".format(
            mape.get("cr") or float("nan"), mape.get("psnr") or float("nan"),
            mape.get("ssim") or float("nan"), mape.get("median") or float("nan")))
    tag = _cell_tag(info)
    written = []
    for zoom, sfx in ((False, ""), (True, "_zoom")):
        fig = make_figure(info, data, metrics, field=field, zoom=zoom)
        fname = "solve_{}_{}{}.png".format(field, tag, sfx)
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=150)
        if not zoom:
            fig.savefig(os.path.join(cell_dir, "solve_illustration.png"), dpi=150)
        plt.close(fig)
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# cell selection
# ---------------------------------------------------------------------------
def rank_tradeoff_cells(tradeoff_dir: str, top_n: int) -> list[str]:
    results_csv = os.path.join(tradeoff_dir, "tradeoff_results.csv")
    if not os.path.isfile(results_csv):
        return []
    rows = []
    with open(results_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            try:
                med = float(r["mape_median_pct"])
            except (KeyError, ValueError, TypeError):
                continue
            if math.isnan(med):
                continue
            out_dir = r.get("out_dir", "")
            # Tolerate stale out_dir paths: rebuild from this tradeoff dir.
            if not os.path.isdir(out_dir):
                nt, np3 = r.get("n_train"), r.get("n_phase3")
                if nt and np3:
                    out_dir = os.path.join(
                        tradeoff_dir, "train{}_phase3{}".format(nt, np3))
            if os.path.isfile(os.path.join(out_dir, "summary.txt")):
                rows.append((med, out_dir))
    rows.sort(key=lambda t: t[0])
    return [d for _, d in rows[:top_n]]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--field", default=DEFAULT_FIELD,
                    help="field stem e.g. CLOUDf01, CLOUDf04 (selects results + baseline)")
    ap.add_argument("--compressor", default="sz3",
                    help="libpressio compressor (default: sz3)")
    ap.add_argument("--cell", nargs="+", default=None,
                    help="explicit run/cell directories (override field defaults)")
    ap.add_argument("--tradeoff-dir", default=None,
                    help="tradeoff sweep dir (default: results/{field}/tradeoff_sweep)")
    ap.add_argument("--top-n", type=int, default=3,
                    help="auto-pick best N tradeoff cells by MAPE median (default: 3)")
    ap.add_argument("--headline", action="store_true",
                    help="also plot the single three_phase_{field} run (off by default)")
    ap.add_argument("--metrics", nargs="+", default=["psnr", "cr"],
                    choices=sorted(METRIC_SPECS), help="panels top->bottom")
    ap.add_argument("--span", type=float, default=0.25,
                    help="log10 half-width for Phase-3 window verticals")
    ap.add_argument("--out", default=None,
                    help="gallery dir (default: results/{field}/solve_illustrations)")
    args = ap.parse_args()

    field = normalize_field_stem(args.field)
    compressor = normalize_compressor(args.compressor)
    out_dir = args.out or os.path.join(
        hp_results_dir(field, compressor=compressor), "solve_illustrations",
    )
    os.makedirs(out_dir, exist_ok=True)

    cells: list[str] = []
    if args.cell:
        cells = list(args.cell)
    else:
        if args.headline:
            headline = hp_three_phase_out(field, compressor=compressor)
            if os.path.isfile(os.path.join(headline, "summary.txt")):
                cells.append(headline)
        tradeoff_dir = args.tradeoff_dir or hp_tradeoff_out(field, compressor=compressor)
        cells += rank_tradeoff_cells(tradeoff_dir, args.top_n)

    if not cells:
        print("ERROR: no cells to plot (field={}, tradeoff dir {})".format(
            field, args.tradeoff_dir or hp_tradeoff_out(field, compressor=compressor)),
            file=sys.stderr)
        sys.exit(2)

    print("Field   : {}".format(field))
    print("Compressor : {}".format(compressor))
    print("Metrics : {}".format(args.metrics))
    print("Output  : {}".format(out_dir))
    print("Cells   : {}".format(len(cells)))
    for cell in cells:
        try:
            written = render_cell(cell, field=field, metrics=args.metrics,
                                  span=args.span, out_dir=out_dir,
                                  compressor=compressor)
        except (FileNotFoundError, ValueError) as exc:
            print("  SKIP {} ({})".format(cell, exc), file=sys.stderr)
            continue
        for w in written:
            print("  wrote {}".format(w))


if __name__ == "__main__":
    main()
