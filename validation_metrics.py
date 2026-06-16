"""Shared MAPE / oracle helpers for model validation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def mape_scalar(actual: float, pred: float) -> float:
    if abs(actual) < 1e-30:
        return float("nan")
    return abs((pred - actual) / actual) * 100.0


def interp_measured(df: pd.DataFrame, e_query: float, col: str) -> float:
    d = df.sort_values("error_bound")
    x = np.log10(d["error_bound"].values.astype(float))
    y = d[col].values.astype(float)
    return float(np.interp(math.log10(e_query), x, y))


def measured_at_e(df: pd.DataFrame, e: float) -> dict:
    return {
        "e": float(e),
        "cr": interp_measured(df, e, "compression_ratio"),
        "psnr": interp_measured(df, e, "psnr"),
        "ssim": interp_measured(df, e, "ssim"),
    }


def pressio_oracle_grid(df: pd.DataFrame, psnr_min: float, ssim_min: float):
    """Baseline: best measured CR on discrete pressio sweep points (no interpolation)."""
    mask = (df["psnr"] >= psnr_min) & (df["ssim"] >= ssim_min)
    if not mask.any():
        return None
    row = df.loc[mask].sort_values("compression_ratio", ascending=False).iloc[0]
    return {
        "e": float(row["error_bound"]),
        "cr": float(row["compression_ratio"]),
        "psnr": float(row["psnr"]),
        "ssim": float(row["ssim"]),
        "n_grid": int(len(df)),
    }


def mape_model_vs_oracle(oracle: dict, model: dict) -> dict:
    """MAPE per field; model dict keys: e, cr, psnr, ssim."""
    fields = {}
    for key, ok, mk in (
        ("e", "e", "e"),
        ("cr", "cr", "cr"),
        ("psnr", "psnr", "psnr"),
        ("ssim", "ssim", "ssim"),
    ):
        fields[key] = mape_scalar(oracle[ok], model[mk])
    vals = [v for v in fields.values() if not np.isnan(v)]
    fields["median"] = float(np.median(vals)) if vals else float("nan")
    return fields


def format_mape_block(oracle: dict, model: dict, mape: dict, n_grid: int | None = None) -> list[str]:
    n_pts = n_grid if n_grid is not None else oracle.get("n_grid")
    grid_note = (
        "pressio oracle on {} measured sweep points".format(int(n_pts))
        if n_pts is not None
        else "pressio oracle on measured sweep grid"
    )
    return [
        "Validation — MODEL vs BASELINE ({})".format(grid_note),
        "  BASELINE (oracle):",
        "    error_bound       = {:.6g}".format(oracle["e"]),
        "    compression_ratio = {:.6f}".format(oracle["cr"]),
        "    psnr              = {:.6f}".format(oracle["psnr"]),
        "    ssim              = {:.6f}".format(oracle["ssim"]),
        "  MODEL FINAL (measured at chosen e via sweep curve):",
        "    error_bound       = {:.6g}".format(model["e"]),
        "    compression_ratio = {:.6f}".format(model["cr"]),
        "    psnr              = {:.6f}".format(model["psnr"]),
        "    ssim              = {:.6f}".format(model["ssim"]),
        "  MAPE (%) = |model - oracle| / |oracle| * 100",
        "    error_bound       {:8.3f}".format(mape["e"]),
        "    compression_ratio {:8.3f}".format(mape["cr"]),
        "    psnr              {:8.3f}".format(mape["psnr"]),
        "    ssim              {:8.3f}".format(mape["ssim"]),
        "    median            {:8.3f}".format(mape["median"]),
    ]
