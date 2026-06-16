#!/usr/bin/env python3
"""Fit and evaluate Lasso polynomial surrogates Q_hat(e), x = log10(e)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

Transform = Literal["none", "logit", "log10"]


@dataclass
class LassoSurrogate:
    """Surrogate Q_hat(e) with optional logit or log10 output transform."""

    name: str
    transform: Transform
    intercept_raw: float
    coefs_raw: np.ndarray
    feature_names: list
    alpha: float
    r2: float
    rmse: float
    n_points: int
    _poly: PolynomialFeatures
    _pipe: Pipeline

    def formula_str(self) -> str:
        terms = ["{:.6g}".format(self.intercept_raw)]
        for c, n in zip(self.coefs_raw, self.feature_names):
            if abs(c) <= 0:
                continue
            sign = "-" if c < 0 else "+"
            terms.append(" {} {:.6g} * {}".format(sign, abs(c), n))
        inner = self.name + "(e) = " + terms[0]
        for t in terms[1:]:
            inner += t
        if self.transform == "logit":
            return inner + "\n{}(e) = 1 / (1 + exp(-{}(e)))".format(
                self.name, self.name.split("_")[0] if "_" in self.name else self.name
            )
        if self.transform == "log10":
            return "log10_" + inner + "\n{}(e) = 10 ** (log10_{}(e))".format(
                self.name, self.name
            )
        return inner

    def _predict_inner(self, e):
        e = np.asarray(e, dtype=float)
        x = np.log10(e)
        X = self._poly.transform(x.reshape(-1, 1))
        z = self._pipe.predict(X)
        if self.transform == "logit":
            return 1.0 / (1.0 + np.exp(-z))
        if self.transform == "log10":
            return np.power(10.0, z)
        return z

    def __call__(self, e):
        out = self._predict_inner(e)
        if np.ndim(e) == 0:
            return float(out[0])
        return out

    def predict_fit_space(self, e):
        """Polynomial prediction in the fit (transformed) space, pre-inverse.

        This is the first-line model: log10(CR), logit(SSIM), or the metric
        itself when transform == "none".
        """
        e = np.asarray(e, dtype=float)
        x = np.log10(e)
        X = self._poly.transform(x.reshape(-1, 1))
        return self._pipe.predict(X)

    def to_fit_space(self, y):
        """Map measured metric values into the surrogate's fit space."""
        y = np.asarray(y, dtype=float)
        if self.transform == "logit":
            eps = 1e-6
            yc = np.clip(y, eps, 1.0 - eps)
            return np.log(yc / (1.0 - yc))
        if self.transform == "log10":
            return np.log10(np.maximum(y, 1e-30))
        return y

    def fit_space_label(self) -> str:
        if self.transform == "logit":
            return "logit({})".format(self.name)
        if self.transform == "log10":
            return "log10({})".format(self.name)
        return self.name


@dataclass
class SurrogateBundle:
    psnr: LassoSurrogate
    ssim: LassoSurrogate
    cr: LassoSurrogate
    csv_path: str
    e_lo: float
    e_hi: float


def fit_lasso_surrogate(
    e: np.ndarray,
    y: np.ndarray,
    name: str,
    degree: int = 4,
    transform: Transform = "none",
    alpha: Optional[float] = None,
) -> LassoSurrogate:
    """Lasso on polynomial features of log10(e).

    alpha is None: LOO-CV via LassoCV (default for Phase 1 pipeline).
    alpha given: fixed regularization strength (hyperparameter grid scans).
    """
    x = np.log10(e)
    if transform == "logit":
        eps = 1e-6
        y_fit = np.log(np.clip(y, eps, 1 - eps) / (1 - np.clip(y, eps, 1 - eps)))
    elif transform == "log10":
        y_fit = np.log10(np.maximum(y, 1e-30))
    else:
        y_fit = y

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(x.reshape(-1, 1))
    feat_names = [
        n.replace("x0", "log10(e)") for n in poly.get_feature_names_out(["x0"])
    ]

    if alpha is None:
        lasso_step = LassoCV(
            cv=LeaveOneOut(), n_alphas=400, max_iter=500000, random_state=0,
        )
    else:
        lasso_step = Lasso(alpha=float(alpha), max_iter=500000)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", lasso_step),
    ])
    pipe.fit(X, y_fit)
    lasso = pipe.named_steps["lasso"]
    alpha_used = float(lasso.alpha_) if alpha is None else float(alpha)
    scaler = pipe.named_steps["scaler"]
    coefs_std = lasso.coef_
    intercept_std = lasso.intercept_
    coefs_raw = coefs_std / scaler.scale_
    intercept_raw = intercept_std - np.sum(
        coefs_std * scaler.mean_ / scaler.scale_
    )

    y_hat = LassoSurrogate(
        name=name, transform=transform,
        intercept_raw=float(intercept_raw),
        coefs_raw=coefs_raw, feature_names=list(feat_names),
        alpha=alpha_used, r2=0.0, rmse=0.0, n_points=len(e),
        _poly=poly, _pipe=pipe,
    )._predict_inner(e)

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    return LassoSurrogate(
        name=name, transform=transform,
        intercept_raw=float(intercept_raw),
        coefs_raw=coefs_raw, feature_names=list(feat_names),
        alpha=alpha_used, r2=r2, rmse=rmse, n_points=len(e),
        _poly=poly, _pipe=pipe,
    )


def _metric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def oracle_mean_r2(bundle: SurrogateBundle, df_baseline: pd.DataFrame) -> float:
    """Mean R² of PSNR, SSIM, CR surrogates vs a dense pressio oracle CSV."""
    e = df_baseline["error_bound"].values.astype(float)
    r2s = [
        _metric_r2(
            df_baseline["psnr"].values.astype(float),
            bundle.psnr(e),
        ),
        _metric_r2(
            df_baseline["ssim"].values.astype(float),
            bundle.ssim(e),
        ),
        _metric_r2(
            df_baseline["compression_ratio"].values.astype(float),
            bundle.cr(e),
        ),
    ]
    return float(np.mean(r2s))


def default_alpha_grid(
    df_train: pd.DataFrame,
    degree: int = 4,
    n_alphas: int = 40,
) -> np.ndarray:
    """Log-spaced alpha grid from LassoCV path on reference training data."""
    e = df_train["error_bound"].values.astype(float)
    y = df_train["compression_ratio"].values.astype(float)
    y_fit = np.log10(np.maximum(y, 1e-30))
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(np.log10(e).reshape(-1, 1))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(
            cv=LeaveOneOut(), n_alphas=400, max_iter=500000, random_state=0,
        )),
    ])
    pipe.fit(X, y_fit)
    alphas = pipe.named_steps["lasso"].alphas_
    lo, hi = float(alphas.min()), float(alphas.max())
    if lo <= 0:
        lo = hi * 1e-6
    return np.logspace(math.log10(lo), math.log10(hi), n_alphas)


def load_sweep_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("error_bound", "compression_ratio", "psnr", "ssim"):
        if col not in df.columns:
            raise KeyError("CSV missing column {!r} (have {})".format(
                col, list(df.columns)))
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["error_bound", "compression_ratio", "psnr", "ssim"])
    df = df[df["error_bound"] > 0].sort_values("error_bound").reset_index(drop=True)
    return df


def fit_surrogates_from_dataframe(
    df: pd.DataFrame,
    degree: int = 4,
    source_label: str = "dataframe",
    alpha: Optional[float] = None,
) -> SurrogateBundle:
    """Fit surrogates from a dataframe with error_bound, psnr, ssim, compression_ratio."""
    e = df["error_bound"].values.astype(float)
    kw = dict(degree=degree, alpha=alpha)
    return SurrogateBundle(
        psnr=fit_lasso_surrogate(e, df["psnr"].values, "PSNR", **kw),
        ssim=fit_lasso_surrogate(
            e, df["ssim"].values, "SSIM", transform="logit", **kw,
        ),
        cr=fit_lasso_surrogate(
            e, df["compression_ratio"].values, "CR", transform="log10", **kw,
        ),
        csv_path=source_label,
        e_lo=float(e.min()),
        e_hi=float(e.max()),
    )


def fit_surrogates_from_csv(
    csv_path: str,
    degree: int = 4,
) -> SurrogateBundle:
    df = load_sweep_csv(csv_path)
    return fit_surrogates_from_dataframe(df, degree=degree, source_label=csv_path)


def write_surrogate_report(bundle: SurrogateBundle, out_dir: str) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "surrogates.txt"), "w") as f:
        f.write("# Phase 1: Lasso surrogates\n")
        f.write("# Source: {}\n\n".format(bundle.csv_path))
        for s in (bundle.psnr, bundle.ssim, bundle.cr):
            f.write("# {} transform={}  N={}  alpha={:.6g}  R2={:.6f}  RMSE={:.6g}\n".format(
                s.name, s.transform, s.n_points, s.alpha, s.r2, s.rmse))
            f.write(s.formula_str() + "\n\n")
