#!/usr/bin/env python3
"""Fit and evaluate Lasso polynomial surrogates Q_hat(e), x = log10(e)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

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


CrSegmentation = Literal["none", "auto", "fixed"]


@dataclass
class PiecewiseLassoSurrogate:
    """Piecewise CR surrogate: independent Lasso models per log10(e) segment."""

    name: str
    transform: Transform
    segments: list[tuple[float, float, LassoSurrogate]]
    breakpoints: list[float]
    n_segments: int
    bic: float
    method: str
    alpha: float
    degree: int
    r2: float
    rmse: float
    n_points: int

    def _pick_segment_indices(self, e: np.ndarray) -> np.ndarray:
        e = np.atleast_1d(np.asarray(e, dtype=float))
        idx = np.zeros(len(e), dtype=int)
        bps = self.breakpoints
        for i, val in enumerate(e):
            seg_i = len(bps)
            for j, bp in enumerate(bps):
                if val < bp:
                    seg_i = j
                    break
            idx[i] = seg_i
        return idx

    def _predict_inner(self, e):
        e = np.atleast_1d(np.asarray(e, dtype=float))
        out = np.empty(len(e), dtype=float)
        idx = self._pick_segment_indices(e)
        for seg_i, (_, _, surr) in enumerate(self.segments):
            mask = idx == seg_i
            if not np.any(mask):
                continue
            out[mask] = surr._predict_inner(e[mask])
        return out

    def __call__(self, e):
        scalar_in = np.ndim(e) == 0
        out = self._predict_inner(e)
        if scalar_in:
            return float(out[0])
        return out

    def predict_fit_space(self, e):
        scalar_in = np.ndim(e) == 0
        e = np.atleast_1d(np.asarray(e, dtype=float))
        out = np.empty(len(e), dtype=float)
        idx = self._pick_segment_indices(e)
        for seg_i, (_, _, surr) in enumerate(self.segments):
            mask = idx == seg_i
            if not np.any(mask):
                continue
            out[mask] = surr.predict_fit_space(e[mask])
        if scalar_in:
            return float(out[0])
        return out

    def to_fit_space(self, y):
        y = np.asarray(y, dtype=float)
        return np.log10(np.maximum(y, 1e-30))

    def fit_space_label(self) -> str:
        return "log10({})".format(self.name)

    def formula_str(self) -> str:
        lines = [
            "# {} piecewise ({} segments, method={}, BIC={:.4g})".format(
                self.name, self.n_segments, self.method, self.bic,
            ),
        ]
        if self.breakpoints:
            lines.append("# breakpoints (error_bound): " + ", ".join(
                "{:.6g}".format(b) for b in self.breakpoints
            ))
        for i, (elo, ehi, surr) in enumerate(self.segments, start=1):
            lines.append(
                "# segment {}: e in [{:.6g}, {:.6g}]  R2={:.6f}  RMSE={:.6g}".format(
                    i, elo, ehi, surr.r2, surr.rmse,
                )
            )
            lines.append(surr.formula_str())
        return "\n".join(lines)


CrSurrogate = Union[LassoSurrogate, PiecewiseLassoSurrogate]


@dataclass
class SurrogateBundle:
    psnr: LassoSurrogate
    ssim: LassoSurrogate
    cr: CrSurrogate
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


def _fit_weighted_lasso_poly(
    x: np.ndarray,
    y_fit: np.ndarray,
    weights: np.ndarray,
    *,
    degree: int,
    alpha: Optional[float],
) -> tuple[PolynomialFeatures, Pipeline, np.ndarray]:
    """Weighted Lasso on polynomial features of x; returns (poly, pipe, y_hat)."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(x.reshape(-1, 1))
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
    w = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    pipe.fit(X, y_fit, lasso__sample_weight=w)
    y_hat = pipe.predict(X)
    return poly, pipe, y_hat


def _fmr_em_select_k(
    e: np.ndarray,
    cr: np.ndarray,
    *,
    degree: int,
    alpha: Optional[float],
    k_max: int,
    min_points_per_segment: int = 3,
    max_iter: int = 80,
    tol: float = 1e-4,
) -> tuple[int, np.ndarray, float]:
    """Mixture-of-regressions EM; return (best_K, responsibilities, best_BIC)."""
    n = len(e)
    x = np.log10(e)
    y = np.log10(np.maximum(cr, 1e-30))
    k_cap = min(k_max, max(1, n // min_points_per_segment))
    if n < 9 or k_cap < 2:
        gamma = np.ones((n, 1), dtype=float)
        return 1, gamma, float("inf")

    best_k = 1
    best_bic = float("inf")
    best_gamma = np.ones((n, 1), dtype=float)

    for k in range(1, k_cap + 1):
        # Initialize responsibilities from quantile bins on x.
        quantiles = np.linspace(0, 100, k + 2)[1:-1]
        edges = np.percentile(x, quantiles) if k > 1 else []
        gamma = np.zeros((n, k), dtype=float)
        for i, xi in enumerate(x):
            seg = int(np.searchsorted(edges, xi, side="right"))
            seg = min(seg, k - 1)
            gamma[i, seg] = 1.0
        pi = gamma.mean(axis=0)
        sigma2 = np.ones(k, dtype=float)

        converged = False
        for _ in range(max_iter):
            # M-step
            preds = np.zeros((n, k), dtype=float)
            models: list[tuple[PolynomialFeatures, Pipeline]] = []
            for j in range(k):
                w = gamma[:, j]
                if w.sum() < 1e-9:
                    preds[:, j] = y.mean()
                    continue
                poly_j, pipe_j, _ = _fit_weighted_lasso_poly(
                    x, y, w, degree=degree, alpha=alpha,
                )
                models.append((poly_j, pipe_j))
                preds[:, j] = pipe_j.predict(
                    poly_j.transform(x.reshape(-1, 1)),
                )

            # E-step (Gaussian on fit-space residuals)
            pi = np.maximum(gamma.mean(axis=0), 1e-12)
            pi /= pi.sum()
            resid2 = (y.reshape(-1, 1) - preds) ** 2
            for j in range(k):
                mask = gamma[:, j] > 1e-9
                if np.any(mask):
                    sigma2[j] = max(
                        float(np.average(resid2[mask, j], weights=gamma[mask, j])),
                        1e-8,
                    )
                else:
                    sigma2[j] = max(float(resid2[:, j].mean()), 1e-8)

            log_resp = np.zeros((n, k), dtype=float)
            for j in range(k):
                log_resp[:, j] = (
                    math.log(pi[j])
                    - 0.5 * math.log(2.0 * math.pi * sigma2[j])
                    - 0.5 * resid2[:, j] / sigma2[j]
                )
            log_resp -= log_resp.max(axis=1, keepdims=True)
            new_gamma = np.exp(log_resp)
            new_gamma /= new_gamma.sum(axis=1, keepdims=True)

            if np.max(np.abs(new_gamma - gamma)) < tol:
                gamma = new_gamma
                converged = True
                break
            gamma = new_gamma

        if not converged:
            pass

        # Hard-assign check: each component needs enough support.
        assign = gamma.argmax(axis=1)
        counts = np.bincount(assign, minlength=k)
        if np.any(counts < min_points_per_segment):
            continue

        # BIC on fit-space mixture likelihood.
        assign = gamma.argmax(axis=1)
        rss = 0.0
        n_params = 0
        for j in range(k):
            mask = assign == j
            if not np.any(mask):
                continue
            poly_j, pipe_j, y_hat_j = _fit_weighted_lasso_poly(
                x[mask], y[mask], np.ones(mask.sum()), degree=degree, alpha=alpha,
            )
            resid = y[mask] - y_hat_j
            rss += float(np.sum(resid ** 2))
            lasso = pipe_j.named_steps["lasso"]
            n_params += int(np.sum(np.abs(lasso.coef_) > 0)) + 1
        n_params += k - 1
        if rss <= 0:
            bic = -float("inf")
        else:
            bic = n * math.log(rss / n) + n_params * math.log(n)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gamma = gamma.copy()

    return best_k, best_gamma, best_bic


def _build_piecewise_from_assignments(
    e: np.ndarray,
    cr: np.ndarray,
    assign: np.ndarray,
    k: int,
    *,
    degree: int,
    alpha: Optional[float],
    bic: float,
    method: str,
) -> PiecewiseLassoSurrogate:
    e_lo, e_hi = float(e.min()), float(e.max())
    unique = sorted(set(int(a) for a in assign))
    ordered: list[tuple[float, float, int]] = []
    for j in unique:
        mask = assign == j
        if not np.any(mask):
            continue
        ej = e[mask]
        ordered.append((float(ej.min()), float(ej.max()), j))
    ordered.sort(key=lambda t: t[0])

    bps: list[float] = []
    for i in range(len(ordered) - 1):
        bps.append(math.sqrt(ordered[i][1] * ordered[i + 1][0]))

    bounds_lo = [e_lo] + bps
    bounds_hi = bps + [e_hi]
    segments: list[tuple[float, float, LassoSurrogate]] = []
    for i, (_, _, j) in enumerate(ordered):
        mask = assign == j
        surr = fit_lasso_surrogate(
            e[mask], cr[mask], "CR", degree=degree,
            transform="log10", alpha=alpha,
        )
        segments.append((bounds_lo[i], bounds_hi[i], surr))

    if not segments:
        surr = fit_lasso_surrogate(
            e, cr, "CR", degree=degree, transform="log10", alpha=alpha,
        )
        segments = [(e_lo, e_hi, surr)]
        bps = []

    pw = PiecewiseLassoSurrogate(
        name="CR",
        transform="log10",
        segments=segments,
        breakpoints=bps,
        n_segments=len(segments),
        bic=bic,
        method=method,
        alpha=float(segments[0][2].alpha),
        degree=degree,
        r2=0.0,
        rmse=0.0,
        n_points=len(e),
    )
    y_hat = pw(e)
    ss_res = float(np.sum((cr - y_hat) ** 2))
    ss_tot = float(np.sum((cr - np.mean(cr)) ** 2))
    pw.r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    pw.rmse = float(np.sqrt(np.mean((cr - y_hat) ** 2)))
    return pw


def _segment_counts(e: np.ndarray, breakpoints: list[float]) -> list[int]:
    """Points per segment for breakpoints (same rules as fixed-segment fit)."""
    e = np.asarray(e, dtype=float)
    e_lo, e_hi = float(e.min()), float(e.max())
    bounds = [e_lo] + list(breakpoints) + [e_hi]
    counts: list[int] = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i < len(bounds) - 2:
            counts.append(int(np.sum((e >= lo) & (e < hi))))
        else:
            counts.append(int(np.sum((e >= lo) & (e <= hi))))
    return counts


def _piecewise_cr_rss(
    e: np.ndarray,
    cr: np.ndarray,
    breakpoints: list[float],
    *,
    degree: int,
    alpha: Optional[float],
) -> float:
    if breakpoints:
        surr = fit_cr_surrogate_fixed_segments(
            e, cr, breakpoints, degree=degree, alpha=alpha,
        )
    else:
        surr = fit_lasso_surrogate(
            e, cr, "CR", degree=degree, transform="log10", alpha=alpha,
        )
    pred = surr(np.asarray(e, dtype=float))
    return float(np.sum((np.asarray(cr, dtype=float) - pred) ** 2))


def _scan_cr_breakpoints(
    e: np.ndarray,
    cr: np.ndarray,
    n_segments: int,
    *,
    degree: int,
    alpha: Optional[float],
    min_points_per_segment: int = 3,
    n_candidates: int = 80,
) -> list[float]:
    """Scan error_bound grid for K-1 breakpoints minimizing training RSS."""
    from itertools import combinations

    e = np.asarray(e, dtype=float)
    cr = np.asarray(cr, dtype=float)
    k = max(1, int(n_segments))
    if k <= 1:
        return []

    e_lo, e_hi = float(e.min()), float(e.max())
    if len(e) < k * min_points_per_segment:
        return []

    grid = np.geomspace(
        max(e_lo * (1.0 + 1e-9), 1e-30),
        e_hi / (1.0 + 1e-9),
        max(8, int(n_candidates)),
    )
    interior = np.unique(e)[1:-1]
    cands = sorted({
        float(x) for x in np.concatenate([grid, interior])
        if e_lo < float(x) < e_hi
    })
    if not cands:
        return []

    def _ok(bps: list[float]) -> bool:
        if len(bps) != k - 1 or not all(bps[i] < bps[i + 1] for i in range(len(bps) - 1)):
            return False
        return all(n >= min_points_per_segment for n in _segment_counts(e, bps))

    if k == 2:
        best_bps: list[float] = []
        best_rss = float("inf")
        for b in cands:
            bps = [b]
            if not _ok(bps):
                continue
            rss = _piecewise_cr_rss(e, cr, bps, degree=degree, alpha=alpha)
            if rss < best_rss:
                best_rss, best_bps = rss, bps
        return best_bps

    if len(cands) > 30:
        idx = np.linspace(0, len(cands) - 1, 30, dtype=int)
        cands = [cands[int(i)] for i in idx]

    best_bps = []
    best_rss = float("inf")
    for combo in combinations(cands, k - 1):
        bps = list(combo)
        if not _ok(bps):
            continue
        rss = _piecewise_cr_rss(e, cr, bps, degree=degree, alpha=alpha)
        if rss < best_rss:
            best_rss, best_bps = rss, bps
    return best_bps


def fit_cr_surrogate_auto_segments(
    e: np.ndarray,
    cr: np.ndarray,
    *,
    degree: int = 4,
    alpha: Optional[float] = None,
    k_max: int = 5,
    min_points_per_segment: int = 3,
) -> PiecewiseLassoSurrogate:
    """Fit CR with exactly k_max segments; scan training data for breakpoints (min RSS)."""
    e = np.asarray(e, dtype=float)
    cr = np.asarray(cr, dtype=float)
    k = max(1, int(k_max))

    if k == 1 or len(e) < k * min_points_per_segment:
        surr = fit_lasso_surrogate(
            e, cr, "CR", degree=degree, transform="log10", alpha=alpha,
        )
        e_lo, e_hi = float(e.min()), float(e.max())
        return PiecewiseLassoSurrogate(
            name="CR",
            transform="log10",
            segments=[(e_lo, e_hi, surr)],
            breakpoints=[],
            n_segments=1,
            bic=float("nan"),
            method="auto_scan",
            alpha=float(surr.alpha),
            degree=degree,
            r2=float(surr.r2),
            rmse=float(surr.rmse),
            n_points=len(e),
        )

    bps = _scan_cr_breakpoints(
        e, cr, k,
        degree=degree, alpha=alpha,
        min_points_per_segment=min_points_per_segment,
    )
    if len(bps) != k - 1:
        import sys
        print(
            "WARNING: auto CR scan could not place {} segments on training data; "
            "using 1 global segment.".format(k),
            file=sys.stderr,
        )
        return fit_cr_surrogate_auto_segments(
            e, cr, degree=degree, alpha=alpha, k_max=1,
            min_points_per_segment=min_points_per_segment,
        )

    pw = fit_cr_surrogate_fixed_segments(
        e, cr, bps, degree=degree, alpha=alpha,
    )
    pw.method = "auto_scan"
    pw.bic = float("nan")
    return pw


def fit_cr_surrogate_fixed_segments(
    e: np.ndarray,
    cr: np.ndarray,
    breakpoints: list[float],
    *,
    degree: int = 4,
    alpha: Optional[float] = None,
) -> PiecewiseLassoSurrogate:
    """Fit CR surrogate on fixed error_bound breakpoints.

    User breakpoints are preserved for inference (not replaced by a geometric
    mean of training-point ranges).
    """
    e = np.asarray(e, dtype=float)
    cr = np.asarray(cr, dtype=float)
    e_lo, e_hi = float(e.min()), float(e.max())
    bps = sorted(float(b) for b in breakpoints if e_lo < float(b) < e_hi)
    bounds = [e_lo] + bps + [e_hi]
    segments: list[tuple[float, float, LassoSurrogate]] = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i < len(bounds) - 2:
            mask = (e >= lo) & (e < hi)
        else:
            mask = (e >= lo) & (e <= hi)
        if not np.any(mask):
            continue
        surr = fit_lasso_surrogate(
            e[mask], cr[mask], "CR", degree=degree,
            transform="log10", alpha=alpha,
        )
        segments.append((lo, hi, surr))

    if not segments:
        surr = fit_lasso_surrogate(
            e, cr, "CR", degree=degree, transform="log10", alpha=alpha,
        )
        segments = [(e_lo, e_hi, surr)]
        bps = []

    pw = PiecewiseLassoSurrogate(
        name="CR",
        transform="log10",
        segments=segments,
        breakpoints=bps,
        n_segments=len(segments),
        bic=float("nan"),
        method="fixed",
        alpha=float(segments[0][2].alpha),
        degree=degree,
        r2=0.0,
        rmse=0.0,
        n_points=len(e),
    )
    y_hat = pw(e)
    ss_res = float(np.sum((cr - y_hat) ** 2))
    ss_tot = float(np.sum((cr - np.mean(cr)) ** 2))
    pw.r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    pw.rmse = float(np.sqrt(np.mean((cr - y_hat) ** 2)))
    return pw


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
    *,
    cr_segmentation: CrSegmentation = "none",
    cr_segments_max: int = 5,
    cr_breakpoints: Optional[list[float]] = None,
) -> SurrogateBundle:
    """Fit surrogates from a dataframe with error_bound, psnr, ssim, compression_ratio."""
    e = df["error_bound"].values.astype(float)
    kw = dict(degree=degree, alpha=alpha)
    if cr_segmentation == "fixed" and cr_breakpoints:
        cr_surr = fit_cr_surrogate_fixed_segments(
            e, df["compression_ratio"].values, cr_breakpoints,
            degree=degree, alpha=alpha,
        )
    elif cr_segmentation == "auto":
        cr_surr = fit_cr_surrogate_auto_segments(
            e, df["compression_ratio"].values,
            degree=degree, alpha=alpha, k_max=cr_segments_max,
        )
    else:
        cr_surr = fit_lasso_surrogate(
            e, df["compression_ratio"].values, "CR", transform="log10", **kw,
        )
    return SurrogateBundle(
        psnr=fit_lasso_surrogate(e, df["psnr"].values, "PSNR", **kw),
        ssim=fit_lasso_surrogate(
            e, df["ssim"].values, "SSIM", transform="logit", **kw,
        ),
        cr=cr_surr,
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
            if isinstance(s, PiecewiseLassoSurrogate):
                f.write(
                    "# {} piecewise transform={}  N={}  segments={}  "
                    "method={}  BIC={:.6g}  R2={:.6f}  RMSE={:.6g}\n".format(
                        s.name, s.transform, s.n_points, s.n_segments,
                        s.method, s.bic, s.r2, s.rmse,
                    )
                )
            else:
                f.write("# {} transform={}  N={}  alpha={:.6g}  R2={:.6f}  RMSE={:.6g}\n".format(
                    s.name, s.transform, s.n_points, s.alpha, s.r2, s.rmse))
            f.write(s.formula_str() + "\n\n")
