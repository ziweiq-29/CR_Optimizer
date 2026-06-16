"""Run dense log-spaced pressio sweeps for oracle baseline (measured points only)."""

from __future__ import annotations

import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

from pressio_run import build_pressio_cmd, parse_pressio_metrics

DEFAULT_BASELINE_N = 10000
FIELDNAMES = [
    "input", "error_bound", "compression_ratio", "psnr", "ssim",
]


def error_bounds_logspace(
    n: int, eb_lo: float, eb_hi: float,
) -> List[float]:
    if n < 1:
        return []
    if n == 1:
        return [float(eb_hi)]
    log_lo, log_hi = math.log10(eb_lo), math.log10(eb_hi)
    step = (log_lo - log_hi) / (n - 1)
    return [10.0 ** (log_hi + i * step) for i in range(n)]


def eb_key(eb: float) -> float:
    return round(math.log10(max(eb, 1e-300)), 12)


def load_existing_rows(path: str) -> Dict[float, dict]:
    if not os.path.isfile(path):
        return {}
    done = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                eb = float(row["error_bound"])
            except (KeyError, ValueError, TypeError):
                continue
            if row.get("compression_ratio") in ("", None):
                continue
            done[eb_key(eb)] = {
                "input": row.get("input", ""),
                "error_bound": eb,
                "compression_ratio": float(row["compression_ratio"]),
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
            }
    return done


def write_csv(path: str, rows_by_key: Dict[float, dict], ebs_order: List[float]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for eb in ebs_order:
            k = eb_key(eb)
            if k in rows_by_key:
                w.writerow(rows_by_key[k])
    os.replace(tmp, path)


def pending_ebs(ebs: List[float], done: Dict[float, dict]) -> List[float]:
    return [eb for eb in ebs if eb_key(eb) not in done]


def _run_one(
    pressio_bin: str,
    input_path: str,
    eb: float,
    dims: Sequence[int],
    dtype: str,
    compressor: str,
    verbose: bool = False,
) -> Tuple[Optional[dict], float]:
    import subprocess

    cmd = build_pressio_cmd(
        pressio_bin, input_path, eb,
        compressor=compressor, dims=dims, dtype=dtype,
    )
    if verbose:
        print("[pressio] " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    elapsed = time.perf_counter() - t0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return None, elapsed
    return parse_pressio_metrics(combined), elapsed


def _row_from_metrics(input_basename: str, eb: float, metrics: Optional[dict]) -> Optional[dict]:
    if metrics is None:
        return None
    cr, psnr, ssim = (
        metrics.get("compression_ratio"),
        metrics.get("psnr"),
        metrics.get("ssim"),
    )
    if cr is None or psnr is None or ssim is None:
        return None
    return {
        "input": input_basename,
        "error_bound": eb,
        "compression_ratio": cr,
        "psnr": psnr,
        "ssim": ssim,
    }


def _worker(args_tuple):
    pressio_bin, input_path, eb, dims, dtype, compressor = args_tuple
    metrics, elapsed = _run_one(
        pressio_bin, input_path, eb, dims, dtype, compressor, verbose=False,
    )
    row = _row_from_metrics(os.path.basename(input_path), eb, metrics)
    return row, elapsed, eb


def run_baseline_sweep(
    *,
    pressio_bin: str,
    input_path: str,
    out_csv: str,
    n: int = DEFAULT_BASELINE_N,
    eb_lo: float,
    eb_hi: float,
    dims: Sequence[int] = (500, 500, 100),
    dtype: str = "float",
    compressor: str = "sz3",
    jobs: int = 1,
    resume: bool = True,
    flush_every: int = 20,
) -> str:
    """
    Run n pressio compressions on a log-spaced error_bound grid.
    Returns path to CSV. Raises RuntimeError if pressio missing or zero successes.
    """
    if not os.path.isfile(pressio_bin):
        raise RuntimeError("pressio not found: {}".format(pressio_bin))
    if not os.path.isfile(input_path):
        raise RuntimeError("input not found: {}".format(input_path))

    ebs = error_bounds_logspace(n, eb_lo, eb_hi)
    input_basename = os.path.basename(input_path)
    rows_by_key: Dict[float, dict] = load_existing_rows(out_csv) if resume else {}

    todo = pending_ebs(ebs, rows_by_key)
    if not todo:
        write_csv(out_csv, rows_by_key, ebs)
        print("Baseline sweep complete: {}/{} rows in {}".format(
            len(rows_by_key), n, out_csv))
        return out_csv

    print("=" * 72)
    print("Baseline pressio sweep (oracle ground truth)")
    print("  input   : {}".format(input_path))
    print("  points  : {} log-spaced in [{:.6g}, {:.6g}]".format(n, eb_hi, eb_lo))
    print("  pending : {} / {}".format(len(todo), n))
    print("  out     : {}".format(out_csv))
    print("  jobs    : {}".format(jobs))
    print("=" * 72)

    t_all = time.perf_counter()
    completed_since_flush = 0
    ok = fail = 0

    def record(row, eb, elapsed, idx, total):
        nonlocal completed_since_flush, ok, fail
        if row is None:
            fail += 1
            print("[FAIL {}/{}] rel={:.6g} ({:.2f}s)".format(
                idx, total, eb, elapsed), flush=True)
            return
        ok += 1
        rows_by_key[eb_key(eb)] = row
        completed_since_flush += 1
        print("[OK {}/{}] rel={:.6g}  CR={}  PSNR={}  SSIM={}  ({:.2f}s)".format(
            idx, total, eb, row["compression_ratio"], row["psnr"], row["ssim"],
            elapsed,
        ), flush=True)
        if completed_since_flush >= flush_every:
            write_csv(out_csv, rows_by_key, ebs)
            completed_since_flush = 0
            print("  [flush] {} rows -> {}".format(len(rows_by_key), out_csv), flush=True)

    if jobs <= 1:
        for i, eb in enumerate(todo, 1):
            metrics, elapsed = _run_one(
                pressio_bin, input_path, float(eb), dims, dtype, compressor,
                verbose=(i <= 3),
            )
            record(_row_from_metrics(input_basename, float(eb), metrics),
                   float(eb), elapsed, i, len(todo))
    else:
        work = [
            (pressio_bin, input_path, float(eb), dims, dtype, compressor)
            for eb in todo
        ]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_worker, w): w[2] for w in work}
            for i, fut in enumerate(as_completed(futures), 1):
                row, elapsed, eb = fut.result()
                record(row, eb, elapsed, i, len(todo))

    write_csv(out_csv, rows_by_key, ebs)
    n_written = sum(1 for eb in ebs if eb_key(eb) in rows_by_key)
    print("Baseline sweep done in {:.1f}s — ok={} fail={} rows={}/{}".format(
        time.perf_counter() - t_all, ok, fail, n_written, n))
    if n_written < n:
        raise RuntimeError(
            "baseline incomplete ({}/{}); re-run same command to resume".format(
                n_written, n))
    return out_csv


def run_pressio_on_ebs(
    *,
    pressio_bin: str,
    input_path: str,
    ebs: List[float],
    dims: Sequence[int] = (500, 500, 100),
    dtype: str = "float",
    compressor: str = "sz3",
    jobs: int = 1,
    verbose: bool = True,
) -> List[dict]:
    """Run pressio at each error_bound in ``ebs``; return successful measurement rows."""
    if not os.path.isfile(pressio_bin):
        raise RuntimeError("pressio not found: {}".format(pressio_bin))
    if not os.path.isfile(input_path):
        raise RuntimeError("input not found: {}".format(input_path))

    input_basename = os.path.basename(input_path)
    rows: List[dict] = []
    ok = fail = 0
    t_all = time.perf_counter()

    def record(row, eb, elapsed, idx, total):
        nonlocal ok, fail
        if row is None:
            fail += 1
            if verbose:
                print("[FAIL {}/{}] rel={:.6g} ({:.2f}s)".format(
                    idx, total, eb, elapsed), flush=True)
            return
        ok += 1
        rows.append(row)
        if verbose:
            print("[OK {}/{}] rel={:.6g}  CR={}  PSNR={}  SSIM={}  ({:.2f}s)".format(
                idx, total, eb, row["compression_ratio"], row["psnr"],
                row["ssim"], elapsed,
            ), flush=True)

    todo = [float(eb) for eb in ebs]
    if jobs <= 1:
        for i, eb in enumerate(todo, 1):
            metrics, elapsed = _run_one(
                pressio_bin, input_path, eb, dims, dtype, compressor,
                verbose=False,
            )
            record(_row_from_metrics(input_basename, eb, metrics), eb, elapsed, i, len(todo))
    else:
        work = [
            (pressio_bin, input_path, eb, dims, dtype, compressor) for eb in todo
        ]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_worker, w): w[2] for w in work}
            for i, fut in enumerate(as_completed(futures), 1):
                row, elapsed, eb = fut.result()
                record(row, eb, elapsed, i, len(todo))

    if verbose:
        print("Phase-3 pressio sweep done in {:.1f}s — ok={} fail={}/{}".format(
            time.perf_counter() - t_all, ok, fail, len(todo)))
    return rows
