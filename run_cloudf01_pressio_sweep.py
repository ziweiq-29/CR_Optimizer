#!/usr/bin/env python3
"""
Sweep sz3 rel error bounds on Hurricane CLOUDf01.bin via libpressio.

Records per run: input filename, error_bound, compression_ratio, psnr, ssim.

Example:
    cd /anvil/projects/x-cis240669/Hurricane
    python3 run_cloudf01_pressio_sweep.py
    python3 run_cloudf01_pressio_sweep.py --dry-run
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time

DEFAULT_PRESSIO = (
    "/anvil/projects/x-cis240669/libpressio-env/.spack-env/view/bin/pressio"
)
DEFAULT_INPUT = (
    "/anvil/projects/x-cis240669/Hurricane/CLOUDf01.bin"
)
DEFAULT_OUT = "CLOUDf01_sz3_sweep.csv"
EB_LO = 1e-6
EB_HI = 1e-1
N_POINTS = 20


def error_bounds_logspace(n=N_POINTS, eb_lo=EB_LO, eb_hi=EB_HI):
    """n points evenly spaced in log10 between eb_hi and eb_lo (inclusive)."""
    if n < 2:
        return [eb_hi] if n == 1 else []
    log_lo, log_hi = math.log10(eb_lo), math.log10(eb_hi)
    step = (log_lo - log_hi) / (n - 1)
    return [10.0 ** (log_hi + i * step) for i in range(n)]


def build_pressio_cmd(
    pressio_bin,
    input_path,
    eb,
    compressor="sz3",
    dims=(500, 500, 100),
    dtype="float",
):
    cmd = [
        pressio_bin,
        "-b", "compressor={}".format(compressor),
        "-i", input_path,
    ]
    for d in dims:
        cmd += ["-d", str(d)]
    cmd += [
        "-t", dtype,
        "-o", "rel={}".format(eb),
        "-b", "external:launch_metric=print",
        "-m", "time",
        "-m", "size",
        "-m", "error_stat",
        "-m", "ssim",
        "-M", "all",
    ]
    return cmd


def parse_pressio_metrics(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if " = " not in line:
            continue
        key_part, value_part = line.split(" = ", 1)
        name = key_part.split(":")[-1] if ":" in key_part else key_part
        if "<" in name:
            name = name.split("<")[0]
        try:
            out[name] = float(value_part.strip())
        except ValueError:
            out[name] = value_part.strip()
    return out


def run_one(pressio_bin, input_path, eb, dims, dtype, compressor, verbose=True):
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
        print(
            "[ERROR] pressio failed (rc={}) at rel={:.6g}".format(
                proc.returncode, eb),
            file=sys.stderr,
        )
        print((proc.stderr or "")[:2000], file=sys.stderr)
        return None, elapsed
    return parse_pressio_metrics(combined), elapsed


def row_from_metrics(input_basename, eb, metrics):
    if metrics is None:
        return None
    cr = metrics.get("compression_ratio")
    psnr = metrics.get("psnr")
    ssim = metrics.get("ssim")
    if cr is None or psnr is None or ssim is None:
        missing = [
            k for k, v in (
                ("compression_ratio", cr),
                ("psnr", psnr),
                ("ssim", ssim),
            )
            if v is None
        ]
        print("[WARN] missing metrics {} at rel={:.6g}".format(missing, eb),
              file=sys.stderr)
    return {
        "input": input_basename,
        "error_bound": eb,
        "compression_ratio": cr,
        "psnr": psnr,
        "ssim": ssim,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pressio", default=DEFAULT_PRESSIO)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--compressor", default="sz3")
    ap.add_argument("--dims", default="500 500 100",
                    help="space-separated dimensions")
    ap.add_argument("--dtype", default="float")
    ap.add_argument("--eb-lo", type=float, default=EB_LO)
    ap.add_argument("--eb-hi", type=float, default=EB_HI)
    ap.add_argument("--n", type=int, default=N_POINTS,
                    help="number of error bounds (log-spaced)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output CSV path (relative to cwd unless absolute)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands only, do not run pressio")
    args = ap.parse_args()

    dims = tuple(int(d) for d in args.dims.strip().split())
    ebs = error_bounds_logspace(args.n, args.eb_lo, args.eb_hi)
    input_basename = os.path.basename(args.input)

    if not args.dry_run:
        if not os.path.isfile(args.pressio):
            print("ERROR: pressio not found: {}".format(args.pressio),
                  file=sys.stderr)
            sys.exit(2)
        if not os.path.isfile(args.input):
            print("ERROR: input not found: {}".format(args.input),
                  file=sys.stderr)
            sys.exit(2)

    fieldnames = [
        "input", "error_bound", "compression_ratio", "psnr", "ssim",
    ]

    print("# input : {}".format(args.input))
    print("# dims  : {}".format(dims))
    print("# eb    : {} points, logspace [{:.6g}, {:.6g}]".format(
        args.n, args.eb_hi, args.eb_lo))
    print("# out   : {}".format(args.out))
    print()

    rows = []
    for i, eb in enumerate(ebs):
        eb_f = float(eb)
        print("--- [{}/{}] rel = {:.6g} ---".format(i + 1, len(ebs), eb_f))
        if args.dry_run:
            cmd = build_pressio_cmd(
                args.pressio, args.input, eb_f,
                compressor=args.compressor, dims=dims, dtype=args.dtype,
            )
            print(" ".join(cmd))
            rows.append({
                "input": input_basename,
                "error_bound": eb_f,
                "compression_ratio": "",
                "psnr": "",
                "ssim": "",
            })
            continue

        metrics, elapsed = run_one(
            args.pressio, args.input, eb_f, dims, args.dtype, args.compressor,
        )
        print("[done] {:.2f} s".format(elapsed))
        row = row_from_metrics(input_basename, eb_f, metrics)
        if row is None:
            continue
        rows.append(row)
        print("  CR={}  PSNR={}  SSIM={}".format(
            row["compression_ratio"], row["psnr"], row["ssim"]))

    if args.dry_run:
        print("\n(dry-run: {} commands, no CSV written)".format(len(ebs)))
        return

    if not rows:
        print("ERROR: no successful runs.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\nWrote {} rows to {}".format(len(rows), out_path))


if __name__ == "__main__":
    main()
