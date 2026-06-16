#!/usr/bin/env python3
"""Run libpressio and parse CR / PSNR / SSIM metrics."""

from __future__ import annotations

import subprocess
import time
from typing import Optional, Sequence, Tuple


DEFAULT_PRESSIO = (
    "/anvil/projects/x-cis240669/libpressio-env/.spack-env/view/bin/pressio"
)


def build_pressio_cmd(
    pressio_bin: str,
    input_path: str,
    eb: float,
    compressor: str = "sz3",
    dims: Sequence[int] = (500, 500, 100),
    dtype: str = "float",
    hdf5_field: Optional[str] = None,
):
    cmd = [pressio_bin, "-b", "compressor={}".format(compressor), "-i", input_path]
    if hdf5_field and input_path.lower().endswith((".h5", ".hdf5")):
        cmd += ["-I", hdf5_field]
    for d in dims:
        cmd += ["-d", str(d)]
    cmd += [
        "-t", dtype,
        "-o", "rel={}".format(eb),
        "-b", "external:launch_metric=print",
        "-m", "time", "-m", "size", "-m", "error_stat", "-m", "ssim",
        "-M", "all",
    ]
    return cmd


def parse_pressio_metrics(text: str) -> dict:
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


def run_pressio(
    pressio_bin: str,
    input_path: str,
    eb: float,
    compressor: str = "sz3",
    dims: Sequence[int] = (500, 500, 100),
    dtype: str = "float",
    hdf5_field: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Optional[dict], float]:
    cmd = build_pressio_cmd(
        pressio_bin, input_path, eb, compressor, dims, dtype, hdf5_field,
    )
    if verbose:
        print("[pressio] " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.perf_counter() - t0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        print("[pressio] failed rc={} at rel={:.6g}".format(proc.returncode, eb))
        return None, elapsed
    return parse_pressio_metrics(combined), elapsed
