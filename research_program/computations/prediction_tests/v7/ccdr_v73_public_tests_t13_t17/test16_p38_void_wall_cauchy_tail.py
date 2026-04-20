#!/usr/bin/env python3
"""
T16 / P38 — Void-wall Cauchy-tail profile (public-data proxy).

This script downloads a public SDSS DR7 VAST void catalog and the public NSA
catalog, stacks galaxies around public void centers, and measures the kurtosis
of the wall-thickness proxy delta_r = distance_to_void_center - void_radius.

This is a public-data screening implementation. It uses radial wall-thickness
stacking rather than the more elaborate dedicated transverse-cut analysis.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from common_public_utils import (
    comoving_distance_mpc_h,
    download_file,
    download_zenodo_file,
    ensure_dir,
    load_fits_table,
    radec_z_to_cartesian,
    save_json,
    table_to_dataframe,
)


NSA_URLS = [
    "https://data.sdss.org/sas/dr17/sdss/atlas/v1/nsa_v1_0_1.fits",
    "https://data.sdss.org/sas/dr16/sdss/atlas/v1/nsa_v1_0_1.fits",
]


def download_vast_files(cache_dir: Path) -> Dict[str, Path]:
    holes = cache_dir / "VoidFinder_nsa_holes.txt"
    download_zenodo_file(
        record_ids=[7406035],
        patterns=[r"VoidFinder-nsa_v1_0_1_Planck2018_comoving_holes\.txt$"],
        dest=holes,
    )
    maximals = cache_dir / "VoidFinder_nsa_maximal.txt"
    download_zenodo_file(
        record_ids=[7406035],
        patterns=[r"VoidFinder-nsa_v1_0_1_Planck2018_comoving_maximal\.txt$", r"VoidFinder-nsa_v1_0_1_Planck2018_comoving_maximals?\.txt$"],
        dest=maximals,
    )
    nsa = cache_dir / "nsa_v1_0_1.fits"
    for url in NSA_URLS:
        try:
            download_file(url, nsa)
            break
        except Exception:
            continue
    if not nsa.exists():
        raise RuntimeError("Failed to download NSA public catalog")
    return {"holes": holes, "maximals": maximals, "nsa": nsa}


def parse_voidfinder_holes(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    pass
            if len(vals) >= 4:
                rows.append(vals[:4])
    if not rows:
        raise RuntimeError(f"No void rows parsed from {path}")
    df = pd.DataFrame(rows, columns=["x_mpc_h", "y_mpc_h", "z_mpc_h", "r_mpc_h"])
    return df


def load_nsa_positions(path: Path, zmax: float, max_galaxies: int) -> pd.DataFrame:
    df = table_to_dataframe(load_fits_table(path, 1))
    # Column names vary slightly by release; try common patterns.
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for name in names:
            for k, v in cols.items():
                if name in k:
                    return v
        return None
    ra = pick("ra")
    dec = pick("dec")
    z = pick("z", "zdist")
    if not all([ra, dec, z]):
        raise RuntimeError(f"Could not identify NSA RA/DEC/z columns: {list(df.columns)[:40]}")
    out = pd.DataFrame({
        "ra": pd.to_numeric(df[ra], errors="coerce"),
        "dec": pd.to_numeric(df[dec], errors="coerce"),
        "z": pd.to_numeric(df[z], errors="coerce"),
    })
    mask = np.isfinite(out["ra"]) & np.isfinite(out["dec"]) & np.isfinite(out["z"]) & (out["z"] > 0) & (out["z"] < zmax)
    out = out.loc[mask].copy()
    if max_galaxies and len(out) > max_galaxies:
        out = out.sample(max_galaxies, random_state=12345)
    xyz = radec_z_to_cartesian(out["ra"].to_numpy(), out["dec"].to_numpy(), out["z"].to_numpy())
    out[["x_mpc_h", "y_mpc_h", "z_mpc_h"]] = xyz
    return out.reset_index(drop=True)


def stack_wall_proxy(voids: pd.DataFrame, gal: pd.DataFrame, shell_width: float, max_voids: int) -> Dict[str, object]:
    vxyz = voids[["x_mpc_h", "y_mpc_h", "z_mpc_h"]].to_numpy(dtype=float)
    vr = voids["r_mpc_h"].to_numpy(dtype=float)
    gxyz = gal[["x_mpc_h", "y_mpc_h", "z_mpc_h"]].to_numpy(dtype=float)
    if max_voids and len(vxyz) > max_voids:
        order = np.argsort(vr)[::-1][:max_voids]
        vxyz = vxyz[order]
        vr = vr[order]
    deltas = []
    counts = []
    for center, r0 in zip(vxyz, vr):
        d = np.linalg.norm(gxyz - center[None, :], axis=1)
        mask = (d >= r0 - shell_width) & (d <= r0 + shell_width)
        dr = d[mask] - r0
        if len(dr) == 0:
            continue
        deltas.extend(dr.tolist())
        counts.append(int(len(dr)))
    arr = np.array(deltas, dtype=float)
    k4 = float(kurtosis(arr, fisher=False, bias=False)) if len(arr) >= 10 else float("nan")
    return {
        "n_voids_used": int(len(counts)),
        "n_shell_points": int(len(arr)),
        "mean_points_per_void": float(np.mean(counts)) if counts else float("nan"),
        "delta_r_values_mpc_h": arr.tolist()[:5000],
        "kurtosis_k4": k4,
    }


def bootstrap_kurtosis(arr: np.ndarray, n_boot: int, rng: np.random.Generator) -> Dict[str, object]:
    vals = []
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(float(kurtosis(arr[idx], fisher=False, bias=False)))
    return {
        "k4_boot_mean": float(np.mean(vals)),
        "k4_boot_ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out_test16_p38_void_wall_cauchy_tail")
    ap.add_argument("--cache-dir", default="test16_cache")
    ap.add_argument("--zmax", type=float, default=0.12)
    ap.add_argument("--max-galaxies", type=int, default=200000)
    ap.add_argument("--max-voids", type=int, default=500)
    ap.add_argument("--shell-width", type=float, default=10.0)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12345)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    outdir = ensure_dir(args.outdir)
    cache_dir = ensure_dir(args.cache_dir)
    rng = np.random.default_rng(args.seed)

    files = download_vast_files(cache_dir)
    voids = parse_voidfinder_holes(files["holes"])
    gal = load_nsa_positions(files["nsa"], zmax=args.zmax, max_galaxies=args.max_galaxies)
    stack = stack_wall_proxy(voids, gal, shell_width=args.shell_width, max_voids=args.max_voids)
    arr = np.array(stack["delta_r_values_mpc_h"], dtype=float)
    boot = bootstrap_kurtosis(arr, n_boot=args.n_boot, rng=rng) if len(arr) >= 50 else {"k4_boot_mean": float("nan"), "k4_boot_ci95": [float("nan"), float("nan")]}

    result = {
        "test_name": "T16 / P38 void-wall Cauchy-tail profile",
        "analysis_mode": "public-data radial wall-thickness proxy",
        "shell_width_mpc_h": args.shell_width,
        **stack,
        **boot,
        "gaussian_baseline": 3.0,
        "threshold_k4": 4.0,
        "support_like": bool(np.isfinite(stack["kurtosis_k4"]) and stack["kurtosis_k4"] > 4.0),
        "falsification_logic": {
            "confirm_like": "Stacked wall-thickness distribution has k4 > 4 with bootstrap support.",
            "falsify_like": "k4 is near-Gaussian or below threshold.",
        },
        "notes": [
            "Uses public VAST VoidFinder void centers plus public NSA galaxies.",
            "Implements a radial wall-thickness proxy rather than the full dedicated transverse-cut pipeline.",
        ],
    }
    save_json(result, outdir / "result.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
