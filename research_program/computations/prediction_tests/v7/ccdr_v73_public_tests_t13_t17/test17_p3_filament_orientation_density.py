#!/usr/bin/env python3
"""
T17 / P3 — Filament orientational correlation (density-dependent proxy).

This script downloads the public SDSS-IV Cosmic Web Catalog from Zenodo,
extracts filament-segment positions and orientation proxies, and tests whether
orientation correlations decay more slowly in denser environments.

It uses the public filament catalog directly, not the full galaxy+filament
reconstruction workflow, so it is a screening/proxy test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from common_public_utils import (
    bin_orientation_correlation,
    comoving_distance_mpc_h,
    download_zenodo_file,
    ensure_dir,
    fit_exponential_positive,
    load_fits_table,
    radec_z_to_cartesian,
    save_json,
    table_to_dataframe,
)


def download_catalog(cache_dir: Path) -> Path:
    path = cache_dir / "Cosmic_filaments_2D_DirSCMS_new1.fits"
    download_zenodo_file(
        record_ids=[6244866],
        patterns=[r"Cosmic_filaments_2D_DirSCMS_new1\.fits$", r"Cosmic_filaments_2D_DirSCMS_new1\.csv$"],
        dest=path,
    )
    return path


def load_filaments(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".fits":
        df = table_to_dataframe(load_fits_table(path, 1))
    else:
        df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    def pick(*tokens):
        for token in tokens:
            for lc, c in cols.items():
                if token in lc:
                    return c
        return None
    ra = pick("ra")
    dec = pick("dec")
    zlow = pick("z_low")
    zhigh = pick("z_high")
    dens = pick("density")
    g1 = pick("grad_dir1", "dir1")
    g2 = pick("grad_dir2", "dir2")
    if not all([ra, dec, zlow, zhigh, dens, g1, g2]):
        raise RuntimeError(f"Missing required columns in cosmic web catalog: {list(df.columns)[:60]}")
    out = pd.DataFrame({
        "ra": pd.to_numeric(df[ra], errors="coerce"),
        "dec": pd.to_numeric(df[dec], errors="coerce"),
        "z_mid": 0.5 * (pd.to_numeric(df[zlow], errors="coerce") + pd.to_numeric(df[zhigh], errors="coerce")),
        "density": pd.to_numeric(df[dens], errors="coerce"),
        "g1": pd.to_numeric(df[g1], errors="coerce"),
        "g2": pd.to_numeric(df[g2], errors="coerce"),
    })
    out["theta"] = np.arctan2(out["g2"], out["g1"])
    mask = np.isfinite(out["ra"]) & np.isfinite(out["dec"]) & np.isfinite(out["z_mid"]) & np.isfinite(out["density"]) & np.isfinite(out["theta"])
    mask &= (out["z_mid"] > 0)
    return out.loc[mask].reset_index(drop=True)


def run_subset(df: pd.DataFrame, bins: np.ndarray, max_points: int, seed: int) -> Dict[str, object]:
    xyz = radec_z_to_cartesian(df["ra"].to_numpy(), df["dec"].to_numpy(), df["z_mid"].to_numpy())
    mids, corr = bin_orientation_correlation(xyz, df["theta"].to_numpy(dtype=float), bins=bins, max_points=max_points, rng=np.random.default_rng(seed))
    fit = fit_exponential_positive(mids, np.clip(corr, 1e-6, None))
    return {
        "s_mid_mpc_h": mids.tolist(),
        "corr": corr.tolist(),
        "fit": fit,
        "n_points": int(len(df)),
        "density_mean": float(np.mean(df["density"])),
        "density_median": float(np.median(df["density"])),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out_test17_p3_filament_orientation_density")
    ap.add_argument("--cache-dir", default="test17_cache")
    ap.add_argument("--max-points", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--s-min", type=float, default=5.0)
    ap.add_argument("--s-max", type=float, default=300.0)
    ap.add_argument("--n-bins", type=int, default=16)
    ap.add_argument("--density-quantile", type=float, default=0.5)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    outdir = ensure_dir(args.outdir)
    cache_dir = ensure_dir(args.cache_dir)

    path = download_catalog(cache_dir)
    df = load_filaments(path)
    bins = np.linspace(args.s_min, args.s_max, args.n_bins + 1)

    q = float(np.quantile(df["density"], args.density_quantile))
    low = df.loc[df["density"] <= q].copy()
    high = df.loc[df["density"] > q].copy()

    full_res = run_subset(df, bins, max_points=args.max_points, seed=args.seed)
    low_res = run_subset(low, bins, max_points=args.max_points, seed=args.seed + 1)
    high_res = run_subset(high, bins, max_points=args.max_points, seed=args.seed + 2)

    r_full = full_res["fit"].get("r_texture_mpc_h", np.nan)
    r_low = low_res["fit"].get("r_texture_mpc_h", np.nan)
    r_high = high_res["fit"].get("r_texture_mpc_h", np.nan)

    result = {
        "test_name": "T17 / P3 filament orientational correlation",
        "analysis_mode": "public cosmic-web filament-catalog proxy",
        "density_quantile_split": args.density_quantile,
        "density_threshold": q,
        "full": full_res,
        "low_density": low_res,
        "high_density": high_res,
        "delta_r_texture_high_minus_low_mpc_h": float(r_high - r_low) if np.isfinite(r_high) and np.isfinite(r_low) else float("nan"),
        "support_like": bool(np.isfinite(r_high) and np.isfinite(r_low) and r_high > r_low),
        "falsification_logic": {
            "confirm_like": "High-density subset shows longer correlation scale or stronger positive correlation than low-density subset.",
            "falsify_like": "No density dependence or wrong-sign dependence.",
        },
        "notes": [
            "Uses the public SDSS-IV Cosmic Web Catalog directly.",
            "Orientation is derived from released direction/gradient columns rather than rerunning filament finding from galaxies.",
        ],
    }
    save_json(result, outdir / "result.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
