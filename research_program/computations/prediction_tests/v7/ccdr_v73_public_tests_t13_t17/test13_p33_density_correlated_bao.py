#!/usr/bin/env python3
"""
T13 / P33 — Density-correlated BAO (public-data proxy implementation).

This script downloads public DESI DR1 LSS clustering catalogs and tests whether
higher-density subsamples show a shifted BAO peak relative to lower-density
subsamples. It is a screening/proxy implementation, not a collaboration-grade
BAO likelihood.

Design choices:
- uses public DESI DR1 clustering catalogs (data + one random file per region)
- estimates local density from k-nearest-neighbor scales in comoving 3D
- splits data into low/high-density quantiles
- builds matched random subsets by nearest-neighbor transfer in (RA, DEC, z)
- computes xi(s) with a Landy–Szalay estimator and fits the BAO peak location

All inputs are downloaded by the script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

from common_public_utils import (
    bootstrap_stat,
    comoving_distance_mpc_h,
    download_first_available,
    ensure_dir,
    fit_bao_peak,
    landy_szalay,
    load_fits_table,
    log,
    radec_z_to_cartesian,
    save_json,
    save_xy_plot_png,
    table_to_dataframe,
)


TARGET_ALIASES = {
    'BGS_ANY': 'BGS_ANY',
    'BGS_BRIGHT': 'BGS_BRIGHT',
    'BGS_BRIGHT-21.5': 'BGS_BRIGHT-21.5',
    'LRG': 'LRG',
    'QSO': 'QSO',
    'ELGnotqso': 'ELG_LOPnotqso',
    'ELG_LOPnotqso': 'ELG_LOPnotqso',
}

REGION_ALIASES = {
    'N': 'NGC',
    'S': 'SGC',
    'NGC': 'NGC',
    'SGC': 'SGC',
}


def desi_candidates(version: str, fname: str) -> List[str]:
    base = f"https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/{version}/{fname}"
    return [base]


REQUIRED_COLUMNS = {
    "ra": ["RA"],
    "dec": ["DEC"],
    "z": ["Z", "Z_not4clus", "Z_HP"],
    "weight": ["WEIGHT", "WEIGHT_COMP", "WEIGHT_SYS", "WEIGHT_ZFAIL"],
}


def pick_first(df: pd.DataFrame, names: List[str], default: float | None = None) -> np.ndarray:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy()
    if default is None:
        raise KeyError(f"None of {names} found in columns")
    return np.full(len(df), float(default), dtype=float)


def load_desi_catalog(path: Path, zmin: float, zmax: float, max_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    df = table_to_dataframe(load_fits_table(path, 1))
    ra = pick_first(df, REQUIRED_COLUMNS["ra"])
    dec = pick_first(df, REQUIRED_COLUMNS["dec"])
    z = pick_first(df, REQUIRED_COLUMNS["z"])
    if "WEIGHT" in df.columns:
        w = pd.to_numeric(df["WEIGHT"], errors="coerce").to_numpy()
    else:
        w = np.ones(len(df), dtype=float)
        for col in ["WEIGHT_COMP", "WEIGHT_SYS", "WEIGHT_ZFAIL", "WEIGHT_FKP"]:
            if col in df.columns:
                w *= np.nan_to_num(pd.to_numeric(df[col], errors="coerce").to_numpy(), nan=1.0, posinf=1.0, neginf=1.0)
    mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & np.isfinite(w)
    mask &= (z >= zmin) & (z <= zmax) & (w > 0)
    sub = pd.DataFrame({"ra": ra[mask], "dec": dec[mask], "z": z[mask], "w": w[mask]})
    if max_rows and len(sub) > max_rows:
        sel = rng.choice(len(sub), size=max_rows, replace=False)
        sub = sub.iloc[sel].reset_index(drop=True)
    return sub.reset_index(drop=True)


def estimate_local_density(xyz: np.ndarray, k: int = 16) -> np.ndarray:
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=min(k + 1, len(xyz)))
    rk = np.asarray(d[:, -1], dtype=float)
    vol = 4.0 / 3.0 * np.pi * np.clip(rk, 1e-6, None) ** 3
    density = k / vol
    return density


def assign_randoms_to_density_split(
    data_ref: pd.DataFrame,
    rand: pd.DataFrame,
    high_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transfer density split from data to randoms by nearest neighbor in (ra, dec, z).
    This is a proxy that keeps angular/redshift selection approximately matched.
    """
    ref = np.column_stack([
        np.asarray(data_ref["ra"]),
        np.asarray(data_ref["dec"]),
        np.asarray(data_ref["z"]),
    ])
    tgt = np.column_stack([
        np.asarray(rand["ra"]),
        np.asarray(rand["dec"]),
        np.asarray(rand["z"]),
    ])
    # normalize axes roughly so z does not dominate
    refn = ref.copy()
    tgtn = tgt.copy()
    refn[:, 0] /= 10.0
    refn[:, 1] /= 10.0
    tgtn[:, 0] /= 10.0
    tgtn[:, 1] /= 10.0
    tree = cKDTree(refn)
    _, idx = tree.query(tgtn, k=1)
    split = high_mask[idx]
    return rand.loc[~split].reset_index(drop=True), rand.loc[split].reset_index(drop=True)


def subset_quintiles(df: pd.DataFrame, dens: np.ndarray, q: float) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    low_thr = float(np.quantile(dens, q))
    high_thr = float(np.quantile(dens, 1.0 - q))
    low = df.loc[dens <= low_thr].reset_index(drop=True)
    high = df.loc[dens >= high_thr].reset_index(drop=True)
    return low, high, low_thr, high_thr


def xi_for_subset(data_df: pd.DataFrame, rand_df: pd.DataFrame, bins: np.ndarray) -> Dict[str, np.ndarray | dict]:
    xyz_d = radec_z_to_cartesian(data_df["ra"].to_numpy(), data_df["dec"].to_numpy(), data_df["z"].to_numpy())
    xyz_r = radec_z_to_cartesian(rand_df["ra"].to_numpy(), rand_df["dec"].to_numpy(), rand_df["z"].to_numpy())

    from common_public_utils import cumulative_pair_counts

    dd = cumulative_pair_counts(xyz_d, xyz_d, bins, self_pairs=True)
    dr = cumulative_pair_counts(xyz_d, xyz_r, bins, self_pairs=False)
    rr = cumulative_pair_counts(xyz_r, xyz_r, bins, self_pairs=True)
    xi = landy_szalay(dd, dr, rr, nd=len(xyz_d), nr=len(xyz_r))
    s_mid = 0.5 * (bins[:-1] + bins[1:])
    fit = fit_bao_peak(s_mid, xi)
    return {"s_mid": s_mid, "xi": xi, "fit": fit}


def run_region(
    cache_dir: Path,
    target: str,
    region: str,
    version: str,
    zmin: float,
    zmax: float,
    max_data: int,
    max_random: int,
    density_quantile: float,
    knn: int,
    bins: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, object]:
    target = TARGET_ALIASES.get(target, target)
    region = REGION_ALIASES.get(region, region)
    data_name = f"{target}_{region}_clustering.dat.fits"
    rand_name = f"{target}_{region}_0_clustering.ran.fits"

    data_path = download_first_available(desi_candidates(version, data_name), cache_dir / data_name)
    rand_path = download_first_available(desi_candidates(version, rand_name), cache_dir / rand_name)

    data_df = load_desi_catalog(data_path, zmin=zmin, zmax=zmax, max_rows=max_data, rng=rng)
    rand_df = load_desi_catalog(rand_path, zmin=zmin, zmax=zmax, max_rows=max_random, rng=rng)

    xyz = radec_z_to_cartesian(data_df["ra"].to_numpy(), data_df["dec"].to_numpy(), data_df["z"].to_numpy())
    density = estimate_local_density(xyz, k=knn)
    data_df = data_df.copy()
    data_df["density"] = density
    low, high, low_thr, high_thr = subset_quintiles(data_df, density, density_quantile)
    high_mask = density >= high_thr
    rand_low, rand_high = assign_randoms_to_density_split(data_df, rand_df, high_mask=high_mask)

    low_res = xi_for_subset(low, rand_low, bins)
    high_res = xi_for_subset(high, rand_high, bins)

    return {
        "region": region,
        "n_data": int(len(data_df)),
        "n_random": int(len(rand_df)),
        "n_low": int(len(low)),
        "n_high": int(len(high)),
        "density_low_threshold": float(low_thr),
        "density_high_threshold": float(high_thr),
        "density_mean_low": float(np.nanmean(low["density"])),
        "density_mean_high": float(np.nanmean(high["density"])),
        "low": low_res,
        "high": high_res,
    }


def summarize(regions: List[Dict[str, object]], bins: np.ndarray) -> Dict[str, object]:
    rd_low = np.array([r["low"]["fit"].get("rd_mpc_h", np.nan) for r in regions], dtype=float)
    rd_high = np.array([r["high"]["fit"].get("rd_mpc_h", np.nan) for r in regions], dtype=float)
    dens_low = np.array([r.get("density_mean_low", np.nan) for r in regions], dtype=float)
    dens_high = np.array([r.get("density_mean_high", np.nan) for r in regions], dtype=float)
    delta = rd_high - rd_low
    frac = delta / rd_low
    if np.sum(np.isfinite(frac)) >= 2:
        r_pear, p_pear = pearsonr(np.concatenate([dens_low, dens_high]), np.concatenate([rd_low, rd_high]))
    else:
        r_pear, p_pear = np.nan, np.nan
    mids = 0.5 * (bins[:-1] + bins[1:])
    xi_low = np.nanmean(np.vstack([r["low"]["xi"] for r in regions]), axis=0)
    xi_high = np.nanmean(np.vstack([r["high"]["xi"] for r in regions]), axis=0)
    return {
        "rd_low_regions_mpc_h": rd_low.tolist(),
        "rd_high_regions_mpc_h": rd_high.tolist(),
        "delta_rd_regions_mpc_h": delta.tolist(),
        "delta_rd_over_rd_regions": frac.tolist(),
        "rd_low_mean_mpc_h": float(np.nanmean(rd_low)),
        "rd_high_mean_mpc_h": float(np.nanmean(rd_high)),
        "delta_rd_mean_mpc_h": float(np.nanmean(delta)),
        "delta_rd_over_rd_mean": float(np.nanmean(frac)),
        "pearson_density_vs_rd_r": float(r_pear),
        "pearson_density_vs_rd_p": float(p_pear),
        "s_mid_mpc_h": mids.tolist(),
        "xi_low_mean": xi_low.tolist(),
        "xi_high_mean": xi_high.tolist(),
        "support_like": bool(np.nanmean(delta) > 0),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out_test13_p33_density_correlated_bao")
    ap.add_argument("--cache-dir", default="test13_cache")
    ap.add_argument("--target", default="BGS_ANY", choices=["BGS_ANY", "BGS_BRIGHT", "BGS_BRIGHT-21.5", "LRG", "ELGnotqso", "ELG_LOPnotqso", "QSO"])
    ap.add_argument("--version", default="v1.5")
    ap.add_argument("--zmin", type=float, default=0.1)
    ap.add_argument("--zmax", type=float, default=0.6)
    ap.add_argument("--max-data-per-region", type=int, default=20000)
    ap.add_argument("--max-random-per-region", type=int, default=40000)
    ap.add_argument("--density-quantile", type=float, default=0.2)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--s-min", type=float, default=40.0)
    ap.add_argument("--s-max", type=float, default=180.0)
    ap.add_argument("--s-step", type=float, default=5.0)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    outdir = ensure_dir(args.outdir)
    cache_dir = ensure_dir(args.cache_dir)
    rng = np.random.default_rng(args.seed)
    bins = np.arange(args.s_min, args.s_max + args.s_step, args.s_step)

    regions = []
    for region in ["NGC", "SGC"]:
        log(f"[region] {region}")
        reg = run_region(
            cache_dir=cache_dir,
            target=args.target,
            region=region,
            version=args.version,
            zmin=args.zmin,
            zmax=args.zmax,
            max_data=args.max_data_per_region,
            max_random=args.max_random_per_region,
            density_quantile=args.density_quantile,
            knn=args.knn,
            bins=bins,
            rng=rng,
        )
        regions.append(reg)

    summary = summarize(regions, bins)
    result = {
        "test_name": "T13 / P33 density-correlated BAO",
        "analysis_mode": "public-data proxy clustering estimator",
        "target": args.target,
        "desi_version": args.version,
        "z_range": [args.zmin, args.zmax],
        "density_quantile": args.density_quantile,
        "knn": args.knn,
        "falsification_logic": {
            "confirm_like": "High-density subsamples show a positive BAO shift (delta rd > 0) with a stable sign across regions and a positive density-vs-rd trend.",
            "falsify_like": "The shift is null within noise or wrong-sign (delta rd <= 0).",
        },
        "regions": [
            {
                k: (v if k not in {"low", "high"} else {"fit": v["fit"]})
                for k, v in reg.items()
            }
            for reg in regions
        ],
        **summary,
    }

    save_json(result, outdir / "result.json")
    save_xy_plot_png(
        outdir / "xi_density_split.png",
        np.asarray(summary["s_mid_mpc_h"], dtype=float),
        [
            ("low density", np.asarray(summary["xi_low_mean"], dtype=float)),
            ("high density", np.asarray(summary["xi_high_mean"], dtype=float)),
        ],
        xlabel="s [Mpc/h]",
        ylabel="xi(s)",
        title="T13 / P33 density-split DESI DR1 proxy BAO",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
