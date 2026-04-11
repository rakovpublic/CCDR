#!/usr/bin/env python3
"""
Test 04: filament dual-finder consistency and null controls.

Runs the same public SDSS galaxy sample through two transparent local filament
estimators: kNN-PCA and MST-edge PCA. It then compares the real correlation
curves and evaluates shuffled-axis nulls for each finder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common_public_data import (
    estimate_filament_axes_knn,
    estimate_filament_axes_mst,
    fetch_sdss_galaxy_sample,
    filament_orientation_correlation,
    null_control_zscores,
    save_json,
    sky_to_cartesian_mpc,
)


def run_nulls(pts: np.ndarray, axes: np.ndarray, n_realizations: int = 20, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(n_realizations):
        shuffled = axes[rng.permutation(len(axes))]
        corr = filament_orientation_correlation(pts, shuffled, r_bins=np.arange(20.0, 281.0, 20.0), max_pairs=120000, seed=int(rng.integers(0, 2**31 - 1)))
        mats.append(np.asarray(corr["corr"], dtype=float))
    return np.vstack(mats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test04_filament_dual_finder_consistency"))
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--z-max", type=float, default=0.12)
    parser.add_argument("--max-rows", type=int, default=12000)
    parser.add_argument("--k-neighbors", type=int, default=12)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    gal = fetch_sdss_galaxy_sample(z_min=args.z_min, z_max=args.z_max, max_rows=args.max_rows)
    pts = sky_to_cartesian_mpc(gal["ra"].to_numpy(float), gal["dec"].to_numpy(float), gal["z"].to_numpy(float))
    axes_knn = estimate_filament_axes_knn(pts, k=args.k_neighbors)
    axes_mst = estimate_filament_axes_mst(pts, k=args.k_neighbors)

    corr_knn = filament_orientation_correlation(pts, axes_knn, r_bins=np.arange(20.0, 281.0, 20.0), max_pairs=200000, seed=123)
    corr_mst = filament_orientation_correlation(pts, axes_mst, r_bins=np.arange(20.0, 281.0, 20.0), max_pairs=200000, seed=456)

    knn_null = run_nulls(pts, axes_knn, n_realizations=20, seed=1001)
    mst_null = run_nulls(pts, axes_mst, n_realizations=20, seed=2002)
    knn_mean, knn_std, knn_z = null_control_zscores(np.asarray(corr_knn["corr"], dtype=float), knn_null)
    mst_mean, mst_std, mst_z = null_control_zscores(np.asarray(corr_mst["corr"], dtype=float), mst_null)

    real_knn = np.asarray(corr_knn["corr"], dtype=float)
    real_mst = np.asarray(corr_mst["corr"], dtype=float)
    agree = float(np.corrcoef(np.nan_to_num(real_knn, nan=0.0), np.nan_to_num(real_mst, nan=0.0))[0, 1])

    summary = {
        "test_name": "Filament dual-finder consistency",
        "n_galaxies": int(len(gal)),
        "r_mid_mpc_over_h": corr_knn["r_mid_mpc_over_h"],
        "knn_corr": corr_knn["corr"],
        "mst_corr": corr_mst["corr"],
        "knn_shuffle_null_mean": knn_mean.tolist(),
        "knn_shuffle_null_std": knn_std.tolist(),
        "knn_z_vs_shuffle": knn_z.tolist(),
        "mst_shuffle_null_mean": mst_mean.tolist(),
        "mst_shuffle_null_std": mst_std.tolist(),
        "mst_z_vs_shuffle": mst_z.tolist(),
        "finder_curve_correlation": agree,
        "falsification_logic": {
            "confirm_like": "Both transparent filament estimators recover a similar-sign large-scale signal and exceed shuffled-axis nulls across multiple bins.",
            "falsify_like": "The sign/scale changes strongly between finders or neither beats the shuffled nulls cleanly.",
        },
        "notes": [
            "This is a dual-finder consistency check using two transparent local filament estimators rather than an external filament catalogue.",
        ],
    }
    save_json(args.outdir / "test04_filament_dual_finder_consistency_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
