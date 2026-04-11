#!/usr/bin/env python3
"""
Test 03: independent public filament replication.

Uses a public SDSS spectroscopic galaxy sample from SkyServer and a transparent
local-PCA filament-axis estimator. This is not DisPerSE or Bisous. It is a
lightweight, independently reproducible public-data reconstruction aimed at the
same falsifiability question: does a large-scale orientational correlation appear
outside the original catalogue pipeline?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common_public_data import (
    estimate_filament_axes_knn,
    fetch_sdss_galaxy_sample,
    filament_orientation_correlation,
    fit_exponential_correlation,
    save_json,
    sky_to_cartesian_mpc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test03_filament_public_replication"))
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--z-max", type=float, default=0.12)
    parser.add_argument("--max-rows", type=int, default=15000)
    parser.add_argument("--k-neighbors", type=int, default=12)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    gal = fetch_sdss_galaxy_sample(z_min=args.z_min, z_max=args.z_max, max_rows=args.max_rows)
    pts = sky_to_cartesian_mpc(gal["ra"].to_numpy(float), gal["dec"].to_numpy(float), gal["z"].to_numpy(float))
    axes = estimate_filament_axes_knn(pts, k=args.k_neighbors)
    corr = filament_orientation_correlation(pts, axes, r_bins=np.arange(20.0, 281.0, 20.0), max_pairs=250000)
    exp_fit = fit_exponential_correlation(np.asarray(corr["r_mid_mpc_over_h"]), np.asarray(corr["corr"]), np.asarray(corr["stderr"]))

    summary = {
        "test_name": "Independent public filament replication",
        "n_galaxies": int(len(gal)),
        "selection": {
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "max_rows": int(args.max_rows),
            "k_neighbors": int(args.k_neighbors),
        },
        **corr,
        "exp_fit": exp_fit,
        "falsification_logic": {
            "confirm_like": "A positive correlation reappears with a characteristic scale of the same broad order as the earlier large-scale claim.",
            "falsify_like": "The public independent reconstruction fluctuates around zero or finds only a small-scale local effect.",
        },
        "notes": [
            "This is intentionally independent of Tempel/Bisous and uses only public SDSS query outputs plus a transparent local-PCA estimator.",
        ],
    }
    save_json(args.outdir / "test03_filament_public_replication_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
