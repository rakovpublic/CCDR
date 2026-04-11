#!/usr/bin/env python3
"""
Test 04: Independent filament replication from public SDSS galaxy positions.

This does not reuse the original Tempel filament catalogue. Instead it builds an
independent local filament-direction field from public SDSS galaxy positions
using kNN + PCA in comoving space, then measures an orientation correlation
function versus pair separation.

Public input downloaded by this script:
- SDSS galaxies via the public SkyServer SQL API

Because the exact schema can evolve, the script intentionally uses a conservative
query and documents the selection it retrieved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common_public_data import (
    fetch_sdss_galaxy_sample,
    fit_exponential_correlation,
    filament_orientation_correlation,
    local_filament_axes,
    save_json,
    sky_to_cartesian_mpc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test04_independent_filament_replication"))
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--z-max", type=float, default=0.12)
    parser.add_argument("--limit-per-slice", type=int, default=2500)
    parser.add_argument("--k-neighbors", type=int, default=12)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    gal = fetch_sdss_galaxy_sample(
        z_min=args.z_min,
        z_max=args.z_max,
        limit_per_slice=args.limit_per_slice,
        chunks=6,
    )
    gal.columns = [str(c).strip().lower() for c in gal.columns]
    if not {"ra", "dec", "z"}.issubset(gal.columns):
        numeric_cols = [c for c in gal.columns if np.issubdtype(gal[c].dtype, np.number)]
        if len(numeric_cols) >= 3:
            gal = gal.rename(columns={numeric_cols[0]: "ra", numeric_cols[1]: "dec", numeric_cols[2]: "z"})
    pts = sky_to_cartesian_mpc(gal["ra"].to_numpy(float), gal["dec"].to_numpy(float), gal["z"].to_numpy(float))
    axes = local_filament_axes(pts, k=args.k_neighbors)
    corr = filament_orientation_correlation(pts, axes, bins=np.linspace(20.0, 260.0, 13), max_pairs=120_000)
    fit = fit_exponential_correlation(np.asarray(corr["r_mid"], float), np.asarray(corr["corr"], float), np.asarray(corr["stderr"], float))

    summary = {
        "test_name": "Independent filament replication",
        "n_galaxies": int(len(gal)),
        "selection": {
            "z_min": args.z_min,
            "z_max": args.z_max,
            "limit_per_slice": args.limit_per_slice,
            "k_neighbors": args.k_neighbors,
        },
        "r_mid_mpc_over_h": [float(x) for x in corr["r_mid"]],
        "corr": [float(x) for x in corr["corr"]],
        "stderr": [float(x) for x in corr["stderr"]],
        "exp_fit": fit,
        "falsification_logic": {
            "confirm_like": "A positive orientation correlation reappears with a characteristic scale of the same order as the earlier claim.",
            "falsify_like": "The signal vanishes in an independent reconstruction.",
        },
        "notes": [
            "This is deliberately independent of the original Tempel catalogue; it trades sophistication for independence and reproducibility.",
            "Use larger samples and deeper cuts for a publication-grade rerun.",
        ],
    }
    save_json(args.outdir / "test04_independent_filament_replication_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
