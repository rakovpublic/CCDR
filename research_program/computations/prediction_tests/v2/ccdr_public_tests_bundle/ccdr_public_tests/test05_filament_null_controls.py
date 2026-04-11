#!/usr/bin/env python3
"""
Test 05: Filament null-control torture test.

Build the same independent filament-direction field as Test 04, then compare the
measured correlation against shuffled and axis-randomized null controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common_public_data import (
    fetch_sdss_galaxy_sample,
    filament_orientation_correlation,
    local_filament_axes,
    save_json,
    sky_to_cartesian_mpc,
)


def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test05_filament_null_controls"))
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--z-max", type=float, default=0.12)
    parser.add_argument("--limit-per-slice", type=int, default=2000)
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
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
    bins = np.linspace(20.0, 260.0, 13)
    real_corr = filament_orientation_correlation(pts, axes, bins=bins, max_pairs=100_000)

    null_stack = []
    for i in range(args.n_null):
        shuffled = axes.copy()
        rng.shuffle(shuffled, axis=0)
        c = filament_orientation_correlation(pts, shuffled, bins=bins, max_pairs=100_000, seed=args.seed + i + 1)
        null_stack.append(c["corr"])

    random_stack = []
    for i in range(args.n_null):
        rand_axes = random_unit_vectors(len(axes), rng)
        c = filament_orientation_correlation(pts, rand_axes, bins=bins, max_pairs=100_000, seed=args.seed + 1000 + i)
        random_stack.append(c["corr"])

    null_stack = np.asarray(null_stack)
    random_stack = np.asarray(random_stack)
    z_vs_shuffle = (real_corr["corr"] - null_stack.mean(axis=0)) / np.maximum(null_stack.std(axis=0, ddof=1), 1e-8)
    z_vs_random = (real_corr["corr"] - random_stack.mean(axis=0)) / np.maximum(random_stack.std(axis=0, ddof=1), 1e-8)

    summary = {
        "test_name": "Filament null-control torture test",
        "n_galaxies": int(len(gal)),
        "real_corr": [float(x) for x in real_corr["corr"]],
        "shuffle_null_mean": [float(x) for x in null_stack.mean(axis=0)],
        "shuffle_null_std": [float(x) for x in null_stack.std(axis=0, ddof=1)],
        "random_null_mean": [float(x) for x in random_stack.mean(axis=0)],
        "random_null_std": [float(x) for x in random_stack.std(axis=0, ddof=1)],
        "z_vs_shuffle": [float(x) for x in z_vs_shuffle],
        "z_vs_random": [float(x) for x in z_vs_random],
        "falsification_logic": {
            "confirm_like": "The real signal exceeds both shuffled-axis and random-axis null controls over a separation range, not just one bin.",
            "falsify_like": "Comparable signal strength appears after shuffling or randomizing the filament axes.",
        },
    }
    save_json(args.outdir / "test05_filament_null_controls_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
