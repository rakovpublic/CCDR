#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from scipy import spatial

from _common_public_data import (
    add_source,
    build_argparser,
    finalize_result,
    json_result_template,
    load_euclid_q1_sample,
    local_density_proxy,
    quantile_split,
    simple_exponential_scale,
)


def estimate_local_axes(ra: np.ndarray, dec: np.ndarray, k: int = 12) -> np.ndarray:
    xyz = np.column_stack([
        np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra)),
        np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra)),
        np.sin(np.deg2rad(dec)),
    ])
    tree = spatial.cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(k + 1, len(xyz)))
    axes = np.zeros((len(xyz), 3), dtype=float)
    for i in range(len(xyz)):
        pts = xyz[idx[i, 1:]]
        cov = np.cov((pts - pts.mean(axis=0)).T)
        w, v = np.linalg.eigh(cov)
        axes[i] = v[:, np.argmax(w)]
    return axes


def pairwise_orientation_proxy(ra: np.ndarray, dec: np.ndarray, axes: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.column_stack([
        np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra)),
        np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra)),
        np.sin(np.deg2rad(dec)),
    ])
    dist = spatial.distance.squareform(spatial.distance.pdist(xyz))
    corr = np.full(len(bins) - 1, np.nan)
    rmid = 0.5 * (bins[1:] + bins[:-1])
    for j in range(len(bins) - 1):
        sel = (dist >= bins[j]) & (dist < bins[j + 1])
        if not np.any(sel):
            continue
        # absolute dot product: orientation, not direction
        dots = np.abs(axes @ axes.T)[sel]
        corr[j] = np.nanmean(dots)
    return rmid, corr


def run_subset(df, bins, seed=0):
    axes = estimate_local_axes(df["ra"].to_numpy(), df["dec"].to_numpy())
    rmid, corr = pairwise_orientation_proxy(df["ra"].to_numpy(), df["dec"].to_numpy(), axes, bins)
    # null via shuffled axes.
    rng = np.random.default_rng(seed)
    shuf = axes.copy()
    rng.shuffle(shuf, axis=0)
    _, corr_null = pairwise_orientation_proxy(df["ra"].to_numpy(), df["dec"].to_numpy(), shuf, bins)
    return {
        "corr": corr.tolist(),
        "corr_null": corr_null.tolist(),
        "corr_mean": float(np.nanmean(corr)),
        "corr_null_mean": float(np.nanmean(corr_null)),
        "observed_minus_null": float(np.nanmean(corr) - np.nanmean(corr_null)),
        "exp_fit_scale": simple_exponential_scale(rmid, np.nan_to_num(corr, nan=0.0) - np.nanmin(np.nan_to_num(corr, nan=0.0)) + 1e-6),
    }


def main() -> None:
    parser = build_argparser("T5 — Euclid-Q1 filament ordering")
    parser.add_argument("--max-rows", type=int, default=2000)
    args = parser.parse_args()

    result = json_result_template(
        "T5 — Euclid-Q1 filament ordering",
        "Estimate local filament axes from Euclid Q1 geometry, measure pairwise orientation correlation, and compare full-sample, low-density, and high-density subsets against a shuffled null.",
    )

    df = load_euclid_q1_sample(max_rows=args.max_rows, seed=args.seed)
    density = local_density_proxy(df["ra"], df["dec"], k=10)
    df["density_proxy"] = density
    lo, hi = quantile_split(density, q=0.25)
    bins = np.linspace(0.005, 0.08, 7)

    full = run_subset(df, bins, seed=args.seed)
    low = run_subset(df[lo].reset_index(drop=True), bins, seed=args.seed)
    high = run_subset(df[hi].reset_index(drop=True), bins, seed=args.seed)

    result["full_sample"] = full
    result["low_density"] = low
    result["high_density"] = high
    result["headline"] = {
        "expected_direction": bool(low["observed_minus_null"] > 0),
        "density_dependence_proxy": float(low["observed_minus_null"] - high["observed_minus_null"]),
    }
    add_source(result, "Euclid Q1 IRSA TAP", "https://irsa.ipac.caltech.edu/TAP/sync")
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
