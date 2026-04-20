#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _common_public_data import (
    add_source,
    build_argparser,
    finalize_result,
    json_result_template,
    load_act_dr6_kappa_sampler,
    sample_euclid_overlap_with_sampler,
    local_density_proxy,
    quantile_split,
)


def delta_kappa(density: np.ndarray, kappa: np.ndarray, q: float) -> float:
    lo, hi = quantile_split(density, q=q)
    return float(np.nanmean(kappa[hi]) - np.nanmean(kappa[lo]))


def main() -> None:
    parser = build_argparser("T2 — P30 Euclid Q1 systematics audit")
    parser.add_argument("--max-rows", type=int, default=5000)
    args = parser.parse_args()

    result = json_result_template(
        "T2 — P30 Euclid-Q1 systematics audit",
        "Repeat the P30 proxy on Euclid Q1 with public ACT lensing while auditing redshift, RA/Dec, quantile-threshold, and sky-patch stability.",
    )

    sampler = load_act_dr6_kappa_sampler()
    gal, kappa = sample_euclid_overlap_with_sampler(sampler, max_rows=args.max_rows, seed=args.seed)
    gal["density_proxy"] = local_density_proxy(gal["ra"], gal["dec"], k=10)
    gal["kappa"] = kappa
    gal = gal[np.isfinite(gal["kappa"])].reset_index(drop=True)

    baseline = delta_kappa(gal["density_proxy"].to_numpy(), gal["kappa"].to_numpy(), q=0.25)
    result["baseline_delta_kappa"] = baseline

    med_z = float(np.nanmedian(gal["z"]))
    med_ra = float(np.nanmedian(gal["ra"]))
    med_dec = float(np.nanmedian(gal["dec"]))
    splits = {
        "z_low": gal[gal["z"] <= med_z],
        "z_high": gal[gal["z"] > med_z],
        "ra_low": gal[gal["ra"] <= med_ra],
        "ra_high": gal[gal["ra"] > med_ra],
        "dec_low": gal[gal["dec"] <= med_dec],
        "dec_high": gal[gal["dec"] > med_dec],
    }
    result["subset_splits"] = {}
    for key, sub in splits.items():
        if len(sub) < 50:
            continue
        val = delta_kappa(sub["density_proxy"], sub["kappa"], q=0.25)
        result["subset_splits"][key] = {
            "delta_kappa": float(val),
            "delta_minus_baseline": float(val - baseline),
            "n": int(len(sub)),
            "same_sign_as_baseline": bool(np.sign(val) == np.sign(baseline)),
        }

    thresholds = {}
    for q in [0.20, 0.25, 0.30, 0.35]:
        thresholds[str(q)] = delta_kappa(gal["density_proxy"], gal["kappa"], q=q)
    result["threshold_scan"] = thresholds

    # Sky patch spread.
    ra_bins = pd.qcut(gal["ra"], 2, duplicates="drop")
    dec_bins = pd.qcut(gal["dec"], 2, duplicates="drop")
    patch_values = []
    patch_out = {}
    for (rab, deb), sub in gal.groupby([ra_bins, dec_bins]):
        if len(sub) < 30:
            continue
        val = delta_kappa(sub["density_proxy"], sub["kappa"], q=0.25)
        name = f"RA[{rab.left:.1f},{rab.right:.1f})_DEC[{deb.left:.1f},{deb.right:.1f})"
        patch_out[name] = {"delta_kappa": float(val), "n": int(len(sub))}
        patch_values.append(val)
    result["sky_patches"] = patch_out
    result["patch_spread"] = float(np.nanmax(patch_values) - np.nanmin(patch_values)) if patch_values else float("nan")

    add_source(result, "Euclid Q1 IRSA TAP", "https://irsa.ipac.caltech.edu/TAP/sync")
    add_source(result, "ACT DR6 lensing release", "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/dr6_lensing_release.tar.gz")
    result["headline"] = {
        "all_subset_splits_same_sign": all(v["same_sign_as_baseline"] for v in result["subset_splits"].values()),
        "threshold_monotonic_nonincreasing": bool(np.all(np.diff(list(thresholds.values())) <= 1e-12)),
    }
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
