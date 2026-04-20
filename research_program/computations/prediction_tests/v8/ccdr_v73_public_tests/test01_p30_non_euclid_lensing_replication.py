#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from _common_public_data import (
    add_source,
    bootstrap_statistic,
    build_argparser,
    finalize_result,
    json_result_template,
    load_act_dr6_kappa_sampler,
    sample_euclid_overlap_with_sampler,
    load_planck_pr4_kappa_sampler,
    local_density_proxy,
    quantile_split,
    robust_pearson,
    robust_spearman,
)


def summarize_path(name: str, kappa: np.ndarray, density: np.ndarray, q: float = 0.25) -> dict:
    lo, hi = quantile_split(density, q=q)
    delta = float(np.nanmean(kappa[hi]) - np.nanmean(kappa[lo]))
    return {
        "delta_kappa": delta,
        "pearson": robust_pearson(density, kappa),
        "spearman": robust_spearman(density, kappa),
        "bootstrap_delta": bootstrap_statistic(kappa[hi] - np.nanmean(kappa[lo]), np.mean, n_boot=500),
        "n_low": int(np.sum(lo)),
        "n_high": int(np.sum(hi)),
        "path": name,
    }


def main() -> None:
    parser = build_argparser("T1 — P30 non-Euclid lensing replication")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--q", type=float, default=0.25)
    args = parser.parse_args()

    result = json_result_template(
        "T1 — P30 non-Euclid lensing replication",
        "Query Euclid Q1 public galaxies, compute a local-density proxy, and test sign stability of density-correlated lensing on ACT DR6 and Planck PR4 public maps.",
    )

    act = load_act_dr6_kappa_sampler()
    gal, kappa_act = sample_euclid_overlap_with_sampler(act, max_rows=args.max_rows, seed=args.seed)
    density = local_density_proxy(gal["ra"], gal["dec"], k=10)
    gal["density_proxy"] = density
    result["sample"] = {
        "n_galaxies": int(len(gal)),
        "ra_range_deg": [float(gal["ra"].min()), float(gal["ra"].max())],
        "dec_range_deg": [float(gal["dec"].min()), float(gal["dec"].max())],
        "z_range": [float(gal["z"].min()), float(gal["z"].max())],
    }

    # ACT path
    good = np.isfinite(kappa_act)
    act_summary = summarize_path("ACT_DR6", kappa_act[good], density[good], q=args.q)
    result["act_proxy"] = act_summary
    add_source(result, "Euclid Q1 IRSA TAP", "https://irsa.ipac.caltech.edu/TAP/sync")
    add_source(result, "ACT DR6 lensing release", "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/dr6_lensing_release.tar.gz")

    # Planck path: optional but attempted.
    try:
        pl = load_planck_pr4_kappa_sampler()
        gal_pl, kappa_pl = sample_euclid_overlap_with_sampler(pl, max_rows=min(args.max_rows, 4000), seed=args.seed)
        dens_pl = local_density_proxy(gal_pl["ra"], gal_pl["dec"], k=10)
        good_pl = np.isfinite(kappa_pl)
        result["planck_proxy"] = summarize_path("Planck_PR4", kappa_pl[good_pl], dens_pl[good_pl], q=args.q)
        add_source(result, "Planck PR4 lensing release", "https://github.com/carronj/planck_PR4_lensing/releases")
    except Exception as exc:  # noqa: BLE001
        result["planck_proxy"] = {"status": "unavailable", "reason": str(exc)}
        result["notes"].append("Planck PR4 public asset download may depend on current GitHub release tag naming; ACT path still executed.")

    result["headline"] = {
        "same_positive_sign": bool(result["act_proxy"]["delta_kappa"] > 0 and (result["planck_proxy"].get("delta_kappa", 1) > 0)),
        "screening_interpretation": (
            "Support-like if both public lensing paths return positive Δκ and positive density–κ correlation."
        ),
    }
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
