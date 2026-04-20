#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from _common_public_data import (
    add_source,
    build_argparser,
    calibrate_sparc_a0_mapping,
    estimate_mond_a0_from_curve,
    finalize_result,
    json_result_template,
    kmos3d_candidates,
    kmos_highz_a0_proxy,
    load_kmos3d_catalog,
    load_sparc_rotation_curves,
    robust_spearman,
)


def main() -> None:
    parser = build_argparser("T3 — High-z a0 with SPARC + KMOS3D proxy")
    parser.add_argument("--max-curves", type=int, default=12)
    args = parser.parse_args()

    result = json_result_template(
        "T3 — High-z a0 with SPARC + KMOS3D proxy",
        "Estimate a local SPARC anchor and a high-z KMOS3D acceleration proxy, then test upward a0(z) behavior and a three-point ν-like gap closure.",
    )

    curves = load_sparc_rotation_curves()
    a0_vals = []
    used = 0
    for name, df in curves.items():
        a0 = estimate_mond_a0_from_curve(df)
        if np.isfinite(a0):
            a0_vals.append(a0)
            used += 1
        if used >= args.max_curves:
            break
    if not a0_vals:
        raise RuntimeError("No usable SPARC curves found")
    local_a0 = float(np.nanmedian(a0_vals))
    mapping = calibrate_sparc_a0_mapping(curves)

    kmos = load_kmos3d_catalog()
    highz = kmos_highz_a0_proxy(kmos, sparc_mapping=mapping)
    highz = highz[(highz["z"] >= 0.6) & (highz["z"] <= 2.7)].sort_values("z").head(64)
    highz["a0_scaled"] = highz["a0_proxy"]
    # Use the upper-redshift half as the "high-z" endpoint to avoid the previous
    # normalization bug where the median was forced to equal the local anchor.
    z_med = float(np.nanmedian(highz["z"]))
    highz_endpoint = highz.loc[highz["z"] >= z_med, "a0_scaled"]
    highz_mean = float(np.nanmedian(highz_endpoint))
    corr = robust_spearman(highz["z"], highz["a0_scaled"])

    cH0_anchor = local_a0 * 5.0  # screening-only asymptotic anchor placeholder
    closed_gap_fraction = float(np.clip((highz_mean - local_a0) / max(cH0_anchor - local_a0, 1e-12), 0.0, 1.0))
    nu_mond = float(np.clip(closed_gap_fraction * 0.0112, 1e-5, 1.0))

    result["local_anchor"] = {
        "a0_proxy": local_a0,
        "n_curves": int(len(a0_vals)),
        "median_absolute_deviation": float(stats_mad(a0_vals)),
    }
    result["highz_proxy"] = {
        "mean_a0_proxy": highz_mean,
        "n_points": int(len(highz)),
        "n_highz_endpoint": int(len(highz_endpoint)),
        "spearman": corr,
        "z_min": float(highz["z"].min()),
        "z_max": float(highz["z"].max()),
    }
    result["three_point_extractor"] = {
        "closed_gap_fraction": closed_gap_fraction,
        "nu_mond_proxy": nu_mond,
        "sparc_mapping": mapping,
    }

    add_source(result, "SPARC rotation curves", "https://zenodo.org/records/16284118")
    add_source(result, "KMOS3D public data", kmos3d_candidates()[0].url)
    finalize_result(__file__, result, args.out)


def stats_mad(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)))


if __name__ == "__main__":
    main()
