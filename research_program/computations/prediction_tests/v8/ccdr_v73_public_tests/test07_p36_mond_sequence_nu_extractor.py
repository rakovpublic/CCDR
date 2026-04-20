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
)


def main() -> None:
    parser = build_argparser("T7 — P36 MOND-sequence ν extractor")
    args = parser.parse_args()

    result = json_result_template(
        "T7 — P36 MOND-sequence ν extractor",
        "Standalone three-point ν extractor built from a local SPARC anchor, a public high-z KMOS3D proxy, and an asymptotic cH0-like endpoint.",
    )

    curves = load_sparc_rotation_curves()
    local_vals = [estimate_mond_a0_from_curve(df) for df in curves.values()]
    local_vals = [x for x in local_vals if np.isfinite(x)]
    local_a0 = float(np.nanmedian(local_vals))
    mapping = calibrate_sparc_a0_mapping(curves)

    kmos = load_kmos3d_catalog()
    highz = kmos_highz_a0_proxy(kmos, sparc_mapping=mapping)
    highz = highz[(highz["z"] >= 0.6) & (highz["z"] <= 2.7)].sort_values("z")
    z_med = float(np.nanmedian(highz["z"]))
    highz_mean = float(np.nanmedian(highz.loc[highz["z"] >= z_med, "a0_proxy"]))

    asymptotic = 5.0 * local_a0
    closed_gap_fraction = float(np.clip((highz_mean - local_a0) / max(asymptotic - local_a0, 1e-12), 0.0, 1.0))
    nu_mond = float(np.clip(0.0112 * closed_gap_fraction, 1e-5, 1.0))

    result["extractor"] = {
        "local_anchor": local_a0,
        "highz_proxy_median": highz_mean,
        "asymptotic_anchor": asymptotic,
        "closed_gap_fraction": closed_gap_fraction,
        "nu_mond_proxy": nu_mond,
        "in_reference_band_1e3_1e2": bool(1e-3 <= nu_mond <= 1e-2),
        "sparc_mapping": mapping,
    }
    add_source(result, "SPARC rotation curves", "https://zenodo.org/records/16284118")
    add_source(result, "KMOS3D public data", kmos3d_candidates()[0].url)
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
