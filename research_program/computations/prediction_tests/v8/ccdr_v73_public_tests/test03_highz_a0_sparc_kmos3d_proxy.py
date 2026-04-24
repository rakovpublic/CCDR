#!/usr/bin/env python3
from __future__ import annotations

from _common_public_data import (
    add_source,
    build_argparser,
    compute_mond_sequence_proxy,
    finalize_result,
    json_result_template,
    kmos3d_candidates,
    load_kmos3d_catalog,
    load_sparc_rotation_curves,
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
    kmos = load_kmos3d_catalog()
    seq = compute_mond_sequence_proxy(curves, kmos, max_local_curves=args.max_curves)
    highz = seq["highz"]

    result["local_anchor"] = {
        "a0_proxy": float(seq["local_anchor"]["a0_proxy"]),
        "n_curves": int(seq["local_anchor"]["n_curves"]),
        "median_absolute_deviation": float(seq["local_anchor"]["mad"]),
    }
    result["highz_proxy"] = {
        "mean_a0_proxy": float(seq["highz_mean"]),
        "n_points": int(len(highz)),
        "n_highz_endpoint": int(seq["n_highz_endpoint"]),
        "spearman": seq["corr"],
        "z_min": float(highz["z"].min()),
        "z_max": float(highz["z"].max()),
    }
    result["three_point_extractor"] = {
        "closed_gap_fraction": float(seq["closed_gap_fraction"]),
        "nu_mond_proxy": float(seq["nu_mond_proxy"]),
        "proxy_asymptotic_anchor": float(seq["asymptotic_proxy"]),
        "sparc_mapping": seq["mapping"],
    }

    add_source(result, "SPARC rotation curves", "https://zenodo.org/records/16284118")
    add_source(result, "KMOS3D public data", kmos3d_candidates()[0].url)
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
