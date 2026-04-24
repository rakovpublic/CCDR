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
    parser = build_argparser("T7 — P36 MOND-sequence ν extractor")
    args = parser.parse_args()

    result = json_result_template(
        "T7 — P36 MOND-sequence ν extractor",
        "Standalone three-point ν extractor built from a local SPARC anchor, a public high-z KMOS3D proxy, and an asymptotic cH0-like endpoint.",
    )

    curves = load_sparc_rotation_curves()
    kmos = load_kmos3d_catalog()
    seq = compute_mond_sequence_proxy(curves, kmos)

    result["extractor"] = {
        "local_anchor": float(seq["local_anchor"]["a0_proxy"]),
        "highz_proxy_median": float(seq["highz_mean"]),
        "asymptotic_anchor": float(seq["asymptotic_proxy"]),
        "closed_gap_fraction": float(seq["closed_gap_fraction"]),
        "nu_mond_proxy": float(seq["nu_mond_proxy"]),
        "in_reference_band_1e3_1e2": bool(1e-3 <= seq["nu_mond_proxy"] <= 1e-2),
        "sparc_mapping": seq["mapping"],
        "local_anchor_scatter_mad": float(seq["local_anchor"]["mad"]),
    }
    add_source(result, "SPARC rotation curves", "https://zenodo.org/records/16284118")
    add_source(result, "KMOS3D public data", kmos3d_candidates()[0].url)
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
