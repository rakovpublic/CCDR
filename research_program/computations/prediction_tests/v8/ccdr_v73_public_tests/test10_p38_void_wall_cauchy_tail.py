#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import spatial, stats

from _common_public_data import (
    add_source,
    build_argparser,
    finalize_result,
    json_result_template,
    query_sdss_galaxies,
)


def main() -> None:
    parser = build_argparser("T10 — P38 void-wall Cauchy tail")
    parser.add_argument("--n-gal", type=int, default=20000)
    parser.add_argument("--n-voids", type=int, default=20)
    args = parser.parse_args()

    result = json_result_template(
        "T10 — P38 void-wall Cauchy tail",
        "Build a lightweight public SDSS void proxy by seeding low-density minima, stacking radial shells, and measuring transverse kurtosis against the k4 > 4 screening threshold.",
    )

    gal = query_sdss_galaxies(args.n_gal, 0.05, 0.20, cache_key=f"void_tail_{args.n_gal}")

    pts = np.column_stack([gal["ra"].to_numpy(), gal["dec"].to_numpy()])
    tree = spatial.cKDTree(pts)
    d, _ = tree.query(pts, k=min(11, len(pts)))
    local = 1.0 / np.clip(d[:, -1], 1e-6, None) ** 2
    gal["density_proxy"] = local
    void_centers = gal.nsmallest(args.n_voids, "density_proxy").copy()

    shell_edges = np.linspace(0.1, 2.2, 12)
    shell_values = []
    for _, vc in void_centers.iterrows():
        rr = np.sqrt((gal["ra"] - vc["ra"]) ** 2 + (gal["dec"] - vc["dec"]) ** 2)
        counts = []
        for a, b in zip(shell_edges[:-1], shell_edges[1:]):
            counts.append(int(((rr >= a) & (rr < b)).sum()))
        shell_values.extend(counts)
    shell_values = np.asarray(shell_values, dtype=float)
    k4 = float(stats.kurtosis(shell_values, fisher=False, bias=False)) if len(shell_values) > 3 else float("nan")

    result["void_proxy"] = {
        "n_voids": int(len(void_centers)),
        "n_shell_values": int(len(shell_values)),
        "transverse_kurtosis_k4": k4,
        "gaussian_baseline": 3.0,
        "threshold_k4_gt_4": bool(k4 > 4.0),
    }
    add_source(result, "SDSS DR17 SkyServer SQL", "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch")
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
