#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from _common_public_data import (
    add_source,
    build_argparser,
    desi_bao_paths,
    finalize_result,
    json_result_template,
    load_desi_bao_measurements,
    load_euclid_q1_sample,
    local_density_proxy,
)


def infer_bao_summary(df: pd.DataFrame) -> dict:
    out = {}
    if df.empty:
        return out
    for q in sorted(df["quantity"].astype(str).unique()):
        sub = df[df["quantity"] == q]
        out[q] = {"n": int(len(sub)), "z_min": float(sub["z"].min()), "z_max": float(sub["z"].max()), "value_median": float(np.nanmedian(sub["value"]))}
    return out


def main() -> None:
    parser = build_argparser("T8 — P33 density-correlated BAO")
    parser.add_argument("--max-rows", type=int, default=3000)
    args = parser.parse_args()

    result = json_result_template(
        "T8 — P33 density-correlated BAO",
        "Use public DESI DR2 summary products to estimate the current rd spread and combine with a Euclid density proxy to quantify the predicted screening-level δrd scale pending density-binned DR3 likelihoods.",
    )

    desi = load_desi_bao_measurements()
    bao_summary = infer_bao_summary(desi)
    euclid = load_euclid_q1_sample(max_rows=args.max_rows, seed=args.seed)
    dens = local_density_proxy(euclid["ra"], euclid["dec"], k=10)
    low = float(np.nanquantile(dens, 0.25))
    high = float(np.nanquantile(dens, 0.75))
    nu_mond = 5.08e-3
    delta_r_over_r = nu_mond ** 2 * (high - low) / max(np.nanmean(dens), 1e-12)
    rd_ref = 147.09
    delta_rd = float(rd_ref * delta_r_over_r)

    result["desi_bao_summary"] = bao_summary
    result["density_proxy_summary"] = {
        "mean": float(np.nanmean(dens)),
        "q25": low,
        "q75": high,
    }
    result["prediction_proxy"] = {
        "nu_mond_reference": nu_mond,
        "delta_r_over_r": delta_r_over_r,
        "delta_rd_mpc": delta_rd,
        "real_test_requires_density_binned_DR3": True,
    }
    add_source(result, "DESI DR2 BAO summary files", desi_bao_paths()["lrg_0p6_0p8"])
    add_source(result, "Euclid Q1 IRSA TAP", "https://irsa.ipac.caltech.edu/TAP/sync")
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
