#!/usr/bin/env python3
"""
Test 01: high-z / low-z nu tomography.

This script checks whether the weak positive-nu signal persists when the public
Pantheon+ and DESI blocks are sliced by redshift. The point is not a precision
measurement of nu; it is a public-data falsifiability test of where the signal
lives.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from _common_public_data import (
    fit_nu_model,
    fit_nu_model_fixed_nu,
    load_desi_dr2,
    load_pantheon_plus,
    nu_significance_from_delta_chi2,
    save_json,
    subset_bao,
    subset_pantheon,
)


def run_case(name: str, sn_df, sn_cov, bao_df, bao_cov) -> dict:
    best = fit_nu_model(
        include_sn=sn_df is not None,
        include_bao=bao_df is not None,
        include_planck=True,
        analytic_intercept=False,
        include_radiation=True,
        diagonal_bao=False,
        sn_df=sn_df,
        sn_cov=sn_cov,
        bao_df=bao_df,
        bao_cov=bao_cov,
    )
    null = fit_nu_model_fixed_nu(
        fixed_nu=0.0,
        include_sn=sn_df is not None,
        include_bao=bao_df is not None,
        include_planck=True,
        analytic_intercept=False,
        include_radiation=True,
        diagonal_bao=False,
        sn_df=sn_df,
        sn_cov=sn_cov,
        bao_df=bao_df,
        bao_cov=bao_cov,
    )
    return {
        "name": name,
        "best_fit": asdict(best),
        "null_fit": asdict(null),
        "delta_chi2_nu0": float(null.chi2 - best.chi2),
        "approx_sigma_against_nu0": nu_significance_from_delta_chi2(best, null),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test01_nu_redshift_tomography"))
    parser.add_argument("--z-split", type=float, default=0.5)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    sn_df, sn_cov = load_pantheon_plus(use_calibrators=False)
    bao_df, bao_cov = load_desi_dr2(diagonal_only=False)

    low_sn = sn_df["z_cmb"].to_numpy(float) < args.z_split
    high_sn = sn_df["z_cmb"].to_numpy(float) >= args.z_split
    low_bao = bao_df["z"].to_numpy(float) < args.z_split
    high_bao = bao_df["z"].to_numpy(float) >= args.z_split

    cases = [
        run_case("all_blocks_full_z", sn_df, sn_cov, bao_df, bao_cov),
        run_case("sn_lowz_plus_all_bao", *subset_pantheon(sn_df, sn_cov, low_sn), bao_df, bao_cov),
        run_case("sn_highz_plus_all_bao", *subset_pantheon(sn_df, sn_cov, high_sn), bao_df, bao_cov),
        run_case("all_sn_plus_bao_lowz", sn_df, sn_cov, *subset_bao(bao_df, bao_cov, low_bao)),
        run_case("all_sn_plus_bao_highz", sn_df, sn_cov, *subset_bao(bao_df, bao_cov, high_bao)),
        run_case("bao_highz_plus_planck_only", None, None, *subset_bao(bao_df, bao_cov, high_bao)),
    ]

    summary = {
        "test_name": "High-z nu tomography",
        "z_split": float(args.z_split),
        "cases": cases,
        "falsification_logic": {
            "confirm_like": "Positive nu survives and is not confined to one low-z SN slice; high-z or BAO-containing blocks retain non-trivial support.",
            "falsify_like": "The nu>0 preference is confined to one low-z SN block and collapses in the BAO/high-z sector.",
        },
        "notes": [
            "This is a tomographic public-data stress test, not a full collaboration likelihood.",
            "The Planck block enters as a public PR3-derived r_drag prior.",
        ],
    }
    save_json(args.outdir / "test01_nu_redshift_tomography_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
