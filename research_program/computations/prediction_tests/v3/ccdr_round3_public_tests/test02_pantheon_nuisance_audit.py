#!/usr/bin/env python3
"""
Test 02: Pantheon+ nuisance-block audit for nu.

This script checks whether the positive-nu preference is carried by one survey
block, by calibrator choices, by the redshift variable choice, or by host-mass
splits. It is aimed at the current v6.2 / v2.2 ambiguity: cosmological signal or
SN-systematics freedom.
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
    subset_pantheon,
)


def run_case(name: str, sn_df, sn_cov, bao_df, bao_cov, *, sn_z_source: str, use_calibrators: bool) -> dict:
    best = fit_nu_model(
        include_sn=True,
        include_bao=True,
        include_planck=True,
        analytic_intercept=False,
        include_radiation=True,
        diagonal_bao=False,
        sn_df=sn_df,
        sn_cov=sn_cov,
        bao_df=bao_df,
        bao_cov=bao_cov,
        sn_z_source=sn_z_source,
        use_calibrators=use_calibrators,
    )
    null = fit_nu_model_fixed_nu(
        fixed_nu=0.0,
        include_sn=True,
        include_bao=True,
        include_planck=True,
        analytic_intercept=False,
        include_radiation=True,
        diagonal_bao=False,
        sn_df=sn_df,
        sn_cov=sn_cov,
        bao_df=bao_df,
        bao_cov=bao_cov,
        sn_z_source=sn_z_source,
        use_calibrators=use_calibrators,
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
    parser.add_argument("--outdir", type=Path, default=Path("out_test02_pantheon_nuisance_audit"))
    parser.add_argument("--top-surveys", type=int, default=5)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    sn_df, sn_cov = load_pantheon_plus(use_calibrators=False)
    bao_df, bao_cov = load_desi_dr2(diagonal_only=False)

    counts = sn_df["survey_id"].value_counts()
    top_surveys = counts.head(args.top_surveys).index.tolist()
    cases = [run_case("baseline_zcmb", sn_df, sn_cov, bao_df, bao_cov, sn_z_source="z_cmb", use_calibrators=False)]

    if "z_hel" in sn_df.columns:
        cases.append(run_case("baseline_zhel", sn_df, sn_cov, bao_df, bao_cov, sn_z_source="z_hel", use_calibrators=False))

    if np.isfinite(sn_df["host_logmass"]).any():
        med = float(np.nanmedian(sn_df["host_logmass"].to_numpy(float)))
        low = np.isfinite(sn_df["host_logmass"]) & (sn_df["host_logmass"].to_numpy(float) < med)
        high = np.isfinite(sn_df["host_logmass"]) & (sn_df["host_logmass"].to_numpy(float) >= med)
        if np.sum(low) > 100:
            cases.append(run_case("hostmass_low", *subset_pantheon(sn_df, sn_cov, low), bao_df, bao_cov, sn_z_source="z_cmb", use_calibrators=False))
        if np.sum(high) > 100:
            cases.append(run_case("hostmass_high", *subset_pantheon(sn_df, sn_cov, high), bao_df, bao_cov, sn_z_source="z_cmb", use_calibrators=False))

    for sid in top_surveys:
        mask = sn_df["survey_id"].astype(str).to_numpy() != str(sid)
        if np.sum(mask) > 300:
            cases.append(run_case(f"leave_out_survey_{sid}", *subset_pantheon(sn_df, sn_cov, mask), bao_df, bao_cov, sn_z_source="z_cmb", use_calibrators=False))

    full_with_cal, cov_with_cal = load_pantheon_plus(use_calibrators=True)
    if len(full_with_cal) > len(sn_df):
        cases.append(run_case("include_calibrators", full_with_cal, cov_with_cal, bao_df, bao_cov, sn_z_source="z_cmb", use_calibrators=True))

    summary = {
        "test_name": "Pantheon+ nuisance-block audit",
        "survey_counts_top": counts.head(args.top_surveys).to_dict(),
        "cases": cases,
        "falsification_logic": {
            "confirm_like": "Positive nu remains qualitatively similar across survey-block removals, z-variable choices, and host-mass splits.",
            "falsify_like": "The nu>0 signal is carried mainly by one survey or one SN-analysis choice.",
        },
        "notes": [
            "This audit is intentionally SN-centric because v6.2 / v2.2 say the current positive-nu signal is SN-driven.",
        ],
    }
    save_json(args.outdir / "test02_pantheon_nuisance_audit_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
