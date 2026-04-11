#!/usr/bin/env python3
"""
Test 01: Unified nu audit on public Pantheon+ + DESI DR2 + Planck PR3 products.

This script implements a compact late-time flat RVM-like audit, not the full
collaboration likelihoods. It is designed to answer a falsifiability question:
does positive nu survive reasonable public-data pipeline choices, or is it an
artifact of one specific construction?

Public inputs downloaded by this script:
- Pantheon+SH0ES distances and covariance (official GitHub release)
- DESI DR2 BAO mean vector and covariance (public Cobaya mirror)
- Planck PR3 chains (public IRSA archive), used here only to derive an r_drag prior

Outputs:
- JSON summary of baseline and variant fits
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from _common_public_data import (
    NuFitResult,
    fit_nu_model,
    fit_nu_model_fixed_nu,
    nu_significance_from_delta_chi2,
    save_json,
)


def run_variant(name: str, **kwargs) -> dict:
    best: NuFitResult = fit_nu_model(**kwargs)
    null: NuFitResult = fit_nu_model_fixed_nu(fixed_nu=0.0, **kwargs)
    return {
        "name": name,
        "settings": kwargs,
        "best_fit": asdict(best),
        "null_fit": asdict(null),
        "delta_chi2_nu0": float(null.chi2 - best.chi2),
        "approx_sigma_against_nu0": nu_significance_from_delta_chi2(best, null),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test01_unified_nu_audit"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    variants = [
        run_variant(
            "baseline_full_public",
            include_sn=True,
            include_bao=True,
            include_planck=True,
            analytic_intercept=False,
            include_radiation=True,
            diagonal_bao=False,
        ),
        run_variant(
            "p01_like_simplified",
            include_sn=True,
            include_bao=True,
            include_planck=True,
            analytic_intercept=True,
            include_radiation=False,
            diagonal_bao=True,
        ),
        run_variant(
            "t01_like_public_covariance",
            include_sn=True,
            include_bao=True,
            include_planck=True,
            analytic_intercept=False,
            include_radiation=True,
            diagonal_bao=False,
        ),
        run_variant(
            "no_planck_prior",
            include_sn=True,
            include_bao=True,
            include_planck=False,
            analytic_intercept=False,
            include_radiation=True,
            diagonal_bao=False,
        ),
        run_variant(
            "sn_plus_planck_only",
            include_sn=True,
            include_bao=False,
            include_planck=True,
            analytic_intercept=False,
            include_radiation=True,
            diagonal_bao=False,
        ),
        run_variant(
            "bao_plus_planck_only",
            include_sn=False,
            include_bao=True,
            include_planck=True,
            analytic_intercept=False,
            include_radiation=True,
            diagonal_bao=False,
        ),
    ]

    nus = [v["best_fit"]["nu"] for v in variants]
    robust_positive = all(nu > 0 for nu in nus)
    baseline_sigma = variants[0]["approx_sigma_against_nu0"]
    summary = {
        "test_name": "Unified nu audit",
        "purpose": "Check whether positive nu is stable across reasonable public-data pipeline choices.",
        "robust_positive_nu": robust_positive,
        "nu_min": min(nus),
        "nu_max": max(nus),
        "baseline_sigma_against_nu0": baseline_sigma,
        "variants": variants,
        "falsification_logic": {
            "confirm_like": "nu stays positive and of similar order across variants; baseline delta-chi2 against nu=0 remains non-trivial.",
            "falsify_like": "nu becomes consistent with zero, flips sign, or only appears in one narrow pipeline choice.",
        },
        "notes": [
            "This is a compact public-data audit, not a reproduction of the full collaboration likelihoods.",
            "The Planck block is represented here by an r_drag prior derived from the public PR3 cosmology chains.",
        ],
    }

    save_json(args.outdir / "test01_unified_nu_audit_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
