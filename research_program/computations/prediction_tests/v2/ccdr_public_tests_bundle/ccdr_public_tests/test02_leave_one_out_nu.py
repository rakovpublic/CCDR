#!/usr/bin/env python3
"""
Test 02: Leave-one-dataset-out robustness for nu.

Runs the same compact public-data nu fit while removing one of the three main
blocks at a time: Pantheon+, DESI DR2 BAO, or Planck PR3-derived r_drag prior.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from _common_public_data import (
    fit_nu_model,
    fit_nu_model_fixed_nu,
    nu_significance_from_delta_chi2,
    save_json,
)


def run_case(name: str, include_sn: bool, include_bao: bool, include_planck: bool) -> dict:
    settings = dict(
        include_sn=include_sn,
        include_bao=include_bao,
        include_planck=include_planck,
        analytic_intercept=False,
        include_radiation=True,
        diagonal_bao=False,
    )
    best = fit_nu_model(**settings)
    null = fit_nu_model_fixed_nu(fixed_nu=0.0, **settings)
    return {
        "name": name,
        "settings": settings,
        "best_fit": asdict(best),
        "null_fit": asdict(null),
        "delta_chi2_nu0": float(null.chi2 - best.chi2),
        "approx_sigma_against_nu0": nu_significance_from_delta_chi2(best, null),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test02_leave_one_out_nu"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = [
        run_case("all_three", True, True, True),
        run_case("leave_out_sn", False, True, True),
        run_case("leave_out_bao", True, False, True),
        run_case("leave_out_planck", True, True, False),
    ]

    surviving = [c for c in cases if c["best_fit"]["nu"] > 0]
    summary = {
        "test_name": "Leave-one-dataset-out nu robustness",
        "survives_positive_in_all_cases": len(surviving) == len(cases),
        "cases": cases,
        "falsification_logic": {
            "confirm_like": "nu remains positive after removing each single block in turn.",
            "falsify_like": "nu only survives when one specific dataset block is included.",
        },
    }

    save_json(args.outdir / "test02_leave_one_out_nu_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
