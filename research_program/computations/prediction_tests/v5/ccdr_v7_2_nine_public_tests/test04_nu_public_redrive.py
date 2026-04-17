#!/usr/bin/env python3
"""Test 04: public re-drive of the nu claim with Pantheon+ + DESI DR2 + Planck PR3.

This is a compact falsifiability rerun focused on the public late-time pieces that currently drive the claim:
baseline z_cmb, z_hel, low/high-z tomography, and host-mass split. It is not a collaboration-grade systematics analysis.
"""
from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from _common_public_data import load_pantheon_plus, load_desi_dr2, fit_nu_model, fit_nu_model_fixed_nu, subset_pantheon, nu_significance_from_delta_chi2, save_json

def run_case(name, sn_df, sn_cov, bao_df, bao_cov, sn_z_source='z_cmb'):
    best = fit_nu_model(include_sn=True, include_bao=True, include_planck=True, analytic_intercept=False,
                        include_radiation=True, diagonal_bao=False, sn_df=sn_df, sn_cov=sn_cov,
                        bao_df=bao_df, bao_cov=bao_cov, sn_z_source=sn_z_source)
    null = fit_nu_model_fixed_nu(0.0, include_sn=True, include_bao=True, include_planck=True, analytic_intercept=False,
                                 include_radiation=True, diagonal_bao=False, sn_df=sn_df, sn_cov=sn_cov,
                                 bao_df=bao_df, bao_cov=bao_cov, sn_z_source=sn_z_source)
    return {
        'name': name,
        'best_fit': asdict(best),
        'null_fit': asdict(null),
        'delta_chi2_nu0': float(null.chi2 - best.chi2),
        'approx_sigma_against_nu0': nu_significance_from_delta_chi2(best, null),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('out_test04_nu_public_redrive'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    bao_df, bao_cov = load_desi_dr2(diagonal_only=False)
    sn_df, sn_cov = load_pantheon_plus(use_calibrators=False)
    cases = [
        run_case('pantheon_zcmb', sn_df, sn_cov, bao_df, bao_cov, 'z_cmb'),
        run_case('pantheon_zhel', sn_df, sn_cov, bao_df, bao_cov, 'z_hel'),
    ]
    z = sn_df['z_cmb'].to_numpy(float) if 'z_cmb' in sn_df.columns else sn_df.iloc[:,0].to_numpy(float)
    low = z < 0.5
    high = z >= 0.5
    if np.sum(low) > 100:
        cases.append(run_case('pantheon_lowz', *subset_pantheon(sn_df, sn_cov, low), bao_df, bao_cov, 'z_cmb'))
    if np.sum(high) > 100:
        cases.append(run_case('pantheon_highz', *subset_pantheon(sn_df, sn_cov, high), bao_df, bao_cov, 'z_cmb'))
    if 'host_logmass' in sn_df.columns:
        vals = sn_df['host_logmass'].to_numpy(float)
        finite = np.isfinite(vals)
        if np.sum(finite) > 100:
            med = float(np.nanmedian(vals[finite]))
            mlow = finite & (vals < med)
            mhigh = finite & (vals >= med)
            if np.sum(mlow) > 100:
                cases.append(run_case('pantheon_hostmass_low', *subset_pantheon(sn_df, sn_cov, mlow), bao_df, bao_cov, 'z_cmb'))
            if np.sum(mhigh) > 100:
                cases.append(run_case('pantheon_hostmass_high', *subset_pantheon(sn_df, sn_cov, mhigh), bao_df, bao_cov, 'z_cmb'))
    summary = {
        'test_name': 'Public nu re-drive',
        'cases': cases,
        'falsification_logic': {
            'confirm_like': 'A positive nu preference survives the public rerun and is not confined to one low-z or nuisance-sensitive slice.',
            'falsify_like': 'The preference remains weak, low-z concentrated, or strongly sensitive to nuisance choices, supporting the systematic interpretation.',
        },
        'notes': [
            'This rerun uses Pantheon+, DESI DR2 BAO, and the Planck PR3 public prior block through the compact public-data likelihood.',
            'The aim is to classify the signal, not to claim a collaboration-grade inference.',
        ],
    }
    save_json(args.outdir / 'test04_nu_public_redrive_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
