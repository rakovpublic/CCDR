#!/usr/bin/env python3
"""
Test 05: local a0 -> Milgrom retreat extended to a high-z precursor proxy.

Updated implementation:
- keeps the local SPARC anchor and high-z proxy a0 estimate
- adds the same joint Test 05 + Test 03 asynchronous-stage fit used by Test 03
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from _common_public_data import (
    estimate_a0_from_rotation_sample,
    fit_async_stage_joint_model,
    fit_rar_hierarchical_like,
    load_desi_fullshape_proxy_points,
    load_highz_rotation_proxy_sample,
    load_sparc_rar,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, default=Path('out_test05_highz_a0_precursor'))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rar = load_sparc_rar().copy()
    if 'Galaxy' in rar.columns:
        gcol = 'Galaxy'
    elif 'galaxy' in rar.columns:
        gcol = 'galaxy'
    else:
        rar['__all__'] = '__all__'
        gcol = '__all__'
    local_fit = fit_rar_hierarchical_like(rar, gcol, 'gobs', 'gbar', offset_prior_dex=0.08)

    highz = estimate_a0_from_rotation_sample(load_highz_rotation_proxy_sample())
    if highz.empty:
        highz_mean = float('nan')
        rho_r = float('nan')
        rho_p = float('nan')
    else:
        highz_mean = float(np.nanmean(highz['a0_proxy_m_per_s2']))
        rho = stats.spearmanr(highz['z'].to_numpy(float), highz['a0_proxy_m_per_s2'].to_numpy(float), nan_policy='omit')
        rho_r = float(rho.statistic)
        rho_p = float(rho.pvalue)
    cH0 = 299792458.0 * (70.0 * 1000.0 / 3.085677581491367e22)
    milgrom = 1.2e-10

    growth_points = load_desi_fullshape_proxy_points()
    joint = fit_async_stage_joint_model(growth_points, highz)
    joint_path = args.outdir / 'test03_test05_joint_async_stage_summary.json'
    save_json(joint_path, joint)

    summary = {
        'test_name': 'High-z a0 precursor proxy',
        'local_anchor_fit': local_fit,
        'local_anchor_point_count': int(local_fit.get('n_points', 0)),
        'highz_points': highz.to_dict(orient='records'),
        'highz_mean_a0_proxy_m_per_s2': highz_mean,
        'highz_fractional_offset_from_milgrom': float((highz_mean - milgrom) / milgrom),
        'highz_fractional_offset_from_cH0': float((highz_mean - cH0) / cH0),
        'spearman_z_vs_a0_proxy_r': rho_r,
        'spearman_z_vs_a0_proxy_p': rho_p,
        'joint_with_test03_async_stage': {
            'joint_summary_json': str(joint_path),
            **joint,
        },
        'falsification_logic': {
            'confirm_like': 'The high-z proxy sample shows a trend of a0 away from the local Milgrom value toward larger asymptotic values, and the coupled async-stage fit with Test 03 improves over separate baseline trends.',
            'falsify_like': 'The proxy a0 stays flat near the local Milgrom value out to z > 1, or the shared async-stage fit adds no explanatory power beyond simple baseline trends.',
        },
        'notes': [
            'This is a high-z precursor proxy, not a full high-z RAR reconstruction.',
            'Public high-z ionised-gas kinematic samples rarely expose SPARC-grade baryonic decompositions, so the script uses a coarse MOND-like estimator.',
            'The joint block is a lightweight asynchronous-stage proxy rather than a first-principles CCDR derivation.',
        ],
    }
    save_json(args.outdir / 'test05_highz_a0_precursor_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
