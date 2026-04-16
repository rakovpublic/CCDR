#!/usr/bin/env python3
"""
Test 03: P29 time-varying DM abundance precursor via multi-redshift growth.

Updated implementation:
- keeps the compressed public DESI full-shape/RSD proxy fit
- adds a joint Test 03 + Test 05 asynchronous-stage fit that couples growth and
  a0(z) under a shared latent stage function with an allowed lag
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common_public_data import (
    estimate_a0_from_rotation_sample,
    fit_async_stage_joint_model,
    fit_dm_growth_proxy_from_fs8,
    load_desi_fullshape_proxy_points,
    load_highz_rotation_proxy_sample,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, default=Path('out_test03_p29_multi_redshift_growth_proxy'))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    points = load_desi_fullshape_proxy_points()
    fit = fit_dm_growth_proxy_from_fs8(points)

    highz = estimate_a0_from_rotation_sample(load_highz_rotation_proxy_sample())
    joint = fit_async_stage_joint_model(points, highz)
    joint_path = args.outdir / 'test03_test05_joint_async_stage_summary.json'
    save_json(joint_path, joint)

    summary = {
        'test_name': 'P29 multi-redshift growth proxy',
        'points': points.to_dict(orient='records'),
        'fit': fit,
        'joint_with_test05_async_stage': {
            'joint_summary_json': str(joint_path),
            **joint,
        },
        'falsification_logic': {
            'confirm_like': 'A nonzero growth-of-DM proxy term improves the fit to the public multi-redshift growth points with the sign expected for ongoing reduction, and the coupled async-stage fit also improves once a0(z) is included.',
            'falsify_like': 'A constant-DM growth history fits equally well, or the shared async-stage fit provides no improvement over separate baseline trends.',
        },
        'notes': [
            'This is a compressed public full-shape/RSD proxy using published DESI Y1 full-shape summary points.',
            'The joint block couples Test 03 and Test 05 through a shared latent stage with an allowed a0 lag, which is a lightweight proxy for asynchronous local stage completion.',
        ],
    }
    save_json(args.outdir / 'test03_p29_multi_redshift_growth_proxy_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
