#!/usr/bin/env python3
"""Test 06: P29 public multi-redshift growth test.

Uses public DESI full-shape/RSD proxy points and fits a simple extra growth-of-DM term. This is a compressed public proxy,
not a full Boltzmann/EFT-LSS implementation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from _common_public_data import load_desi_fullshape_proxy_points, fit_dm_growth_proxy_from_fs8, save_json

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('out_test06_p29_multi_redshift_growth'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    pts = load_desi_fullshape_proxy_points()
    fit = fit_dm_growth_proxy_from_fs8(pts)
    summary = {
        'test_name': 'P29 multi-redshift growth test',
        'points': pts.to_dict(orient='records'),
        'fit': fit,
        'falsification_logic': {
            'confirm_like': 'A positive growth-of-DM proxy term improves the fit to public multi-redshift growth points with the sign expected for ongoing reduction.',
            'falsify_like': 'A constant-DM growth history fits equally well, or the preferred proxy term remains zero or wrong-sign.',
        },
        'notes': [
            'This is a compressed public full-shape/RSD proxy based on published DESI summary points.',
            'A full perturbation-level implementation with public chains would be the next refinement once the proxy sign is established.',
        ],
    }
    save_json(args.outdir / 'test06_p29_multi_redshift_growth_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
