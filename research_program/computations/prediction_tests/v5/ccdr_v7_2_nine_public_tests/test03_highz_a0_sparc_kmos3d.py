#!/usr/bin/env python3
"""Test 03: high-z a0 with public SPARC anchor and KMOS3D public proxy sample.

This extends the local SPARC anchor to a public high-z ionised-gas proxy sample. Because public high-z products rarely expose full
SPARC-grade baryonic decompositions, the script uses a coarse MOND-like estimator and explicitly reports the result as suggestive/proxy-level.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from scipy import stats
import numpy as np
from _common_public_data import load_highz_rotation_proxy_sample, estimate_a0_from_rotation_sample, load_sparc_rar, fit_rar_hierarchical_like, scrape_links, save_json

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('out_test03_highz_a0_sparc_kmos3d'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    # public links
    kmos_links = scrape_links('https://www.mpe.mpg.de/ir/KMOS3D/data', pattern=r'fits|cube|catalog|txt|dat|tbl|pdf')
    rar = load_sparc_rar().copy()
    if 'Galaxy' in rar.columns:
        gcol = 'Galaxy'
    elif 'galaxy' in rar.columns:
        gcol = 'galaxy'
    else:
        rar['__all__'] = '__all__'
        gcol = '__all__'
    local_fit = fit_rar_hierarchical_like(rar, gcol, 'gobs', 'gbar', offset_prior_dex=0.08)
    hz = estimate_a0_from_rotation_sample(load_highz_rotation_proxy_sample())
    rho = stats.spearmanr(hz['z'].to_numpy(float), hz['a0_proxy_m_per_s2'].to_numpy(float), nan_policy='omit')
    milgrom = 1.2e-10
    mean_a0 = float(np.nanmean(hz['a0_proxy_m_per_s2']))
    summary = {
        'test_name': 'High-z a0 with SPARC + KMOS3D proxy',
        'kmos3d_public_links': kmos_links[:30],
        'n_kmos3d_links_discovered': int(len(kmos_links)),
        'local_anchor_fit': local_fit,
        'highz_points': hz.to_dict(orient='records'),
        'highz_mean_a0_proxy_m_per_s2': mean_a0,
        'highz_over_milgrom': float(mean_a0 / milgrom),
        'spearman_z_vs_a0_r': float(rho.statistic),
        'spearman_z_vs_a0_p': float(rho.pvalue),
        'falsification_logic': {
            'confirm_like': 'The high-z proxy trend moves away from the local Milgrom scale toward larger asymptotic values with increasing redshift.',
            'falsify_like': 'The high-z proxy stays statistically flat around the local Milgrom value once the public sample is re-estimated.',
        },
        'notes': [
            'The local anchor is the public SPARC machine-readable RAR table.',
            'The high-z component is a public-proxy implementation because machine-readable high-z baryonic decompositions are limited.',
        ],
    }
    save_json(args.outdir / 'test03_highz_a0_sparc_kmos3d_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
