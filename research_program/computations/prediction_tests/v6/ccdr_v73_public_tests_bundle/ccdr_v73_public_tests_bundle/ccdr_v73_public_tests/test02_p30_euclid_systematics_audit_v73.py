from __future__ import annotations

import argparse

from _common_public_data_v73 import enrich_density_catalog, fetch_sdss_galaxy_sample, high_low_split_stat, save_json


def subset_stat(df, mask, quantile: float = 0.25):
    sub = df.loc[mask].copy()
    if len(sub) < 20:
        return {'n': int(len(sub)), 'high_minus_low_kappa': float('nan')}
    out = high_low_split_stat(sub, 'kappa_act_proxy', quantile=quantile)
    out = {'high_minus_low_kappa': float(out['high_minus_low']), 'low_count': out['low_count'], 'high_count': out['high_count']}
    out['n'] = int(len(sub))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='P30 Euclid-Q1 systematics audit proxy, updated for v7.3 patchiness checks.')
    ap.add_argument('--max-rows', type=int, default=12000)
    args = ap.parse_args()

    gal, source_used = fetch_sdss_galaxy_sample(max_rows=args.max_rows)
    gal = enrich_density_catalog(gal)
    baseline = high_low_split_stat(gal, 'kappa_act_proxy', quantile=0.25)['high_minus_low']

    cases = []
    cases.append({'subset': 'z_low', **subset_stat(gal, gal['z'] <= gal['z'].median())})
    cases.append({'subset': 'z_high', **subset_stat(gal, gal['z'] > gal['z'].median())})
    cases.append({'subset': 'ra_low', **subset_stat(gal, gal['ra'] <= gal['ra'].median())})
    cases.append({'subset': 'ra_high', **subset_stat(gal, gal['ra'] > gal['ra'].median())})
    cases.append({'subset': 'dec_low', **subset_stat(gal, gal['dec'] <= gal['dec'].median())})
    cases.append({'subset': 'dec_high', **subset_stat(gal, gal['dec'] > gal['dec'].median())})
    for row in cases:
        row['delta_minus_baseline'] = float(row['high_minus_low_kappa'] - baseline)

    threshold_scan = []
    for q in (0.20, 0.25, 0.30, 0.35):
        threshold_scan.append({'quantile': q, 'high_minus_low_kappa': high_low_split_stat(gal, 'kappa_act_proxy', quantile=q)['high_minus_low']})

    patch_means = gal.groupby('patch_id')['kappa_act_proxy'].mean().to_dict()
    payload = {
        'test_name': 'P30 Euclid-Q1 systematics audit v7.3',
        'source_used': source_used,
        'n_objects': int(len(gal)),
        'baseline_high_minus_low_kappa': float(baseline),
        'subset_cases': cases,
        'threshold_scan': threshold_scan,
        'patch_mean_kappa': {str(k): float(v) for k, v in patch_means.items()},
        'patch_spread': float(gal.groupby('patch_id')['kappa_act_proxy'].mean().std()),
        'falsification_logic': {
            'confirm_like': 'The P30 sign remains stable across reasonable threshold choices. Residual RA/Dec/z asymmetries are small enough to be read as patchiness or reducing-volume structure rather than a one-field artifact.',
            'falsify_like': 'The signal localizes to one field-like split or collapses under modest threshold changes, indicating likely survey/systematics origin rather than physical density correlation.',
        },
        'notes': [
            'In v7.3, some split-to-split variation is not automatically fatal because the new prior allows genuine patch-to-patch DM-composition variation.',
            'This script therefore reports split asymmetries explicitly instead of auto-classifying every asymmetry as a survey systematic.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
