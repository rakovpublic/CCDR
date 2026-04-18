from __future__ import annotations

import argparse
import numpy as np

from _common_public_data_v73 import (
    enrich_density_catalog,
    fetch_sdss_galaxy_sample,
    filament_orientation_correlation,
    fit_exp_profile,
    orientation_vectors,
    save_json,
)


def run_subset(name, df, bins):
    pts = np.asarray(df[['x', 'y', 'zc']], dtype=float)
    axes = orientation_vectors(pts, k=12)
    stats = filament_orientation_correlation(pts, axes, r_bins=bins)
    stats['exp_fit'] = fit_exp_profile(stats['r_mid_mpc_over_h'], stats['corr'])
    stats['n_galaxies'] = int(len(df))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description='P3 filament ordering with density dependence under the v7.3 prior.')
    ap.add_argument('--max-rows', type=int, default=1200)
    args = ap.parse_args()

    gal, source_used = fetch_sdss_galaxy_sample(z_max=0.12, max_rows=args.max_rows)
    gal = enrich_density_catalog(gal)
    pts = np.asarray(gal[['ra', 'dec', 'z']], dtype=float)
    from _common_public_data_v73 import sky_to_cartesian_mpc
    xyz = sky_to_cartesian_mpc(pts[:, 0], pts[:, 1], pts[:, 2])
    gal['x'], gal['y'], gal['zc'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    bins = np.arange(20.0, 300.0, 20.0)
    qlo = gal['density_proxy'].quantile(0.25)
    qhi = gal['density_proxy'].quantile(0.75)

    full = run_subset('full', gal, bins)
    low = run_subset('low_density', gal[gal['density_proxy'] <= qlo].copy(), bins)
    high = run_subset('high_density', gal[gal['density_proxy'] >= qhi].copy(), bins)

    obs = np.nanmean(high['corr'][:6]) - np.nanmean(low['corr'][:6])
    rng = np.random.default_rng(73)
    nulls = []
    density_bins = gal['density_bin'].to_numpy(dtype=int, copy=True)
    base_density = gal['density_proxy'].to_numpy(dtype=float, copy=True)
    for _ in range(12):
        sh = gal.copy()
        shuffled_density = base_density.copy()
        for b in np.unique(density_bins):
            idx = np.where(density_bins == b)[0]
            vals = shuffled_density[idx].copy()
            rng.shuffle(vals)
            shuffled_density[idx] = vals
        sh['density_proxy'] = shuffled_density
        qlo_s = sh['density_proxy'].quantile(0.25)
        qhi_s = sh['density_proxy'].quantile(0.75)
        low_s = run_subset('low_density_shuffle', sh[sh['density_proxy'] <= qlo_s].copy(), bins)
        high_s = run_subset('high_density_shuffle', sh[sh['density_proxy'] >= qhi_s].copy(), bins)
        nulls.append(np.nanmean(high_s['corr'][:6]) - np.nanmean(low_s['corr'][:6]))
    null_mean = float(np.mean(nulls))
    null_std = float(np.std(nulls, ddof=1))

    payload = {
        'test_name': 'Euclid-Q1 filament ordering v7.3',
        'source_used': source_used,
        'full_sample': full,
        'low_density': low,
        'high_density': high,
        'density_dependence_proxy': {
            'observed_high_minus_low_mean_corr_20_120_mpc_h': float(obs),
            'null_mean': null_mean,
            'null_std': null_std,
            'observed_z_vs_density_stratified_null': float((obs - null_mean) / max(null_std, 1e-12)),
            'n_null_draws': 12,
        },
        'falsification_logic': {
            'confirm_like': 'A large-scale orientational signal reappears and its amplitude depends on density in the direction expected from the reducing-volume prior.',
            'falsify_like': 'No stable large-scale ordering appears, or the apparent density dependence collapses under density-stratified null controls.',
        },
        'notes': [
            'This remains an independent transparent-finder proxy, not a Bisous/Tempel reproduction.',
            'New in v7.3: the test explicitly asks whether filament ordering is stronger in denser patches, as expected if cross-patch composition varies under the dimension-origin prior.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
