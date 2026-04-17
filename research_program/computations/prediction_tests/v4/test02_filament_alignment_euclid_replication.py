#!/usr/bin/env python3
"""
Test 02: independent replication of filament alignment (legacy P3 stress test).

Updated public implementation:
- forces the public Euclid Q1 catalogue through IRSA TAP
- compares full/high-density/low-density subsamples
- adds a density-stratified null control that shuffles filament axes only within
  local-density bins to test whether the high-density excess survives beyond a
  coarse density-selection degeneracy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common_public_data import (
    estimate_filament_axes_knn,
    estimate_local_density_knn,
    filament_orientation_correlation,
    fit_exponential_correlation,
    load_euclid_q1_public_sample,
    local_axis_order_score,
    save_json,
    sky_to_cartesian_mpc,
    stratified_permutation_indices,
)


def run_subset_from_arrays(name: str, pts: np.ndarray, axes: np.ndarray, bins: np.ndarray) -> dict:
    stats = filament_orientation_correlation(pts, axes, r_bins=bins)
    mids = np.asarray(stats['r_mid_mpc_over_h'], dtype=float)
    corr = np.asarray(stats['corr'], dtype=float)
    stderr = np.asarray(stats['stderr'], dtype=float)
    fit = fit_exponential_correlation(mids, corr, stderr)
    return {
        'name': name,
        'n_galaxies': int(len(pts)),
        'r_mid_mpc_over_h': [float(x) for x in mids],
        'corr': [float(x) if np.isfinite(x) else None for x in corr],
        'stderr': [float(x) if np.isfinite(x) else None for x in stderr],
        'pair_counts': [int(x) for x in stats.get('counts', [])],
        'exp_fit': fit,
    }


def scalar_alignment_stat(result: dict, max_bin_mid: float = 120.0) -> float:
    mids = np.asarray(result['r_mid_mpc_over_h'], float)
    corr = np.asarray([np.nan if v is None else float(v) for v in result['corr']], float)
    mask = np.isfinite(mids) & np.isfinite(corr) & (mids <= max_bin_mid)
    if not np.any(mask):
        return float('nan')
    return float(np.nanmean(corr[mask]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, default=Path('out_test02_filament_alignment_euclid_replication'))
    parser.add_argument('--max-rows', type=int, default=18000)
    parser.add_argument('--n-density-nulls', type=int, default=32)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    gal = load_euclid_q1_public_sample(max_rows=args.max_rows, z_max=1.2, strict=True)
    source_used = gal.attrs.get('source_used', 'euclid_q1_public')

    pts = sky_to_cartesian_mpc(gal['ra'].to_numpy(float), gal['dec'].to_numpy(float), gal['z'].to_numpy(float))
    dens = estimate_local_density_knn(pts, k=12)
    logd = dens['log10_density']
    axes = estimate_filament_axes_knn(pts, k=12)
    local_order = local_axis_order_score(pts, axes, k=12)

    q25, q75 = np.nanquantile(logd, [0.25, 0.75])
    mask_high = logd >= q75
    mask_low = logd <= q25
    bins = np.arange(20.0, 281.0, 20.0)

    full = run_subset_from_arrays('all', pts, axes, bins)
    high = run_subset_from_arrays('high_density', pts[mask_high], axes[mask_high], bins) if np.sum(mask_high) > 1000 else None
    low = run_subset_from_arrays('low_density', pts[mask_low], axes[mask_low], bins) if np.sum(mask_low) > 1000 else None

    high_stat = scalar_alignment_stat(high) if high is not None else float('nan')
    low_stat = scalar_alignment_stat(low) if low is not None else float('nan')
    high_minus_low = float(high_stat - low_stat) if np.isfinite(high_stat) and np.isfinite(low_stat) else float('nan')

    rng = np.random.default_rng(24680)
    null_stats = []
    for _ in range(args.n_density_nulls):
        perm = stratified_permutation_indices(logd, n_bins=8, rng=rng)
        axes_perm = axes[perm]
        high_perm = run_subset_from_arrays('high_density_perm', pts[mask_high], axes_perm[mask_high], bins) if np.sum(mask_high) > 1000 else None
        low_perm = run_subset_from_arrays('low_density_perm', pts[mask_low], axes_perm[mask_low], bins) if np.sum(mask_low) > 1000 else None
        hs = scalar_alignment_stat(high_perm) if high_perm is not None else float('nan')
        ls = scalar_alignment_stat(low_perm) if low_perm is not None else float('nan')
        null_stats.append(float(hs - ls) if np.isfinite(hs) and np.isfinite(ls) else float('nan'))
    null_stats = np.asarray(null_stats, float)

    object_map = pd.DataFrame({
        'ra': gal['ra'].to_numpy(float),
        'dec': gal['dec'].to_numpy(float),
        'z': gal['z'].to_numpy(float),
        'rk_mpc': dens['rk_mpc'],
        'density': dens['density'],
        'log10_density': logd,
        'local_order_score': local_order,
    })
    object_map_path = args.outdir / 'test02_filament_density_object_map.csv'
    object_map.to_csv(object_map_path, index=False)

    summary = {
        'test_name': 'Independent filament alignment replication',
        'source_used': source_used,
        'full_sample': full,
        'high_density': high,
        'low_density': low,
        'density_stratified_null_control': {
            'object_map_csv': str(object_map_path),
            'observed_high_minus_low_mean_corr_20_120_mpc_h': float(high_minus_low) if np.isfinite(high_minus_low) else None,
            'null_mean': float(np.nanmean(null_stats)) if np.any(np.isfinite(null_stats)) else None,
            'null_std': float(np.nanstd(null_stats)) if np.any(np.isfinite(null_stats)) else None,
            'observed_z_vs_density_stratified_null': float((high_minus_low - np.nanmean(null_stats)) / np.nanstd(null_stats)) if np.nanstd(null_stats) > 0 and np.isfinite(high_minus_low) else None,
            'n_null_draws': int(args.n_density_nulls),
        },
        'falsification_logic': {
            'confirm_like': 'A positive orientational correlation with characteristic scale above ~100 Mpc/h reappears, is stronger in denser environments, and survives density-stratified shuffles of filament axes.',
            'falsify_like': 'No stable large-scale correlation appears, or the apparent high-density enhancement collapses under density-stratified null controls.',
        },
        'notes': [
            'This is an independent transparent-finder replication, not a Bisous/Tempel catalogue reproduction.',
            'The script now forces the public Euclid Q1 sample instead of defaulting to an SDSS fallback path.',
            'The new null control preserves the coarse density distribution while breaking within-bin orientational coherence.',
        ],
    }
    save_json(args.outdir / 'test02_filament_alignment_euclid_replication_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
