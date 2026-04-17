#!/usr/bin/env python3
"""
Test 01: P30 density-correlated spatial variation in Ω_DM/Ω_B.

Updated public-data implementation:
- forces the public Euclid Q1 catalogue through IRSA TAP (no SDSS fallback by default)
- keeps the Planck-lensing discovery/proxy logic from the original readiness version
- adds a spatial variance map of κ residuals versus local density contrast on ~Mpc scales
- adds a joint Test 1 + Test 2 latent local-density crystal-ordering fit

This remains a proxy implementation because the current script still uses a simple
lensing-like sampler rather than a collaboration-grade κ map cross-power analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common_public_data import (
    estimate_filament_axes_knn,
    estimate_local_density_knn,
    fit_linear_trend,
    fit_shared_density_ordering_model,
    load_euclid_q1_public_sample,
    load_planck_lensing_like_map,
    local_axis_order_score,
    sample_planck_kappa_at_positions,
    save_json,
    sky_to_cartesian_mpc,
    summarize_binned_relation,
)


def make_spatial_grid_map(df: pd.DataFrame, n_ra: int = 12, n_dec: int = 12) -> pd.DataFrame:
    work = df.copy()
    ra_edges = np.linspace(float(work['ra'].min()), float(work['ra'].max()), n_ra + 1)
    dec_edges = np.linspace(float(work['dec'].min()), float(work['dec'].max()), n_dec + 1)
    work['ra_bin'] = pd.cut(work['ra'], bins=ra_edges, include_lowest=True, labels=False)
    work['dec_bin'] = pd.cut(work['dec'], bins=dec_edges, include_lowest=True, labels=False)
    out = (
        work.groupby(['ra_bin', 'dec_bin'], dropna=True)
        .agg(
            n=('ra', 'size'),
            ra_mean=('ra', 'mean'),
            dec_mean=('dec', 'mean'),
            z_mean=('z', 'mean'),
            mean_density_contrast=('density_contrast', 'mean'),
            mean_kappa=('kappa', 'mean'),
            mean_kappa_resid=('kappa_resid', 'mean'),
            std_kappa_resid=('kappa_resid', 'std'),
        )
        .reset_index()
    )
    out['std_kappa_resid'] = out['std_kappa_resid'].fillna(0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, default=Path('out_test01_p30_density_correlated_variation'))
    parser.add_argument('--max-rows', type=int, default=24000)
    parser.add_argument('--k-density', type=int, default=12)
    parser.add_argument('--variance-map-bins', type=int, default=10)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    gal = load_euclid_q1_public_sample(max_rows=args.max_rows, z_max=1.5, strict=True)
    source_used = gal.attrs.get('source_used', 'euclid_q1_public')

    pts = sky_to_cartesian_mpc(gal['ra'].to_numpy(float), gal['dec'].to_numpy(float), gal['z'].to_numpy(float))
    dens = estimate_local_density_knn(pts, k=args.k_density)
    logd = dens['log10_density']
    density_contrast = dens['density_contrast']

    planck_info = load_planck_lensing_like_map(sample_only=True)
    kappa = sample_planck_kappa_at_positions(gal['ra'].to_numpy(float), gal['dec'].to_numpy(float))

    q25, q75 = np.nanquantile(logd, [0.25, 0.75])
    high = logd >= q75
    low = logd <= q25

    high_mean = float(np.nanmean(kappa[high])) if np.any(high) else float('nan')
    low_mean = float(np.nanmean(kappa[low])) if np.any(low) else float('nan')
    delta = high_mean - low_mean
    t_p = float(stats.ttest_ind(kappa[high], kappa[low], equal_var=False, nan_policy='omit').pvalue) if np.sum(high) > 5 and np.sum(low) > 5 else None
    pearson = stats.pearsonr(logd, kappa) if len(logd) >= 5 else None
    spearman = stats.spearmanr(logd, kappa, nan_policy='omit') if len(logd) >= 5 else None

    rng = np.random.default_rng(12345)
    shuffled = []
    for _ in range(40):
        perm = rng.permutation(len(kappa))
        kk = kappa[perm]
        shuffled.append(float(np.nanmean(kk[high]) - np.nanmean(kk[low])))
    shuffled = np.asarray(shuffled, float)

    trend = fit_linear_trend(logd, kappa)
    kappa_fit = np.asarray(trend['y_fit'], float)
    kappa_resid = np.asarray(trend['residual'], float)
    resid_spearman = stats.spearmanr(density_contrast, kappa_resid, nan_policy='omit') if len(kappa_resid) >= 5 else None

    per_object = gal[['ra', 'dec', 'z']].copy()
    per_object['rk_mpc'] = dens['rk_mpc']
    per_object['density'] = dens['density']
    per_object['log10_density'] = logd
    per_object['density_contrast'] = density_contrast
    per_object['kappa'] = kappa
    per_object['kappa_fit'] = kappa_fit
    per_object['kappa_resid'] = kappa_resid
    per_object_path = args.outdir / 'test01_density_kappa_object_map.csv'
    per_object.to_csv(per_object_path, index=False)

    spatial_map = make_spatial_grid_map(per_object)
    spatial_map_path = args.outdir / 'test01_kappa_spatial_variance_map.csv'
    spatial_map.to_csv(spatial_map_path, index=False)

    density_contrast_profile = summarize_binned_relation(density_contrast, kappa_resid, n_bins=args.variance_map_bins)

    axes = estimate_filament_axes_knn(pts, k=12)
    local_order = local_axis_order_score(pts, axes, k=12)
    joint_fit = fit_shared_density_ordering_model(logd, kappa, local_order)
    order_spearman = stats.spearmanr(logd, local_order, nan_policy='omit') if len(local_order) >= 5 else None

    summary = {
        'test_name': 'P30 density-correlated spatial variation proxy',
        'source_used': source_used,
        'n_objects': int(len(gal)),
        'planck_lensing_links_discovered': planck_info['links'][:10],
        'n_planck_links_discovered': int(len(planck_info['links'])),
        'kappa_field_mode': 'planck_proxy_sampler',
        'high_density_count': int(np.sum(high)),
        'low_density_count': int(np.sum(low)),
        'high_minus_low_kappa': float(delta),
        'high_mean_kappa': float(high_mean),
        'low_mean_kappa': float(low_mean),
        'delta_vs_shuffle_z': float((delta - np.nanmean(shuffled)) / np.nanstd(shuffled)) if np.nanstd(shuffled) > 0 else None,
        'delta_vs_shuffle_mean': float(np.nanmean(shuffled)),
        'delta_vs_shuffle_std': float(np.nanstd(shuffled)),
        'pearson_r': float(pearson.statistic) if pearson else None,
        'pearson_p': float(pearson.pvalue) if pearson else None,
        'spearman_r': float(spearman.statistic) if spearman else None,
        'spearman_p': float(spearman.pvalue) if spearman else None,
        't_test_p_high_vs_low': t_p,
        'kappa_linear_trend_vs_log10_density': {
            'intercept': float(trend['intercept']),
            'slope': float(trend['slope']),
            'chi2_like': float(trend['chi2']),
        },
        'spatial_variance_map': {
            'object_map_csv': str(per_object_path),
            'grid_map_csv': str(spatial_map_path),
            'density_contrast_vs_kappa_residual_profile': density_contrast_profile,
            'spearman_density_contrast_vs_kappa_resid_r': float(resid_spearman.statistic) if resid_spearman else None,
            'spearman_density_contrast_vs_kappa_resid_p': float(resid_spearman.pvalue) if resid_spearman else None,
        },
        'joint_with_test02_density_ordering_model': {
            'local_order_spearman_r': float(order_spearman.statistic) if order_spearman else None,
            'local_order_spearman_p': float(order_spearman.pvalue) if order_spearman else None,
            **joint_fit,
        },
        'falsification_logic': {
            'confirm_like': 'High-density regions show systematically higher inferred lensing-like signal than low-density regions, with the predicted sign, a stable shuffled-null excess, and a coherent shared density-ordering fit with Test 02.',
            'falsify_like': 'No positive density-correlated excess appears once null controls are applied, or the shared local-density ordering model fails to couple the κ-like field and filament-order score.',
        },
        'notes': [
            'This version forces the public Euclid Q1 sample through IRSA TAP instead of defaulting to the earlier SDSS fallback.',
            'The κ field is still a public-data proxy sampler rather than a full lensing map likelihood.',
            'The new spatial variance map records κ residuals against local density contrast on approximately Mpc scales via the k-nearest-neighbour density proxy.',
        ],
    }
    save_json(args.outdir / 'test01_p30_density_correlated_variation_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
