#!/usr/bin/env python3
"""Test 05: Euclid-Q1 filament ordering with density-stratified null controls.

This is a transparent-finder replication of P3 in its v7.2 form. It uses a kNN local-PCA axis estimator, density splits,
and a density-stratified axis shuffle that preserves the coarse density distribution while destroying within-bin orientational coherence.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import spatial
from _common_public_data import try_load_euclid_public_sample, fetch_sdss_galaxy_sample, sky_to_cartesian_mpc, estimate_filament_axes_knn, filament_orientation_correlation, fit_exponential_correlation, save_json

def density_proxy(points: np.ndarray, k: int = 12) -> np.ndarray:
    tree = spatial.cKDTree(points)
    d, _ = tree.query(points, k=min(k + 1, len(points)))
    if d.ndim == 1:
        d = d[:, None]
    rk = np.clip(d[:, -1], 1e-12, None)
    return k / ((4.0 / 3.0) * np.pi * rk**3)

def run_corr(gal, bins):
    pts = sky_to_cartesian_mpc(gal['ra'].to_numpy(float), gal['dec'].to_numpy(float), gal['z'].to_numpy(float))
    axes = estimate_filament_axes_knn(pts, k_neighbors=12)
    res = filament_orientation_correlation(pts, axes, r_bins=bins)
    fit = fit_exponential_correlation(np.asarray(res['r_mid_mpc_over_h'], float), np.asarray(res['corr'], float), np.asarray(res['stderr'], float))
    res['exp_fit'] = fit
    res['n_galaxies'] = int(len(gal))
    return pts, axes, res

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('out_test05_p3_euclid_filament_ordering'))
    ap.add_argument('--max-rows', type=int, default=18000)
    ap.add_argument('--null-draws', type=int, default=32)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    gal = try_load_euclid_public_sample(max_rows=args.max_rows, z_max=1.2)
    source_used = 'euclid_q1_public'
    if gal.empty:
        gal = fetch_sdss_galaxy_sample(z_min=0.02, z_max=0.12, max_rows=args.max_rows)
        source_used = 'sdss_fallback'
    bins = np.arange(20.0, 281.0, 20.0)
    pts, axes, full = run_corr(gal, bins)
    dens = density_proxy(pts)
    logd = np.log10(np.clip(dens, 1e-30, None))
    q25, q75 = np.nanquantile(logd, [0.25, 0.75])
    mask_high = logd >= q75
    mask_low = logd <= q25
    _, _, high = run_corr(gal.loc[mask_high].reset_index(drop=True), bins) if np.sum(mask_high) > 1000 else (None,None,None)
    _, _, low = run_corr(gal.loc[mask_low].reset_index(drop=True), bins) if np.sum(mask_low) > 1000 else (None,None,None)
    # density-stratified null control on full sample
    rng = np.random.default_rng(1234)
    nbin_edges = np.nanquantile(logd, [0.0, 0.25, 0.5, 0.75, 1.0])
    null_means = []
    real_corr = np.asarray(full['corr'], float)
    for _ in range(args.null_draws):
        shuf_axes = axes.copy()
        for i in range(len(nbin_edges)-1):
            m = (logd >= nbin_edges[i]) & (logd <= nbin_edges[i+1] if i == len(nbin_edges)-2 else logd < nbin_edges[i+1])
            idx = np.where(m)[0]
            if len(idx) > 1:
                shuf_axes[idx] = shuf_axes[rng.permutation(idx)]
        rr = filament_orientation_correlation(pts, shuf_axes, r_bins=bins)
        null_means.append(rr['corr'])
    null_mat = np.asarray(null_means, float)
    # compare 20-120 Mpc/h average corr high-low
    mids = np.asarray(full['r_mid_mpc_over_h'], float)
    win = (mids >= 20) & (mids <= 120)
    obs_diff = None
    z_vs_null = None
    if high and low:
        hc = np.asarray(high['corr'], float)
        lc = np.asarray(low['corr'], float)
        obs_diff = float(np.nanmean(hc[win]) - np.nanmean(lc[win]))
        null_diff = []
        for row in null_mat:
            null_diff.append(float(np.nanmean(row[win]) - np.nanmean(row[win])))
        # use full null std as conservative baseline
        z_vs_null = float((np.nanmean(np.asarray(high['corr'], float)[win]) - np.nanmean(null_mat[:, win])) / np.nanstd(null_mat[:, win])) if np.nanstd(null_mat[:, win]) > 0 else None
    summary = {
        'test_name': 'Euclid-Q1 filament ordering test',
        'source_used': source_used,
        'full_sample': full,
        'high_density': high,
        'low_density': low,
        'density_stratified_null_control': {
            'n_null_draws': int(args.null_draws),
            'null_mean': float(np.nanmean(null_mat)),
            'null_std': float(np.nanstd(null_mat, ddof=1)),
            'observed_high_minus_low_mean_corr_20_120_mpc_h': obs_diff,
            'observed_z_vs_density_stratified_null': z_vs_null,
        },
        'falsification_logic': {
            'confirm_like': 'A positive orientational correlation with characteristic scale above ~100 Mpc/h reappears and survives density-stratified shuffles of filament axes.',
            'falsify_like': 'No stable large-scale correlation appears, or the apparent ordering collapses under density-stratified null controls.',
        },
        'notes': [
            'This is an independent transparent-finder replication, not a Bisous/Tempel catalogue reproduction.',
            'The density-stratified null preserves the coarse density distribution while breaking within-bin orientational coherence.',
        ],
    }
    save_json(args.outdir / 'test05_p3_euclid_filament_ordering_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
