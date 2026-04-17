#!/usr/bin/env python3
"""Test 01: P30 replication using Euclid Q1 galaxy density against public ACT/Planck lensing access paths.

This is a public-data proxy implementation. It discovers official ACT/Planck lensing pages and, if no directly
readable map is obtained in the runtime environment, evaluates deterministic ACT/Planck sky-proxy fields on the
same Euclid/SDSS coordinates so the replication, null controls, and sign logic still run end-to-end.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import spatial, stats
from _common_public_data import (
    try_load_euclid_public_sample, fetch_sdss_galaxy_sample, sky_to_cartesian_mpc,
    load_act_lensing_links, load_planck_lensing_like_map, sample_act_kappa_at_positions,
    sample_planck_kappa_at_positions, save_json,
)

def _density_log(points: np.ndarray, k: int = 12) -> np.ndarray:
    tree = spatial.cKDTree(points)
    d, _ = tree.query(points, k=min(k + 1, len(points)))
    if d.ndim == 1:
        d = d[:, None]
    rk = np.clip(d[:, -1], 1e-12, None)
    dens = k / ((4.0 / 3.0) * np.pi * rk**3)
    return np.log10(np.clip(dens, 1e-30, None))

def _run_one(name: str, kappa: np.ndarray, logd: np.ndarray) -> dict:
    q25, q75 = np.nanquantile(logd, [0.25, 0.75])
    high = logd >= q75
    low = logd <= q25
    delta = float(np.nanmean(kappa[high]) - np.nanmean(kappa[low]))
    rng = np.random.default_rng(12345)
    null = []
    for _ in range(64):
        kk = kappa[rng.permutation(len(kappa))]
        null.append(float(np.nanmean(kk[high]) - np.nanmean(kk[low])))
    null = np.asarray(null)
    pear = stats.pearsonr(logd, kappa)
    spear = stats.spearmanr(logd, kappa, nan_policy='omit')
    return {
        'name': name,
        'high_minus_low_kappa': delta,
        'high_count': int(np.sum(high)),
        'low_count': int(np.sum(low)),
        'pearson_r': float(pear.statistic),
        'pearson_p': float(pear.pvalue),
        'spearman_r': float(spear.statistic),
        'spearman_p': float(spear.pvalue),
        'delta_vs_shuffle_mean': float(np.nanmean(null)),
        'delta_vs_shuffle_std': float(np.nanstd(null, ddof=1)),
        'delta_vs_shuffle_z': float((delta - np.nanmean(null)) / np.nanstd(null, ddof=1)) if np.nanstd(null, ddof=1) > 0 else None,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('out_test01_p30_non_euclid_lensing_replication'))
    ap.add_argument('--max-rows', type=int, default=12000)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    gal = try_load_euclid_public_sample(max_rows=args.max_rows, z_max=1.5)
    source_used = 'euclid_q1_public'
    if gal.empty:
        gal = fetch_sdss_galaxy_sample(z_min=0.02, z_max=0.20, max_rows=args.max_rows)
        source_used = 'sdss_fallback'
    pts = sky_to_cartesian_mpc(gal['ra'].to_numpy(float), gal['dec'].to_numpy(float), gal['z'].to_numpy(float))
    logd = _density_log(pts)
    act_links = load_act_lensing_links()
    planck_info = load_planck_lensing_like_map(sample_only=True)
    act = sample_act_kappa_at_positions(gal['ra'].to_numpy(float), gal['dec'].to_numpy(float))
    planck = sample_planck_kappa_at_positions(gal['ra'].to_numpy(float), gal['dec'].to_numpy(float))
    summary = {
        'test_name': 'P30 non-Euclid lensing replication proxy',
        'source_used': source_used,
        'n_objects': int(len(gal)),
        'act_links_discovered': act_links[:20],
        'n_act_links_discovered': int(len(act_links)),
        'planck_links_discovered': planck_info.get('links', [])[:20],
        'n_planck_links_discovered': int(len(planck_info.get('links', []))),
        'act_result': _run_one('act_proxy', act, logd),
        'planck_result': _run_one('planck_proxy', planck, logd),
        'falsification_logic': {
            'confirm_like': 'The density-correlated signal reappears with the predicted sign under independent public lensing access paths and survives shuffled nulls.',
            'falsify_like': 'The sign flips or vanishes under independent lensing access paths, or the apparent effect collapses under null controls.',
        },
        'notes': [
            'This is a replication proxy using official public ACT/Planck access paths when discoverable.',
            'If no directly readable lensing map is obtained at runtime, deterministic sky-proxy samplers are used so the sign/null logic can still be tested.',
        ],
    }
    save_json(args.outdir / 'test01_p30_non_euclid_lensing_replication_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
