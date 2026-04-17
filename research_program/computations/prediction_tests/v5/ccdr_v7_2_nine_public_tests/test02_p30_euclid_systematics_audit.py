#!/usr/bin/env python3
"""Test 02: Euclid-Q1 internal systematics audit of the P30 density-kappa signal.

This uses the same public galaxy sample and a Planck-like kappa proxy, then checks whether the sign and size of
the density-correlated signal remain stable across simple field-like spatial splits, redshift splits, and density-threshold changes.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import spatial
from _common_public_data import try_load_euclid_public_sample, fetch_sdss_galaxy_sample, sky_to_cartesian_mpc, sample_planck_kappa_at_positions, save_json

def _density_log(points, k=12):
    tree = spatial.cKDTree(points)
    d, _ = tree.query(points, k=min(k + 1, len(points)))
    if d.ndim == 1:
        d = d[:, None]
    rk = np.clip(d[:, -1], 1e-12, None)
    return np.log10(np.clip(k / ((4.0 / 3.0) * np.pi * rk**3), 1e-30, None))

def _delta(logd, kappa, qlo=0.25, qhi=0.75):
    a, b = np.nanquantile(logd, [qlo, qhi])
    high = logd >= b
    low = logd <= a
    if np.sum(high) < 20 or np.sum(low) < 20:
        return None
    return float(np.nanmean(kappa[high]) - np.nanmean(kappa[low]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('out_test02_p30_euclid_systematics_audit'))
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
    kappa = sample_planck_kappa_at_positions(gal['ra'].to_numpy(float), gal['dec'].to_numpy(float))
    base = _delta(logd, kappa)
    zmed = float(np.nanmedian(gal['z']))
    ramed = float(np.nanmedian(gal['ra']))
    decmed = float(np.nanmedian(gal['dec']))
    cases = []
    masks = {
        'z_low': gal['z'].to_numpy(float) < zmed,
        'z_high': gal['z'].to_numpy(float) >= zmed,
        'ra_low': gal['ra'].to_numpy(float) < ramed,
        'ra_high': gal['ra'].to_numpy(float) >= ramed,
        'dec_low': gal['dec'].to_numpy(float) < decmed,
        'dec_high': gal['dec'].to_numpy(float) >= decmed,
    }
    for name, mask in masks.items():
        if int(np.sum(mask)) < 200:
            continue
        d = _delta(logd[mask], kappa[mask])
        cases.append({'subset': name, 'n': int(np.sum(mask)), 'high_minus_low_kappa': d, 'delta_minus_baseline': None if d is None or base is None else float(d - base)})
    thresholds = []
    for q in [0.2, 0.25, 0.3, 0.35]:
        d = _delta(logd, kappa, qlo=q, qhi=1-q)
        thresholds.append({'quantile': q, 'high_minus_low_kappa': d})
    summary = {
        'test_name': 'P30 Euclid-Q1 systematics audit proxy',
        'source_used': source_used,
        'n_objects': int(len(gal)),
        'baseline_high_minus_low_kappa': base,
        'subset_cases': cases,
        'threshold_scan': thresholds,
        'falsification_logic': {
            'confirm_like': 'The P30 sign remains stable across spatial and redshift splits and across reasonable density-threshold choices.',
            'falsify_like': 'The signal localizes to one field-like split or collapses under modest threshold changes, indicating likely survey/systematics origin.',
        },
        'notes': [
            'This is an internal audit proxy using publicly accessible columns rather than a full survey-quality photo-z and mask systematics pipeline.',
            'Field-like splits are approximated by RA/Dec partitions when explicit Euclid field labels are unavailable in the queried sample.',
        ],
    }
    save_json(args.outdir / 'test02_p30_euclid_systematics_audit_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
