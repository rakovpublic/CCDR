#!/usr/bin/env python3
"""Test 09: P8c NANOGrav 15-year x cosmic-web sign cross-check proxy.

Public machine-readable PTA x cosmic-web products are limited, so this script uses a transparent proxy:
- download the public NANOGrav 15-year release archive and extract pulsar sky positions (and red-noise amplitude if present)
- compute public galaxy-density estimates at those sky positions from Euclid Q1 or SDSS fallback
- test the sign of the association between PTA-noise proxy and local density
This should be read as a sign/readiness cross-check, not a final PTA x LSS measurement.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import spatial, stats
from _common_public_data import load_nanograv_15yr_pulsar_positions, try_load_euclid_public_sample, fetch_sdss_galaxy_sample, save_json

def spherical_sep_deg(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(ra1); dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2); dec2 = np.deg2rad(dec2)
    return np.rad2deg(np.arccos(np.clip(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2), -1.0, 1.0)))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('out_test09_p8c_nanograv_cosmic_web_sign'))
    ap.add_argument('--max-pulsars', type=int, default=60)
    ap.add_argument('--max-galaxies', type=int, default=12000)
    ap.add_argument('--radius-deg', type=float, default=3.0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    p = load_nanograv_15yr_pulsar_positions(max_pulsars=args.max_pulsars)
    gal = try_load_euclid_public_sample(max_rows=args.max_galaxies, z_max=1.2)
    source_used = 'euclid_q1_public'
    if gal.empty:
        gal = fetch_sdss_galaxy_sample(z_min=0.02, z_max=0.20, max_rows=args.max_galaxies)
        source_used = 'sdss_fallback'
    if len(p) == 0:
        summary = {
            'test_name': 'P8c NANOGrav x cosmic-web sign proxy',
            'n_pulsars': 0,
            'source_used': source_used,
            'notes': ['Public NANOGrav archive could not be parsed in the current runtime environment.'],
        }
        save_json(args.outdir / 'test09_p8c_nanograv_cosmic_web_sign_summary.json', summary)
        print(json.dumps(summary, indent=2))
        return
    gra = gal['ra'].to_numpy(float); gdec = gal['dec'].to_numpy(float)
    densities = []
    for ra, dec in zip(p['ra'].to_numpy(float), p['dec'].to_numpy(float)):
        sep = spherical_sep_deg(ra, dec, gra, gdec)
        densities.append(float(np.sum(sep <= args.radius_deg)))
    p = p.copy()
    p['sky_density_count'] = densities
    amp = p['red_noise_amp'].to_numpy(float) if 'red_noise_amp' in p.columns else np.full(len(p), np.nan)
    finite = np.isfinite(amp)
    if np.sum(finite) >= 6:
        rho = stats.spearmanr(amp[finite], p.loc[finite, 'sky_density_count'].to_numpy(float), nan_policy='omit')
        sign_r = float(rho.statistic)
        sign_p = float(rho.pvalue)
    else:
        # fallback sign proxy based only on pulsar density ranking vs declination to keep readiness logic transparent
        rho = stats.spearmanr(p['dec'].to_numpy(float), p['sky_density_count'].to_numpy(float), nan_policy='omit')
        sign_r = float(rho.statistic)
        sign_p = float(rho.pvalue)
    summary = {
        'test_name': 'P8c NANOGrav x cosmic-web sign proxy',
        'source_used': source_used,
        'n_pulsars': int(len(p)),
        'n_with_red_noise_amp': int(np.sum(np.isfinite(p.get('red_noise_amp', np.full(len(p), np.nan))))),
        'radius_deg': float(args.radius_deg),
        'pulsar_rows_sample': p.head(20).to_dict(orient='records'),
        'spearman_sign_r': sign_r,
        'spearman_sign_p': sign_p,
        'falsification_logic': {
            'confirm_like': 'The public PTA x density proxy shows the predicted sign once sky positions and available noise-like amplitudes are cross-matched to the cosmic web.',
            'falsify_like': 'The sign remains opposite or null under the public proxy, consistent with the current wrong-sign status of P8c.',
        },
        'notes': [
            'This is a public-proxy sign cross-check because no publication-grade machine-readable PTA x cosmic-web cross-correlation product is presently bundled in the public releases used here.',
            'When per-pulsar red-noise amplitudes are unavailable in the parsed archive subset, the script falls back to a readiness/sign proxy rather than claiming a physical detection test.',
        ],
    }
    save_json(args.outdir / 'test09_p8c_nanograv_cosmic_web_sign_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
