from __future__ import annotations

import numpy as np

from _common_public_data_v73 import (
    enrich_density_catalog,
    estimate_nu_from_mond_sequence,
    fetch_sdss_galaxy_sample,
    fit_local_a0_from_rar,
    load_bao_summary_points,
    load_highz_kmos3d_proxy,
    load_sparc_anchor_sample,
    mean_highz_a0,
    save_json,
)


def main() -> None:
    gal, source_used = fetch_sdss_galaxy_sample(max_rows=12000)
    gal = enrich_density_catalog(gal)
    bao = load_bao_summary_points()
    sparc, _ = load_sparc_anchor_sample()
    highz_points, _, _ = load_highz_kmos3d_proxy()
    local_fit = fit_local_a0_from_rar(sparc)
    nu_mond = estimate_nu_from_mond_sequence(local_fit['best_a0_m_per_s2'], mean_highz_a0(highz_points))['nu_mond_sequence']

    qlo = gal['density_proxy'].quantile(0.25)
    qhi = gal['density_proxy'].quantile(0.75)
    rho_low = float(gal.loc[gal['density_proxy'] <= qlo, 'density_proxy'].mean())
    rho_high = float(gal.loc[gal['density_proxy'] >= qhi, 'density_proxy'].mean())
    dr_over_r = float(nu_mond * (rho_high - rho_low) / max(abs(gal['density_proxy'].mean()), 1e-12) * 1e-2)
    rd_baseline = float(np.mean([row['rd_mpc'] for row in bao]))

    payload = {
        'test_name': 'P33 density-correlated BAO v7.3',
        'source_used': source_used,
        'bao_summary_points': bao,
        'density_bins_match_kappa_pipeline': True,
        'density_proxy_means': {'low': rho_low, 'high': rho_high},
        'nu_input_from_mond_sequence': float(nu_mond),
        'predicted_delta_r_over_r': dr_over_r,
        'predicted_delta_rd_mpc': float(rd_baseline * dr_over_r),
        'baseline_rd_mpc': rd_baseline,
        'falsification_logic': {
            'confirm_like': 'The density-stratified BAO proxy shifts in the same direction as the κ-based density ordering, supporting early reduction in denser patches.',
            'falsify_like': 'The inferred sign or scale of δr/r is inconsistent with the ν and density ordering preferred by the κ pipeline.',
        },
        'notes': [
            'This is a sister proxy to P30 rather than a full density-binned BAO likelihood. Public summary tables do not carry the full bin-resolved clustering information needed for a collaboration-grade measurement.',
            'The density binning is intentionally matched to the κ pipeline so the test can function as a v7.3 consistency check.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
