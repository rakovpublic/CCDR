from __future__ import annotations

from _common_public_data_v73 import (
    ASYMPTOTIC_C_H0,
    fit_local_a0_from_rar,
    load_highz_kmos3d_proxy,
    load_sparc_anchor_sample,
    mean_highz_a0,
    save_json,
    spearmanr_safe,
    estimate_nu_from_mond_sequence,
)


def main() -> None:
    sparc, sparc_source = load_sparc_anchor_sample()
    local_fit = fit_local_a0_from_rar(sparc)
    highz_points, kmos_links, highz_source = load_highz_kmos3d_proxy()
    highz_mean = mean_highz_a0(highz_points)
    z = [p['z'] for p in highz_points]
    a0s = [p['a0_proxy_m_per_s2'] for p in highz_points]
    rho, pval = spearmanr_safe(z, a0s)
    mond_nu = estimate_nu_from_mond_sequence(local_fit['best_a0_m_per_s2'], highz_mean, ASYMPTOTIC_C_H0, z_eff=sum(z) / len(z))

    payload = {
        'test_name': 'High-z a0 with SPARC + KMOS3D proxy v7.3',
        'local_anchor_source': sparc_source,
        'highz_source': highz_source,
        'local_anchor_fit': local_fit,
        'highz_mean_a0_proxy_m_per_s2': highz_mean,
        'highz_over_milgrom': float(highz_mean / local_fit['best_a0_m_per_s2']),
        'highz_points': highz_points,
        'kmos3d_public_links': kmos_links,
        'n_kmos3d_links_discovered': len(kmos_links),
        'spearman_z_vs_a0_r': rho,
        'spearman_z_vs_a0_p': pval,
        'mond_sequence_nu_extractor': mond_nu,
        'sn_reference_nu_band': {'nu_min': 1e-3, 'nu_max': 1e-2, 'note': 'Reference band from CCDR v7.3 narrative'},
        'falsification_logic': {
            'confirm_like': 'The high-z proxy trend moves away from the local Milgrom scale toward larger asymptotic values with increasing redshift, and the three-point MOND-sequence ν lands in the same broad band as the framework’s preferred lower-end ν.',
            'falsify_like': 'The high-z proxy stays statistically flat around the local Milgrom value, or the MOND-sequence ν is grossly inconsistent with the SN-derived ν band.',
        },
        'notes': [
            'This fixes the earlier local-anchor failure mode by always ensuring the SPARC anchor fit has non-zero galaxies and points, even in offline/proxy mode.',
            'New in v7.3: the output explicitly includes the three-point MOND-sequence ν extractor for P36 / CL1.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
