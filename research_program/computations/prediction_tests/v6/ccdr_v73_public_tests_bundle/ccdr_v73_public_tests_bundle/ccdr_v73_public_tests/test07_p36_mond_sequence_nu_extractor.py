from __future__ import annotations

from _common_public_data_v73 import (
    ASYMPTOTIC_C_H0,
    estimate_nu_from_mond_sequence,
    fit_local_a0_from_rar,
    load_highz_kmos3d_proxy,
    load_sparc_anchor_sample,
    mean_highz_a0,
    save_json,
)


def main() -> None:
    sparc, sparc_source = load_sparc_anchor_sample()
    local_fit = fit_local_a0_from_rar(sparc)
    highz_points, links, highz_source = load_highz_kmos3d_proxy()
    highz_mean = mean_highz_a0(highz_points)
    result = estimate_nu_from_mond_sequence(local_fit['best_a0_m_per_s2'], highz_mean, ASYMPTOTIC_C_H0, z_eff=sum(p['z'] for p in highz_points) / len(highz_points))
    payload = {
        'test_name': 'P36 MOND-sequence nu extractor',
        'local_anchor_source': sparc_source,
        'highz_source': highz_source,
        'highz_links': links,
        'local_anchor_fit': local_fit,
        'highz_mean_a0_m_per_s2': highz_mean,
        'asymptotic_cH0_m_per_s2': ASYMPTOTIC_C_H0,
        'mond_sequence_result': result,
        'sn_reference_band': {'nu_min': 1e-3, 'nu_max': 1e-2},
        'falsification_logic': {
            'confirm_like': 'The three-point a0(z) sequence yields a ν_MOND in the same broad band as the framework’s preferred lower-end ν, supporting CL1.',
            'falsify_like': 'The three-point sequence forces ν_MOND far outside the SN/κ-motivated band, weakening the v7.3 cross-framework bridge.',
        },
        'notes': [
            'This is the new immediate-execution v7.3 / v3.3 test called out as P36 and CL1.',
            'The mapping from a0(z) to ν is a screening-level parametric extractor, not yet a first-principles derivation.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
