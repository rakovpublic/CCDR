from __future__ import annotations

from _common_public_data_v73 import (
    estimate_nu_from_mond_sequence,
    estimate_time_crystal_q_proxy,
    fallback_pantheon_cases,
    fit_local_a0_from_rar,
    load_highz_kmos3d_proxy,
    load_sparc_anchor_sample,
    mean_highz_a0,
    save_json,
)


def main() -> None:
    sn_cases = fallback_pantheon_cases()
    sparc, _ = load_sparc_anchor_sample()
    local_fit = fit_local_a0_from_rar(sparc)
    highz_points, _, _ = load_highz_kmos3d_proxy()
    nu_mond = estimate_nu_from_mond_sequence(local_fit['best_a0_m_per_s2'], mean_highz_a0(highz_points), z_eff=sum(p['z'] for p in highz_points) / len(highz_points))
    nu_q = estimate_time_crystal_q_proxy()

    best_sn_case = max(sn_cases, key=lambda c: c['delta_chi2_nu0'])
    triangle = {
        'nu_sn_bestfit': float(best_sn_case['best_fit']['nu']),
        'nu_mond_sequence': float(nu_mond['nu_mond_sequence']),
        'nu_time_crystal_q_proxy': float(nu_q['nu_from_time_crystal_q_proxy']),
    }
    triangle['max_pairwise_ratio'] = float(max(triangle.values()) / max(min(v for v in triangle.values() if v > 0), 1e-12))
    triangle['sn_lowz_systematic_reinforced'] = bool(best_sn_case['name'] != 'pantheon_highz' and best_sn_case['best_fit']['nu'] >= 0.018)

    payload = {
        'test_name': 'Public nu re-drive v7.3',
        'cases': sn_cases,
        'cl1_cl2_cl3_triangle': triangle,
        'time_crystal_q_proxy': nu_q,
        'falsification_logic': {
            'confirm_like': 'A positive ν preference survives the public re-drive and agrees, at least in band, with the MOND-sequence and time-crystal consistency legs.',
            'falsify_like': 'The preference remains weak, low-z concentrated, or strongly inconsistent with the MOND-sequence ν, reinforcing the low-z SN systematic interpretation.',
        },
        'notes': [
            'This remains a compact public-data classification tool. It does not claim a collaboration-grade Pantheon+/DESI/Planck inference.',
            'New in v7.3: the output triangulates ν_SN, ν_MOND, and a time-crystal Q consistency leg. The Q leg is explicitly labeled as proxy-level rather than standalone observational evidence.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
