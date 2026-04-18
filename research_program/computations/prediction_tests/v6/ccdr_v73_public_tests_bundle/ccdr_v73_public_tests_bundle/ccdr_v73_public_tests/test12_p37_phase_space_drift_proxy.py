from __future__ import annotations

from _common_public_data_v73 import (
    estimate_nu_from_mond_sequence,
    fit_local_a0_from_rar,
    gaia_rotation_proxy,
    load_highz_kmos3d_proxy,
    load_sparc_anchor_sample,
    mean_highz_a0,
    phase_space_drift_proxy,
    save_json,
)


def main() -> None:
    sparc, _ = load_sparc_anchor_sample()
    local_fit = fit_local_a0_from_rar(sparc)
    highz_points, _, _ = load_highz_kmos3d_proxy()
    nu = estimate_nu_from_mond_sequence(local_fit['best_a0_m_per_s2'], mean_highz_a0(highz_points))['nu_mond_sequence']
    gaia = gaia_rotation_proxy()
    drift = phase_space_drift_proxy(nu, n_events=3000)
    payload = {
        'test_name': 'P37 phase-space drift proxy',
        'gaia_rotation_proxy': gaia.to_dict(orient='records'),
        'drift_proxy': drift,
        'falsification_logic': {
            'confirm_like': 'Opposite galactic-orbit phases exhibit a coherent spectral offset of the sign expected from live-cascade DM mass drift, even if the effect is subthreshold per event.',
            'falsify_like': 'The orbit-phase proxy shows no coherent offset or the sign is opposite to the live-cascade expectation.',
        },
        'notes': [
            'This upgrades the old standalone drift audit into the v7.3 orbit-phase proxy logic emphasized by P37.',
            'The detector population is simulated because public event-level dark-matter detector datasets are not currently available in the right form.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
