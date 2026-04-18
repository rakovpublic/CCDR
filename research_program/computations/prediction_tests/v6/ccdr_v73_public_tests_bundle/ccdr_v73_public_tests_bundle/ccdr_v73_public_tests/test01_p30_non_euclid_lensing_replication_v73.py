from __future__ import annotations

import argparse

from _common_public_data_v73 import (
    ACT_LINKS,
    PLANCK_LINKS,
    enrich_density_catalog,
    fetch_sdss_galaxy_sample,
    reducing_volume_null,
    save_json,
    summarize_lensing_channel,
)


def main() -> None:
    ap = argparse.ArgumentParser(description='P30 replication with non-Euclid public lensing access paths, updated for v7.3.')
    ap.add_argument('--max-rows', type=int, default=12000)
    ap.add_argument('--null-draws', type=int, default=64)
    args = ap.parse_args()

    gal, source_used = fetch_sdss_galaxy_sample(max_rows=args.max_rows)
    gal = enrich_density_catalog(gal)

    act_result = summarize_lensing_channel(gal, 'kappa_act_proxy', 'act_proxy', n_null_draws=args.null_draws)
    planck_result = summarize_lensing_channel(gal, 'kappa_planck_proxy', 'planck_proxy', n_null_draws=args.null_draws)

    payload = {
        'test_name': 'P30 non-Euclid lensing replication v7.3',
        'source_used': source_used,
        'n_objects': int(len(gal)),
        'act_links_discovered': ACT_LINKS,
        'n_act_links_discovered': len(ACT_LINKS),
        'planck_links_discovered': PLANCK_LINKS,
        'n_planck_links_discovered': len(PLANCK_LINKS),
        'act_result': act_result,
        'planck_result': planck_result,
        'reducing_volume_null_act': reducing_volume_null(gal, 'kappa_act_proxy', n_draws=args.null_draws),
        'reducing_volume_null_planck': reducing_volume_null(gal, 'kappa_planck_proxy', n_draws=args.null_draws),
        'falsification_logic': {
            'confirm_like': 'The density-correlated signal reappears with the predicted sign under independent public lensing access paths, survives shuffled nulls, and shows cross-patch structure compatible with the v7.3 reducing-volume prior.',
            'falsify_like': 'The sign flips or vanishes under independent lensing access paths, or the effect collapses under density-stratified and reducing-volume null controls.',
        },
        'notes': [
            'This remains a public-data replication proxy: if collaboration-grade machine-readable lensing maps are unavailable, deterministic sky-proxy samplers are used rather than claiming a full map-level replication.',
            'New in v7.3: reducing-volume nulls preserve the density ranking while shuffling patch-level DM-composition offsets to test cross-patch variation under the dimension-origin prior.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
