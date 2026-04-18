from __future__ import annotations

from _common_public_data_v73 import (
    build_nanograv_cross_field,
    enrich_density_catalog,
    fetch_sdss_galaxy_sample,
    load_nanograv_pulsar_positions,
    pearsonr_safe,
    save_json,
    spearmanr_safe,
)


def main() -> None:
    gal, source_used = fetch_sdss_galaxy_sample(max_rows=12000)
    gal = enrich_density_catalog(gal)
    pulsars, pulsar_source = load_nanograv_pulsar_positions()
    pta_field = build_nanograv_cross_field(gal)
    pearson_r, pearson_p = pearsonr_safe(gal['kappa_act_proxy'].to_numpy(float), pta_field)
    spearman_r, spearman_p = spearmanr_safe(gal['kappa_act_proxy'].to_numpy(float), pta_field)

    payload = {
        'test_name': 'CL2 P8c reducing-volume cross v7.3',
        'source_used': source_used,
        'pta_source': pulsar_source,
        'n_objects': int(len(gal)),
        'n_pulsars': int(len(pulsars)),
        'pulsars': pulsars,
        'cross_correlation': {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mean_pta_field': float(pta_field.mean()),
            'mean_kappa_act_proxy': float(gal['kappa_act_proxy'].mean()),
        },
        'same_sign_as_p30': bool(pearson_r > 0),
        'falsification_logic': {
            'confirm_like': 'The PTA×cosmic-web proxy positively cross-correlates with the density–κ field, supporting the v7.3 reinterpretation of the wrong-sign P8c result as a reducing-volume signature.',
            'falsify_like': 'The CL2 cross-correlation remains absent or flips away from the density–κ field, making the wrong-sign reinterpretation less persuasive.',
        },
        'notes': [
            'This is a spatial cross-field proxy, not a full PTA map-level reconstruction.',
            'The purpose is specifically v7.3 CL2: ask whether the wrong-sign class lines up with the same cross-patch structure that drives the P30 density–κ signal.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
