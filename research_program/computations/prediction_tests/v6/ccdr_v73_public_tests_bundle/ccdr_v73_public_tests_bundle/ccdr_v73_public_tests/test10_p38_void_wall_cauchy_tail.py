from __future__ import annotations

from _common_public_data_v73 import enrich_density_catalog, fetch_sdss_galaxy_sample, make_void_wall_profile, save_json


def main() -> None:
    gal, source_used = fetch_sdss_galaxy_sample(z_max=0.15, max_rows=14000)
    gal = enrich_density_catalog(gal)
    profile = make_void_wall_profile(gal)
    payload = {
        'test_name': 'P38 void-wall Cauchy-tail proxy',
        'source_used': source_used,
        'profile': profile,
        'passes_k4_gt_4': bool(profile['transverse_kurtosis_k4'] > 4.0),
        'falsification_logic': {
            'confirm_like': 'Stacked void-wall transverse density cuts exhibit kurtosis above the Gaussian baseline, consistent with power-law-like wall tails.',
            'falsify_like': 'The stacked transverse profile is close to Gaussian and does not show the excess-tailed behavior expected from grain-boundary scattering.',
        },
        'notes': [
            'This is a public-proxy implementation using density-field minima as void seeds when full public void catalogues are not available in the runtime environment.',
            'A positive proxy result should still be followed by a dedicated void-catalogue analysis.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
