from __future__ import annotations

from _common_public_data_v73 import fit_growth_live_frozen, load_growth_points, save_json


def main() -> None:
    points = load_growth_points()
    fit = fit_growth_live_frozen(points)
    payload = {
        'test_name': 'P29 multi-redshift growth v7.3',
        'points': points,
        'fit': fit,
        'falsification_logic': {
            'confirm_like': 'A positive live-cascade growth term improves the fit, and the live/frozen decomposition is qualitatively compatible with the probability-weighted prior.',
            'falsify_like': 'A constant-DM or effectively frozen history fits equally well, or the preferred live-growth proxy remains zero or wrong-sign.',
        },
        'notes': [
            'This is a compressed public full-shape/RSD proxy based on summary points rather than a perturbation-level chain analysis.',
            'New in v7.3: the output decomposes the fit into live and frozen fractions rather than forcing every null into a simple yes/no verdict.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
