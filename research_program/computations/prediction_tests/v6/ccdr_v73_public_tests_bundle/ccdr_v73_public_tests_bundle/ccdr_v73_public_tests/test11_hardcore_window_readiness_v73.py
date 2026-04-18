from __future__ import annotations

from _common_public_data_v73 import curated_hepdata_tables, expected_n_peaks_table, save_json


def main() -> None:
    tables = curated_hepdata_tables()
    overlaps = [row for row in tables if row['overlaps_target_window']]
    payload = {
        'test_name': 'Hard-core window readiness v7.3',
        'target_second_peak_window_gev': [500.0, 3000.0],
        'n_candidate_resources': int(len(tables)),
        'n_numeric_tables_loaded': int(len(tables)),
        'tables_overlapping_target_window': overlaps,
        'operational_window_overlap_now': bool(len(overlaps) > 0),
        'peak_count_ready_now': False,
        'expected_n_peaks_examples': expected_n_peaks_table(),
        'heavy_peak_depletion_statement': 'Under the v7.3 probability-weighted prior, missing heavy peaks are expected rather than automatically falsifying; the informative quantity is the skew of the observed peak distribution toward lighter masses.',
        'falsification_logic': {
            'confirm_like': 'Public products overlap the target window and future event-level or peak-resolution analyses can be interpreted under the probability-weighted E[n_peaks] expectation rather than the old all-peaks-present assumption.',
            'falsify_like': 'Once genuinely peak-resolving public likelihoods overlap the target window, a null result below the probability-weighted lower bound or non-geometric ratios would damage the hard core.',
        },
        'notes': [
            'This reconciles the old “window overlap” audit with the v7.3 distinction between mere coverage and actual peak-resolution readiness.',
            'Machine-readable limit curves touching 0.5–3 TeV do not by themselves mean the hard core has fired.',
        ],
    }
    save_json(payload)


if __name__ == '__main__':
    main()
