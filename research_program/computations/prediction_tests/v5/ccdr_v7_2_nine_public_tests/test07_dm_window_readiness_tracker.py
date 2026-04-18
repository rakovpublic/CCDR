#!/usr/bin/env python3
"""Test 07: hard-core readiness tracker for the 0.5--3 TeV second-peak window."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from _common_public_data import inspect_direct_detection_resources_latest, save_json

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('out_test07_dm_window_readiness_tracker'))
    ap.add_argument('--window-min-gev', type=float, default=500.0)
    ap.add_argument('--window-max-gev', type=float, default=3000.0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    info = inspect_direct_detection_resources_latest()
    tables = []
    overlap = []
    for row in info['numeric_tables']:
        o = max(0.0, min(args.window_max_gev, row['mass_max']) - max(args.window_min_gev, row['mass_min']))
        rr = dict(row)
        rr['window_overlap_gev'] = float(o)
        rr['overlaps_target_window'] = o > 0
        tables.append(rr)
        if o > 0:
            overlap.append(rr)
    summary = {
        'test_name': 'DM window readiness tracker',
        'target_second_peak_window_gev': [float(args.window_min_gev), float(args.window_max_gev)],
        'n_candidate_resources': info['n_candidate_resources'],
        'n_numeric_tables_loaded': len(tables),
        'tables_sample': tables[:40],
        'tables_overlapping_target_window': overlap,
        'hard_core_ready_now': bool(overlap),
        'falsification_logic': {
            'confirm_like': 'Public products overlap the target window and reveal countable static or time-varying candidate species.',
            'falsify_like': 'Once public sensitivity truly overlaps the target window, no species or peak structure appears.',
        },
        'notes': [
            'This operationalises the v7.2/v3.2 observable-first hard-core readiness question.',
            'No overlap means the hard core remains structurally unavailable in public data, not that it is falsified.',
        ],
    }
    save_json(args.outdir / 'test07_dm_window_readiness_tracker_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
