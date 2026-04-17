#!/usr/bin/env python3
"""Test 08: P31 drifting DM peak audit on public HEPData direct-detection tables.

This keeps only simple numeric tables and focuses on repeated candidate features across related public products.
It is a preregistered audit, not a claim that current tables contain accepted dark-matter mass detections.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
from scipy import signal
from _common_public_data import inspect_direct_detection_resources_latest, load_numeric_table_from_url, save_json

def extract_record_key(url: str) -> str:
    m = re.search(r'ins\d+', url)
    return m.group(0) if m else url

def find_candidate_peaks(df) -> list[float]:
    x = np.asarray(df.iloc[:, 0], float)
    y = np.asarray(df.iloc[:, 1], float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 40:
        return []
    # Restrict to csv-like smooth curves by using only positive x and avoiding file-format artifacts downstream.
    ly = np.log10(np.abs(y) + 1e-30)
    win = 11 if len(ly) >= 11 else max(5, len(ly)//2*2+1)
    smooth = signal.savgol_filter(ly, win, 2)
    resid = ly - smooth
    peaks, _ = signal.find_peaks(resid, height=max(np.std(resid), 0.05), distance=5)
    vals = [float(x[i]) for i in peaks if 0.5 <= x[i] <= 5000]
    return vals[:20]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('out_test08_p31_dm_peak_drift_audit'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    info = inspect_direct_detection_resources_latest()
    grouped = {}
    loaded = 0
    for row in info['numeric_tables']:
        src = row['source']
        if not src.lower().endswith('/csv') and '.csv' not in src.lower():
            continue
        df = load_numeric_table_from_url(src)
        if df is None:
            continue
        loaded += 1
        peaks = find_candidate_peaks(df)
        key = extract_record_key(src)
        grouped.setdefault(key, []).append({'source': src, 'n_points': row['n_points'], 'candidate_peaks_gev': peaks})
    repeated = []
    for key, items in grouped.items():
        all_peaks = sorted([p for item in items for p in item['candidate_peaks_gev']])
        if len(all_peaks) < 2:
            continue
        clusters = []
        cur = [all_peaks[0]]
        for p in all_peaks[1:]:
            if abs(np.log(p / cur[-1])) < 0.08:
                cur.append(p)
            else:
                if len(cur) >= 2:
                    clusters.append(cur)
                cur = [p]
        if len(cur) >= 2:
            clusters.append(cur)
        if clusters:
            repeated.append({'record': key, 'clusters_gev': [[float(v) for v in c] for c in clusters]})
    summary = {
        'test_name': 'P31 DM peak-drift audit',
        'n_candidate_resources': info['n_candidate_resources'],
        'n_numeric_tables_loaded': loaded,
        'grouped_records': grouped,
        'repeated_candidate_clusters': repeated,
        'falsification_logic': {
            'confirm_like': 'A recurring candidate feature is seen across public direct-detection products and can be tracked for monotonic drift over time.',
            'falsify_like': 'A genuinely repeated detected mass feature shows no drift within achievable comparison precision.',
        },
        'notes': [
            'This audit is intentionally conservative and restricts attention to simple numeric CSV-style curves when possible.',
            'Repeated grid artifacts or model-scan masses should not be treated as real DM detections.',
        ],
    }
    save_json(args.outdir / 'test08_p31_dm_peak_drift_audit_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
