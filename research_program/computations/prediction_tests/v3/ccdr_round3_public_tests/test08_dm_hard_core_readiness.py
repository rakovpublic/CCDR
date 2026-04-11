#!/usr/bin/env python3
"""
Test 08: direct-detection hard-core readiness map.

Inspects public machine-readable HEPData resources linked to current
high-mass-direct-detection result records and asks a narrower question than the
old peak-scan: do public numeric products actually give mass coverage and table
resolution in the predicted second-peak window, and is the input rich enough to
fire the CCDR hard-core falsification rule?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common_public_data import inspect_direct_detection_resources, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test08_dm_hard_core_readiness"))
    parser.add_argument("--window-min-gev", type=float, default=500.0)
    parser.add_argument("--window-max-gev", type=float, default=3000.0)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    info = inspect_direct_detection_resources()
    tables = info["numeric_tables"]
    overlapping = []
    for row in tables:
        overlap = max(0.0, min(args.window_max_gev, row["mass_max"]) - max(args.window_min_gev, row["mass_min"]))
        covered = overlap > 0
        row = dict(row)
        row["overlaps_target_window"] = covered
        row["window_overlap_gev"] = float(overlap)
        if covered:
            overlapping.append(row)

    summary = {
        "test_name": "DM hard-core readiness map",
        "target_second_peak_window_gev": [float(args.window_min_gev), float(args.window_max_gev)],
        "n_candidate_resources": info["n_candidate_resources"],
        "n_numeric_tables_loaded": len(tables),
        "tables": tables,
        "tables_overlapping_target_window": overlapping,
        "hard_core_ready_now": False,
        "falsification_logic": {
            "confirm_like": "Future machine-readable public products reach the target mass window with enough structure to search for a genuine second peak.",
            "falsify_like": "Once such products exist and no second peak appears, the CCDR hard-core tower prediction fails.",
        },
        "notes": [
            "This is a readiness and coverage audit, not a hard-core falsification by itself.",
            "Machine-readable limit curves are not the same thing as public event-level likelihoods or peak-resolution analyses.",
        ],
    }
    save_json(args.outdir / "test08_dm_hard_core_readiness_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
