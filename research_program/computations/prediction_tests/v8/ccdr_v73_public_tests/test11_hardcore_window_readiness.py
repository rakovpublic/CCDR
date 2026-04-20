#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from _common_public_data import (
    add_source,
    build_argparser,
    curve_window_coverage,
    finalize_result,
    json_result_template,
    load_direct_detection_curves,
    pairwise_ratio_summary,
)


def expected_n_peaks_table() -> dict:
    return {
        "0.3": {"N6": 1.30, "N8": 1.42, "N11": 1.43},
        "0.5": {"N6": 1.50, "N8": 1.88, "N11": 1.98},
        "0.7": {"N6": 1.70, "N8": 2.53, "N11": 3.06},
    }


def main() -> None:
    parser = build_argparser("T11 — Hard-core window readiness")
    args = parser.parse_args()

    result = json_result_template(
        "T11 — Hard-core window readiness",
        "Download public direct-detection limit curves, measure how much of the 0.5–3 TeV target window they cover, and report whether public products have window overlap even if they do not yet have peak-resolution readiness.",
    )

    curves = load_direct_detection_curves()
    cover = {}
    for name, df in curves.items():
        try:
            cover[name] = curve_window_coverage(df)
        except Exception as exc:  # noqa: BLE001
            cover[name] = {"status": "unparsed", "reason": str(exc)}
    good = [v.get("coverage_gev", 0.0) for v in cover.values() if isinstance(v, dict)]
    overlap_now = sum(1 for x in good if x > 0) >= 1
    peak_ready = sum(1 for x in good if x >= 2000.0) >= 3

    result["curve_coverage"] = cover
    result["readiness"] = {
        "operational_window_overlap_now": bool(overlap_now),
        "peak_count_ready_now": bool(peak_ready),
        "expected_n_peaks_under_prior": expected_n_peaks_table(),
        "screening_interpretation": "Window overlap can be true while peak-resolution readiness remains false.",
    }
    add_source(result, "XENONnT public WIMP data release", "https://github.com/XENONnT/wimp_data_release")
    add_source(result, "PandaX-4T first-analysis public data", "https://pandax.sjtu.edu.cn/public/data_release/PandaX-4T/first_analysis")
    add_source(result, "LZ HEPData record", "https://www.hepdata.net/record/145090")
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
