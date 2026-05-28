#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccdr_r10_common import safe_json_main, run_by_kind

META = {
  "falsification_logic": {
    "confirm_like": "Public data show the predicted sign/scale/stability after null controls.",
    "data_limited": "If public data are insufficient or event-level products are unavailable, return data_limited/readiness_only rather than claiming a result.",
    "falsify_like": "The predicted effect is absent, reversed, or entirely explained by public-data null controls at adequate sensitivity."
  },
  "group": "direct_detection",
  "kind": "direct_detection_units_v22",
  "prediction_id": "P10/P25/P31",
  "prediction_name": "PandaX quantified mass-window positive-ready",
  "sources": [
    {
      "url": "https://pandax.sjtu.edu.cn/public/data_release"
    }
  ],
  "test_id": "R10-T27",
  "tier": "public-current",
  "urls": [
    "https://pandax.sjtu.edu.cn/public/data_release",
    "https://static.pandax.sjtu.edu.cn/download/data-share/p4-light-dark-matter/run0_data.csv",
    "https://static.pandax.sjtu.edu.cn/download/data-share/p4-light-dark-matter/run1_data.csv"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
