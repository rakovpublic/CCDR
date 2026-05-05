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
  "kind": "direct_detection_columns_confirm_v14",
  "prediction_id": "P10/P25/P31",
  "prediction_name": "XENONnT mass-window measured coverage confirmation-ready",
  "sources": [
    {
      "url": "https://www.xenonexperiment.org/results"
    },
    {
      "url": "https://arxiv.org/abs/2502.18005"
    }
  ],
  "test_id": "R10-T25",
  "tier": "public-current",
  "urls": [
    "https://www.xenonexperiment.org/results",
    "https://arxiv.org/abs/2502.18005"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
