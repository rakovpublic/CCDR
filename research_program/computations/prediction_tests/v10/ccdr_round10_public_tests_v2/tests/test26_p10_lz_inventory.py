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
  "prediction_name": "LZ explicit-unit coverage parser",
  "sources": [
    {
      "url": "https://www.hepdata.net/record/155182?version=1"
    }
  ],
  "test_id": "R10-T26",
  "tier": "public-current",
  "urls": [
    "https://www.hepdata.net/record/155182?version=1",
    "https://www.hepdata.net/record/155182?format=json",
    "https://www.hepdata.net/record/158595"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
