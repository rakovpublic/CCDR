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
  "group": "collider",
  "kind": "hepdata_units_columns_v10",
  "prediction_id": "P5",
  "prediction_name": "P5 QGP/KSS units-column bound-positive",
  "sources": [
    {
      "url": "https://www.hepdata.net/record/133408"
    },
    {
      "url": "https://www.hepdata.net/record/ins1419244"
    }
  ],
  "test_id": "R10-T30",
  "tier": "public-current",
  "urls": [
    "https://www.hepdata.net/record/133408?format=json",
    "https://www.hepdata.net/record/ins2093750?format=json",
    "https://www.hepdata.net/record/ins1419244?format=json",
    "https://doi.org/10.17182/hepdata.72886.v2/t7"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
