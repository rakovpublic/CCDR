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
  "prediction_id": "P9b/P9e/P9f",
  "prediction_name": "P9 MET/DY/HH units-column bound-positive",
  "sources": [
    {
      "url": "https://www.hepdata.net/record/102093"
    },
    {
      "url": "https://www.hepdata.net/record/129940"
    },
    {
      "url": "https://www.hepdata.net/record/166053"
    }
  ],
  "test_id": "R10-T33",
  "tier": "public-current",
  "urls": [
    "https://www.hepdata.net/record/102093?version=3",
    "https://www.hepdata.net/record/126746",
    "https://www.hepdata.net/record/129940",
    "https://www.hepdata.net/record/ins1711625",
    "https://www.hepdata.net/record/166053",
    "https://www.hepdata.net/record/166074"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
