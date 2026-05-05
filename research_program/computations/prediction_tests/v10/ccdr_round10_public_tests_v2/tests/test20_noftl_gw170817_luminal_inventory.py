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
  "group": "gw",
  "kind": "gw170817_inventory",
  "prediction_id": "No-FTL",
  "prediction_name": "GW170817 luminal multimessenger inventory",
  "sources": [
    {
      "url": "https://www.gwosc.org/eventapi/json/GWTC-1-confident/"
    }
  ],
  "test_id": "R10-T20",
  "tier": "public-current",
  "urls": [
    "https://www.gwosc.org/eventapi/json/GWTC-1-confident/",
    "https://gwosc.org/eventapi/json/GWTC-1-confident/",
    "https://www.gwosc.org/eventapi/json/GWTC-1-confident/GW170817-v2/"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
