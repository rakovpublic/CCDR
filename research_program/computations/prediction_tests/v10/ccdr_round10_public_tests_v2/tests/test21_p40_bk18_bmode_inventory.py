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
  "group": "cmb",
  "kind": "bk18_bandpower_bound_v10",
  "prediction_id": "P40",
  "prediction_name": "P40 BK18 bandpower-bound positive-ready",
  "sources": [
    {
      "url": "http://bicepkeck.org/BK18_datarelease/BK18_cosmomc.tgz"
    },
    {
      "url": "https://bicepkeck.org/BK18_datarelease/BK18_cosmomc.tgz"
    }
  ],
  "test_id": "R10-T21",
  "tier": "public-current",
  "urls": [
    "http://bicepkeck.org/BK18_datarelease/BK18_cosmomc.tgz",
    "https://bicepkeck.org/BK18_datarelease/BK18_cosmomc.tgz"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
