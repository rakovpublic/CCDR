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
  "kind": "firas_standard_mu_y_bounds",
  "prediction_id": "P28",
  "prediction_name": "FIRAS standard \u03bc/y 95%-bound proxy",
  "sources": [
    {
      "url": "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt"
    },
    {
      "url": "https://lambda.gsfc.nasa.gov/product/cobe/firas_monopole_get.html"
    }
  ],
  "test_id": "R10-T23",
  "tier": "public-current",
  "urls": [
    "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt",
    "https://lambda.gsfc.nasa.gov/product/cobe/firas_monopole_get.html"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
