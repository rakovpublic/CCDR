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
  "group": "lensing",
  "interpretation": "Correct ACT AdvACT DR6 lensing maps endpoints. Large FITS products still require --allow-large and healpy for science sampling.",
  "kind": "p30_act_confirm_squeeze_v14",
  "prediction_id": "P30",
  "prediction_name": "P30 ACT-Euclid/SDSS density-kappa confirmation squeeze",
  "sources": [
    {
      "url": "https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_info.html"
    },
    {
      "url": "https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html"
    }
  ],
  "test_id": "R10-T04",
  "tier": "public-current",
  "urls": [
    "https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_info.html",
    "https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_get.html"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
