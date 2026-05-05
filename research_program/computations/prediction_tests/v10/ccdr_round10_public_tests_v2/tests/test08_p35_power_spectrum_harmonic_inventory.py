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
  "group": "bao",
  "kind": "harmonic_comb_proxy",
  "prediction_id": "P35",
  "prediction_name": "P35 BAO harmonic-comb proxy readiness-positive",
  "sources": [
    {
      "url": "https://data.desi.lbl.gov/public/dr2/README"
    },
    {
      "url": "https://data.sdss.org/sas/dr17/eboss/lss/catalogs/DR16/"
    }
  ],
  "test_id": "R10-T08",
  "tier": "public-current",
  "urls": [
    "https://data.desi.lbl.gov/public/dr2/README",
    "https://data.sdss.org/sas/dr17/eboss/lss/catalogs/DR16/"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
