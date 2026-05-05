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
  "interpretation": "Euclid Q1 access pages reachable; actual catalogue downloads can require TAP/ESA authentication/session handling.",
  "kind": "euclid_mer_catalogue_only_v13",
  "prediction_id": "P30",
  "prediction_name": "P30 Euclid mer_catalogue object-coordinate sample",
  "sources": [
    {
      "url": "https://eas.unige.ch/EAS/Q1/"
    },
    {
      "url": "https://www.cosmos.esa.int/web/euclid/euclid-q1-data-release"
    }
  ],
  "test_id": "R10-T06",
  "tier": "public-current",
  "urls": [
    "https://eas.unige.ch/EAS/Q1/",
    "https://www.cosmos.esa.int/web/euclid/euclid-q1-data-release"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
