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
  "group": "voids",
  "interpretation": "Correct VAST VoidFinder/V2 Zenodo record.",
  "kind": "p38_catalogue_nulls_v22",
  "prediction_id": "P38",
  "prediction_name": "P38 void morphology robust-confirm null hardening",
  "sources": [
    {
      "label": "VAST void catalogs for SDSS DR7",
      "url": "https://zenodo.org/records/7406035"
    }
  ],
  "test_id": "R10-T10",
  "tier": "public-current",
  "zenodo_record": "7406035"
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
