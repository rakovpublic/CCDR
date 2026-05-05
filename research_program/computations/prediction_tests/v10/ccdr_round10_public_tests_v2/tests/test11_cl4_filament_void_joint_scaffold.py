#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccdr_r10_common import safe_json_main, run_by_kind

META = {
  "bridge_type": "CL4",
  "falsification_logic": {
    "confirm_like": "Public data show the predicted sign/scale/stability after null controls.",
    "data_limited": "If public data are insufficient or event-level products are unavailable, return data_limited/readiness_only rather than claiming a result.",
    "falsify_like": "The predicted effect is absent, reversed, or entirely explained by public-data null controls at adequate sensitivity."
  },
  "group": "cross_link",
  "implementation_notes": [
    "Consumes outputs from filament and void tests; does not download new science data."
  ],
  "kind": "bridge_whitelist_v9",
  "prediction_id": "CL4",
  "prediction_name": "CL4 P3+P38 whitelist-updated bridge-positive",
  "sources": [],
  "test_id": "R10-T11",
  "tier": "public-current"
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
