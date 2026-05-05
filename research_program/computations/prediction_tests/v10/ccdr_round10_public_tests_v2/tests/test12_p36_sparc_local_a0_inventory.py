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
  "group": "galaxy",
  "kind": "sparc_robust_a0",
  "prediction_id": "P36/local a0",
  "prediction_name": "P36 SPARC robust local RAR/a0 bootstrap test",
  "sources": [
    {
      "label": "SPARC Rotmod LTG",
      "url": "https://zenodo.org/records/16284118"
    }
  ],
  "test_id": "R10-T12",
  "tier": "public-current"
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
