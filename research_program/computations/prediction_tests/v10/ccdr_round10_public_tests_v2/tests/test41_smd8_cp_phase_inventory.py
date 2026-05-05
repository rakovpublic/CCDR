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
  "group": "smd",
  "kind": "smd_constants_pack",
  "needles": [
    "Jarlskog",
    "delta",
    "CKM"
  ],
  "prediction_id": "SM-D8",
  "prediction_name": "SM-D8 CKM CP/Jarlskog inventory check",
  "sources": [
    {
      "url": "https://pdg.lbl.gov/2025/reviews/rpp2025-rev-cp-violation.pdf"
    },
    {
      "url": "https://pdg.lbl.gov/2024/reviews/rpp2024-rev-cp-violation.pdf"
    }
  ],
  "test_id": "R10-SMD08",
  "tier": "public-current",
  "urls": []
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
