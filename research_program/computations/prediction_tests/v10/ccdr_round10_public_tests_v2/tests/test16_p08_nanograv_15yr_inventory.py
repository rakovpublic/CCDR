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
  "group": "pta",
  "kind": "nanograv_kappa_sky_ready_v8",
  "prediction_id": "P8/P8c",
  "prediction_name": "P8/P8c NANOGrav \u03ba-sky positive-ready",
  "sources": [
    {
      "label": "Zenodo record 16051178",
      "url": "https://zenodo.org/records/16051178"
    }
  ],
  "test_id": "R10-T16",
  "tier": "public-current",
  "zenodo_record": "16051178"
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
