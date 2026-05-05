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
  "group": "cosmology",
  "interpretation": "Uses public CobayaSampler bao_data DESI DR2 BAO files instead of DESI file-server URLs that returned 401.",
  "kind": "bao_grid_positive_v8",
  "prediction_id": "P39",
  "prediction_name": "P39 DESI DR2 BAO grid positive-compatible",
  "sources": [
    {
      "url": "https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2"
    }
  ],
  "test_id": "R10-T03",
  "tier": "public-current",
  "urls": [
    "https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
