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
  "kind": "kross_vrot_parser_v14",
  "prediction_id": "P36",
  "prediction_name": "P36 high-z a0 cross-catalogue Vrot^2/R squeeze",
  "sources": [
    {
      "url": "https://www.mpe.mpg.de/ir/KMOS3D/data"
    },
    {
      "url": "https://www.nottingham.ac.uk/astronomy/kross/data.aspx"
    }
  ],
  "test_id": "R10-T14",
  "tier": "public-current",
  "urls": [
    "https://www.mpe.mpg.de/ir/KMOS3D/data",
    "https://www.nottingham.ac.uk/astronomy/kross/data.aspx"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
