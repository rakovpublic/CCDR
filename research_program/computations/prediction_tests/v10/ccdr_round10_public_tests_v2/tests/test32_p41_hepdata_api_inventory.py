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
  "group": "flavour",
  "kind": "p41_structured_cp_v22",
  "prediction_id": "P41",
  "prediction_name": "P41 control supplementary archive structured-table parser",
  "sources": [
    {
      "url": "https://cds.cern.ch/record/2951844/export/xm"
    },
    {
      "url": "https://arxiv.org/abs/2512.18053"
    }
  ],
  "test_id": "R10-T32",
  "tier": "public-current",
  "urls": [
    "https://www.hepdata.net/search/?q=LHCb%20B0%20K%2A0%20mu%2B%20mu-%20angular%20analysis&format=json",
    "https://www.hepdata.net/search/?q=b%20to%20s%20mu%20mu%20C9&format=json"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
