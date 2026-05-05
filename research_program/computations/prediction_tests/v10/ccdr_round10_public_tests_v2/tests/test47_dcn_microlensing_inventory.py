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
  "group": "dcn",
  "kind": "dcn_digitized_window_v10",
  "prediction_id": "DCN_k",
  "prediction_name": "DCN/AQN microlensing digitized-window positive-ready",
  "sources": [
    {
      "url": "https://arxiv.org/abs/2402.00212"
    },
    {
      "url": "https://link.aps.org/doi/10.1103/PhysRevD.99.083503"
    }
  ],
  "test_id": "R10-DCN01",
  "tier": "public-proxy",
  "urls": [
    "https://arxiv.org/abs/2402.00212",
    "https://arxiv.org/abs/2507.00770",
    "https://link.aps.org/doi/10.1103/PhysRevD.99.083503",
    "https://inspirehep.net/literature/468330"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
