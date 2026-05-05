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
  "group": "cmb",
  "kind": "p40_planck_cross_bound_v9",
  "prediction_id": "P40",
  "prediction_name": "P40 Planck/BK18 cross-bound positive-ready",
  "sources": [
    {
      "url": "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/"
    }
  ],
  "test_id": "R10-T22",
  "tier": "public-current",
  "urls": [
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/",
    "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CMB_IQU-smica_2048_R3.00_full.fits"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
