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
  "kind": "planck_y_firas_cross_bound",
  "prediction_id": "P28",
  "prediction_name": "P28 Planck y-map plus FIRAS bound cross-check",
  "sources": [
    {
      "url": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/ysz_index.html"
    }
  ],
  "test_id": "R10-T24",
  "tier": "public-current",
  "urls": [
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/ysz_index.html",
    "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/ysz_index.html",
    "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits",
    "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_Compton-SZMap-milca-ymaps_2048_R2.00.fits"
  ]
}

if __name__ == "__main__":
    safe_json_main(META, run_by_kind)
