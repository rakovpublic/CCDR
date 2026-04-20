#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from _common_public_data import (
    add_source,
    build_argparser,
    desi_bao_paths,
    extract_desi_fs8_points,
    finalize_result,
    fit_live_fraction,
    json_result_template,
    load_public_growth_points,
    public_growth_paths,
)


def main() -> None:
    parser = build_argparser("T6 — P29 multi-redshift growth")
    args = parser.parse_args()

    result = json_result_template(
        "T6 — P29 multi-redshift growth",
        "Extract public multi-redshift fσ8 points from public BAO+FS tables and fit a simple live/frozen dark-sector mixture to screen for wrong-sign or positive α behavior.",
    )
    growth = load_public_growth_points()
    fs8 = extract_desi_fs8_points(growth)
    fit = fit_live_fraction(fs8["z"], fs8["fs8"])
    delta_chi2_against_zero = fit.chi2 - fit_live_fraction(fs8["z"], np.full_like(fs8["fs8"], np.nanmedian(fs8["fs8"]))).chi2

    result["fs8_points"] = fs8.to_dict(orient="records")
    result["fit"] = {
        **fit.params,
        "chi2": fit.chi2,
        "ndof": fit.ndof,
        "delta_chi2_against_zeroish": float(delta_chi2_against_zero),
        "wrong_sign_standalone": bool(fit.params["alpha"] < 0),
    }
    add_source(result, "SDSS DR16 LRG BAO+FS file", public_growth_paths()["sdss_dr16_lrg"])
    add_source(result, "SDSS DR16 QSO BAO+FS file", public_growth_paths()["sdss_dr16_qso"])
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
