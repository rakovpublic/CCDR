#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg, stats

from _common_public_data import (
    add_source,
    build_argparser,
    comoving_distance_mpc,
    desi_bao_paths,
    finalize_result,
    json_result_template,
    load_desi_consensus,
    load_pantheon_plus,
    pairwise_ratio_summary,
)


def infer_pantheon_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = {c.lower(): c for c in df.columns}
    return {
        "zcmb": cols.get("zcmb", next(c for c in df.columns if "zcmb" in c.lower())),
        "zhel": cols.get("zhel", next(c for c in df.columns if "zhel" in c.lower())),
        "mu": next(c for c in df.columns if c.lower() in {"mu_shoes", "mu", "mb"} or "mu" in c.lower()),
    }


def sigma_like_from_slope(z: np.ndarray, mu: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    m = np.isfinite(z) & np.isfinite(mu) & (z > 0)
    if m.sum() < 10:
        return float("nan")
    x = np.log10(z[m])
    y = mu[m]
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = linalg.lstsq(X, y)
    resid = y - X @ beta
    sigma = np.std(resid, ddof=2)
    slope_sigma = abs(beta[1]) / max(sigma / np.sqrt(len(y)), 1e-12)
    return float(slope_sigma)


def main() -> None:
    parser = build_argparser("T4 — Public ν re-drive")
    args = parser.parse_args()

    result = json_result_template(
        "T4 — Public ν re-drive",
        "Use public Pantheon+ and DESI DR2 products to compare low-z and high-z ν-sensitive residual behavior and quantify whether the preference is carried by low-z SNe.",
    )

    pan, cov = load_pantheon_plus()
    desi = load_desi_consensus()
    cols = infer_pantheon_columns(pan)

    zcmb = pd.to_numeric(pan[cols["zcmb"]], errors="coerce").to_numpy(dtype=float)
    zhel = pd.to_numeric(pan[cols["zhel"]], errors="coerce").to_numpy(dtype=float)
    mu = pd.to_numeric(pan[cols["mu"]], errors="coerce").to_numpy(dtype=float)

    sigmas = {
        "pantheon_zcmb": sigma_like_from_slope(zcmb, mu),
        "pantheon_zhel": sigma_like_from_slope(zhel, mu),
        "pantheon_lowz": sigma_like_from_slope(zcmb[zcmb <= 0.5], mu[zcmb <= 0.5]),
        "pantheon_highz": sigma_like_from_slope(zcmb[zcmb > 0.5], mu[zcmb > 0.5]),
    }
    # A compact ν-like proxy from relative low-vs-high z significance.
    low = sigmas["pantheon_lowz"]
    high = sigmas["pantheon_highz"]
    nu_sn_proxy = float(np.clip(0.001 + 0.01 * max(low - high, 0) / max(low, 1e-12), 0, 0.03))
    nu_q_proxy = 0.001
    nu_mond_reference = 5.08e-3
    ratios = pairwise_ratio_summary([nu_sn_proxy, nu_mond_reference, nu_q_proxy])

    result["pantheon_sigma_like"] = sigmas
    result["desi_consensus_shape"] = {"rows": int(len(desi)), "columns": [str(c) for c in desi.columns]}
    result["nu_triangle_proxy"] = {
        "nu_sn_proxy": nu_sn_proxy,
        "nu_mond_reference": nu_mond_reference,
        "nu_q_proxy": nu_q_proxy,
        **ratios,
        "sn_lowz_systematic_reinforced": bool(low > high),
    }

    add_source(result, "Pantheon+ DataRelease", "https://github.com/PantheonPlusSH0ES/DataRelease")
    add_source(result, "DESI DR2 BAO data", desi_bao_paths()["consensus"])
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
