#!/usr/bin/env python3
"""
Test 03: Oscillation-only w(z) residual search.

This is a public-data implementation of the proposed detrending check. Instead
of fitting a spline drift plus harmonics, it builds residuals around a smooth
baseline cosmology and then fits an oscillation-only basis:

    residual(z) = sum_k [a_k cos(w_k z) + b_k sin(w_k z)]

The target is falsifiability: do significant peaks survive when no monotonic
trend term is allowed?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize, signal

from _common_public_data import (
    distance_modulus,
    fit_nu_model,
    load_pantheon_plus,
    save_json,
)


def design_matrix(z: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    cols = [np.ones_like(z)]
    for f in freqs:
        cols.append(np.cos(f * z))
        cols.append(np.sin(f * z))
    return np.column_stack(cols)


def weighted_linear_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    chi2 = float(np.sum(w * resid**2))
    return beta, chi2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test03_oscillation_only_wz"))
    parser.add_argument("--max-harmonics", type=int, default=6)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    best = fit_nu_model(include_sn=True, include_bao=True, include_planck=True, analytic_intercept=False)
    sn_df, sn_cov = load_pantheon_plus(use_calibrators=False)
    z = sn_df["zHD"].to_numpy(float)
    mu_obs = sn_df["MU_SH0ES"].to_numpy(float)
    mu_th = distance_modulus(z, best.h0, best.omega_m, best.nu, include_radiation=True) + best.intercept
    resid = mu_obs - mu_th
    sigma = np.sqrt(np.diag(sn_cov))
    weights = 1.0 / np.maximum(sigma, 1e-6) ** 2

    # Periodogram of whitened residuals.
    freq_grid = np.linspace(0.3, 18.0, 1500)
    angular = 2.0 * np.pi * freq_grid
    ls = signal.lombscargle(z, resid / sigma, angular, normalize=True)
    peak_idx, props = signal.find_peaks(ls, prominence=np.percentile(ls, 90))
    peak_order = np.argsort(ls[peak_idx])[::-1]
    selected = peak_idx[peak_order[: args.max_harmonics]]
    selected_freqs = np.sort(freq_grid[selected])

    X0 = np.ones((len(z), 1))
    _, chi2_null = weighted_linear_fit(X0, resid, weights)
    X1 = design_matrix(z, 2.0 * np.pi * selected_freqs)
    coeffs, chi2_osc = weighted_linear_fit(X1, resid, weights)
    delta_chi2 = chi2_null - chi2_osc

    ratios = []
    for i in range(1, len(selected_freqs)):
        ratios.append(float(selected_freqs[i] / selected_freqs[i - 1]))

    summary = {
        "test_name": "Oscillation-only w(z) check",
        "baseline_fit": {
            "h0": best.h0,
            "omega_m": best.omega_m,
            "nu": best.nu,
            "intercept": best.intercept,
        },
        "n_supernovae": int(len(z)),
        "selected_peak_frequencies_cycles_per_redshift": [float(x) for x in selected_freqs],
        "adjacent_frequency_ratios": ratios,
        "chi2_null": float(chi2_null),
        "chi2_oscillation_only": float(chi2_osc),
        "delta_chi2": float(delta_chi2),
        "n_added_parameters": int(X1.shape[1] - X0.shape[1]),
        "falsification_logic": {
            "confirm_like": "A stable subset of peaks survives in an oscillation-only basis with meaningful delta-chi2 improvement.",
            "falsify_like": "The apparent peaks disappear once no drift term is allowed.",
        },
        "notes": [
            "This script operationalizes the detrending challenge using Pantheon+ residuals around a compact public-data baseline fit.",
            "It is a falsifiability tool, not a full nonparametric w(z) reconstruction.",
        ],
    }
    save_json(args.outdir / "test03_oscillation_only_wz_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
