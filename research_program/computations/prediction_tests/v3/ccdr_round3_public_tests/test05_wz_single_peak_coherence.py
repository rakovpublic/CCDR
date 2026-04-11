#!/usr/bin/env python3
"""
Test 05: split-sample coherence of the surviving oscillation-only w(z) peak.

Builds Pantheon+ residuals around the compact public-data baseline fit and asks
whether the leading oscillation-only frequency is coherent across redshift and
survey splits. No drift term is allowed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

from _common_public_data import distance_modulus, fit_nu_model, load_pantheon_plus, save_json


def lomb_best_frequency(z: np.ndarray, resid: np.ndarray, sigma: np.ndarray, fmin: float = 0.3, fmax: float = 18.0, nfreq: int = 2000) -> tuple[float, float]:
    freq_grid = np.linspace(fmin, fmax, nfreq)
    ang = 2.0 * np.pi * freq_grid
    y = resid / np.maximum(sigma, 1e-6)
    power = signal.lombscargle(z, y, ang, normalize=True)
    idx = int(np.argmax(power))
    return float(freq_grid[idx]), float(power[idx])


def amplitude_at_frequency(z: np.ndarray, resid: np.ndarray, sigma: np.ndarray, freq: float) -> dict:
    w = 2.0 * np.pi * freq
    X = np.column_stack([np.ones_like(z), np.cos(w * z), np.sin(w * z)])
    W = np.diag(1.0 / np.maximum(sigma, 1e-6) ** 2)
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ resid)
    cov = np.linalg.inv(XtW @ X)
    amp = float(np.hypot(beta[1], beta[2]))
    amp_err = float(np.sqrt(max(cov[1, 1] + cov[2, 2], 1e-16)))
    return {"amplitude": amp, "amplitude_over_error": amp / amp_err if amp_err > 0 else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test05_wz_single_peak_coherence"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    best = fit_nu_model(include_sn=True, include_bao=True, include_planck=True, analytic_intercept=False)
    sn_df, sn_cov = load_pantheon_plus(use_calibrators=False)
    z = sn_df["z_cmb"].to_numpy(float)
    mu_obs = sn_df["mu"].to_numpy(float)
    mu_th = distance_modulus(z, best.h0, best.omega_m, best.nu, include_radiation=True) + best.intercept
    resid = mu_obs - mu_th
    sigma = np.sqrt(np.clip(np.diag(sn_cov), 1e-12, None))

    freq_full, power_full = lomb_best_frequency(z, resid, sigma)
    survey_counts = sn_df["survey_id"].value_counts()
    top_survey = str(survey_counts.index[0])
    median_z = float(np.median(z))

    masks = {
        "all": np.ones(len(z), dtype=bool),
        "z_low": z < median_z,
        "z_high": z >= median_z,
        f"survey_{top_survey}": sn_df["survey_id"].astype(str).to_numpy() == top_survey,
        f"without_survey_{top_survey}": sn_df["survey_id"].astype(str).to_numpy() != top_survey,
    }

    subsets = []
    for name, mask in masks.items():
        if np.sum(mask) < 80:
            continue
        f_best, p_best = lomb_best_frequency(z[mask], resid[mask], sigma[mask])
        amp = amplitude_at_frequency(z[mask], resid[mask], sigma[mask], freq_full)
        subsets.append({
            "name": name,
            "n": int(np.sum(mask)),
            "best_frequency_cycles_per_redshift": f_best,
            "best_power": p_best,
            "global_frequency_amplitude": amp["amplitude"],
            "global_frequency_amp_over_err": amp["amplitude_over_error"],
        })

    summary = {
        "test_name": "w(z) single-peak coherence",
        "baseline_fit": {"h0": best.h0, "omega_m": best.omega_m, "nu": best.nu, "intercept": best.intercept},
        "full_sample_best_frequency_cycles_per_redshift": freq_full,
        "full_sample_best_power": power_full,
        "subsets": subsets,
        "falsification_logic": {
            "confirm_like": "The surviving oscillation-only peak recurs at a similar frequency across disjoint subsets and is not concentrated in one survey block.",
            "falsify_like": "The single remaining peak wanders or disappears across redshift and survey splits.",
        },
        "notes": [
            "This script tests coherence of the surviving single oscillation-only peak; it does not attempt to resurrect the retracted six-peak claim.",
        ],
    }
    save_json(args.outdir / "test05_wz_single_peak_coherence_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
