#!/usr/bin/env python3
"""
P8_nanograv_beat_test_v2.py

Tests for sinusoidal modulation of NANOGrav 15-yr sensitivity curve
in log-frequency space, with PROPER null hypothesis comparison.

Key fix from v1: generates synthetic null curves (smooth power-law +
realistic structure but no beats) and compares the measured amp_log
against the null distribution to compute a real significance.
"""
import os
import sys
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import percentileofscore

DATA_DIR = Path("data/p8_nanograv")
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = ("https://zenodo.org/records/8092346/files/"
       "NANOGrav15yr_Sensitivity-Curves_v1.0.0.tar.gz?download=1")
ARCHIVE = DATA_DIR / "NANOGrav15yr_Sensitivity-Curves_v1.0.0.tar.gz"
EXTRACT = DATA_DIR / "extracted"


def download_data():
    """Download NANOGrav 15-yr sensitivity curves from Zenodo."""
    if ARCHIVE.exists():
        print(f"[cache] {ARCHIVE}")
    else:
        print(f"[download] {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
        print(f"[saved]    {ARCHIVE} ({ARCHIVE.stat().st_size / 1e6:.1f} MB)")

    if not EXTRACT.exists():
        print(f"[extract]  {ARCHIVE} -> {EXTRACT}")
        EXTRACT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ARCHIVE) as tf:
            tf.extractall(EXTRACT)


def find_sensitivity_curve():
    """Locate the full-PTA sensitivity curve file."""
    candidates = list(EXTRACT.rglob("*fullPTA*.txt"))
    if not candidates:
        candidates = list(EXTRACT.rglob("sensitivity_curves*.txt"))
    if not candidates:
        raise FileNotFoundError("No sensitivity curve file found in extracted data")
    return candidates[0]


def load_curve(filepath):
    """Load (frequency, characteristic strain) from the curve file."""
    data = np.loadtxt(filepath, comments="#")
    if data.shape[1] < 2:
        raise ValueError(f"Unexpected format in {filepath}")
    freq = data[:, 0]      # Hz
    h_c = data[:, 1]       # characteristic strain (dimensionless)
    return freq, h_c


def fit_baseline(log_f, log_h):
    """Fit smooth power-law baseline (linear in log-log)."""
    coeffs = np.polyfit(log_f, log_h, deg=1)
    return coeffs  # [slope, intercept]


def model_baseline_plus_beat(log_f, slope, intercept, amp, omega, phase):
    """Power-law + sinusoidal modulation in log-frequency space."""
    return slope * log_f + intercept + amp * np.sin(omega * log_f + phase)


def fit_beat(log_f, log_h, baseline_coeffs):
    """Fit sinusoidal residual after baseline subtraction."""
    slope, intercept = baseline_coeffs
    residual = log_h - (slope * log_f + intercept)

    # Try multiple omega initial values to avoid local minima
    best_params = None
    best_rss = np.inf

    for omega_init in [0.5, 1.0, 2.0, 4.0, 8.0]:
        for amp_init in [0.01, 0.1, 1.0]:
            try:
                p0 = [amp_init, omega_init, 0.0]
                popt, _ = curve_fit(
                    lambda x, a, w, p: a * np.sin(w * x + p),
                    log_f, residual, p0=p0, maxfev=5000)
                pred = popt[0] * np.sin(popt[1] * log_f + popt[2])
                rss = np.sum((residual - pred) ** 2)
                if rss < best_rss:
                    best_rss = rss
                    best_params = popt
            except RuntimeError:
                continue

    if best_params is None:
        return None, None
    return best_params, best_rss


def measure_amp(freq, h_c):
    """Measure the beat amplitude on a given (freq, h_c) curve."""
    log_f = np.log10(freq)
    log_h = np.log10(h_c)
    valid = np.isfinite(log_f) & np.isfinite(log_h)
    log_f, log_h = log_f[valid], log_h[valid]

    baseline = fit_baseline(log_f, log_h)
    beat_params, rss = fit_beat(log_f, log_h, baseline)
    if beat_params is None:
        return None

    amp, omega, phase = beat_params
    return {
        "baseline_slope": float(baseline[0]),
        "baseline_intercept": float(baseline[1]),
        "amp_log": float(abs(amp)),
        "omega": float(omega),
        "phase": float(phase),
        "rss": float(rss),
    }


def generate_null_curves(freq, h_c_real, n_realisations=1000, seed=42):
    """
    Generate synthetic null sensitivity curves with the SAME smooth shape
    as the real one but NO beat-like modulation. The null is a power-law
    plus correlated Gaussian noise matching the real curve's residual scatter.
    """
    rng = np.random.default_rng(seed)
    log_f = np.log10(freq)
    log_h = np.log10(h_c_real)

    # Fit smooth baseline (degree 3 polynomial captures real curve shape
    # without imposing an oscillation)
    poly = np.polyfit(log_f, log_h, deg=3)
    smooth = np.polyval(poly, log_f)
    residual_scale = np.std(log_h - smooth)

    null_curves = []
    for _ in range(n_realisations):
        # Add correlated Gaussian noise (length scale ~10 points)
        noise = rng.normal(0, residual_scale, size=len(log_f))
        # Smooth the noise slightly to mimic finite spectral resolution
        kernel = np.exp(-np.arange(-5, 6) ** 2 / 4)
        kernel /= kernel.sum()
        smoothed_noise = np.convolve(noise, kernel, mode="same")
        log_h_null = smooth + smoothed_noise
        h_c_null = 10 ** log_h_null
        null_curves.append(h_c_null)

    return null_curves


def main():
    print("=" * 70)
    print("P8: NANOGrav 15-yr Beat-Frequency Test (with NULL hypothesis)")
    print("=" * 70)

    download_data()
    curve_file = find_sensitivity_curve()
    print(f"[load] {curve_file}")
    freq, h_c = load_curve(curve_file)
    print(f"[load] {len(freq)} points from {freq[0]:.2e} to {freq[-1]:.2e} Hz")

    # Measure amplitude on real data
    print("\n[fit] real curve...")
    real = measure_amp(freq, h_c)
    if real is None:
        print("FATAL: could not fit real curve")
        sys.exit(1)
    print(f"  amp_log = {real['amp_log']:.4f}")
    print(f"  omega   = {real['omega']:.4f}")
    print(f"  rss     = {real['rss']:.4f}")

    # Generate null distribution
    n_null = 1000
    print(f"\n[null] generating {n_null} synthetic null curves...")
    null_curves = generate_null_curves(freq, h_c, n_realisations=n_null)

    print(f"[null] fitting beat model to each null realisation...")
    null_amps = []
    for i, h_null in enumerate(null_curves):
        result = measure_amp(freq, h_null)
        if result is not None:
            null_amps.append(result["amp_log"])
        if (i + 1) % 100 == 0:
            print(f"   {i + 1}/{n_null}")

    null_amps = np.array(null_amps)
    print(f"\n[null] {len(null_amps)} successful fits")
    print(f"[null] amp_log distribution:")
    print(f"   median = {np.median(null_amps):.4f}")
    print(f"   mean   = {np.mean(null_amps):.4f}")
    print(f"   std    = {np.std(null_amps):.4f}")
    print(f"   95%    = {np.percentile(null_amps, 95):.4f}")
    print(f"   99%    = {np.percentile(null_amps, 99):.4f}")

    # Statistical significance
    pct = percentileofscore(null_amps, real["amp_log"], kind="strict")
    p_value = (100.0 - pct) / 100.0
    n_above = np.sum(null_amps >= real["amp_log"])

    # Sigma equivalent (one-sided)
    if p_value > 0:
        from scipy.stats import norm
        sigma = norm.isf(p_value)
    else:
        sigma = float("inf")

    print(f"\n{'=' * 70}")
    print(f"RESULT")
    print(f"{'=' * 70}")
    print(f"Real amp_log:          {real['amp_log']:.4f}")
    print(f"Null median amp_log:   {np.median(null_amps):.4f}")
    print(f"Null 95th percentile:  {np.percentile(null_amps, 95):.4f}")
    print(f"Real / null median:    {real['amp_log'] / max(np.median(null_amps), 1e-9):.2f}x")
    print(f"P-value (one-sided):   {p_value:.4f}")
    print(f"Significance:          {sigma:.2f}σ")
    print(f"Null curves >= real:   {n_above}/{len(null_amps)}")

    if sigma > 3:
        verdict = "DETECTION (3σ)"
    elif sigma > 2:
        verdict = "MARGINAL (2σ)"
    elif sigma > 1:
        verdict = "WEAK HINT (1σ)"
    else:
        verdict = "CONSISTENT WITH NULL (no beat)"
    print(f"Verdict:               {verdict}")

    output = {
        "data_url": URL,
        "curve_file": str(curve_file),
        "n_points": int(len(freq)),
        "real": real,
        "null": {
            "n_realisations": int(len(null_amps)),
            "median": float(np.median(null_amps)),
            "mean": float(np.mean(null_amps)),
            "std": float(np.std(null_amps)),
            "p95": float(np.percentile(null_amps, 95)),
            "p99": float(np.percentile(null_amps, 99)),
        },
        "p_value": float(p_value),
        "sigma": float(sigma),
        "verdict": verdict,
    }
    out_path = DATA_DIR / "result_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 9))

        axes[0].loglog(freq, h_c, "o-", markersize=3, label="Real NANOGrav 15-yr")
        # Reconstructed model
        log_f = np.log10(freq)
        model = (real["baseline_slope"] * log_f + real["baseline_intercept"] +
                 real["amp_log"] * np.sin(real["omega"] * log_f + real["phase"]))
        axes[0].loglog(freq, 10 ** model, "r--",
                       label=f"baseline + beat (amp={real['amp_log']:.3f})")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("Characteristic strain h_c")
        axes[0].set_title("NANOGrav 15-yr Sensitivity Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(null_amps, bins=50, alpha=0.7, color="gray", label="Null distribution")
        axes[1].axvline(real["amp_log"], color="red", lw=2,
                        label=f"Real: {real['amp_log']:.3f}")
        axes[1].axvline(np.percentile(null_amps, 95), color="orange",
                        ls="--", label="95th percentile")
        axes[1].set_xlabel("amp_log")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Null vs Real: {sigma:.1f}σ ({verdict})")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("P8_nanograv_v2.png", dpi=150)
        print("Plot: P8_nanograv_v2.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
