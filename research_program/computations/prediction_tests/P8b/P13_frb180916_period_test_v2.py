#!/usr/bin/env python3
"""
P13_frb180916_period_test_v2.py

Tests the 16.35-day period of FRB 20180916B with:
  1. Rayleigh/Lomb-Scargle-style event-time periodogram
  2. Comparison with CCDR predicted period range
  3. Joint test with other published periodic FRBs

This implementation follows the uploaded P13 specification, while making
it easier to run from the command line and save outputs reproducibly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

DATA_DIR = Path("data/p13_frb180916")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Burst arrival times from CHIME/FRB Collaboration 2020, Nature 582, 351.
# Table S1 fallback values as provided in the user's spec.
PUBLISHED_BURSTS_MJD = np.array([
    58370.21, 58372.34, 58372.37, 58372.71, 58375.49, 58386.84,
    58387.20, 58388.14, 58400.45, 58401.97, 58415.67, 58416.18,
    58420.73, 58428.79, 58432.32, 58435.32, 58448.95, 58452.07,
    58453.66, 58467.28, 58467.49, 58468.04, 58468.85, 58475.37,
    58479.65, 58481.38, 58482.85, 58495.78, 58504.81, 58522.86,
    58530.93, 58535.97, 58563.30, 58569.00, 58572.95, 58587.26,
    58592.98, 58611.55, 58616.85, 58620.78, 58632.40, 58654.51,
    58665.99, 58672.53,
], dtype=float)

CCDR_PERIOD_MIN = 1.0
CCDR_PERIOD_MAX = 1000.0

OTHER_PERIODIC_FRBS: Dict[str, float] = {
    "FRB 121102": 157.0,
    "FRB 20180916B": 16.35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P13: FRB 20180916B 16-day period vs CCDR range"
    )
    parser.add_argument(
        "--period-min",
        type=float,
        default=1.0,
        help="Minimum trial period in days (default: 1.0)",
    )
    parser.add_argument(
        "--period-max",
        type=float,
        default=100.0,
        help="Maximum trial period in days (default: 100.0)",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=5000,
        help="Number of trial periods in the scan (default: 5000)",
    )
    parser.add_argument(
        "--published-period",
        type=float,
        default=16.35,
        help="Published reference period in days (default: 16.35)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory (default: {DATA_DIR})",
    )
    return parser.parse_args()


def rayleigh_periodogram(times: np.ndarray, period_grid: np.ndarray) -> np.ndarray:
    """
    Compute the Rayleigh event-time statistic across a period grid.

    For event times t_i and test period P:
        Z(P) = 2/N * [ (sum cos(2π t_i/P))² + (sum sin(2π t_i/P))² ]

    Under the null of a Poisson process with no preferred phase, Z is
    asymptotically chi-square distributed with 2 degrees of freedom.
    """
    times = np.asarray(times, dtype=float)
    period_grid = np.asarray(period_grid, dtype=float)

    n = times.size
    if n == 0:
        raise ValueError("times must contain at least one burst")
    if np.any(period_grid <= 0):
        raise ValueError("all trial periods must be positive")

    phases = 2.0 * np.pi * times[:, None] / period_grid[None, :]
    cos_sum = np.cos(phases).sum(axis=0)
    sin_sum = np.sin(phases).sum(axis=0)
    return 2.0 * (cos_sum**2 + sin_sum**2) / n


def false_alarm_probability(z: float, n_trials: int = 1) -> float:
    """
    Bonferroni-style multi-trial false-alarm probability for Rayleigh Z.

    Single-trial tail probability under the null is exp(-Z/2).
    """
    p_single = float(np.exp(-z / 2.0))
    fap = 1.0 - (1.0 - p_single) ** max(1, int(n_trials))
    return float(np.clip(fap, 0.0, 1.0))


def fap_to_sigma(fap: float) -> float:
    """Convert false-alarm probability to an approximate two-sided sigma."""
    if fap <= 0.0:
        return float("inf")
    if fap >= 1.0:
        return 0.0

    from scipy.stats import norm

    return float(norm.isf(fap / 2.0))


def summarize_other_periodic_frbs(
    period_min: float,
    period_max: float,
) -> Tuple[int, Dict[str, Dict[str, object]]]:
    summary: Dict[str, Dict[str, object]] = {}
    n_in_range = 0
    for name, period in OTHER_PERIODIC_FRBS.items():
        in_range = period_min <= period <= period_max
        n_in_range += int(in_range)
        summary[name] = {
            "period_days": float(period),
            "in_ccdr_range": bool(in_range),
        }
    return n_in_range, summary


def make_plot(
    output_path: Path,
    period_grid: np.ndarray,
    powers: np.ndarray,
    times: np.ndarray,
    published_period: float,
    peak_period: float,
    ccdr_min: float,
    ccdr_max: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(period_grid, powers, lw=0.7)
    axes[0].axvline(published_period, linestyle="--", label=f"Published {published_period:.2f} d")
    axes[0].axvline(peak_period, linestyle=":", label=f"Peak {peak_period:.2f} d")
    axes[0].axvspan(ccdr_min, ccdr_max, alpha=0.12, label="CCDR range")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Period (days)")
    axes[0].set_ylabel("Rayleigh Z")
    axes[0].set_title(f"FRB 20180916B periodogram ({len(times)} bursts)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    phase = np.mod(times, published_period) / published_period
    axes[1].hist(phase, bins=20, alpha=0.75)
    axes[1].set_xlabel(f"Phase (P = {published_period:.2f} d)")
    axes[1].set_ylabel("Number of bursts")
    axes[1].set_title("Phase distribution at published period")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    times = np.array(PUBLISHED_BURSTS_MJD, dtype=float)
    period_grid = np.linspace(args.period_min, args.period_max, args.n_grid)
    powers = rayleigh_periodogram(times, period_grid)

    peak_idx = int(np.argmax(powers))
    peak_period = float(period_grid[peak_idx])
    peak_power = float(powers[peak_idx])

    # Keep the effective-trials logic close to the user's spec.
    n_eff = len(times)
    fap_peak = false_alarm_probability(peak_power, n_trials=len(period_grid))
    sigma_peak = fap_to_sigma(fap_peak)

    idx_pub = int(np.argmin(np.abs(period_grid - args.published_period)))
    z_at_published = float(powers[idx_pub])
    fap_published_single = false_alarm_probability(z_at_published, n_trials=1)

    in_range = CCDR_PERIOD_MIN <= peak_period <= CCDR_PERIOD_MAX
    n_in_range, other_summary = summarize_other_periodic_frbs(
        CCDR_PERIOD_MIN,
        CCDR_PERIOD_MAX,
    )

    if n_in_range == len(OTHER_PERIODIC_FRBS):
        verdict = "All known periodic FRBs consistent with CCDR range"
    elif n_in_range > 0:
        verdict = "Partial consistency"
    else:
        verdict = "All published periods outside CCDR range — falsified"

    print("=" * 70)
    print("P13: FRB 20180916B 16-Day Period — CCDR Prediction Test")
    print("=" * 70)
    print(f"[data] {len(times)} bursts spanning {times.max() - times.min():.1f} days")
    print(
        f"[search] {len(period_grid)} trial periods from "
        f"{period_grid[0]:.1f} to {period_grid[-1]:.1f} days"
    )

    print(f"\n{'=' * 70}")
    print("PERIODOGRAM RESULT")
    print(f"{'=' * 70}")
    print(f"Peak period:           {peak_period:.3f} days")
    print(f"Published period:      {args.published_period:.2f} days")
    print(
        "Match:                 "
        f"{abs(peak_period - args.published_period) / args.published_period * 100:.1f}% offset"
    )
    print(f"Peak Rayleigh Z:       {peak_power:.2f}")
    print(f"False-alarm prob:      {fap_peak:.2e}")
    print(f"Significance:          {sigma_peak:.1f}σ")
    print()
    print(f"At published period {args.published_period:.2f} d:")
    print(f"  Z = {z_at_published:.2f}")
    print(f"  Single-trial FAP = {fap_published_single:.2e}")

    print(f"\n{'=' * 70}")
    print("CCDR PREDICTION TEST")
    print(f"{'=' * 70}")
    print(f"CCDR predicted period range: {CCDR_PERIOD_MIN:.0f} to {CCDR_PERIOD_MAX:.0f} days")
    print(f"Measured period:             {peak_period:.2f} days")
    print(
        "Status: "
        + ("CONSISTENT with CCDR predicted range" if in_range else "OUTSIDE CCDR predicted range")
    )

    print(f"\n{'=' * 70}")
    print("OTHER PERIODIC FRBs (literature)")
    print(f"{'=' * 70}")
    for name, info in other_summary.items():
        status = "in range" if info["in_ccdr_range"] else "OUT OF RANGE"
        print(f"  {name}: P = {info['period_days']:.2f} days  [{status}]")
    print(f"\n{n_in_range}/{len(OTHER_PERIODIC_FRBS)} periodic FRBs in CCDR range")
    print(f"Verdict: {verdict}")
    print()
    print("Note: the CCDR range 1-1000 days is broad. With only 2 known")
    print("periodic FRBs in this test, it remains a weak consistency check.")
    print("The prediction would need to be sharpened to become strongly diagnostic.")

    result = {
        "data_source": "CHIME/FRB Collaboration 2020, Nature 582, 351",
        "n_bursts": int(len(times)),
        "time_span_days": float(times.max() - times.min()),
        "search_range_days": [float(period_grid[0]), float(period_grid[-1])],
        "n_trial_periods": int(len(period_grid)),
        "effective_trials_assumed": int(n_eff),
        "peak_period_days": peak_period,
        "peak_rayleigh_Z": peak_power,
        "false_alarm_probability": fap_peak,
        "sigma": sigma_peak,
        "published_period_days": float(args.published_period),
        "z_at_published_period": z_at_published,
        "single_trial_fap_at_published_period": fap_published_single,
        "ccdr_period_range_days": [CCDR_PERIOD_MIN, CCDR_PERIOD_MAX],
        "in_ccdr_range": bool(in_range),
        "other_periodic_frbs": other_summary,
        "n_in_range": int(n_in_range),
        "verdict": verdict,
    }

    json_path = output_dir / "result_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {json_path}")

    try:
        plot_path = output_dir / "P13_frb180916_v2.png"
        make_plot(
            plot_path,
            period_grid,
            powers,
            times,
            args.published_period,
            peak_period,
            CCDR_PERIOD_MIN,
            CCDR_PERIOD_MAX,
        )
        print(f"Plot: {plot_path}")
    except ImportError:
        print("[plot] matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
