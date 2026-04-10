#!/usr/bin/env python3
"""
P14_quark_koide_test_v2.py

Tests the quark Koide prediction Q_up ≈ 8/9 and Q_down ≈ 3/4
using FLAG 2024 + PDG 2024 mass values with full uncertainty propagation.

Fix from v1:
  - No PDF parsing (v1 failed because PDG PDF uses subscripts/special chars)
  - Hardcoded mass values from canonical published sources
  - 200,000 Monte Carlo samples for proper uncertainty propagation
  - Reports CI, sigma deviation, and conservative consistency status
"""
from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

DEFAULT_DATA_DIR = Path("data/p14_quark_koide")

# ============================================================
# MASS DATA (PDG 2024 + FLAG 2024)
# Format: (central, +sigma, -sigma) in MeV
# Source URLs (manually verifiable):
#   https://pdg.lbl.gov/2024/reviews/rpp2024-rev-quark-masses.pdf
#   https://flag.unibe.ch/2024/Media?action=show&category=quark%20masses
# ============================================================

MassDatum = Tuple[float, float, float]
MassTable = Dict[str, MassDatum]

# MSbar masses at μ = M_Z (running from PDG/FLAG values)
MSBAR_MZ: MassTable = {
    "u": (1.27, 0.29, 0.15),       # MeV
    "d": (2.84, 0.29, 0.10),       # MeV
    "s": (55.5, 5.1, 5.1),         # MeV
    "c": (619.0, 10.0, 10.0),      # MeV
    "b": (2855.0, 21.0, 21.0),     # MeV
    "t": (171700.0, 1100.0, 1100.0),  # MeV
}

# Same masses at μ = 2 GeV (from FLAG 2024)
MSBAR_2GEV: MassTable = {
    "u": (2.16, 0.49, 0.26),       # MeV
    "d": (4.67, 0.48, 0.17),       # MeV
    "s": (93.4, 8.6, 8.6),         # MeV
    "c": (1270.0, 20.0, 20.0),     # MeV  (at μ = m_c)
    "b": (4180.0, 30.0, 30.0),     # MeV  (at μ = m_b)
    "t": (162500.0, 1100.0, 1100.0),  # MeV
}

# Predictions
TARGET_UP = 8.0 / 9.0      # 0.888889
TARGET_DOWN = 3.0 / 4.0    # 0.750000


def koide_q(m1: np.ndarray | float, m2: np.ndarray | float, m3: np.ndarray | float) -> np.ndarray | float:
    """Koide ratio Q = (sum m) / (sum sqrt(m))²."""
    numerator = m1 + m2 + m3
    denominator = (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)) ** 2
    return numerator / denominator


def sample_mass_split_normal(data: MassDatum, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample masses from an asymmetric Gaussian (split-normal proxy)."""
    central, sig_p, sig_m = data
    if sig_p == 0 and sig_m == 0:
        return np.full(n_samples, central, dtype=float)

    choose_left = rng.random(n_samples) < 0.5
    draws = np.empty(n_samples, dtype=float)
    n_left = int(np.sum(choose_left))
    n_right = int(n_samples - n_left)

    if n_left:
        draws[choose_left] = central - np.abs(rng.normal(0.0, sig_m, size=n_left))
    if n_right:
        draws[~choose_left] = central + np.abs(rng.normal(0.0, sig_p, size=n_right))
    return draws


def koide_with_errors(
    m1_data: MassDatum,
    m2_data: MassDatum,
    m3_data: MassDatum,
    n_samples: int = 200_000,
    seed: int = 42,
) -> dict:
    """Monte Carlo error propagation for the Koide ratio."""
    rng = np.random.default_rng(seed)

    m1 = sample_mass_split_normal(m1_data, n_samples, rng)
    m2 = sample_mass_split_normal(m2_data, n_samples, rng)
    m3 = sample_mass_split_normal(m3_data, n_samples, rng)

    valid = (m1 > 0.0) & (m2 > 0.0) & (m3 > 0.0)
    samples = koide_q(m1[valid], m2[valid], m3[valid])

    if samples.size == 0:
        raise RuntimeError("No valid Monte Carlo samples survived positivity cuts.")

    return {
        "samples": samples,
        "n_samples_requested": int(n_samples),
        "n_samples_valid": int(samples.size),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "ci_68": [float(np.percentile(samples, 16)), float(np.percentile(samples, 84))],
        "ci_95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
    }


def evaluate_target(stats: dict, target: float, label: str) -> dict:
    """Compare measured Q against a rational target."""
    std = stats["std"]
    z = (stats["mean"] - target) / std if std > 0 else float("inf")
    in_68 = stats["ci_68"][0] <= target <= stats["ci_68"][1]
    in_95 = stats["ci_95"][0] <= target <= stats["ci_95"][1]

    if abs(z) < 2:
        status = "CONSISTENT within 2σ"
    elif abs(z) < 3:
        status = "MARGINAL (2-3σ)"
    else:
        status = "EXCLUDED at 3σ"

    return {
        "label": label,
        "target": float(target),
        "z": float(z),
        "in_68_CI": bool(in_68),
        "in_95_CI": bool(in_95),
        "status": status,
    }


def nearest_rationals(value: float, sigma: float, max_denominator: int = 12, top_k: int = 5) -> list[dict]:
    """Find nearest low-denominator rational approximants in (0.5, 1.0)."""
    candidates: list[tuple[float, int, int, float, float]] = []
    for q in range(2, max_denominator + 1):
        for p in range(1, q):
            if gcd(p, q) != 1:
                continue
            frac = p / q
            if 0.5 < frac < 1.0:
                delta = abs(frac - value)
                z = delta / sigma if sigma > 0 else float("inf")
                candidates.append((z, p, q, frac, delta))
    candidates.sort()

    return [
        {
            "fraction": f"{p}/{q}",
            "value": float(frac),
            "delta": float(delta),
            "z": float(z),
        }
        for z, p, q, frac, delta in candidates[:top_k]
    ]


def print_stats_block(label: str, stats: dict, target: float, eval_result: dict) -> None:
    print(f"{label}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"  68% CI: [{stats['ci_68'][0]:.6f}, {stats['ci_68'][1]:.6f}]")
    print(f"  95% CI: [{stats['ci_95'][0]:.6f}, {stats['ci_95'][1]:.6f}]")
    print(f"  Target {target:.6f}: |z| = {abs(eval_result['z']):.2f}σ → {eval_result['status']}")


def save_outputs(data_dir: Path, payload: dict) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "result_v2.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def make_plot(data_dir: Path, up_mz: dict, down_mz: dict, eval_up: dict, eval_down: dict) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(up_mz["samples"], bins=80, density=True, alpha=0.7, label="Q_up MC samples")
    axes[0].axvline(TARGET_UP, lw=2, label="8/9")
    axes[0].axvline(up_mz["mean"], ls="--", label=f"mean = {up_mz['mean']:.4f}")
    axes[0].set_xlabel("Q_up")
    axes[0].set_ylabel("PDF")
    axes[0].set_title(f"Q_up at MSbar(M_Z): {abs(eval_up['z']):.2f}σ from 8/9")
    axes[0].legend()

    axes[1].hist(down_mz["samples"], bins=80, density=True, alpha=0.7, label="Q_down MC samples")
    axes[1].axvline(TARGET_DOWN, lw=2, label="3/4")
    axes[1].axvline(down_mz["mean"], ls="--", label=f"mean = {down_mz['mean']:.4f}")
    axes[1].set_xlabel("Q_down")
    axes[1].set_ylabel("PDF")
    axes[1].set_title(f"Q_down at MSbar(M_Z): {abs(eval_down['z']):.2f}σ from 3/4")
    axes[1].legend()

    plt.tight_layout()
    plot_path = data_dir / "P14_quark_koide_v2.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quark Koide test using hardcoded FLAG 2024 + PDG 2024 masses.")
    parser.add_argument("--n-samples", type=int, default=200_000, help="Monte Carlo sample count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Monte Carlo.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory for JSON and plot outputs.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip matplotlib plot generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir: Path = args.output_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P14: Quark Koide Test (PDG 2024 + FLAG 2024)")
    print("=" * 70)
    print(f"Predictions: Q_up ≈ 8/9 = {TARGET_UP:.6f}")
    print(f"             Q_down ≈ 3/4 = {TARGET_DOWN:.6f}")
    print(f"Monte Carlo samples: {args.n_samples:,}")
    print()

    up_mz = koide_with_errors(MSBAR_MZ["u"], MSBAR_MZ["c"], MSBAR_MZ["t"], n_samples=args.n_samples, seed=args.seed)
    eval_up_mz = evaluate_target(up_mz, TARGET_UP, "Q_up vs 8/9 (M_Z)")
    print_stats_block("Q_up (MSbar at M_Z)", up_mz, TARGET_UP, eval_up_mz)

    down_mz = koide_with_errors(MSBAR_MZ["d"], MSBAR_MZ["s"], MSBAR_MZ["b"], n_samples=args.n_samples, seed=args.seed + 1)
    eval_down_mz = evaluate_target(down_mz, TARGET_DOWN, "Q_down vs 3/4 (M_Z)")
    print()
    print_stats_block("Q_down (MSbar at M_Z)", down_mz, TARGET_DOWN, eval_down_mz)

    up_2gev = koide_with_errors(MSBAR_2GEV["u"], MSBAR_2GEV["c"], MSBAR_2GEV["t"], n_samples=args.n_samples, seed=args.seed + 2)
    down_2gev = koide_with_errors(MSBAR_2GEV["d"], MSBAR_2GEV["s"], MSBAR_2GEV["b"], n_samples=args.n_samples, seed=args.seed + 3)

    shift_up = up_mz["mean"] - up_2gev["mean"]
    shift_down = down_mz["mean"] - down_2gev["mean"]
    sigma_up = abs(shift_up) / max(up_mz["std"], up_2gev["std"])
    sigma_down = abs(shift_down) / max(down_mz["std"], down_2gev["std"])

    print(f"\n{'=' * 70}")
    print("SCALE DEPENDENCE")
    print(f"{'=' * 70}")
    print(f"Q_up   at 2 GeV:  {up_2gev['mean']:.6f} ± {up_2gev['std']:.6f}")
    print(f"Q_up   at M_Z:    {up_mz['mean']:.6f} ± {up_mz['std']:.6f}")
    print(f"Shift: {shift_up:+.6f} = {sigma_up:.1f}σ")
    print()
    print(f"Q_down at 2 GeV:  {down_2gev['mean']:.6f} ± {down_2gev['std']:.6f}")
    print(f"Q_down at M_Z:    {down_mz['mean']:.6f} ± {down_mz['std']:.6f}")
    print(f"Shift: {shift_down:+.6f} = {sigma_down:.1f}σ")
    print()
    print("If shifts are >>1σ, Q is NOT scale-invariant.")
    print("If Q_up = 8/9 only at M_Z (not at 2 GeV), the prediction picks")
    print("out a specific energy scale.")

    up_rationals = nearest_rationals(up_mz["mean"], up_mz["std"])
    down_rationals = nearest_rationals(down_mz["mean"], down_mz["std"])

    print(f"\n{'=' * 70}")
    print("NEAREST LOW-DENOMINATOR RATIONALS (denom ≤ 12)")
    print(f"{'=' * 70}")
    for label, stats, rationals in (
        ("Q_up at M_Z", up_mz, up_rationals),
        ("Q_down at M_Z", down_mz, down_rationals),
    ):
        print(f"\n{label}: {stats['mean']:.6f} ± {stats['std']:.6f}")
        for item in rationals:
            marker = "  ←" if item["z"] < 1 else ""
            print(
                f"  {item['fraction']:<4} = {item['value']:.6f}  "
                f"Δ={item['delta']:.6f}  ({item['z']:.2f}σ){marker}"
            )

    payload = {
        "data_sources": {
            "PDG_2024": "https://pdg.lbl.gov/2024/reviews/rpp2024-rev-quark-masses.pdf",
            "FLAG_2024": "https://flag.unibe.ch/2024",
        },
        "predictions": {
            "Q_up_target": TARGET_UP,
            "Q_down_target": TARGET_DOWN,
        },
        "results_at_MZ": {
            "Q_up": {k: v for k, v in up_mz.items() if k != "samples"},
            "Q_down": {k: v for k, v in down_mz.items() if k != "samples"},
            "evaluation_up": eval_up_mz,
            "evaluation_down": eval_down_mz,
        },
        "results_at_2GeV": {
            "Q_up": {k: v for k, v in up_2gev.items() if k != "samples"},
            "Q_down": {k: v for k, v in down_2gev.items() if k != "samples"},
        },
        "scale_dependence": {
            "shift_up_2GeV_to_MZ": float(shift_up),
            "sigma_up": float(sigma_up),
            "shift_down_2GeV_to_MZ": float(shift_down),
            "sigma_down": float(sigma_down),
        },
        "nearest_rationals": {
            "Q_up_at_MZ": up_rationals,
            "Q_down_at_MZ": down_rationals,
        },
    }
    out_path = save_outputs(data_dir, payload)
    print(f"\nSaved: {out_path}")

    if args.skip_plot:
        print("Plot skipped (--skip-plot).")
    else:
        plot_path = make_plot(data_dir, up_mz, down_mz, eval_up_mz, eval_down_mz)
        if plot_path is None:
            print("Plot skipped (matplotlib not available).")
        else:
            print(f"Plot: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
