# P14: Quark Koide Test (Wrapper Around SM-D5 v3)

## Status
**P14 in v1 was a duplicate of SM-D5** that tried to parse the PDG PDF
with regex. The PDF parsing failed because PDG uses subscripts and
asymmetric errors that don't match simple regex patterns.

This v2 abandons PDF parsing entirely and uses **hardcoded FLAG 2024 +
PDG 2024 values** (the same approach as `SMD5_complete_test.py`). It is
explicitly a streamlined wrapper around SM-D5 v3, focused only on the
Q_up and Q_down predictions.

## Prediction
- Q_up = (m_u + m_c + m_t) / (√m_u + √m_c + √m_t)² ≈ 8/9
- Q_down = (m_d + m_s + m_b) / (√m_d + √m_s + √m_b)² ≈ 3/4
- Both at MSbar(M_Z), with full uncertainty propagation

## Hardware
Any laptop. <1 minute.

## Software
```bash
pip install numpy scipy matplotlib
```

## Data
**No download needed** — values from PDG 2024 (https://pdg.lbl.gov)
and FLAG 2024 (https://flag.unibe.ch) are hardcoded as constants.
Both sources are publicly accessible; the values can be verified by hand.

## Script

```python
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
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/p14_quark_koide")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MASS DATA (PDG 2024 + FLAG 2024)
# Format: (central, +sigma, -sigma) in MeV
# Source URLs (manually verifiable):
#   https://pdg.lbl.gov/2024/reviews/rpp2024-rev-quark-masses.pdf
#   https://flag.unibe.ch/2024/Media?action=show&category=quark%20masses
# ============================================================

# MSbar masses at μ = M_Z (running from PDG/FLAG values)
MSBAR_MZ = {
    "u": (1.27, 0.29, 0.15),     # MeV
    "d": (2.84, 0.29, 0.10),     # MeV
    "s": (55.5, 5.1, 5.1),       # MeV
    "c": (619.0, 10.0, 10.0),    # MeV
    "b": (2855.0, 21.0, 21.0),   # MeV
    "t": (171700.0, 1100.0, 1100.0),  # MeV
}

# Same masses at μ = 2 GeV (from FLAG 2024)
MSBAR_2GEV = {
    "u": (2.16, 0.49, 0.26),     # MeV
    "d": (4.67, 0.48, 0.17),     # MeV
    "s": (93.4, 8.6, 8.6),       # MeV
    "c": (1270.0, 20.0, 20.0),   # MeV  (at μ = m_c)
    "b": (4180.0, 30.0, 30.0),   # MeV  (at μ = m_b)
    "t": (162500.0, 1100.0, 1100.0),  # MeV
}

# Predictions
TARGET_UP = 8.0 / 9.0     # 0.888889
TARGET_DOWN = 3.0 / 4.0   # 0.750000


def koide_Q(m1, m2, m3):
    """Koide ratio Q = (sum m) / (sum sqrt(m))²."""
    num = m1 + m2 + m3
    den = (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)) ** 2
    return num / den


def sample_mass(central, sig_p, sig_m, rng):
    """Sample a single mass from an asymmetric Gaussian (split-normal)."""
    if sig_p == 0 and sig_m == 0:
        return central
    if rng.random() < 0.5:
        return central - abs(rng.normal(0, sig_m))
    return central + abs(rng.normal(0, sig_p))


def koide_with_errors(m1_data, m2_data, m3_data, n_samples=200_000, seed=42):
    """Monte Carlo error propagation for the Koide ratio."""
    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples)
    n_valid = 0
    for _ in range(n_samples):
        m1 = sample_mass(*m1_data, rng)
        m2 = sample_mass(*m2_data, rng)
        m3 = sample_mass(*m3_data, rng)
        if m1 > 0 and m2 > 0 and m3 > 0:
            samples[n_valid] = koide_Q(m1, m2, m3)
            n_valid += 1
    samples = samples[:n_valid]
    return {
        "samples": samples,
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "ci_68": [float(np.percentile(samples, 16)),
                   float(np.percentile(samples, 84))],
        "ci_95": [float(np.percentile(samples, 2.5)),
                   float(np.percentile(samples, 97.5))],
    }


def evaluate_target(stats, target, label):
    """Compare measured Q against a rational target."""
    z = (stats["mean"] - target) / stats["std"]
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
        "target": target,
        "z": float(z),
        "in_68_CI": bool(in_68),
        "in_95_CI": bool(in_95),
        "status": status,
    }


def main():
    print("=" * 70)
    print("P14: Quark Koide Test (PDG 2024 + FLAG 2024)")
    print("=" * 70)
    print(f"Predictions: Q_up ≈ 8/9 = {TARGET_UP:.6f}")
    print(f"             Q_down ≈ 3/4 = {TARGET_DOWN:.6f}")
    print()

    # Up-type at M_Z
    up_mz = koide_with_errors(
        MSBAR_MZ["u"], MSBAR_MZ["c"], MSBAR_MZ["t"])
    print(f"Q_up (MSbar at M_Z): {up_mz['mean']:.6f} ± {up_mz['std']:.6f}")
    print(f"  68% CI: [{up_mz['ci_68'][0]:.6f}, {up_mz['ci_68'][1]:.6f}]")
    print(f"  95% CI: [{up_mz['ci_95'][0]:.6f}, {up_mz['ci_95'][1]:.6f}]")
    eval_up_mz = evaluate_target(up_mz, TARGET_UP, "Q_up vs 8/9 (M_Z)")
    print(f"  Target 8/9: |z| = {abs(eval_up_mz['z']):.2f}σ → {eval_up_mz['status']}")

    # Down-type at M_Z
    down_mz = koide_with_errors(
        MSBAR_MZ["d"], MSBAR_MZ["s"], MSBAR_MZ["b"])
    print(f"\nQ_down (MSbar at M_Z): {down_mz['mean']:.6f} ± {down_mz['std']:.6f}")
    print(f"  68% CI: [{down_mz['ci_68'][0]:.6f}, {down_mz['ci_68'][1]:.6f}]")
    print(f"  95% CI: [{down_mz['ci_95'][0]:.6f}, {down_mz['ci_95'][1]:.6f}]")
    eval_down_mz = evaluate_target(down_mz, TARGET_DOWN, "Q_down vs 3/4 (M_Z)")
    print(f"  Target 3/4: |z| = {abs(eval_down_mz['z']):.2f}σ → {eval_down_mz['status']}")

    # Same at 2 GeV — to check scale dependence
    up_2gev = koide_with_errors(
        MSBAR_2GEV["u"], MSBAR_2GEV["c"], MSBAR_2GEV["t"])
    down_2gev = koide_with_errors(
        MSBAR_2GEV["d"], MSBAR_2GEV["s"], MSBAR_2GEV["b"])

    print(f"\n{'=' * 70}")
    print("SCALE DEPENDENCE")
    print(f"{'=' * 70}")
    print(f"Q_up   at 2 GeV:  {up_2gev['mean']:.6f} ± {up_2gev['std']:.6f}")
    print(f"Q_up   at M_Z:    {up_mz['mean']:.6f} ± {up_mz['std']:.6f}")
    shift_up = up_mz["mean"] - up_2gev["mean"]
    sigma_up = abs(shift_up) / max(up_mz["std"], up_2gev["std"])
    print(f"Shift: {shift_up:+.6f} = {sigma_up:.1f}σ")
    print()
    print(f"Q_down at 2 GeV:  {down_2gev['mean']:.6f} ± {down_2gev['std']:.6f}")
    print(f"Q_down at M_Z:    {down_mz['mean']:.6f} ± {down_mz['std']:.6f}")
    shift_down = down_mz["mean"] - down_2gev["mean"]
    sigma_down = abs(shift_down) / max(down_mz["std"], down_2gev["std"])
    print(f"Shift: {shift_down:+.6f} = {sigma_down:.1f}σ")
    print()
    print("If shifts are >>1σ, Q is NOT scale-invariant.")
    print("If Q_up = 8/9 only at M_Z (not at 2 GeV), the prediction picks")
    print("out a specific energy scale.")

    # Nearest rationals (anti-cherry-pick)
    print(f"\n{'=' * 70}")
    print("NEAREST LOW-DENOMINATOR RATIONALS (denom ≤ 12)")
    print(f"{'=' * 70}")
    for label, stats in [("Q_up at M_Z", up_mz), ("Q_down at M_Z", down_mz)]:
        print(f"\n{label}: {stats['mean']:.6f} ± {stats['std']:.6f}")
        candidates = []
        for q in range(2, 13):
            for p in range(1, q):
                from math import gcd
                if gcd(p, q) != 1:
                    continue
                frac = p / q
                if 0.5 < frac < 1.0:
                    delta = abs(frac - stats["mean"])
                    z = delta / stats["std"]
                    candidates.append((z, p, q, frac, delta))
        candidates.sort()
        for z, p, q, frac, delta in candidates[:5]:
            marker = "  ←" if z < 1 else ""
            print(f"  {p}/{q:<2} = {frac:.6f}  Δ={delta:.6f}  ({z:.2f}σ){marker}")

    # Save
    out = {
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
        "scale_dependence": {
            "shift_up_2GeV_to_MZ": float(shift_up),
            "sigma_up": float(sigma_up),
            "shift_down_2GeV_to_MZ": float(shift_down),
            "sigma_down": float(sigma_down),
        },
    }
    out_path = DATA_DIR / "result_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(up_mz["samples"], bins=80, density=True, alpha=0.7,
                    color="blue", label="Q_up MC samples")
        axes[0].axvline(TARGET_UP, color="red", lw=2, label="8/9")
        axes[0].axvline(up_mz["mean"], color="black", ls="--",
                       label=f"mean = {up_mz['mean']:.4f}")
        axes[0].set_xlabel("Q_up")
        axes[0].set_ylabel("PDF")
        axes[0].set_title(f"Q_up at MSbar(M_Z): {abs(eval_up_mz['z']):.2f}σ from 8/9")
        axes[0].legend()

        axes[1].hist(down_mz["samples"], bins=80, density=True, alpha=0.7,
                    color="red", label="Q_down MC samples")
        axes[1].axvline(TARGET_DOWN, color="blue", lw=2, label="3/4")
        axes[1].axvline(down_mz["mean"], color="black", ls="--",
                       label=f"mean = {down_mz['mean']:.4f}")
        axes[1].set_xlabel("Q_down")
        axes[1].set_ylabel("PDF")
        axes[1].set_title(f"Q_down at MSbar(M_Z): {abs(eval_down_mz['z']):.2f}σ from 3/4")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("P14_quark_koide_v2.png", dpi=150)
        print("Plot: P14_quark_koide_v2.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
```

## Why No Download Required

The PDG and FLAG values are stable, infrequently-updated, fully public,
and trivially small (12 numbers total). Hardcoding them is more reliable
than parsing PDFs and lets the script run completely offline. The data
sources are documented in comments and in the JSON output for verification.

If new PDG/FLAG values are released, update the constants at the top
of the script.

## Expected Result

Based on SMD5_complete_test_updated.py results:
- Q_up = 0.888494 ± 0.000909 → 8/9 at 0.44σ → CONSISTENT
- Q_down = 0.743974 ± 0.007038 → 3/4 at 0.86σ → CONSISTENT
- Q_up shift between 2 GeV and M_Z: ~48σ → NOT scale-invariant

## Timeline: <1 minute
