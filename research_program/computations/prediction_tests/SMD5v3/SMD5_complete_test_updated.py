#!/usr/bin/env python3
"""
SMD5_complete_test_updated.py

Conservative Koide-analysis script that addresses common methodological criticisms:
  - separates descriptive results from hypothesis tests
  - uses reproducible Monte Carlo with vectorized sampling
  - avoids overclaiming PASS/FAIL unless a threshold is explicitly chosen
  - scans nearby low-denominator rationals instead of cherry-picking a single "next" rational
  - labels cross-scheme / constituent-mass comparisons as heuristic only
  - removes speculative theory claims (RG invariance, UV origin, future precision timelines)

Important:
  This script does not validate the underlying mass inputs. It treats the numbers below as the
  user-supplied dataset and propagates their quoted uncertainties consistently.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

# ============================================================
# SECTION 1: USER-SUPPLIED MASS DATA
# ============================================================
# Format: (central, sigma_plus, sigma_minus)

MSBAR_2GEV = {
    'u': (2.16, 0.49, 0.26),
    'd': (4.67, 0.48, 0.17),
    's': (93.4, 8.6, 8.6),
    'c': (1270, 20, 20),
    'b': (4180, 30, 30),
    't': (162500, 1100, 1100),
}

POLE_OR_CONSTITUENT = {
    'c': (1670, 70, 70),
    'b': (4780, 60, 60),
    't': (172690, 300, 300),
    # Light-quark entries below are constituent-model proxies, not pole masses.
    'u': (336, 10, 10),
    'd': (340, 10, 10),
    's': (486, 10, 10),
}

LEPTONS = {
    'e': (0.51099895, 0.0, 0.0),
    'mu': (105.6584, 0.0, 0.0),
    'tau': (1776.86, 0.12, 0.12),
}

MSBAR_MZ = {
    'u': (1.27, 0.29, 0.15),
    'd': (2.84, 0.29, 0.10),
    's': (55.5, 5.1, 5.1),
    'c': (619, 10, 10),
    'b': (2855, 21, 21),
    't': (171700, 1100, 1100),
}

# ============================================================
# SECTION 2: CORE MATH
# ============================================================

@dataclass
class KoideSummary:
    mean: float
    std: float
    median: float
    q16: float
    q84: float
    q025: float
    q975: float
    samples: np.ndarray


def koide_Q(m1: np.ndarray, m2: np.ndarray, m3: np.ndarray) -> np.ndarray:
    num = m1 + m2 + m3
    den = (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)) ** 2
    return num / den


def sample_split_normal(data: Tuple[float, float, float], size: int, rng: np.random.Generator) -> np.ndarray:
    """Simple two-piece Gaussian sampler around a positive central value."""
    central, sig_plus, sig_minus = data
    if sig_plus == 0 and sig_minus == 0:
        return np.full(size, central, dtype=float)

    side = rng.random(size) < 0.5
    draws = np.empty(size, dtype=float)
    draws[side] = central - np.abs(rng.normal(0.0, sig_minus, side.sum()))
    draws[~side] = central + np.abs(rng.normal(0.0, sig_plus, (~side).sum()))

    # Resample any nonphysical nonpositive values.
    bad = draws <= 0
    while np.any(bad):
        nbad = int(bad.sum())
        side_bad = rng.random(nbad) < 0.5
        repl = np.empty(nbad, dtype=float)
        repl[side_bad] = central - np.abs(rng.normal(0.0, sig_minus, side_bad.sum()))
        repl[~side_bad] = central + np.abs(rng.normal(0.0, sig_plus, (~side_bad).sum()))
        draws[bad] = repl
        bad = draws <= 0
    return draws


def koide_with_errors(m1_data, m2_data, m3_data, n_samples=200_000, seed=12345) -> KoideSummary:
    rng = np.random.default_rng(seed)
    m1 = sample_split_normal(m1_data, n_samples, rng)
    m2 = sample_split_normal(m2_data, n_samples, rng)
    m3 = sample_split_normal(m3_data, n_samples, rng)
    q = koide_Q(m1, m2, m3)
    p = np.percentile(q, [2.5, 16, 50, 84, 97.5])
    return KoideSummary(
        mean=float(np.mean(q)),
        std=float(np.std(q, ddof=1)),
        median=float(p[2]),
        q16=float(p[1]),
        q84=float(p[3]),
        q025=float(p[0]),
        q975=float(p[4]),
        samples=q,
    )


def nearest_rationals(x: float, qmax: int = 20, n_show: int = 8) -> List[Tuple[int, int, float, float]]:
    found = []
    for q in range(2, qmax + 1):
        for p in range(1, q):
            frac = p / q
            found.append((p, q, frac, abs(frac - x)))
    found.sort(key=lambda t: (t[3], t[1], t[0]))
    return found[:n_show]


def empirical_rational_hit_rate(samples: np.ndarray, qmax: int = 9, window_sigma: float = 2.0, n_grid: int = 5000) -> float:
    """
    Approximate post-hoc look-elsewhere rate on [2/3, 1].
    For each grid point x with local sigma set to sample std, ask whether there exists p/q (q<=qmax)
    within window_sigma * sigma. Compare to the observed situation descriptively.
    """
    sigma = float(np.std(samples, ddof=1))
    xs = np.linspace(2/3, 1.0, n_grid)
    rationals = [p / q for q in range(2, qmax + 1) for p in range(1, q)]
    hits = 0
    for x in xs:
        if any(abs(r - x) <= window_sigma * sigma for r in rationals):
            hits += 1
    return hits / n_grid


def z_to_target(summary: KoideSummary, target: float) -> float:
    return abs(summary.mean - target) / summary.std if summary.std > 0 else math.inf


def ci_contains(summary: KoideSummary, target: float, level: float = 0.95) -> bool:
    if level == 0.68:
        return summary.q16 <= target <= summary.q84
    if level == 0.95:
        return summary.q025 <= target <= summary.q975
    raise ValueError("Supported levels: 0.68, 0.95")


def report_target_test(name: str, summary: KoideSummary, target: float, label: str, sigma_threshold: float = 2.0) -> None:
    z = z_to_target(summary, target)
    inside95 = ci_contains(summary, target, 0.95)
    print(f"  {name}: target {label} = {target:.6f}")
    print(f"    |Q - target| = {abs(summary.mean - target):.6f}")
    print(f"    z = {z:.2f} (using propagated mass uncertainty only)")
    print(f"    In 68% CI: {'YES' if ci_contains(summary, target, 0.68) else 'NO'}")
    print(f"    In 95% CI: {'YES' if inside95 else 'NO'}")
    if z < sigma_threshold:
        print(f"    Conservative status: CONSISTENT within {sigma_threshold:.1f}σ")
    else:
        print(f"    Conservative status: TENSION at > {sigma_threshold:.1f}σ")


def print_nearest_rationals(name: str, summary: KoideSummary, qmax: int = 20) -> None:
    print(f"  {name}: nearest reduced fractions with denominator ≤ {qmax}")
    shown = []
    for p, q, frac, dist in nearest_rationals(summary.mean, qmax=qmax):
        g = math.gcd(p, q)
        rp, rq = p // g, q // g
        key = (rp, rq)
        if key in shown:
            continue
        shown.append(key)
        z = dist / summary.std if summary.std > 0 else math.inf
        print(f"    {rp}/{rq} = {rp/rq:.6f}   Δ={dist:.6f}   ({z:.2f}σ)")
        if len(shown) >= 6:
            break


def sensitivity_scan(reference_masses: Dict[str, Tuple[float, float, float]], keys: Tuple[str, str, str], frac_shift=0.01) -> None:
    k1, k2, k3 = keys
    base = [reference_masses[k1][0], reference_masses[k2][0], reference_masses[k3][0]]
    q0 = float(koide_Q(np.array([base[0]]), np.array([base[1]]), np.array([base[2]]))[0])
    print("  Local one-at-a-time sensitivity (1% upward shift of central value):")
    for i, key in enumerate(keys):
        shifted = base.copy()
        shifted[i] *= (1.0 + frac_shift)
        q1 = float(koide_Q(np.array([shifted[0]]), np.array([shifted[1]]), np.array([shifted[2]]))[0])
        dq = q1 - q0
        print(f"    {key}: ΔQ = {dq:+.6e} for +1% shift")


# ============================================================
# SECTION 3: MAIN ANALYSIS
# ============================================================

def main() -> None:
    N_MC = 200_000
    SEED = 12345

    print("=" * 76)
    print("SM-D5 COMPLETE TEST (CONSERVATIVE VERSION)")
    print(f"Monte Carlo samples: {N_MC:,}   RNG seed: {SEED}")
    print("Interpretation policy: descriptive first, hypothesis test second, no theory overclaim")
    print("=" * 76)

    lep = koide_with_errors(LEPTONS['e'], LEPTONS['mu'], LEPTONS['tau'], N_MC, seed=SEED)
    up_mz = koide_with_errors(MSBAR_MZ['u'], MSBAR_MZ['c'], MSBAR_MZ['t'], N_MC, seed=SEED + 1)
    down_mz = koide_with_errors(MSBAR_MZ['d'], MSBAR_MZ['s'], MSBAR_MZ['b'], N_MC, seed=SEED + 2)
    up_2g = koide_with_errors(MSBAR_2GEV['u'], MSBAR_2GEV['c'], MSBAR_2GEV['t'], N_MC, seed=SEED + 3)
    down_2g = koide_with_errors(MSBAR_2GEV['d'], MSBAR_2GEV['s'], MSBAR_2GEV['b'], N_MC, seed=SEED + 4)
    up_proxy = koide_with_errors(POLE_OR_CONSTITUENT['u'], POLE_OR_CONSTITUENT['c'], POLE_OR_CONSTITUENT['t'], N_MC, seed=SEED + 5)
    down_proxy = koide_with_errors(POLE_OR_CONSTITUENT['d'], POLE_OR_CONSTITUENT['s'], POLE_OR_CONSTITUENT['b'], N_MC, seed=SEED + 6)

    print("\n" + "=" * 76)
    print("DESCRIPTIVE RESULTS")
    print("=" * 76)
    for label, summary in [
        ("Leptons", lep),
        ("Up-type quarks MSbar(M_Z)", up_mz),
        ("Down-type quarks MSbar(M_Z)", down_mz),
        ("Up-type quarks MSbar(2 GeV)", up_2g),
        ("Down-type quarks MSbar(2 GeV)", down_2g),
    ]:
        print(f"\n  {label}")
        print(f"    Q mean   = {summary.mean:.8f}")
        print(f"    Q median = {summary.median:.8f}")
        print(f"    68% CI   = [{summary.q16:.8f}, {summary.q84:.8f}]")
        print(f"    95% CI   = [{summary.q025:.8f}, {summary.q975:.8f}]")
        print(f"    sigma    = {summary.std:.8f}")

    print("\n" + "=" * 76)
    print("TARGETED HYPOTHESIS CHECKS")
    print("=" * 76)
    report_target_test("Leptons", lep, 2/3, "2/3")
    report_target_test("Up-type quarks MSbar(M_Z)", up_mz, 8/9, "8/9")
    report_target_test("Down-type quarks MSbar(M_Z)", down_mz, 3/4, "3/4")

    print("\n" + "=" * 76)
    print("NEAREST LOW-DENOMINATOR RATIONALS")
    print("Avoids cherry-picking a single competitor such as 17/19 or 14/19.")
    print("=" * 76)
    print_nearest_rationals("Up-type quarks MSbar(M_Z)", up_mz, qmax=20)
    print_nearest_rationals("Down-type quarks MSbar(M_Z)", down_mz, qmax=20)

    print("\n" + "=" * 76)
    print("POST-HOC LOOK-ELSEWHERE CHECK")
    print("This is descriptive, not a formal p-value for a preregistered model.")
    print("=" * 76)
    up_hit = empirical_rational_hit_rate(up_mz.samples, qmax=9, window_sigma=2.0)
    down_hit = empirical_rational_hit_rate(down_mz.samples, qmax=9, window_sigma=2.0)
    print(f"  Fraction of Q values in [2/3,1] lying within 2σ of some p/q with q≤9:")
    print(f"    Up-type sigma scale:   {up_hit:.3f}")
    print(f"    Down-type sigma scale: {down_hit:.3f}")
    print("  Interpretation: small-denominator proximity is not automatically rare once one scans many rationals.")

    print("\n" + "=" * 76)
    print("SCHEME / SCALE COMPARISON")
    print("Important: this is descriptive only. It does NOT establish RG invariance.")
    print("=" * 76)
    print(f"  {'Scheme':<24} {'Q_up':>12} {'σ_up':>12} {'Q_down':>12} {'σ_down':>12}")
    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'MSbar(M_Z)':<24} {up_mz.mean:12.6f} {up_mz.std:12.6f} {down_mz.mean:12.6f} {down_mz.std:12.6f}")
    print(f"  {'MSbar(2 GeV inputs)':<24} {up_2g.mean:12.6f} {up_2g.std:12.6f} {down_2g.mean:12.6f} {down_2g.std:12.6f}")
    print(f"  {'Constituent/pole proxy':<24} {up_proxy.mean:12.6f} {up_proxy.std:12.6f} {down_proxy.mean:12.6f} {down_proxy.std:12.6f}")
    print("\n  Caveat: the last row mixes constituent-model light masses with heavy-quark pole-like inputs.")
    print("  It can illustrate scheme dependence, but should not be treated as a clean field-theoretic comparison.")

    print("\n" + "=" * 76)
    print("SENSITIVITY")
    print("=" * 76)
    print("\n  Up-type MSbar(M_Z)")
    sensitivity_scan(MSBAR_MZ, ('u', 'c', 't'))
    print("\n  Down-type MSbar(M_Z)")
    sensitivity_scan(MSBAR_MZ, ('d', 's', 'b'))

    print("\n" + "=" * 76)
    print("OPTIONAL RATIO DIAGNOSTIC")
    print("This is a derived descriptive quantity, not an independent test unless preregistered.")
    print("=" * 76)
    d_up = up_mz.samples - 2/3
    d_down = down_mz.samples - 2/3
    ratio = d_up / d_down
    ratio = ratio[np.isfinite(ratio)]
    ratio = ratio[np.abs(ratio) < 100]
    r16, r50, r84 = np.percentile(ratio, [16, 50, 84])
    r025, r975 = np.percentile(ratio, [2.5, 97.5])
    target = 8 / 3
    rmean = float(np.mean(ratio))
    rstd = float(np.std(ratio, ddof=1))
    print(f"  mean   = {rmean:.6f}")
    print(f"  median = {r50:.6f}")
    print(f"  68% CI = [{r16:.6f}, {r84:.6f}]")
    print(f"  95% CI = [{r025:.6f}, {r975:.6f}]")
    print(f"  target 8/3 = {target:.6f}")
    print(f"  |ratio - target| / σ = {abs(rmean - target) / rstd:.2f}")

    print("\n" + "=" * 76)
    print("BOTTOM LINE")
    print("=" * 76)
    print(f"  Leptons remain extremely close to 2/3: Q = {lep.mean:.8f} ± {lep.std:.8f}")
    print(f"  Up-type quarks at MSbar(M_Z):   Q = {up_mz.mean:.6f} ± {up_mz.std:.6f}")
    print(f"  Down-type quarks at MSbar(M_Z): Q = {down_mz.mean:.6f} ± {down_mz.std:.6f}")
    print("  On the supplied input dataset, 8/9 and 3/4 can be discussed as candidate targets,")
    print("  but the script now avoids stronger claims than the uncertainty model supports.")
    print("  No claim of RG invariance, UV origin, or decisive exclusion/confirmation is made here.")

    np.savez(
        'SMD5_complete_results_updated.npz',
        Q_lep=lep.mean, sig_lep=lep.std,
        Q_up_mz=up_mz.mean, sig_up_mz=up_mz.std,
        Q_down_mz=down_mz.mean, sig_down_mz=down_mz.std,
        Q_up_2g=up_2g.mean, sig_up_2g=up_2g.std,
        Q_down_2g=down_2g.mean, sig_down_2g=down_2g.std,
        Q_up_proxy=up_proxy.mean, sig_up_proxy=up_proxy.std,
        Q_down_proxy=down_proxy.mean, sig_down_proxy=down_proxy.std,
        ratio_mean=rmean, ratio_std=rstd,
        samples_up=up_mz.samples[:10000],
        samples_down=down_mz.samples[:10000],
    )
    print("\n  Saved: SMD5_complete_results_updated.npz")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].hist(up_mz.samples, bins=100, alpha=0.75)
        axes[0, 0].axvline(8/9, ls='--', lw=2, label='8/9')
        axes[0, 0].axvline(2/3, ls=':', lw=1.5, label='2/3')
        axes[0, 0].set_title('Q_up distribution (MSbar at M_Z)')
        axes[0, 0].set_xlabel('Q')
        axes[0, 0].legend()

        axes[0, 1].hist(down_mz.samples, bins=100, alpha=0.75)
        axes[0, 1].axvline(3/4, ls='--', lw=2, label='3/4')
        axes[0, 1].axvline(2/3, ls=':', lw=1.5, label='2/3')
        axes[0, 1].set_title('Q_down distribution (MSbar at M_Z)')
        axes[0, 1].set_xlabel('Q')
        axes[0, 1].legend()

        schemes = ['MSbar(M_Z)', 'MSbar(2GeV)', 'Proxy']
        axes[1, 0].errorbar(range(3), [up_mz.mean, up_2g.mean, up_proxy.mean],
                            yerr=[up_mz.std, up_2g.std, up_proxy.std], fmt='o', capsize=5)
        axes[1, 0].axhline(8/9, ls='--', label='8/9')
        axes[1, 0].axhline(2/3, ls=':', label='2/3')
        axes[1, 0].set_xticks(range(3))
        axes[1, 0].set_xticklabels(schemes)
        axes[1, 0].set_ylabel('Q_up')
        axes[1, 0].set_title('Up-type Koide vs input scheme')
        axes[1, 0].legend()

        axes[1, 1].errorbar(range(3), [down_mz.mean, down_2g.mean, down_proxy.mean],
                            yerr=[down_mz.std, down_2g.std, down_proxy.std], fmt='o', capsize=5)
        axes[1, 1].axhline(3/4, ls='--', label='3/4')
        axes[1, 1].axhline(2/3, ls=':', label='2/3')
        axes[1, 1].set_xticks(range(3))
        axes[1, 1].set_xticklabels(schemes)
        axes[1, 1].set_ylabel('Q_down')
        axes[1, 1].set_title('Down-type Koide vs input scheme')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('SMD5_complete_test_updated.png', dpi=150)
        print("  Plot: SMD5_complete_test_updated.png")
    except ImportError:
        print("  matplotlib not available — skipping plot")


if __name__ == '__main__':
    main()
