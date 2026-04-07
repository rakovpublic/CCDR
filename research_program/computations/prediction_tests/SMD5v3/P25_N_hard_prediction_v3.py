#!/usr/bin/env python3
"""
P25_N_hard_prediction.py

Rewrites the old "three independent routes give N=6" script into a harder,
more falsifiable CCDR module.

Main changes relative to the earlier version:
- treats N as a DISCRETE HYPOTHESIS to be tested, not something "derived"
  independently three times
- uses a FORWARD prediction for Omega_DM / Omega_B
- separates PREDICTIONS from mere CONSISTENCY / COMPATIBILITY checks
- adds explicit falsification criteria
- compares nearby integer hypotheses N in {4,5,6,7,8}
- avoids claiming that DA and HaPPY independently prove N=6

Default observed ratio is kept at 5.36 to match the prior script.
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class HypothesisResult:
    N: int
    visible: int
    dark: int
    central_ratio: float
    delta_tol: float
    band_low: float
    band_high: float
    z_score: float
    gaussian_weight: float
    in_band: bool
    verdict: str


def gaussian_weight(obs: float, pred: float, sigma_eff: float) -> float:
    if sigma_eff <= 0:
        return 0.0
    return math.exp(-0.5 * ((obs - pred) / sigma_eff) ** 2)


def evaluate_hypothesis(N: int, obs_ratio: float, delta_tol: float) -> HypothesisResult:
    central = float(N - 1)
    sigma_eff = delta_tol * central
    low = central * (1.0 - delta_tol)
    high = central * (1.0 + delta_tol)
    z = (obs_ratio - central) / sigma_eff if sigma_eff > 0 else float("inf")
    weight = gaussian_weight(obs_ratio, central, sigma_eff)
    in_band = (low <= obs_ratio <= high)

    if abs(z) <= 1.0:
        verdict = "strongly compatible"
    elif abs(z) <= 2.0:
        verdict = "weak / borderline"
    else:
        verdict = "disfavoured"

    return HypothesisResult(
        N=N,
        visible=1,
        dark=N - 1,
        central_ratio=central,
        delta_tol=delta_tol,
        band_low=low,
        band_high=high,
        z_score=z,
        gaussian_weight=weight,
        in_band=in_band,
        verdict=verdict,
    )


def normalize_weights(results: List[HypothesisResult]) -> List[Tuple[int, float]]:
    total = sum(r.gaussian_weight for r in results)
    if total <= 0:
        return [(r.N, 0.0) for r in results]
    return [(r.N, r.gaussian_weight / total) for r in results]


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P25 hard-prediction / discrete model-selection script for CCDR."
    )
    parser.add_argument(
        "--omega-ratio",
        type=float,
        default=5.36,
        help="Observed Omega_DM / Omega_B ratio to test (default: 5.36).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.10,
        help="Pre-registered fractional tolerance Delta for corrections (default: 0.10).",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="*",
        default=[4, 5, 6, 7, 8],
        help="Discrete sector hypotheses N to compare (default: 4 5 6 7 8).",
    )
    args = parser.parse_args()

    obs_ratio = float(args.omega_ratio)
    delta_tol = float(args.delta)
    n_values = sorted(set(args.n_values))

    print("P25 (HARD VERSION): BULK SECTOR COUNT N AS A TESTABLE DISCRETE HYPOTHESIS")
    print("=" * 78)
    print("CORE POSTULATE:")
    print("  The cosmological crystal has:")
    print("    - N total internal sectors")
    print("    - 1 visible (acoustic) sector")
    print("    - N-1 dark (optical) sectors")
    print()
    print("IMPORTANT:")
    print("  This script does NOT claim that N=6 is independently derived")
    print("  three times. Instead, it tests discrete hypotheses N against a")
    print("  forward prediction for Omega_DM / Omega_B and then reports")
    print("  algebraic / holographic compatibility checks separately.")
    print()
    print(f"Observed input ratio: Omega_DM / Omega_B = {obs_ratio:.4f}")
    print(f"Pre-registered correction tolerance: |delta| <= Delta = {delta_tol:.3f}")

    print_header("1. PRIMARY FORWARD PREDICTION")
    print("CCDR forward prediction:")
    print("  Omega_DM / Omega_B = (N - 1) * (1 + delta)")
    print("where delta represents finite corrections from:")
    print("  - defect geometry")
    print("  - crystallisation dynamics")
    print("  - freeze-out / sequestration asymmetry")
    print()
    print("This is a MODEL-SELECTION test over nearby integer hypotheses,")
    print("not a nearest-integer rounding exercise.")

    results = [evaluate_hypothesis(N, obs_ratio, delta_tol) for N in n_values]

    print("\nPredicted bands:")
    print(f"{'N':>3} {'visible':>8} {'dark':>6} {'central':>10} {'band':>23} {'z':>9} {'status':>22}")
    print("-" * 90)
    for r in results:
        band = f"[{r.band_low:.2f}, {r.band_high:.2f}]"
        print(f"{r.N:>3} {r.visible:>8} {r.dark:>6} {r.central_ratio:>10.2f} {band:>23} {r.z_score:>9.2f} {r.verdict:>22}")

    probs = normalize_weights(results)
    ranking = {N: p for N, p in probs}

    print("\nApproximate relative support (Gaussian score with sigma_eff = Delta*(N-1)):")
    for r in sorted(results, key=lambda x: x.gaussian_weight, reverse=True):
        print(f"  H_{r.N}: weight={r.gaussian_weight:.6f}, normalized≈{ranking[r.N]:.3f}")

    best = max(results, key=lambda x: x.gaussian_weight)
    print("\nBest-supported discrete hypothesis under the current tolerance model:")
    print(f"  N = {best.N}  ->  visible={best.visible}, dark={best.dark},")
    print(f"             predicted central ratio={best.central_ratio:.2f},")
    print(f"             predicted band=[{best.band_low:.2f}, {best.band_high:.2f}]")
    print(f"  Verdict: {best.verdict}")

    print_header("2. SECONDARY STRUCTURAL PREDICTION")
    print("If CCDR has sector count N, then:")
    print("  N_dark = N - 1")
    print()
    print("For the best-supported current hypothesis:")
    print(f"  N = {best.N}  ->  N_dark = {best.dark}")
    print()
    print("Interpretation:")
    print("  CCDR predicts a DISCRETE multiplicity of dark sectors, not an")
    print("  arbitrary continuum. This remains only weakly testable at present,")
    print("  but it is a logical consequence of the hypothesis.")

    print_header("3. CONSISTENCY CHECKS (NOT INDEPENDENT DERIVATIONS)")
    print("(a) Division algebra compatibility")
    N_DA = 1 + 2 + 4 - 1
    print("  CCDR counting convention:")
    print("    dim(R) + dim(C) + dim(H) - 1(phase) = 1 + 2 + 4 - 1 = 6")
    print(f"  N_DA = {N_DA}")
    if best.N == N_DA:
        print("  [COMPATIBLE] The current best-supported N matches the DA convention.")
    else:
        print("  [NOT MATCHED] The current best-supported N does not match the DA convention.")
    print("  Note: this is convention-dependent and does NOT independently prove N=6.")
    print()
    print("(b) HaPPY / holographic-QEC compatibility")
    N_HAPPY = 6
    print("  Minimal redundancy toy-code arguments can be arranged to allow N=6.")
    print(f"  N_HaPPY (compatibility marker) = {N_HAPPY}")
    if best.N == N_HAPPY:
        print("  [COMPATIBLE] The current best-supported N lies in the quoted viable regime.")
    else:
        print("  [NOT MATCHED] The current best-supported N is not the quoted HaPPY-compatible value.")
    print("  Note: this is a consistency check only; it depends on code choice and is not predictive.")

    print_header("4. HONEST SUMMARY")
    print("What this script DOES claim:")
    print("  - There is a forward prediction: Omega_DM / Omega_B ≈ N - 1, up to")
    print("    a pre-registered correction tolerance Delta.")
    print("  - Nearby discrete hypotheses N can be compared explicitly.")
    print("  - Under the default observed ratio and Delta, one can ask whether")
    print("    N=6 is preferred over N=5 or N=7.")
    print()
    print("What this script DOES NOT claim:")
    print("  - DA and HaPPY independently derive N=6.")
    print("  - The current result is decisive by itself.")
    print("  - Sector multiplicity is already observationally established.")

    print_header("5. FALSIFICATION CRITERIA")
    print("P25 should be treated as falsified or seriously weakened if:")
    print("  (F1) future precision cosmology robustly pushes Omega_DM / Omega_B")
    print(f"       outside the pre-registered N=6 band [{6*(1-delta_tol)-1:.2f}, {6*(1+delta_tol)-1:.2f}]?  No.")
    # Correct the specific N=6 band display cleanly
    n6_low = (6 - 1) * (1 - delta_tol)
    n6_high = (6 - 1) * (1 + delta_tol)
    print(f"       More precisely, for N=6: [{n6_low:.2f}, {n6_high:.2f}]")
    print("  (F2) a neighbouring discrete hypothesis (especially N=5 or N=7)")
    print("       becomes robustly preferred under the same pre-registered tolerance.")
    print("  (F3) any future dark-sector evidence strongly favours a single smooth")
    print("       component with no meaningful discrete multiplicity.")
    print("  (F4) the correction term delta must be tuned beyond ~20-30% just to")
    print("       keep N=6 viable.")

    print_header("6. SAFE CLAIM")
    print(f"Under the input ratio Omega_DM/Omega_B = {obs_ratio:.4f} and")
    print(f"pre-registered tolerance Delta = {delta_tol:.3f}, the script currently")
    print(f"finds that N = {best.N} is the best-supported discrete sector-count")
    print("hypothesis in the tested set, while DA and HaPPY remain compatibility")
    print("checks rather than independent proofs.")

    print_header("7. NEXT STEP")
    print("To make P25 genuinely stronger, CCDR still needs at least one of:")
    print("  - a first-principles derivation of delta")
    print("  - a second observable that depends on N")
    print("  - a way to distinguish N=6 from nearby N without relying on a single")
    print("    cosmological ratio alone")


if __name__ == "__main__":
    main()
