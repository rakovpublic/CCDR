#!/usr/bin/env python3
"""
P25 v4: Independent prediction test for CCDR bulk sector count N.

Purpose
-------
Turn P25 into a harder, more falsifiable test by combining:
  1) primary cosmological density-ratio observable:
       Omega_DM / Omega_B = (N - 1) * (1 + delta_total)
  2) optional second observable:
       f_int = k / (N - 1),   k integer in [0, N-1]
     where f_int is the interacting/acoustic fraction of the dark sector.

This script does NOT pretend to derive delta_total from first principles.
Instead, it gives you a place to insert a microphysical decomposition:
    delta_total = (1+delta_geom)(1+delta_seq)(1+delta_fo)(1+delta_mass) - 1

If you can justify those pieces from CCDR microphysics, P25 becomes much stronger.

Use
---
Default quick test:
    python3 P25_v4_independent_prediction_test.py

With custom data:
    python3 P25_v4_independent_prediction_test.py \
        --omega-ratio 5.364 --omega-sigma 0.11 \
        --f-int 0.21 --f-int-sigma 0.04 \
        --delta-geom 0.01 --delta-seq 0.02 --delta-fo -0.01 --delta-mass 0.00

Interpretation
--------------
- If only omega-ratio is supplied, this is still a one-observable test.
- If f_int is also supplied from an independent public analysis, the script performs
  a genuine two-observable ranking over (N, k).
- The script reports whether N=6 is preferred over nearby N values and whether the
  inference is robust or still ambiguous.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict


@dataclass
class Inputs:
    omega_ratio: float = 5.364
    omega_sigma: float = 0.11  # observational uncertainty only; not theory tolerance
    # Optional second observable
    f_int: Optional[float] = None
    f_int_sigma: Optional[float] = None
    # Microphysical decomposition of delta_total
    delta_geom: float = 0.0
    delta_seq: float = 0.0
    delta_fo: float = 0.0
    delta_mass: float = 0.0
    # Extra theory uncertainty if first-principles delta is not trusted yet
    theory_sigma_frac: float = 0.05
    n_min: int = 4
    n_max: int = 8


@dataclass
class HypothesisResult:
    N: int
    k_int: Optional[int]
    omega_pred: float
    omega_sigma_eff: float
    omega_pull: float
    f_int_pred: Optional[float]
    f_int_pull: Optional[float]
    chi2_total: float
    rel_weight: float


def delta_total(inp: Inputs) -> float:
    """Compose delta multiplicatively to avoid linearization artifacts."""
    return (
        (1.0 + inp.delta_geom)
        * (1.0 + inp.delta_seq)
        * (1.0 + inp.delta_fo)
        * (1.0 + inp.delta_mass)
        - 1.0
    )


def omega_prediction(N: int, inp: Inputs) -> Tuple[float, float]:
    """
    Predict Omega_DM/Omega_B for a given N.

    Effective uncertainty combines:
      - observational uncertainty (inp.omega_sigma)
      - residual theory uncertainty from incomplete first-principles control
        represented as a fractional uncertainty on the predicted ratio
    """
    dtot = delta_total(inp)
    pred = (N - 1) * (1.0 + dtot)
    theory_sigma = abs(pred) * inp.theory_sigma_frac
    sigma_eff = math.sqrt(inp.omega_sigma ** 2 + theory_sigma ** 2)
    return pred, sigma_eff


def f_int_candidates(N: int) -> List[Tuple[int, float]]:
    """Allowed discrete interacting fractions f_int = k/(N-1)."""
    denom = N - 1
    return [(k, k / denom) for k in range(0, denom + 1)]


def gaussian_chi2(obs: float, pred: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return ((obs - pred) / sigma) ** 2


def evaluate_hypotheses(inp: Inputs) -> List[HypothesisResult]:
    results: List[HypothesisResult] = []

    for N in range(inp.n_min, inp.n_max + 1):
        omega_pred, omega_sigma_eff = omega_prediction(N, inp)
        omega_chi2 = gaussian_chi2(inp.omega_ratio, omega_pred, omega_sigma_eff)
        omega_pull = (inp.omega_ratio - omega_pred) / omega_sigma_eff

        if inp.f_int is None:
            results.append(
                HypothesisResult(
                    N=N,
                    k_int=None,
                    omega_pred=omega_pred,
                    omega_sigma_eff=omega_sigma_eff,
                    omega_pull=omega_pull,
                    f_int_pred=None,
                    f_int_pull=None,
                    chi2_total=omega_chi2,
                    rel_weight=0.0,
                )
            )
        else:
            if inp.f_int_sigma is None or inp.f_int_sigma <= 0:
                raise ValueError("When --f-int is supplied, --f-int-sigma must be positive.")
            for k, fint_pred in f_int_candidates(N):
                fint_chi2 = gaussian_chi2(inp.f_int, fint_pred, inp.f_int_sigma)
                fint_pull = (inp.f_int - fint_pred) / inp.f_int_sigma
                results.append(
                    HypothesisResult(
                        N=N,
                        k_int=k,
                        omega_pred=omega_pred,
                        omega_sigma_eff=omega_sigma_eff,
                        omega_pull=omega_pull,
                        f_int_pred=fint_pred,
                        f_int_pull=fint_pull,
                        chi2_total=omega_chi2 + fint_chi2,
                        rel_weight=0.0,
                    )
                )

    # Relative weights from chi2 differences
    if not results:
        return results
    chi2_min = min(r.chi2_total for r in results)
    weights = [math.exp(-0.5 * (r.chi2_total - chi2_min)) for r in results]
    norm = sum(weights)
    for r, w in zip(results, weights):
        r.rel_weight = w / norm if norm > 0 else 0.0

    results.sort(key=lambda r: r.chi2_total)
    return results


def aggregate_by_N(results: List[HypothesisResult]) -> List[Dict[str, float]]:
    """Sum relative weights over k_int to get per-N support."""
    byN: Dict[int, float] = {}
    best_byN: Dict[int, float] = {}
    for r in results:
        byN[r.N] = byN.get(r.N, 0.0) + r.rel_weight
        best_byN[r.N] = min(best_byN.get(r.N, float("inf")), r.chi2_total)
    rows = [{"N": N, "support": byN[N], "best_chi2": best_byN[N]} for N in sorted(byN)]
    rows.sort(key=lambda x: x["best_chi2"])
    return rows


def strength_label(best_support: float, second_support: float) -> str:
    if best_support >= 0.85 and (best_support / max(second_support, 1e-12)) >= 3.0:
        return "strong"
    if best_support >= 0.65 and (best_support / max(second_support, 1e-12)) >= 1.5:
        return "moderate"
    return "weak/ambiguous"


def print_header(inp: Inputs) -> None:
    print("P25 v4 — INDEPENDENT PREDICTION TEST")
    print("=" * 72)
    print("Core model:")
    print("  Omega_DM / Omega_B = (N - 1) * (1 + delta_total)")
    print("  delta_total = (1+delta_geom)(1+delta_seq)(1+delta_fo)(1+delta_mass) - 1")
    print()
    print(f"Inputs:")
    print(f"  Observed Omega_DM / Omega_B = {inp.omega_ratio:.4f} ± {inp.omega_sigma:.4f}")
    print(f"  delta_geom = {inp.delta_geom:+.4f}")
    print(f"  delta_seq  = {inp.delta_seq:+.4f}")
    print(f"  delta_fo   = {inp.delta_fo:+.4f}")
    print(f"  delta_mass = {inp.delta_mass:+.4f}")
    print(f"  => delta_total = {delta_total(inp):+.4f}")
    print(f"  Residual theory sigma (fractional) = {inp.theory_sigma_frac:.3f}")
    if inp.f_int is not None:
        print(f"  Independent observable: f_int = {inp.f_int:.4f} ± {inp.f_int_sigma:.4f}")
        print("  Allowed model values: f_int = k / (N - 1), k integer")
    else:
        print("  No second observable supplied: this remains a 1-observable test.")
    print("=" * 72)
    print()


def print_top_results(results: List[HypothesisResult], top_n: int = 10) -> None:
    print("TOP HYPOTHESES")
    print("-" * 72)
    for i, r in enumerate(results[:top_n], start=1):
        fint_str = (
            f"k={r.k_int}, f_int_pred={r.f_int_pred:.4f}, f_pull={r.f_int_pull:+.2f}"
            if r.f_int_pred is not None
            else "no f_int test"
        )
        print(
            f"{i:2d}. N={r.N}, omega_pred={r.omega_pred:.4f}, "
            f"omega_pull={r.omega_pull:+.2f}, {fint_str}, "
            f"chi2={r.chi2_total:.3f}, weight={r.rel_weight:.3f}"
        )
    print()


def print_per_N_summary(perN: List[Dict[str, float]]) -> None:
    print("AGGREGATED SUPPORT BY N")
    print("-" * 72)
    for row in perN:
        print(f"N={row['N']}: support={row['support']:.3f}, best_chi2={row['best_chi2']:.3f}")
    print()


def print_conclusion(inp: Inputs, perN: List[Dict[str, float]]) -> None:
    best = perN[0]
    second = perN[1] if len(perN) > 1 else {"support": 0.0}
    label = strength_label(best["support"], second["support"])

    print("CONCLUSION")
    print("-" * 72)
    print(
        f"Best-supported sector count: N={best['N']} "
        f"(support={best['support']:.3f}, strength={label})"
    )

    if inp.f_int is None:
        print("Caution: only one observable was used.")
        print("This can rank N, but it does NOT yet make P25 robust against nearby N.")
    else:
        print("This uses two observables:")
        print("  (1) Omega_DM / Omega_B")
        print("  (2) an independent interacting-fraction observable f_int")
        print("That is the minimum structure needed to start distinguishing N=6 from nearby N.")

    # Specific diagnostic around N=6 vs N=7 if present
    support_map = {row["N"]: row["support"] for row in perN}
    if 6 in support_map and 7 in support_map:
        ratio = support_map[6] / max(support_map[7], 1e-12)
        print(f"Support ratio N=6 / N=7 = {ratio:.3f}")
        if ratio < 1.5:
            print("Interpretation: N=6 is NOT cleanly separated from N=7 yet.")
        elif ratio < 3.0:
            print("Interpretation: N=6 is preferred over N=7, but not decisively.")
        else:
            print("Interpretation: N=6 is cleanly preferred over N=7.")
    print()

    print("FALSIFICATION FLAGS")
    print("-" * 72)
    print("* If N=6 is not preferred once a second observable is added, P25 weakens sharply.")
    print("* If keeping N=6 requires |delta_total| or residual theory uncertainty > 10%,")
    print("  the model remains underived rather than predictive.")
    print("* If all nearby N values fit comparably well, P25 is still only a consistency story.")
    print()


def save_json(inp: Inputs, results: List[HypothesisResult], perN: List[Dict[str, float]], out: str) -> None:
    payload = {
        "inputs": asdict(inp),
        "delta_total": delta_total(inp),
        "top_results": [asdict(r) for r in results[:20]],
        "per_N_support": perN,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON summary to: {out}")


def parse_args() -> Inputs:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega-ratio", type=float, default=5.364)
    ap.add_argument("--omega-sigma", type=float, default=0.11)
    ap.add_argument("--f-int", type=float, default=None)
    ap.add_argument("--f-int-sigma", type=float, default=None)

    ap.add_argument("--delta-geom", type=float, default=0.0)
    ap.add_argument("--delta-seq", type=float, default=0.0)
    ap.add_argument("--delta-fo", type=float, default=0.0)
    ap.add_argument("--delta-mass", type=float, default=0.0)
    ap.add_argument("--theory-sigma-frac", type=float, default=0.05)

    ap.add_argument("--n-min", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--json-out", type=str, default="p25_v4_results.json")
    ns = ap.parse_args()

    return Inputs(
        omega_ratio=ns.omega_ratio,
        omega_sigma=ns.omega_sigma,
        f_int=ns.f_int,
        f_int_sigma=ns.f_int_sigma,
        delta_geom=ns.delta_geom,
        delta_seq=ns.delta_seq,
        delta_fo=ns.delta_fo,
        delta_mass=ns.delta_mass,
        theory_sigma_frac=ns.theory_sigma_frac,
        n_min=ns.n_min,
        n_max=ns.n_max,
    )


def main() -> None:
    inp = parse_args()
    print_header(inp)
    results = evaluate_hypotheses(inp)
    perN = aggregate_by_N(results)
    print_top_results(results)
    print_per_N_summary(perN)
    print_conclusion(inp, perN)
    save_json(inp, results, perN, out="p25_v4_results.json")


if __name__ == "__main__":
    main()
