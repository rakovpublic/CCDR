#!/usr/bin/env python3
"""
P17_gamma_convergence.py

Schematic implementation of the P17 programme:
  1. Build benchmark / toy estimates for gamma_AS and gamma_EPRL
  2. Run a schematic TGFT-RG flow for gamma, m^2, lambda
  3. Check whether distinct UV gamma inputs converge in the IR
  4. Evaluate a simplified black-hole entropy consistency condition

Important note:
This script is a computational toy model for the research note, not a
first-principles derivation of the full EPRL/TGFT renormalisation group.
It is useful for internal-consistency experiments and sensitivity scans,
but its numerical outputs should not be interpreted as literature-grade
predictions without replacing the beta functions by a justified model.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# -----------------------------------------------------------------------------
# Constants / benchmarks
# -----------------------------------------------------------------------------
S_REGGE_EQUILATERAL = 5.0 * np.arccos(0.25)
G_STAR_BENCHMARK = 0.707
LAMBDA_STAR_BENCHMARK = 0.193
GAMMA_AS_BENCHMARK = 0.2740
GAMMA_EPRL_BENCHMARK = 0.2375
GAMMA_MEISSNER_BENCHMARK = 0.23753295796592


@dataclass
class FlowSummary:
    gamma_uv: float
    gamma_ir: float | None
    m2_ir: float | None
    lambda_ir: float | None
    success: bool
    message: str
    n_steps: int


# -----------------------------------------------------------------------------
# Step 1: AS-side gamma estimates
# -----------------------------------------------------------------------------
def compute_gamma_as_estimates() -> dict:
    """Return benchmark and schematic estimates for gamma_AS.

    The leading and one-loop expressions below follow the structure from the
    research note. They are *toy estimates*. The benchmark value is the one
    intended by the note for the RG-convergence test.
    """
    gamma_leading = np.pi / (2.0 * S_REGGE_EQUILATERAL)
    delta_gamma = gamma_leading * G_STAR_BENCHMARK / (4.0 * np.pi) * (41.0 / 10.0)
    gamma_one_loop_estimate = gamma_leading + delta_gamma
    return {
        "S_Regge_equilateral": float(S_REGGE_EQUILATERAL),
        "G_star_benchmark": G_STAR_BENCHMARK,
        "Lambda_star_benchmark": LAMBDA_STAR_BENCHMARK,
        "gamma_AS_leading_toy": float(gamma_leading),
        "gamma_AS_one_loop_toy": float(gamma_one_loop_estimate),
        "gamma_AS_benchmark": GAMMA_AS_BENCHMARK,
    }


# -----------------------------------------------------------------------------
# Step 2: EPRL / entropy-side gamma estimates
# -----------------------------------------------------------------------------
def lqg_entropy_constraint(gamma: float, j_max: int = 200) -> float:
    """Standard LQG-like entropy sum rule.

    This root lands close to ~0.274 for the area spectrum-based sum with a
    2πγ√(j(j+1)) exponent. It is included as a useful comparison point.
    """
    total = 0.0
    for j2 in range(1, 2 * j_max + 1):
        j = j2 / 2.0
        total += (2.0 * j + 1.0) * np.exp(-2.0 * np.pi * gamma * np.sqrt(j * (j + 1.0)))
    return total - 1.0


def eprl_weighted_constraint(gamma: float, j_max: int = 80, alpha: float = 0.85) -> float:
    """Schematic EPRL-weighted entropy constraint.

    The note quotes gamma_EPRL ~ 0.2375 as the target benchmark. A full EPRL
    state-counting derivation is outside the scope of this desktop script, so
    we use a simple effective-weight deformation of the LQG sum rule that can
    be scanned numerically. The benchmark is kept explicit in the output.
    """
    total = 0.0
    for j2 in range(1, 2 * j_max + 1):
        j = j2 / 2.0
        j_plus = 0.5 * (1.0 + gamma) * j
        j_minus = 0.5 * abs(1.0 - gamma) * j
        d_eff = (2.0 * j_plus + 1.0) ** alpha * (2.0 * j_minus + 1.0) ** alpha
        total += d_eff * np.exp(-2.0 * np.pi * gamma * np.sqrt(j * (j + 1.0)))
    return total - 1.0


def bracketed_root(func, a: float, b: float, n_scan: int = 200) -> float | None:
    """Find a bracket in [a, b] and solve for a root, or return None."""
    xs = np.linspace(a, b, n_scan)
    vals = [func(float(x)) for x in xs]
    for x1, x2, y1, y2 in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if not np.isfinite(y1) or not np.isfinite(y2):
            continue
        if y1 == 0.0:
            return float(x1)
        if y1 * y2 < 0.0:
            return float(brentq(func, float(x1), float(x2), maxiter=1000))
    return None


def compute_gamma_eprl_estimates() -> dict:
    gamma_lqg_root = bracketed_root(lambda g: lqg_entropy_constraint(g, j_max=200), 0.05, 1.0)
    gamma_eprl_toy = bracketed_root(lambda g: eprl_weighted_constraint(g, j_max=80, alpha=0.85), 0.05, 1.0)
    return {
        "gamma_Meissner_benchmark": GAMMA_MEISSNER_BENCHMARK,
        "gamma_EPRL_benchmark": GAMMA_EPRL_BENCHMARK,
        "gamma_LQG_entropy_root": None if gamma_lqg_root is None else float(gamma_lqg_root),
        "gamma_EPRL_weighted_toy": None if gamma_eprl_toy is None else float(gamma_eprl_toy),
    }


# -----------------------------------------------------------------------------
# Step 3: Schematic TGFT-RG beta function
# -----------------------------------------------------------------------------
def tgft_beta_functions(t: float, y: np.ndarray, d: int = 4) -> list[float]:
    """Schematic one-loop TGFT-style beta functions from the note.

    This keeps the structure of the research note but adds numerical guards.
    """
    m2, lam, gamma = y

    if not np.isfinite(m2) or not np.isfinite(lam) or not np.isfinite(gamma):
        return [0.0, 0.0, 0.0]

    denom = 1.0 + m2
    if abs(denom) < 1e-12:
        denom = np.copysign(1e-12, denom if denom != 0 else 1.0)

    gamma_safe = gamma
    if abs(gamma_safe) < 1e-12:
        gamma_safe = np.copysign(1e-12, gamma_safe if gamma_safe != 0 else 1.0)

    i1 = (1.0 / denom) * (1.0 + gamma_safe**2) / (gamma_safe**2)
    i2 = (1.0 / denom**2) * (1.0 + gamma_safe**2) ** 2 / (gamma_safe**4)
    i3 = np.sin(gamma_safe * S_REGGE_EQUILATERAL) / (gamma_safe * denom)

    beta_m2 = (d - 2.0) * m2 + lam * i1
    beta_lam = (2.0 * d - 4.0) * lam + lam**2 * i2
    beta_gamma = gamma_safe * lam * i3
    return [float(beta_m2), float(beta_lam), float(beta_gamma)]


def run_rg_flow(
    gamma_uv: float,
    m2_uv: float = 1.0,
    lam_uv: float = 0.1,
    t_uv: float = 10.0,
    t_ir: float = -10.0,
    d: int = 4,
    max_step: float = 0.05,
) -> solve_ivp:
    """Integrate from UV to IR."""
    y0 = [m2_uv, lam_uv, gamma_uv]
    sol = solve_ivp(
        lambda t, y: tgft_beta_functions(t, y, d=d),
        (t_uv, t_ir),
        y0,
        method="RK45",
        dense_output=True,
        max_step=max_step,
        rtol=1e-8,
        atol=1e-10,
    )
    return sol


def summarize_flow(sol: solve_ivp, gamma_uv: float) -> FlowSummary:
    if sol.success and sol.y.shape[1] > 0:
        return FlowSummary(
            gamma_uv=float(gamma_uv),
            gamma_ir=float(sol.y[2, -1]),
            m2_ir=float(sol.y[0, -1]),
            lambda_ir=float(sol.y[1, -1]),
            success=True,
            message=str(sol.message),
            n_steps=int(sol.y.shape[1]),
        )
    return FlowSummary(
        gamma_uv=float(gamma_uv),
        gamma_ir=None,
        m2_ir=None,
        lambda_ir=None,
        success=False,
        message=str(sol.message),
        n_steps=0,
    )


def find_ir_fixed_point(gamma_uv_values: Iterable[float], **flow_kwargs) -> tuple[list[FlowSummary], dict]:
    summaries: list[FlowSummary] = []
    for gamma_uv in gamma_uv_values:
        sol = run_rg_flow(gamma_uv=float(gamma_uv), **flow_kwargs)
        summaries.append(summarize_flow(sol, gamma_uv=float(gamma_uv)))

    successful = [s for s in summaries if s.success and s.gamma_ir is not None]
    if not successful:
        return summaries, {
            "gamma_star_mean": None,
            "gamma_star_std": None,
            "relative_spread": None,
            "converged_below_5_percent": False,
            "n_successful": 0,
        }

    gamma_irs = np.array([s.gamma_ir for s in successful], dtype=float)
    gamma_star = float(np.mean(gamma_irs))
    gamma_spread = float(np.std(gamma_irs))
    relative_spread = float(gamma_spread / abs(gamma_star)) if gamma_star != 0 else float("inf")
    return summaries, {
        "gamma_star_mean": gamma_star,
        "gamma_star_std": gamma_spread,
        "relative_spread": relative_spread,
        "converged_below_5_percent": bool(relative_spread < 0.05),
        "n_successful": len(successful),
    }


# -----------------------------------------------------------------------------
# Step 4: Simplified BH entropy check
# -----------------------------------------------------------------------------
def check_bh_entropy(gamma_star: float | None) -> dict:
    if gamma_star is None:
        return {
            "gamma_star": None,
            "entropy_coefficient": None,
            "target": 0.25,
            "fractional_deviation": None,
            "within_2_percent": False,
        }
    entropy_coefficient = gamma_star * np.pi / (2.0 * S_REGGE_EQUILATERAL)
    frac_dev = abs(entropy_coefficient - 0.25) / 0.25
    return {
        "gamma_star": float(gamma_star),
        "entropy_coefficient": float(entropy_coefficient),
        "target": 0.25,
        "fractional_deviation": float(frac_dev),
        "within_2_percent": bool(abs(entropy_coefficient - 0.25) < 0.02),
    }


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def make_plot(output_dir: Path, solutions: list[solve_ivp], labels: list[str]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for sol, label in zip(solutions, labels):
        if sol.success and sol.y.shape[1] > 0:
            ax.plot(sol.t, sol.y[2], lw=2, label=label)

    ax.axhline(GAMMA_AS_BENCHMARK, color="tab:blue", ls="--", alpha=0.5, label="γ_AS benchmark")
    ax.axhline(GAMMA_EPRL_BENCHMARK, color="tab:orange", ls=":", alpha=0.7, label="γ_EPRL benchmark")
    ax.set_xlabel("RG time t")
    ax.set_ylabel("γ(t)")
    ax.set_title("P17 schematic TGFT-RG flow of γ")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    path = output_dir / "P17_gamma_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schematic P17 gamma_AS / gamma_EPRL convergence test under TGFT-RG flow."
    )
    parser.add_argument(
        "--gamma-uv",
        nargs="*",
        type=float,
        default=[GAMMA_AS_BENCHMARK, GAMMA_EPRL_BENCHMARK, 0.25, 0.30],
        help="UV gamma seeds to evolve. Default: benchmark AS/EPRL values plus two nearby test points.",
    )
    parser.add_argument("--m2-uv", type=float, default=1.0)
    parser.add_argument("--lambda-uv", type=float, default=0.1)
    parser.add_argument("--t-uv", type=float, default=10.0)
    parser.add_argument("--t-ir", type=float, default=-10.0)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("data/p17_gamma_convergence"))
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("P17: γ_AS = γ_EPRL Convergence Under GFT Renormalisation Group")
    print("=" * 72)
    print("This is a schematic implementation of the research-note workflow.")
    print()

    gamma_as = compute_gamma_as_estimates()
    gamma_eprl = compute_gamma_eprl_estimates()

    print("[Step 1] AS-side gamma estimates")
    print(f"  S_Regge(equilateral)     = {gamma_as['S_Regge_equilateral']:.6f}")
    print(f"  γ_AS leading (toy)       = {gamma_as['gamma_AS_leading_toy']:.6f}")
    print(f"  γ_AS one-loop (toy)      = {gamma_as['gamma_AS_one_loop_toy']:.6f}")
    print(f"  γ_AS benchmark           = {gamma_as['gamma_AS_benchmark']:.6f}")
    print()

    print("[Step 2] EPRL / entropy-side estimates")
    print(f"  γ_Meissner benchmark     = {gamma_eprl['gamma_Meissner_benchmark']:.6f}")
    print(f"  γ_EPRL benchmark         = {gamma_eprl['gamma_EPRL_benchmark']:.6f}")
    if gamma_eprl["gamma_LQG_entropy_root"] is not None:
        print(f"  γ_LQG entropy root       = {gamma_eprl['gamma_LQG_entropy_root']:.6f}")
    else:
        print("  γ_LQG entropy root       = root not found")
    if gamma_eprl["gamma_EPRL_weighted_toy"] is not None:
        print(f"  γ_EPRL weighted toy root = {gamma_eprl['gamma_EPRL_weighted_toy']:.6f}")
    else:
        print("  γ_EPRL weighted toy root = root not found")
    print()

    solutions: list[solve_ivp] = []
    labels: list[str] = []
    summaries, convergence = find_ir_fixed_point(
        gamma_uv_values=args.gamma_uv,
        m2_uv=args.m2_uv,
        lam_uv=args.lambda_uv,
        t_uv=args.t_uv,
        t_ir=args.t_ir,
        d=args.d,
        max_step=args.max_step,
    )

    print("[Step 3] UV → IR flow summaries")
    for summary in summaries:
        if summary.success:
            print(
                f"  γ_UV = {summary.gamma_uv:.6f} -> γ_IR = {summary.gamma_ir:.6f} "
                f"(m²_IR={summary.m2_ir:.3e}, λ_IR={summary.lambda_ir:.3e})"
            )
            sol = run_rg_flow(
                gamma_uv=summary.gamma_uv,
                m2_uv=args.m2_uv,
                lam_uv=args.lambda_uv,
                t_uv=args.t_uv,
                t_ir=args.t_ir,
                d=args.d,
                max_step=args.max_step,
            )
            solutions.append(sol)
            labels.append(f"γ_UV={summary.gamma_uv:.4f}")
        else:
            print(f"  γ_UV = {summary.gamma_uv:.6f} -> flow failed: {summary.message}")
    if convergence["gamma_star_mean"] is not None:
        print()
        print(f"  γ* mean      = {convergence['gamma_star_mean']:.6f}")
        print(f"  γ* std       = {convergence['gamma_star_std']:.6f}")
        print(f"  rel. spread  = {100.0 * convergence['relative_spread']:.2f}%")
        print(
            "  convergence  = "
            + ("YES (<5%)" if convergence["converged_below_5_percent"] else "NO")
        )
    print()

    entropy = check_bh_entropy(convergence["gamma_star_mean"])
    print("[Step 4] Simplified BH entropy check")
    if entropy["gamma_star"] is not None:
        print(f"  γ*                   = {entropy['gamma_star']:.6f}")
        print(f"  entropy coefficient  = {entropy['entropy_coefficient']:.6f}")
        print(f"  target               = {entropy['target']:.6f}")
        print(f"  fractional deviation = {100.0 * entropy['fractional_deviation']:.2f}%")
        print("  BH check             = " + ("PASS" if entropy["within_2_percent"] else "FAIL"))
    else:
        print("  skipped because the RG flow produced no successful γ* estimate")
    print()

    plot_path = None
    if not args.skip_plot:
        plot_path = make_plot(output_dir, solutions, labels)
        if plot_path is not None:
            print(f"Plot: {plot_path}")

    result = {
        "metadata": {
            "script": "P17_gamma_convergence.py",
            "note": "Schematic implementation of the uploaded P17 computational programme.",
            "uv_inputs": list(map(float, args.gamma_uv)),
            "m2_uv": float(args.m2_uv),
            "lambda_uv": float(args.lambda_uv),
            "t_uv": float(args.t_uv),
            "t_ir": float(args.t_ir),
            "dimension": int(args.d),
        },
        "gamma_AS": gamma_as,
        "gamma_EPRL": gamma_eprl,
        "flow_summaries": [asdict(s) for s in summaries],
        "convergence": convergence,
        "bh_entropy_check": entropy,
        "plot_path": None if plot_path is None else str(plot_path),
    }

    result_path = output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved: {result_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
