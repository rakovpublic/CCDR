#!/usr/bin/env python3
"""
P18_f_geo_from_EPRL.py

Compute the geometric baryogenesis factor f_geo from the EPRL chiral amplitude.

This script implements the analytic pieces from the P18 research note and adds:
- a small CLI
- JSON output
- a plot of f_geo sensitivity
- optional sl2cfoam-next integration if a local binary is available

Important diagnostic built into the implementation:
for the naive compact equilateral Regge action S = 5 arccos(1/4) and
gamma_AS = 0.274, one gets |sin(gamma S)| ~ O(1), not 1e-4.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


GAMMA_AS_DEFAULT = 0.274
NU_DEFAULT = 1e-3
DELTA_CP_DEFAULT = 1e-3
OBSERVED_ETA = 6.1e-10


@dataclass
class ReggeActionSet:
    theta_dihedral: float
    area_triangle: float
    deficit_single_simplex: float
    s_hinge_sum: float
    s_compact: float


@dataclass
class AnalyticResult:
    gamma: float
    S_regge: float
    gamma_times_S: float
    f_geo: float
    eta: float
    log10_f_geo: float
    ratio_to_observed_eta: float


@dataclass
class NumericalAmplitudeResult:
    j: float
    j2: int
    gamma: float
    real_part: float
    imag_part: float
    magnitude: float
    phase: float
    f_geo_direct: float
    f_geo_analytic_same_j: float
    expected_gamma_S: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute f_geo from analytic EPRL expressions and optional sl2cfoam amplitudes."
    )
    parser.add_argument("--gamma", type=float, default=GAMMA_AS_DEFAULT,
                        help="Barbero-Immirzi parameter gamma. Default: %(default)s")
    parser.add_argument("--nu", type=float, default=NU_DEFAULT,
                        help="Dimensionless prefactor nu in eta = nu * delta_cp * f_geo.")
    parser.add_argument("--delta-cp", type=float, default=DELTA_CP_DEFAULT,
                        help="Dimensionless CP factor delta_CP in eta = nu * delta_cp * f_geo.")
    parser.add_argument("--j-max", type=int, default=10,
                        help="Maximum spin j included in higher-spin average.")
    parser.add_argument("--l-ratio", type=float, default=1.0,
                        help="l_CDT / l_Pl used in higher-spin Boltzmann weighting.")
    parser.add_argument("--use-hinge-sum", action="store_true",
                        help="Use the 10-hinge single-simplex action instead of the compact 5 arccos(1/4) value.")
    parser.add_argument("--gamma-grid-min", type=float, default=0.20)
    parser.add_argument("--gamma-grid-max", type=float, default=0.35)
    parser.add_argument("--gamma-grid-n", type=int, default=151)
    parser.add_argument("--s-grid-min", type=float, default=4.0)
    parser.add_argument("--s-grid-max", type=float, default=10.0)
    parser.add_argument("--s-grid-n", type=int, default=151)
    parser.add_argument("--sl2cfoam-bin", default="sl2cfoam",
                        help="Path to sl2cfoam-next executable, if available.")
    parser.add_argument("--run-sl2cfoam", action="store_true",
                        help="Actually run sl2cfoam for the requested j2 values if the binary is available.")
    parser.add_argument("--dl-min", type=int, default=0)
    parser.add_argument("--dl-max", type=int, default=30)
    parser.add_argument("--j2-values", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6],
                        help="Twice-spin values passed to sl2cfoam, e.g. 1 2 3 4 5 6.")
    parser.add_argument("--amplitude-dir", default=None,
                        help="Directory of precomputed sl2cfoam output files to parse instead of running the binary.")
    parser.add_argument("--output-dir", default="data/p18_f_geo",
                        help="Directory for JSON and plot output.")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def compute_regge_actions() -> ReggeActionSet:
    theta_dihedral = math.acos(0.25)
    area_triangle = math.sqrt(3.0) / 4.0
    deficit_single_simplex = math.pi - theta_dihedral
    s_hinge_sum = 10.0 * area_triangle * deficit_single_simplex
    s_compact = 5.0 * theta_dihedral
    return ReggeActionSet(
        theta_dihedral=theta_dihedral,
        area_triangle=area_triangle,
        deficit_single_simplex=deficit_single_simplex,
        s_hinge_sum=s_hinge_sum,
        s_compact=s_compact,
    )


def compute_f_geo(gamma: float, s_regge: float, nu: float, delta_cp: float) -> AnalyticResult:
    gamma_times_S = gamma * s_regge
    f_geo = abs(math.sin(gamma_times_S))
    eta = nu * delta_cp * f_geo
    log10_f_geo = math.log10(f_geo) if f_geo > 0 else float("-inf")
    ratio_to_observed_eta = eta / OBSERVED_ETA if OBSERVED_ETA > 0 else float("inf")
    return AnalyticResult(
        gamma=gamma,
        S_regge=s_regge,
        gamma_times_S=gamma_times_S,
        f_geo=f_geo,
        eta=eta,
        log10_f_geo=log10_f_geo,
        ratio_to_observed_eta=ratio_to_observed_eta,
    )


def higher_spin_average(gamma: float, s_half: float, j_max: int, l_ratio: float) -> dict:
    rows: list[dict] = []
    total_weight = 0.0
    total_f_geo = 0.0
    for j2 in range(1, 2 * j_max + 1):
        j = j2 / 2.0
        d_j = 2 * j + 1
        weight = d_j * math.exp(-j * (j + 1) / max(l_ratio ** 2, 1e-12))
        s_j = 2.0 * j * s_half
        f_j = abs(math.sin(gamma * s_j))
        contribution = weight * f_j
        rows.append(
            {
                "j": j,
                "j2": j2,
                "weight": weight,
                "S_regge": s_j,
                "f_geo_j": f_j,
                "contribution": contribution,
            }
        )
        total_weight += weight
        total_f_geo += contribution

    avg = total_f_geo / total_weight if total_weight > 0 else 0.0
    f_half = abs(math.sin(gamma * s_half))
    correction_fraction = (avg / f_half - 1.0) if f_half > 0 else float("inf")
    return {
        "j_max": j_max,
        "l_ratio": l_ratio,
        "total_weight": total_weight,
        "f_geo_weighted_average": avg,
        "f_geo_j_half": f_half,
        "correction_fraction": correction_fraction,
        "rows": rows,
    }


def sensitivity_scan_gamma(gamma_values: np.ndarray, s_regge: float, nu: float, delta_cp: float) -> list[dict]:
    out: list[dict] = []
    for gamma in gamma_values:
        res = compute_f_geo(float(gamma), s_regge, nu, delta_cp)
        out.append(asdict(res))
    return out


def sensitivity_scan_s(gamma: float, s_values: np.ndarray, nu: float, delta_cp: float) -> list[dict]:
    out: list[dict] = []
    for s_regge in s_values:
        res = compute_f_geo(gamma, float(s_regge), nu, delta_cp)
        out.append(asdict(res))
    return out


def parse_sl2cfoam_output(path: Path) -> complex:
    """Parse a sl2cfoam output file.

    Accepts either a final pair of floats on the last data line or a simple
    two-column output. Lines beginning with # are ignored.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    numeric_lines: list[list[float]] = []
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [float(tok) for tok in float_re.findall(line)]
        if len(parts) >= 2:
            numeric_lines.append(parts)
    if not numeric_lines:
        raise ValueError(f"Could not parse any numeric amplitude from {path}")
    vals = numeric_lines[-1]
    return complex(vals[0], vals[1])


def f_geo_direct_from_amplitude(amplitude: complex) -> float:
    mag2 = amplitude.real ** 2 + amplitude.imag ** 2
    if mag2 <= 1e-300:
        return 0.0
    return abs(2.0 * amplitude.real * amplitude.imag / mag2)


def run_sl2cfoam(sl2cfoam_bin: str, gamma: float, j2: int, dl_min: int, dl_max: int) -> complex:
    spins = [str(j2)] * 10
    cmd = [sl2cfoam_bin, f"{gamma}", *spins, str(dl_min), str(dl_max)]
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    tmp_path = None
    try:
        tmp_path = Path("/tmp/sl2cfoam_output_p18.txt")
        tmp_path.write_text(completed.stdout, encoding="utf-8")
        return parse_sl2cfoam_output(tmp_path)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def collect_numerical_results(
    gamma: float,
    s_half: float,
    sl2cfoam_bin: str,
    run_binary: bool,
    amplitude_dir: str | None,
    j2_values: Iterable[int],
    dl_min: int,
    dl_max: int,
) -> list[dict]:
    results: list[dict] = []
    amp_dir_path = Path(amplitude_dir) if amplitude_dir else None
    bin_path = shutil.which(sl2cfoam_bin) if run_binary else None

    for j2 in j2_values:
        j = j2 / 2.0
        amplitude: complex | None = None
        source = None

        if amp_dir_path is not None:
            candidate = amp_dir_path / f"amplitude_j{j2}_g{gamma}.dat"
            if not candidate.exists():
                candidate = amp_dir_path / f"amplitude_j{j2}_g{gamma:.4f}.dat"
            if candidate.exists():
                amplitude = parse_sl2cfoam_output(candidate)
                source = str(candidate)

        if amplitude is None and run_binary and bin_path:
            try:
                amplitude = run_sl2cfoam(bin_path, gamma, j2, dl_min, dl_max)
                source = bin_path
            except Exception as exc:  # pragma: no cover
                results.append(
                    {
                        "j": j,
                        "j2": j2,
                        "gamma": gamma,
                        "error": f"sl2cfoam failed: {exc}",
                    }
                )
                continue

        if amplitude is None:
            continue

        expected_gamma_S = gamma * (2.0 * j * s_half)
        analytic_same_j = abs(math.sin(expected_gamma_S))
        res = NumericalAmplitudeResult(
            j=j,
            j2=j2,
            gamma=gamma,
            real_part=float(amplitude.real),
            imag_part=float(amplitude.imag),
            magnitude=float(abs(amplitude)),
            phase=float(np.angle(amplitude)),
            f_geo_direct=float(f_geo_direct_from_amplitude(amplitude)),
            f_geo_analytic_same_j=float(analytic_same_j),
            expected_gamma_S=float(expected_gamma_S),
            source=source or "unknown",
        )
        results.append(asdict(res))
    return results


def build_decision_tree(f_geo: float) -> dict:
    if 1e-5 <= f_geo <= 1e-3:
        verdict = "parameter_free_baryogenesis_viable"
        note = "f_geo is near the 1e-4 target scale."
    elif f_geo > 1e-2:
        verdict = "too_large_for_naive_eta"
        note = "Naive eta = nu * delta_CP * f_geo overshoots observation unless extra suppression exists."
    elif f_geo == 0.0:
        verdict = "exact_cancellation"
        note = "No chiral asymmetry in this channel."
    else:
        verdict = "intermediate_regime"
        note = "Not at the target scale, but also not O(1)."
    return {"verdict": verdict, "note": note}


def make_plot(
    output_path: Path,
    gamma_scan: list[dict],
    s_scan: list[dict],
    gamma_target: float,
    s_target: float,
) -> None:
    if plt is None:
        return
    gamma_vals = np.array([row["gamma"] for row in gamma_scan])
    f_gamma = np.array([row["f_geo"] for row in gamma_scan])
    eta_gamma = np.array([row["eta"] for row in gamma_scan])

    s_vals = np.array([row["S_regge"] for row in s_scan])
    f_s = np.array([row["f_geo"] for row in s_scan])
    eta_s = np.array([row["eta"] for row in s_scan])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(gamma_vals, f_gamma)
    axes[0, 0].axvline(gamma_target, linestyle="--")
    axes[0, 0].axhline(1e-4, linestyle=":")
    axes[0, 0].set_xlabel("gamma")
    axes[0, 0].set_ylabel("f_geo")
    axes[0, 0].set_title("f_geo vs gamma")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(gamma_vals, eta_gamma)
    axes[0, 1].axvline(gamma_target, linestyle="--")
    axes[0, 1].axhline(OBSERVED_ETA, linestyle=":")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("gamma")
    axes[0, 1].set_ylabel("eta")
    axes[0, 1].set_title("eta vs gamma")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(s_vals, f_s)
    axes[1, 0].axvline(s_target, linestyle="--")
    axes[1, 0].axhline(1e-4, linestyle=":")
    axes[1, 0].set_xlabel("S_Regge")
    axes[1, 0].set_ylabel("f_geo")
    axes[1, 0].set_title("f_geo vs S_Regge")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(s_vals, eta_s)
    axes[1, 1].axvline(s_target, linestyle="--")
    axes[1, 1].axhline(OBSERVED_ETA, linestyle=":")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("S_Regge")
    axes[1, 1].set_ylabel("eta")
    axes[1, 1].set_title("eta vs S_Regge")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regge = compute_regge_actions()
    s_target = regge.s_hinge_sum if args.use_hinge_sum else regge.s_compact
    analytic = compute_f_geo(args.gamma, s_target, args.nu, args.delta_cp)
    higher_spins = higher_spin_average(args.gamma, regge.s_compact, args.j_max, args.l_ratio)

    gamma_scan = sensitivity_scan_gamma(
        np.linspace(args.gamma_grid_min, args.gamma_grid_max, args.gamma_grid_n),
        s_target,
        args.nu,
        args.delta_cp,
    )
    s_scan = sensitivity_scan_s(
        args.gamma,
        np.linspace(args.s_grid_min, args.s_grid_max, args.s_grid_n),
        args.nu,
        args.delta_cp,
    )

    numerical = collect_numerical_results(
        gamma=args.gamma,
        s_half=regge.s_compact,
        sl2cfoam_bin=args.sl2cfoam_bin,
        run_binary=args.run_sl2cfoam,
        amplitude_dir=args.amplitude_dir,
        j2_values=args.j2_values,
        dl_min=args.dl_min,
        dl_max=args.dl_max,
    )

    result = {
        "inputs": vars(args),
        "constants": {
            "observed_eta": OBSERVED_ETA,
        },
        "regge_actions": asdict(regge),
        "analytic_primary": asdict(analytic),
        "analytic_diagnostic": {
            "compact_vs_hinge_ratio": regge.s_hinge_sum / regge.s_compact,
            "note": "The compact choice gives the note's critical O(1) diagnostic for f_geo at gamma=0.274.",
        },
        "higher_spin_average": higher_spins,
        "numerical_amplitudes": numerical,
        "decision_tree": build_decision_tree(analytic.f_geo),
        "summary": {
            "f_geo_target_scale": 1e-4,
            "naive_prediction_matches_target": bool(1e-5 <= analytic.f_geo <= 1e-3),
            "eta_matches_observation_within_order_of_magnitude": bool(0.1 <= analytic.ratio_to_observed_eta <= 10.0),
        },
    }

    json_path = output_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    plot_path = output_dir / "P18_f_geo_from_EPRL.png"
    if not args.skip_plot:
        make_plot(plot_path, gamma_scan, s_scan, args.gamma, s_target)

    print("=" * 70)
    print("P18: Computing f_geo from the EPRL Chiral Amplitude")
    print("=" * 70)
    print(f"gamma = {args.gamma:.6f}")
    print(f"S_Regge (compact)   = {regge.s_compact:.6f}")
    print(f"S_Regge (hinge sum) = {regge.s_hinge_sum:.6f}")
    print(f"Using S_Regge       = {s_target:.6f}")
    print()
    print(f"gamma * S = {analytic.gamma_times_S:.6f}")
    print(f"f_geo     = {analytic.f_geo:.6e}")
    print(f"eta       = {analytic.eta:.6e}")
    print(f"eta / eta_obs = {analytic.ratio_to_observed_eta:.3e}")
    print()
    if analytic.f_geo > 1e-2:
        print("Diagnostic: naive analytic f_geo is O(1e-2 to 1), not near 1e-4.")
    else:
        print("Diagnostic: naive analytic f_geo is not O(1).")
    print(f"Higher-spin weighted average = {higher_spins['f_geo_weighted_average']:.6e}")
    if numerical:
        print(f"Loaded {len(numerical)} numerical amplitude result(s).")
    else:
        print("No numerical sl2cfoam amplitudes were loaded.")
    print(f"Saved JSON: {json_path}")
    if not args.skip_plot and plt is not None:
        print(f"Saved plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
