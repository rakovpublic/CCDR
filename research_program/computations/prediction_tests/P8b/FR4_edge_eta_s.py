#!/usr/bin/env python3
"""FR4_edge_eta_s.py

Test whether edge-plasma eta/s shows non-monotonic behaviour as a function
of collisionality nu* and estimate the minimum relative to the KSS bound.

This implementation supports either:
  1) an embedded approximate literature-inspired dataset, or
  2) a user-supplied CSV with columns:
       nu_star, chi_edge_m2_s, T_eV, n_1e19_m3[, source]

Outputs:
  - printed summary
  - JSON results
  - PNG plot
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# Physical constants
HBAR = 1.054_571_817e-34  # J s
KB = 1.380_649e-23        # J / K
M_D = 3.343_583_719e-27   # kg, deuteron mass
KSS = HBAR / (4.0 * math.pi * KB)
EV_TO_K = 11604.51812

# Approximate literature-style fallback data from the research note.
DEFAULT_DATA = [
    {"nu_star": 0.01, "chi_edge_m2_s": 0.50, "T_eV": 500.0, "n_1e19_m3": 3.0,  "source": "approx_low_nu"},
    {"nu_star": 0.03, "chi_edge_m2_s": 0.30, "T_eV": 300.0, "n_1e19_m3": 4.0,  "source": "approx_low_nu"},
    {"nu_star": 0.10, "chi_edge_m2_s": 0.15, "T_eV": 200.0, "n_1e19_m3": 5.0,  "source": "approx_mid_nu"},
    {"nu_star": 0.30, "chi_edge_m2_s": 0.08, "T_eV": 150.0, "n_1e19_m3": 6.0,  "source": "approx_mid_nu"},
    {"nu_star": 1.00, "chi_edge_m2_s": 0.10, "T_eV": 100.0, "n_1e19_m3": 8.0,  "source": "approx_high_nu"},
    {"nu_star": 3.00, "chi_edge_m2_s": 0.20, "T_eV": 80.0,  "n_1e19_m3": 10.0, "source": "approx_high_nu"},
    {"nu_star": 10.0, "chi_edge_m2_s": 0.50, "T_eV": 50.0,  "n_1e19_m3": 15.0, "source": "approx_very_high_nu"},
]


@dataclass
class EdgePoint:
    nu_star: float
    chi_edge_m2_s: float
    T_eV: float
    n_1e19_m3: float
    source: str = ""


@dataclass
class DerivedPoint:
    nu_star: float
    chi_edge_m2_s: float
    T_eV: float
    n_1e19_m3: float
    eta_over_s: float
    eta_over_s_over_kss: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FR4 edge-plasma eta/s non-monotonicity test")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional input CSV with columns nu_star, chi_edge_m2_s, T_eV, n_1e19_m3[, source]")
    parser.add_argument("--output-dir", type=Path, default=Path("data/fr4_edge_eta_s"),
                        help="Directory for JSON and plot outputs")
    parser.add_argument("--poly-degree", type=int, default=2,
                        help="Polynomial degree in log10(nu*) used for a smooth trend fit (default: 2)")
    parser.add_argument("--skip-plot", action="store_true", help="Do not generate PNG plot")
    return parser.parse_args()


def chi_to_eta_over_s(chi_m2_s: float, T_eV: float, n_1e19_m3: float) -> float:
    """Convert thermal diffusivity to eta/s using the note's approximation.

    Approximation used in the research note:
      eta ~ n m chi
      s   ~ n k_B
    therefore
      eta/s ~ m chi / k_B

    Temperature and density are retained in the interface for transparency and
    future refinement, even though they cancel in this approximation.
    """
    _ = T_eV
    _ = n_1e19_m3
    return M_D * chi_m2_s / KB


def validate_point(point: EdgePoint) -> None:
    if point.nu_star <= 0:
        raise ValueError(f"nu_star must be positive, got {point.nu_star}")
    if point.chi_edge_m2_s <= 0:
        raise ValueError(f"chi_edge_m2_s must be positive, got {point.chi_edge_m2_s}")
    if point.T_eV <= 0:
        raise ValueError(f"T_eV must be positive, got {point.T_eV}")
    if point.n_1e19_m3 <= 0:
        raise ValueError(f"n_1e19_m3 must be positive, got {point.n_1e19_m3}")


def load_points_from_csv(path: Path) -> List[EdgePoint]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"nu_star", "chi_edge_m2_s", "T_eV", "n_1e19_m3"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        points: List[EdgePoint] = []
        for i, row in enumerate(reader, start=2):
            try:
                point = EdgePoint(
                    nu_star=float(row["nu_star"]),
                    chi_edge_m2_s=float(row["chi_edge_m2_s"]),
                    T_eV=float(row["T_eV"]),
                    n_1e19_m3=float(row["n_1e19_m3"]),
                    source=(row.get("source") or "csv"),
                )
                validate_point(point)
                points.append(point)
            except Exception as exc:
                raise ValueError(f"Failed to parse row {i}: {row!r} ({exc})") from exc

    if len(points) < 3:
        raise ValueError("Need at least 3 data points")
    return points


def load_default_points() -> List[EdgePoint]:
    points = [EdgePoint(**row) for row in DEFAULT_DATA]
    for p in points:
        validate_point(p)
    return points


def derive_points(points: Iterable[EdgePoint]) -> List[DerivedPoint]:
    derived: List[DerivedPoint] = []
    for p in points:
        eta_over_s = chi_to_eta_over_s(p.chi_edge_m2_s, p.T_eV, p.n_1e19_m3)
        derived.append(
            DerivedPoint(
                nu_star=p.nu_star,
                chi_edge_m2_s=p.chi_edge_m2_s,
                T_eV=p.T_eV,
                n_1e19_m3=p.n_1e19_m3,
                eta_over_s=eta_over_s,
                eta_over_s_over_kss=eta_over_s / KSS,
                source=p.source,
            )
        )
    return sorted(derived, key=lambda x: x.nu_star)


def check_discrete_non_monotonicity(values: np.ndarray) -> bool:
    if len(values) < 3:
        return False
    diffs = np.diff(values)
    has_down = np.any(diffs < 0)
    has_up = np.any(diffs > 0)
    interior_min = int(np.argmin(values)) not in {0, len(values) - 1}
    return bool(has_down and has_up and interior_min)


def fit_log_poly(nu_star: np.ndarray, eta_over_s: np.ndarray, degree: int = 2) -> dict:
    degree = max(1, min(degree, len(nu_star) - 1))
    x = np.log10(nu_star)
    coeffs = np.polyfit(x, eta_over_s, degree)
    poly = np.poly1d(coeffs)

    x_grid = np.linspace(x.min(), x.max(), 1000)
    y_grid = poly(x_grid)
    idx_min = int(np.argmin(y_grid))
    x_min = float(x_grid[idx_min])
    y_min = float(y_grid[idx_min])

    if degree >= 2:
        dpoly = np.polyder(poly)
        roots = [r.real for r in np.roots(dpoly) if abs(r.imag) < 1e-9 and x.min() <= r.real <= x.max()]
        interior_roots = list(sorted(roots))
    else:
        interior_roots = []

    return {
        "degree": degree,
        "coefficients": [float(c) for c in coeffs],
        "x_grid": x_grid,
        "y_grid": y_grid,
        "nu_star_at_min": 10.0 ** x_min,
        "eta_over_s_min": y_min,
        "eta_over_s_min_over_kss": y_min / KSS,
        "interior_stationary_points_log10_nu": interior_roots,
        "suggests_interior_minimum": bool(len(interior_roots) > 0 and x.min() < x_min < x.max()),
    }


def summarise(derived: List[DerivedPoint], poly_fit: dict) -> dict:
    nu = np.array([d.nu_star for d in derived], dtype=float)
    eta_s = np.array([d.eta_over_s for d in derived], dtype=float)
    discrete_min_idx = int(np.argmin(eta_s))
    discrete_min = derived[discrete_min_idx]
    discrete_nonmono = check_discrete_non_monotonicity(eta_s)

    kss_ratio_min = float(np.min(eta_s) / KSS)
    if kss_ratio_min < 10:
        closeness = "within 1 order of magnitude"
    elif kss_ratio_min < 100:
        closeness = "within 2 orders of magnitude"
    elif kss_ratio_min < 1000:
        closeness = "within 3 orders of magnitude"
    else:
        closeness = "more than 3 orders of magnitude above"

    if discrete_nonmono or poly_fit["suggests_interior_minimum"]:
        verdict = "NON_MONOTONICITY SUPPORTED by current dataset"
    else:
        verdict = "NO clear non-monotonicity from current dataset"

    return {
        "n_points": len(derived),
        "kss_bound": KSS,
        "discrete_minimum": asdict(discrete_min),
        "discrete_non_monotonicity": discrete_nonmono,
        "smooth_fit": {
            "degree": poly_fit["degree"],
            "nu_star_at_min": float(poly_fit["nu_star_at_min"]),
            "eta_over_s_min": float(poly_fit["eta_over_s_min"]),
            "eta_over_s_min_over_kss": float(poly_fit["eta_over_s_min_over_kss"]),
            "suggests_interior_minimum": bool(poly_fit["suggests_interior_minimum"]),
        },
        "closest_approach_to_kss_ratio": kss_ratio_min,
        "closest_approach_to_kss_description": closeness,
        "verdict": verdict,
    }


def print_report(derived: List[DerivedPoint], summary: dict) -> None:
    print("FR4: Edge Plasma eta/s vs Collisionality")
    print("=" * 68)
    print(f"KSS bound: eta/s = hbar/(4 pi k_B) = {KSS:.6e}")
    print()
    for d in derived:
        print(
            f"nu* = {d.nu_star:7.3g} | chi = {d.chi_edge_m2_s:6.3f} m^2/s | "
            f"T = {d.T_eV:7.1f} eV | n = {d.n_1e19_m3:5.2f}e19 m^-3 | "
            f"eta/s = {d.eta_over_s:.6e} | (eta/s)/KSS = {d.eta_over_s_over_kss:9.2f}"
        )

    dmin = summary["discrete_minimum"]
    print()
    print(f"Discrete minimum at nu* = {dmin['nu_star']:.6g}")
    print(f"  eta/s_min = {dmin['eta_over_s']:.6e}")
    print(f"  (eta/s)_min / KSS = {dmin['eta_over_s_over_kss']:.2f}")
    print(f"Discrete non-monotonicity: {summary['discrete_non_monotonicity']}")
    print()
    print("Smooth-fit estimate:")
    print(f"  nu*_min ~ {summary['smooth_fit']['nu_star_at_min']:.6g}")
    print(f"  eta/s_min ~ {summary['smooth_fit']['eta_over_s_min']:.6e}")
    print(f"  (eta/s)_min / KSS ~ {summary['smooth_fit']['eta_over_s_min_over_kss']:.2f}")
    print(f"  interior minimum suggested: {summary['smooth_fit']['suggests_interior_minimum']}")
    print()
    print(f"Closest approach to KSS: {summary['closest_approach_to_kss_description']}")
    print(f"Verdict: {summary['verdict']}")


def save_json(output_path: Path, derived: List[DerivedPoint], summary: dict, poly_fit: dict) -> None:
    payload = {
        "inputs": [asdict(d) for d in derived],
        "summary": summary,
        "smooth_fit_full": {
            "degree": poly_fit["degree"],
            "coefficients": poly_fit["coefficients"],
            "nu_star_grid": [float(10.0 ** x) for x in poly_fit["x_grid"]],
            "eta_over_s_grid": [float(y) for y in poly_fit["y_grid"]],
            "interior_stationary_points_log10_nu": [float(v) for v in poly_fit["interior_stationary_points_log10_nu"]],
        },
        "constants": {
            "HBAR": HBAR,
            "KB": KB,
            "M_D": M_D,
            "KSS": KSS,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_plot(output_path: Path, derived: List[DerivedPoint], poly_fit: dict) -> None:
    import matplotlib.pyplot as plt

    nu = np.array([d.nu_star for d in derived], dtype=float)
    eta_s = np.array([d.eta_over_s for d in derived], dtype=float)
    chi = np.array([d.chi_edge_m2_s for d in derived], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(nu, eta_s / KSS, s=70, c=chi)
    ax.plot(np.array([10.0 ** x for x in poly_fit["x_grid"]]), poly_fit["y_grid"] / KSS, lw=2)
    ax.axhline(1.0, linestyle="--", linewidth=1.5)
    ax.axvline(poly_fit["nu_star_at_min"], linestyle=":", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Collisionality $\nu_*$")
    ax.set_ylabel(r"$(\eta/s) / (\hbar / 4\pi k_B)$")
    ax.set_title(r"FR4 edge plasma $\eta/s$ vs collisionality")
    ax.grid(True, which="both", alpha=0.3)
    plt.colorbar(sc, ax=ax, label=r"$\chi_{edge}$ (m$^2$/s)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv is not None:
        points = load_points_from_csv(args.csv)
        input_mode = "csv"
    else:
        points = load_default_points()
        input_mode = "embedded_approximate_dataset"

    derived = derive_points(points)
    nu = np.array([d.nu_star for d in derived], dtype=float)
    eta_s = np.array([d.eta_over_s for d in derived], dtype=float)
    poly_fit = fit_log_poly(nu, eta_s, degree=args.poly_degree)
    summary = summarise(derived, poly_fit)
    summary["input_mode"] = input_mode

    print_report(derived, summary)

    json_path = args.output_dir / "result.json"
    save_json(json_path, derived, summary, poly_fit)
    print(f"\nSaved JSON: {json_path}")

    if not args.skip_plot:
        plot_path = args.output_dir / "FR4_edge_eta_s.png"
        save_plot(plot_path, derived, poly_fit)
        print(f"Saved plot: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
