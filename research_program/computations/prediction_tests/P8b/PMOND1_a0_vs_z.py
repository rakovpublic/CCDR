#!/usr/bin/env python3
"""
PMOND1_a0_vs_z.py

Practical implementation of the P-MOND-1 idea:
  1. Fit a local z≈0 MOND acceleration scale a0 from the public SPARC
     radial-acceleration-relation table (RAR.mrt).
  2. Optionally ingest high-z measurements from a summary CSV or derive
     them from user-provided rotation-curve CSV data.
  3. Compare redshift-trend models:
       - fixed MOND constant a0 = 1.2e-10 m/s^2
       - free constant a0 = const
       - raw CCDR a0(z) = c H(z)
       - shape-only CCDR a0(z) = A_ref * H(z)/H0

This script is intentionally more robust than the placeholder spec:
  - it uses the live SPARC RAR table directly;
  - it does not fabricate high-z measurements;
  - it supports user CSV inputs for summary measurements or raw curves.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
C_LIGHT = 2.99792458e8  # m/s
DEFAULT_H0 = 67.4       # km/s/Mpc
DEFAULT_OMEGA_M = 0.315
A0_STANDARD = 1.2e-10   # m/s^2
MPC_M = 3.085677581491367e22

DEFAULT_RAR_URL = "https://astroweb.cwru.edu/SPARC/RAR.mrt"


@dataclass
class Measurement:
    z: float
    a0: float
    a0_err: float
    source: str


@dataclass
class ModelFit:
    name: str
    parameter: float | None
    chi2: float
    dof: int
    chi2_red: float
    aic: float
    bic: float


# -----------------------------------------------------------------------------
# Cosmology and MOND helpers
# -----------------------------------------------------------------------------
def h0_si(h0_km_s_mpc: float) -> float:
    return h0_km_s_mpc * 1e3 / MPC_M


def e_z(z: np.ndarray | float, omega_m: float = DEFAULT_OMEGA_M) -> np.ndarray | float:
    z = np.asarray(z, dtype=float)
    omega_l = 1.0 - omega_m
    return np.sqrt(omega_m * (1.0 + z) ** 3 + omega_l)


def h_z(z: np.ndarray | float, h0_km_s_mpc: float = DEFAULT_H0,
        omega_m: float = DEFAULT_OMEGA_M) -> np.ndarray | float:
    return h0_si(h0_km_s_mpc) * e_z(z, omega_m)


def a0_ccdr(z: np.ndarray | float, h0_km_s_mpc: float = DEFAULT_H0,
            omega_m: float = DEFAULT_OMEGA_M) -> np.ndarray | float:
    return C_LIGHT * h_z(z, h0_km_s_mpc, omega_m)


def mond_simple_interp(g_bar: np.ndarray | float, a0: float) -> np.ndarray | float:
    """McGaugh/Lelli simple RAR interpolation in linear acceleration units."""
    g_bar = np.asarray(g_bar, dtype=float)
    x = np.sqrt(np.clip(g_bar / a0, 1e-300, None))
    denom = 1.0 - np.exp(-x)
    denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
    return g_bar / denom


def mond_rotation_kms(r_kpc: np.ndarray, m_bary_solar: float, a0: float) -> np.ndarray:
    """Simple circular-speed approximation from the placeholder note."""
    g_newton = 6.67430e-11
    m_sun = 1.98847e30
    r_m = np.asarray(r_kpc, dtype=float) * 3.085677581491367e19
    m_kg = m_bary_solar * m_sun
    v2_n = g_newton * m_kg / np.clip(r_m, 1e-30, None)
    x = v2_n / np.clip(a0 * r_m, 1e-300, None)
    mu_eff = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / np.clip(x, 1e-300, None) ** 2))
    v2_mond = v2_n / mu_eff
    return np.sqrt(np.clip(v2_mond, 0.0, None)) / 1e3


# -----------------------------------------------------------------------------
# SPARC RAR handling
# -----------------------------------------------------------------------------
def download_file(url: str, dest: Path, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PMOND1)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dest.write_bytes(resp.read())
    return dest


def parse_rar_mrt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse SPARC RAR.mrt as fixed-width/log10 columns."""
    log_gbar, e_log_gbar, log_gobs, e_log_gobs = [], [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("Title:", "Authors:", "Table:", "Byte-by-byte", "Bytes ", "Format ")):
                continue
            if set(stripped) <= {"=", "-"}:
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                a, b, c, d = map(float, parts[:4])
            except ValueError:
                continue
            log_gbar.append(a)
            e_log_gbar.append(abs(b))
            log_gobs.append(c)
            e_log_gobs.append(max(abs(d), 1e-6))

    if len(log_gbar) < 50:
        raise RuntimeError(f"Failed to parse enough RAR rows from {path}")

    return (np.asarray(log_gbar), np.asarray(e_log_gbar),
            np.asarray(log_gobs), np.asarray(e_log_gobs))


def chi2_rar(a0: float, log_gbar: np.ndarray, log_gobs: np.ndarray,
             e_log_gobs: np.ndarray) -> float:
    g_bar = 10.0 ** log_gbar
    pred = mond_simple_interp(g_bar, a0)
    log_pred = np.log10(pred)
    resid = (log_gobs - log_pred) / e_log_gobs
    return float(np.sum(resid ** 2))


def fit_a0_from_rar(log_gbar: np.ndarray, log_gobs: np.ndarray,
                    e_log_gobs: np.ndarray,
                    lower: float = 1e-12, upper: float = 1e-8) -> tuple[float, float, float]:
    result = minimize_scalar(
        lambda log10_a0: chi2_rar(10.0 ** log10_a0, log_gbar, log_gobs, e_log_gobs),
        bounds=(math.log10(lower), math.log10(upper)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    best_log10_a0 = float(result.x)
    best_a0 = 10.0 ** best_log10_a0
    chi2_min = float(result.fun)

    grid = np.linspace(best_log10_a0 - 1.0, best_log10_a0 + 1.0, 4000)
    chis = np.array([chi2_rar(10.0 ** lg, log_gbar, log_gobs, e_log_gobs) for lg in grid])
    mask = chis <= chi2_min + 1.0
    if np.any(mask):
        lo = 10.0 ** grid[mask][0]
        hi = 10.0 ** grid[mask][-1]
        err = max(best_a0 - lo, hi - best_a0)
    else:
        err = np.nan
    return best_a0, err, chi2_min


# -----------------------------------------------------------------------------
# High-z inputs
# -----------------------------------------------------------------------------
def load_measurements_csv(path: Path) -> list[Measurement]:
    rows: list[Measurement] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        needed = {"z", "a0", "a0_err"}
        if not needed.issubset(reader.fieldnames or []):
            raise RuntimeError(
                f"Measurement CSV must contain columns {sorted(needed)}; got {reader.fieldnames}"
            )
        for row in reader:
            if not row or not row.get("z"):
                continue
            rows.append(
                Measurement(
                    z=float(row["z"]),
                    a0=float(row["a0"]),
                    a0_err=max(float(row["a0_err"]), 1e-30),
                    source=row.get("source", "user_csv") or "user_csv",
                )
            )
    return rows


def fit_a0_to_rotation_curve(r_kpc: np.ndarray, v_kms: np.ndarray, v_err_kms: np.ndarray,
                             m_bary_solar: float,
                             lower: float = 1e-12, upper: float = 1e-8) -> tuple[float, float, float]:
    def chi2(log10_a0: float) -> float:
        a0 = 10.0 ** log10_a0
        v_model = mond_rotation_kms(r_kpc, m_bary_solar, a0)
        resid = (v_kms - v_model) / np.clip(v_err_kms, 1e-9, None)
        return float(np.sum(resid ** 2))

    result = minimize_scalar(
        chi2,
        bounds=(math.log10(lower), math.log10(upper)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    best_log10 = float(result.x)
    best_a0 = 10.0 ** best_log10
    chi2_min = float(result.fun)

    grid = np.linspace(best_log10 - 1.0, best_log10 + 1.0, 2500)
    chis = np.array([chi2(x) for x in grid])
    mask = chis <= chi2_min + 1.0
    if np.any(mask):
        lo = 10.0 ** grid[mask][0]
        hi = 10.0 ** grid[mask][-1]
        err = max(best_a0 - lo, hi - best_a0)
    else:
        err = np.nan
    return best_a0, err, chi2_min


def load_rotation_curve_measurements(path: Path) -> list[Measurement]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for --rotation-curves-csv") from exc

    df = pd.read_csv(path)
    required = {"galaxy", "z", "r_kpc", "v_kms", "v_err_kms", "M_bary_msun"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Rotation-curve CSV must contain columns {sorted(required)}")

    rows: list[Measurement] = []
    for galaxy, sub in df.groupby("galaxy"):
        z_vals = np.unique(sub["z"].astype(float).values)
        m_vals = np.unique(sub["M_bary_msun"].astype(float).values)
        if len(z_vals) != 1 or len(m_vals) != 1:
            raise RuntimeError(f"Galaxy {galaxy} must have a single z and M_bary_msun")
        a0, a0_err, _ = fit_a0_to_rotation_curve(
            sub["r_kpc"].astype(float).values,
            sub["v_kms"].astype(float).values,
            sub["v_err_kms"].astype(float).values,
            float(m_vals[0]),
        )
        rows.append(Measurement(float(z_vals[0]), a0, max(a0_err, 1e-30), str(galaxy)))
    return rows


# -----------------------------------------------------------------------------
# Model comparison
# -----------------------------------------------------------------------------
def fit_weighted_constant(measurements: Iterable[Measurement]) -> ModelFit:
    ms = list(measurements)
    y = np.array([m.a0 for m in ms])
    s = np.array([m.a0_err for m in ms])
    w = 1.0 / s ** 2
    a_best = float(np.sum(w * y) / np.sum(w))
    chi2 = float(np.sum(((y - a_best) / s) ** 2))
    n = len(ms)
    k = 1
    return ModelFit("free_constant", a_best, chi2, n - k, chi2 / max(n - k, 1), chi2 + 2 * k,
                    chi2 + k * np.log(max(n, 1)))


def fit_fixed_constant(measurements: Iterable[Measurement], a0_fixed: float) -> ModelFit:
    ms = list(measurements)
    y = np.array([m.a0 for m in ms])
    s = np.array([m.a0_err for m in ms])
    chi2 = float(np.sum(((y - a0_fixed) / s) ** 2))
    n = len(ms)
    k = 0
    return ModelFit("fixed_constant_milgrom", a0_fixed, chi2, n - 1, chi2 / max(n - 1, 1), chi2 + 2 * k,
                    chi2 + k * np.log(max(n, 1)))


def fit_scaled_ccdr(measurements: Iterable[Measurement], omega_m: float) -> ModelFit:
    ms = list(measurements)
    y = np.array([m.a0 for m in ms])
    s = np.array([m.a0_err for m in ms])
    ez = np.array([float(e_z(m.z, omega_m)) for m in ms])
    w = 1.0 / s ** 2
    a_ref = float(np.sum(w * ez * y) / np.sum(w * ez ** 2))
    pred = a_ref * ez
    chi2 = float(np.sum(((y - pred) / s) ** 2))
    n = len(ms)
    k = 1
    return ModelFit("scaled_ccdr_shape", a_ref, chi2, n - k, chi2 / max(n - k, 1), chi2 + 2 * k,
                    chi2 + k * np.log(max(n, 1)))


def fit_raw_ccdr(measurements: Iterable[Measurement], h0_km_s_mpc: float,
                 omega_m: float) -> ModelFit:
    ms = list(measurements)
    y = np.array([m.a0 for m in ms])
    s = np.array([m.a0_err for m in ms])
    pred = np.array([float(a0_ccdr(m.z, h0_km_s_mpc, omega_m)) for m in ms])
    chi2 = float(np.sum(((y - pred) / s) ** 2))
    n = len(ms)
    k = 0
    return ModelFit("raw_ccdr_cH", None, chi2, n - 1, chi2 / max(n - 1, 1), chi2 + 2 * k,
                    chi2 + k * np.log(max(n, 1)))


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def write_template_measurements_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["z", "a0", "a0_err", "source"])
        writer.writerow([0.9, "", "", "Genzel+2017"])
        writer.writerow([2.0, "", "", "Ubler/Lang+2017"])
        writer.writerow([4.2, "", "", "Rizzo+2020"])


def make_plot(out_path: Path,
              log_gbar: np.ndarray,
              log_gobs: np.ndarray,
              best_local_a0: float,
              measurements: list[Measurement],
              h0_km_s_mpc: float,
              omega_m: float,
              scaled_ccdr_fit: ModelFit | None,
              free_const_fit: ModelFit | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Local RAR fit
    axes[0].scatter(10 ** log_gbar, 10 ** log_gobs, s=8, alpha=0.25)
    xs = np.logspace(np.min(log_gbar), np.max(log_gbar), 400)
    axes[0].plot(xs, mond_simple_interp(xs, best_local_a0), label=f"best local a0={best_local_a0:.2e}")
    axes[0].plot(xs, mond_simple_interp(xs, A0_STANDARD), linestyle="--", label=f"Milgrom {A0_STANDARD:.2e}")
    axes[0].plot(xs, mond_simple_interp(xs, float(a0_ccdr(0.0, h0_km_s_mpc, omega_m))), linestyle=":",
                 label=f"cH0 {float(a0_ccdr(0.0, h0_km_s_mpc, omega_m)):.2e}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("g_bar (m/s²)")
    axes[0].set_ylabel("g_obs (m/s²)")
    axes[0].set_title("SPARC local RAR fit")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    # a0(z) comparison
    if measurements:
        zs = np.array([m.z for m in measurements])
        ys = np.array([m.a0 for m in measurements])
        es = np.array([m.a0_err for m in measurements])
        axes[1].errorbar(zs, ys, yerr=es, fmt="o", capsize=3, label="measurements")

    z_grid = np.linspace(0.0, max([m.z for m in measurements], default=4.2) + 0.2, 400)
    axes[1].plot(z_grid, np.full_like(z_grid, A0_STANDARD), linestyle="--", label="fixed MOND")
    axes[1].plot(z_grid, a0_ccdr(z_grid, h0_km_s_mpc, omega_m), linestyle=":", label="raw CCDR cH(z)")
    if scaled_ccdr_fit and scaled_ccdr_fit.parameter is not None:
        axes[1].plot(z_grid, scaled_ccdr_fit.parameter * e_z(z_grid, omega_m),
                     label="scaled CCDR shape")
    if free_const_fit and free_const_fit.parameter is not None:
        axes[1].plot(z_grid, np.full_like(z_grid, free_const_fit.parameter),
                     label="free constant fit")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("a0 (m/s²)")
    axes[1].set_title("a0 vs redshift")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test constant a0 vs a0(z) scaling using SPARC + user high-z data.")
    p.add_argument("--output-dir", default="data/pmond1_a0_vs_z", help="Directory for outputs")
    p.add_argument("--rar-file", default="", help="Path to local SPARC RAR.mrt file")
    p.add_argument("--rar-url", default=DEFAULT_RAR_URL, help="SPARC RAR URL to use when downloading")
    p.add_argument("--measurements-csv", default="", help="CSV with columns z,a0,a0_err,source")
    p.add_argument("--rotation-curves-csv", default="", help="Long-format CSV with galaxy,z,r_kpc,v_kms,v_err_kms,M_bary_msun")
    p.add_argument("--no-local-anchor", action="store_true", help="Do not include the SPARC local fit as z=0 measurement")
    p.add_argument("--h0", type=float, default=DEFAULT_H0, help="H0 in km/s/Mpc")
    p.add_argument("--omega-m", type=float, default=DEFAULT_OMEGA_M, help="Matter density parameter")
    p.add_argument("--lower-bound", type=float, default=1e-12)
    p.add_argument("--upper-bound", type=float, default=1e-8)
    p.add_argument("--skip-plot", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_path = out_dir / "highz_measurements_template.csv"
    write_template_measurements_csv(template_path)

    # Local SPARC anchor
    rar_path = Path(args.rar_file) if args.rar_file else (out_dir / "RAR.mrt")
    if not rar_path.exists():
        try:
            download_file(args.rar_url, rar_path)
            print(f"[downloaded] {rar_path}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Could not download SPARC RAR file from {args.rar_url}. "
                f"Use --rar-file to point to a local copy."
            ) from exc
    log_gbar, e_log_gbar, log_gobs, e_log_gobs = parse_rar_mrt(rar_path)
    best_local_a0, best_local_err, local_chi2 = fit_a0_from_rar(
        log_gbar, log_gobs, e_log_gobs, args.lower_bound, args.upper_bound
    )

    print("=" * 72)
    print("PMOND1: a0(z) test")
    print("=" * 72)
    print(f"[local] SPARC RAR points: {len(log_gbar)}")
    print(f"[local] Best-fit a0(z≈0): {best_local_a0:.4e} ± {best_local_err:.2e} m/s²")
    print(f"[local] Fixed MOND a0:     {A0_STANDARD:.4e} m/s²")
    print(f"[local] Raw CCDR cH0:      {float(a0_ccdr(0.0, args.h0, args.omega_m)):.4e} m/s²")

    measurements: list[Measurement] = []
    if not args.no_local_anchor:
        measurements.append(Measurement(0.0, best_local_a0, max(best_local_err, 1e-30), "SPARC RAR fit"))

    if args.measurements_csv:
        ms = load_measurements_csv(Path(args.measurements_csv))
        measurements.extend(ms)
        print(f"[input] Loaded {len(ms)} summary measurements from {args.measurements_csv}")

    if args.rotation_curves_csv:
        ms = load_rotation_curve_measurements(Path(args.rotation_curves_csv))
        measurements.extend(ms)
        print(f"[input] Derived {len(ms)} measurements from rotation-curve CSV")

    model_fits: list[ModelFit] = []
    if len(measurements) >= 2:
        fixed_const_fit = fit_fixed_constant(measurements, A0_STANDARD)
        free_const_fit = fit_weighted_constant(measurements)
        raw_ccdr_fit = fit_raw_ccdr(measurements, args.h0, args.omega_m)
        scaled_ccdr_fit = fit_scaled_ccdr(measurements, args.omega_m)
        model_fits = [fixed_const_fit, free_const_fit, raw_ccdr_fit, scaled_ccdr_fit]

        print("\n[model comparison]")
        for fit in sorted(model_fits, key=lambda m: m.aic):
            param = "n/a" if fit.parameter is None else f"{fit.parameter:.3e}"
            print(f"  {fit.name:20s} param={param:>12s}  chi2={fit.chi2:8.3f}  aic={fit.aic:8.3f}  bic={fit.bic:8.3f}")
        best_by_aic = min(model_fits, key=lambda m: m.aic)
        print(f"[best] Lowest AIC: {best_by_aic.name}")
    else:
        fixed_const_fit = free_const_fit = raw_ccdr_fit = scaled_ccdr_fit = None
        print("\n[info] Need at least one non-local high-z measurement to compare redshift models.")
        print(f"[info] Template CSV written to: {template_path}")

    prediction_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.2]
    predictions = [
        {
            "z": z,
            "raw_ccdr_a0": float(a0_ccdr(z, args.h0, args.omega_m)),
            "ratio_to_fixed_mond": float(a0_ccdr(z, args.h0, args.omega_m) / A0_STANDARD),
        }
        for z in prediction_grid
    ]

    out = {
        "config": {
            "h0_km_s_mpc": args.h0,
            "omega_m": args.omega_m,
            "rar_url": args.rar_url,
            "fixed_mond_a0": A0_STANDARD,
        },
        "local_sparc_fit": {
            "rar_file": str(rar_path),
            "n_points": int(len(log_gbar)),
            "best_fit_a0": best_local_a0,
            "best_fit_a0_err": best_local_err,
            "chi2": local_chi2,
        },
        "measurements": [asdict(m) for m in measurements],
        "prediction_grid": predictions,
        "model_fits": [asdict(m) for m in model_fits],
        "template_measurements_csv": str(template_path),
    }
    out_json = out_dir / "result.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[saved] {out_json}")

    if not args.skip_plot:
        plot_path = out_dir / "PMOND1_a0_vs_z.png"
        make_plot(
            plot_path,
            log_gbar,
            log_gobs,
            best_local_a0,
            measurements,
            args.h0,
            args.omega_m,
            scaled_ccdr_fit,
            free_const_fit,
        )
        print(f"[saved] {plot_path}")


if __name__ == "__main__":
    main()
