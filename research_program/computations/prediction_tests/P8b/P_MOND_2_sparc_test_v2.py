#!/usr/bin/env python3
"""
P_MOND_2_sparc_test_v2.py

Fit the MOND acceleration scale a0 to the SPARC RAR table and compare
standard MOND against the CCDR alternative a0(z=0) = c * H0.

Key points:
  1. Uses wide search bounds to avoid optimizer boundary artefacts.
  2. Parses the published SPARC RAR.mrt machine-readable table directly.
  3. Fits in log-space, with optional weighting by the published g_obs
     uncertainties from the table.
  4. Reports fit quality at Milgrom, CCDR, and the best-fit value.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar

DATA_DIR = Path("data/p_mond_2_sparc")
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL_PRIMARY = "https://astroweb.case.edu/SPARC/RAR.mrt"
URL_FALLBACK = "https://astroweb.cwru.edu/SPARC/RAR.mrt"
RAR_FILE = DATA_DIR / "RAR.mrt"

# Predictions
A0_MILGROM = 1.20e-10  # m/s^2
H0_KMS_MPC_DEFAULT = 67.4
MPC_METERS = 3.086e22
C_LIGHT = 2.998e8


def compute_ccdr_a0(h0_kms_mpc: float) -> float:
    """Convert H0 from km/s/Mpc to SI and return c*H0."""
    h0_si = h0_kms_mpc * 1e3 / MPC_METERS
    return C_LIGHT * h0_si


FLOAT_PATTERN = re.compile(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?")


def download_rar(force: bool = False, input_path: str | None = None) -> Path:
    """Download the SPARC RAR table from one of the public mirrors."""
    candidate_paths = []
    if input_path:
        candidate_paths.append(Path(input_path))
    candidate_paths.extend([
        RAR_FILE,
        Path("RAR.mrt"),
        Path(__file__).resolve().with_name("RAR.mrt"),
    ])

    if not force:
        for candidate in candidate_paths:
            if candidate.exists() and candidate.stat().st_size > 1000:
                print(f"[cache] {candidate}")
                return candidate

    for url in (URL_PRIMARY, URL_FALLBACK):
        try:
            print(f"[download] {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (P-MOND-2 test)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 1000:
                raise RuntimeError("downloaded file is unexpectedly small")
            RAR_FILE.write_bytes(data)
            print(f"  [saved] {RAR_FILE} ({len(data) / 1e3:.1f} KB)")
            return RAR_FILE
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f"  [fail] {exc}")
            continue

    raise RuntimeError(
        "Could not download SPARC RAR table. "
        f"Please place it manually at {RAR_FILE}."
    )


def load_rar(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse the SPARC RAR machine-readable table.

    The published table header states:
      gbar   = log10 of baryonic acceleration
      e_gbar = error on gbar
      gobs   = log10 of observed acceleration
      e_gobs = error on gobs

    Returns:
      g_bar_lin, g_obs_lin, e_gobs_dex, raw_log_rows
    """
    rows = []
    with filepath.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip header / metadata lines containing letters.
            if any(ch.isalpha() for ch in stripped):
                continue
            nums = FLOAT_PATTERN.findall(stripped)
            if len(nums) != 4:
                continue
            try:
                gbar_log, e_gbar, gobs_log, e_gobs = map(float, nums)
            except ValueError:
                continue
            rows.append((gbar_log, e_gbar, gobs_log, e_gobs))

    if len(rows) < 100:
        raise RuntimeError(
            f"Only parsed {len(rows)} rows from {filepath}; table format may have changed."
        )

    raw = np.asarray(rows, dtype=float)
    gbar_log = raw[:, 0]
    gobs_log = raw[:, 2]
    e_gobs_dex = raw[:, 3]

    g_bar_lin = np.power(10.0, gbar_log)
    g_obs_lin = np.power(10.0, gobs_log)

    print(f"[load] {len(raw)} rows from {filepath.name}")
    print(
        f"       g_bar range: {g_bar_lin.min():.2e} to {g_bar_lin.max():.2e} m/s^2"
    )
    print(
        f"       g_obs range: {g_obs_lin.min():.2e} to {g_obs_lin.max():.2e} m/s^2"
    )

    return g_bar_lin, g_obs_lin, e_gobs_dex, raw


def mond_simple_interp(g_bar: np.ndarray, a0: float) -> np.ndarray:
    """
    McGaugh, Lelli & Schombert 2016 fitting form:
        g_obs = g_bar / (1 - exp(-sqrt(g_bar / a0)))
    """
    x = np.sqrt(np.clip(g_bar / a0, 1e-300, None))
    denom = 1.0 - np.exp(-x)
    return g_bar / np.clip(denom, 1e-300, None)


def residuals_log10(a0: float, g_bar: np.ndarray, g_obs: np.ndarray) -> np.ndarray:
    pred = mond_simple_interp(g_bar, a0)
    return np.log10(g_obs) - np.log10(pred)


def mse_log10(a0: float, g_bar: np.ndarray, g_obs: np.ndarray) -> float:
    res = residuals_log10(a0, g_bar, g_obs)
    return float(np.mean(res * res))


def chi2_log10(
    a0: float,
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    sigma_dex: np.ndarray,
) -> float:
    res = residuals_log10(a0, g_bar, g_obs)
    sigma = np.clip(sigma_dex, 1e-6, None)
    return float(np.sum((res / sigma) ** 2))


def reduced_chi2_log10(
    a0: float,
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    sigma_dex: np.ndarray,
) -> float:
    chi2 = chi2_log10(a0, g_bar, g_obs, sigma_dex)
    dof = max(len(g_bar) - 1, 1)
    return chi2 / dof


def fit_a0(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    sigma_dex: np.ndarray,
    bounds: Tuple[float, float],
    weighted: bool,
) -> Tuple[float, float, bool]:
    """Fit a0 and flag if the solution is pinned near a boundary."""
    objective = (
        (lambda a0: reduced_chi2_log10(a0, g_bar, g_obs, sigma_dex))
        if weighted
        else (lambda a0: mse_log10(a0, g_bar, g_obs))
    )

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-16, "maxiter": 2000},
    )
    best_a0 = float(result.x)
    best_score = float(result.fun)

    lower, upper = bounds
    hit_lower = abs(best_a0 - lower) / lower < 0.01
    hit_upper = abs(best_a0 - upper) / upper < 0.01
    return best_a0, best_score, (hit_lower or hit_upper)


def make_plot(
    out_png: Path,
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    best_a0: float,
    a0_milgrom: float,
    a0_ccdr: float,
    weighted: bool,
    sigma_dex: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].loglog(g_bar, g_obs, ".", markersize=2, alpha=0.45, label="SPARC data")
    g_smooth = np.logspace(np.log10(g_bar.min()), np.log10(g_bar.max()), 300)
    axes[0].loglog(
        g_smooth,
        mond_simple_interp(g_smooth, best_a0),
        lw=2,
        label=f"Best fit: {best_a0:.3e}",
    )
    axes[0].loglog(
        g_smooth,
        mond_simple_interp(g_smooth, a0_milgrom),
        "--",
        lw=1.5,
        label=f"Milgrom: {a0_milgrom:.3e}",
    )
    axes[0].loglog(
        g_smooth,
        mond_simple_interp(g_smooth, a0_ccdr),
        ":",
        lw=1.8,
        label=f"CCDR: {a0_ccdr:.3e}",
    )
    axes[0].set_xlabel("g_bar (m/s^2)")
    axes[0].set_ylabel("g_obs (m/s^2)")
    axes[0].set_title("SPARC radial acceleration relation")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend()

    a0_scan = np.logspace(-12, -8, 250)
    if weighted:
        y_scan = [reduced_chi2_log10(a, g_bar, g_obs, sigma_dex) for a in a0_scan]
        ylabel = r"reduced $\chi^2$ (log-space)"
    else:
        y_scan = [mse_log10(a, g_bar, g_obs) for a in a0_scan]
        ylabel = "MSE (log-space)"

    axes[1].loglog(a0_scan, y_scan)
    axes[1].axvline(a0_milgrom, linestyle="--", label="Milgrom")
    axes[1].axvline(a0_ccdr, linestyle=":", label="CCDR")
    axes[1].axvline(best_a0, linestyle="-", label="Best fit")
    axes[1].set_xlabel("a0 (m/s^2)")
    axes[1].set_ylabel(ylabel)
    axes[1].set_title("Fit scan")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Plot: {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit MOND a0 to the SPARC RAR table and compare with CCDR."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the SPARC RAR table even if it is cached.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to a local RAR.mrt file.",
    )
    parser.add_argument(
        "--h0",
        type=float,
        default=H0_KMS_MPC_DEFAULT,
        help="H0 in km/s/Mpc for the CCDR comparison. Default: 67.4",
    )
    parser.add_argument(
        "--lower-bound",
        type=float,
        default=1e-12,
        help="Lower fit bound for a0 in m/s^2.",
    )
    parser.add_argument(
        "--upper-bound",
        type=float,
        default=1e-8,
        help="Upper fit bound for a0 in m/s^2.",
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Use unweighted log-space MSE instead of reduced chi^2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bounds = (args.lower_bound, args.upper_bound)
    if not (0 < bounds[0] < bounds[1]):
        raise SystemExit("Invalid bounds: require 0 < lower_bound < upper_bound")

    weighted = not args.unweighted
    a0_ccdr = compute_ccdr_a0(args.h0)

    print("=" * 72)
    print("P-MOND-2: SPARC RAR Test (Milgrom vs CCDR)")
    print("=" * 72)
    print(f"Milgrom prediction: a0 = {A0_MILGROM:.3e} m/s^2")
    print(f"CCDR prediction:    a0 = c*H0 = {a0_ccdr:.3e} m/s^2  (H0={args.h0:.2f})")
    print(f"Ratio CCDR / Milgrom: {a0_ccdr / A0_MILGROM:.3f}x")
    print(f"Objective: {'reduced chi^2 in log-space' if weighted else 'unweighted log-space MSE'}")
    print(f"Fit bounds: [{bounds[0]:.1e}, {bounds[1]:.1e}]")
    print()

    rar_path = download_rar(force=args.force_download, input_path=args.input)
    g_bar, g_obs, e_gobs_dex, raw = load_rar(rar_path)

    best_a0, best_score, hit_boundary = fit_a0(
        g_bar, g_obs, e_gobs_dex, bounds=bounds, weighted=weighted
    )

    print("[fit]")
    print(f"  Best-fit a0:      {best_a0:.6e} m/s^2")
    if weighted:
        print(f"  Reduced chi^2:    {best_score:.6f}")
    else:
        print(f"  Log-space MSE:    {best_score:.6f}")
    print(f"  Hit boundary:     {hit_boundary}")
    if hit_boundary:
        print("  WARNING: fit is pinned near a search boundary; result is not trustworthy.")

    frac_dev_milgrom = abs(best_a0 - A0_MILGROM) / A0_MILGROM
    frac_dev_ccdr = abs(best_a0 - a0_ccdr) / a0_ccdr

    mse_at_best = mse_log10(best_a0, g_bar, g_obs)
    mse_at_milgrom = mse_log10(A0_MILGROM, g_bar, g_obs)
    mse_at_ccdr = mse_log10(a0_ccdr, g_bar, g_obs)
    redchi2_at_best = reduced_chi2_log10(best_a0, g_bar, g_obs, e_gobs_dex)
    redchi2_at_milgrom = reduced_chi2_log10(A0_MILGROM, g_bar, g_obs, e_gobs_dex)
    redchi2_at_ccdr = reduced_chi2_log10(a0_ccdr, g_bar, g_obs, e_gobs_dex)

    print()
    print("Comparison to predictions")
    print(f"  Fractional deviation from Milgrom: {frac_dev_milgrom:.3%}")
    print(f"  Fractional deviation from CCDR:    {frac_dev_ccdr:.3%}")

    print()
    print("Fit quality at reference values")
    print(f"  Milgrom: reduced chi^2 = {redchi2_at_milgrom:.6f}, MSE = {mse_at_milgrom:.6f}")
    print(f"  CCDR:    reduced chi^2 = {redchi2_at_ccdr:.6f}, MSE = {mse_at_ccdr:.6f}")
    print(f"  Best:    reduced chi^2 = {redchi2_at_best:.6f}, MSE = {mse_at_best:.6f}")

    if hit_boundary:
        verdict = "FIT INVALID: optimizer hit a boundary"
    elif frac_dev_milgrom < 0.10:
        verdict = "MILGROM CONFIRMED, CCDR EXCLUDED at z=0"
    elif frac_dev_ccdr < 0.10:
        verdict = "CCDR CONFIRMED at z=0"
    else:
        verdict = "INCONCLUSIVE — best fit matches neither prediction closely"

    print()
    print("VERDICT")
    print(f"  {verdict}")
    print("  Note: SPARC constrains the local z≈0 value only.")

    out = {
        "data_url_primary": URL_PRIMARY,
        "data_url_fallback": URL_FALLBACK,
        "n_points": int(len(g_bar)),
        "fit_bounds": {"lower": bounds[0], "upper": bounds[1]},
        "weighted_fit": weighted,
        "best_fit_a0": best_a0,
        "predictions": {
            "milgrom_a0": A0_MILGROM,
            "ccdr_a0_z0": a0_ccdr,
            "h0_km_s_mpc": args.h0,
            "ratio_ccdr_to_milgrom": a0_ccdr / A0_MILGROM,
        },
        "comparison": {
            "frac_dev_from_milgrom": frac_dev_milgrom,
            "frac_dev_from_ccdr": frac_dev_ccdr,
            "reduced_chi2_at_best": redchi2_at_best,
            "reduced_chi2_at_milgrom": redchi2_at_milgrom,
            "reduced_chi2_at_ccdr": redchi2_at_ccdr,
            "mse_at_best": mse_at_best,
            "mse_at_milgrom": mse_at_milgrom,
            "mse_at_ccdr": mse_at_ccdr,
        },
        "hit_boundary": hit_boundary,
        "verdict": verdict,
    }

    out_json = DATA_DIR / "result_v2.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved: {out_json}")

    try:
        make_plot(
            DATA_DIR / "P_MOND_2_v2.png",
            g_bar,
            g_obs,
            best_a0,
            A0_MILGROM,
            a0_ccdr,
            weighted,
            e_gobs_dex,
        )
    except ImportError:
        print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
