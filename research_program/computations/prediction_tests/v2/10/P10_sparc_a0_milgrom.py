#!/usr/bin/env python3
"""
P10_sparc_a0_milgrom.py

Standalone implementation of CCDR Test 10:
fit the local SPARC radial-acceleration relation (RAR) acceleration scale a0
from public data downloaded automatically by the script.

Public sources used by default:
- SPARC RAR all-data table: https://astroweb.case.edu/SPARC/RAR.mrt
- SPARC parent catalog:     https://astroweb.case.edu/SPARC/Table1.mrt

Outputs:
- result.json
- rar_fit.png
- fit_curve.csv

This script requires: numpy, scipy, matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import brentq, minimize_scalar

DEFAULT_RAR_URL = "https://astroweb.case.edu/SPARC/RAR.mrt"
DEFAULT_PARENT_URL = "https://astroweb.case.edu/SPARC/SPARC_Lelli2016c.mrt"
M_PER_MPC = 3.085677581491367e22
C_LIGHT = 299792458.0  # m/s
MILGROM_A0 = 1.20e-10  # m/s^2
DEFAULT_H0 = 67.4  # km/s/Mpc


@dataclass
class RARData:
    log10_gbar: np.ndarray
    err_log10_gbar: np.ndarray
    log10_gobs: np.ndarray
    err_log10_gobs: np.ndarray
    source_path: Path

    @property
    def n(self) -> int:
        return int(self.log10_gbar.size)


@dataclass
class FitSummary:
    model_name: str
    a0_fit: float
    a0_sigma_stat: float
    a0_sigma_sys: float
    a0_sigma_total: float
    chi2: float
    chi2_reduced: float
    dof: int
    dev_from_milgrom_pct: float
    dev_from_cH0_pct: float


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_if_needed(url: str, destination: Path, refresh: bool = False) -> Path:
    ensure_dir(destination.parent)
    if refresh or not destination.exists():
        eprint(f"[data] Downloading {url}")
        with urllib.request.urlopen(url) as response:
            destination.write_bytes(response.read())
    else:
        eprint(f"[data] Using cached {destination.name}")
    return destination


def load_rar_mrt(path: Path) -> RARData:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    data_started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped and set(stripped) <= set("-=:"):
            continue
        if stripped[:1].isdigit() or stripped[:1] in "+-":
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    rows.append(tuple(float(x) for x in parts[:4]))
                    data_started = True
                    continue
                except ValueError:
                    pass
        elif data_started:
            break

    if not rows:
        raise RuntimeError(f"No numeric RAR rows found in {path}")

    arr = np.asarray(rows, dtype=float)
    log10_gbar = arr[:, 0]
    err_log10_gbar = arr[:, 1]
    log10_gobs = arr[:, 2]
    err_log10_gobs = arr[:, 3]

    mask = np.isfinite(arr).all(axis=1)
    log10_gbar = log10_gbar[mask]
    err_log10_gbar = err_log10_gbar[mask]
    log10_gobs = log10_gobs[mask]
    err_log10_gobs = err_log10_gobs[mask]

    if log10_gbar.size == 0:
        raise RuntimeError(f"All rows were filtered out from {path}")

    return RARData(
        log10_gbar=log10_gbar,
        err_log10_gbar=err_log10_gbar,
        log10_gobs=log10_gobs,
        err_log10_gobs=err_log10_gobs,
        source_path=path,
    )


def cH0_acceleration(h0_km_s_mpc: float) -> float:
    h0_si = (h0_km_s_mpc * 1000.0) / M_PER_MPC
    return C_LIGHT * h0_si


def model_published_rar(gbar: np.ndarray, a0: float) -> np.ndarray:
    x = np.maximum(gbar / a0, 1e-300)
    denom = 1.0 - np.exp(-np.sqrt(x))
    denom = np.maximum(denom, 1e-300)
    return gbar / denom


def model_simple_mu(gbar: np.ndarray, a0: float) -> np.ndarray:
    # MOND simple-mu interpolating function: mu(y)=y/(1+y)
    # Solving g_N = g * mu(g/a0) gives:
    return 0.5 * (gbar + np.sqrt(gbar * gbar + 4.0 * gbar * a0))


def model_log10_gobs(log10_gbar: np.ndarray, a0: float, model_name: str) -> np.ndarray:
    gbar = np.power(10.0, log10_gbar)
    if model_name == "published_rar":
        gobs = model_published_rar(gbar, a0)
    elif model_name == "simple_mu":
        gobs = model_simple_mu(gbar, a0)
    else:
        raise ValueError(f"Unknown model_name={model_name}")
    return np.log10(np.maximum(gobs, 1e-300))


def dy_dx(log10_gbar: np.ndarray, a0: float, model_name: str, eps: float = 1e-5) -> np.ndarray:
    yp = model_log10_gobs(log10_gbar + eps, a0, model_name)
    ym = model_log10_gobs(log10_gbar - eps, a0, model_name)
    return (yp - ym) / (2.0 * eps)


def sigma_eff_dex(data: RARData, a0: float, model_name: str, include_x_errors: bool = True) -> np.ndarray:
    sig_y = data.err_log10_gobs
    if not include_x_errors:
        return np.maximum(sig_y, 1e-8)
    slope = dy_dx(data.log10_gbar, a0, model_name)
    sig = np.sqrt(sig_y * sig_y + (slope * data.err_log10_gbar) ** 2)
    return np.maximum(sig, 1e-8)


def chi2_for_a0(a0: float, data: RARData, model_name: str, include_x_errors: bool = True) -> float:
    if not np.isfinite(a0) or a0 <= 0.0:
        return np.inf
    y_model = model_log10_gobs(data.log10_gbar, a0, model_name)
    sigma = sigma_eff_dex(data, a0, model_name, include_x_errors=include_x_errors)
    resid = (data.log10_gobs - y_model) / sigma
    chi2 = float(np.sum(resid * resid))
    if not np.isfinite(chi2):
        return np.inf
    return chi2


def fit_a0(data: RARData, model_name: str, include_x_errors: bool = True) -> Tuple[float, float, int]:
    bounds = (1e-12, 2e-9)
    objective = lambda a0: chi2_for_a0(a0, data, model_name, include_x_errors=include_x_errors)
    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-15, "maxiter": 2000})
    if not result.success:
        raise RuntimeError(f"Fit failed for {model_name}: {result}")
    a0_best = float(result.x)
    chi2_best = float(result.fun)
    dof = data.n - 1
    return a0_best, chi2_best, dof


def delta_chi2_crossing(
    data: RARData,
    model_name: str,
    a0_best: float,
    chi2_best: float,
    target_delta: float = 1.0,
    side: str = "left",
    include_x_errors: bool = True,
) -> float | None:
    target = chi2_best + target_delta

    def f(log10_a0: float) -> float:
        a0 = 10.0 ** log10_a0
        return chi2_for_a0(a0, data, model_name, include_x_errors=include_x_errors) - target

    log_best = math.log10(a0_best)
    lo_bound = math.log10(1e-12)
    hi_bound = math.log10(2e-9)

    if side == "left":
        scan = np.linspace(log_best - 0.001, lo_bound, 400)
    else:
        scan = np.linspace(log_best + 0.001, hi_bound, 400)

    prev_x = log_best
    prev_f = f(prev_x)
    for x in scan:
        fx = f(float(x))
        if np.sign(prev_f) != np.sign(fx):
            try:
                return float(10.0 ** brentq(f, prev_x, float(x), maxiter=500))
            except ValueError:
                return None
        prev_x, prev_f = float(x), fx
    return None


def compute_stat_sigma(data: RARData, model_name: str, a0_best: float, chi2_best: float, include_x_errors: bool = True) -> float:
    left = delta_chi2_crossing(data, model_name, a0_best, chi2_best, side="left", include_x_errors=include_x_errors)
    right = delta_chi2_crossing(data, model_name, a0_best, chi2_best, side="right", include_x_errors=include_x_errors)
    errs = []
    if left is not None:
        errs.append(a0_best - left)
    if right is not None:
        errs.append(right - a0_best)
    if not errs:
        return float("nan")
    return float(np.mean(errs))


def summarize_fit(data: RARData, h0_km_s_mpc: float) -> Tuple[FitSummary, dict]:
    cH0 = cH0_acceleration(h0_km_s_mpc)

    # Primary estimate: published RAR fit in log space using the observed-acceleration
    # uncertainties, matching the simplest one-parameter implementation of the test.
    a0_pub, chi2_pub, dof = fit_a0(data, "published_rar", include_x_errors=False)
    stat_sigma = compute_stat_sigma(data, "published_rar", a0_pub, chi2_pub, include_x_errors=False)

    # Systematic checks: include x-errors, and swap to the simple-mu interpolating function.
    a0_pub_with_x, chi2_pub_with_x, _ = fit_a0(data, "published_rar", include_x_errors=True)
    a0_simple, chi2_simple, _ = fit_a0(data, "simple_mu", include_x_errors=False)

    sys_components = [abs(a0_pub - a0_pub_with_x), abs(a0_pub - a0_simple)]
    sys_sigma = max(sys_components)
    total_sigma = float(np.sqrt(stat_sigma ** 2 + sys_sigma ** 2)) if np.isfinite(stat_sigma) else sys_sigma

    summary = FitSummary(
        model_name="published_rar",
        a0_fit=a0_pub,
        a0_sigma_stat=stat_sigma,
        a0_sigma_sys=sys_sigma,
        a0_sigma_total=total_sigma,
        chi2=chi2_pub,
        chi2_reduced=chi2_pub / dof,
        dof=dof,
        dev_from_milgrom_pct=100.0 * (a0_pub - MILGROM_A0) / MILGROM_A0,
        dev_from_cH0_pct=100.0 * (a0_pub - cH0) / cH0,
    )

    aux = {
        "a0_fit_published_with_x_errors": a0_pub_with_x,
        "chi2_published_with_x_errors": chi2_pub_with_x,
        "a0_fit_simple_mu": a0_simple,
        "chi2_simple_mu": chi2_simple,
        "cH0_a0": cH0,
        "pass_v6_local": abs(summary.dev_from_milgrom_pct) <= 5.0,
        "within_5pct_cH0": abs(summary.dev_from_cH0_pct) <= 5.0,
        "between_milgrom_and_cH0": min(MILGROM_A0, cH0) <= a0_pub <= max(MILGROM_A0, cH0),
        "systematic_components": {
            "published_with_x_errors_shift": abs(a0_pub - a0_pub_with_x),
            "simple_mu_shift": abs(a0_pub - a0_simple),
        },
    }
    return summary, aux


def make_plot(data: RARData, summary: FitSummary, aux: dict, outpath: Path) -> None:
    x = data.log10_gbar
    y = data.log10_gobs
    order = np.argsort(x)
    x_sorted = x[order]

    cH0 = aux["cH0_a0"]
    y_best = model_log10_gobs(x_sorted, summary.a0_fit, "published_rar")
    y_milgrom = model_log10_gobs(x_sorted, MILGROM_A0, "published_rar")
    y_cH0 = model_log10_gobs(x_sorted, cH0, "published_rar")
    resid_best = y - model_log10_gobs(x, summary.a0_fit, "published_rar")

    fig = plt.figure(figsize=(8.5, 7.0))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(x, y, s=8, alpha=0.28, label=f"SPARC RAR ({data.n} points)")
    ax1.plot(x_sorted, y_best, lw=2.2, label=fr"Best fit: $a_0={summary.a0_fit:.3e}$ m s$^{{-2}}$")
    ax1.plot(x_sorted, y_milgrom, lw=1.6, ls="--", label=fr"Milgrom $a_0={MILGROM_A0:.2e}$")
    ax1.plot(x_sorted, y_cH0, lw=1.6, ls=":", label=fr"$cH_0={cH0:.2e}$")
    ax1.plot(x_sorted, x_sorted, lw=1.0, ls="-.", label=r"$g_{obs}=g_{bar}$")
    ax1.set_xlabel(r"$\log_{10} g_{bar}$ [m s$^{-2}$]")
    ax1.set_ylabel(r"$\log_{10} g_{obs}$ [m s$^{-2}$]")
    ax1.legend(loc="best", fontsize=9)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.scatter(x, resid_best, s=8, alpha=0.28)
    ax2.axhline(0.0, lw=1.2, ls="--")
    ax2.set_xlabel(r"$\log_{10} g_{bar}$ [m s$^{-2}$]")
    ax2.set_ylabel("Residual [dex]")

    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def write_csv_curve(data: RARData, summary: FitSummary, aux: dict, outpath: Path) -> None:
    x_grid = np.linspace(np.min(data.log10_gbar), np.max(data.log10_gbar), 400)
    cH0 = aux["cH0_a0"]
    rows = np.column_stack([
        x_grid,
        model_log10_gobs(x_grid, summary.a0_fit, "published_rar"),
        model_log10_gobs(x_grid, MILGROM_A0, "published_rar"),
        model_log10_gobs(x_grid, cH0, "published_rar"),
    ])
    header = "log10_gbar,log10_gobs_best,log10_gobs_milgrom,log10_gobs_cH0"
    np.savetxt(outpath, rows, delimiter=",", header=header, comments="")


def build_result_payload(summary: FitSummary, aux: dict, rar_path: Path, parent_path: Path, h0_km_s_mpc: float) -> dict:
    return {
        "test_name": "SPARC a0 local vs cH0 asymptotic",
        "data_sources": {
            "rar_all_data": str(rar_path),
            "parent_catalog": str(parent_path),
        },
        "fit_model": "published_rar",
        "alternate_systematic_model": "simple_mu",
        "h0_km_s_mpc": h0_km_s_mpc,
        "a0_fit": summary.a0_fit,
        "a0_fit_sigma": summary.a0_sigma_total,
        "a0_fit_sigma_stat": summary.a0_sigma_stat,
        "a0_fit_sigma_sys": summary.a0_sigma_sys,
        "chi2": summary.chi2,
        "chi2_reduced": summary.chi2_reduced,
        "degrees_of_freedom": summary.dof,
        "milgrom_a0": MILGROM_A0,
        "cH0_a0": aux["cH0_a0"],
        "dev_from_milgrom_pct": summary.dev_from_milgrom_pct,
        "dev_from_cH0_pct": summary.dev_from_cH0_pct,
        "a0_fit_published_with_x_errors": aux["a0_fit_published_with_x_errors"],
        "chi2_published_with_x_errors": aux["chi2_published_with_x_errors"],
        "a0_fit_simple_mu": aux["a0_fit_simple_mu"],
        "chi2_simple_mu": aux["chi2_simple_mu"],
        "systematic_components": aux["systematic_components"],
        "pass_v6_local": aux["pass_v6_local"],
        "within_5pct_cH0": aux["within_5pct_cH0"],
        "between_milgrom_and_cH0": aux["between_milgrom_and_cH0"],
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit SPARC RAR a0 from public data")
    p.add_argument("--outdir", default="out_p10", help="Output directory")
    p.add_argument("--data-cache-dir", default="p10_public_data", help="Directory for downloaded public data")
    p.add_argument("--refresh-data", action="store_true", help="Re-download public data")
    p.add_argument("--rar-url", default=DEFAULT_RAR_URL, help="Public URL for SPARC RAR table")
    p.add_argument("--parent-url", default=DEFAULT_PARENT_URL, help="Public URL for SPARC parent catalog")
    p.add_argument("--h0", type=float, default=DEFAULT_H0, help="H0 in km/s/Mpc for cH0 comparison")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    cache_dir = Path(args.data_cache_dir)
    ensure_dir(outdir)
    ensure_dir(cache_dir)

    rar_path = download_if_needed(args.rar_url, cache_dir / "RAR.mrt", refresh=args.refresh_data)
    parent_path = download_if_needed(args.parent_url, cache_dir / "Table1.mrt", refresh=args.refresh_data)

    eprint("[fit] Loading SPARC RAR data...")
    data = load_rar_mrt(rar_path)
    eprint(f"[fit] Loaded {data.n} RAR points")

    eprint("[fit] Fitting published RAR model and simple-mu cross-check...")
    summary, aux = summarize_fit(data, h0_km_s_mpc=args.h0)

    payload = build_result_payload(summary, aux, rar_path, parent_path, h0_km_s_mpc=args.h0)
    (outdir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv_curve(data, summary, aux, outdir / "fit_curve.csv")
    make_plot(data, summary, aux, outdir / "rar_fit.png")

    eprint("[done] Wrote:")
    for name in ["result.json", "fit_curve.csv", "rar_fit.png"]:
        eprint(f"   {outdir / name}")
    eprint(
        f"[result] a0 = {summary.a0_fit:.4e} +/- {summary.a0_sigma_total:.2e} m/s^2 "
        f"(stat {summary.a0_sigma_stat:.2e}, sys {summary.a0_sigma_sys:.2e})"
    )
    eprint(
        f"[result] dev from Milgrom = {summary.dev_from_milgrom_pct:+.2f}%, "
        f"dev from cH0 = {summary.dev_from_cH0_pct:+.2f}%"
    )
    eprint(f"[result] pass_v6_local = {aux['pass_v6_local']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
