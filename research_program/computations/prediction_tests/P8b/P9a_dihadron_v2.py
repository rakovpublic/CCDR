#!/usr/bin/env python3
"""
P9a_dihadron_v2.py

Fit a Fourier series to a projected dihadron angular-correlation function
C(Δφ) and test whether the cos(6Δφ) coefficient is significantly nonzero.

Supported inputs:
  1) local CSV with columns like: dphi, corr, err
  2) HEPData JSON/YAML via a HEPData table/record URL
  3) --demo synthetic dataset for smoke testing

This is a practical implementation of the P9a note: it does not attempt
full event mixing from raw CERN Open Data inside one small script. Instead,
it works directly with tabulated projected correlation data, which is the
natural input for a fast v6 / cos(6Δφ) scan.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


OUT_DIR_DEFAULT = Path("data/p9a_dihadron")
HEPDATA_FORMATS = {"json", "yaml"}


@dataclass
class FitSummary:
    a0: float
    a0_err: float
    harmonics: Dict[str, float]
    harmonic_errors: Dict[str, float]
    chi2: float
    ndof: int
    chi2_ndof: float
    v2_est: Optional[float]
    v6_est: Optional[float]
    v6_err: Optional[float]
    v6_significance_sigma: Optional[float]
    v6_qcd_baseline: Optional[float]
    v6_excess_above_qcd: Optional[float]
    v6_excess_sigma: Optional[float]


# ------------------------------------------------------------
# Input loading
# ------------------------------------------------------------

def fetch_url(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (P9a-dihadron-v2)",
            "Accept": "application/json, application/x-yaml, text/yaml, text/plain, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def hepdata_download_url(url: str, fmt: str, table: Optional[str] = None) -> str:
    if fmt not in HEPDATA_FORMATS:
        raise ValueError(f"Unsupported HEPData format: {fmt}")

    parsed = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qs(parsed.query)
    q["format"] = [fmt]
    if table:
        q["table"] = [table]
    new_query = urllib.parse.urlencode(q, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _import_yaml():
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "YAML support requires PyYAML. Install with: pip install pyyaml"
        ) from exc
    return yaml


def parse_hepdata_json_or_yaml(payload: bytes, fmt: str) -> dict:
    if fmt == "json":
        return json.loads(payload.decode("utf-8"))
    yaml = _import_yaml()
    return yaml.safe_load(payload.decode("utf-8"))


def _pick_first_numeric_values(values: Sequence[dict]) -> np.ndarray:
    out: List[float] = []
    for row in values:
        if isinstance(row, dict):
            value = row.get("value")
            if value is None:
                low = row.get("low")
                high = row.get("high")
                if low is not None and high is not None:
                    value = 0.5 * (float(low) + float(high))
            if value is not None:
                try:
                    out.append(float(value))
                except Exception:
                    pass
    if not out:
        raise RuntimeError("Could not extract numeric values from HEPData table")
    return np.asarray(out, dtype=float)


def _pick_errors(values: Sequence[dict]) -> Optional[np.ndarray]:
    errs: List[float] = []
    found_any = False
    for row in values:
        err_val = None
        if isinstance(row, dict) and "errors" in row:
            for err in row["errors"] or []:
                if isinstance(err, dict):
                    if "symerror" in err:
                        try:
                            err_val = float(err["symerror"])
                            break
                        except Exception:
                            pass
                    elif "asymerror" in err and isinstance(err["asymerror"], dict):
                        try:
                            plus = abs(float(err["asymerror"].get("plus", np.nan)))
                            minus = abs(float(err["asymerror"].get("minus", np.nan)))
                            if np.isfinite(plus) and np.isfinite(minus):
                                err_val = 0.5 * (plus + minus)
                                break
                        except Exception:
                            pass
        if err_val is None:
            errs.append(np.nan)
        else:
            found_any = True
            errs.append(float(err_val))
    if not found_any:
        return None
    arr = np.asarray(errs, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    fill = np.nanmedian(arr[finite])
    arr[~finite] = fill if np.isfinite(fill) and fill > 0 else 1.0
    return arr


def load_hepdata_projection(url: str, table: Optional[str] = None, fmt: str = "json") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    download_url = hepdata_download_url(url, fmt=fmt, table=table)
    payload = fetch_url(download_url)
    data = parse_hepdata_json_or_yaml(payload, fmt)

    if "independent_variables" not in data or "dependent_variables" not in data:
        raise RuntimeError("Unexpected HEPData payload structure")

    indep = data["independent_variables"][0]
    dep = data["dependent_variables"][0]

    phi = _pick_first_numeric_values(indep["values"])
    corr = _pick_first_numeric_values(dep["values"])
    err = _pick_errors(dep["values"])

    meta = {
        "source": "hepdata",
        "download_url": download_url,
        "x_name": indep.get("header", {}).get("name"),
        "y_name": dep.get("header", {}).get("name"),
        "n_points": int(len(phi)),
    }
    return phi, corr, err, meta


def load_csv_projection(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    import csv

    aliases = {
        "phi": {"phi", "dphi", "delta_phi", "deltaphi", "x"},
        "corr": {"c", "corr", "correlation", "y", "yield", "value"},
        "err": {"err", "error", "sigma", "yerr", "c_err", "corr_err"},
    }

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no header row")
        fields = {name.strip().lower(): name for name in reader.fieldnames}

        def find_col(kind: str) -> Optional[str]:
            for alias in aliases[kind]:
                if alias in fields:
                    return fields[alias]
            return None

        phi_col = find_col("phi")
        corr_col = find_col("corr")
        err_col = find_col("err")
        if phi_col is None or corr_col is None:
            raise RuntimeError(
                f"Could not find phi/corr columns in CSV. Got: {reader.fieldnames}"
            )

        phi_vals: List[float] = []
        corr_vals: List[float] = []
        err_vals: List[float] = []
        for row in reader:
            try:
                phi_vals.append(float(row[phi_col]))
                corr_vals.append(float(row[corr_col]))
                if err_col and row.get(err_col, "") != "":
                    err_vals.append(float(row[err_col]))
                else:
                    err_vals.append(np.nan)
            except Exception:
                continue

    phi = np.asarray(phi_vals, dtype=float)
    corr = np.asarray(corr_vals, dtype=float)
    err_arr = np.asarray(err_vals, dtype=float)
    err = None if not np.isfinite(err_arr).any() else np.where(np.isfinite(err_arr), err_arr, np.nanmedian(err_arr[np.isfinite(err_arr)]))
    meta = {
        "source": "csv",
        "path": str(path),
        "n_points": int(len(phi)),
    }
    return phi, corr, err, meta


def demo_dataset(n_points: int = 72, noise_sigma: float = 3e-4, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    phi = np.linspace(-math.pi, math.pi, n_points, endpoint=False)
    true = (
        0.004
        + 0.010 * np.cos(phi)
        + 0.0035 * np.cos(2 * phi)
        + 0.0008 * np.cos(3 * phi)
        + 0.0005 * np.cos(4 * phi)
        + 0.0002 * np.cos(5 * phi)
        + 0.0010 * np.cos(6 * phi)
    )
    err = np.full_like(phi, noise_sigma)
    corr = true + rng.normal(0.0, noise_sigma, size=n_points)
    meta = {"source": "demo", "n_points": int(n_points), "noise_sigma": float(noise_sigma)}
    return phi, corr, err, meta


# ------------------------------------------------------------
# Fitting
# ------------------------------------------------------------

def to_radians_if_needed(phi: np.ndarray) -> Tuple[np.ndarray, str]:
    finite_max = np.nanmax(np.abs(phi))
    if finite_max > 2 * math.pi + 0.25:
        return np.deg2rad(phi), "degrees"
    return phi.astype(float), "radians"


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    return ((phi + math.pi) % (2 * math.pi)) - math.pi


def design_matrix(phi_rad: np.ndarray, max_harmonic: int) -> np.ndarray:
    cols = [np.ones_like(phi_rad)]
    for n in range(1, max_harmonic + 1):
        cols.append(np.cos(n * phi_rad))
    return np.column_stack(cols)


def weighted_linear_fit(phi_rad: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray], max_harmonic: int) -> FitSummary:
    X = design_matrix(phi_rad, max_harmonic)

    if yerr is None:
        sigma = np.full_like(y, max(np.std(y) * 0.05, 1e-6))
    else:
        sigma = np.asarray(yerr, dtype=float).copy()
        bad = ~np.isfinite(sigma) | (sigma <= 0)
        if bad.any():
            fill = np.nanmedian(sigma[~bad]) if (~bad).any() else max(np.std(y) * 0.05, 1e-6)
            sigma[bad] = fill

    W = np.diag(1.0 / sigma**2)
    XT_W = X.T @ W
    cov = np.linalg.inv(XT_W @ X)
    beta = cov @ XT_W @ y
    y_fit = X @ beta
    chi2 = float(np.sum(((y - y_fit) / sigma) ** 2))
    ndof = int(len(y) - len(beta))
    chi2_ndof = chi2 / ndof if ndof > 0 else float("nan")

    errs = np.sqrt(np.diag(cov))
    a0 = float(beta[0])
    a0_err = float(errs[0])

    harmonics = {f"a{n}": float(beta[n]) for n in range(1, max_harmonic + 1)}
    harmonic_errors = {f"a{n}_err": float(errs[n]) for n in range(1, max_harmonic + 1)}

    a2 = harmonics.get("a2")
    a6 = harmonics.get("a6")
    a6_err = harmonic_errors.get("a6_err")
    v2_est = None
    v6_est = None
    v6_qcd_baseline = None
    v6_excess = None
    v6_sigma = None
    v6_excess_sigma = None
    if a0 != 0:
        if a2 is not None:
            v2_est = a2 / a0
        if a6 is not None:
            v6_est = a6 / a0
            if a6_err is not None:
                v6_sigma = abs(a6) / a6_err if a6_err > 0 else None
        if v2_est is not None and v6_est is not None:
            v6_qcd_baseline = v2_est**3
            v6_excess = v6_est - v6_qcd_baseline
            if a6_err is not None and a6_err > 0:
                v6_err = abs(a6_err / a0)
                v6_excess_sigma = abs(v6_excess) / v6_err if v6_err > 0 else None
            else:
                v6_err = None
        else:
            v6_err = None
    else:
        v6_err = None

    return FitSummary(
        a0=a0,
        a0_err=a0_err,
        harmonics=harmonics,
        harmonic_errors=harmonic_errors,
        chi2=chi2,
        ndof=ndof,
        chi2_ndof=chi2_ndof,
        v2_est=v2_est,
        v6_est=v6_est,
        v6_err=v6_err,
        v6_significance_sigma=v6_sigma,
        v6_qcd_baseline=v6_qcd_baseline,
        v6_excess_above_qcd=v6_excess,
        v6_excess_sigma=v6_excess_sigma,
    )


def classify_v6(summary: FitSummary, ccdr_raw_target: float = 1e-3) -> str:
    a6 = summary.harmonics.get("a6")
    a6_err = summary.harmonic_errors.get("a6_err")
    if a6 is None or a6_err is None or a6_err <= 0:
        return "INCONCLUSIVE: missing a6 uncertainty"

    sig = abs(a6) / a6_err
    if sig < 2:
        return "NULL: a6 consistent with zero"
    if a6 > 0 and abs(a6) >= 0.5 * ccdr_raw_target:
        if sig >= 3:
            return "POSITIVE: positive a6 at the ~10^-3 scale"
        return "HINT: positive a6 near the target scale"
    if a6 > 0:
        return "SMALL SIGNAL: positive a6 but below the ~10^-3 target scale"
    return "WRONG SIGN: a6 is significantly negative"


# ------------------------------------------------------------
# Plotting / saving
# ------------------------------------------------------------

def evaluate_series(phi_rad: np.ndarray, summary: FitSummary) -> np.ndarray:
    y = np.full_like(phi_rad, summary.a0, dtype=float)
    for name, coeff in summary.harmonics.items():
        n = int(name[1:])
        y += coeff * np.cos(n * phi_rad)
    return y


def save_plot(phi_rad: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray], summary: FitSummary, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    phi_grid = np.linspace(-math.pi, math.pi, 600)
    y_fit = evaluate_series(phi_grid, summary)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    if yerr is not None:
        axes[0].errorbar(phi_rad, y, yerr=yerr, fmt="o", markersize=4, alpha=0.8, label="data")
    else:
        axes[0].plot(phi_rad, y, "o", markersize=4, alpha=0.8, label="data")
    axes[0].plot(phi_grid, y_fit, lw=2, label="Fourier fit")
    axes[0].set_xlabel(r"$\Delta\phi$ [rad]")
    axes[0].set_ylabel(r"$C(\Delta\phi)$")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    ns = []
    coeffs = []
    errs = []
    for name, coeff in summary.harmonics.items():
        ns.append(int(name[1:]))
        coeffs.append(coeff)
        errs.append(summary.harmonic_errors[f"{name}_err"])
    axes[1].errorbar(ns, coeffs, yerr=errs, fmt="o")
    axes[1].axhline(0.0, color="k", ls="--", alpha=0.6)
    axes[1].set_xlabel("harmonic n")
    axes[1].set_ylabel(r"$a_n$")
    axes[1].set_title("Fourier coefficients")
    axes[1].grid(True, alpha=0.3)

    text = []
    if summary.v2_est is not None:
        text.append(f"v2≈a2/a0={summary.v2_est:.4g}")
    if summary.v6_est is not None:
        text.append(f"v6≈a6/a0={summary.v6_est:.4g}")
    if summary.v6_qcd_baseline is not None:
        text.append(f"v2^3={summary.v6_qcd_baseline:.4g}")
    if summary.v6_excess_above_qcd is not None:
        text.append(f"excess={summary.v6_excess_above_qcd:.4g}")
    if text:
        axes[1].text(0.02, 0.98, "\n".join(text), transform=axes[1].transAxes, va="top")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit Fourier harmonics to a dihadron angular-correlation projection and test cos(6Δφ).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Local CSV with columns like dphi,corr,err")
    src.add_argument("--hepdata-url", type=str, help="HEPData table or record URL")
    src.add_argument("--demo", action="store_true", help="Run on a synthetic demo dataset")
    p.add_argument("--table", type=str, default=None, help="HEPData table name, e.g. 'Table 28' if using a record URL")
    p.add_argument("--hepdata-format", choices=["json", "yaml"], default="json")
    p.add_argument("--max-harmonic", type=int, default=6)
    p.add_argument("--outdir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--prefix", type=str, default="p9a")
    p.add_argument("--skip-plot", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.max_harmonic < 1:
        raise SystemExit("--max-harmonic must be >= 1")

    if args.demo:
        phi, corr, err, meta = demo_dataset()
    elif args.csv is not None:
        phi, corr, err, meta = load_csv_projection(args.csv)
    else:
        phi, corr, err, meta = load_hepdata_projection(args.hepdata_url, table=args.table, fmt=args.hepdata_format)

    if len(phi) < args.max_harmonic + 2:
        raise RuntimeError("Too few points for requested harmonic fit")

    phi_rad, units = to_radians_if_needed(phi)
    phi_rad = wrap_phi(phi_rad)
    order = np.argsort(phi_rad)
    phi_rad = phi_rad[order]
    corr = corr[order]
    if err is not None:
        err = err[order]

    summary = weighted_linear_fit(phi_rad, corr, err, max_harmonic=args.max_harmonic)
    verdict = classify_v6(summary)

    args.outdir.mkdir(parents=True, exist_ok=True)
    result_path = args.outdir / f"{args.prefix}_result.json"
    plot_path = args.outdir / f"{args.prefix}_fit.png"

    result = {
        "input_meta": meta,
        "phi_input_units": units,
        "n_points": int(len(phi_rad)),
        "max_harmonic": int(args.max_harmonic),
        "fit": asdict(summary),
        "verdict": verdict,
        "interpretation": {
            "ccdr_raw_a6_target": 1e-3,
            "note": "Primary test here is the raw fitted a6 coefficient in C(Δφ)=a0+...+a6 cos(6Δφ). Normalised v_n-like quantities are reported only as heuristics.",
        },
    }
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)

    if not args.skip_plot:
        title = f"P9a Fourier fit ({meta.get('source', 'unknown')})"
        save_plot(phi_rad, corr, err, summary, plot_path, title)

    print("=" * 72)
    print("P9a: Hadronic Angular Correlations — cos(6Δφ) scan")
    print("=" * 72)
    print(f"[input] source={meta.get('source')}  n_points={len(phi_rad)}  units={units}")
    if meta.get("download_url"):
        print(f"[input] fetched from: {meta['download_url']}")
    print(f"[fit] a0 = {summary.a0:+.6e} ± {summary.a0_err:.2e}")
    for n in range(1, args.max_harmonic + 1):
        a = summary.harmonics[f"a{n}"]
        ea = summary.harmonic_errors[f"a{n}_err"]
        sig = abs(a) / ea if ea > 0 else float("nan")
        print(f"[fit] a{n} = {a:+.6e} ± {ea:.2e}  ({sig:.2f}σ)")
    print(f"[fit] chi2/ndof = {summary.chi2:.2f}/{summary.ndof} = {summary.chi2_ndof:.3f}")
    if summary.v2_est is not None:
        print(f"[derived] v2 ≈ a2/a0 = {summary.v2_est:+.6e}")
    if summary.v6_est is not None:
        print(f"[derived] v6 ≈ a6/a0 = {summary.v6_est:+.6e}")
    if summary.v6_qcd_baseline is not None:
        print(f"[derived] QCD-like v6 baseline v2^3 = {summary.v6_qcd_baseline:+.6e}")
    if summary.v6_excess_above_qcd is not None:
        print(f"[derived] v6 excess above v2^3 = {summary.v6_excess_above_qcd:+.6e}")
    print(f"[verdict] {verdict}")
    print(f"[saved] {result_path}")
    if not args.skip_plot:
        print(f"[saved] {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
