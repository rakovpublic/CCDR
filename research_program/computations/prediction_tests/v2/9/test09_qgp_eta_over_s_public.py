#!/usr/bin/env python3
"""
Test 09 — QGP eta/s saturation and oscillation near T_c

Public-source implementation.

This script downloads the public Parkkila-Onnerstad-Kim 2021 paper
(arXiv:2106.05019 / Phys. Rev. C 104, 054904) and extracts the Table III
MAP parameters for the eta/s(T) parametrization:

    eta/s(T) = eta/s(T_c) + slope * (T - T_c) * (T / T_c)**crv

It then evaluates the curve at the temperatures requested by the test,
compares it to the KSS bound 1/(4*pi), and checks whether a dip-peak
model is preferred over monotonic alternatives in the 140-170 MeV region.

Design choice:
- The uploaded test note cites JETSCAPE / Phys. Rev. C 103, 054904 as the
  primary posterior source, but the most direct public source with an
  explicit eta/s(T) parametrization and published MAP values is
  Parkkila et al., Phys. Rev. C 104, 054904 (2021).
- To keep runtime practical and avoid multi-gigabyte downloads, this script
  uses that explicit published parametrization instead of downloading a full
  MCMC chain.

Outputs:
- result.json
- summary.txt
- eta_over_s_curve.png
- public_sources.json
- cached copy of the downloaded PDF

No user-supplied data files are required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


def ensure_package(import_name: str, pip_name: str | None = None) -> None:
    pip_name = pip_name or import_name
    try:
        importlib.import_module(import_name)
    except Exception:
        print(f"[setup] Installing missing dependency: {pip_name}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


# Pure-python PDF extraction library. Installed on demand.
ensure_package("pypdf", "pypdf")

import requests
import numpy as np
from matplotlib import pyplot as plt
from pypdf import PdfReader


ARXIV_ABS_URL = "https://arxiv.org/abs/2106.05019"
ARXIV_PDF_URL = "https://arxiv.org/pdf/2106.05019"
APS_DOI_URL = "https://doi.org/10.1103/PhysRevC.104.054904"
JETSCAPE_REF_URL = "https://doi.org/10.1103/PhysRevC.103.054904"
UPLOADED_TEST_NAME = "09_qgp_eta_over_s.md"


@dataclass
class MapParameters:
    norm: float
    eta_tc: float
    p: float
    eta_slope: float
    sigma_k: float
    eta_crv: float
    dmin3: float
    zeta_peak: float
    tau_fs: float
    zeta_max: float
    T_c: float
    zeta_width: float
    T_switch: float


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception as exc:
            raise RuntimeError(f"Failed to extract text from PDF page {idx}: {exc}") from exc
        pages.append(txt)
    return "\n\n".join(pages)


def normalize_text(text: str) -> str:
    # Make regexing more robust against PDF extraction quirks.
    text = text.replace("\xa0", " ")
    text = text.replace("−", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    return text


FLOAT_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+")


def parse_table_iii_map_parameters(text: str) -> MapParameters:
    norm_text = normalize_text(text)

    # Focus on the Table III block. The first 13 numeric values after the table
    # header correspond to the MAP values in row-major order.
    m = re.search(r"TABLE\s+III\.(.*?)(?:FIG\.\s*12|FIG\.\s*11|V\.\s*RESULTS)", norm_text, re.S | re.I)
    if not m:
        raise RuntimeError("Could not locate Table III in the downloaded paper.")

    block = m.group(1)
    numbers = FLOAT_RE.findall(block)
    if len(numbers) < 13:
        raise RuntimeError(
            "Could not parse the 13 MAP values from Table III. "
            f"Only found {len(numbers)} numeric tokens."
        )

    vals = [float(x) for x in numbers[:13]]
    return MapParameters(
        norm=vals[0],
        eta_tc=vals[1],
        p=vals[2],
        eta_slope=vals[3],
        sigma_k=vals[4],
        eta_crv=vals[5],
        dmin3=vals[6],
        zeta_peak=vals[7],
        tau_fs=vals[8],
        zeta_max=vals[9],
        T_c=vals[10],
        zeta_width=vals[11],
        T_switch=vals[12],
    )


def parse_summary_snippets(text: str) -> Dict[str, str]:
    norm_text = normalize_text(text)
    snippets: Dict[str, str] = {}

    patterns = {
        "equation": r"\(η/s\)\(T\) = .*?\(8\)",
        "weak_dependence": r"temperature dependence of η/s\(T\) is similar.*?weak temperature dependence of η/s\.",
        "minimum_near_tc": r"lowest value of η/s\(T\) is around the critical temperature Tc, close to the universal minimum 1/\(4π\)\.",
    }

    for key, pat in patterns.items():
        m = re.search(pat, norm_text, re.S)
        if m:
            snippets[key] = m.group(0)

    return snippets


def eta_over_s(T_gev: np.ndarray | float, params: MapParameters) -> np.ndarray | float:
    T = np.asarray(T_gev, dtype=float)
    # Published parametrization from Parkkila et al. Eq. (8).
    y = params.eta_tc + params.eta_slope * (T - params.T_c) * np.power(T / params.T_c, params.eta_crv)
    return y if isinstance(T_gev, np.ndarray) else float(y)


def fit_constant(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, int]:
    yhat = np.full_like(y, y.mean())
    sse = float(np.sum((y - yhat) ** 2))
    return yhat, sse, 1


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, int]:
    coeff = np.polyfit(x, y, deg=1)
    yhat = np.polyval(coeff, x)
    sse = float(np.sum((y - yhat) ** 2))
    return yhat, sse, 2


def fit_cubic_dip_peak_candidate(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, int, bool, List[float]]:
    coeff = np.polyfit(x, y, deg=3)
    yhat = np.polyval(coeff, x)
    sse = float(np.sum((y - yhat) ** 2))

    # Valid dip-peak candidate must have two distinct real extrema inside the scan window,
    # ordered as a minimum followed by a maximum.
    dcoeff = np.polyder(coeff)
    roots = np.roots(dcoeff)
    real_roots = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-8 and x.min() <= r.real <= x.max())

    is_valid = False
    if len(real_roots) == 2:
        second = np.polyder(dcoeff)
        c1 = np.polyval(second, real_roots[0])
        c2 = np.polyval(second, real_roots[1])
        is_valid = (c1 > 0) and (c2 < 0) and (real_roots[0] < real_roots[1])

    return yhat, sse, 4, is_valid, real_roots


def bic(n: int, sse: float, k: int, floor: float = 1e-18) -> float:
    sse = max(sse, floor)
    return n * math.log(sse / n) + k * math.log(n)


def analyse_transition_region(params: MapParameters) -> Dict[str, object]:
    x_mev = np.arange(140.0, 171.0, 1.0)
    x_gev = x_mev / 1000.0
    y = eta_over_s(x_gev, params)

    y_const, sse_const, k_const = fit_constant(x_mev, y)
    y_lin, sse_lin, k_lin = fit_linear(x_mev, y)
    y_cub, sse_cub, k_cub, cubic_valid, cubic_roots = fit_cubic_dip_peak_candidate(x_mev, y)

    bic_const = bic(len(x_mev), sse_const, k_const)
    bic_lin = bic(len(x_mev), sse_lin, k_lin)
    bic_cub = bic(len(x_mev), sse_cub, k_cub) if cubic_valid else float("inf")

    best_monotonic_bic = min(bic_const, bic_lin)
    dip_peak_bic_preferred = bool(np.isfinite(bic_cub) and (bic_cub + 2.0 < best_monotonic_bic))

    imin = int(np.argmin(y))
    imax = int(np.argmax(y))

    return {
        "scan_T_mev": x_mev.tolist(),
        "scan_eta_over_s": np.asarray(y).tolist(),
        "bic_constant": bic_const,
        "bic_linear": bic_lin,
        "bic_dip_peak": bic_cub,
        "dip_peak_model_valid": cubic_valid,
        "dip_peak_extrema_mev": cubic_roots,
        "dip_peak_bic_preferred": dip_peak_bic_preferred,
        "min_T_mev": float(x_mev[imin]),
        "max_T_mev": float(x_mev[imax]),
        "min_eta_over_s": float(y[imin]),
        "max_eta_over_s": float(y[imax]),
        "is_monotonic_non_decreasing": bool(np.all(np.diff(y) >= -1e-12)),
        "constant_fit": np.asarray(y_const).tolist(),
        "linear_fit": np.asarray(y_lin).tolist(),
        "dip_peak_fit": np.asarray(y_cub).tolist(),
    }


def make_plot(outdir: Path, params: MapParameters, transition: Dict[str, object], high_T: Dict[str, float], kss: float) -> Path:
    temps_mev = np.linspace(140.0, 320.0, 600)
    temps_gev = temps_mev / 1000.0
    curve = eta_over_s(temps_gev, params)

    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(temps_mev, curve, label="Published MAP eta/s(T)")
    ax.axhline(kss, linestyle="--", label="KSS = 1/(4π)")

    scan_x = np.array(transition["scan_T_mev"], dtype=float)
    scan_y = np.array(transition["scan_eta_over_s"], dtype=float)
    ax.plot(scan_x, scan_y, linewidth=2, alpha=0.7, label="140–170 MeV scan")
    ax.plot(scan_x, np.array(transition["linear_fit"], dtype=float), linestyle=":", label="Linear fit")

    for T_mev, y in high_T.items():
        ax.scatter([float(T_mev)], [float(y)], s=35)
        ax.annotate(f"{T_mev} MeV", (float(T_mev), float(y)), xytext=(4, 4), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Temperature [MeV]")
    ax.set_ylabel("η/s")
    ax.set_title("Test 09: QGP η/s(T) from public Parkkila et al. MAP parameters")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out = outdir / "eta_over_s_curve.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_summary(
    outdir: Path,
    params: MapParameters,
    transition: Dict[str, object],
    high_T: Dict[str, float],
    ratios: Dict[str, float],
    kss: float,
    pass_p5: bool,
    pass_p9c: bool,
    snippets: Dict[str, str],
) -> Path:
    lines = []
    lines.append("TEST 09 — QGP eta/s saturation and oscillation near T_c")
    lines.append("=" * 68)
    lines.append("")
    lines.append("Public source used for computation:")
    lines.append(f"- arXiv abstract: {ARXIV_ABS_URL}")
    lines.append(f"- arXiv PDF: {ARXIV_PDF_URL}")
    lines.append(f"- APS DOI landing page: {APS_DOI_URL}")
    lines.append(f"- JETSCAPE reference mentioned in test note: {JETSCAPE_REF_URL}")
    lines.append("")
    lines.append("Parsed Table III MAP parameters:")
    lines.append(f"- T_c = {params.T_c:.3f} GeV")
    lines.append(f"- eta/s(T_c) = {params.eta_tc:.3f}")
    lines.append(f"- (eta/s)_slope = {params.eta_slope:.3f} GeV^-1")
    lines.append(f"- (eta/s)_crv = {params.eta_crv:.3f}")
    lines.append("")
    lines.append(f"KSS bound = 1/(4*pi) = {kss:.6f}")
    lines.append("")
    lines.append("High-temperature checks:")
    for key in ["200", "250", "300"]:
        lines.append(
            f"- T = {key} MeV: eta/s = {high_T[key]:.6f}, ratio_to_KSS = {ratios[key]:.3f}"
        )
    lines.append("")
    lines.append("Transition-region scan (140-170 MeV):")
    lines.append(f"- monotonic non-decreasing: {transition['is_monotonic_non_decreasing']}")
    lines.append(f"- minimum in scan at {transition['min_T_mev']:.1f} MeV")
    lines.append(f"- maximum in scan at {transition['max_T_mev']:.1f} MeV")
    lines.append(f"- BIC constant = {transition['bic_constant']:.6f}")
    lines.append(f"- BIC linear   = {transition['bic_linear']:.6f}")
    lines.append(
        "- BIC dip-peak = "
        + (f"{transition['bic_dip_peak']:.6f}" if np.isfinite(transition['bic_dip_peak']) else "inf (invalid dip-peak cubic)")
    )
    lines.append(f"- dip_peak_bic_preferred = {transition['dip_peak_bic_preferred']}")
    lines.append("")
    lines.append("Verdicts:")
    lines.append(f"- pass_p5 = {pass_p5}")
    lines.append(f"- pass_p9c = {pass_p9c}")
    lines.append("")
    if snippets:
        lines.append("Supporting snippets parsed from the paper:")
        for key, val in snippets.items():
            lines.append(f"[{key}] {val}")
            lines.append("")

    out = outdir / "summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Public-source implementation of Test 09.")
    parser.add_argument("--outdir", default="out_test09_public", help="Output directory")
    parser.add_argument(
        "--pdf-url",
        default=ARXIV_PDF_URL,
        help="Public PDF URL for the Parkkila et al. paper (default: arXiv PDF)",
    )
    parser.add_argument(
        "--strict-p5-threshold",
        type=float,
        default=0.50,
        help="Require |eta/s - KSS| / KSS <= threshold at all high-T checkpoints (default: 0.50)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / "cache"
    cache_dir.mkdir(exist_ok=True)

    pdf_path = cache_dir / "parkkila_2021_eta_over_s.pdf"
    print(f"[data] Downloading public paper PDF from {args.pdf_url}")
    download_file(args.pdf_url, pdf_path)
    print(f"[data] Saved PDF to {pdf_path}")

    print("[parse] Extracting text from downloaded PDF")
    pdf_text = extract_pdf_text(pdf_path)
    if not pdf_text.strip():
        raise RuntimeError("PDF text extraction returned empty text.")

    params = parse_table_iii_map_parameters(pdf_text)
    snippets = parse_summary_snippets(pdf_text)

    kss = 1.0 / (4.0 * math.pi)
    high_T_mev = [200, 250, 300]
    high_T = {str(T): float(eta_over_s(T / 1000.0, params)) for T in high_T_mev}
    ratios = {T: (val / kss) for T, val in high_T.items()}

    transition = analyse_transition_region(params)

    # Strict operationalization of the uploaded criterion.
    pass_p5 = all(abs(r - 1.0) <= args.strict_p5_threshold for r in ratios.values())
    pass_p9c = bool(transition["dip_peak_bic_preferred"])

    plot_path = make_plot(outdir, params, transition, high_T, kss)
    summary_path = write_summary(outdir, params, transition, high_T, ratios, kss, pass_p5, pass_p9c, snippets)

    result = {
        "test_name": "Test 09 — QGP eta/s Saturation and Oscillation Near T_c",
        "implementation": "public-source script using published MAP eta/s(T) parametrization from Parkkila et al. (2021)",
        "source_pdf_url": args.pdf_url,
        "source_abs_url": ARXIV_ABS_URL,
        "source_doi_url": APS_DOI_URL,
        "jetscape_reference_url": JETSCAPE_REF_URL,
        "downloaded_pdf_sha256": sha256_of_file(pdf_path),
        "map_parameters": {
            "T_c_GeV": params.T_c,
            "eta_over_s_Tc": params.eta_tc,
            "eta_over_s_slope_GeV_inv": params.eta_slope,
            "eta_over_s_curvature": params.eta_crv,
            "zeta_over_s_peak_GeV": params.zeta_peak,
            "zeta_over_s_max": params.zeta_max,
            "zeta_over_s_width_GeV": params.zeta_width,
            "T_switch_GeV": params.T_switch,
        },
        "eta_over_s_high_T": {f"{k}_MeV": v for k, v in high_T.items()},
        "kss_bound": kss,
        "ratio_to_kss": {f"{k}_MeV": v for k, v in ratios.items()},
        "transition_region": {
            "scan_range_MeV": [140, 170],
            "min_T_mev": transition["min_T_mev"],
            "max_T_mev": transition["max_T_mev"],
            "min_eta_over_s": transition["min_eta_over_s"],
            "max_eta_over_s": transition["max_eta_over_s"],
            "bic_constant": transition["bic_constant"],
            "bic_linear": transition["bic_linear"],
            "bic_dip_peak": transition["bic_dip_peak"],
            "dip_peak_model_valid": transition["dip_peak_model_valid"],
            "dip_peak_extrema_mev": transition["dip_peak_extrema_mev"],
            "dip_peak_bic_preferred": transition["dip_peak_bic_preferred"],
            "is_monotonic_non_decreasing": transition["is_monotonic_non_decreasing"],
        },
        "pass_p5": pass_p5,
        "pass_p9c": pass_p9c,
        "notes": {
            "method_limit": "Uses published MAP parameters, not the full posterior chain.",
            "why": "Practical public-source implementation without multi-gigabyte MCMC downloads.",
        },
        "artifacts": {
            "plot": str(plot_path),
            "summary": str(summary_path),
            "pdf": str(pdf_path),
        },
    }

    result_path = outdir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    public_sources = {
        "analysis_primary": {
            "title": "Bayesian estimation of the specific shear and bulk viscosity of the quark-gluon plasma with additional flow harmonic observables",
            "arxiv_abs": ARXIV_ABS_URL,
            "arxiv_pdf": args.pdf_url,
            "doi": APS_DOI_URL,
        },
        "test_note_reference": {
            "name": UPLOADED_TEST_NAME,
            "mentioned_reference": JETSCAPE_REF_URL,
        },
    }
    (outdir / "public_sources.json").write_text(json.dumps(public_sources, indent=2), encoding="utf-8")

    print("[done] Wrote:")
    print(f"  - {result_path}")
    print(f"  - {summary_path}")
    print(f"  - {plot_path}")
    print(f"  - {outdir / 'public_sources.json'}")


if __name__ == "__main__":
    main()
