#!/usr/bin/env python3
"""
Test 07 — R_K / R_K* Geometric Ratio Test

Standalone script. Downloads public source material itself and requires no user-supplied
input files.

Current implementation uses the published LHCb public paper source (arXiv:2212.09152)
for the simultaneous R_K and R_K* measurements in the central q^2 bin, extracts the
reported central values and stat/syst uncertainties, and evaluates the CCDR v6 P9d test.

Why central-q^2?
The uploaded test spec quotes the LHCb 2022/2023 values R_K = 0.949 ± 0.047 and
R_K* = 1.027 ± 0.072 in the central-q^2 bin, and its note explicitly uses
|delta R_K| / |delta R_K*| ~ 1.7 as the motivating quantity.

Outputs:
- result.json
- report.md
- rk_rkstar_contour.png

No manual data-file arguments are required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import textwrap
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import requests

# Use a non-interactive backend before importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

try:
    from pypdf import PdfReader
except Exception as exc:  # pragma: no cover
    PdfReader = None
    PYPDF_IMPORT_ERROR = exc
else:
    PYPDF_IMPORT_ERROR = None

ARXIV_PDF_URL = "https://arxiv.org/pdf/2212.09152.pdf"
CDS_PDF_URL = "https://cds.cern.ch/record/2845047/files/2212.09152.pdf"
SM_RATIO = math.sqrt(3.0)
EPS = 1e-6


@dataclass
class Measurement:
    value: float
    stat_up: float
    stat_down: float
    syst_up: float
    syst_down: float

    @property
    def sigma_up(self) -> float:
        return math.sqrt(self.stat_up ** 2 + self.syst_up ** 2)

    @property
    def sigma_down(self) -> float:
        return math.sqrt(self.stat_down ** 2 + self.syst_down ** 2)

    @property
    def sigma_sym(self) -> float:
        return 0.5 * (self.sigma_up + self.sigma_down)

    @property
    def delta(self) -> float:
        return self.value - 1.0


@dataclass
class Result:
    source_pdf: str
    q2_region: str
    R_K: float
    R_K_sigma: float
    R_Kstar: float
    R_Kstar_sigma: float
    delta_RK: float
    delta_RKstar: float
    ratio: float
    ratio_sigma: float
    signed_ratio: float
    signed_ratio_sigma: float
    z_score_vs_sqrt3: float
    z_score_vs_one: float
    z_score_vs_zero: float
    pass_v6_p9d: bool
    null_test_passed: bool
    status: str
    covariance_assumption: str
    rho_delta_RK_RKstar: float


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "ChatGPT-Test07-RK-RKstar/1.0",
        "Accept": "application/pdf,application/octet-stream,text/plain,*/*",
    })
    return s


def download_with_cache(url: str, path: Path, timeout: int = 60) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    sess = _session()
    resp = sess.get(url, timeout=timeout)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def extract_pdf_text(pdf_path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError(
            "pypdf is required to extract text from the public PDF source "
            f"but could not be imported: {PYPDF_IMPORT_ERROR}"
        )
    reader = PdfReader(str(pdf_path))
    pieces = []
    for page in reader.pages:
        pieces.append(page.extract_text() or "")
    return "\n".join(pieces)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "∗": "*",
        "′": "'",
        "“": '"',
        "”": '"',
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Preserve enough structure for regex but make whitespace manageable.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text


def parse_central_q2_measurements(text: str) -> Tuple[Measurement, Measurement]:
    flat = " ".join(text.split())

    pattern = re.compile(
        r"central-q\s*2\s*\(\s*"
        r"RK\s*=\s*(?P<rk>[0-9.]+)\s*\+(?P<rk_stat_up>[0-9.]+)\s*-(?P<rk_stat_dn>[0-9.]+)\s*\(stat\)\s*"
        r"\+(?P<rk_syst_up>[0-9.]+)\s*-(?P<rk_syst_dn>[0-9.]+)\s*\(syst\)\s*,\s*"
        r"RK\*\s*=\s*(?P<rks>[0-9.]+)\s*\+(?P<rks_stat_up>[0-9.]+)\s*-(?P<rks_stat_dn>[0-9.]+)\s*\(stat\)\s*"
        r"\+(?P<rks_syst_up>[0-9.]+)\s*-(?P<rks_syst_dn>[0-9.]+)\s*\(syst\)",
        flags=re.IGNORECASE,
    )
    m = pattern.search(flat)
    if not m:
        snippet_re = re.compile(r"central-q.{0,300}?RK.{0,300}?RK\*", flags=re.IGNORECASE | re.DOTALL)
        snippet = snippet_re.search(flat)
        raise RuntimeError(
            "Could not parse the central-q^2 R_K / R_K* block from the downloaded public paper."
            + (f" Nearby text: {snippet.group(0)!r}" if snippet else "")
        )

    rk = Measurement(
        value=float(m.group("rk")),
        stat_up=float(m.group("rk_stat_up")),
        stat_down=float(m.group("rk_stat_dn")),
        syst_up=float(m.group("rk_syst_up")),
        syst_down=float(m.group("rk_syst_dn")),
    )
    rks = Measurement(
        value=float(m.group("rks")),
        stat_up=float(m.group("rks_stat_up")),
        stat_down=float(m.group("rks_stat_dn")),
        syst_up=float(m.group("rks_syst_up")),
        syst_down=float(m.group("rks_syst_dn")),
    )
    return rk, rks


def split_normal_sample(rng: np.random.Generator, mean: float, sigma_down: float, sigma_up: float, size: int) -> np.ndarray:
    sigma_down = max(float(sigma_down), 1e-9)
    sigma_up = max(float(sigma_up), 1e-9)
    z = rng.standard_normal(size)
    out = np.empty(size, dtype=float)
    pos = z >= 0
    out[pos] = mean + z[pos] * sigma_up
    out[~pos] = mean + z[~pos] * sigma_down
    return out


def robust_ratio_sigma(delta1: np.ndarray, delta2: np.ndarray) -> Tuple[float, float]:
    valid = np.abs(delta2) > EPS
    if valid.sum() < max(1000, int(0.1 * len(valid))):
        raise RuntimeError("Too few valid Monte Carlo draws for a stable ratio estimate.")

    signed = delta1[valid] / delta2[valid]
    mag = np.abs(delta1[valid]) / np.abs(delta2[valid])

    def sigma_from_quantiles(arr: np.ndarray) -> float:
        q16, q84 = np.quantile(arr, [0.16, 0.84])
        return float(0.5 * (q84 - q16))

    return sigma_from_quantiles(mag), sigma_from_quantiles(signed)


def draw_covariance_ellipse(ax, mean: np.ndarray, cov: np.ndarray, n_std: float, **kwargs) -> None:
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0.0))
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ell)


def make_plot(rk: Measurement, rks: Measurement, samples_delta_rk: np.ndarray, samples_delta_rks: np.ndarray, outpath: Path) -> None:
    mean = np.array([rks.delta, rk.delta])
    cov = np.cov(np.vstack([samples_delta_rks, samples_delta_rk]))

    xlim = max(0.25, 4 * max(abs(rks.delta), rks.sigma_sym))
    ylim = max(0.25, 4 * max(abs(rk.delta), rk.sigma_sym))

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    ax.scatter(samples_delta_rks[::200], samples_delta_rk[::200], s=2, alpha=0.03)
    draw_covariance_ellipse(ax, mean, cov, n_std=1.0, linewidth=2.0)
    draw_covariance_ellipse(ax, mean, cov, n_std=2.0, linewidth=1.5, linestyle="--")

    xs = np.linspace(-xlim, xlim, 400)
    ax.plot(xs, SM_RATIO * xs, linestyle=":", linewidth=2.0, label=r"$\delta R_K = \sqrt{3}\,\delta R_{K^*}$")
    ax.plot(xs, -SM_RATIO * xs, linestyle=":", linewidth=1.4, label=r"$|\delta R_K| = \sqrt{3}\,|\delta R_{K^*}|$")

    ax.scatter([0.0], [0.0], marker="x", s=80, label="SM origin")
    ax.scatter([mean[0]], [mean[1]], marker="*", s=140, label="LHCb central value")

    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel(r"$\delta R_{K^*} = R_{K^*}-1$")
    ax.set_ylabel(r"$\delta R_K = R_K-1$")
    ax.set_title("Test 07: R_K / R_K* geometric-ratio check")
    ax.legend(fontsize=9)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def evaluate(rk: Measurement, rks: Measurement, samples: int, seed: int) -> Result:
    rng = np.random.default_rng(seed)
    rk_samples = split_normal_sample(rng, rk.value, rk.sigma_down, rk.sigma_up, samples)
    rks_samples = split_normal_sample(rng, rks.value, rks.sigma_down, rks.sigma_up, samples)

    delta_rk = rk_samples - 1.0
    delta_rks = rks_samples - 1.0

    ratio_sigma, signed_ratio_sigma = robust_ratio_sigma(delta_rk, delta_rks)
    ratio = abs(rk.delta) / max(abs(rks.delta), EPS)
    signed_ratio = rk.delta / (rks.delta if abs(rks.delta) > EPS else math.copysign(EPS, rks.delta if rks.delta != 0 else 1.0))

    null_test_passed = (abs(rk.delta) <= 2.0 * rk.sigma_sym) and (abs(rks.delta) <= 2.0 * rks.sigma_sym)
    z_vs_sqrt3 = abs(ratio - SM_RATIO) / max(ratio_sigma, 1e-12)
    z_vs_one = abs(ratio - 1.0) / max(ratio_sigma, 1e-12)
    z_vs_zero = abs(ratio - 0.0) / max(ratio_sigma, 1e-12)

    if null_test_passed:
        status = "cannot_fire"
        pass_v6_p9d = False
    elif z_vs_sqrt3 <= 1.0:
        status = "strong_support"
        pass_v6_p9d = True
    elif z_vs_sqrt3 <= 2.0:
        status = "consistent"
        pass_v6_p9d = True
    elif z_vs_sqrt3 > 3.0:
        status = "refuted"
        pass_v6_p9d = False
    else:
        status = "inconclusive"
        pass_v6_p9d = False

    return Result(
        source_pdf=ARXIV_PDF_URL,
        q2_region="central",
        R_K=rk.value,
        R_K_sigma=rk.sigma_sym,
        R_Kstar=rks.value,
        R_Kstar_sigma=rks.sigma_sym,
        delta_RK=rk.delta,
        delta_RKstar=rks.delta,
        ratio=ratio,
        ratio_sigma=ratio_sigma,
        signed_ratio=signed_ratio,
        signed_ratio_sigma=signed_ratio_sigma,
        z_score_vs_sqrt3=z_vs_sqrt3,
        z_score_vs_one=z_vs_one,
        z_score_vs_zero=z_vs_zero,
        pass_v6_p9d=pass_v6_p9d,
        null_test_passed=null_test_passed,
        status=status,
        covariance_assumption=(
            "No public machine-readable RK/RK* covariance matrix was located in the source used here; "
            "the Monte Carlo propagation therefore assumes rho = 0 between delta_RK and delta_RKstar."
        ),
        rho_delta_RK_RKstar=0.0,
    )


def write_report(result: Result, rk: Measurement, rks: Measurement, outpath: Path) -> None:
    report = f"""# Test 07 — R_K / R_K* Geometric Ratio Test

## Source
- Public paper PDF: `{result.source_pdf}`
- q² region used for the main test: `{result.q2_region}`

## Extracted measurements
- R_K = {rk.value:.3f} +{rk.sigma_up:.3f}/-{rk.sigma_down:.3f} (total)
- R_K* = {rks.value:.3f} +{rks.sigma_up:.3f}/-{rks.sigma_down:.3f} (total)

## Derived quantities
- delta_RK = {result.delta_RK:+.6f}
- delta_RKstar = {result.delta_RKstar:+.6f}
- magnitude ratio |delta_RK|/|delta_RKstar| = {result.ratio:.6f} ± {result.ratio_sigma:.6f}
- signed ratio delta_RK/delta_RKstar = {result.signed_ratio:.6f} ± {result.signed_ratio_sigma:.6f}
- sqrt(3) = {SM_RATIO:.6f}
- z-score vs sqrt(3) = {result.z_score_vs_sqrt3:.6f}

## Verdict
- status = `{result.status}`
- pass_v6_p9d = `{result.pass_v6_p9d}`
- null_test_passed = `{result.null_test_passed}`

## Important ambiguity handled explicitly
The uploaded Test 07 note compares **|delta_RK|/|delta_RKstar|** to ~1.7.
However, the published central values in the central-q² bin give delta_RK < 0 and
delta_RKstar > 0, so the **signed** ratio is negative. To keep the implementation aligned
with the note, the primary `ratio` reported in `result.json` is the **magnitude ratio**,
while `signed_ratio` is also reported for transparency.

## Covariance treatment
{result.covariance_assumption}
"""
    outpath.write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Test 07 from public sources only.")
    parser.add_argument("--outdir", default="test07_out", help="Output directory.")
    parser.add_argument("--cache-dir", default=".cache_test07", help="Download cache directory.")
    parser.add_argument("--samples", type=int, default=300_000, help="Monte Carlo draws for uncertainty propagation.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cache_dir = Path(args.cache_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = cache_dir / "2212.09152.pdf"
    source_url = ARXIV_PDF_URL
    try:
        download_with_cache(ARXIV_PDF_URL, pdf_path)
    except Exception:
        source_url = CDS_PDF_URL
        download_with_cache(CDS_PDF_URL, pdf_path)

    text = extract_pdf_text(pdf_path)
    text = normalize_text(text)
    rk, rks = parse_central_q2_measurements(text)
    result = evaluate(rk, rks, samples=args.samples, seed=args.seed)
    result.source_pdf = source_url

    json_path = outdir / "result.json"
    report_path = outdir / "report.md"
    plot_path = outdir / "rk_rkstar_contour.png"

    rng = np.random.default_rng(args.seed)
    rk_samples = split_normal_sample(rng, rk.value, rk.sigma_down, rk.sigma_up, 100_000)
    rks_samples = split_normal_sample(rng, rks.value, rks.sigma_down, rks.sigma_up, 100_000)
    make_plot(rk, rks, rk_samples - 1.0, rks_samples - 1.0, plot_path)

    json_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    write_report(result, rk, rks, report_path)

    print(json.dumps(asdict(result), indent=2))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
