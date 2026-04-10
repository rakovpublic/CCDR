#!/usr/bin/env python3
"""
Test 06 — LHCb / CMS 1–2 GeV dimuon excess audit using only public sources.

This script is intentionally self-contained and does not require the user to pass any
local data files. It downloads public material directly from CMS/LHCb/HEPData and
produces:

- result.json
- cms_limits_figure5a.csv (if HEPData parse succeeds)
- cms_limits_figure5a.png
- downloaded source files under cache/
- summary.txt

Important limitation
--------------------
For this test, the relevant public CMS HEPData record exposes limit/efficiency tables,
not the raw low-mass dimuon spectrum itself. The LHCb paper gives explicit public
statements about the largest local excesses below 20 GeV and their global significance.
Accordingly, this implementation is a *public-source audit* rather than a full raw-spectrum
re-fit. Where machine-readable spectrum tables are unavailable, the script uses official
public statements from the papers and public results pages.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

# Optional dependencies. The script still runs if these are missing.
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


USER_AGENT = "Mozilla/5.0 (compatible; public-data-audit/1.0; +https://openai.com)"

CMS_HPDATA_RECORD_JSON = "https://www.hepdata.net/record/ins2704121?format=json"
CMS_FIG3_PNG = (
    "https://cms-results.web.cern.ch/cms-results/public-results/publications/"
    "EXO-21-005/CMS-EXO-21-005_Figure_003.png"
)
CMS_RESULTS_HTML = (
    "https://cms-results.web.cern.ch/cms-results/public-results/publications/"
    "EXO-21-005/index.html"
)
LHCB_PDF = "https://cds.cern.ch/record/2722971/files/LHCb-PAPER-2020-013.pdf"
LHCB_ARXIV_PDF = "https://arxiv.org/pdf/2007.03923.pdf"

# Public findings manually mirrored from the official papers as a robust fallback in case
# PDF text extraction is unavailable on the target machine.
KNOWN_PUBLIC_FINDINGS = {
    "lhcb_below_20gev": {
        "paper": "Searches for low-mass dimuon resonances, LHCb, JHEP 10 (2020) 156",
        "inclusive_largest_local_sigma": 3.7,
        "inclusive_mass_gev": 0.349,
        "inclusive_pt_bin_gev": [3.0, 5.0],
        "inclusive_global_sigma_approx": 1.0,
        "xb_largest_local_sigma": 3.1,
        "xb_mass_gev": 2.424,
        "xb_pt_bin_gev": [10.0, 20.0],
        "xb_global_sigma_below": 1.0,
        "statement": (
            "No significant excess is found in either prompt-like spectrum for m(X) < 20 GeV."
        ),
    },
    "cms_low_mass_summary": {
        "paper": "Search for direct production of GeV-scale resonances decaying to a pair of muons, CMS, JHEP 12 (2023) 070",
        "mass_range_gev": [1.1, 2.6],
        "statement": (
            "No significant excess of events above the expectation from the standard model background is observed."
        ),
    },
}


@dataclass
class Finding:
    source: str
    statement: str
    local_sigma: Optional[float] = None
    global_sigma: Optional[float] = None
    mass_gev: Optional[float] = None
    width_mev: Optional[float] = None
    category: Optional[str] = None


@dataclass
class ResultSummary:
    search_window_gev: Tuple[float, float]
    excess_mass_gev: Optional[float]
    excess_width_mev: Optional[float]
    local_significance_sigma: Optional[float]
    look_elsewhere_significance: Optional[float]
    angular_distribution_isotropic: Optional[bool]
    pass_v6_p9a: bool
    status: str
    public_source_assessment: str
    cms_summary: Dict[str, Any]
    lhcb_summary: Dict[str, Any]
    files_downloaded: List[str]


def fetch_url(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def download_to(url: str, path: Path, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_url(url)
    path.write_bytes(data)
    return path


def fetch_json(url: str) -> Any:
    raw = fetch_url(url)
    return json.loads(raw.decode("utf-8"))


def safe_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return out.strip("_") or "file"


def extract_text_from_pdf(path: Path) -> Optional[str]:
    if PdfReader is None:
        return None
    try:
        reader = PdfReader(str(path))
        chunks: List[str] = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                chunks.append("")
        text = "\n".join(chunks)
        return text if text.strip() else None
    except Exception:
        return None


def parse_lhcb_findings_from_text(text: str) -> Dict[str, Any]:
    compact = re.sub(r"\s+", " ", text)

    out: Dict[str, Any] = {
        "source": "parsed_pdf_text",
        "inclusive_largest_local_sigma": None,
        "inclusive_mass_gev": None,
        "inclusive_global_sigma_approx": None,
        "xb_largest_local_sigma": None,
        "xb_mass_gev": None,
        "xb_global_sigma_below": None,
        "statement": None,
    }

    m1 = re.search(
        r"largest local excess in the inclusive search.*?is\s+([0-9]+(?:\.[0-9]+)?)σ\s+at\s+([0-9]+)\s*MeV.*?global significance is only\s*[≈~]?\s*([0-9]+(?:\.[0-9]+)?)σ",
        compact,
        re.IGNORECASE,
    )
    if m1:
        out["inclusive_largest_local_sigma"] = float(m1.group(1))
        out["inclusive_mass_gev"] = float(m1.group(2)) / 1000.0
        out["inclusive_global_sigma_approx"] = float(m1.group(3))

    m2 = re.search(
        r"largest local excess in the X \+ b search below 20 GeV is\s+([0-9]+(?:\.[0-9]+)?)σ\s+at\s+([0-9]+)\s*MeV.*?global significance is below\s+([0-9]+(?:\.[0-9]+)?)σ",
        compact,
        re.IGNORECASE,
    )
    if m2:
        out["xb_largest_local_sigma"] = float(m2.group(1))
        out["xb_mass_gev"] = float(m2.group(2)) / 1000.0
        out["xb_global_sigma_below"] = float(m2.group(3))

    m3 = re.search(
        r"Therefore, no significant excess is found in either prompt\S* spectrum for m\(X\) < 20 GeV",
        compact,
        re.IGNORECASE,
    )
    if m3:
        out["statement"] = m3.group(0)

    return out


def get_lhcb_summary(cache_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    downloaded: List[str] = []

    pdf_path = cache_dir / "lhcb_low_mass_dimuon_search.pdf"
    try:
        download_to(LHCB_PDF, pdf_path)
        downloaded.append(str(pdf_path))
    except Exception:
        # Fallback to arXiv PDF if the CDS mirror is not reachable.
        pdf_path = cache_dir / "lhcb_low_mass_dimuon_search_arxiv.pdf"
        download_to(LHCB_ARXIV_PDF, pdf_path)
        downloaded.append(str(pdf_path))

    text = extract_text_from_pdf(pdf_path)
    if text:
        parsed = parse_lhcb_findings_from_text(text)
        if parsed.get("statement"):
            parsed["paper_url"] = LHCB_PDF
            return parsed, downloaded

    fallback = dict(KNOWN_PUBLIC_FINDINGS["lhcb_below_20gev"])
    fallback["source"] = "embedded_public_fallback"
    fallback["paper_url"] = LHCB_PDF
    return fallback, downloaded


def extract_cms_table_urls(record_json: Dict[str, Any]) -> Dict[str, str]:
    urls: Dict[str, str] = {}
    for table in record_json.get("data_tables", []):
        name = str(table.get("name", ""))
        data = table.get("data", {}) or {}
        json_url = data.get("json")
        if json_url:
            urls[name] = str(json_url)
    return urls


def flatten_hepdata_name(dep: Dict[str, Any]) -> str:
    parts: List[str] = []
    header = dep.get("header") or {}
    if isinstance(header, dict):
        if header.get("name"):
            parts.append(str(header.get("name")))
    for q in dep.get("qualifiers", []) or []:
        if isinstance(q, dict):
            nm = q.get("name")
            val = q.get("value")
            if nm and val:
                parts.append(f"{nm}={val}")
    return " | ".join(parts) if parts else "series"


def parse_hepdata_xy(table_json: Dict[str, Any]) -> Dict[str, List[float]]:
    indep = table_json.get("independent_variables", []) or []
    dep = table_json.get("dependent_variables", []) or []
    if not indep or not dep:
        raise ValueError("Unexpected HEPData JSON structure: no independent/dependent variables")

    xvals: List[float] = []
    for item in indep[0].get("values", []) or []:
        if "value" in item:
            xvals.append(float(item["value"]))
        elif "low" in item and "high" in item:
            xvals.append(0.5 * (float(item["low"]) + float(item["high"])))
        else:
            raise ValueError(f"Cannot parse x value from item: {item!r}")

    series: Dict[str, List[float]] = {"x": xvals}
    for d in dep:
        label = flatten_hepdata_name(d)
        vals: List[float] = []
        for item in d.get("values", []) or []:
            value = item.get("value")
            if value in (None, ""):
                vals.append(float("nan"))
            else:
                try:
                    vals.append(float(value))
                except Exception:
                    vals.append(float("nan"))
        series[label] = vals
    return series


def pick_observed_series(series: Dict[str, List[float]]) -> Tuple[str, List[float]]:
    candidates = []
    for key, vals in series.items():
        if key == "x":
            continue
        score = 0
        low = key.lower()
        if "observ" in low:
            score += 3
        if "limit" in low:
            score += 2
        if "95" in low:
            score += 1
        if score > 0:
            candidates.append((score, key, vals))
    if not candidates:
        # fallback to first non-x series
        for key, vals in series.items():
            if key != "x":
                return key, vals
        raise ValueError("No dependent series found")
    candidates.sort(reverse=True)
    _, key, vals = candidates[0]
    return key, vals


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def plot_cms_limits(x: List[float], y: List[float], out_png: Path, window: Tuple[float, float]) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y)
    ax.axvspan(window[0], window[1], alpha=0.2)
    ax.set_xlabel("Dimuon mass [GeV]")
    ax.set_ylabel("Observed 95% CL limit")
    ax.set_title("CMS EXO-21-005 observed model-independent limits")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def get_cms_summary(cache_dir: Path, outdir: Path, window: Tuple[float, float]) -> Tuple[Dict[str, Any], List[str]]:
    downloaded: List[str] = []
    record_path = cache_dir / "cms_exo_21_005_record.json"
    record = fetch_json(CMS_HPDATA_RECORD_JSON)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    downloaded.append(str(record_path))

    fig3_path = cache_dir / "cms_figure3_scouting_mass_spectrum.png"
    try:
        download_to(CMS_FIG3_PNG, fig3_path)
        downloaded.append(str(fig3_path))
    except Exception:
        pass

    summary = dict(KNOWN_PUBLIC_FINDINGS["cms_low_mass_summary"])
    summary["source"] = "official_public_summary"
    summary["results_page"] = CMS_RESULTS_HTML
    summary["hepdata_record"] = CMS_HPDATA_RECORD_JSON

    # Best effort: parse Figure 5-a HEPData observed limits and save a simple plot.
    try:
        table_urls = extract_cms_table_urls(record)
        fig5a_url = table_urls.get("Figure 5-a") or table_urls.get("Figure 5")
        if fig5a_url:
            fig5a = fetch_json(fig5a_url)
            fig5a_path = cache_dir / "cms_figure5a_limits.json"
            fig5a_path.write_text(json.dumps(fig5a, indent=2), encoding="utf-8")
            downloaded.append(str(fig5a_path))

            series = parse_hepdata_xy(fig5a)
            obs_key, obs_vals = pick_observed_series(series)
            xvals = series["x"]

            csv_path = outdir / "cms_limits_figure5a.csv"
            rows = zip(xvals, obs_vals)
            write_csv(csv_path, ["mass_gev", "observed_limit"], rows)

            png_path = outdir / "cms_limits_figure5a.png"
            plot_cms_limits(xvals, obs_vals, png_path, window)

            summary["figure5a_observed_series_name"] = obs_key
            in_window = [
                {"mass_gev": x, "observed_limit": y}
                for x, y in zip(xvals, obs_vals)
                if window[0] <= x <= window[1] and not math.isnan(y)
            ]
            summary["observed_limits_in_window"] = in_window
    except Exception as exc:
        summary["hepdata_parse_warning"] = f"Could not parse Figure 5-a automatically: {exc}"

    return summary, downloaded


def decide_result(window: Tuple[float, float], cms: Dict[str, Any], lhcb: Dict[str, Any], files: List[str]) -> ResultSummary:
    cms_no_sig = "no significant excess" in str(cms.get("statement", "")).lower()
    lhcb_no_sig = "no significant excess" in str(lhcb.get("statement", "")).lower()

    # Public-source audit logic:
    # - CMS official result says no significant excess in 1.1–2.6 GeV.
    # - LHCb official result says no significant excess below 20 GeV and the largest local
    #   excesses below 20 GeV are outside the 1.0–1.8 GeV target window.
    lhcb_masses_outside_window = True
    for key in ("inclusive_mass_gev", "xb_mass_gev"):
        val = lhcb.get(key)
        if isinstance(val, (int, float)) and window[0] <= float(val) <= window[1]:
            lhcb_masses_outside_window = False

    no_public_support = cms_no_sig and lhcb_no_sig and lhcb_masses_outside_window

    if no_public_support:
        status = "no_public_support_for_target_excess"
        assessment = (
            "Official public CMS and LHCb results do not support a narrow 1.0–1.8 GeV dimuon excess. "
            "CMS reports no significant excess in 1.1–2.6 GeV, and LHCb reports no significant excess "
            "below 20 GeV, with the largest local excesses occurring outside the target window."
        )
        return ResultSummary(
            search_window_gev=window,
            excess_mass_gev=None,
            excess_width_mev=None,
            local_significance_sigma=None,
            look_elsewhere_significance=None,
            angular_distribution_isotropic=None,
            pass_v6_p9a=False,
            status=status,
            public_source_assessment=assessment,
            cms_summary=cms,
            lhcb_summary=lhcb,
            files_downloaded=files,
        )

    assessment = (
        "Public sources do not yield a machine-readable raw residual spectrum for a faithful re-fit in the target window. "
        "This audit remains inconclusive unless a public binned spectrum becomes available or a dedicated figure-digitization "
        "workflow is added."
    )
    return ResultSummary(
        search_window_gev=window,
        excess_mass_gev=None,
        excess_width_mev=None,
        local_significance_sigma=None,
        look_elsewhere_significance=None,
        angular_distribution_isotropic=None,
        pass_v6_p9a=False,
        status="inconclusive_public_data_audit",
        public_source_assessment=assessment,
        cms_summary=cms,
        lhcb_summary=lhcb,
        files_downloaded=files,
    )


def write_summary_txt(path: Path, result: ResultSummary) -> None:
    text = textwrap.dedent(
        f"""
        Test 06 — LHCb/CMS 1–2 GeV dimuon excess public-source audit
        ==============================================================

        Search window: {result.search_window_gev[0]:.3f}–{result.search_window_gev[1]:.3f} GeV
        Status: {result.status}
        pass_v6_p9a: {result.pass_v6_p9a}

        Assessment
        ----------
        {result.public_source_assessment}

        CMS public summary
        ------------------
        {result.cms_summary.get('statement')}
        Mass range: {result.cms_summary.get('mass_range_gev')}

        LHCb public summary
        -------------------
        {result.lhcb_summary.get('statement')}
        Largest local excess below 20 GeV (inclusive):
          sigma = {result.lhcb_summary.get('inclusive_largest_local_sigma')}
          mass  = {result.lhcb_summary.get('inclusive_mass_gev')} GeV
          global ~ {result.lhcb_summary.get('inclusive_global_sigma_approx')} sigma

        Largest local excess below 20 GeV (X+b):
          sigma = {result.lhcb_summary.get('xb_largest_local_sigma')}
          mass  = {result.lhcb_summary.get('xb_mass_gev')} GeV
          global < {result.lhcb_summary.get('xb_global_sigma_below')} sigma

        Output JSON fields
        ------------------
        excess_mass_gev            = {result.excess_mass_gev}
        excess_width_mev           = {result.excess_width_mev}
        local_significance_sigma   = {result.local_significance_sigma}
        look_elsewhere_significance= {result.look_elsewhere_significance}
        angular_distribution_isotropic = {result.angular_distribution_isotropic}

        Downloaded files
        ----------------
        """
    ).strip()
    more = "\n".join(f"- {p}" for p in result.files_downloaded)
    path.write_text(text + "\n" + more + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Public-source implementation of Test 06 (LHCb/CMS 1–2 GeV dimuon excess audit)."
    )
    parser.add_argument("--outdir", default="out_test06_public", help="Output directory")
    parser.add_argument("--window-min", type=float, default=1.0, help="Search window minimum in GeV")
    parser.add_argument("--window-max", type=float, default=1.8, help="Search window maximum in GeV")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    cache_dir = outdir / "cache"
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    window = (float(args.window_min), float(args.window_max))
    downloaded: List[str] = []

    cms_summary, cms_files = get_cms_summary(cache_dir, outdir, window)
    downloaded.extend(cms_files)

    lhcb_summary, lhcb_files = get_lhcb_summary(cache_dir)
    downloaded.extend(lhcb_files)

    result = decide_result(window, cms_summary, lhcb_summary, downloaded)

    result_json_path = outdir / "result.json"
    result_json_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    summary_txt_path = outdir / "summary.txt"
    write_summary_txt(summary_txt_path, result)

    print(json.dumps(asdict(result), indent=2))
    print(f"[done] Wrote {result_json_path}")
    print(f"[done] Wrote {summary_txt_path}")
    if (outdir / "cms_limits_figure5a.csv").exists():
        print(f"[done] Wrote {outdir / 'cms_limits_figure5a.csv'}")
    if (outdir / "cms_limits_figure5a.png").exists():
        print(f"[done] Wrote {outdir / 'cms_limits_figure5a.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
