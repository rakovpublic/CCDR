#!/usr/bin/env python3
"""
V10 automated evidence discovery layer for CCDR Tier-B tests.

Goal: reduce data_limited outcomes without adding manual data-entry steps.
The module automatically tries, in order:
  - data-contract driven required-column scoring
  - article/HTML supplementary-file discovery
  - arXiv source package extraction
  - repository/metadata expansion via DOI, Crossref, DataCite, OpenAlex hints
  - optional PDF table extraction (secondary evidence only)
  - optional vector/PDF figure-data diagnostics (exploratory only)
  - OSF parent/sibling link traversal and schema-PDF extraction
  - HEPData/Zenodo/Figshare/OSF connector style direct artifact discovery
  - unit normalization, nearest-miss, and sensitivity diagnostics

No local manual files are required. Any extracted table is tagged by evidence tier.
Only primary machine-readable public tables are allowed to confirm/falsify.
Secondary PDF/figure/source-package extractions may run exploratory diagnostics but
must not be interpreted as decisive.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse, quote

import numpy as np
import pandas as pd

from .tierb_common import (
    cache_level,
    clean_numeric_series,
    column_match_report,
    download_bytes,
    ensure_dir,
    guarded_download_bytes,
    head_metadata,
    numeric_columns,
    read_tabular_bytes,
    safe_name,
    spearman,
    to_jsonable,
    utc_now,
)


# v17 crash fix: keep autodiscovery path constants local to this module.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# ---------------------------------------------------------------------------
# Data contracts: centralize gates for parsers, reports, and final verdicts.
# ---------------------------------------------------------------------------

CONTRACTS: Dict[str, Dict[str, Any]] = {
    "T26": {
        "prediction_id": "FR3",
        "name": "ELM energy scaling",
        "required_column_groups": [
            ["E_ELM", "W_ELM", "ELM energy", "dW", "ΔW", "delta W"],
            ["P_ped", "pedestal pressure", "Wped", "pedestal energy"],
            ["V_ped", "pedestal volume", "dP/P", "dP_over_P", "deltaP"],
            ["device", "shot", "machine", "discharge"],
        ],
        "min_rows": {"primary": 20, "secondary_auto_pdf_table": 8, "secondary_auto_figure_digitized": 5, "arxiv_source_table": 8},
        "primary_only_decisive": True,
    },
    "T27": {
        "prediction_id": "FR6",
        "name": "ELM frequency / RMP-helicity proxy",
        "required_column_groups": [
            ["ELM frequency", "f_ELM", "ELM suppression", "ELM mitigation"],
            ["RMP current", "I-coil", "coil current", "phasing", "n=2", "n=3", "helicity", "H_mag"],
            ["shot", "discharge", "device", "machine"],
        ],
        "min_rows": {"primary": 15, "secondary_auto_pdf_table": 6, "secondary_auto_figure_digitized": 5, "arxiv_source_table": 6},
        "primary_only_decisive": True,
    },
    "T28": {
        "prediction_id": "FR7",
        "name": "H-mode / KSS transport margin",
        "required_column_groups": [
            ["tau_E", "TAUTH", "TAUE", "energy confinement", "H-factor", "H98"],
            ["density", "ne", "nbar", "NEL", "NEBAR"],
            ["power", "PLOSS", "PLTH", "ip", "bt", "stored_energy", "wmhd"],
            ["device", "machine", "tokamak"],
        ],
        "min_rows": {"primary": 50, "secondary_auto_pdf_table": 12, "arxiv_source_table": 12},
        "primary_only_decisive": True,
    },
    "T29": {
        "prediction_id": "FR8",
        "name": "stellarator/tokamak profile transport proxy",
        "required_column_groups": [
            ["device", "device_type", "stellarator", "tokamak", "W7-X", "W7-AS", "LHD"],
            ["radius", "rho", "r/a", "normalized flux", "profile"],
            ["Te", "Ti", "ne", "temperature", "density"],
            ["heat flux", "power flux", "diffusivity", "chi", "transport"],
        ],
        "min_rows": {"primary": 20, "secondary_auto_pdf_table": 8, "secondary_auto_figure_digitized": 6},
        "primary_only_decisive": True,
    },
    "T30": {
        "prediction_id": "FR10",
        "name": "density+curvature confinement residual",
        "required_column_groups": [
            ["tau_E", "TAUTH", "TAUE", "energy confinement", "H-factor", "H98"],
            ["density", "ne", "nbar", "NEL", "NEBAR"],
            ["elongation", "kappa", "triangularity", "delta", "q95", "R_major", "a_minor"],
            ["device", "machine", "tokamak"],
        ],
        "min_rows": {"primary": 50, "secondary_auto_pdf_table": 12, "arxiv_source_table": 12},
        "primary_only_decisive": True,
    },
    "T44": {
        "prediction_id": "EL?",
        "name": "3D NAND area/volume scaling",
        "required_column_groups": [["company", "vendor"], ["year", "generation"], ["layers", "layer count"], ["capacity", "Gb", "Gbit"], ["die area", "mm2", "mm^2"]],
        "min_rows": {"primary": 10, "html_table": 8},
        "primary_only_decisive": True,
    },
    "T45": {
        "prediction_id": "EL?",
        "name": "optical interconnect energy trend",
        "required_column_groups": [["energy per bit", "pJ/bit", "fJ/bit"], ["bandwidth", "Gbps", "bandwidth/mm"], ["link length", "distance", "reach"], ["node", "process"]],
        "min_rows": {"primary": 8, "html_table": 6},
        "primary_only_decisive": True,
    },
    "T47": {
        "prediction_id": "EL?",
        "name": "neuromorphic graph-energy audit",
        "required_column_groups": [["chip", "processor"], ["energy", "inference", "spike"], ["accuracy", "benchmark"], ["topology", "neurons", "cores"]],
        "min_rows": {"primary": 6, "html_table": 5},
        "primary_only_decisive": True,
    },
    "T50": {"prediction_id": "SE?", "name": "Casimir residual upper bound", "required_column_groups": [["residual", "pressure", "force"], ["uncertainty", "noise", "systematic"], ["distance", "separation"]], "min_rows": {"primary": 5, "secondary_auto_pdf_table": 4}, "primary_only_decisive": True},
    "T51": {"prediction_id": "SE?", "name": "optical clock drift upper bound", "required_column_groups": [["frequency", "drift", "fractional"], ["uncertainty", "systematic", "noise"], ["integration", "time", "baseline"]], "min_rows": {"primary": 5, "secondary_auto_pdf_table": 4}, "primary_only_decisive": True},
    "T52": {"prediction_id": "SE?", "name": "atom interferometer noise floor", "required_column_groups": [["noise", "sensitivity", "strain", "acceleration"], ["integration", "time", "baseline"], ["uncertainty", "systematic"]], "min_rows": {"primary": 5, "secondary_auto_pdf_table": 4}, "primary_only_decisive": True},
    "T54": {"prediction_id": "BI?", "name": "photosynthetic coherence meta-analysis", "required_column_groups": [["coherence", "lifetime", "dephasing"], ["temperature", "K"], ["complex", "system", "sample"]], "min_rows": {"primary": 8, "secondary_auto_pdf_table": 5}, "primary_only_decisive": True},
    "T57": {"prediction_id": "HEP?", "name": "cosmic ray cross-section enhancement", "required_column_groups": [["energy", "TeV", "PeV", "GeV"], ["cross section", "sigma"], ["uncertainty", "error"]], "min_rows": {"primary": 8, "hepdata_table": 5}, "primary_only_decisive": True},
    "T59": {"prediction_id": "HEP?", "name": "public HEP anomaly ledger", "required_column_groups": [["mass", "mT", "energy", "GeV", "TeV"], ["cross section", "yield", "events", "limit"], ["uncertainty", "error", "observed", "expected"]], "min_rows": {"primary": 8, "hepdata_table": 5}, "primary_only_decisive": True},
}

STRUCTURED_EXT_RE = re.compile(r"\.(csv|tsv|txt|dat|xls|xlsx|json|zip)(\?|$)", re.I)
PDF_RE = re.compile(r"\.pdf(\?|$)|application/pdf", re.I)
DATA_LINK_RE = re.compile(r"(supplement|supplementary|source[-_ ]?data|data|dataset|csv|xlsx?|tsv|zip|figshare|zenodo|osf|dryad|github|hepdata)", re.I)

# ---------------------------------------------------------------------------
# Unit normalization and sensitivity diagnostics.
# ---------------------------------------------------------------------------

UNIT_FACTORS = [
    (re.compile(r"\bMJ\b", re.I), 1e6, "J"),
    (re.compile(r"\bkJ\b", re.I), 1e3, "J"),
    (re.compile(r"\bJ\b", re.I), 1.0, "J"),
    (re.compile(r"\bfJ\s*/\s*bit\b", re.I), 1e-3, "pJ/bit"),
    (re.compile(r"\bpJ\s*/\s*bit\b", re.I), 1.0, "pJ/bit"),
    (re.compile(r"\bMW\b", re.I), 1e6, "W"),
    (re.compile(r"\bkW\b", re.I), 1e3, "W"),
    (re.compile(r"\bms\b", re.I), 1e-3, "s"),
    (re.compile(r"\bs\b", re.I), 1.0, "s"),
    (re.compile(r"10\s*\^\s*19\s*m\s*[-^]?3|10\s*19\s*/\s*m\^?3", re.I), 1e19, "m^-3"),
    (re.compile(r"cm\s*\^?2", re.I), 100.0, "mm^2"),
    (re.compile(r"mm\s*\^?2", re.I), 1.0, "mm^2"),
]


def normalize_unit_from_text(text: str) -> Dict[str, Any]:
    text = str(text or "")
    for rx, fac, unit in UNIT_FACTORS:
        if rx.search(text):
            return {"unit_original_hint": rx.pattern, "scale_factor": fac, "unit_normalized": unit}
    return {"unit_original_hint": None, "scale_factor": 1.0, "unit_normalized": None}


def sensitivity_classification(n_rows: int, x_range_dex: Optional[float] = None, min_rows: int = 10) -> Dict[str, Any]:
    if n_rows <= 0:
        status = "no_data"
    elif n_rows < min_rows:
        status = "some_data_below_min_rows"
    elif x_range_dex is not None and x_range_dex < 0.4:
        status = "enough_rows_insufficient_range"
    elif x_range_dex is not None and x_range_dex < 1.0:
        status = "sufficient_for_large_effect_only"
    else:
        status = "sufficient_for_predicted_effect"
    return {"n_rows": int(n_rows), "x_range_dex": x_range_dex, "minimum_rows": int(min_rows), "sensitivity_status": status}


def _regex_groups_from_contract(test_id: str) -> List[List[str]]:
    c = CONTRACTS.get(test_id, {})
    return [[re.escape(x) for x in group] for group in c.get("required_column_groups", [])]


def _min_rows_for(test_id: str, tier: str) -> int:
    mins = (CONTRACTS.get(test_id, {}) or {}).get("min_rows", {})
    return int(mins.get(tier) or mins.get("primary") or 5)

# ---------------------------------------------------------------------------
# Link/source discovery.
# ---------------------------------------------------------------------------


def html_links(html: str, base_url: str) -> List[Dict[str, Any]]:
    out = []
    for m in re.finditer(r"(?:href|src)=[\"']([^\"']+)[\"']", html, flags=re.I):
        u = urljoin(base_url, m.group(1))
        label = m.group(1)[:160]
        if DATA_LINK_RE.search(u) or STRUCTURED_EXT_RE.search(u) or PDF_RE.search(u):
            out.append({"url": u, "label": label, "reason": "html_href_data_like"})
    # Plain URLs embedded in text.
    for m in re.finditer(r"https?://[^\s'\"<>]+", html):
        u = m.group(0).rstrip(".,);]")
        if DATA_LINK_RE.search(u) or STRUCTURED_EXT_RE.search(u) or PDF_RE.search(u):
            out.append({"url": u, "label": u[:160], "reason": "embedded_data_like_url"})
    return _dedupe_link_dicts(out)


def _dedupe_link_dicts(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for it in items:
        u = it.get("url")
        if not u or u in seen:
            continue
        seen.add(u); out.append(it)
    return out


def doi_candidates(text: str) -> List[str]:
    # Conservative DOI pattern; trim common punctuation/HTML tails.
    out = []
    for m in re.finditer(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text or ""):
        doi = m.group(0).rstrip(".);,]}'\"<")
        if len(doi) < 8:
            continue
        out.append(doi)
    return sorted(set(out))[:12]


def arxiv_id_from_url_or_text(url: str, text: str = "") -> Optional[str]:
    joined = (url or "") + "\n" + (text or "")[:2000]
    for pat in [r"arxiv\.org/(?:abs|html|pdf|e-print)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", r"arXiv[:\s]+([0-9]{4}\.[0-9]{4,5})(?:v\d+)?"]:
        m = re.search(pat, joined, re.I)
        if m:
            return m.group(1)
    return None


def metadata_expansion_links(url: str, text: str = "") -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    aid = arxiv_id_from_url_or_text(url, text)
    if aid:
        links.append({"url": f"https://arxiv.org/e-print/{aid}", "label": f"arXiv source package {aid}", "reason": "arxiv_source_package"})
    for doi in doi_candidates(url + "\n" + text[:50000]):
        q = quote(doi, safe="")
        links.extend([
            {"url": f"https://api.crossref.org/works/{q}", "label": f"Crossref metadata {doi}", "reason": "doi_crossref_metadata"},
            {"url": f"https://api.datacite.org/dois/{q}", "label": f"DataCite metadata {doi}", "reason": "doi_datacite_metadata"},
            {"url": f"https://api.openalex.org/works/doi:{doi}", "label": f"OpenAlex metadata {doi}", "reason": "doi_openalex_metadata"},
        ])
    return _dedupe_link_dicts(links)


def links_from_metadata_json(obj: Any, base_url: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    def walk(x: Any, path: str = ""):
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if isinstance(v, str):
                    if v.startswith("http") and (DATA_LINK_RE.search(v) or STRUCTURED_EXT_RE.search(v) or PDF_RE.search(v)):
                        links.append({"url": v, "label": path + "/" + k, "reason": "metadata_url"})
                    if lk in {"download_url", "content", "self", "href", "url"} and isinstance(v, str) and v.startswith("http"):
                        if DATA_LINK_RE.search(v) or STRUCTURED_EXT_RE.search(v) or PDF_RE.search(v):
                            links.append({"url": v, "label": path + "/" + k, "reason": "metadata_download_like"})
                elif isinstance(v, (dict, list)):
                    walk(v, path + "/" + str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x[:300]):
                walk(v, path + f"[{i}]")
    walk(obj)
    return _dedupe_link_dicts(links)

# ---------------------------------------------------------------------------
# Table extraction from artifacts.
# ---------------------------------------------------------------------------


def _frames_from_pdf_pdfplumber(data: bytes, source_url: str, max_pages: int = 8) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return frames
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for pageno, page in enumerate(pdf.pages[:max_pages]):
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for ti, tab in enumerate(tables[:8]):
                    if not tab or len(tab) < 2:
                        continue
                    header = [str(x or "").strip() for x in tab[0]]
                    body = tab[1:]
                    try:
                        df = pd.DataFrame(body, columns=header)
                        df.attrs["source_url"] = source_url
                        df.attrs["evidence_tier"] = "secondary_auto_pdf_table"
                        df.attrs["pdf_page"] = pageno + 1
                        df.attrs["pdf_table_index"] = ti
                        frames.append(df)
                    except Exception:
                        pass
    except Exception:
        pass
    return frames


def _text_from_pdf(data: bytes, max_pages: int = 10) -> str:
    try:
        import pdfplumber  # type: ignore
        parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages[:max_pages]:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    pass
        return "\n".join(parts)
    except Exception:
        return ""


def _frames_from_arxiv_or_tar(data: bytes, source_url: str, max_members: int = 80) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]], str]:
    frames: List[pd.DataFrame] = []
    files: List[Dict[str, Any]] = []
    text_sample_parts: List[str] = []
    # arXiv e-print is usually tar.gz, sometimes plain gzipped TeX.
    bio = io.BytesIO(data)
    try:
        with tarfile.open(fileobj=bio, mode="r:*") as tf:
            for member in tf.getmembers()[:max_members]:
                if not member.isfile():
                    continue
                name = member.name
                files.append({"name": name, "size": member.size})
                if member.size > 5_000_000:
                    continue
                f = tf.extractfile(member)
                if not f:
                    continue
                blob = f.read()
                if re.search(r"\.(csv|tsv|dat|txt|json|xlsx?)$", name, re.I):
                    for df in read_tabular_bytes(blob, name):
                        df.attrs["source_url"] = source_url + "#" + name
                        df.attrs["evidence_tier"] = "arxiv_source_table"
                        frames.append(df)
                elif name.lower().endswith(".tex"):
                    txt = blob.decode("utf-8", errors="replace")
                    text_sample_parts.append(txt[:20000])
                    frames.extend(_frames_from_latex_tables(txt, source_url + "#" + name))
        return frames, files, "\n".join(text_sample_parts)[:200000]
    except Exception:
        pass
    # try gzipped single file
    try:
        txt = gzip.decompress(data).decode("utf-8", errors="replace")
        frames.extend(_frames_from_latex_tables(txt, source_url + "#source.tex"))
        return frames, [{"name": "source.tex.gz", "size": len(data)}], txt[:200000]
    except Exception:
        return frames, files, ""


def _frames_from_latex_tables(text: str, source_url: str) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for m in re.finditer(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, flags=re.S):
        raw = m.group(0)
        # Remove latex commands but keep row/column separators.
        body = re.sub(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}", "", raw)
        body = re.sub(r"\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]*\})", "", body)
        rows = []
        for row in re.split(r"\\\\", body):
            row = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", lambda mm: mm.group(1) or "", row)
            cols = [re.sub(r"\s+", " ", c).strip(" $\t\n{}") for c in row.split("&")]
            if len(cols) >= 2 and any(c for c in cols):
                rows.append(cols)
        if len(rows) >= 2:
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df.attrs["source_url"] = source_url
            df.attrs["evidence_tier"] = "arxiv_latex_table"
            frames.append(df)
    return frames


def _frames_from_zip(data: bytes, source_url: str) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]], str]:
    frames: List[pd.DataFrame] = []
    files: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for zi in z.infolist()[:200]:
                if zi.is_dir() or zi.file_size > 10_000_000:
                    continue
                name = zi.filename
                files.append({"name": name, "size": zi.file_size})
                blob = z.read(zi)
                if re.search(r"\.(csv|tsv|txt|dat|json|xlsx?)$", name, re.I):
                    for df in read_tabular_bytes(blob, name):
                        df.attrs["source_url"] = source_url + "#" + name
                        df.attrs["evidence_tier"] = "primary_structured_zip_member"
                        frames.append(df)
                elif name.lower().endswith(".tex"):
                    txt = blob.decode("utf-8", errors="replace")
                    text_parts.append(txt[:20000])
                    frames.extend(_frames_from_latex_tables(txt, source_url + "#" + name))
    except Exception:
        pass
    return frames, files, "\n".join(text_parts)[:200000]


def frames_from_artifact(test_id: str, data: bytes, url: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[List[pd.DataFrame], Dict[str, Any], str]:
    meta = meta or {}
    ctype = (meta.get("content_type") or "").lower()
    diag: Dict[str, Any] = {"url": url, "content_type": ctype, "extractors_tried": [], "archive_members_sample": []}
    text_sample = ""
    frames: List[pd.DataFrame] = []

    # Structured files first: primary evidence when a direct table was published.
    if STRUCTURED_EXT_RE.search(url) or any(x in ctype for x in ["csv", "excel", "spreadsheet", "json", "zip"]):
        diag["extractors_tried"].append("read_tabular_bytes")
        try:
            if url.lower().split("?")[0].endswith(".zip") or "zip" in ctype:
                zframes, zfiles, ztext = _frames_from_zip(data, url)
                frames.extend(zframes); diag["archive_members_sample"] = zfiles[:50]; text_sample += ztext
            else:
                for df in read_tabular_bytes(data, url):
                    df.attrs["source_url"] = url
                    df.attrs["evidence_tier"] = "primary_structured_public_table"
                    frames.append(df)
        except Exception as e:
            diag["structured_parse_error"] = f"{type(e).__name__}: {e}"

    # HTML table and links; HTML tables are secondary unless clearly source-data tables.
    if (b"<html" in data[:2000].lower()) or "html" in ctype:
        diag["extractors_tried"].append("html_read_tabular_bytes")
        text_sample = data.decode("utf-8", errors="replace")[:300000]
        try:
            for df in read_tabular_bytes(data, url):
                df.attrs["source_url"] = url
                df.attrs["evidence_tier"] = "html_table"
                frames.append(df)
        except Exception as e:
            diag["html_table_error"] = f"{type(e).__name__}: {e}"

    # PDF table extraction is secondary only.
    if (url.lower().split("?")[0].endswith(".pdf") or "pdf" in ctype):
        diag["extractors_tried"].append("pdfplumber_tables_secondary")
        pframes = _frames_from_pdf_pdfplumber(data, url)
        frames.extend(pframes)
        text_sample = _text_from_pdf(data, max_pages=10)
        diag["pdf_table_frames"] = len(pframes)
        # Optional vector/figure diagnostics. We record, not decide.
        diag["figure_digitization_attempt"] = _figure_vector_diagnostic(data)

    # arXiv/source packages and tarballs.
    if "arxiv.org/e-print" in url or url.lower().endswith((".tar", ".tar.gz", ".tgz")) or "x-eprint" in ctype:
        diag["extractors_tried"].append("arxiv_or_tar_source_package")
        aframes, files, txt = _frames_from_arxiv_or_tar(data, url)
        frames.extend(aframes); diag["archive_members_sample"] = (diag.get("archive_members_sample") or []) + files[:50]
        text_sample += txt

    diag["frames_extracted"] = len(frames)
    return frames, diag, text_sample


def _figure_vector_diagnostic(data: bytes) -> Dict[str, Any]:
    out = {"attempted": False, "available": False, "candidate_vector_pages": 0, "note": "exploratory only; never decisive"}
    try:
        import fitz  # PyMuPDF  # type: ignore
    except Exception:
        out["available"] = False
        return out
    out["attempted"] = True; out["available"] = True
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        cnt = 0
        for page in list(doc)[:8]:
            try:
                drawings = page.get_drawings()
                text = page.get_text("text") or ""
                if len(drawings) > 10 and re.search(r"ELM|pedestal|frequency|diffus|coherence|noise|sensitivity|efficiency", text, re.I):
                    cnt += 1
            except Exception:
                pass
        out["candidate_vector_pages"] = cnt
        if cnt:
            out["evidence_tier"] = "secondary_auto_figure_digitized_pending_axis_solver"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out

# ---------------------------------------------------------------------------
# Source-specific connectors.
# ---------------------------------------------------------------------------


def connector_candidate_urls(source_url: str, data: Optional[bytes] = None, meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    meta = meta or {}
    links: List[Dict[str, Any]] = []
    text = ""
    if data:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    # Generic HTML/source links and DOI/arXiv expansion.
    if text:
        links.extend(html_links(text, source_url))
        links.extend(metadata_expansion_links(source_url, text))
    else:
        links.extend(metadata_expansion_links(source_url, ""))
    # HEPData direct query by arXiv/DOI-ish metadata.
    aid = arxiv_id_from_url_or_text(source_url, text)
    if aid:
        links.append({"url": f"https://www.hepdata.net/search/?q={quote(aid)}&format=json", "label": "HEPData search by arXiv", "reason": "hepdata_search"})
    for doi in doi_candidates(source_url + "\n" + text[:50000]):
        links.append({"url": f"https://www.hepdata.net/search/?q={quote(doi)}&format=json", "label": "HEPData search by DOI", "reason": "hepdata_search"})
    # OSF parent container exploration when file item exposes parent link.
    if "api.osf.io" in source_url:
        links.append({"url": source_url, "label": "OSF API self/parent exploration", "reason": "osf_connector_seed"})
    return _dedupe_link_dicts(links)[:120]


def links_from_connector_json(obj: Any, url: str) -> List[Dict[str, Any]]:
    links = links_from_metadata_json(obj, url)
    # HEPData search JSON can contain records or links.
    if isinstance(obj, dict):
        # OSF file API: data[].links.download and parent/related.
        if "api.osf.io" in url and isinstance(obj.get("data"), list):
            for it in obj.get("data") or []:
                if not isinstance(it, dict):
                    continue
                attrs = it.get("attributes") or {}
                lks = it.get("links") or {}
                rel = it.get("relationships") or {}
                name = attrs.get("name") or attrs.get("materialized_path") or ""
                dl = lks.get("download")
                html = lks.get("html")
                if dl:
                    links.append({"url": dl, "label": name, "reason": "osf_download_link"})
                if html:
                    links.append({"url": html, "label": name, "reason": "osf_html_link"})
                for key in ["files", "parent_folder", "node"]:
                    try:
                        u = (((rel.get(key) or {}).get("links") or {}).get("related") or {}).get("href")
                        if u:
                            links.append({"url": u, "label": f"osf_related_{key}", "reason": "osf_related"})
                    except Exception:
                        pass
            try:
                nxt = (obj.get("links") or {}).get("next")
                if nxt:
                    links.append({"url": nxt, "label": "OSF next page", "reason": "pagination"})
            except Exception:
                pass
        # Figshare article search / article object.
        if isinstance(obj.get("files"), list):
            for f in obj["files"]:
                if isinstance(f, dict) and f.get("download_url"):
                    links.append({"url": f["download_url"], "label": f.get("name", "figshare file"), "reason": "figshare_file"})
        # Zenodo / Invenio.
        hits = (((obj.get("hits") or {}).get("hits")) or [])
        if isinstance(hits, list):
            for rec in hits:
                if not isinstance(rec, dict):
                    continue
                for f in rec.get("files") or []:
                    if isinstance(f, dict):
                        dl = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download")
                        if dl:
                            links.append({"url": dl, "label": f.get("key") or f.get("filename") or "zenodo file", "reason": "zenodo_file"})
        # HEPData search: look recursively for /record/ links.
        txt = json.dumps(obj)[:2_000_000]
        for m in re.finditer(r"https?://(?:www\.)?hepdata\.net/record/[^\"'\s<>]+", txt):
            u = m.group(0).rstrip('.,}\"')
            links.append({"url": u, "label": "HEPData record", "reason": "hepdata_record_link"})
    return _dedupe_link_dicts(links)[:120]

# ---------------------------------------------------------------------------
# Candidate table scoring, nearest miss, and output augmentation.
# ---------------------------------------------------------------------------


def score_frame(test_id: str, df: pd.DataFrame, source_url: str, tier: str) -> Dict[str, Any]:
    required = _regex_groups_from_contract(test_id)
    report = column_match_report(df, required)
    nums = numeric_columns(df)
    n_rows = int(df.shape[0])
    # Simple x-range diagnostic over the first numeric column with variability.
    x_range = None
    for c in nums[:10]:
        vals = clean_numeric_series(df[c]).dropna().astype(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) >= 3:
            try:
                x_range = float(np.log10(vals.max()) - np.log10(vals.min()))
                break
            except Exception:
                pass
    unit_hints = {str(c): normalize_unit_from_text(str(c)) for c in list(df.columns)[:60]}
    min_rows = _min_rows_for(test_id, tier)
    ok = bool(report.get("ok") and len(nums) >= 2 and n_rows >= min_rows)
    return {
        "source_url": source_url,
        "evidence_tier": tier,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in list(df.columns)[:80]],
        "numeric_columns": [str(c) for c in nums[:40]],
        "unit_normalization_hints": unit_hints,
        "physical_column_match": report,
        "sensitivity": sensitivity_classification(n_rows, x_range, min_rows=min_rows),
        "qualifies_for_model": ok,
        "confirmation_allowed": bool(tier in {"primary_structured_public_table", "primary_structured_zip_member", "hepdata_table"} and n_rows >= (CONTRACTS.get(test_id, {}).get("min_rows", {}).get("primary", 10))),
        "falsification_allowed": bool(tier in {"primary_structured_public_table", "primary_structured_zip_member", "hepdata_table"} and n_rows >= (CONTRACTS.get(test_id, {}).get("min_rows", {}).get("primary", 10))),
        "attrs": {k: v for k, v in getattr(df, "attrs", {}).items() if isinstance(v, (str, int, float, bool))},
    }


def nearest_miss(test_id: str, candidates: Sequence[Dict[str, Any]], sources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Rank by matched required groups, numeric cols, rows.
    best = None
    best_score = -1
    for c in candidates:
        match = c.get("physical_column_match") or {}
        matched = len(match.get("matched_groups") or [])
        rows = (c.get("shape") or [0])[0]
        nums = len(c.get("numeric_columns") or [])
        score = matched * 100 + min(rows, 50) + nums * 3
        if score > best_score:
            best_score = score; best = c
    if best:
        miss = best.get("physical_column_match") or {}
        return {
            "nearest_candidate_source": best.get("source_url"),
            "nearest_candidate_tier": best.get("evidence_tier"),
            "nearest_candidate_shape": best.get("shape"),
            "matched_groups": miss.get("matched_groups"),
            "missing_required_groups": miss.get("missing_groups"),
            "numeric_columns": best.get("numeric_columns"),
            "suggested_next_auto_strategy": "repository_expansion_or_pdf_table_extraction" if best.get("evidence_tier") in {"html_table", "secondary_auto_pdf_table"} else "search_associated_supplements_or_source_package",
        }
    return {
        "nearest_source": sources[0].get("url") if sources else None,
        "missing_required_groups": CONTRACTS.get(test_id, {}).get("required_column_groups"),
        "suggested_next_auto_strategy": "supplement_crawler_arxiv_source_pdf_table_extraction",
    }


def source_urls_from_result(test_id: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec_key in ["manifest_records", "downloaded_sources", "nrel_downloads"]:
        for rec in result.get(rec_key) or []:
            if isinstance(rec, dict):
                u = rec.get("url") or rec.get("source_url") or ((rec.get("manifest_row") or {}).get("url") if isinstance(rec.get("manifest_row"), dict) else None)
                lab = rec.get("label") or rec.get("source_label") or ((rec.get("manifest_row") or {}).get("label") if isinstance(rec.get("manifest_row"), dict) else None) or rec_key
                if u:
                    out.append({"url": str(u), "label": str(lab), "reason": rec_key})
                # Some records hold meta.final_url.
                meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else None
                if meta and meta.get("final_url"):
                    out.append({"url": str(meta["final_url"]), "label": str(lab) + " final", "reason": rec_key + ":final_url"})
    # If manifest CSV path exists, include selected manifest URLs by reading it is done outside from result already.
    return _dedupe_link_dicts(out)[:80]


def automated_source_scan(test_id: str, args: Any, seed_sources: Sequence[Dict[str, Any]], max_sources: int = 25, max_depth: int = 2) -> Dict[str, Any]:
    cache = cache_level(args.cache, f"v10_auto_discovery_{test_id}")
    queue = list(seed_sources)
    seen = set()
    source_records: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    schema_artifacts: List[Dict[str, Any]] = []
    depth_by_url = {s.get("url"): 0 for s in queue}
    while queue and len(source_records) < max_sources:
        src = queue.pop(0)
        url = src.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        depth = depth_by_url.get(url, 0)
        data, meta = guarded_download_bytes(url, cache / "files", timeout=getattr(args, "timeout", 45), force=getattr(args, "force", False), max_bytes=getattr(args, "max_bytes", 50_000_000), manifest_approved=True)
        rec: Dict[str, Any] = {"url": url, "label": src.get("label"), "seed_reason": src.get("reason"), "depth": depth, "meta": meta, "extracted_links": [], "artifact_diag": {}, "candidate_tables": []}
        source_records.append(rec)
        if not data:
            continue
        frames, diag, text_sample = frames_from_artifact(test_id, data, url, meta)
        rec["artifact_diag"] = diag
        # If DB variables/schema PDF, extract schema text preview and table-like variable names.
        if re.search(r"variables|schema|dictionary", url + " " + str(src.get("label")), re.I) and (url.lower().endswith(".pdf") or "pdf" in str(meta.get("content_type", "")).lower()):
            schema = extract_schema_from_text(text_sample, test_id)
            if schema.get("variables_count"):
                schema_artifacts.append(schema)
                rec["schema_extracted"] = schema
        for df in frames[:80]:
            tier = str(df.attrs.get("evidence_tier") or ("primary_structured_public_table" if STRUCTURED_EXT_RE.search(url) else "html_table"))
            sc = score_frame(test_id, df, str(df.attrs.get("source_url") or url), tier)
            candidates.append(sc); rec["candidate_tables"].append(sc)
        # Expand links only for shallow levels.
        if depth < max_depth:
            links = connector_candidate_urls(url, data, meta)
            # metadata JSON can yield links.
            try:
                if data[:1] in [b"{", b"["]:
                    obj = json.loads(data.decode("utf-8", errors="replace"))
                    links.extend(links_from_connector_json(obj, url))
            except Exception:
                pass
            links = _dedupe_link_dicts(links)
            rec["extracted_links"] = links[:80]
            for l in links:
                u = l.get("url")
                if u and u not in seen and len(queue) < max_sources * 3:
                    depth_by_url[u] = depth + 1
                    queue.append(l)
    qualifying = [c for c in candidates if c.get("qualifies_for_model")]
    primary = [c for c in qualifying if c.get("confirmation_allowed")]
    secondary = [c for c in qualifying if not c.get("confirmation_allowed")]
    contract = CONTRACTS.get(test_id, {})
    if primary:
        status = "primary_table_model_possible"
    elif secondary:
        status = "secondary_model_possible_nonprimary"
    elif candidates:
        status = "candidate_tables_found_but_missing_columns_or_power"
    elif source_records:
        status = "sources_scanned_no_candidate_tables"
    else:
        status = "no_sources_scanned"
    return {
        "version": "v10_automated_discovery_no_manual_steps",
        "generated_utc": utc_now(),
        "data_contract": contract,
        "seed_sources_count": len(seed_sources),
        "sources_scanned_count": len(source_records),
        "source_records_sample": source_records[:60],
        "candidate_table_count": len(candidates),
        "qualifying_table_count": len(qualifying),
        "primary_qualifying_table_count": len(primary),
        "secondary_qualifying_table_count": len(secondary),
        "qualifying_tables_sample": qualifying[:30],
        "schema_artifacts": schema_artifacts[:10],
        "nearest_miss": nearest_miss(test_id, candidates, source_records),
        "automated_readiness_status": status,
        "evidence_ladder": {
            "E0": "no source found",
            "E1": "source found but no usable table",
            "E2": "secondary auto-extracted table/model possible; not decisive",
            "E3": "primary machine-readable public table/model possible",
            "E4": "primary table with uncertainties/controls and adequate sensitivity",
        },
    }


def extract_schema_from_text(text: str, test_id: str) -> Dict[str, Any]:
    rows = []
    for line in (text or "").splitlines():
        ln = re.sub(r"\s+", " ", line).strip()
        if not ln or len(ln) > 300:
            continue
        # ITPA/HDB variables often look like NAME description units.
        m = re.match(r"^([A-Z][A-Z0-9_]{2,20})\s+(.{5,220})$", ln)
        if m:
            name = m.group(1)
            desc = m.group(2)
            rows.append({"variable_name": name, "description": desc[:220], "unit_hint": normalize_unit_from_text(desc).get("unit_normalized")})
    return {"test_id": test_id, "schema_source": "auto_pdf_text_schema_extraction", "variables_count": len(rows), "variables_sample": rows[:120]}

# ---------------------------------------------------------------------------
# Lightweight automatic models for extracted tables. Non-primary tiers are
# diagnostic only and cannot confirm/falsify.
# ---------------------------------------------------------------------------


def simple_model_diagnostics(test_id: str, candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # We intentionally do not rerun full per-test models here, because many tables
    # lack canonical units. This returns enough diagnostics to triage automation.
    usable = [c for c in candidates if c.get("qualifies_for_model")]
    if not usable:
        return {"status": "no_model_candidate"}
    primary = [c for c in usable if c.get("confirmation_allowed")]
    return {
        "status": "primary_model_candidate_available" if primary else "secondary_model_candidate_available_nonprimary",
        "n_candidate_tables": len(usable),
        "n_primary_candidate_tables": len(primary),
        "confirmation_allowed": bool(primary),
        "falsification_allowed": bool(primary),
        "note": "v10 auto-discovery does not promote secondary PDF/figure/source extraction to confirmation; use per-test model after table is promoted to primary machine-readable evidence.",
    }


def augment_result_with_autodiscovery(test_id: str, result: Dict[str, Any], args: Any, extra_sources: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    seeds = source_urls_from_result(test_id, result)
    if extra_sources:
        seeds.extend(extra_sources)
    # Generic fallback from manifests / known connectors if no seed sources were returned.
    if not seeds:
        # HEPData generic seeds for HEP tests; no manual input.
        if test_id in {"T57", "T59"}:
            seeds.extend([
                {"url": "https://www.hepdata.net/search/?q=ATLAS%20CMS%20Drell-Yan%20Higgs%20MET&format=json", "label": "HEPData broad exact-table search", "reason": "hepdata_connector"},
            ])
    scan = automated_source_scan(test_id, args, seeds, max_sources=30 if test_id not in {"T50", "T51", "T52"} else 18, max_depth=2)
    scan["model_diagnostics"] = simple_model_diagnostics(test_id, scan.get("qualifying_tables_sample") or [])
    result["automated_discovery_v10"] = scan
    result["quality_patch_version"] = str(result.get("quality_patch_version", "")) + "+v10_auto_discovery_no_manual_steps"
    # Upgrade readiness if automation found data, without overstating evidence.
    if scan.get("primary_qualifying_table_count", 0):
        result["readiness_status"] = "primary_auto_discovered_table_candidate"
        result["evidence_status"] = "analysis_ready_primary_auto_discovered"
    elif scan.get("secondary_qualifying_table_count", 0):
        result["readiness_status"] = "secondary_auto_extracted_table_candidate_nonprimary"
        result["evidence_status"] = "data_limited_secondary_diagnostic_available"
    else:
        # preserve stronger null/ok statuses when present, otherwise clarify data-limited mode
        if result.get("evidence_status") == "data_limited":
            result["readiness_status"] = result.get("readiness_status") or scan.get("automated_readiness_status")
    result["automated_no_manual_steps_policy"] = "All v10 discovery artifacts are downloaded or extracted automatically from public URLs; manual curated CSVs are not required for this route. Secondary auto-extracted PDF/figure tables cannot confirm/falsify."
    return result

# ---------------------------------------------------------------------------
# Auto-generated microstructure support from public metadata (no manual rows).
# ---------------------------------------------------------------------------

MICROSTRUCTURE_PATTERNS = [
    ("measured_or_explicit_nanocrystalline", re.compile(r"nano[- ]?crystalline|nanograin|grain size|\b\d+(?:\.\d+)?\s*(?:nm|um|µm)\s*grain", re.I), 0.9),
    ("composite_fiber_boundary_proxy", re.compile(r"CFRP|fiber|fibre|graphlite|composite|clearwater", re.I), 0.55),
    ("amorphous_control", re.compile(r"amorphous|polymer|epoxy|kapton|teflon|PTFE|PEEK", re.I), 0.45),
    ("bulk_crystal_or_metal_control", re.compile(r"aluminum|copper|brass|silicon|sapphire|beryllium|steel|titanium", re.I), 0.35),
]


def auto_microstructure_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for rec in result.get("downloaded_sources") or []:
        path = str(rec.get("path") or rec.get("url") or "")
        label = path
        matches = []
        for cls, rx, conf in MICROSTRUCTURE_PATTERNS:
            if rx.search(label):
                matches.append({"class": cls, "confidence": conf, "pattern": rx.pattern})
        if matches:
            best = sorted(matches, key=lambda x: x["confidence"], reverse=True)[0]
            rows.append({"source_path": path, "auto_microstructure_class": best["class"], "confidence": best["confidence"], "all_matches": matches})
    decisive = [r for r in rows if r["auto_microstructure_class"] == "measured_or_explicit_nanocrystalline" and r["confidence"] >= 0.85]
    return {
        "version": "v10_auto_generated_microstructure_manifest_no_manual_steps",
        "rows_generated": len(rows),
        "decisive_candidate_rows": len(decisive),
        "class_counts": {c: sum(1 for r in rows if r["auto_microstructure_class"] == c) for c, _, _ in MICROSTRUCTURE_PATTERNS},
        "rows_sample": rows[:80],
        "interpretation": "Auto-generated manifest is useful for triage; decisive MAT1/MAT3 language still requires enough high-confidence measured/explcit nanocrystalline rows.",
    }


def augment_material_result_v10(test_id: str, result: Dict[str, Any], args: Any) -> Dict[str, Any]:
    auto = auto_microstructure_from_result(result)
    result["auto_generated_microstructure_manifest_v10"] = auto
    # Keep v9 decisive gate: do not force decisive_ready from weak auto labels.
    q = result.get("decisive_quality_gate") or {}
    if auto.get("decisive_candidate_rows", 0) >= 10 and q.get("grain_or_nano_known_usable", 0) >= 10:
        q["auto_decisive_ready_suggestion"] = True
    else:
        q["auto_decisive_ready_suggestion"] = False
    result["decisive_quality_gate"] = q
    result["quality_patch_version"] = str(result.get("quality_patch_version", "")) + "+v10_auto_microstructure_manifest"
    return result

# ---------------------------------------------------------------------------
# Persist contracts and schemas for inspectability.
# ---------------------------------------------------------------------------


def write_contract_files(data_dir: Path) -> Dict[str, Any]:
    outdir = ensure_dir(data_dir / "contracts")
    written = []
    for tid, contract in CONTRACTS.items():
        path = outdir / f"{tid}.json"
        path.write_text(json.dumps(to_jsonable(contract), indent=2, sort_keys=True), encoding="utf-8")
        written.append(str(path))
    return {"contracts_written": len(written), "contract_paths_sample": written[:20]}


# ---------------------------------------------------------------------------
# v11 quality layer: stricter artifact typing, richer automatic extraction,
# domain-specific rejection summaries, and expanded no-manual discovery.
# ---------------------------------------------------------------------------

METADATA_API_HOST_PATTERNS = [
    re.compile(r"api\.crossref\.org/works", re.I),
    re.compile(r"api\.openalex\.org/works", re.I),
    re.compile(r"api\.datacite\.org/dois", re.I),
    re.compile(r"hepdata\.net/search/", re.I),
    re.compile(r"api\.semanticscholar\.org/", re.I),
]

PHYSICAL_TABLE_HINT_RE = re.compile(
    r"(E[_ -]?ELM|W[_ -]?ELM|pedestal|P[_ -]?ped|dP/P|RMP|I[-_ ]?coil|f[_ -]?ELM|"
    r"TAUE|TAUTH|H98|PLOSS|q95|elongation|triangularity|rho|diffusiv|heat flux|"
    r"energy per bit|pJ/bit|fJ/bit|bandwidth|die area|layers|capacity|noise|sensitivity|"
    r"coherence|lifetime|cross section|observed|expected|uncertainty)", re.I
)


def is_metadata_record_url(url: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    u = str(url or "")
    if any(rx.search(u) for rx in METADATA_API_HOST_PATTERNS):
        return True
    ctype = str((meta or {}).get("content_type") or "").lower()
    # JSON from repository APIs can be physical if it is a data file; only known bibliographic/search APIs are metadata.
    return False


def artifact_role(url: str, meta: Optional[Dict[str, Any]] = None) -> str:
    if is_metadata_record_url(url, meta):
        return "metadata_record"
    u = str(url or "").lower()
    ctype = str((meta or {}).get("content_type") or "").lower()
    if "arxiv.org/e-print" in u or u.endswith((".tar", ".tar.gz", ".tgz")):
        return "source_package"
    if u.endswith(".zip") or "zip" in ctype:
        return "archive"
    if u.endswith((".csv", ".tsv", ".dat", ".txt", ".xls", ".xlsx")) or any(x in ctype for x in ["csv", "excel", "spreadsheet"]):
        return "primary_table_file"
    if u.endswith(".json") or "application/json" in ctype:
        return "json_candidate"
    if u.endswith(".pdf") or "pdf" in ctype:
        return "pdf"
    if "html" in ctype or u.endswith((".html", ".htm")):
        return "html"
    return "unknown"


def _frames_from_pdf_camelot(data: bytes, source_url: str, tmp_dir: Optional[Path] = None, max_pages: int = 12) -> List[pd.DataFrame]:
    """Optional Camelot extraction. Disabled by default because it can hang on some PDFs/Java setups.
    Enable with CCDR_ENABLE_CAMELOT=1.
    """
    frames: List[pd.DataFrame] = []
    if os.environ.get("CCDR_ENABLE_CAMELOT", "0") not in {"1", "true", "TRUE", "yes"}:
        return frames
    try:
        import camelot  # type: ignore
    except Exception:
        return frames
    tmp_dir = ensure_dir(tmp_dir or Path(".") / ".tmp_pdf_tables")
    pdf_path = tmp_dir / (safe_name(source_url)[-80:] + ".pdf")
    try:
        pdf_path.write_bytes(data)
        pages = ",".join(str(i) for i in range(1, max_pages + 1))
        for flavor in ["lattice", "stream"]:
            try:
                tables = camelot.read_pdf(str(pdf_path), pages=pages, flavor=flavor)
            except Exception:
                continue
            for i, table in enumerate(tables[:20]):
                try:
                    df = table.df
                    if df is None or df.shape[0] < 2 or df.shape[1] < 2:
                        continue
                    # Promote first row to header when it looks header-like.
                    header = [str(x).strip() for x in list(df.iloc[0])]
                    body = df.iloc[1:].copy()
                    body.columns = header
                    body.attrs["source_url"] = source_url
                    body.attrs["evidence_tier"] = "secondary_auto_pdf_table"
                    body.attrs["pdf_extractor"] = f"camelot_{flavor}"
                    body.attrs["pdf_table_index"] = i
                    frames.append(body)
                except Exception:
                    pass
    except Exception:
        pass
    return frames


def _frames_from_pdf_text_lines(data: bytes, source_url: str, max_pages: int = 20) -> List[pd.DataFrame]:
    """Conservative text-table fallback for PDFs with aligned columns."""
    frames: List[pd.DataFrame] = []
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return frames
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for pageno, page in enumerate(pdf.pages[:max_pages]):
                try:
                    text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                except Exception:
                    text = ""
                if not PHYSICAL_TABLE_HINT_RE.search(text):
                    continue
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                # Find blocks with repeated multi-space or tab separators and numeric content.
                block: List[List[str]] = []
                for ln in lines:
                    if re.search(r"\d", ln) and ("  " in ln or "\t" in ln):
                        cells = [c.strip() for c in re.split(r"\t+|\s{2,}", ln) if c.strip()]
                        if len(cells) >= 2:
                            block.append(cells)
                    elif len(block) >= 3:
                        frames.extend(_block_to_frames(block, source_url, pageno + 1))
                        block = []
                    else:
                        block = []
                if len(block) >= 3:
                    frames.extend(_block_to_frames(block, source_url, pageno + 1))
    except Exception:
        pass
    return frames


def _block_to_frames(block: List[List[str]], source_url: str, pageno: int) -> List[pd.DataFrame]:
    out: List[pd.DataFrame] = []
    try:
        width = max(len(r) for r in block)
        rows = [r + [""] * (width - len(r)) for r in block]
        # Use a synthetic header if no obvious header exists; this still lets nearest-miss report numeric columns.
        header = rows[0]
        if sum(bool(re.search(r"[A-Za-z]", c)) for c in header) < max(1, width // 2):
            header = [f"col_{i+1}" for i in range(width)]
            body = rows
        else:
            body = rows[1:]
        df = pd.DataFrame(body, columns=header)
        if df.shape[0] >= 2 and df.shape[1] >= 2:
            df.attrs["source_url"] = source_url
            df.attrs["evidence_tier"] = "secondary_auto_pdf_table"
            df.attrs["pdf_extractor"] = "text_line_block"
            df.attrs["pdf_page"] = pageno
            out.append(df)
    except Exception:
        pass
    return out


def _frames_from_latex_tables_v11(text: str, source_url: str) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    # Handle tabular, tabular*, longtable, and deluxetable-ish bodies.
    patterns = [
        r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}",
        r"\\begin\{longtable\}.*?\\end\{longtable\}",
        r"\\begin\{deluxetable\}.*?\\end\{deluxetable\}",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.S):
            raw = m.group(0)
            body = re.sub(r"\\begin\{[^}]+\}(?:\[[^\]]*\])?(?:\{[^}]*\})?|\\end\{[^}]+\}", "", raw)
            body = re.sub(r"\\(?:hline|toprule|midrule|bottomrule|tablehead|startdata|enddata|colnumbers|cline\{[^}]*\})", "", body)
            body = re.sub(r"\\colhead\{([^{}]*)\}", r"\1", body)
            rows = []
            for row in re.split(r"\\\\", body):
                row = re.sub(r"%.*", "", row)
                row = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", lambda mm: mm.group(1) or "", row)
                cols = [re.sub(r"\s+", " ", c).strip(" $\t\n{}") for c in row.split("&")]
                if len(cols) >= 2 and any(c for c in cols):
                    rows.append(cols)
            if len(rows) >= 2:
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                df = pd.DataFrame(rows[1:], columns=rows[0])
                df.attrs["source_url"] = source_url
                df.attrs["evidence_tier"] = "arxiv_latex_table"
                frames.append(df)
    return frames

# Override the v10 LaTeX parser with the richer v11 parser.
_frames_from_latex_tables = _frames_from_latex_tables_v11


def frames_from_artifact(test_id: str, data: bytes, url: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[List[pd.DataFrame], Dict[str, Any], str]:
    """v11 override: metadata APIs are link sources only, not candidate tables."""
    meta = meta or {}
    ctype = (meta.get("content_type") or "").lower()
    role = artifact_role(url, meta)
    diag: Dict[str, Any] = {"url": url, "content_type": ctype, "artifact_role": role, "extractors_tried": [], "archive_members_sample": []}
    text_sample = ""
    frames: List[pd.DataFrame] = []

    if role == "metadata_record":
        diag["extractors_tried"].append("metadata_link_extraction_only")
        try:
            text_sample = data.decode("utf-8", errors="replace")[:500000]
        except Exception:
            text_sample = ""
        diag["frames_extracted"] = 0
        diag["metadata_not_physical_table"] = True
        return frames, diag, text_sample

    if role in {"primary_table_file", "json_candidate", "archive"}:
        diag["extractors_tried"].append("read_tabular_bytes")
        try:
            if role == "archive":
                zframes, zfiles, ztext = _frames_from_zip(data, url)
                frames.extend(zframes); diag["archive_members_sample"] = zfiles[:80]; text_sample += ztext
            else:
                for df in read_tabular_bytes(data, url):
                    df.attrs["source_url"] = url
                    # JSON candidates are primary only when not bibliographic/search metadata and physical columns pass later.
                    df.attrs["evidence_tier"] = "primary_structured_public_table"
                    frames.append(df)
        except Exception as e:
            diag["structured_parse_error"] = f"{type(e).__name__}: {e}"

    if role == "html" or (b"<html" in data[:2000].lower()):
        diag["extractors_tried"].append("html_read_tabular_bytes")
        text_sample = data.decode("utf-8", errors="replace")[:500000]
        try:
            for df in read_tabular_bytes(data, url):
                df.attrs["source_url"] = url
                # HTML is primary only if URL itself is a data endpoint; generic pages are secondary/html.
                df.attrs["evidence_tier"] = "html_table"
                frames.append(df)
        except Exception as e:
            diag["html_table_error"] = f"{type(e).__name__}: {e}"

    if role == "pdf":
        diag["extractors_tried"].append("pdfplumber_tables_secondary")
        pframes = _frames_from_pdf_pdfplumber(data, url, max_pages=_pdf_page_budget_v12())
        frames.extend(pframes)
        diag["pdfplumber_table_frames"] = len(pframes)
        cframes = _frames_from_pdf_camelot(data, url)
        if cframes:
            diag["extractors_tried"].append("camelot_secondary")
            frames.extend(cframes)
        diag["camelot_table_frames"] = len(cframes)
        tframes = _frames_from_pdf_text_lines(data, url, max_pages=_pdf_page_budget_v12())
        if tframes:
            diag["extractors_tried"].append("pdf_text_line_tables_secondary")
            frames.extend(tframes)
        diag["text_line_table_frames"] = len(tframes)
        text_sample = _text_from_pdf(data, max_pages=_pdf_page_budget_v12())
        diag["figure_digitization_attempt"] = _figure_vector_diagnostic(data)

    if role == "source_package":
        diag["extractors_tried"].append("arxiv_or_tar_source_package")
        aframes, files, txt = _frames_from_arxiv_or_tar(data, url, max_members=220)
        frames.extend(aframes); diag["archive_members_sample"] = (diag.get("archive_members_sample") or []) + files[:100]
        text_sample += txt

    diag["frames_extracted"] = len(frames)
    return frames, diag, text_sample


def score_frame(test_id: str, df: pd.DataFrame, source_url: str, tier: str) -> Dict[str, Any]:
    required = _regex_groups_from_contract(test_id)
    report = column_match_report(df, required)
    nums = numeric_columns(df)
    n_rows = int(df.shape[0])
    x_range = None
    for c in nums[:10]:
        vals = clean_numeric_series(df[c]).dropna().astype(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) >= 3:
            try:
                x_range = float(np.log10(vals.max()) - np.log10(vals.min()))
                break
            except Exception:
                pass
    unit_hints = {str(c): normalize_unit_from_text(str(c)) for c in list(df.columns)[:80]}
    min_rows = _min_rows_for(test_id, tier)
    role = artifact_role(source_url, {"content_type": ""})
    # Known metadata APIs never qualify and are not allowed to inflate candidate evidence.
    metadata_only = is_metadata_record_url(source_url)
    has_physical = bool(report.get("ok"))
    enough_numeric = len(nums) >= 2
    enough_rows = n_rows >= min_rows
    ok = bool((not metadata_only) and has_physical and enough_numeric and enough_rows)
    primary_tiers = {"primary_structured_public_table", "primary_structured_zip_member", "hepdata_table"}
    primary_allowed = bool(ok and tier in primary_tiers)
    rejection_reasons = []
    if metadata_only:
        rejection_reasons.append("metadata_record_not_physical_table")
    if not has_physical:
        rejection_reasons.append("missing_required_physical_column_groups")
    if not enough_numeric:
        rejection_reasons.append("too_few_numeric_columns")
    if not enough_rows:
        rejection_reasons.append("below_min_rows_for_contract")
    if tier not in primary_tiers:
        rejection_reasons.append("secondary_or_nonprimary_evidence_tier")
    return {
        "source_url": source_url,
        "artifact_role": role,
        "evidence_tier": "metadata_record" if metadata_only else tier,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in list(df.columns)[:100]],
        "numeric_columns": [str(c) for c in nums[:60]],
        "unit_normalization_hints": unit_hints,
        "physical_column_match": report,
        "sensitivity": sensitivity_classification(n_rows, x_range, min_rows=min_rows),
        "qualifies_for_model": ok,
        "confirmation_allowed": primary_allowed,
        "falsification_allowed": primary_allowed,
        "rejection_reasons": rejection_reasons,
        "attrs": {k: v for k, v in getattr(df, "attrs", {}).items() if isinstance(v, (str, int, float, bool))},
    }


def links_from_connector_json(obj: Any, url: str) -> List[Dict[str, Any]]:
    """v11 override: extract links, including HEPData table downloads, but do not treat search JSON as tables."""
    links = links_from_metadata_json(obj, url)
    if isinstance(obj, dict):
        if "api.osf.io" in url and isinstance(obj.get("data"), list):
            for it in obj.get("data") or []:
                if not isinstance(it, dict):
                    continue
                attrs = it.get("attributes") or {}
                lks = it.get("links") or {}
                rel = it.get("relationships") or {}
                name = attrs.get("name") or attrs.get("materialized_path") or ""
                dl = lks.get("download")
                html = lks.get("html")
                if dl:
                    links.append({"url": dl, "label": name, "reason": "osf_download_link"})
                if html:
                    links.append({"url": html, "label": name, "reason": "osf_html_link"})
                for key in ["files", "parent_folder", "node"]:
                    try:
                        u = (((rel.get(key) or {}).get("links") or {}).get("related") or {}).get("href")
                        if u:
                            links.append({"url": u, "label": f"osf_related_{key}", "reason": "osf_related"})
                    except Exception:
                        pass
            try:
                nxt = (obj.get("links") or {}).get("next")
                if nxt:
                    links.append({"url": nxt, "label": "OSF next page", "reason": "pagination"})
            except Exception:
                pass
        if isinstance(obj.get("files"), list):
            for f in obj["files"]:
                if isinstance(f, dict) and f.get("download_url"):
                    links.append({"url": f["download_url"], "label": f.get("name", "figshare file"), "reason": "figshare_file"})
        hits = (((obj.get("hits") or {}).get("hits")) or [])
        if isinstance(hits, list):
            for rec in hits:
                if not isinstance(rec, dict):
                    continue
                for f in rec.get("files") or []:
                    if isinstance(f, dict):
                        dl = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download")
                        if dl:
                            links.append({"url": dl, "label": f.get("key") or f.get("filename") or "zenodo file", "reason": "zenodo_file"})
        txt = json.dumps(obj)[:3_000_000]
        # HEPData record and table download patterns.
        for m in re.finditer(r"https?://(?:www\.)?hepdata\.net/record/[^\"'\s<>]+", txt):
            u = m.group(0).rstrip('.,}\"')
            links.append({"url": u, "label": "HEPData record", "reason": "hepdata_record_link"})
        for m in re.finditer(r"https?://(?:www\.)?hepdata\.net/download/table/[^\"'\s<>]+/(?:csv|yaml|json)", txt):
            u = m.group(0).rstrip('.,}\"')
            links.append({"url": u, "label": "HEPData table download", "reason": "hepdata_table_download"})
        # Sometimes search JSON has recid/table_num rather than direct links.
        for recid in sorted(set(re.findall(r'"recid"\s*:\s*"?(\d+)"?', txt)))[:20]:
            links.append({"url": f"https://www.hepdata.net/record/{recid}", "label": f"HEPData record {recid}", "reason": "hepdata_record_from_json"})
    return _dedupe_link_dicts(links)[:160]


def connector_candidate_urls(source_url: str, data: Optional[bytes] = None, meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    meta = meta or {}
    links: List[Dict[str, Any]] = []
    text = ""
    if data:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    if text:
        links.extend(html_links(text, source_url))
        links.extend(metadata_expansion_links(source_url, text))
    else:
        links.extend(metadata_expansion_links(source_url, ""))
    aid = arxiv_id_from_url_or_text(source_url, text)
    if aid:
        links.append({"url": f"https://arxiv.org/e-print/{aid}", "label": f"arXiv source package {aid}", "reason": "arxiv_source_package"})
    for doi in doi_candidates(source_url + "\n" + text[:50000]):
        links.append({"url": f"https://www.hepdata.net/search/?q={quote(doi)}&format=json", "label": "HEPData search by DOI", "reason": "hepdata_search"})
    if "api.osf.io" in source_url:
        links.append({"url": source_url, "label": "OSF API self/parent exploration", "reason": "osf_connector_seed"})
    # Domain-specific repository search expansions for data-limited groups.
    # These are query/API sources, not evidence; they can only yield artifact links.
    if any(k in source_url.lower() + text[:2000].lower() for k in ["elm", "rmp", "pedestal", "w7-x", "stellarator", "confinement"]):
        for q in ["ELM pedestal energy supplementary data", "RMP ELM frequency supplementary data", "W7-X profile transport data"]:
            links.append({"url": f"https://zenodo.org/api/records/?q={quote(q)}&size=10", "label": f"Zenodo search {q}", "reason": "zenodo_search"})
    return _dedupe_link_dicts(links)[:180]


def nearest_miss(test_id: str, candidates: Sequence[Dict[str, Any]], sources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    best = None; best_score = -1
    for c in candidates:
        # Ignore pure metadata for nearest physical miss unless no other candidates exist.
        metadata_penalty = -1000 if "metadata_record_not_physical_table" in (c.get("rejection_reasons") or []) else 0
        match = c.get("physical_column_match") or {}
        matched = len(match.get("matched_groups") or [])
        rows = (c.get("shape") or [0])[0]
        nums = len(c.get("numeric_columns") or [])
        score = metadata_penalty + matched * 100 + min(rows, 50) + nums * 3
        if score > best_score:
            best_score = score; best = c
    if best:
        miss = best.get("physical_column_match") or {}
        return {
            "nearest_candidate_source": best.get("source_url"),
            "nearest_candidate_tier": best.get("evidence_tier"),
            "nearest_candidate_shape": best.get("shape"),
            "matched_groups": miss.get("matched_groups"),
            "missing_required_groups": miss.get("missing_groups"),
            "numeric_columns": best.get("numeric_columns"),
            "rejection_reasons": best.get("rejection_reasons"),
            "suggested_next_auto_strategy": _suggest_strategy_from_rejections(best),
        }
    return {
        "nearest_source": sources[0].get("url") if sources else None,
        "missing_required_groups": CONTRACTS.get(test_id, {}).get("required_column_groups"),
        "suggested_next_auto_strategy": "supplement_crawler_arxiv_source_pdf_table_extraction",
    }


def _suggest_strategy_from_rejections(c: Dict[str, Any]) -> str:
    reasons = set(c.get("rejection_reasons") or [])
    if "metadata_record_not_physical_table" in reasons:
        return "follow_metadata_links_only_do_not_parse_metadata_as_data"
    if "missing_required_physical_column_groups" in reasons:
        return "search_associated_supplements_or_source_package_with_required_column_contract"
    if "below_min_rows_for_contract" in reasons:
        return "find_same_schema_tables_or_expand_repository_search"
    if "secondary_or_nonprimary_evidence_tier" in reasons:
        return "locate_primary_machine_readable_supplement_for_decisive_verdict"
    return "repository_expansion_or_domain_specific_parser"


def automated_source_scan(test_id: str, args: Any, seed_sources: Sequence[Dict[str, Any]], max_sources: int = 25, max_depth: int = 2) -> Dict[str, Any]:
    cache = cache_level(args.cache, f"v11_auto_discovery_{test_id}")
    queue = list(seed_sources) + additional_seed_sources_v11(test_id)
    seen = set(); source_records: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []; metadata_records: List[Dict[str, Any]] = []
    schema_artifacts: List[Dict[str, Any]] = []
    depth_by_url = {s.get("url"): 0 for s in queue}
    while queue and len(source_records) < max_sources:
        src = queue.pop(0); url = src.get("url")
        if not url or url in seen or str(url).startswith("data:"):
            continue
        seen.add(url); depth = depth_by_url.get(url, 0)
        data, meta = guarded_download_bytes(url, cache / "files", timeout=getattr(args, "timeout", 45), force=getattr(args, "force", False), max_bytes=getattr(args, "max_bytes", 60_000_000), manifest_approved=True)
        rec: Dict[str, Any] = {"url": url, "label": src.get("label"), "seed_reason": src.get("reason"), "depth": depth, "meta": meta, "extracted_links": [], "artifact_diag": {}, "candidate_tables": []}
        source_records.append(rec)
        if not data:
            continue
        frames, diag, text_sample = frames_from_artifact(test_id, data, url, meta)
        rec["artifact_diag"] = diag
        if diag.get("metadata_not_physical_table"):
            metadata_records.append({"url": url, "label": src.get("label"), "reason": src.get("reason"), "artifact_role": "metadata_record"})
        if re.search(r"variables|schema|dictionary", url + " " + str(src.get("label")), re.I) and (url.lower().endswith(".pdf") or "pdf" in str(meta.get("content_type", "")).lower()):
            schema = extract_schema_from_text(text_sample, test_id)
            if schema.get("variables_count"):
                schema_artifacts.append(schema); rec["schema_extracted"] = schema
        for df in frames[:100]:
            tier = str(df.attrs.get("evidence_tier") or ("primary_structured_public_table" if STRUCTURED_EXT_RE.search(url) else "html_table"))
            sc = score_frame(test_id, df, str(df.attrs.get("source_url") or url), tier)
            # Do not count metadata records as candidate tables.
            if "metadata_record_not_physical_table" in sc.get("rejection_reasons", []):
                metadata_records.append({"url": sc.get("source_url"), "columns_sample": sc.get("columns", [])[:12], "reason": "parsed_metadata_frame_rejected"})
                continue
            candidates.append(sc); rec["candidate_tables"].append(sc)
        if depth < max_depth:
            links = connector_candidate_urls(url, data, meta)
            try:
                if data[:1] in [b"{", b"["]:
                    obj = json.loads(data.decode("utf-8", errors="replace"))
                    links.extend(links_from_connector_json(obj, url))
            except Exception:
                pass
            links = _dedupe_link_dicts(links)
            rec["extracted_links"] = links[:100]
            for l in links:
                u = l.get("url")
                if u and u not in seen and len(queue) < max_sources * 4:
                    depth_by_url[u] = depth + 1; queue.append(l)
    qualifying = [c for c in candidates if c.get("qualifies_for_model")]
    primary = [c for c in qualifying if c.get("confirmation_allowed")]
    secondary = [c for c in qualifying if not c.get("confirmation_allowed")]
    if primary:
        status = "primary_table_model_possible"
    elif secondary:
        status = "secondary_model_possible_nonprimary"
    elif candidates:
        status = "candidate_tables_found_but_missing_columns_or_power"
    elif source_records:
        status = "sources_scanned_no_candidate_tables"
    else:
        status = "no_sources_scanned"
    return {
        "version": "v11_automated_discovery_strict_artifact_typing_no_manual_steps",
        "generated_utc": utc_now(),
        "data_contract": CONTRACTS.get(test_id, {}),
        "seed_sources_count": len(seed_sources),
        "additional_auto_seed_sources_count": len(additional_seed_sources_v11(test_id)),
        "sources_scanned_count": len(source_records),
        "source_records_sample": source_records[:60],
        "metadata_records_seen_count": len(metadata_records),
        "metadata_records_sample": metadata_records[:20],
        "candidate_table_count": len(candidates),
        "qualifying_table_count": len(qualifying),
        "primary_qualifying_table_count": len(primary),
        "secondary_qualifying_table_count": len(secondary),
        "candidate_rejection_summary": rejection_summary(candidates),
        "candidate_tables_sample": candidates[:30],
        "qualifying_tables_sample": qualifying[:30],
        "schema_artifacts": schema_artifacts[:10],
        "nearest_miss": nearest_miss(test_id, candidates, source_records),
        "automated_readiness_status": status,
        "strict_verdict_rule_v11": "Only primary machine-readable physical tables (E3/E4) may confirm/falsify. Metadata records are link sources only. PDF/HTML/arXiv extracted tables are secondary unless they are explicit source-data files.",
        "evidence_ladder": {
            "E0": "no source found",
            "E1": "source found but no usable physical table",
            "E2": "secondary auto-extracted table/model possible; not decisive",
            "E3": "primary machine-readable public physical table/model possible",
            "E4": "primary table with uncertainties/controls and adequate sensitivity",
        },
    }


def rejection_summary(candidates: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in candidates:
        for r in c.get("rejection_reasons") or []:
            out[r] = out.get(r, 0) + 1
    return out


def additional_seed_sources_v11(test_id: str) -> List[Dict[str, Any]]:
    """More automated public-data seeds. These are discovery sources, not evidence."""
    seeds: List[Dict[str, Any]] = []
    if test_id in {"T26", "T27", "T29"}:
        queries = {
            "T26": ["ELM pedestal energy loss supplementary data", "type I ELM energy pedestal pressure table"],
            "T27": ["RMP ELM frequency coil current supplementary data", "ELM suppression phasing n=2 n=3 data"],
            "T29": ["W7-X profile transport data Te ne heat flux", "stellarator tokamak confinement profile transport data"],
        }[test_id]
        for q in queries:
            seeds.append({"url": f"https://zenodo.org/api/records/?q={quote(q)}&size=10", "label": f"Zenodo API search: {q}", "reason": "v11_zenodo_query_seed"})
            seeds.append({"url": f"https://api.figshare.com/v2/articles/search", "label": f"Figshare POST search unavailable by GET for: {q}", "reason": "v11_figshare_search_note"})
    if test_id in {"T44", "T45", "T47"}:
        # Vendor/spec pages that often contain usable structured tables or links.
        if test_id == "T44":
            seeds += [
                {"url": "https://en.wikichip.org/wiki/3d_nand", "label": "WikiChip 3D NAND", "reason": "v11_electronics_seed"},
                {"url": "https://en.wikipedia.org/wiki/3D_XPoint", "label": "Wikipedia 3D XPoint", "reason": "v11_electronics_seed"},
                {"url": "https://www.techinsights.com/blog", "label": "TechInsights blog discovery", "reason": "v11_electronics_seed"},
            ]
        elif test_id == "T45":
            seeds += [
                {"url": "https://irds.ieee.org/editions", "label": "IRDS editions", "reason": "v11_electronics_seed"},
                {"url": "https://www.opencompute.org/wiki/Networking/Optics", "label": "OCP optics", "reason": "v11_electronics_seed"},
            ]
        elif test_id == "T47":
            seeds += [
                {"url": "https://en.wikichip.org/wiki/intel/loihi", "label": "WikiChip Loihi", "reason": "v11_electronics_seed"},
                {"url": "https://en.wikichip.org/wiki/ibm/truenorth", "label": "WikiChip TrueNorth", "reason": "v11_electronics_seed"},
                {"url": "https://open-neuromorphic.org/neuromorphic-computing/hardware/", "label": "Open Neuromorphic hardware", "reason": "v11_electronics_seed"},
            ]
    if test_id in {"T50", "T51", "T52"}:
        queries = {
            "T50": "Casimir force residual uncertainty table data",
            "T51": "optical clock fractional frequency drift uncertainty budget table",
            "T52": "atom interferometer sensitivity noise floor table",
        }
        q = queries[test_id]
        seeds.append({"url": f"https://zenodo.org/api/records/?q={quote(q)}&size=10", "label": f"Zenodo metrology search: {q}", "reason": "v11_metrology_seed"})
    if test_id in {"T57", "T59"}:
        for q in ["ATLAS MET observed expected", "CMS Drell-Yan 1 TeV", "ATLAS di-Higgs observed expected", "cosmic ray cross section"]:
            seeds.append({"url": f"https://www.hepdata.net/search/?q={quote(q)}&format=json", "label": f"HEPData search: {q}", "reason": "v11_hepdata_query_seed"})
    if test_id == "T54":
        for q in ["photosynthetic coherence lifetime supplementary data", "2D electronic spectroscopy coherence lifetime data"]:
            seeds.append({"url": f"https://zenodo.org/api/records/?q={quote(q)}&size=10", "label": f"Zenodo coherence search: {q}", "reason": "v11_bio_seed"})
    return seeds


def augment_result_with_autodiscovery(test_id: str, result: Dict[str, Any], args: Any, extra_sources: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    seeds = source_urls_from_result(test_id, result)
    if extra_sources:
        seeds.extend(extra_sources)
    if not seeds and test_id in {"T57", "T59"}:
        seeds.append({"url": "https://www.hepdata.net/search/?q=ATLAS%20CMS%20Drell-Yan%20Higgs%20MET&format=json", "label": "HEPData broad exact-table search", "reason": "hepdata_connector"})
    scan = automated_source_scan(test_id, args, seeds, max_sources=42 if test_id not in {"T50", "T51", "T52"} else 24, max_depth=2)
    scan["model_diagnostics"] = simple_model_diagnostics(test_id, scan.get("qualifying_tables_sample") or [])
    result["automated_discovery_v11"] = scan
    # Retain v10 key for backward compatibility but point to v11 result.
    result["automated_discovery_v10"] = scan
    result["quality_patch_version"] = str(result.get("quality_patch_version", "")) + "+v11_strict_artifact_typing_more_auto_sources"
    if scan.get("primary_qualifying_table_count", 0):
        result["readiness_status"] = "primary_auto_discovered_physical_table_candidate"
        result["evidence_status"] = "analysis_ready_primary_auto_discovered"
    elif scan.get("secondary_qualifying_table_count", 0):
        result["readiness_status"] = "secondary_auto_extracted_physical_table_candidate_nonprimary"
        result["evidence_status"] = "data_limited_secondary_diagnostic_available"
    else:
        if result.get("evidence_status") == "data_limited":
            result["readiness_status"] = result.get("readiness_status") or scan.get("automated_readiness_status")
    result["automated_no_manual_steps_policy"] = "All v11 discovery artifacts are downloaded or extracted automatically from public URLs. Metadata APIs are link sources only, not evidence. Secondary auto-extracted PDF/figure/arXiv tables cannot confirm/falsify."
    return result


def auto_microstructure_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """v11: richer auto triage, exact path labels, and reference-harvest suggestions without manual rows."""
    rows = []
    for rec in result.get("downloaded_sources") or []:
        path = str(rec.get("path") or rec.get("url") or "")
        label = path.lower()
        matches = []
        for cls, rx, conf in MICROSTRUCTURE_PATTERNS:
            if rx.search(label):
                matches.append({"class": cls, "confidence": conf, "pattern": rx.pattern})
        # Infer measured/explicit only from strong terms, not merely directory names.
        if re.search(r"nano[-_ ]?crystal|grain[_ -]?size|\b\d+(?:\.\d+)?\s*(nm|um|µm)", label, re.I):
            matches.append({"class": "measured_or_explicit_nanocrystalline", "confidence": 0.92, "pattern": "strong_filename_microstructure"})
        if matches:
            best = sorted(matches, key=lambda x: x["confidence"], reverse=True)[0]
            rows.append({
                "source_path": path,
                "auto_microstructure_class": best["class"],
                "confidence": best["confidence"],
                "all_matches": matches,
                "reference_harvest_strategy": "download sibling references.txt / README and DOI supplements when available",
            })
    decisive = [r for r in rows if r["auto_microstructure_class"] == "measured_or_explicit_nanocrystalline" and r["confidence"] >= 0.90]
    return {
        "version": "v11_auto_generated_microstructure_manifest_no_manual_steps",
        "rows_generated": len(rows),
        "decisive_candidate_rows": len(decisive),
        "class_counts": {c: sum(1 for r in rows if r["auto_microstructure_class"] == c) for c, _, _ in MICROSTRUCTURE_PATTERNS},
        "rows_sample": rows[:100],
        "next_auto_harvest": ["CMB-S4 references.txt siblings", "paper DOI supplements", "GitHub README/source metadata"],
        "interpretation": "Auto-generated manifest is triage only. Decisive MAT1/MAT3 language still requires enough high-confidence measured/explicit nanocrystalline rows matched to kappa tables.",
    }

# ---------------------------------------------------------------------------
# v12 quality layer: strict repository artifact typing, file-level repository
# downloads, physical-table candidate counts, exact connector seeds, and
# stronger automatic microstructure enrichment. These definitions intentionally
# override v11 functions above.
# ---------------------------------------------------------------------------

import urllib.request
import urllib.parse
import time
import os

DATA_ROOT_V12 = Path(__file__).resolve().parents[1] / "data"

REPOSITORY_METADATA_URL_RE = re.compile(
    r"(zenodo\.org/api/records(?:/\d+)?(?:/versions/[^/?]+|/media-files|/files)?(?:\?|$)|"
    r"api\.figshare\.com/v2/articles(?:/search|/\d+)?(?:\?|$)|"
    r"api\.osf\.io/v2/|"
    r"hepdata\.net/(?:search|record)(?:/|\?|$)|"
    r"api\.crossref\.org/works|api\.openalex\.org/works|api\.datacite\.org/dois|"
    r"api\.semanticscholar\.org/)",
    re.I,
)

ACTUAL_FILE_URL_RE = re.compile(
    r"(zenodo\.org/api/files/|zenodo\.org/record/\d+/files/|zenodo\.org/records/\d+/files/|"
    r"osf\.io/download/|figshare\.com/ndownloader/files/|hepdata\.net/download/table/|"
    r"raw\.githubusercontent\.com/|github\.com/.*/raw/|github\.com/.*/releases/download/)",
    re.I,
)



def _pdf_page_budget_v12(default: int = 4) -> int:
    try:
        return max(1, min(25, int(os.environ.get("CCDR_PDF_TABLE_PAGES", str(default)))))
    except Exception:
        return default

PHYSICAL_MIN_COLS_BY_TEST = {
    "T26": ["elm", "ped", "shot", "device", "discharge", "machine", "dW", "W_ELM", "P_ped"],
    "T27": ["elm", "rmp", "coil", "phasing", "frequency", "shot", "device", "discharge"],
    "T28": ["taue", "tauth", "h98", "density", "nbar", "ploss", "ip", "bt", "device"],
    "T29": ["w7", "stellarator", "tokamak", "rho", "profile", "te", "ne", "heat", "diffus"],
    "T30": ["taue", "h98", "density", "elong", "triang", "q95", "kappa", "device"],
    "T44": ["nand", "layers", "capacity", "die", "area", "gb", "company", "year"],
    "T45": ["energy", "bit", "bandwidth", "link", "reach", "node", "pj", "fj"],
    "T47": ["loihi", "truenorth", "energy", "inference", "accuracy", "benchmark", "neurons"],
    "T50": ["casimir", "force", "pressure", "residual", "separation", "uncertainty"],
    "T51": ["clock", "frequency", "drift", "fractional", "allan", "uncertainty"],
    "T52": ["interferometer", "noise", "sensitivity", "acceleration", "strain", "baseline"],
    "T54": ["coherence", "lifetime", "dephasing", "temperature", "complex", "sample"],
    "T57": ["energy", "cross", "section", "sigma", "cosmic", "uncertainty"],
    "T59": ["mass", "gev", "tev", "observed", "expected", "limit", "events", "cross"],
}


def is_repository_metadata_record_url_v12(url: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    u = str(url or "")
    if ACTUAL_FILE_URL_RE.search(u):
        return False
    if REPOSITORY_METADATA_URL_RE.search(u):
        return True
    return False


def artifact_role(url: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """v12 override: repository search/record APIs are metadata only; file endpoints are data."""
    u = str(url or "").lower()
    ctype = str((meta or {}).get("content_type") or "").lower()
    if u.startswith("figshare_search://"):
        return "metadata_record"
    if is_repository_metadata_record_url_v12(url, meta):
        return "metadata_record"
    if ACTUAL_FILE_URL_RE.search(url):
        # Content type/extension below decides exact parsing; it is at least not metadata.
        if "pdf" in ctype or u.endswith(".pdf"):
            return "pdf"
        if "zip" in ctype or u.endswith(".zip"):
            return "archive"
        if any(x in ctype for x in ["csv", "excel", "spreadsheet", "json", "text/plain"]) or re.search(r"\.(csv|tsv|txt|dat|xls|xlsx|json)(\?|$)", u):
            return "primary_table_file"
        return "primary_file_endpoint"
    if "arxiv.org/e-print" in u or u.endswith((".tar", ".tar.gz", ".tgz")):
        return "source_package"
    if u.endswith(".zip") or "zip" in ctype:
        return "archive"
    if u.endswith((".csv", ".tsv", ".dat", ".txt", ".xls", ".xlsx")) or any(x in ctype for x in ["csv", "excel", "spreadsheet"]):
        return "primary_table_file"
    if u.endswith(".json") or "application/json" in ctype:
        return "json_candidate"
    if u.endswith(".pdf") or "pdf" in ctype:
        return "pdf"
    if "html" in ctype or u.endswith((".html", ".htm")):
        return "html"
    return "unknown"


def is_metadata_record_url(url: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    """Compatibility override used by v11 scoring."""
    return is_repository_metadata_record_url_v12(url, meta)


def _figshare_post_search_bytes_v12(query: str, page_size: int = 25) -> Tuple[bytes, Dict[str, Any]]:
    endpoint = "https://api.figshare.com/v2/articles/search"
    payload = json.dumps({"search_for": query, "page_size": page_size, "order_direction": "desc"}).encode("utf-8")
    meta = {"url": endpoint, "method": "POST", "query": query, "ok": False, "content_type": "application/json"}
    try:
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json", "User-Agent": "ccdr-tierb-v12-autodiscovery"}, method="POST")
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
            meta.update({"ok": True, "status_code": getattr(resp, "status", None), "bytes": len(data), "final_url": endpoint})
            return data, meta
    except Exception as e:
        meta["error"] = f"figshare_post_failed: {type(e).__name__}: {e}"
        return b"", meta


def _download_for_scan_v12(src: Dict[str, Any], cache: Path, args: Any) -> Tuple[bytes, Dict[str, Any]]:
    url = src.get("url") or ""
    if str(url).startswith("figshare_search://"):
        q = urllib.parse.unquote(str(url).split("://", 1)[1])
        return _figshare_post_search_bytes_v12(q)
    return guarded_download_bytes(str(url), cache / "files", timeout=getattr(args, "timeout", 45), force=getattr(args, "force", False), max_bytes=getattr(args, "max_bytes", 50_000_000), manifest_approved=True)


def _zenodo_file_links_v12(obj: Any, url: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    def add_file(f: Dict[str, Any], label_prefix: str = "zenodo file"):
        lks = f.get("links") or {}
        # In Zenodo legacy API, links.self is often the API file download URL.
        for key in ["download", "self", "content", "archive"]:
            dl = lks.get(key) or f.get(key)
            if isinstance(dl, str) and dl.startswith("http"):
                links.append({"url": dl, "label": f.get("key") or f.get("filename") or f.get("name") or label_prefix, "reason": "zenodo_file_download"})
    if isinstance(obj, dict):
        # Search endpoint: hits.hits contains records.
        hits = (((obj.get("hits") or {}).get("hits")) or [])
        if isinstance(hits, list):
            for rec in hits[:30]:
                if not isinstance(rec, dict):
                    continue
                rec_id = rec.get("id") or rec.get("recid")
                if rec_id:
                    links.append({"url": f"https://zenodo.org/api/records/{rec_id}", "label": f"Zenodo record {rec_id}", "reason": "zenodo_record_metadata"})
                    links.append({"url": f"https://zenodo.org/api/records/{rec_id}/files", "label": f"Zenodo record {rec_id} files", "reason": "zenodo_files_endpoint"})
                for f in rec.get("files") or []:
                    if isinstance(f, dict):
                        add_file(f)
        # Record endpoint.
        for f in obj.get("files") or []:
            if isinstance(f, dict):
                add_file(f)
        # Invenio RDM files endpoint variants.
        entries = obj.get("entries") or obj.get("files") or []
        if isinstance(entries, list):
            for f in entries[:100]:
                if isinstance(f, dict):
                    add_file(f)
        elif isinstance(entries, dict):
            for name, f in list(entries.items())[:100]:
                if isinstance(f, dict):
                    f = dict(f); f.setdefault("key", name); add_file(f)
        # Follow files/media/archive links, but as metadata until actual files are reached.
        lks = obj.get("links") or {}
        for key in ["files", "media_files", "archive"]:
            u = lks.get(key)
            if isinstance(u, str) and u.startswith("http"):
                links.append({"url": u, "label": f"Zenodo {key}", "reason": "zenodo_files_or_archive_endpoint"})
    return _dedupe_link_dicts(links)


def _figshare_file_links_v12(obj: Any, url: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        for art in obj[:30]:
            if isinstance(art, dict) and art.get("id"):
                aid = art["id"]
                links.append({"url": f"https://api.figshare.com/v2/articles/{aid}", "label": art.get("title") or f"Figshare article {aid}", "reason": "figshare_article_metadata"})
    elif isinstance(obj, dict):
        if isinstance(obj.get("files"), list):
            for f in obj["files"]:
                if isinstance(f, dict):
                    dl = f.get("download_url") or ((f.get("links") or {}).get("download"))
                    if dl:
                        links.append({"url": dl, "label": f.get("name") or "figshare file", "reason": "figshare_file_download"})
    return _dedupe_link_dicts(links)


def _hepdata_links_v12(obj: Any, url: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    text = ""
    try:
        text = json.dumps(obj)
    except Exception:
        text = str(obj)[:500000]
    # Direct download/table links in JSON.
    for m in re.finditer(r"https?://(?:www\.)?hepdata\.net/download/table/[^\"'\s<>]+", text):
        links.append({"url": m.group(0).rstrip('.,}\"'), "label": "HEPData table download", "reason": "hepdata_table_download"})
    for m in re.finditer(r"/(?:download/table|record)/[^\"'\s<>]+", text):
        u = urljoin("https://www.hepdata.net", m.group(0).rstrip('.,}\"'))
        links.append({"url": u, "label": "HEPData relative link", "reason": "hepdata_relative_link"})
    # If we have record pages but no tables, try common export URLs could be discovered from HTML later.
    for m in re.finditer(r"https?://(?:www\.)?hepdata\.net/record/[^\"'\s<>]+", text):
        links.append({"url": m.group(0).rstrip('.,}\"'), "label": "HEPData record", "reason": "hepdata_record_link"})
    return _dedupe_link_dicts(links)


def links_from_connector_json(obj: Any, url: str) -> List[Dict[str, Any]]:
    """v12 override: metadata JSON only generates links to actual downloadable files/tables."""
    links: List[Dict[str, Any]] = []
    # Generic metadata URLs are useful as link sources; use a focused extractor to avoid Wikidata/API noise.
    if "zenodo.org/api/records" in url:
        links.extend(_zenodo_file_links_v12(obj, url))
    elif "api.figshare.com/v2/articles" in url or str(url).startswith("figshare_search://"):
        links.extend(_figshare_file_links_v12(obj, url))
    elif "hepdata.net" in url:
        links.extend(_hepdata_links_v12(obj, url))
    elif "api.osf.io" in url:
        # Keep the v11 OSF logic, since it explicitly discovers download links.
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            for it in obj.get("data") or []:
                if not isinstance(it, dict):
                    continue
                attrs = it.get("attributes") or {}
                lks = it.get("links") or {}
                rel = it.get("relationships") or {}
                name = attrs.get("name") or attrs.get("materialized_path") or "osf file"
                if lks.get("download"):
                    links.append({"url": lks["download"], "label": name, "reason": "osf_download_link"})
                if lks.get("html"):
                    links.append({"url": lks["html"], "label": name, "reason": "osf_html_link"})
                for key in ["files", "parent_folder", "node"]:
                    try:
                        u = (((rel.get(key) or {}).get("links") or {}).get("related") or {}).get("href")
                        if u:
                            links.append({"url": u, "label": f"osf_related_{key}", "reason": "osf_related"})
                    except Exception:
                        pass
            nxt = (obj.get("links") or {}).get("next")
            if nxt:
                links.append({"url": nxt, "label": "OSF next page", "reason": "pagination"})
    else:
        # Crossref/OpenAlex/DataCite: only follow explicit data/supplement links, not every Wikidata/reference URL.
        links.extend([l for l in links_from_metadata_json(obj, url) if DATA_LINK_RE.search(l.get("url", "")) or STRUCTURED_EXT_RE.search(l.get("url", "")) or PDF_RE.search(l.get("url", ""))])
    return _dedupe_link_dicts(links)[:160]


def connector_candidate_urls(source_url: str, data: Optional[bytes] = None, meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """v12 override: conservative link discovery; repository metadata first, physical files later."""
    meta = meta or {}
    links: List[Dict[str, Any]] = []
    text = ""
    if data:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    role = artifact_role(source_url, meta)
    if role == "html" and text:
        links.extend(html_links(text, source_url))
    # DOI/arXiv expansion is useful for PDFs/HTML/source pages, but not for already-scanned metadata APIs.
    if role not in {"metadata_record"}:
        links.extend(metadata_expansion_links(source_url, text[:300000] if text else ""))
    # arXiv source package from arXiv HTML/PDF/abs URLs.
    aid = arxiv_id_from_url_or_text(source_url, text[:20000] if text else "")
    if aid:
        links.append({"url": f"https://arxiv.org/e-print/{aid}", "label": f"arXiv source package {aid}", "reason": "arxiv_source_package"})
    # HEPData search by DOI/arXiv only for HEP-like tests; harmless as metadata.
    for doi in doi_candidates(source_url + "\n" + (text[:50000] if text else ""))[:5]:
        if re.search(r"atlas|cms|lhc|drell|higgs|met|cross.section|cosmic|hep", source_url + " " + text[:10000], re.I):
            links.append({"url": f"https://www.hepdata.net/search/?q={quote(doi)}&format=json", "label": "HEPData search by DOI", "reason": "hepdata_search"})
    return _dedupe_link_dicts(links)[:120]


def physical_hint_score_v12(test_id: str, df: pd.DataFrame) -> int:
    cols = " ".join(str(c).lower() for c in list(df.columns)[:200])
    hints = PHYSICAL_MIN_COLS_BY_TEST.get(test_id, [])
    return sum(1 for h in hints if h.lower() in cols)


def table_relevance_score_v12(sc: Dict[str, Any]) -> float:
    match = sc.get("physical_column_match") or {}
    matched = len(match.get("matched_groups") or [])
    required = len((match.get("matched_groups") or [])) + len((match.get("missing_groups") or []))
    numeric = len(sc.get("numeric_columns") or [])
    rows = (sc.get("shape") or [0])[0] or 0
    tier = sc.get("evidence_tier") or ""
    tier_bonus = 2.0 if tier in {"primary_structured_public_table", "primary_structured_zip_member", "hepdata_table"} else 0.5
    return float((matched / max(required, 1)) * 100 + min(numeric, 10) * 3 + min(rows, 50) * 0.2 + tier_bonus)


def score_frame(test_id: str, df: pd.DataFrame, source_url: str, tier: str) -> Dict[str, Any]:
    """v12 override: stricter physical-candidate scoring and metadata exclusion."""
    required = _regex_groups_from_contract(test_id)
    report = column_match_report(df, required)
    nums = numeric_columns(df)
    n_rows = int(df.shape[0])
    x_range = None
    for c in nums[:10]:
        vals = clean_numeric_series(df[c]).dropna().astype(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) >= 3:
            try:
                x_range = float(np.log10(vals.max()) - np.log10(vals.min()))
                break
            except Exception:
                pass
    unit_hints = {str(c): normalize_unit_from_text(str(c)) for c in list(df.columns)[:80]}
    min_rows = _min_rows_for(test_id, tier)
    role = artifact_role(source_url, {"content_type": ""})
    metadata_only = is_repository_metadata_record_url_v12(source_url)
    matched_groups = len(report.get("matched_groups") or [])
    physical_hint_score = physical_hint_score_v12(test_id, df)
    has_some_physical = matched_groups > 0 or physical_hint_score > 0
    has_all_physical = bool(report.get("ok"))
    enough_numeric = len(nums) >= 2
    enough_rows = n_rows >= min_rows
    primary_tiers = {"primary_structured_public_table", "primary_structured_zip_member", "hepdata_table"}
    ok = bool((not metadata_only) and has_all_physical and enough_numeric and enough_rows)
    primary_allowed = bool(ok and tier in primary_tiers)
    rejection_reasons = []
    if metadata_only:
        rejection_reasons.append("metadata_record_not_physical_table")
    if not has_some_physical:
        rejection_reasons.append("no_physical_observable_hint")
    if not has_all_physical:
        rejection_reasons.append("missing_required_physical_column_groups")
    if not enough_numeric:
        rejection_reasons.append("too_few_numeric_columns")
    if not enough_rows:
        rejection_reasons.append("below_min_rows_for_contract")
    if tier not in primary_tiers:
        rejection_reasons.append("secondary_or_nonprimary_evidence_tier")
    out = {
        "source_url": source_url,
        "artifact_role": "metadata_record" if metadata_only else role,
        "evidence_tier": "metadata_record" if metadata_only else tier,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in list(df.columns)[:100]],
        "numeric_columns": [str(c) for c in nums[:60]],
        "unit_normalization_hints": unit_hints,
        "physical_column_match": report,
        "physical_hint_score": physical_hint_score,
        "has_some_physical_evidence": bool(has_some_physical),
        "sensitivity": sensitivity_classification(n_rows, x_range, min_rows=min_rows),
        "qualifies_for_model": ok,
        "confirmation_allowed": primary_allowed,
        "falsification_allowed": primary_allowed,
        "rejection_reasons": rejection_reasons,
        "attrs": {k: v for k, v in getattr(df, "attrs", {}).items() if isinstance(v, (str, int, float, bool))},
    }
    out["table_relevance_score"] = table_relevance_score_v12(out)
    return out


def nearest_miss(test_id: str, candidates: Sequence[Dict[str, Any]], sources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    physical = [c for c in candidates if c.get("has_some_physical_evidence")]
    pool = physical if physical else list(candidates)
    if pool:
        best = sorted(pool, key=lambda c: c.get("table_relevance_score", 0), reverse=True)[0]
        miss = best.get("physical_column_match") or {}
        return {
            "nearest_candidate_source": best.get("source_url"),
            "nearest_candidate_tier": best.get("evidence_tier"),
            "nearest_candidate_shape": best.get("shape"),
            "nearest_candidate_relevance_score": best.get("table_relevance_score"),
            "matched_groups": miss.get("matched_groups"),
            "missing_required_groups": miss.get("missing_groups"),
            "numeric_columns": best.get("numeric_columns"),
            "rejection_reasons": best.get("rejection_reasons"),
            "suggested_next_auto_strategy": "download_files_listed_by_repository_metadata" if best.get("artifact_role") == "metadata_record" else "domain_specific_parser_or_source_data_search",
        }
    return {
        "nearest_source": sources[0].get("url") if sources else None,
        "missing_required_groups": CONTRACTS.get(test_id, {}).get("required_column_groups"),
        "suggested_next_auto_strategy": "supplement_crawler_arxiv_source_pdf_table_extraction",
    }


def _read_csv_manifest_rows_v12(path: Path, test_id: Optional[str] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if test_id is None or str(r.get("test_id", "")).strip() == test_id:
                    rows.append(r)
    except Exception:
        return []
    return rows


def additional_seed_sources_v11(test_id: str) -> List[Dict[str, Any]]:
    """v12 override: more automated seeds, including exact manifest file URLs where available."""
    seeds: List[Dict[str, Any]] = []
    # First include exact HEPData table URLs from the bundled manifest.
    if test_id in {"T57", "T59"}:
        for r in _read_csv_manifest_rows_v12(DATA_ROOT_V12 / "hep_manifest.csv", test_id):
            u = r.get("url")
            if u:
                seeds.append({"url": u, "label": r.get("label") or "HEPData manifest table", "reason": "v12_exact_hepdata_manifest"})
    # Existing domain seed logic, but use typed repository search and Figshare POST custom scheme.
    if test_id in {"T26", "T27", "T29"}:
        queries = {
            "T26": [
                "ELM pedestal energy loss supplementary data",
                "type I ELM energy pedestal pressure table",
                "JET ELM energy loss pedestal pressure dW ELM data",
                "DIII-D ELM energy pedestal database supplementary",
            ],
            "T27": [
                "RMP ELM frequency coil current supplementary data",
                "ELM suppression phasing n=2 n=3 data",
                "DIII-D RMP coil current ELM frequency table",
                "KSTAR ELM suppression RMP current frequency data",
            ],
            "T29": [
                "W7-X profile transport data Te ne heat flux",
                "stellarator tokamak confinement profile transport data",
                "W7-X heat flux diffusivity ne Te profile supplementary data",
                "LHD stellarator edge transport profile data",
            ],
        }[test_id]
        for q in queries:
            seeds.append({"url": f"https://zenodo.org/api/records/?q={quote(q)}&size=25", "label": f"Zenodo API search: {q}", "reason": "v12_zenodo_query_seed"})
            seeds.append({"url": "figshare_search://" + quote(q), "label": f"Figshare POST search: {q}", "reason": "v12_figshare_post_seed"})
    if test_id in {"T44", "T45", "T47"}:
        for r in _read_csv_manifest_rows_v12(DATA_ROOT_V12 / "electronics_source_manifest.csv", test_id):
            u = r.get("url")
            if u:
                seeds.append({"url": u, "label": r.get("label") or "electronics manifest source", "reason": "v12_electronics_manifest"})
        if test_id == "T44":
            seeds += [
                {"url": "https://en.wikichip.org/wiki/3d_nand", "label": "WikiChip 3D NAND", "reason": "v12_electronics_seed"},
                {"url": "https://en.wikipedia.org/wiki/Flash_memory", "label": "Wikipedia Flash memory", "reason": "v12_electronics_seed"},
                {"url": "https://en.wikipedia.org/wiki/Multi-level_cell", "label": "Wikipedia multi-level cell", "reason": "v12_electronics_seed"},
            ]
        elif test_id == "T45":
            seeds += [
                {"url": "https://irds.ieee.org/editions", "label": "IRDS editions", "reason": "v12_electronics_seed"},
                {"url": "https://www.opencompute.org/wiki/Networking/Optics", "label": "OCP optics", "reason": "v12_electronics_seed"},
            ]
        elif test_id == "T47":
            seeds += [
                {"url": "https://en.wikichip.org/wiki/intel/loihi", "label": "WikiChip Loihi", "reason": "v12_electronics_seed"},
                {"url": "https://en.wikichip.org/wiki/ibm/truenorth", "label": "WikiChip TrueNorth", "reason": "v12_electronics_seed"},
                {"url": "https://open-neuromorphic.org/neuromorphic-computing/hardware/", "label": "Open Neuromorphic hardware", "reason": "v12_electronics_seed"},
            ]
    if test_id in {"T50", "T51", "T52"}:
        queries = {
            "T50": ["Casimir force residual uncertainty table data", "Casimir pressure residual separation uncertainty data"],
            "T51": ["optical clock fractional frequency drift uncertainty budget table", "Allan deviation optical clock data table"],
            "T52": ["atom interferometer sensitivity noise floor table", "atom interferometer acceleration noise spectral density data"],
        }[test_id]
        for q in queries:
            seeds.append({"url": f"https://zenodo.org/api/records/?q={quote(q)}&size=25", "label": f"Zenodo metrology search: {q}", "reason": "v12_metrology_seed"})
            seeds.append({"url": "figshare_search://" + quote(q), "label": f"Figshare POST metrology search: {q}", "reason": "v12_figshare_post_seed"})
    if test_id == "T54":
        for q in ["photosynthetic coherence lifetime supplementary data", "2D electronic spectroscopy coherence lifetime data", "FMO coherence lifetime data table"]:
            seeds.append({"url": f"https://zenodo.org/api/records/?q={quote(q)}&size=25", "label": f"Zenodo coherence search: {q}", "reason": "v12_bio_seed"})
            seeds.append({"url": "figshare_search://" + quote(q), "label": f"Figshare POST coherence search: {q}", "reason": "v12_figshare_post_seed"})
    if test_id in {"T57", "T59"}:
        for q in ["ATLAS MET observed expected", "CMS Drell-Yan 1 TeV", "ATLAS di-Higgs observed expected", "cosmic ray cross section"]:
            seeds.append({"url": f"https://www.hepdata.net/search/?q={quote(q)}&format=json", "label": f"HEPData search: {q}", "reason": "v12_hepdata_query_seed"})
    return _dedupe_link_dicts(seeds)


def rejection_summary(candidates: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in candidates:
        for r in c.get("rejection_reasons") or []:
            out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def automated_source_scan(test_id: str, args: Any, seed_sources: Sequence[Dict[str, Any]], max_sources: int = 25, max_depth: int = 2) -> Dict[str, Any]:
    """v12 override: count only physical table candidates; metadata only discovers files."""
    cache = cache_level(args.cache, f"v12_auto_discovery_{test_id}")
    queue = _dedupe_link_dicts(list(seed_sources) + additional_seed_sources_v11(test_id))
    seen = set()
    source_records: List[Dict[str, Any]] = []
    physical_candidates: List[Dict[str, Any]] = []
    parsed_nonphysical: List[Dict[str, Any]] = []
    metadata_records: List[Dict[str, Any]] = []
    schema_artifacts: List[Dict[str, Any]] = []
    depth_by_url = {s.get("url"): 0 for s in queue}
    max_sources_eff = max(max_sources, 36 if test_id in {"T26", "T27", "T29", "T44", "T45", "T47", "T57", "T59"} else max_sources)
    # Keep discovery bounded. --timeout is per-request, so add a process-level autodiscovery budget.
    try:
        per_req_timeout = int(getattr(args, "timeout", 45) or 45)
    except Exception:
        per_req_timeout = 45
    deadline = time.monotonic() + max(90, min(420, per_req_timeout * 8))
    timeout_stop_reason = None
    while queue and len(source_records) < max_sources_eff and time.monotonic() < deadline:
        src = queue.pop(0)
        url = src.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        depth = depth_by_url.get(url, 0)
        data, meta = _download_for_scan_v12(src, cache, args)
        rec: Dict[str, Any] = {"url": url, "label": src.get("label"), "seed_reason": src.get("reason"), "depth": depth, "meta": meta, "extracted_links": [], "artifact_diag": {}, "candidate_tables": [], "nonphysical_tables_sample": []}
        source_records.append(rec)
        if not data:
            continue
        frames, diag, text_sample = frames_from_artifact(test_id, data, url, meta)
        rec["artifact_diag"] = diag
        if diag.get("artifact_role") == "metadata_record":
            metadata_records.append({"url": url, "reason": src.get("reason"), "content_type": meta.get("content_type"), "bytes": meta.get("bytes")})
        if re.search(r"variables|schema|dictionary", str(url) + " " + str(src.get("label")), re.I) and ("pdf" in str(meta.get("content_type", "")).lower() or str(url).lower().endswith(".pdf")):
            schema = extract_schema_from_text(text_sample, test_id)
            if schema.get("variables_count"):
                schema_artifacts.append(schema); rec["schema_extracted"] = schema
        for df in frames[:120]:
            tier = str(df.attrs.get("evidence_tier") or ("primary_structured_public_table" if STRUCTURED_EXT_RE.search(str(url)) else "html_table"))
            sc = score_frame(test_id, df, str(df.attrs.get("source_url") or url), tier)
            if sc.get("artifact_role") == "metadata_record" or "metadata_record_not_physical_table" in sc.get("rejection_reasons", []):
                metadata_records.append({"url": sc.get("source_url"), "columns_sample": sc.get("columns", [])[:15], "reason": "parsed_metadata_frame_rejected"})
                continue
            # Only physical-ish tables go into candidate_table_count.
            if sc.get("has_some_physical_evidence"):
                physical_candidates.append(sc); rec["candidate_tables"].append(sc)
            else:
                parsed_nonphysical.append(sc)
                if len(rec["nonphysical_tables_sample"]) < 3:
                    rec["nonphysical_tables_sample"].append({"source_url": sc.get("source_url"), "shape": sc.get("shape"), "columns": sc.get("columns", [])[:12], "rejection_reasons": sc.get("rejection_reasons")})
        if depth < max_depth:
            links = connector_candidate_urls(str(url), data, meta)
            try:
                stripped = data.lstrip()[:1]
                if stripped in [b"{", b"["]:
                    obj = json.loads(data.decode("utf-8", errors="replace"))
                    links.extend(links_from_connector_json(obj, str(url)))
            except Exception:
                pass
            links = _dedupe_link_dicts(links)
            rec["extracted_links"] = links[:120]
            for l in links:
                u = l.get("url")
                if u and u not in seen and len(queue) < max_sources_eff * 5:
                    depth_by_url[u] = depth + 1
                    queue.append(l)
    if queue and time.monotonic() >= deadline:
        timeout_stop_reason = "autodiscovery_wall_clock_budget_reached"
    elif len(source_records) >= max_sources_eff:
        timeout_stop_reason = "autodiscovery_max_sources_reached"
    qualifying = [c for c in physical_candidates if c.get("qualifies_for_model")]
    primary = [c for c in qualifying if c.get("confirmation_allowed")]
    secondary = [c for c in qualifying if not c.get("confirmation_allowed")]
    if primary:
        status = "primary_table_model_possible"
    elif secondary:
        status = "secondary_model_possible_nonprimary"
    elif physical_candidates:
        status = "physical_candidate_tables_found_but_missing_columns_or_power"
    elif source_records:
        status = "sources_scanned_no_physical_candidate_tables"
    else:
        status = "no_sources_scanned"
    # Sort candidates by relevance.
    physical_candidates_sorted = sorted(physical_candidates, key=lambda c: c.get("table_relevance_score", 0), reverse=True)
    return {
        "version": "v12_strict_physical_artifact_discovery_no_manual_steps",
        "generated_utc": utc_now(),
        "data_contract": CONTRACTS.get(test_id, {}),
        "seed_sources_count": len(seed_sources),
        "additional_auto_seed_sources_count": len(additional_seed_sources_v11(test_id)),
        "sources_scanned_count": len(source_records),
        "autodiscovery_stop_reason": timeout_stop_reason,
        "autodiscovery_queue_remaining": len(queue),
        "source_records_sample": source_records[:50],
        "metadata_records_seen_count": len(metadata_records),
        "metadata_records_sample": metadata_records[:25],
        "nonphysical_tables_parsed_count": len(parsed_nonphysical),
        "nonphysical_tables_sample": parsed_nonphysical[:20],
        "candidate_table_count": len(physical_candidates_sorted),
        "physical_candidate_table_count": len(physical_candidates_sorted),
        "qualifying_table_count": len(qualifying),
        "primary_qualifying_table_count": len(primary),
        "secondary_qualifying_table_count": len(secondary),
        "candidate_rejection_summary": rejection_summary(physical_candidates_sorted),
        "candidate_tables_sample": physical_candidates_sorted[:30],
        "qualifying_tables_sample": qualifying[:30],
        "schema_artifacts": schema_artifacts[:10],
        "nearest_miss": nearest_miss(test_id, physical_candidates_sorted, source_records),
        "automated_readiness_status": status,
        "strict_verdict_rule_v12": "Only actual physical table files or table-like source-data with required physical columns count as candidates. Repository/API/search metadata only discovers downloadable files and never increments candidate_table_count.",
        "evidence_ladder": {
            "E0": "no source found",
            "E1": "source found but no usable physical table",
            "E2": "secondary auto-extracted physical diagnostic table; not decisive",
            "E3": "primary machine-readable public physical table/model possible",
            "E4": "primary table with uncertainties/controls and adequate sensitivity",
        },
    }


def simple_model_diagnostics(test_id: str, candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [c for c in candidates if c.get("qualifies_for_model")]
    if not usable:
        return {"status": "no_model_candidate", "reason": "no physical table passed all required groups, numeric, and row-count gates"}
    primary = [c for c in usable if c.get("confirmation_allowed")]
    return {
        "status": "primary_model_candidate_available" if primary else "secondary_model_candidate_available_nondecisive",
        "usable_table_count": len(usable),
        "primary_table_count": len(primary),
        "best_relevance_score": max(float(c.get("table_relevance_score", 0)) for c in usable),
        "decisive_allowed": bool(primary),
        "interpretation": "This is a readiness signal. The domain test runner must still run the scientific model before support/null claims.",
    }


def augment_result_with_autodiscovery(test_id: str, result: Dict[str, Any], args: Any, extra_sources: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    seeds = source_urls_from_result(test_id, result)
    if extra_sources:
        seeds.extend(extra_sources)
    if not seeds and test_id in {"T57", "T59"}:
        seeds.append({"url": "https://www.hepdata.net/search/?q=ATLAS%20CMS%20Drell-Yan%20Higgs%20MET&format=json", "label": "HEPData broad exact-table search", "reason": "hepdata_connector"})
    scan = automated_source_scan(test_id, args, seeds, max_sources=60 if test_id in {"T26", "T27", "T29", "T44", "T45", "T47", "T57", "T59"} else 35, max_depth=2)
    scan["model_diagnostics"] = simple_model_diagnostics(test_id, scan.get("qualifying_tables_sample") or [])
    result["automated_discovery_v12"] = scan
    result["automated_discovery_v11"] = scan
    result["automated_discovery_v10"] = scan
    result["quality_patch_version"] = str(result.get("quality_patch_version", "")) + "+v12_physical_artifact_typing_exact_file_connectors"
    if scan.get("primary_qualifying_table_count", 0):
        result["readiness_status"] = "primary_auto_discovered_physical_table_candidate"
        result["evidence_status"] = "analysis_ready_primary_auto_discovered"
    elif scan.get("secondary_qualifying_table_count", 0):
        result["readiness_status"] = "secondary_auto_extracted_physical_table_candidate_nonprimary"
        result["evidence_status"] = "data_limited_secondary_diagnostic_available"
    elif result.get("evidence_status") == "data_limited":
        result["readiness_status"] = result.get("readiness_status") or scan.get("automated_readiness_status")
    result["automated_no_manual_steps_policy"] = "All v12 discovery artifacts are downloaded or extracted automatically from public URLs. Repository/search metadata only discovers file links; candidate_table_count counts physical-table candidates only. Secondary auto-extracted PDF/figure/arXiv tables cannot confirm/falsify."
    return result


def _download_text_v12(url: str, cache: Path, args: Any) -> str:
    data, meta = guarded_download_bytes(url, cache / "micro_refs", timeout=getattr(args, "timeout", 30), force=False, max_bytes=2_000_000, manifest_approved=True)
    if not data:
        return ""
    try:
        return data.decode("utf-8", errors="replace")[:300000]
    except Exception:
        return ""


def _cmbs4_reference_urls_for_path_v12(path: str) -> List[str]:
    # CMB-S4 raw paths look like thermal_conductivity/lib/Material/RAW/file.csv.
    m = re.search(r"thermal_conductivity/lib/([^/]+)/", path)
    if not m:
        return []
    mat = m.group(1)
    base = f"https://raw.githubusercontent.com/CMB-S4/Cryogenic_Material_Properties/main/thermal_conductivity/lib/{mat}/"
    return [base + "references.txt", base + "README.md", base + f"{mat}_fits.csv"]


def auto_microstructure_from_result_v12(result: Dict[str, Any], args: Any) -> Dict[str, Any]:
    cache = cache_level(args.cache, "v12_auto_microstructure_refs")
    rows = []
    ref_text_by_material: Dict[str, str] = {}
    for rec in result.get("downloaded_sources") or []:
        path = str(rec.get("path") or rec.get("url") or "")
        label = path.lower()
        ref_urls = _cmbs4_reference_urls_for_path_v12(path)
        ref_text = ""
        for u in ref_urls:
            if u not in ref_text_by_material:
                ref_text_by_material[u] = _download_text_v12(u, cache, args)
            ref_text += "\n" + ref_text_by_material.get(u, "")
        combined = (label + "\n" + ref_text.lower())[:400000]
        matches = []
        # Highest-confidence measured rows require explicit terms in references or filenames, not generic material names.
        if re.search(r"nano[-_ ]?crystal|nanograin|nanocrystalline|grain\s*size|crystallite\s*size|\b\d+(?:\.\d+)?\s*(?:nm|um|µm)\b.{0,40}(?:grain|crystallite|particle)", combined, re.I):
            matches.append({"class": "measured_or_explicit_nanocrystalline", "confidence": 0.93, "pattern": "reference_or_filename_measured_microstructure"})
        if re.search(r"sinter|powder|porous|polycrystal|polycrystalline|anneal|cold[- ]?worked", combined, re.I):
            matches.append({"class": "grain_boundary_candidate_nondecisive", "confidence": 0.65, "pattern": "processing_microstructure_terms"})
        for cls, rx, conf in MICROSTRUCTURE_PATTERNS:
            if rx.search(combined):
                matches.append({"class": cls, "confidence": conf, "pattern": rx.pattern})
        if matches:
            best = sorted(matches, key=lambda x: x["confidence"], reverse=True)[0]
            rows.append({
                "source_path": path,
                "auto_microstructure_class": best["class"],
                "confidence": best["confidence"],
                "all_matches": matches,
                "reference_urls_checked": ref_urls,
                "reference_text_found": bool(ref_text.strip()),
            })
    decisive = [r for r in rows if r["auto_microstructure_class"] == "measured_or_explicit_nanocrystalline" and r["confidence"] >= 0.90]
    class_names = sorted({r["auto_microstructure_class"] for r in rows} | {c for c, _, _ in MICROSTRUCTURE_PATTERNS} | {"grain_boundary_candidate_nondecisive"})
    return {
        "version": "v12_auto_microstructure_reference_harvest_no_manual_steps",
        "rows_generated": len(rows),
        "decisive_candidate_rows": len(decisive),
        "class_counts": {c: sum(1 for r in rows if r["auto_microstructure_class"] == c) for c in class_names},
        "rows_sample": rows[:120],
        "references_checked_count": len(ref_text_by_material),
        "interpretation": "Reference-harvested microstructure is automatic triage only. Decisive MAT1/MAT3 language still requires enough high-confidence measured/explicit nanocrystalline rows matched to kappa tables.",
    }


def augment_material_result_v10(test_id: str, result: Dict[str, Any], args: Any) -> Dict[str, Any]:
    auto = auto_microstructure_from_result_v12(result, args)
    result["auto_generated_microstructure_manifest_v12"] = auto
    result["auto_generated_microstructure_manifest_v10"] = auto
    q = result.get("decisive_quality_gate") or {}
    if auto.get("decisive_candidate_rows", 0) >= 10 and q.get("grain_or_nano_known_usable", 0) >= 10:
        q["auto_decisive_ready_suggestion"] = True
    else:
        q["auto_decisive_ready_suggestion"] = False
    q["auto_microstructure_decisive_candidate_rows"] = auto.get("decisive_candidate_rows", 0)
    result["decisive_quality_gate"] = q
    result["quality_patch_version"] = str(result.get("quality_patch_version", "")) + "+v12_auto_microstructure_reference_harvest"
    return result

# ---------------------------------------------------------------------------
# v13 data-limited quality layer: strict metadata typing, HTML-noise filter,
# repository file-first connectors, recursive archives, and domain gates.
# ---------------------------------------------------------------------------

METADATA_SEARCH_JSON_COLUMNS_V13 = {
    "hits.hits", "hits.total", "aggregations.publication_date.buckets",
    "aggregations.resource_type.buckets", "aggregations.file_type.buckets",
    "links.self", "links.next", "metadata.title", "metadata.creators",
}

HTML_NOISE_RE_V13 = re.compile(
    r"(<svg|</svg|<path|</path|<script|</script|<style|</style|gtag\(|googletag|"
    r"stylesheet|favicon|apple-touch-icon|material-symbols|fontawesome|bootstrap|jquery|polyfill|"
    r"cookie|analytics|csrf|navbar|breadcrumb|aria-|xmlns=|viewbox=)",
    re.I,
)

DOMAIN_PHYSICAL_RE_V13 = re.compile(
    r"(ELM|pedestal|W[_ -]?ELM|E[_ -]?ELM|RMP|I[-_ ]?coil|TAUE|TAUTH|H98|q95|"
    r"elongation|triangularity|diffusiv|heat\s*flux|W7[- ]?X|tokamak|stellarator|"
    r"die\s*area|3D\s*NAND|layers|energy\s*/?\s*bit|pJ/bit|fJ/bit|Loihi|TrueNorth|"
    r"Casimir|Allan|clock\s*drift|interferometer|coherence|lifetime|cross\s*section|"
    r"observed|expected|Drell|Higgs|MET|GeV|TeV|bandpower|BB)",
    re.I,
)

FALSE_POSITIVE_SOURCE_RE_V13 = re.compile(
    r"(earth\s+land\s+model|\bDELM\b|\bNoDELM\b|SNOTEL|squirrel|white\s+dwarf|"
    r"favicon|apple-touch-icon|css_|\.css(\?|$)|\.js(\?|$)|\.svg(\?|$)|\.png(\?|$)|\.jpg(\?|$)|\.ico(\?|$))",
    re.I,
)

ARCHIVE_EXT_RE_V13 = re.compile(r"\.(zip|tar|tar\.gz|tgz|gz)(\?|$)", re.I)
TABLE_EXT_RE_V13 = re.compile(r"\.(csv|tsv|txt|dat|json|xls|xlsx)(\?|$)", re.I)
SOURCE_EXT_RE_V13 = re.compile(r"\.(tex|latex)(\?|$)", re.I)


def is_repository_metadata_record_url_v12(url: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    """v13 override: repository search/record/API JSON is metadata unless it is an actual file-content endpoint."""
    u = str(url or "")
    lu = u.lower()
    ctype = str((meta or {}).get("content_type") or "").lower()
    if u.startswith("figshare_search://"):
        return True
    # Explicit file-content endpoints are not metadata even under repository API hosts.
    if re.search(r"/files/[^/?#]+/(?:content|download)(?:\?|$)|/files-archive(?:\?|$)|/media-files-archive(?:\?|$)", lu):
        return False
    if re.search(r"zenodo\.org/api/records(?:/\d+)?(?:/files)?(?:\?|$)", lu):
        return True
    if re.search(r"zenodo\.org/api/records/\d+$", lu):
        return True
    if re.search(r"zenodo\.org/api/records/\d+/(?:versions|access|communities|draft|pids|quota|request|media-files)(?:/|\?|$)", lu):
        return True
    if re.search(r"api\.figshare\.com/v2/articles(?:/search|/\d+)?(?:\?|$)", lu):
        return True
    if re.search(r"api\.osf\.io/v2/", lu):
        return True
    if re.search(r"hepdata\.net/search/", lu):
        return True
    if re.search(r"api\.(?:crossref|openalex|datacite|semanticscholar)\.org/", lu):
        return True
    # JSON content from known search/metadata sources is metadata unless it is a file endpoint.
    if "application/json" in ctype and re.search(r"(api/records\?|/search\?|/works/|/dois/)", lu):
        return True
    return False


def artifact_role(url: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """v13 override: type artifacts before any table parsing."""
    u = str(url or "")
    lu = u.lower()
    ctype = str((meta or {}).get("content_type") or "").lower()
    if FALSE_POSITIVE_SOURCE_RE_V13.search(u):
        return "nondata_asset_or_false_positive"
    if is_repository_metadata_record_url_v12(url, meta):
        return "metadata_record"
    if "arxiv.org/e-print" in lu or lu.endswith((".tar", ".tar.gz", ".tgz")) or "x-eprint" in ctype:
        return "source_package"
    if ARCHIVE_EXT_RE_V13.search(lu) or any(x in ctype for x in ["zip", "gzip", "tar"]):
        return "archive"
    if TABLE_EXT_RE_V13.search(lu) or any(x in ctype for x in ["csv", "excel", "spreadsheet", "text/plain"]):
        return "primary_table_file"
    if lu.endswith(".json") or "application/json" in ctype:
        return "json_candidate"
    if lu.endswith(".pdf") or "pdf" in ctype:
        return "pdf"
    if "html" in ctype or lu.endswith((".html", ".htm")):
        return "html"
    return "unknown"


def _frame_text_v13(df: pd.DataFrame, max_cells: int = 500) -> str:
    try:
        cols = [str(c) for c in list(df.columns)[:100]]
        vals = []
        if df is not None and not df.empty:
            sub = df.iloc[:20, : min(25, df.shape[1])]
            vals = [str(x) for x in sub.astype(str).values.ravel().tolist()[:max_cells]]
        return " ".join(cols + vals)
    except Exception:
        return " ".join(str(c) for c in list(getattr(df, "columns", []))[:100])


def is_metadata_like_frame_v13(df: pd.DataFrame, source_url: str = "") -> bool:
    cols = {str(c) for c in list(getattr(df, "columns", []))}
    if len(cols & METADATA_SEARCH_JSON_COLUMNS_V13) >= 2:
        return True
    joined = " ".join(cols).lower()
    if all(x in joined for x in ["hits", "aggregations", "links"]):
        return True
    if is_repository_metadata_record_url_v12(source_url):
        return True
    return False


def is_html_noise_frame_v13(df: pd.DataFrame) -> bool:
    txt = _frame_text_v13(df)
    if HTML_NOISE_RE_V13.search(txt):
        return True
    # Many read_html results from generic pages contain no physical terms and no numeric values.
    nums = []
    try:
        nums = numeric_columns(df)
    except Exception:
        nums = []
    if not DOMAIN_PHYSICAL_RE_V13.search(txt) and len(nums) == 0:
        return True
    return False


def has_domain_physical_hint_v13(test_id: str, df: pd.DataFrame, url: str = "") -> bool:
    txt = (str(url) + " " + _frame_text_v13(df)).lower()
    # Match contract group terms or the domain hint regex. At least one contract-group term is enough for nearest-miss.
    contract = CONTRACTS.get(test_id, {})
    for group in contract.get("required_column_groups", []) or []:
        for term in group:
            if str(term).lower().replace("_", " ") in txt or str(term).lower() in txt:
                return True
    if DOMAIN_PHYSICAL_RE_V13.search(txt):
        return True
    return False


def _read_table_member_v13(blob: bytes, name: str, source_url: str, tier: str) -> List[pd.DataFrame]:
    out: List[pd.DataFrame] = []
    try:
        for df in read_tabular_bytes(blob, name):
            df.attrs["source_url"] = source_url + "#" + name
            df.attrs["evidence_tier"] = tier
            out.append(df)
    except Exception:
        pass
    return out


def _frames_from_archive_v13(data: bytes, source_url: str, depth: int = 0, max_depth: int = 3, max_members: int = 400) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]], str]:
    """Recursive archive traversal for zip/tar/tar.gz/gz nested artifacts."""
    frames: List[pd.DataFrame] = []
    members: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    if depth > max_depth:
        return frames, members, ""

    def handle_member(name: str, blob: bytes):
        nonlocal frames, members, text_parts
        lname = name.lower()
        members.append({"name": name, "size": len(blob), "depth": depth})
        if FALSE_POSITIVE_SOURCE_RE_V13.search(name):
            return
        if TABLE_EXT_RE_V13.search(lname):
            frames.extend(_read_table_member_v13(blob, name, source_url, "primary_structured_zip_member"))
        elif SOURCE_EXT_RE_V13.search(lname):
            try:
                txt = blob.decode("utf-8", errors="replace")
                text_parts.append(txt[:50000])
                frames.extend(_frames_from_latex_tables(txt, source_url + "#" + name))
            except Exception:
                pass
        elif ARCHIVE_EXT_RE_V13.search(lname):
            subframes, submembers, subtext = _frames_from_archive_v13(blob, source_url + "#" + name, depth + 1, max_depth=max_depth, max_members=max_members)
            frames.extend(subframes); members.extend(submembers); text_parts.append(subtext)
        elif lname.endswith(".pdf"):
            # Do not parse every nested PDF in huge bundles unless it looks domain-relevant by filename.
            if DOMAIN_PHYSICAL_RE_V13.search(name):
                pframes = _frames_from_pdf_pdfplumber(blob, source_url + "#" + name, max_pages=_pdf_page_budget_v12())
                pframes.extend(_frames_from_pdf_text_lines(blob, source_url + "#" + name, max_pages=_pdf_page_budget_v12()))
                frames.extend(pframes)

    # ZIP
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for i, zi in enumerate(z.infolist()):
                if i >= max_members or zi.is_dir() or zi.file_size > 75_000_000:
                    continue
                try:
                    handle_member(zi.filename, z.read(zi))
                except Exception:
                    continue
            return frames, members[:200], "\n".join(text_parts)[:500000]
    except Exception:
        pass
    # TAR / TAR.GZ / TGZ
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            for i, ti in enumerate(tf.getmembers()):
                if i >= max_members or not ti.isfile() or (ti.size and ti.size > 75_000_000):
                    continue
                try:
                    fh = tf.extractfile(ti)
                    if fh:
                        handle_member(ti.name, fh.read())
                except Exception:
                    continue
            return frames, members[:200], "\n".join(text_parts)[:500000]
    except Exception:
        pass
    # Single gzip member.
    try:
        blob = gzip.decompress(data)
        name = source_url.split("/")[-1]
        if name.lower().endswith(".gz"):
            name = name[:-3]
        handle_member(name or "decompressed_member", blob)
    except Exception:
        pass
    return frames, members[:200], "\n".join(text_parts)[:500000]


def frames_from_artifact(test_id: str, data: bytes, url: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[List[pd.DataFrame], Dict[str, Any], str]:
    """v13 override: file-first parsing; metadata/search JSON never parsed as table; noisy HTML suppressed."""
    meta = meta or {}
    ctype = (meta.get("content_type") or "").lower()
    role = artifact_role(url, meta)
    diag: Dict[str, Any] = {"url": url, "content_type": ctype, "artifact_role": role, "extractors_tried": [], "archive_members_sample": []}
    text_sample = ""
    frames: List[pd.DataFrame] = []

    if role in {"metadata_record", "nondata_asset_or_false_positive"}:
        diag["extractors_tried"].append("metadata_or_nondata_link_extraction_only")
        try:
            text_sample = data.decode("utf-8", errors="replace")[:500000]
        except Exception:
            text_sample = ""
        diag["frames_extracted"] = 0
        diag["metadata_not_physical_table"] = role == "metadata_record"
        diag["nondata_asset_rejected"] = role == "nondata_asset_or_false_positive"
        return frames, diag, text_sample

    if role in {"archive", "source_package"}:
        diag["extractors_tried"].append("recursive_archive_or_source_package")
        aframes, members, txt = _frames_from_archive_v13(data, url)
        frames.extend(aframes); diag["archive_members_sample"] = members[:100]; text_sample += txt
        # Source packages may not be standard archives under Python tar/zip; fall back to existing parser.
        if role == "source_package" and not frames:
            try:
                aframes2, files2, txt2 = _frames_from_arxiv_or_tar(data, url, max_members=220)
                frames.extend(aframes2); diag["archive_members_sample"] = (diag.get("archive_members_sample") or []) + files2[:100]; text_sample += txt2
            except Exception:
                pass

    elif role in {"primary_table_file", "json_candidate", "primary_file_endpoint"}:
        diag["extractors_tried"].append("read_tabular_bytes_file_first")
        # Do not parse repository metadata JSON, even if extension/content-type is JSON.
        if is_metadata_like_frame_v13(pd.DataFrame(), url):
            diag["frames_extracted"] = 0
            diag["metadata_not_physical_table"] = True
            return frames, diag, text_sample
        try:
            for df in read_tabular_bytes(data, url):
                df.attrs["source_url"] = url
                df.attrs["evidence_tier"] = "primary_structured_public_table"
                if not is_metadata_like_frame_v13(df, url):
                    frames.append(df)
        except Exception as e:
            diag["structured_parse_error"] = f"{type(e).__name__}: {e}"

    elif role == "html" or (b"<html" in data[:2000].lower()):
        diag["extractors_tried"].append("html_read_table_with_noise_filter")
        text_sample = data.decode("utf-8", errors="replace")[:500000]
        if HTML_NOISE_RE_V13.search(text_sample[:20000]) and not DOMAIN_PHYSICAL_RE_V13.search(text_sample[:50000]):
            diag["html_noise_page_rejected_before_table_parse"] = True
        else:
            try:
                for df in read_tabular_bytes(data, url):
                    if is_metadata_like_frame_v13(df, url) or is_html_noise_frame_v13(df):
                        continue
                    df.attrs["source_url"] = url
                    df.attrs["evidence_tier"] = "html_table"
                    frames.append(df)
            except Exception as e:
                diag["html_table_error"] = f"{type(e).__name__}: {e}"

    elif role == "pdf":
        diag["extractors_tried"].append("pdf_tables_secondary")
        pframes = _frames_from_pdf_pdfplumber(data, url, max_pages=_pdf_page_budget_v12())
        tframes = _frames_from_pdf_text_lines(data, url, max_pages=_pdf_page_budget_v12())
        cframes = _frames_from_pdf_camelot(data, url)
        frames.extend(pframes); frames.extend(tframes); frames.extend(cframes)
        diag["pdfplumber_table_frames"] = len(pframes)
        diag["text_line_table_frames"] = len(tframes)
        diag["camelot_table_frames"] = len(cframes)
        text_sample = _text_from_pdf(data, max_pages=_pdf_page_budget_v12())
        diag["figure_digitization_attempt"] = _figure_vector_diagnostic(data)

    # Final physical/noise gate before returning frames. This keeps data-limited reports clean.
    filtered: List[pd.DataFrame] = []
    rejected_noise = 0
    for df in frames:
        src = str(df.attrs.get("source_url") or url)
        if is_metadata_like_frame_v13(df, src) or is_html_noise_frame_v13(df):
            rejected_noise += 1
            continue
        if not has_domain_physical_hint_v13(test_id, df, src):
            # Keep direct physical table files only if later contract matching can decide; reject generic HTML/source noise.
            if str(df.attrs.get("evidence_tier")) in {"html_table", "arxiv_latex_table", "secondary_auto_pdf_table"}:
                rejected_noise += 1
                continue
        filtered.append(df)
    diag["frames_extracted_before_v13_filter"] = len(frames)
    diag["frames_rejected_by_v13_noise_or_domain_gate"] = rejected_noise
    diag["frames_extracted"] = len(filtered)
    return filtered, diag, text_sample


def score_frame(test_id: str, df: pd.DataFrame, source_url: str, tier: str) -> Dict[str, Any]:
    """v13 override: require domain signal before candidate-table accounting; metadata never primary."""
    required = _regex_groups_from_contract(test_id)
    report = column_match_report(df, required)
    nums = numeric_columns(df)
    n_rows = int(df.shape[0])
    x_range = None
    for c in nums[:10]:
        vals = clean_numeric_series(df[c]).dropna().astype(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) >= 3:
            try:
                x_range = float(np.log10(vals.max()) - np.log10(vals.min()))
                break
            except Exception:
                pass
    min_rows = _min_rows_for(test_id, tier)
    role = artifact_role(source_url, {"content_type": ""})
    metadata_only = is_repository_metadata_record_url_v12(source_url) or is_metadata_like_frame_v13(df, source_url)
    matched_groups = len(report.get("matched_groups") or [])
    physical_hint_score = physical_hint_score_v12(test_id, df)
    domain_hint = has_domain_physical_hint_v13(test_id, df, source_url)
    html_noise = is_html_noise_frame_v13(df)
    has_some_physical = bool((matched_groups > 0 or physical_hint_score > 0 or domain_hint) and not html_noise and not metadata_only)
    has_all_physical = bool(report.get("ok") and has_some_physical)
    enough_numeric = len(nums) >= 2
    enough_rows = n_rows >= min_rows
    primary_tiers = {"primary_structured_public_table", "primary_structured_zip_member", "hepdata_table"}
    # Secondary extraction can qualify for model diagnostics, but not confirm/falsify.
    ok = bool((not metadata_only) and has_all_physical and enough_numeric and enough_rows)
    primary_allowed = bool(ok and tier in primary_tiers)
    rejection_reasons = []
    if metadata_only:
        rejection_reasons.append("metadata_record_not_physical_table")
    if html_noise:
        rejection_reasons.append("html_svg_or_boilerplate_noise")
    if not domain_hint and not physical_hint_score and not matched_groups:
        rejection_reasons.append("no_domain_physical_hint")
    if not has_some_physical:
        rejection_reasons.append("no_physical_observable_hint")
    if not has_all_physical:
        rejection_reasons.append("missing_required_physical_column_groups")
    if not enough_numeric:
        rejection_reasons.append("too_few_numeric_columns")
    if not enough_rows:
        rejection_reasons.append("below_min_rows_for_contract")
    if tier not in primary_tiers:
        rejection_reasons.append("secondary_or_nonprimary_evidence_tier")
    out = {
        "source_url": source_url,
        "artifact_role": "metadata_record" if metadata_only else role,
        "evidence_tier": "metadata_only" if metadata_only else tier,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in list(df.columns)[:100]],
        "numeric_columns": [str(c) for c in nums[:60]],
        "unit_normalization_hints": {str(c): normalize_unit_from_text(str(c)) for c in list(df.columns)[:80]},
        "physical_column_match": report,
        "physical_hint_score": physical_hint_score,
        "domain_physical_hint": bool(domain_hint),
        "has_some_physical_evidence": bool(has_some_physical),
        "sensitivity": sensitivity_classification(n_rows, x_range, min_rows=min_rows),
        "qualifies_for_model": ok,
        "confirmation_allowed": primary_allowed,
        "falsification_allowed": primary_allowed,
        "rejection_reasons": sorted(set(rejection_reasons)),
        "attrs": {k: v for k, v in getattr(df, "attrs", {}).items() if isinstance(v, (str, int, float, bool))},
    }
    out["table_relevance_score"] = table_relevance_score_v12(out)
    return out


def _zenodo_file_links_v12(obj: Any, url: str) -> List[Dict[str, Any]]:
    """v13 override: extract only record/file/content links; no metadata tables."""
    links: List[Dict[str, Any]] = []
    def add(u: str, label: str, reason: str):
        if u and isinstance(u, str) and u.startswith("http"):
            links.append({"url": u, "label": label, "reason": reason})
    def add_file(f: Dict[str, Any], label_prefix: str = "zenodo file"):
        lks = f.get("links") or {}
        key = f.get("key") or f.get("filename") or f.get("name") or label_prefix
        for lk in ["download", "self", "content", "archive"]:
            u = lks.get(lk) or f.get(lk)
            if isinstance(u, str) and u.startswith("http"):
                # Prefer explicit content endpoint when the API file link is only metadata.
                if "/api/records/" in u and "/files/" in u and not re.search(r"/(content|download)(\?|$)", u):
                    add(u.rstrip("/") + "/content", str(key), "zenodo_file_content_download")
                else:
                    add(u, str(key), "zenodo_file_download")
    if isinstance(obj, dict):
        hits = (((obj.get("hits") or {}).get("hits")) or [])
        if isinstance(hits, list):
            for rec in hits[:40]:
                if not isinstance(rec, dict):
                    continue
                rec_id = rec.get("id") or rec.get("recid")
                if rec_id:
                    add(f"https://zenodo.org/api/records/{rec_id}", f"Zenodo record {rec_id}", "zenodo_record_metadata")
                    add(f"https://zenodo.org/api/records/{rec_id}/files", f"Zenodo record {rec_id} files", "zenodo_files_endpoint")
                for f in rec.get("files") or []:
                    if isinstance(f, dict):
                        add_file(f)
        for f in obj.get("files") or []:
            if isinstance(f, dict):
                add_file(f)
        entries = obj.get("entries") or []
        if isinstance(entries, dict):
            for name, f in list(entries.items())[:200]:
                if isinstance(f, dict):
                    f = dict(f); f.setdefault("key", name); add_file(f)
        elif isinstance(entries, list):
            for f in entries[:200]:
                if isinstance(f, dict):
                    add_file(f)
        lks = obj.get("links") or {}
        for key in ["files", "archive", "media_files"]:
            u = lks.get(key)
            if isinstance(u, str) and u.startswith("http"):
                add(u, f"Zenodo {key}", "zenodo_files_or_archive_endpoint")
    return _dedupe_link_dicts(links)[:200]


def links_from_connector_json(obj: Any, url: str) -> List[Dict[str, Any]]:
    """v13 override: repository metadata only emits file/download/API follow-up links."""
    links: List[Dict[str, Any]] = []
    u = str(url or "")
    if "zenodo.org/api/records" in u:
        links.extend(_zenodo_file_links_v12(obj, u))
    elif "api.figshare.com/v2/articles" in u or u.startswith("figshare_search://"):
        links.extend(_figshare_file_links_v12(obj, u))
    elif "hepdata.net" in u:
        links.extend(_hepdata_links_v12(obj, u))
    elif "api.osf.io" in u:
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            for it in obj.get("data") or []:
                if not isinstance(it, dict):
                    continue
                attrs = it.get("attributes") or {}
                lks = it.get("links") or {}
                rel = it.get("relationships") or {}
                name = attrs.get("name") or attrs.get("materialized_path") or "osf file"
                if lks.get("download"):
                    links.append({"url": lks["download"], "label": name, "reason": "osf_download_link"})
                if lks.get("html"):
                    links.append({"url": lks["html"], "label": name, "reason": "osf_html_link"})
                for key in ["files", "parent_folder", "node"]:
                    try:
                        uu = (((rel.get(key) or {}).get("links") or {}).get("related") or {}).get("href")
                        if uu:
                            links.append({"url": uu, "label": f"osf_related_{key}", "reason": "osf_related"})
                    except Exception:
                        pass
            nxt = (obj.get("links") or {}).get("next")
            if nxt:
                links.append({"url": nxt, "label": "OSF next page", "reason": "pagination"})
    else:
        # For bibliographic metadata, keep only explicit supplementary/data/file links.
        try:
            raw_links = links_from_metadata_json(obj, u)
            for l in raw_links:
                lu = l.get("url", "")
                if DATA_LINK_RE.search(lu) or TABLE_EXT_RE_V13.search(lu) or ARCHIVE_EXT_RE_V13.search(lu) or PDF_RE.search(lu):
                    links.append(l)
        except Exception:
            pass
    # Drop known non-data assets and false-positive domains at link level.
    filtered = [l for l in links if not FALSE_POSITIVE_SOURCE_RE_V13.search(str(l.get("url", "")) + " " + str(l.get("label", "")))]
    return _dedupe_link_dicts(filtered)[:200]


def connector_candidate_urls(source_url: str, data: Optional[bytes] = None, meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """v13 override: conservative discovery with strong nondata filtering."""
    meta = meta or {}
    links: List[Dict[str, Any]] = []
    text = ""
    if data:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    role = artifact_role(source_url, meta)
    if role == "html" and text:
        links.extend(html_links(text, source_url))
    if role not in {"metadata_record", "nondata_asset_or_false_positive"}:
        links.extend(metadata_expansion_links(source_url, text[:300000] if text else ""))
    aid = arxiv_id_from_url_or_text(source_url, text[:20000] if text else "")
    if aid:
        links.append({"url": f"https://arxiv.org/e-print/{aid}", "label": f"arXiv source package {aid}", "reason": "arxiv_source_package"})
    # DOI to HEPData only when high-energy keywords exist.
    hay = source_url + " " + text[:12000]
    if re.search(r"atlas|cms|lhc|drell|higgs|met\b|cross[ -]?section|cosmic|hep|qgp|eta/s", hay, re.I):
        for doi in doi_candidates(hay)[:5]:
            links.append({"url": f"https://www.hepdata.net/search/?q={quote(doi)}&format=json", "label": "HEPData search by DOI", "reason": "hepdata_search"})
    filtered = [l for l in links if not FALSE_POSITIVE_SOURCE_RE_V13.search(str(l.get("url", "")) + " " + str(l.get("label", "")))]
    return _dedupe_link_dicts(filtered)[:160]


def automated_source_scan(test_id: str, args: Any, seed_sources: Sequence[Dict[str, Any]], max_sources: int = 25, max_depth: int = 2) -> Dict[str, Any]:
    """v13 override: strict data-limited scanner. Counts only domain-physical tables."""
    cache = cache_level(args.cache, f"v13_auto_discovery_{test_id}")
    queue = _dedupe_link_dicts([s for s in (list(seed_sources) + additional_seed_sources_v11(test_id)) if not FALSE_POSITIVE_SOURCE_RE_V13.search(str(s.get("url", "")) + " " + str(s.get("label", "")))])
    seen = set()
    source_records: List[Dict[str, Any]] = []
    physical_candidates: List[Dict[str, Any]] = []
    parsed_nonphysical: List[Dict[str, Any]] = []
    metadata_records: List[Dict[str, Any]] = []
    schema_artifacts: List[Dict[str, Any]] = []
    depth_by_url = {s.get("url"): 0 for s in queue}
    max_sources_eff = max(max_sources, 44 if test_id in {"T26", "T27", "T29", "T44", "T45", "T47", "T57", "T59"} else max_sources)
    try:
        per_req_timeout = int(getattr(args, "timeout", 45) or 45)
    except Exception:
        per_req_timeout = 45
    deadline = time.monotonic() + max(100, min(480, per_req_timeout * 9))
    stop_reason = None
    while queue and len(source_records) < max_sources_eff and time.monotonic() < deadline:
        src = queue.pop(0)
        url = src.get("url")
        if not url or url in seen or FALSE_POSITIVE_SOURCE_RE_V13.search(str(url) + " " + str(src.get("label", ""))):
            continue
        seen.add(url)
        depth = depth_by_url.get(url, 0)
        data, meta = _download_for_scan_v12(src, cache, args)
        rec: Dict[str, Any] = {"url": url, "label": src.get("label"), "seed_reason": src.get("reason"), "depth": depth, "meta": meta, "extracted_links": [], "artifact_diag": {}, "candidate_tables": [], "nonphysical_tables_sample": []}
        source_records.append(rec)
        if not data:
            continue
        frames, diag, text_sample = frames_from_artifact(test_id, data, url, meta)
        rec["artifact_diag"] = diag
        if diag.get("artifact_role") == "metadata_record":
            metadata_records.append({"url": url, "reason": src.get("reason"), "content_type": meta.get("content_type"), "bytes": meta.get("bytes")})
        if re.search(r"variables|schema|dictionary", str(url) + " " + str(src.get("label")), re.I) and ("pdf" in str(meta.get("content_type", "")).lower() or str(url).lower().endswith(".pdf")):
            schema = extract_schema_from_text(text_sample, test_id)
            if schema.get("variables_count"):
                schema_artifacts.append(schema); rec["schema_extracted"] = schema
        for df in frames[:160]:
            tier = str(df.attrs.get("evidence_tier") or ("primary_structured_public_table" if TABLE_EXT_RE_V13.search(str(url)) else "html_table"))
            sc = score_frame(test_id, df, str(df.attrs.get("source_url") or url), tier)
            if sc.get("artifact_role") == "metadata_record" or "metadata_record_not_physical_table" in sc.get("rejection_reasons", []):
                metadata_records.append({"url": sc.get("source_url"), "columns_sample": sc.get("columns", [])[:15], "reason": "parsed_metadata_frame_rejected"})
                continue
            if sc.get("has_some_physical_evidence"):
                physical_candidates.append(sc); rec["candidate_tables"].append(sc)
            else:
                parsed_nonphysical.append(sc)
                if len(rec["nonphysical_tables_sample"]) < 3:
                    rec["nonphysical_tables_sample"].append({"source_url": sc.get("source_url"), "shape": sc.get("shape"), "columns": sc.get("columns", [])[:12], "rejection_reasons": sc.get("rejection_reasons")})
        if depth < max_depth:
            links = connector_candidate_urls(str(url), data, meta)
            try:
                stripped = data.lstrip()[:1]
                if stripped in [b"{", b"["]:
                    obj = json.loads(data.decode("utf-8", errors="replace"))
                    links.extend(links_from_connector_json(obj, str(url)))
            except Exception:
                pass
            links = _dedupe_link_dicts([l for l in links if not FALSE_POSITIVE_SOURCE_RE_V13.search(str(l.get("url", "")) + " " + str(l.get("label", "")))])
            rec["extracted_links"] = links[:160]
            for l in links:
                u = l.get("url")
                if u and u not in seen and len(queue) < max_sources_eff * 6:
                    depth_by_url[u] = depth + 1
                    queue.append(l)
    if queue and time.monotonic() >= deadline:
        stop_reason = "autodiscovery_wall_clock_budget_reached"
    elif len(source_records) >= max_sources_eff:
        stop_reason = "autodiscovery_max_sources_reached"
    qualifying = [c for c in physical_candidates if c.get("qualifies_for_model")]
    primary = [c for c in qualifying if c.get("confirmation_allowed")]
    secondary = [c for c in qualifying if not c.get("confirmation_allowed")]
    if primary:
        status = "primary_table_model_possible"
    elif secondary:
        status = "secondary_model_possible_nonprimary"
    elif physical_candidates:
        status = "physical_candidate_tables_found_but_missing_columns_or_power"
    elif source_records:
        status = "sources_scanned_no_physical_candidate_tables"
    else:
        status = "no_sources_scanned"
    physical_candidates_sorted = sorted(physical_candidates, key=lambda c: c.get("table_relevance_score", 0), reverse=True)
    return {
        "version": "v13_data_limited_hardening_file_first_domain_gated",
        "generated_utc": utc_now(),
        "data_contract": CONTRACTS.get(test_id, {}),
        "seed_sources_count": len(seed_sources),
        "additional_auto_seed_sources_count": len(additional_seed_sources_v11(test_id)),
        "sources_scanned_count": len(source_records),
        "autodiscovery_stop_reason": stop_reason,
        "autodiscovery_queue_remaining": len(queue),
        "source_records_sample": source_records[:60],
        "metadata_records_seen_count": len(metadata_records),
        "metadata_records_sample": metadata_records[:30],
        "nonphysical_tables_parsed_count": len(parsed_nonphysical),
        "nonphysical_tables_sample": parsed_nonphysical[:20],
        "candidate_table_count": len(physical_candidates_sorted),
        "physical_candidate_table_count": len(physical_candidates_sorted),
        "qualifying_table_count": len(qualifying),
        "primary_qualifying_table_count": len(primary),
        "secondary_qualifying_table_count": len(secondary),
        "candidate_rejection_summary": rejection_summary(physical_candidates_sorted + parsed_nonphysical),
        "candidate_tables_sample": physical_candidates_sorted[:30],
        "qualifying_tables_sample": qualifying[:30],
        "schema_artifacts": schema_artifacts[:10],
        "nearest_miss": nearest_miss(test_id, physical_candidates_sorted, source_records),
        "automated_readiness_status": status,
        "strict_verdict_rule_v13": "Metadata/search JSON, HTML/SVG/icon fragments, and false-positive domains are link sources only; candidate_table_count includes only domain-physical tables after contract and noise gates. Repository connectors are file-first; archives are recursively traversed.",
        "evidence_ladder": {
            "E0": "no source found",
            "E1": "source found but no usable physical table",
            "E2": "secondary auto-extracted physical diagnostic table; not decisive",
            "E3": "primary machine-readable public physical table/model possible",
            "E4": "primary table with uncertainties/controls and adequate sensitivity",
        },
    }


def augment_result_with_autodiscovery(test_id: str, result: Dict[str, Any], args: Any, extra_sources: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    seeds = source_urls_from_result(test_id, result)
    if extra_sources:
        seeds.extend(extra_sources)
    if not seeds and test_id in {"T57", "T59"}:
        seeds.append({"url": "https://www.hepdata.net/search/?q=ATLAS%20CMS%20Drell-Yan%20Higgs%20MET&format=json", "label": "HEPData broad exact-table search", "reason": "hepdata_connector"})
    scan = automated_source_scan(test_id, args, seeds, max_sources=70 if test_id in {"T26", "T27", "T29", "T44", "T45", "T47", "T57", "T59"} else 40, max_depth=3)
    scan["model_diagnostics"] = simple_model_diagnostics(test_id, scan.get("qualifying_tables_sample") or [])
    result["automated_discovery_v13"] = scan
    result["automated_discovery_v12"] = scan
    result["automated_discovery_v11"] = scan
    result["automated_discovery_v10"] = scan
    result["quality_patch_version"] = str(result.get("quality_patch_version", "")) + "+v13_data_limited_hardening"
    if scan.get("primary_qualifying_table_count", 0):
        result["readiness_status"] = "primary_auto_discovered_physical_table_candidate"
        result["evidence_status"] = "analysis_ready_primary_auto_discovered"
    elif scan.get("secondary_qualifying_table_count", 0):
        result["readiness_status"] = "secondary_auto_extracted_physical_table_candidate_nonprimary"
        result["evidence_status"] = "data_limited_secondary_diagnostic_available"
    elif result.get("evidence_status") == "data_limited":
        result["readiness_status"] = result.get("readiness_status") or scan.get("automated_readiness_status")
    result["automated_no_manual_steps_policy"] = "All v13 discovery artifacts are downloaded or extracted automatically from public URLs. Metadata/search JSON and HTML/SVG boilerplate are never evidence. Candidate tables are domain-gated physical artifacts only; secondary PDF/figure/arXiv tables cannot confirm/falsify."
    return result

# v13.1 compatibility fix: Zenodo and HEPData commonly use a trailing slash before query strings.
def is_repository_metadata_record_url_v12(url: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    u = str(url or "")
    lu = u.lower()
    ctype = str((meta or {}).get("content_type") or "").lower()
    if u.startswith("figshare_search://"):
        return True
    # Explicit file/content downloads are physical file endpoints, not metadata.
    if re.search(r"/files/[^/?#]+/(?:content|download)(?:\?|$)|/files-archive(?:\?|$)|/media-files-archive(?:\?|$)", lu):
        return False
    if re.search(r"zenodo\.org/api/records/?(?:\?|$)", lu):
        return True
    if re.search(r"zenodo\.org/api/records/\d+/?(?:\?|$)", lu):
        return True
    if re.search(r"zenodo\.org/api/records/\d+/(?:files|versions|access|communities|draft|pids|quota|request|media-files)/?(?:\?|$)", lu):
        return True
    if re.search(r"api\.figshare\.com/v2/articles(?:/search|/\d+)?/?(?:\?|$)", lu):
        return True
    if re.search(r"api\.osf\.io/v2/", lu):
        return True
    if re.search(r"hepdata\.net/(?:search|record)/?", lu):
        return True
    if re.search(r"api\.(?:crossref|openalex|datacite|semanticscholar)\.org/", lu):
        return True
    if "application/json" in ctype and re.search(r"(api/records/?\?|/search/?\?|/works/|/dois/)", lu):
        return True
    return False

# ---------------------------------------------------------------------------
# v14 data-limited resolution layer: strict candidate accounting + targeted
# source-specific fixes for the 10 known data-limited blockers.
# ---------------------------------------------------------------------------

# Keep references to the v13 implementations for wrapping.
_frames_from_artifact_v13 = frames_from_artifact
_score_frame_v13 = score_frame
_artifact_role_v13 = artifact_role
_additional_seed_sources_v13 = additional_seed_sources_v11

README_NONDATA_RE_V14 = re.compile(
    r"(readme|read_me|license|citation|authors?|description|manifest|metadata|changelog|requirements|environment|questionnaire)",
    re.I,
)
ADDRESS_AFFILIATION_RE_V14 = re.compile(
    r"(institute|university|department|division\s+of|faculty|laborator|\broad\b|\bstreet\b|avenue|london|email|correspondence|postal|fulham|cancer\s+biology|dental|covid)",
    re.I,
)

# Test-specific domain gates. A table must normally hit at least one term and
# have numeric columns before it is counted as a candidate.
DOMAIN_TERMS_V14: Dict[str, List[str]] = {
    "T26": ["ELM", "E_ELM", "W_ELM", "pedestal", "P_ped", "Wped", "dP/P", "discharge", "shot", "tokamak", "JET", "DIII-D", "ASDEX", "AUG"],
    "T27": ["ELM frequency", "f_ELM", "RMP", "I-coil", "coil", "phasing", "helicity", "shot", "discharge", "DIII-D", "KSTAR"],
    "T28": ["tau_E", "TAUE", "TAUTH", "energy confinement", "H98", "H-factor", "nbar", "PLOSS", "Ip", "Bt", "tokamak"],
    "T29": ["W7-X", "W7-AS", "stellarator", "tokamak", "rho", "Te", "Ti", "ne", "diffusivity", "heat flux", "transport"],
    "T30": ["tau_E", "TAUE", "H98", "density", "elongation", "kappa", "triangularity", "q95", "R_major", "a_minor"],
    "T44": ["NAND", "V-NAND", "layers", "die area", "Gb", "bits/cell", "TLC", "QLC", "wafer"],
    "T45": ["energy per bit", "pJ/bit", "fJ/bit", "bandwidth", "Gbps", "link length", "interconnect", "photonic", "optical"],
    "T47": ["Loihi", "TrueNorth", "SpiNNaker", "BrainScaleS", "neuromorphic", "inference", "spike", "accuracy", "energy"],
    "T50": ["Casimir", "residual", "pressure", "force", "separation", "uncertainty", "systematic"],
    "T51": ["clock", "frequency", "drift", "fractional", "Allan", "baseline", "uncertainty", "systematic"],
    "T52": ["atom interferometer", "interferometer", "noise", "sensitivity", "strain", "acceleration", "baseline", "integration"],
    "T54": ["coherence", "lifetime", "dephasing", "oscillation", "FMO", "LH2", "photosystem", "temperature"],
    "T57": ["cross section", "sigma", "energy", "TeV", "PeV", "GeV", "cosmic", "uncertainty"],
    "T59": ["mass", "mT", "GeV", "TeV", "observed", "expected", "limit", "events", "Drell", "Higgs", "MET"],
}

# Better false-positive filter than the v13 global regex. This is used before
# expanding repository records and before treating text files as tables.
FALSE_POSITIVE_SOURCE_RE_V14 = re.compile(
    r"(earth\s+land\s+model|\bDELM\b|\bNoDELM\b|SNOTEL|squirrel|white\s+dwarf|"
    r"cancer\s+biology|dental|covid|questionnaire|survey|fomite|solar\s+radiation|"
    r"favicon|apple-touch-icon|css_|\.css(\?|$)|\.js(\?|$)|\.svg(\?|$)|\.png(\?|$)|\.jpg(\?|$)|\.ico(\?|$))",
    re.I,
)


def _domain_text_v14(df: Optional[pd.DataFrame] = None, source_url: str = "") -> str:
    txt = str(source_url or "")
    if df is not None:
        try:
            txt += " " + " ".join(str(c) for c in list(df.columns)[:120])
            if not df.empty:
                txt += " " + " ".join(str(x) for x in df.iloc[:30, : min(20, df.shape[1])].astype(str).values.ravel().tolist()[:800])
        except Exception:
            pass
    return txt


def _matched_domain_terms_v14(test_id: str, text: str) -> List[str]:
    low = str(text or "").lower().replace("_", " ")
    hits = []
    for term in DOMAIN_TERMS_V14.get(test_id, []):
        t = term.lower().replace("_", " ")
        if t in low:
            hits.append(term)
    return sorted(set(hits))


def _numeric_count_v14(df: pd.DataFrame) -> int:
    try:
        return len(numeric_columns(df))
    except Exception:
        return 0


def _is_nondata_text_frame_v14(df: pd.DataFrame, source_url: str = "") -> bool:
    name = str(source_url or "").lower()
    txt = _domain_text_v14(df, source_url)
    nums = _numeric_count_v14(df)
    if README_NONDATA_RE_V14.search(name) and nums < 2:
        return True
    if ADDRESS_AFFILIATION_RE_V14.search(txt) and nums == 0:
        return True
    # Plain text parsed into a table with no numeric content and no domain terms is never data.
    if nums == 0 and not _matched_domain_terms_v14("T26", txt) and not DOMAIN_PHYSICAL_RE_V13.search(txt):
        return True
    return False


def _looks_like_nondata_text_artifact_v14(url: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    u = str(url or "")
    ctype = str((meta or {}).get("content_type") or "").lower()
    if README_NONDATA_RE_V14.search(u) and ("text" in ctype or u.lower().endswith(('.txt','.md','.rst'))):
        return True
    return False


def artifact_role(url: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """v14 override: README/description/citation text is metadata-only unless later proven numeric+physical."""
    if _looks_like_nondata_text_artifact_v14(url, meta):
        return "metadata_or_readme_text"
    if FALSE_POSITIVE_SOURCE_RE_V14.search(str(url or "")):
        return "nondata_asset_or_false_positive"
    return _artifact_role_v13(url, meta)


def _record_is_relevant_v14(test_id: str, obj: Any, url: str = "") -> bool:
    """Repository record-level relevance gate before following every file link."""
    text = str(url or "")
    try:
        if isinstance(obj, dict):
            # Keep only semantically useful metadata fields; avoid huge nested link dumps.
            for key in ["title", "description", "publication_title", "metadata", "subjects", "keywords", "resource_type", "creators"]:
                if key in obj:
                    text += " " + json.dumps(obj.get(key), ensure_ascii=False)[:8000]
            if isinstance(obj.get("metadata"), dict):
                md = obj["metadata"]
                for key in ["title", "description", "keywords", "subjects", "resource_type"]:
                    text += " " + json.dumps(md.get(key, ""), ensure_ascii=False)[:8000]
        else:
            text += " " + json.dumps(obj, ensure_ascii=False)[:20000]
    except Exception:
        text += " " + str(obj)[:20000]
    if FALSE_POSITIVE_SOURCE_RE_V14.search(text):
        return False
    hits = _matched_domain_terms_v14(test_id, text)
    # HEP exact-table searches can be kept with fewer domain hits because table URLs are often neutral.
    if test_id in {"T57", "T59"} and re.search(r"hepdata|atlas|cms|lhc|drell|higgs|met|cross.section|cosmic", text, re.I):
        return True
    return len(hits) > 0


def _fusion_pdf_text_frames_v14(test_id: str, data: bytes, source_url: str) -> List[pd.DataFrame]:
    """Secondary-only fusion/metrology/coherence text-layer extractor.

    It does not claim primary evidence. It finds unit-bearing lines near domain terms
    and turns them into triage rows so the run can say what the PDF contains.
    """
    if test_id not in {"T26", "T27", "T29", "T50", "T51", "T52", "T54"}:
        return []
    text = _text_from_pdf(data, max_pages=20)
    if not text:
        return []
    rows = []
    unit_rx = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*(?:e[-+]?\d+)?)?\s*(MJ|kJ|J|MW|kW|ms|s|kPa|Pa|m\^?3|m-3|10\^19|pJ/bit|fJ/bit|Hz|K|nm|um|µm|mm|cm|GeV|TeV|PeV|%)", re.I)
    line_terms = DOMAIN_TERMS_V14.get(test_id, [])
    for i, line in enumerate(text.splitlines()):
        ln = re.sub(r"\s+", " ", line).strip()
        if len(ln) < 8 or len(ln) > 400:
            continue
        hits = [t for t in line_terms if t.lower().replace("_", " ") in ln.lower().replace("_", " ")]
        nums = unit_rx.findall(ln)
        # Also allow nearby domain keywords around unit rows for PDFs whose numeric tables split headers and rows.
        if hits and (nums or re.search(r"\d", ln)):
            rows.append({"line_index": i, "raw_line": ln, "matched_terms": ";".join(hits), "units_found": ";".join(nums), "source_url": source_url})
    if len(rows) < 3:
        return []
    df = pd.DataFrame(rows)
    df.attrs["source_url"] = source_url + "#auto_pdf_text_units"
    df.attrs["evidence_tier"] = "secondary_auto_pdf_text_table"
    df.attrs["artifact_role"] = "secondary_pdf_text_units"
    return [df]


def frames_from_artifact(test_id: str, data: bytes, url: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[List[pd.DataFrame], Dict[str, Any], str]:
    """v14 wrapper around v13: refuse README/nondata frames and add text-unit extraction."""
    role = artifact_role(url, meta)
    if role in {"metadata_or_readme_text", "nondata_asset_or_false_positive"}:
        text = ""
        try:
            text = data.decode("utf-8", errors="replace")[:200000]
        except Exception:
            pass
        diag = {"url": url, "content_type": (meta or {}).get("content_type"), "artifact_role": role, "extractors_tried": ["metadata_or_readme_text_link_extraction_only"], "frames_extracted": 0, "metadata_not_physical_table": True}
        return [], diag, text
    frames, diag, text_sample = _frames_from_artifact_v13(test_id, data, url, meta)
    # Add secondary text-unit extraction for PDFs when generic table extraction failed or was sparse.
    ctype = str((meta or {}).get("content_type") or "").lower()
    if (str(url).lower().endswith(".pdf") or "pdf" in ctype) and test_id in {"T26", "T27", "T29", "T50", "T51", "T52", "T54"}:
        extra = _fusion_pdf_text_frames_v14(test_id, data, url)
        if extra:
            frames.extend(extra)
            diag.setdefault("extractors_tried", []).append("v14_pdf_text_unit_extractor_secondary")
            diag["v14_pdf_text_unit_frames"] = len(extra)
    filtered = []
    rejected = 0
    for df in frames:
        src = str(getattr(df, "attrs", {}).get("source_url") or url)
        if _is_nondata_text_frame_v14(df, src):
            rejected += 1
            continue
        filtered.append(df)
    diag["frames_extracted_before_v14_filter"] = len(frames)
    diag["frames_rejected_by_v14_readme_or_nondata_gate"] = rejected
    diag["frames_extracted"] = len(filtered)
    return filtered, diag, text_sample


def score_frame(test_id: str, df: pd.DataFrame, source_url: str, tier: str) -> Dict[str, Any]:
    """v14 strict candidate score. A candidate must be numeric and domain/contract-linked."""
    sc = _score_frame_v13(test_id, df, source_url, tier)
    nums = sc.get("numeric_columns") or []
    match = sc.get("physical_column_match") or {}
    matched_groups = match.get("matched_groups") or []
    text = _domain_text_v14(df, source_url)
    domain_terms = _matched_domain_terms_v14(test_id, text)
    nondata = _is_nondata_text_frame_v14(df, source_url)
    if nondata:
        sc["artifact_role"] = "metadata_or_readme_text"
        sc["evidence_tier"] = "metadata_only"
    sc["domain_terms_v14"] = domain_terms[:40]
    sc["candidate_table_v14"] = bool((not nondata) and len(nums) >= 2 and (len(matched_groups) >= 1 or len(domain_terms) >= 1))
    sc["physical_candidate_v14"] = bool((not nondata) and len(nums) >= 2 and len(matched_groups) >= 1)
    # Model eligibility remains all groups + rows + numeric, but metadata/readme cannot qualify.
    if nondata or not sc["candidate_table_v14"]:
        sc["qualifies_for_model"] = False
        sc["confirmation_allowed"] = False
        sc["falsification_allowed"] = False
    reasons = set(sc.get("rejection_reasons") or [])
    if nondata:
        reasons.add("metadata_or_readme_text_not_data")
    if len(nums) < 2:
        reasons.add("too_few_numeric_columns")
    if len(matched_groups) == 0:
        reasons.add("no_required_group_matched")
    if not domain_terms and len(matched_groups) == 0:
        reasons.add("no_test_domain_terms")
    sc["rejection_reasons"] = sorted(reasons)
    return sc


def links_from_connector_json_v14(obj: Any, url: str, test_id: str) -> List[Dict[str, Any]]:
    """File-first connector with record-level relevance filter."""
    if not _record_is_relevant_v14(test_id, obj, url):
        return []
    return links_from_connector_json(obj, url)


def connector_candidate_urls_v14(source_url: str, data: Optional[bytes], meta: Optional[Dict[str, Any]], test_id: str) -> List[Dict[str, Any]]:
    links = connector_candidate_urls(source_url, data, meta)
    role = artifact_role(source_url, meta)
    # Do not DOI-expand broad repository search metadata unless record is domain-relevant.
    if role == "metadata_record":
        try:
            obj = json.loads((data or b"").decode("utf-8", errors="replace")) if data else {}
            if not _record_is_relevant_v14(test_id, obj, source_url):
                return []
        except Exception:
            pass
    return [l for l in links if not FALSE_POSITIVE_SOURCE_RE_V14.search(str(l.get("url", "")) + " " + str(l.get("label", "")))]


def additional_seed_sources_v11(test_id: str) -> List[Dict[str, Any]]:
    """v14 targeted seeds: exact/near-exact public repositories, not broad web noise."""
    seeds = _additional_seed_sources_v13(test_id)
    add: List[Dict[str, Any]] = []
    if test_id in {"T26", "T27"}:
        add.extend([
            {"url": "https://zenodo.org/api/records/?q=%22ELM%20energy%20loss%22%20tokamak%20pedestal&size=10", "label": "Zenodo exact ELM energy tokamak pedestal", "reason": "v14_exact_fusion_query"},
            {"url": "https://zenodo.org/api/records/?q=%22ELM%20frequency%22%20RMP%20tokamak&size=10", "label": "Zenodo exact ELM frequency RMP", "reason": "v14_exact_fusion_query"},
            {"url": "https://zenodo.org/api/records/?q=%22DIII-D%22%20RMP%20%22ELM%20frequency%22&size=10", "label": "Zenodo DIII-D RMP ELM frequency", "reason": "v14_exact_fusion_query"},
        ])
    if test_id in {"T28", "T30"}:
        add.extend([
            {"url": "https://zenodo.org/api/records/?q=%22energy%20confinement%20time%22%20H98%20tokamak%20density&size=10", "label": "Alternative H-mode public table search", "reason": "v14_hmode_schema_search"},
            {"url": "https://zenodo.org/api/records/?q=TAUE%20H98%20q95%20elongation%20tokamak&size=10", "label": "Alternative TAUE H98 q95 tokamak search", "reason": "v14_hmode_schema_search"},
        ])
    if test_id == "T29":
        add.extend([
            {"url": "https://zenodo.org/api/records/?q=%22W7-X%22%20profile%20transport%20diffusivity&size=10", "label": "W7-X profile transport search", "reason": "v14_profile_search"},
            {"url": "https://zenodo.org/api/records/?q=stellarator%20tokamak%20Te%20ne%20profile%20transport&size=10", "label": "stellarator/tokamak profile table search", "reason": "v14_profile_search"},
        ])
    if test_id == "T44":
        add.extend([
            {"url": "https://en.wikichip.org/wiki/3d_nand", "label": "WikiChip 3D NAND", "reason": "v14_electronics_connector"},
            {"url": "https://zenodo.org/api/records/?q=3D%20NAND%20layers%20die%20area%20Gb&size=10", "label": "3D NAND die area dataset search", "reason": "v14_electronics_connector"},
        ])
    if test_id == "T45":
        add.extend([
            {"url": "https://zenodo.org/api/records/?q=optical%20interconnect%20energy%20per%20bit%20bandwidth&size=10", "label": "optical interconnect energy per bit dataset search", "reason": "v14_electronics_connector"},
            {"url": "https://zenodo.org/api/records/?q=IRDS%20interconnect%20energy%20per%20bit%20optical&size=10", "label": "IRDS optical interconnect table search", "reason": "v14_electronics_connector"},
        ])
    if test_id == "T47":
        add.extend([
            {"url": "https://zenodo.org/api/records/?q=Loihi%20TrueNorth%20neuromorphic%20energy%20inference%20accuracy&size=10", "label": "neuromorphic benchmark dataset search", "reason": "v14_electronics_connector"},
            {"url": "https://zenodo.org/api/records/?q=SpiNNaker%20Loihi%20energy%20benchmark%20neuromorphic&size=10", "label": "SpiNNaker Loihi benchmark dataset search", "reason": "v14_electronics_connector"},
        ])
    if test_id in {"T50", "T51", "T52"}:
        add.extend([
            {"url": "https://zenodo.org/api/records/?q=Casimir%20residual%20force%20uncertainty%20separation&size=10", "label": "Casimir residual table search", "reason": "v14_metrology_bound_connector"},
            {"url": "https://zenodo.org/api/records/?q=optical%20clock%20frequency%20drift%20uncertainty%20baseline&size=10", "label": "optical clock drift table search", "reason": "v14_metrology_bound_connector"},
            {"url": "https://zenodo.org/api/records/?q=atom%20interferometer%20sensitivity%20noise%20baseline%20uncertainty&size=10", "label": "atom interferometer sensitivity table search", "reason": "v14_metrology_bound_connector"},
        ])
    if test_id == "T54":
        add.extend([
            {"url": "https://zenodo.org/api/records/?q=photosynthetic%20coherence%20lifetime%202D%20spectroscopy%20temperature&size=10", "label": "photosynthetic coherence lifetime dataset search", "reason": "v14_bio_direct_connector"},
            {"url": "https://zenodo.org/api/records/?q=FMO%20coherence%20oscillation%20lifetime%20temperature&size=10", "label": "FMO coherence table search", "reason": "v14_bio_direct_connector"},
        ])
    if test_id == "T57":
        add.extend([
            {"url": "https://www.hepdata.net/search/?q=cosmic%20ray%20cross%20section%20energy&format=json", "label": "HEPData cosmic cross-section exact search", "reason": "v14_exact_hepdata_manifest"},
            {"url": "https://zenodo.org/api/records/?q=cosmic%20ray%20cross%20section%20energy%20uncertainty&size=10", "label": "cosmic ray cross-section data search", "reason": "v14_exact_hepdata_manifest"},
        ])
    if test_id == "T59":
        add.extend([
            {"url": "https://www.hepdata.net/search/?q=ATLAS%20MET%20observed%20expected%20limit%20GeV&format=json", "label": "T59a MET HEPData exact search", "reason": "v14_exact_hepdata_manifest"},
            {"url": "https://www.hepdata.net/search/?q=Drell-Yan%20mass%20spectrum%20observed%20expected%20TeV&format=json", "label": "T59b DY HEPData exact search", "reason": "v14_exact_hepdata_manifest"},
            {"url": "https://www.hepdata.net/search/?q=di-Higgs%20mass%20observed%20expected%20limit&format=json", "label": "T59c di-Higgs HEPData exact search", "reason": "v14_exact_hepdata_manifest"},
        ])
    return _dedupe_link_dicts(seeds + add)


def _bound_summary_v14(test_id: str, qualifying: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if test_id not in {"T50", "T51", "T52"}:
        return {}
    return {
        "upper_limit_only": True,
        "confirmation_forbidden": True,
        "bound_ready": bool(qualifying),
        "required_bound_fields": CONTRACTS.get(test_id, {}).get("required_column_groups", []),
        "interpretation": "Metrology tests can only produce bounds/exclusions. They cannot confirm CCDR-like effects without a predefined residual likelihood.",
    }


def automated_source_scan(test_id: str, args: Any, seed_sources: Sequence[Dict[str, Any]], max_sources: int = 25, max_depth: int = 2) -> Dict[str, Any]:
    """v14 override: targeted data-limited scanner with strict candidate accounting."""
    cache = cache_level(args.cache, f"v14_auto_discovery_{test_id}")
    queue = _dedupe_link_dicts([s for s in (list(seed_sources) + additional_seed_sources_v11(test_id)) if not FALSE_POSITIVE_SOURCE_RE_V14.search(str(s.get("url", "")) + " " + str(s.get("label", "")))])
    # Prioritize exact/source-data artifacts over broad repository searches.
    def prio(s: Dict[str, Any]) -> int:
        txt = (str(s.get("reason", "")) + " " + str(s.get("url", "")) + " " + str(s.get("label", ""))).lower()
        if "file_download" in txt or "/content" in txt or "source-data" in txt or "supplement" in txt:
            return 0
        if "manifest_records" in txt or "exact" in txt or "schema" in txt:
            return 1
        if "api/records/?q" in txt or "search" in txt:
            return 3
        return 2
    queue = sorted(queue, key=prio)
    seen = set(); source_records=[]; physical_candidates=[]; parsed_nonphysical=[]; metadata_records=[]; schema_artifacts=[]
    depth_by_url = {s.get("url"): 0 for s in queue}
    max_sources_eff = max(max_sources, 56 if test_id in {"T26", "T27", "T28", "T29", "T30", "T44", "T45", "T47", "T50", "T51", "T52", "T54", "T57", "T59"} else max_sources)
    try:
        per_req_timeout = int(getattr(args, "timeout", 45) or 45)
    except Exception:
        per_req_timeout = 45
    deadline = time.monotonic() + max(100, min(540, per_req_timeout * 9))
    stop_reason = None
    broad_searches_used = 0
    while queue and len(source_records) < max_sources_eff and time.monotonic() < deadline:
        src = queue.pop(0); url = src.get("url")
        if not url or url in seen or FALSE_POSITIVE_SOURCE_RE_V14.search(str(url) + " " + str(src.get("label", ""))):
            continue
        # Stop broad search expansion sooner after enough exact/source attempts.
        if "api/records/?q" in str(url) or "search" in str(src.get("reason", "")):
            broad_searches_used += 1
            if broad_searches_used > 12 and not physical_candidates:
                # keep the run bounded and explain the stop reason later
                continue
        seen.add(url); depth = depth_by_url.get(url, 0)
        data, meta = _download_for_scan_v12(src, cache, args)
        rec = {"url": url, "label": src.get("label"), "seed_reason": src.get("reason"), "depth": depth, "meta": meta, "extracted_links": [], "artifact_diag": {}, "candidate_tables": [], "nonphysical_tables_sample": []}
        source_records.append(rec)
        if not data:
            continue
        frames, diag, text_sample = frames_from_artifact(test_id, data, url, meta)
        rec["artifact_diag"] = diag
        role = diag.get("artifact_role") or artifact_role(url, meta)
        if role in {"metadata_record", "metadata_or_readme_text"}:
            metadata_records.append({"url": url, "label": src.get("label"), "reason": src.get("reason"), "artifact_role": role, "content_type": meta.get("content_type"), "bytes": meta.get("bytes")})
        if re.search(r"variables|schema|dictionary", str(url) + " " + str(src.get("label")), re.I) and ("pdf" in str(meta.get("content_type", "")).lower() or str(url).lower().endswith(".pdf")):
            schema = extract_schema_from_text(text_sample, test_id)
            if schema.get("variables_count"):
                schema["schema_csv_suggested_path"] = "data/generated/itpa_db523_schema.csv"
                schema_artifacts.append(schema); rec["schema_extracted"] = schema
        for df in frames[:200]:
            tier = str(df.attrs.get("evidence_tier") or ("primary_structured_public_table" if TABLE_EXT_RE_V13.search(str(url)) else "html_table"))
            sc = score_frame(test_id, df, str(df.attrs.get("source_url") or url), tier)
            if sc.get("artifact_role") in {"metadata_record", "metadata_or_readme_text"} or "metadata_record_not_physical_table" in sc.get("rejection_reasons", []):
                metadata_records.append({"url": sc.get("source_url"), "columns_sample": sc.get("columns", [])[:12], "reason": "parsed_metadata_or_readme_rejected"})
                continue
            if sc.get("candidate_table_v14"):
                physical_candidates.append(sc); rec["candidate_tables"].append(sc)
            else:
                parsed_nonphysical.append(sc)
                if len(rec["nonphysical_tables_sample"]) < 4:
                    rec["nonphysical_tables_sample"].append({"source_url": sc.get("source_url"), "shape": sc.get("shape"), "columns": sc.get("columns", [])[:12], "rejection_reasons": sc.get("rejection_reasons")})
        if depth < max_depth:
            links=[]
            try:
                stripped=data.lstrip()[:1]
                if stripped in [b"{", b"["]:
                    obj=json.loads(data.decode("utf-8", errors="replace"))
                    links.extend(links_from_connector_json_v14(obj, str(url), test_id))
            except Exception:
                pass
            links.extend(connector_candidate_urls_v14(str(url), data, meta, test_id))
            links=_dedupe_link_dicts([l for l in links if not FALSE_POSITIVE_SOURCE_RE_V14.search(str(l.get("url", "")) + " " + str(l.get("label", "")))])
            # Prioritize file content and domain labels.
            links=sorted(links, key=prio)
            rec["extracted_links"] = links[:180]
            for l in links:
                u=l.get("url")
                if u and u not in seen and len(queue) < max_sources_eff*7:
                    depth_by_url[u]=depth+1; queue.append(l)
            queue=sorted(_dedupe_link_dicts(queue), key=prio)
    if queue and time.monotonic() >= deadline:
        stop_reason="autodiscovery_wall_clock_budget_reached"
    elif len(source_records) >= max_sources_eff:
        stop_reason="autodiscovery_max_sources_reached"
    elif broad_searches_used > 12 and not physical_candidates:
        stop_reason="broad_search_budget_exhausted_no_domain_physical_files"
    qualifying=[c for c in physical_candidates if c.get("qualifies_for_model")]
    primary=[c for c in qualifying if c.get("confirmation_allowed")]
    secondary=[c for c in qualifying if not c.get("confirmation_allowed")]
    if primary:
        status="primary_table_model_possible"
    elif secondary:
        status="secondary_model_possible_nonprimary"
    elif physical_candidates:
        status="candidate_physical_tables_found_but_missing_columns_or_power"
    elif source_records:
        status="sources_scanned_no_physical_candidate_tables"
    else:
        status="no_sources_scanned"
    physical_candidates_sorted=sorted(physical_candidates, key=lambda c: c.get("table_relevance_score", 0), reverse=True)
    bound_summary=_bound_summary_v14(test_id, qualifying)
    return {
        "version":"v14_data_limited_resolution_targeted_connectors",
        "generated_utc":utc_now(),
        "data_contract":CONTRACTS.get(test_id, {}),
        "seed_sources_count":len(seed_sources),
        "additional_auto_seed_sources_count":len(additional_seed_sources_v11(test_id)),
        "sources_scanned_count":len(source_records),
        "autodiscovery_stop_reason":stop_reason,
        "autodiscovery_queue_remaining":len(queue),
        "source_quality_ladder_v14":["curated/source-data file links", "exact repository records/files", "arXiv/source packages", "publisher supplements", "broad repository search as last resort"],
        "source_records_sample":source_records[:80],
        "metadata_records_seen_count":len(metadata_records),
        "metadata_records_sample":metadata_records[:40],
        "nonphysical_tables_parsed_count":len(parsed_nonphysical),
        "nonphysical_tables_sample":parsed_nonphysical[:20],
        "candidate_table_count":len(physical_candidates_sorted),
        "physical_candidate_table_count":len(physical_candidates_sorted),
        "qualifying_table_count":len(qualifying),
        "primary_qualifying_table_count":len(primary),
        "secondary_qualifying_table_count":len(secondary),
        "candidate_rejection_summary":rejection_summary(physical_candidates_sorted+parsed_nonphysical),
        "candidate_tables_sample":physical_candidates_sorted[:30],
        "qualifying_tables_sample":qualifying[:30],
        "schema_artifacts":schema_artifacts[:12],
        "bound_only_summary_v14":bound_summary,
        "nearest_miss":nearest_miss(test_id, physical_candidates_sorted, source_records),
        "automated_readiness_status":status,
        "strict_verdict_rule_v14":"README/plain text metadata, repository search JSON, HTML/SVG/icon fragments, and false-positive domains are not candidates. candidate_table_count requires numeric columns plus a matched contract/domain term. Metrology tests are bound-only.",
        "evidence_ladder":{
            "E0":"no source found",
            "E1":"source found but no usable physical table",
            "E2":"secondary auto-extracted physical diagnostic table; not decisive",
            "E3":"primary machine-readable public physical table/model possible",
            "E4":"primary table with uncertainties/controls and adequate sensitivity",
        },
    }


def augment_result_with_autodiscovery(test_id: str, result: Dict[str, Any], args: Any, extra_sources: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    seeds = source_urls_from_result(test_id, result)
    if extra_sources:
        seeds.extend(extra_sources)
    if not seeds and test_id in {"T57", "T59"}:
        seeds.append({"url": "https://www.hepdata.net/search/?q=ATLAS%20CMS%20Drell-Yan%20Higgs%20MET&format=json", "label": "HEPData broad exact-table search", "reason": "hepdata_connector"})
    scan = automated_source_scan(test_id, args, seeds, max_sources=80 if test_id in {"T26", "T27", "T28", "T29", "T30", "T44", "T45", "T47", "T50", "T51", "T52", "T54", "T57", "T59"} else 45, max_depth=3)
    scan["model_diagnostics"] = simple_model_diagnostics(test_id, scan.get("qualifying_tables_sample") or [])
    result["automated_discovery_v14"] = scan
    result["automated_discovery_v13"] = scan
    result["automated_discovery_v12"] = scan
    result["automated_discovery_v11"] = scan
    result["automated_discovery_v10"] = scan
    result["quality_patch_version"] = str(result.get("quality_patch_version", "")) + "+v14_data_limited_resolution"
    result["automated_no_manual_steps_policy"] = "All v14 discovery artifacts are downloaded or extracted automatically from public URLs. Candidate tables require numeric columns plus test-domain/contract evidence. README/search/metadata records are never data."
    return result



# ---------------------------------------------------------------------------
# v15 data-limited precision layer: exact-source manifests, record-metadata-only
# relevance, archive guards, and explicit public-unavailable statuses.
# ---------------------------------------------------------------------------

_automated_source_scan_v14 = automated_source_scan
_additional_seed_sources_v14 = additional_seed_sources_v11
_download_for_scan_base_v15 = _download_for_scan_v12
_record_is_relevant_base_v15 = _record_is_relevant_v14

EXACT_MANIFEST_FILES_V15 = [
    "fusion_exact_source_manifest.csv",
    "electronics_exact_source_manifest.csv",
    "metrology_bound_manifest.csv",
    "biology_coherence_source_manifest.csv",
    "exact_hepdata_manifest.csv",
]

DOMAIN_POSITIVE_FIELDS_V15 = [
    "title", "description", "keywords", "subjects", "resource_type", "creators",
    "publication_title", "abstract", "notes", "communities", "related_identifiers",
]

# Exact public URLs found or used as official/near-official discovery endpoints.
# These are URLs only: no manual numeric rows are added.
EXTRA_EXACT_SOURCE_URLS_V15: Dict[str, List[Dict[str, str]]] = {
    "T26": [
        {"url": "https://scipub.euro-fusion.org/wp-content/uploads/2014/11/EFDP03032.pdf", "label": "Loarte Type-I ELM energy loss PDF", "reason": "v15_exact_fusion_pdf"},
        {"url": "https://www.iter.org/sites/default/files/education/Liang_Yunfeng_talk.pdf", "label": "ITER/RMP ELM control talk PDF", "reason": "v15_exact_fusion_pdf"},
        {"url": "https://zenodo.org/api/records/?q=%22Type%20I%20ELM%22%20%22pedestal%20energy%22%20tokamak&size=10", "label": "exact Type-I ELM pedestal energy repository query", "reason": "v15_exact_repository_query"},
    ],
    "T27": [
        {"url": "https://www.iter.org/sites/default/files/education/Liang_Yunfeng_talk.pdf", "label": "RMP ELM control public PDF", "reason": "v15_exact_fusion_pdf"},
        {"url": "https://zenodo.org/api/records/?q=%22RMP%22%20%22ELM%20frequency%22%20tokamak%20shot&size=10", "label": "exact RMP ELM frequency repository query", "reason": "v15_exact_repository_query"},
    ],
    "T28": [
        {"url": "https://osf.io/drwcq/", "label": "International Global H-Mode Confinement Database OSF page", "reason": "v15_itpa_osf_page"},
        {"url": "https://api.osf.io/v2/nodes/drwcq/files/osfstorage/", "label": "International Global H-Mode Confinement Database OSF API", "reason": "v15_itpa_osf_api"},
        {"url": "https://pure.mpg.de/rest/items/item_3325255_5/component/file_3357083/content", "label": "Updated ITPA global H-mode confinement database paper PDF", "reason": "v15_itpa_schema_pdf"},
        {"url": "https://zenodo.org/api/records/?q=%22H98%22%20%22energy%20confinement%20time%22%20tokamak%20q95&size=10", "label": "H98 tau_E q95 exact public table query", "reason": "v15_alt_hmode_query"},
    ],
    "T30": [
        {"url": "https://api.osf.io/v2/nodes/drwcq/files/osfstorage/", "label": "ITPA DB5.2.3 OSF API for shaping/density", "reason": "v15_itpa_osf_api"},
        {"url": "https://pure.mpg.de/rest/items/item_3325255_5/component/file_3357083/content", "label": "Updated ITPA database paper for variable/schema extraction", "reason": "v15_itpa_schema_pdf"},
        {"url": "https://zenodo.org/api/records/?q=%22elongation%22%20%22triangularity%22%20%22H98%22%20tokamak%20density&size=10", "label": "H98 shaping density exact public table query", "reason": "v15_alt_hmode_query"},
    ],
    "T29": [
        {"url": "https://zenodo.org/api/records/?q=%22W7-X%22%20%22electron%20temperature%22%20density%20profile%20transport&size=10", "label": "W7-X Te/ne profile transport exact query", "reason": "v15_profile_query"},
        {"url": "https://zenodo.org/api/records/?q=%22stellarator%22%20%22tokamak%22%20profile%20diffusivity%20heat%20flux&size=10", "label": "stellarator tokamak diffusivity profile exact query", "reason": "v15_profile_query"},
    ],
    "T44": [
        {"url": "https://en.wikichip.org/wiki/3d_nand", "label": "WikiChip 3D NAND page", "reason": "v15_exact_electronics_page"},
        {"url": "https://library.techinsights.com/sectioned-blog-viewer/91b66ff4-0a0f-4c01-89a4-b0260626b7e4", "label": "TechInsights 3D NAND TLC bit-density ranking", "reason": "v15_exact_electronics_page"},
        {"url": "https://zenodo.org/api/records/?q=%223D%20NAND%22%20%22die%20area%22%20%22layers%22%20%22Gb%22&size=10", "label": "3D NAND die area layers repository query", "reason": "v15_exact_repository_query"},
    ],
    "T45": [
        {"url": "https://opg.optica.org/abstract.cfm?uri=oe-23-3-2085", "label": "Optical interconnect bandwidth-density table source", "reason": "v15_exact_electronics_page"},
        {"url": "https://zenodo.org/api/records/?q=%22pJ%2Fbit%22%20%22optical%20interconnect%22%20bandwidth&size=10", "label": "pJ/bit optical interconnect repository query", "reason": "v15_exact_repository_query"},
    ],
    "T47": [
        {"url": "https://zenodo.org/api/records/?q=%22Loihi%22%20%22energy%20per%20inference%22%20accuracy&size=10", "label": "Loihi energy per inference exact query", "reason": "v15_exact_repository_query"},
        {"url": "https://zenodo.org/api/records/?q=%22TrueNorth%22%20neuromorphic%20benchmark%20energy%20accuracy&size=10", "label": "TrueNorth neuromorphic benchmark exact query", "reason": "v15_exact_repository_query"},
    ],
    "T50": [
        {"url": "https://zenodo.org/api/records/?q=%22Casimir%22%20%22residual%20force%22%20separation%20uncertainty&size=10", "label": "Casimir residual force exact query", "reason": "v15_bound_query"},
    ],
    "T51": [
        {"url": "https://zenodo.org/api/records/?q=%22optical%20clock%22%20%22frequency%20ratio%22%20drift%20uncertainty&size=10", "label": "optical clock ratio drift exact query", "reason": "v15_bound_query"},
    ],
    "T52": [
        {"url": "https://zenodo.org/api/records/?q=%22atom%20interferometer%22%20acceleration%20sensitivity%20noise%20baseline&size=10", "label": "atom interferometer sensitivity exact query", "reason": "v15_bound_query"},
    ],
    "T54": [
        {"url": "https://zenodo.org/api/records/?q=%22FMO%22%20coherence%20lifetime%20%222D%20spectroscopy%22&size=10", "label": "FMO coherence 2D spectroscopy exact query", "reason": "v15_bio_query"},
        {"url": "https://zenodo.org/api/records/?q=%22photosynthetic%20complex%22%20coherence%20dephasing%20temperature&size=10", "label": "photosynthetic complex dephasing exact query", "reason": "v15_bio_query"},
    ],
    "T57": [
        {"url": "https://www.hepdata.net/search/?q=%22cosmic%20ray%22%20%22cross%20section%22%20energy&format=json", "label": "HEPData cosmic-ray cross-section exact search", "reason": "v15_exact_hepdata"},
    ],
    "T59": [
        {"url": "https://www.hepdata.net/download/table/ins1305430/Table7/csv", "label": "HEPData Drell-Yan Table7 CSV", "reason": "v15_exact_hepdata_csv"},
        {"url": "https://www.hepdata.net/search/?q=%22Drell%20Yan%22%20%22mass%20spectrum%22%20observed%20expected&format=json", "label": "HEPData Drell-Yan exact search", "reason": "v15_exact_hepdata"},
        {"url": "https://www.hepdata.net/search/?q=%22missing%20transverse%20momentum%22%20observed%20expected%20limit%20GeV&format=json", "label": "HEPData MET observed expected exact search", "reason": "v15_exact_hepdata"},
        {"url": "https://www.hepdata.net/search/?q=%22di-Higgs%22%20observed%20expected%20limit&format=json", "label": "HEPData di-Higgs limit exact search", "reason": "v15_exact_hepdata"},
    ],
}


def _manifest_sources_v15(test_id: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    base = Path(__file__).resolve().parents[1] / "data"
    for fn in EXACT_MANIFEST_FILES_V15:
        p = base / fn
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if (row.get("test_id") or "").strip() != test_id:
                        continue
                    url = (row.get("source_url") or row.get("search_url") or row.get("record_or_search_url") or row.get("url") or "").strip()
                    if not url:
                        continue
                    out.append({
                        "url": url,
                        "label": (row.get("label") or row.get("subtest_id") or fn),
                        "reason": "v15_exact_manifest:" + fn,
                    })
        except Exception:
            pass
    out.extend(EXTRA_EXACT_SOURCE_URLS_V15.get(test_id, []))
    return _dedupe_link_dicts(out)


def additional_seed_sources_v11(test_id: str) -> List[Dict[str, Any]]:
    """v15 override: exact manifests first, then the v14 discovery seeds."""
    exact = _manifest_sources_v15(test_id)
    rest = _additional_seed_sources_v14(test_id)
    return _dedupe_link_dicts(exact + rest)


def _metadata_text_from_record_v15(obj: Any) -> str:
    pieces: List[str] = []
    def add(x: Any, limit: int = 6000):
        if x is None:
            return
        try:
            if isinstance(x, (dict, list)):
                pieces.append(json.dumps(x, ensure_ascii=False)[:limit])
            else:
                pieces.append(str(x)[:limit])
        except Exception:
            pieces.append(str(x)[:limit])
    if isinstance(obj, dict):
        md = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
        attrs = obj.get("attributes") if isinstance(obj.get("attributes"), dict) else {}
        for d in [obj, md, attrs]:
            for k in DOMAIN_POSITIVE_FIELDS_V15:
                if k in d:
                    add(d.get(k))
        # Zenodo search hit records often put title/description in metadata; HEPData often in title/abstract.
        for k in ["title", "abstract", "abstracts", "collaboration", "experiment", "analyses", "data_abstract"]:
            if k in obj:
                add(obj.get(k))
    else:
        add(obj, 8000)
    return "\n".join(pieces)


def _record_is_relevant_v14(test_id: str, obj: Any, url: str = "") -> bool:
    """v15 override: score repository records from record metadata, not the search URL query."""
    text = _metadata_text_from_record_v15(obj)
    url_l = str(url or "").lower()
    # Exact direct downloads/manifest URLs may be neutral in metadata.
    if re.search(r"/download/table/|/files/.+/content$|/api/records/\d+/?$|osf\.io/drwcq|pure\.mpg\.de", url_l):
        # still reject obvious false positives in metadata.
        if FALSE_POSITIVE_SOURCE_RE_V14.search(text):
            return False
        # For exact manifest/record URLs keep them only if there is at least a weak domain hit in metadata or exact source URL.
        return bool(_matched_domain_terms_v14(test_id, text + " " + url_l)) or "exact" in url_l or "download/table" in url_l or "drwcq" in url_l
    if FALSE_POSITIVE_SOURCE_RE_V14.search(text):
        return False
    hits = _matched_domain_terms_v14(test_id, text)
    if test_id in {"T57", "T59"} and re.search(r"hepdata|atlas|cms|lhc|drell|higgs|met|cross\s*section|cosmic", text, re.I):
        return True
    return len(hits) > 0


def _source_is_exact_or_curated_v15(src: Dict[str, Any]) -> bool:
    txt = (str(src.get("reason", "")) + " " + str(src.get("label", "")) + " " + str(src.get("url", ""))).lower()
    return any(x in txt for x in ["v15_exact", "exact_manifest", "manifest_records", "curated", "osf_api", "itpa", "hepdata_csv"])


def _download_for_scan_v12(src: Dict[str, Any], cache: Path, args: Any) -> Tuple[bytes, Dict[str, Any]]:
    """v15 override: file-size/relevance guard; avoid multi-GB broad false-positive downloads."""
    url = str(src.get("url") or "")
    if url.startswith("figshare_search://"):
        q = urllib.parse.unquote(url.split("://", 1)[1])
        return _figshare_post_search_bytes_v12(q)
    exact = _source_is_exact_or_curated_v15(src)
    # HEAD before download; skip huge broad artifacts unless exact/curated.
    try:
        head = head_metadata(url, cache_level(cache / "files", "metadata"), timeout=getattr(args, "timeout", 45), force=getattr(args, "force", False))
        clen = head.get("content_length")
        if clen is not None:
            # Broad archives above 100 MB have been mostly false positives; exact sources get a higher cap.
            broad_cap = int(getattr(args, "broad_max_bytes", 100_000_000) or 100_000_000)
            exact_cap = int(getattr(args, "exact_max_bytes", 600_000_000) or 600_000_000)
            cap = exact_cap if exact else broad_cap
            if clen > cap:
                meta = dict(head)
                meta.update({"ok": False, "skipped": True, "skip_reason": f"v15_content_length>{cap}", "url": url, "exact_or_curated": exact})
                return b"", meta
    except Exception:
        pass
    max_bytes = int(getattr(args, "max_bytes", 80_000_000) or 80_000_000)
    if exact:
        max_bytes = max(max_bytes, 600_000_000)
    return guarded_download_bytes(url, cache / "files", timeout=getattr(args, "timeout", 45), force=getattr(args, "force", False), max_bytes=max_bytes, manifest_approved=exact)


def _augment_status_v15(test_id: str, scan: Dict[str, Any]) -> Dict[str, Any]:
    """Add explicit unavailability/status classifications after the v14 scan."""
    status = scan.get("automated_readiness_status")
    schema = scan.get("schema_artifacts") or []
    qual = scan.get("qualifying_table_count") or 0
    candidates = scan.get("candidate_table_count") or 0
    records = scan.get("source_records_sample") or []
    # ITPA schema-only condition: schema/source paper exists but no public data table.
    if test_id in {"T28", "T30"} and qual == 0:
        txt = json.dumps(records[:60] + schema[:10], ensure_ascii=False).lower()
        if "drwcq" in txt or "itpa" in txt or "db5" in txt or "h-mode" in txt or schema:
            scan["automated_readiness_status"] = "schema_or_publication_found_data_table_not_public"
            scan["public_unavailability_status_v15"] = {
                "status": "schema_found_data_file_not_public_or_not_discovered",
                "meaning": "Public sources expose the ITPA/H-mode paper, schema or OSF page, but no machine-readable DB table passed the contract.",
                "next_step": "search alternative public H-mode tables by schema aliases or obtain official DB file if public access exists",
            }
    # No primary table after exact-source search.
    if candidates == 0 and qual == 0 and status in {"sources_scanned_no_physical_candidate_tables", "no_sources_scanned"}:
        scan.setdefault("public_unavailability_status_v15", {
            "status": "no_public_primary_physical_table_found_after_exact_source_search",
            "meaning": "Exact manifests and targeted public repositories were scanned but no physical table met even candidate gates.",
        })
    # For metrology, force bound-only labels.
    if test_id in {"T50", "T51", "T52"}:
        scan["automated_readiness_status"] = "upper_limit_only_" + str(scan.get("automated_readiness_status", "data_limited"))
        scan["bound_only_summary_v15"] = {
            "upper_limit_only": True,
            "confirmation_forbidden": True,
            "qualifying_bound_tables": qual,
            "status": "bound_ready" if qual else "bound_data_limited",
        }
    return scan


def automated_source_scan(test_id: str, args: Any, seed_sources: Sequence[Dict[str, Any]], max_sources: int = 25, max_depth: int = 2) -> Dict[str, Any]:
    """v15 override: v14 scan + exact manifest seeds + status augmentation."""
    extra = additional_seed_sources_v11(test_id)
    merged = _dedupe_link_dicts(list(seed_sources) + extra)
    scan = _automated_source_scan_v14(test_id, args, merged, max_sources=max_sources, max_depth=max_depth)
    scan["version"] = "v15_exact_sources_and_public_unavailability"
    scan["exact_manifest_sources_v15"] = _manifest_sources_v15(test_id)[:50]
    scan["source_quality_ladder_v15"] = [
        "exact public machine-readable table URLs/manifests",
        "curated source-data/supplement links",
        "repository record files only after record-metadata relevance passes",
        "arXiv/source packages",
        "publisher supplements",
        "broad repository search last and size-limited",
    ]
    scan["archive_size_guard_v15"] = {
        "broad_archive_default_cap_bytes": 100_000_000,
        "exact_or_curated_cap_bytes": 600_000_000,
        "reason": "Avoid multi-GB false-positive archives from broad repository searches while still allowing exact curated artifacts.",
    }
    scan["strict_record_relevance_v15"] = "Repository records are scored from title/description/keywords/subjects metadata, not from the search URL text. False-positive domains are rejected before following files."
    scan = _augment_status_v15(test_id, scan)
    return scan


def augment_result_with_autodiscovery(test_id: str, result: Dict[str, Any], args: Any, extra_sources: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    seeds = source_urls_from_result(test_id, result)
    if extra_sources:
        seeds.extend(extra_sources)
    # Always inject exact manifests for data-limited targets.
    seeds.extend(_manifest_sources_v15(test_id))
    if not seeds and test_id in {"T57", "T59"}:
        seeds.append({"url": "https://www.hepdata.net/search/?q=ATLAS%20CMS%20Drell-Yan%20Higgs%20MET&format=json", "label": "HEPData broad exact-table search", "reason": "hepdata_connector"})
    scan = automated_source_scan(test_id, args, seeds, max_sources=96 if test_id in {"T26", "T27", "T28", "T29", "T30", "T44", "T45", "T47", "T50", "T51", "T52", "T54", "T57", "T59"} else 45, max_depth=3)
    scan["model_diagnostics"] = simple_model_diagnostics(test_id, scan.get("qualifying_tables_sample") or [])
    result["automated_discovery_v15"] = scan
    result["automated_discovery_v14"] = scan
    result["automated_discovery_v13"] = scan
    result["automated_discovery_v12"] = scan
    result["automated_discovery_v11"] = scan
    result["automated_discovery_v10"] = scan
    result["quality_patch_version"] = str(result.get("quality_patch_version", "")) + "+v15_exact_source_resolution"
    result["automated_no_manual_steps_policy"] = "All v15 discovery artifacts are downloaded or extracted automatically from public URLs. Exact manifests contain only URLs, never manual numeric rows."
    return result


# ---------------------------------------------------------------------------
# v16 exact-source improvements: URL-only manifests for all remaining quality
# sections.  This intentionally keeps metadata as metadata; it only adds more
# focused public routes for the scanner to try before broad search.
# ---------------------------------------------------------------------------
_additional_seed_sources_v15_ref = additional_seed_sources_v11


def _v16_manifest_seed_file_rows(test_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    files = [
        "fusion_exact_source_manifest.csv",
        "electronics_exact_source_manifest.csv",
        "metrology_bound_manifest.csv",
        "biology_coherence_source_manifest.csv",
        "exact_hepdata_manifest.csv",
        "material_microstructure_source_manifest.csv",
        "koide_mass_source_manifest.csv",
    ]
    for fn in files:
        p = DATA_DIR / fn
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    if str(r.get("test_id", "")).upper() == test_id.upper() and r.get("url"):
                        out.append({
                            "url": r.get("url", ""),
                            "label": r.get("label", f"v16 source {fn}"),
                            "reason": r.get("reason", f"v16_exact_manifest:{fn}"),
                            "tier": r.get("tier", "metadata_only"),
                            "subtest_id": r.get("subtest_id", ""),
                            "required_columns": r.get("required_columns", ""),
                        })
        except Exception:
            continue
    return out


def additional_seed_sources_v11(test_id: str) -> List[Dict[str, Any]]:
    base = []
    try:
        base = list(_additional_seed_sources_v15_ref(test_id))
    except Exception:
        base = []
    extra = _v16_manifest_seed_file_rows(test_id)
    # small hard-coded exact routes for parser-specific tests
    if test_id.upper() in {"T28", "T30"}:
        extra.extend([
            {"url": "https://osf.io/drwcq/", "label": "OSF International Global H-mode Confinement Database landing page", "reason": "v16_itpa_osf_landing", "tier": "schema_or_data_discovery"},
            {"url": "https://zenodo.org/api/records/?q=%22H98%22+%22energy+confinement+time%22+tokamak+%22q95%22&size=10", "label": "H98 tau_E q95 public-table query", "reason": "v16_hmode_alias_query", "tier": "metadata_only"},
        ])
    return _dedupe_link_dicts(base + extra)


# ---------------------------------------------------------------------------
# v18 source hygiene for positive-path runs
# Avoid broad HTML boilerplate link explosion (e.g. arXiv search -> GitHub UI ->
# unrelated GitHub marketing pages). Exact source manifests remain allowed.
# ---------------------------------------------------------------------------
_HTML_BOILERPLATE_HOSTS_V18 = {
    "github.com", "github.githubassets.com", "avatars.githubusercontent.com", "github.blog",
    "docs.github.com", "skills.github.com", "support.github.com", "github.community",
}
_HTML_BOILERPLATE_PATH_PATTERNS_V18 = re.compile(
    r"/(features|resources|customer-stories|pricing|enterprise|marketplace|topics|trending|collections|security|solutions|blog|changelog|manifest\.json|opensearch\.xml)(/|$)",
    re.I,
)

# Wrap discover_data_links behavior at the record level by filtering extracted_links
# after automated_source_scan builds them. This leaves exact manifest URLs intact.
_v18_auto_scan_ref = automated_source_scan

def _v18_filter_extracted_links(links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = []
    for x in links or []:
        url = str(x.get("url") or "")
        try:
            pr = urlparse(url)
            host = pr.netloc.lower()
            path = pr.path or "/"
        except Exception:
            kept.append(x); continue
        reason = str(x.get("reason") or "")
        # Keep raw/code/data-like files; drop generic UI/marketing HTML.
        is_file = bool(re.search(r"\.(csv|tsv|xlsx?|json|yaml|yml|h5|hdf5|npz|zip|tar|gz|tgz|fits|dat|txt|pdf)(\?|$)", path, re.I))
        if (host in _HTML_BOILERPLATE_HOSTS_V18 or host.endswith("github.com")) and not is_file:
            if _HTML_BOILERPLATE_PATH_PATTERNS_V18.search(path) or reason in {"html_href_data_like", "embedded_data_like_url"}:
                continue
        kept.append(x)
    return kept

def automated_source_scan(test_id, args, seed_sources, max_sources=45, max_depth=2):
    res = _v18_auto_scan_ref(test_id, args, seed_sources, max_sources=max_sources, max_depth=max_depth)
    try:
        for rec in res.get("source_records_sample", []) or []:
            rec["extracted_links_before_v18_filter_count"] = len(rec.get("extracted_links") or [])
            rec["extracted_links"] = _v18_filter_extracted_links(rec.get("extracted_links") or [])
            rec["extracted_links_after_v18_filter_count"] = len(rec.get("extracted_links") or [])
        res["html_boilerplate_link_filter_v18"] = {
            "enabled": True,
            "policy": "Drop generic GitHub/arXiv-search UI/marketing links unless they are direct data/code/archive/document files.",
        }
    except Exception:
        pass
    return res


# ---------------------------------------------------------------------------
# v19 positive-focused source discipline
# Fusion scientific mode now stops after exact/curated routes instead of spending
# the whole queue on broad UI links.  Discovery candidates can still be collected
# separately, but the scientific result is kept clean and fast.
# ---------------------------------------------------------------------------
_v19_auto_scan_ref = automated_source_scan

_V19_STRICT_BOILERPLATE_HOSTS = {
    "github.com", "github.githubassets.com", "avatars.githubusercontent.com", "github.blog",
    "docs.github.com", "skills.github.com", "support.github.com", "github.community",
    "github-cloud.s3.amazonaws.com", "user-images.githubusercontent.com",
}
_V19_ALLOWED_DATA_FILE_RE = re.compile(r"\.(csv|tsv|xlsx?|json|yaml|yml|h5|hdf5|npz|zip|tar|gz|tgz|fits|dat|txt|pdf|tex)(\?|$)", re.I)
_V19_ALLOWED_GITHUB_DATA_RE = re.compile(r"/(raw|download|releases/download|archive/refs)/", re.I)


def _v19_is_data_link(url: str) -> bool:
    try:
        pr = urlparse(url)
        host = pr.netloc.lower()
        path = pr.path or "/"
    except Exception:
        return True
    if _V19_ALLOWED_DATA_FILE_RE.search(path) or _V19_ALLOWED_GITHUB_DATA_RE.search(path):
        return True
    if host in _V19_STRICT_BOILERPLATE_HOSTS or host.endswith("githubusercontent.com"):
        return False
    # Keep repository APIs/known science repos, but reject generic social/marketing hosts.
    if host.endswith("github.com"):
        return False
    return True


def _v19_filter_links_strict(links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for x in links or []:
        u = str(x.get("url") or "")
        if not u:
            continue
        if _v19_is_data_link(u):
            out.append(x)
    return out


def automated_source_scan(test_id, args, seed_sources, max_sources=45, max_depth=2):
    # Positive-focused discipline: exact/curated scientific pass for fusion;
    # broad discovery belongs in a candidate-manifest updater, not the main score.
    if str(test_id).upper() in {"T26", "T27", "T28", "T29", "T30"}:
        max_sources = min(int(max_sources or 45), 32)
        max_depth = min(int(max_depth or 2), 2)
    res = _v19_auto_scan_ref(test_id, args, seed_sources, max_sources=max_sources, max_depth=max_depth)
    try:
        dropped = 0
        for rec in res.get("source_records_sample", []) or []:
            before = len(rec.get("extracted_links") or [])
            rec["extracted_links"] = _v19_filter_links_strict(rec.get("extracted_links") or [])
            after = len(rec.get("extracted_links") or [])
            rec["extracted_links_after_v19_filter_count"] = after
            rec["extracted_links_before_v19_filter_count"] = before
            dropped += max(0, before - after)
        res["html_boilerplate_link_filter_v19"] = {
            "enabled": True,
            "dropped_links_in_sample": dropped,
            "policy": "Drop generic GitHub/arXiv/UI/social/asset links unless they are direct data/code/archive/document files.",
        }
        if str(test_id).upper() in {"T26", "T27", "T28", "T29", "T30"}:
            res["fusion_scientific_mode_v19"] = {
                "enabled": True,
                "max_sources_cap": max_sources,
                "max_depth_cap": max_depth,
                "mode": "exact_curated_sources_first; broad discovery should be exported separately as candidate_manifest_updates.json",
                "reason": "Fusion public primary event-level tables are scarce; avoid treating broad crawler exhaustion as a scientific signal.",
            }
            if res.get("autodiscovery_stop_reason") == "autodiscovery_max_sources_reached":
                res["automated_readiness_status"] = res.get("automated_readiness_status") or "exact_sources_scanned_broad_discovery_remaining"
    except Exception:
        pass
    return res


# ---------------------------------------------------------------------------
# v20 fusion-positive discovery and parser refinements
# Adds a conservative unit-line PDF extractor for fusion PDFs, stricter exact-only
# mode controls, and additional public-data seeds. Secondary extracted rows are
# diagnostic only and cannot confirm/falsify.
# ---------------------------------------------------------------------------
_v20_auto_scan_ref = automated_source_scan

_FUSION_UNIT_LINE_RE_V20 = re.compile(
    r"(?i)(ELM|pedestal|RMP|H98|tau[_\s-]*E|q95|W7-X|tokamak|DIII-D|JET|ASDEX|AUG|ITER|Wped|Pped|dW|energy loss|frequency).{0,120}?(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(MJ|kJ|J|MW|kPa|Pa|ms|s|Hz|kHz|m\^-?3|10\^19|%)"
)


def _fusion_pdf_unit_line_frames_v20(data: bytes, source_url: str, max_pages: int = 14) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return frames
    rows: List[Dict[str, Any]] = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for pageno, page in enumerate(pdf.pages[:max_pages], start=1):
                try:
                    text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                except Exception:
                    text = ""
                if not re.search(r"(?i)ELM|pedestal|RMP|tokamak|H-mode|W7-X|DIII-D|JET|ASDEX|ITER", text):
                    continue
                for line in text.splitlines():
                    if not re.search(r"(?i)ELM|pedestal|RMP|H98|tau|q95|Wped|Pped|dW|energy|frequency|DIII-D|JET|ASDEX|ITER", line):
                        continue
                    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line)
                    if not nums:
                        continue
                    units = re.findall(r"(?i)\b(MJ|kJ|J|MW|kPa|Pa|ms|s|Hz|kHz|m\^-?3|10\^19|%)\b", line)
                    if not units:
                        continue
                    rows.append({
                        "page": pageno,
                        "line_text": line.strip()[:500],
                        "numeric_values": ";".join(nums[:12]),
                        "units_found": ";".join(units[:12]),
                        "source_url": source_url,
                    })
        if rows:
            df = pd.DataFrame(rows)
            df.attrs["source_url"] = source_url
            df.attrs["evidence_tier"] = "secondary_auto_pdf_text_table"
            df.attrs["pdf_extractor"] = "v20_fusion_unit_line_extractor"
            df.attrs["confirmation_allowed"] = False
            df.attrs["falsification_allowed"] = False
            frames.append(df)
    except Exception:
        pass
    return frames

# Wrap artifact extraction so fusion PDFs can yield secondary unit-line diagnostics.
try:
    _v20_extract_frames_ref = extract_frames_from_artifact  # type: ignore[name-defined]
    def extract_frames_from_artifact(data: bytes, url: str, meta: Dict[str, Any], cache_dir: Path) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:  # type: ignore[override]
        frames, diag = _v20_extract_frames_ref(data, url, meta, cache_dir)
        ctype = str((meta or {}).get("content_type") or "").lower()
        if ("pdf" in ctype or str(url).lower().endswith(".pdf")) and re.search(r"(?i)ELM|pedestal|RMP|H-mode|tokamak|W7-X|DIII-D|JET|ITER|ASDEX", str(url)):
            extra = _fusion_pdf_unit_line_frames_v20(data, url)
            if extra:
                frames.extend(extra)
            diag["v20_fusion_unit_line_frames"] = len(extra)
            diag.setdefault("extractors_tried", []).append("v20_fusion_unit_line_extractor")
        return frames, diag
except Exception:
    pass

# More exact public-data seeds for fusion and positive-path tests. These are URLs only.
_v20_additional_seed_ref = additional_seed_sources_v11

def additional_seed_sources_v11(test_id: str) -> List[Dict[str, Any]]:  # type: ignore[override]
    base = []
    try:
        base = list(_v20_additional_seed_ref(test_id))
    except Exception:
        base = []
    tid = str(test_id).upper()
    extra: List[Dict[str, Any]] = []
    if tid in {"T26", "T27"}:
        extra.extend([
            {"url": "https://zenodo.org/api/records/?q=%22ELM%20energy%20loss%22%20%22DIII-D%22%20tokamak&size=10", "label": "v20 DIII-D ELM energy loss exact query", "reason": "v20_fusion_exact_query", "tier": "metadata_only"},
            {"url": "https://zenodo.org/api/records/?q=%22ELM%20frequency%22%20%22RMP%22%20%22DIII-D%22&size=10", "label": "v20 DIII-D RMP ELM frequency exact query", "reason": "v20_fusion_exact_query", "tier": "metadata_only"},
            {"url": "https://zenodo.org/api/records/?q=%22pedestal%20pressure%22%20%22ELM%22%20tokamak&size=10", "label": "v20 pedestal pressure ELM exact query", "reason": "v20_fusion_exact_query", "tier": "metadata_only"},
        ])
    if tid in {"T28", "T30"}:
        extra.extend([
            {"url": "https://zenodo.org/api/records/?q=%22H98%22%20%22tau_E%22%20tokamak%20database&size=10", "label": "v20 H98 tau_E tokamak table query", "reason": "v20_hmode_alias_query", "tier": "metadata_only"},
            {"url": "https://zenodo.org/api/records/?q=%22energy%20confinement%20time%22%20%22q95%22%20elongation%20tokamak&size=10", "label": "v20 confinement q95 elongation table query", "reason": "v20_hmode_alias_query", "tier": "metadata_only"},
        ])
    if tid == "T29":
        extra.extend([
            {"url": "https://zenodo.org/api/records/?q=%22W7-X%22%20profile%20transport%20temperature%20density&size=10", "label": "v20 W7-X profile transport query", "reason": "v20_w7x_profile_query", "tier": "metadata_only"},
            {"url": "https://zenodo.org/api/records/?q=tokamak%20edge%20profile%20transport%20temperature%20density&size=10", "label": "v20 tokamak edge profile query", "reason": "v20_tokamak_profile_query", "tier": "metadata_only"},
        ])
    return _dedupe_link_dicts(base + extra)


def automated_source_scan(test_id, args, seed_sources, max_sources=45, max_depth=2):  # type: ignore[override]
    tid = str(test_id).upper()
    # Exact mode still keeps fusion alive but keeps it from drowning in generic HTML.
    if tid in {"T26", "T27", "T28", "T29", "T30"}:
        max_sources = min(int(max_sources or 45), 36)
        max_depth = min(int(max_depth or 2), 2)
    res = _v20_auto_scan_ref(test_id, args, seed_sources, max_sources=max_sources, max_depth=max_depth)
    try:
        if tid in {"T26", "T27", "T28", "T29", "T30"}:
            # Build a compact candidate-manifest suggestion object instead of expanding queue forever.
            res["fusion_candidate_manifest_updates_v20"] = {
                "status": "export_suggested_sources_not_scored",
                "suggested_file": "candidate_manifest_updates.json",
                "reason": "broad fusion search remains useful but should not dominate scientific scoring",
                "remaining_queue_count": res.get("autodiscovery_queue_remaining"),
                "secondary_unit_line_extractor_active": True,
                "exact_mode_caps": {"max_sources": max_sources, "max_depth": max_depth},
            }
            res["fusion_scientific_mode_v20"] = {
                "enabled": True,
                "mode": "exact_curated_plus_secondary_pdf_diagnostics",
                "do_not_give_up": "Fusion exact sources and secondary diagnostics remain active; broad discovery is redirected to manifest updates.",
                "primary_confirmation_requires": "machine-readable event/profile table passing all contract groups",
            }
    except Exception:
        pass
    return res


# ---------------------------------------------------------------------------
# v21 exact-domain parser hooks for EL/electronics and fusion secondary diagnostics.
# These are conservative text/table extractors for exact public files only. They
# produce diagnostic candidate rows, but do not relax primary evidence gates.
# ---------------------------------------------------------------------------

def _text_from_pdf_or_bytes_v21(data: bytes, url: str, max_pages: int = 8) -> str:
    c = str(url).lower()
    if c.endswith('.pdf') or data[:5] == b'%PDF-':
        try:
            import pdfplumber  # type: ignore
            chunks = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages[:max_pages]:
                    try:
                        chunks.append(page.extract_text(x_tolerance=1, y_tolerance=3) or '')
                    except Exception:
                        pass
            return '\n'.join(chunks)
        except Exception:
            return ''
    try:
        return data.decode('utf-8', errors='replace')
    except Exception:
        return ''


def _electronics_line_frames_v21(data: bytes, url: str, meta: Dict[str, Any]) -> List[pd.DataFrame]:
    text = _text_from_pdf_or_bytes_v21(data, url)
    if not text:
        return []
    rows = []
    lower_url = str(url).lower()
    for line in text.splitlines():
        l = line.strip()
        if len(l) < 8 or len(l) > 800:
            continue
        low = l.lower()
        # T44 / NAND rows
        if re.search(r'\b(3d\s*nand|v-?nand|nand|flash)\b', low) and re.search(r'\b(layer|layers|gb|tb|die|cell|tlc|qlc|bits?)\b', low):
            nums = re.findall(r'\d+(?:\.\d+)?', l)
            if nums:
                rows.append({'parser_family': 'T44_NAND_exact_text', 'line_text': l[:500], 'numbers': ';'.join(nums[:12]), 'source_url': url})
        # T45 / optical interconnect rows
        if re.search(r'\b(optical|photon|photonic|interconnect|link)\b', low) and re.search(r'(pj\s*/\s*bit|fj\s*/\s*bit|gbps|tbps|bandwidth|reach|mm|cm)', low):
            nums = re.findall(r'\d+(?:\.\d+)?', l)
            if nums:
                rows.append({'parser_family': 'T45_optical_interconnect_exact_text', 'line_text': l[:500], 'numbers': ';'.join(nums[:12]), 'source_url': url})
        # T47 / neuromorphic rows
        if re.search(r'\b(loihi|truenorth|spinnaker|brainscales|neuromorphic)\b', low) and re.search(r'\b(energy|power|joule|pj|nj|uj|accuracy|benchmark|spike|inference|core|neuron)\b', low):
            nums = re.findall(r'\d+(?:\.\d+)?', l)
            if nums:
                rows.append({'parser_family': 'T47_neuromorphic_exact_text', 'line_text': l[:500], 'numbers': ';'.join(nums[:12]), 'source_url': url})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df.attrs['source_url'] = url
    df.attrs['evidence_tier'] = 'secondary_exact_spec_text_table'
    df.attrs['confirmation_allowed'] = False
    df.attrs['falsification_allowed'] = False
    df.attrs['parser'] = 'v21_electronics_exact_text_parser'
    return [df]


def _fusion_pdf_unit_line_frames_v21(data: bytes, source_url: str, max_pages: int = 20) -> List[pd.DataFrame]:
    text = _text_from_pdf_or_bytes_v21(data, source_url, max_pages=max_pages)
    if not text:
        return []
    rows = []
    for pageno, chunk in enumerate(text.split('\f') if '\f' in text else [text], start=1):
        if not re.search(r'(?i)ELM|pedestal|RMP|tokamak|H-mode|W7-X|DIII-D|JET|ASDEX|ITER|AUG|H98|q95|tau', chunk):
            continue
        for line in chunk.splitlines():
            if not re.search(r'(?i)ELM|pedestal|RMP|H98|tau|q95|Wped|Pped|dW|energy|frequency|DIII-D|JET|ASDEX|ITER|W7-X', line):
                continue
            nums = re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', line)
            units = re.findall(r'(?i)\b(MJ|kJ|J|MW|kPa|Pa|ms|Hz|kHz|m\^-?3|10\^19|%)\b', line)
            if nums and units:
                rows.append({'page_or_chunk': pageno, 'line_text': line.strip()[:700], 'numeric_values': ';'.join(nums[:16]), 'units_found': ';'.join(units[:16]), 'source_url': source_url})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df.attrs['source_url'] = source_url
    df.attrs['evidence_tier'] = 'secondary_auto_pdf_text_table'
    df.attrs['confirmation_allowed'] = False
    df.attrs['falsification_allowed'] = False
    df.attrs['pdf_extractor'] = 'v21_fusion_unit_line_extractor'
    return [df]


try:
    _v21_extract_frames_ref = extract_frames_from_artifact  # type: ignore[name-defined]
    def extract_frames_from_artifact(data: bytes, url: str, meta: Dict[str, Any], cache_dir: Path) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:  # type: ignore[override]
        frames, diag = _v21_extract_frames_ref(data, url, meta, cache_dir)
        ctype = str((meta or {}).get('content_type') or '').lower()
        url_s = str(url)
        # Electronics exact text extraction for EL branch.
        if re.search(r'(?i)nand|wikichip|techinsights|irds|optical|interconnect|loihi|truenorth|spinnaker|neuromorphic', url_s):
            try:
                extra_el = _electronics_line_frames_v21(data, url_s, meta or {})
                if extra_el:
                    frames.extend(extra_el)
                diag['v21_electronics_exact_text_frames'] = len(extra_el)
                diag.setdefault('extractors_tried', []).append('v21_electronics_exact_text_parser')
            except Exception as e:
                diag['v21_electronics_exact_text_error'] = repr(e)
        # Fusion diagnostic extraction, broader and page/depth aware.
        if ('pdf' in ctype or url_s.lower().endswith('.pdf')) and re.search(r'(?i)ELM|pedestal|RMP|H-mode|tokamak|W7-X|DIII-D|JET|ITER|ASDEX|AUG|Loarte|Liang', url_s):
            try:
                extra_f = _fusion_pdf_unit_line_frames_v21(data, url_s)
                if extra_f:
                    frames.extend(extra_f)
                diag['v21_fusion_unit_line_frames'] = len(extra_f)
                diag.setdefault('extractors_tried', []).append('v21_fusion_unit_line_extractor')
            except Exception as e:
                diag['v21_fusion_unit_line_error'] = repr(e)
        return frames, diag
except Exception:
    pass

# v21: more exact source seeds for the EL branch and fusion diagnostics.
_v21_additional_seed_ref = additional_seed_sources_v11

def additional_seed_sources_v11(test_id: str) -> List[Dict[str, Any]]:  # type: ignore[override]
    try:
        base = list(_v21_additional_seed_ref(test_id))
    except Exception:
        base = []
    tid = str(test_id).upper()
    extra: List[Dict[str, Any]] = []
    if tid == 'T44':
        extra.extend([
            {'url': 'https://en.wikichip.org/wiki/3d_nand', 'label': 'v21 WikiChip 3D NAND exact page', 'reason': 'v21_el_t44_exact_parser_seed', 'tier': 'html_table_candidate'},
            {'url': 'https://en.wikichip.org/wiki/flash_memory', 'label': 'v21 WikiChip flash memory exact page', 'reason': 'v21_el_t44_exact_parser_seed', 'tier': 'html_table_candidate'},
            {'url': 'https://www.techinsights.com/blog/3d-nand-flash-memory-density-ranking', 'label': 'v21 TechInsights 3D NAND density ranking', 'reason': 'v21_el_t44_exact_parser_seed', 'tier': 'html_table_candidate'},
        ])
    if tid == 'T45':
        extra.extend([
            {'url': 'https://irds.ieee.org/editions', 'label': 'v21 IRDS editions optical/interconnect roadmap source', 'reason': 'v21_el_t45_irds_seed', 'tier': 'roadmap_pdf_discovery'},
            {'url': 'https://zenodo.org/api/records/?q=%22pJ%2Fbit%22%20%22optical%20interconnect%22%20%22bandwidth%22&size=10', 'label': 'v21 exact optical interconnect pJ/bit query', 'reason': 'v21_el_t45_exact_query', 'tier': 'metadata_only'},
        ])
    if tid == 'T47':
        extra.extend([
            {'url': 'https://zenodo.org/api/records/?q=%22Loihi%22%20%22energy%22%20%22benchmark%22&size=10', 'label': 'v21 Loihi exact benchmark query', 'reason': 'v21_el_t47_exact_query', 'tier': 'metadata_only'},
            {'url': 'https://zenodo.org/api/records/?q=%22TrueNorth%22%20%22energy%22%20%22benchmark%22&size=10', 'label': 'v21 TrueNorth exact benchmark query', 'reason': 'v21_el_t47_exact_query', 'tier': 'metadata_only'},
            {'url': 'https://zenodo.org/api/records/?q=%22SpiNNaker%22%20%22energy%22%20%22benchmark%22&size=10', 'label': 'v21 SpiNNaker exact benchmark query', 'reason': 'v21_el_t47_exact_query', 'tier': 'metadata_only'},
        ])
    if tid in {'T26', 'T27'}:
        extra.extend([
            {'url': 'https://zenodo.org/api/records/?q=%22DIII-D%22%20%22ELM%20energy%20loss%22%20%22pedestal%22&size=10', 'label': 'v21 DIII-D ELM loss pedestal exact query', 'reason': 'v21_fusion_extra_query', 'tier': 'metadata_only'},
            {'url': 'https://zenodo.org/api/records/?q=%22JET%22%20%22ELM%20energy%20loss%22%20%22pedestal%22&size=10', 'label': 'v21 JET ELM loss pedestal exact query', 'reason': 'v21_fusion_extra_query', 'tier': 'metadata_only'},
        ])
    return _dedupe_link_dicts(base + extra)


# ---------------------------------------------------------------------------
# v22 exact positive-path extractors: stronger EL/NAND, optical, neuromorphic,
# grain-size phrase mining hooks, and fusion secondary diagnostics. These do not
# relax evidence gates; they add normalized diagnostic frames/metadata only.
# ---------------------------------------------------------------------------


def _v22_text_from_bytes(data: bytes, url: str, max_pages: int = 30) -> str:
    # Multi-backend PDF text fallback, then plain text/HTML decode.
    if data[:5] == b'%PDF-' or str(url).lower().endswith('.pdf'):
        chunks: List[str] = []
        try:
            import pdfplumber  # type: ignore
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages[:max_pages]:
                    try:
                        chunks.append(page.extract_text(x_tolerance=1, y_tolerance=3) or '')
                    except Exception:
                        pass
        except Exception:
            pass
        if not ''.join(chunks).strip():
            try:
                import PyPDF2  # type: ignore
                reader = PyPDF2.PdfReader(io.BytesIO(data))
                for page in reader.pages[:max_pages]:
                    try:
                        chunks.append(page.extract_text() or '')
                    except Exception:
                        pass
            except Exception:
                pass
        return '\n'.join(chunks)
    try:
        return data.decode('utf-8', errors='replace')
    except Exception:
        return ''


def _guess_maker_v22(line: str) -> Optional[str]:
    for maker in ['Samsung','Micron','Intel','SK hynix','SK Hynix','Kioxia','Toshiba','Western Digital','SanDisk','YMTC']:
        if maker.lower() in line.lower():
            return maker
    return None


def _nand_structured_frames_v22(data: bytes, url: str) -> List[pd.DataFrame]:
    text = _v22_text_from_bytes(data, url, max_pages=8)
    if not text:
        return []
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        l = re.sub(r'\s+', ' ', line.strip())
        low = l.lower()
        if not re.search(r'3d\s*nand|v-?nand|nand|flash', low):
            continue
        if not re.search(r'layer|layers|gb|gbit|tb|tbit|die|cell|tlc|qlc|slc|mlc', low):
            continue
        layers = None
        m = re.search(r'(\d{2,3})\s*(?:-|\s)?(?:layer|layers|l\b)', low)
        if m:
            layers = int(m.group(1))
        cap = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(tb|tbit|gb|gbit)\b', low)
        if m:
            cap = float(m.group(1)) * (1000.0 if m.group(2).startswith('t') else 1.0)
        die_area = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm\s*(?:2|\^2)|mm²)', low)
        if m:
            die_area = float(m.group(1))
        bits = None
        if 'qlc' in low: bits = 4
        elif 'tlc' in low: bits = 3
        elif 'mlc' in low: bits = 2
        elif 'slc' in low: bits = 1
        if layers or cap or die_area or bits:
            rows.append({
                'manufacturer': _guess_maker_v22(l),
                'generation_or_product': l[:120],
                'layers': layers,
                'die_capacity_Gb': cap,
                'die_area_mm2': die_area,
                'bits_per_cell': bits,
                'density_Gb_per_mm2': (cap / die_area) if cap and die_area else None,
                'line_text': l[:600],
                'source_url': url,
            })
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df.attrs['source_url'] = url
    df.attrs['evidence_tier'] = 'secondary_exact_spec_text_table'
    df.attrs['parser'] = 'v22_nand_structured_parser'
    df.attrs['v22_nand_structured_rows'] = len(df)
    df.attrs['confirmation_allowed'] = False
    df.attrs['falsification_allowed'] = False
    return [df]


def _optical_structured_frames_v22(data: bytes, url: str) -> List[pd.DataFrame]:
    text = _v22_text_from_bytes(data, url, max_pages=20)
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        l = re.sub(r'\s+', ' ', line.strip())
        low = l.lower()
        if not re.search(r'optical|photonic|photon|interconnect|link|serdes|i/o', low):
            continue
        if not re.search(r'pj\s*/\s*bit|fj\s*/\s*bit|gb/s|gbps|tb/s|tbps|bandwidth|reach|mm|cm', low):
            continue
        epb = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(fJ|pJ)\s*/\s*bit', l, flags=re.I)
        if m:
            epb = float(m.group(1)) * (0.001 if m.group(2).lower() == 'fj' else 1.0)
        bw = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(Gb/s|Gbps|Tb/s|Tbps)', l, flags=re.I)
        if m:
            bw = float(m.group(1)) * (1000.0 if m.group(2).lower().startswith('t') else 1.0)
        reach = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|m)\b', l, flags=re.I)
        if m:
            reach = float(m.group(1)) * ({'mm':1.0,'cm':10.0,'m':1000.0}[m.group(2).lower()])
        rows.append({'energy_pJ_per_bit': epb, 'bandwidth_Gbps': bw, 'reach_mm': reach, 'line_text': l[:600], 'source_url': url})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df.attrs['source_url'] = url
    df.attrs['evidence_tier'] = 'secondary_exact_spec_text_table'
    df.attrs['parser'] = 'v22_optical_interconnect_unit_parser'
    df.attrs['v22_optical_interconnect_rows'] = len(df)
    df.attrs['confirmation_allowed'] = False
    df.attrs['falsification_allowed'] = False
    return [df]


def _neuromorphic_structured_frames_v22(data: bytes, url: str) -> List[pd.DataFrame]:
    text = _v22_text_from_bytes(data, url, max_pages=20)
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        l = re.sub(r'\s+', ' ', line.strip())
        low = l.lower()
        if not re.search(r'loihi|truenorth|spinnaker|brainscales|neuromorphic', low):
            continue
        if not re.search(r'energy|power|pj|nj|uj|µj|spike|inference|accuracy|benchmark|core|neuron', low):
            continue
        energy = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(pJ|nJ|uJ|µJ|mJ)\b', l, flags=re.I)
        if m:
            scale = {'pj':1e-12,'nj':1e-9,'uj':1e-6,'µj':1e-6,'mj':1e-3}[m.group(2).lower()]
            energy = float(m.group(1)) * scale
        rows.append({'chip_hint': re.search(r'(Loihi\s*2?|TrueNorth|SpiNNaker|BrainScaleS)', l, flags=re.I).group(0) if re.search(r'(Loihi\s*2?|TrueNorth|SpiNNaker|BrainScaleS)', l, flags=re.I) else None, 'energy_J_hint': energy, 'line_text': l[:600], 'source_url': url})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df.attrs['source_url'] = url
    df.attrs['evidence_tier'] = 'secondary_exact_spec_text_table'
    df.attrs['parser'] = 'v22_neuromorphic_benchmark_parser'
    df.attrs['v22_neuromorphic_rows'] = len(df)
    df.attrs['confirmation_allowed'] = False
    df.attrs['falsification_allowed'] = False
    return [df]


def _fusion_unit_frames_v22(data: bytes, url: str) -> List[pd.DataFrame]:
    text = _v22_text_from_bytes(data, url, max_pages=40)
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(text.splitlines()):
        l = re.sub(r'\s+', ' ', line.strip())
        if len(l) < 10 or len(l) > 800:
            continue
        if not re.search(r'(?i)ELM|pedestal|RMP|H98|tau_E|tauE|q95|Wped|Pped|dW|energy loss|frequency|DIII-D|JET|ASDEX|AUG|ITER|W7-X', l):
            continue
        nums = re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', l)
        units = re.findall(r'(?i)\b(MJ|kJ|J|MW|kPa|Pa|ms|s|Hz|kHz|m\^-?3|10\^19|%)\b', l)
        if nums and units:
            rows.append({'line_index': idx, 'line_text': l[:700], 'numeric_values': ';'.join(nums[:16]), 'units_found': ';'.join(units[:16]), 'source_url': url})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df.attrs['source_url'] = url
    df.attrs['evidence_tier'] = 'secondary_auto_pdf_text_table'
    df.attrs['pdf_extractor'] = 'v22_fusion_unit_line_extractor'
    df.attrs['v22_fusion_unit_line_frames'] = len(df)
    df.attrs['confirmation_allowed'] = False
    df.attrs['falsification_allowed'] = False
    return [df]


try:
    _v22_extract_frames_ref = extract_frames_from_artifact  # type: ignore[name-defined]
    def extract_frames_from_artifact(data: bytes, url: str, meta: Dict[str, Any], cache_dir: Path) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:  # type: ignore[override]
        frames, diag = _v22_extract_frames_ref(data, url, meta, cache_dir)
        us = str(url)
        low = us.lower()
        ctype = str((meta or {}).get('content_type') or '').lower()
        try:
            if re.search(r'3d\s*nand|v-?nand|nand|flash|wikichip|techinsights', low):
                xs = _nand_structured_frames_v22(data, us)
                frames.extend(xs)
                diag['v22_nand_structured_rows'] = sum(int(x.attrs.get('v22_nand_structured_rows', len(x))) for x in xs)
                diag.setdefault('extractors_tried', []).append('v22_nand_structured_parser')
        except Exception as e:
            diag['v22_nand_structured_error'] = repr(e)
        try:
            if re.search(r'optical|photonic|interconnect|irds|pJ|fJ', us, flags=re.I):
                xs = _optical_structured_frames_v22(data, us)
                frames.extend(xs)
                diag['v22_optical_interconnect_rows'] = sum(int(x.attrs.get('v22_optical_interconnect_rows', len(x))) for x in xs)
                diag.setdefault('extractors_tried', []).append('v22_optical_interconnect_unit_parser')
        except Exception as e:
            diag['v22_optical_interconnect_error'] = repr(e)
        try:
            if re.search(r'loihi|truenorth|spinnaker|brainscales|neuromorphic', us, flags=re.I):
                xs = _neuromorphic_structured_frames_v22(data, us)
                frames.extend(xs)
                diag['v22_neuromorphic_rows'] = sum(int(x.attrs.get('v22_neuromorphic_rows', len(x))) for x in xs)
                diag.setdefault('extractors_tried', []).append('v22_neuromorphic_benchmark_parser')
        except Exception as e:
            diag['v22_neuromorphic_error'] = repr(e)
        try:
            if ('pdf' in ctype or low.endswith('.pdf')) and re.search(r'ELM|pedestal|RMP|H-mode|tokamak|W7-X|DIII-D|JET|ITER|ASDEX|AUG|Loarte|Liang', us, flags=re.I):
                xs = _fusion_unit_frames_v22(data, us)
                frames.extend(xs)
                diag['v22_fusion_unit_line_frames'] = sum(int(x.attrs.get('v22_fusion_unit_line_frames', len(x))) for x in xs)
                diag.setdefault('extractors_tried', []).append('v22_fusion_unit_line_extractor')
        except Exception as e:
            diag['v22_fusion_unit_line_error'] = repr(e)
        return frames, diag
except Exception:
    pass

# v22: direct-source additions for positive paths. Still URL-only; no manual rows.
_v22_additional_seed_ref = additional_seed_sources_v11

def additional_seed_sources_v11(test_id: str) -> List[Dict[str, Any]]:  # type: ignore[override]
    try:
        base = list(_v22_additional_seed_ref(test_id))
    except Exception:
        base = []
    tid = str(test_id).upper()
    extra: List[Dict[str, Any]] = []
    if tid == 'T44':
        extra.extend([
            {'url': 'https://en.wikichip.org/wiki/3d_nand', 'label': 'v22 WikiChip 3D NAND exact parser target', 'reason': 'v22_t44_exact_nand_parser', 'tier': 'html_table_candidate'},
            {'url': 'https://en.wikichip.org/wiki/list_of_flash_memory_cells', 'label': 'v22 WikiChip flash cells list', 'reason': 'v22_t44_exact_nand_parser', 'tier': 'html_table_candidate'},
            {'url': 'https://www.techinsights.com/blog/3d-nand-flash-memory-density-ranking', 'label': 'v22 TechInsights NAND density ranking', 'reason': 'v22_t44_exact_nand_parser', 'tier': 'html_table_candidate'},
        ])
    if tid == 'T45':
        extra.extend([
            {'url': 'https://zenodo.org/api/records/?q=%22fJ%2Fbit%22%20%22optical%20interconnect%22&size=10', 'label': 'v22 optical interconnect fJ/bit exact query', 'reason': 'v22_t45_optical_unit_query', 'tier': 'metadata_only'},
            {'url': 'https://zenodo.org/api/records/?q=%22pJ%2Fbit%22%20%22silicon%20photonics%22%20%22Gb%2Fs%22&size=10', 'label': 'v22 silicon photonics pJ/bit Gb/s exact query', 'reason': 'v22_t45_optical_unit_query', 'tier': 'metadata_only'},
        ])
    if tid == 'T47':
        extra.extend([
            {'url': 'https://zenodo.org/api/records/?q=%22Loihi%202%22%20%22energy%20per%20inference%22&size=10', 'label': 'v22 Loihi 2 energy inference query', 'reason': 'v22_t47_neuro_query', 'tier': 'metadata_only'},
            {'url': 'https://zenodo.org/api/records/?q=%22SpiNNaker%22%20%22energy%20per%20spike%22&size=10', 'label': 'v22 SpiNNaker energy per spike query', 'reason': 'v22_t47_neuro_query', 'tier': 'metadata_only'},
        ])
    if tid in {'T26','T27'}:
        extra.extend([
            {'url': 'https://zenodo.org/api/records/?q=%22ELM%20energy%20loss%22%20%22Wped%22%20%22Pped%22&size=10', 'label': 'v22 Wped Pped ELM exact query', 'reason': 'v22_fusion_unit_query', 'tier': 'metadata_only'},
            {'url': 'https://zenodo.org/api/records/?q=%22Type-I%20ELM%22%20%22dW%22%20%22pedestal%22&size=10', 'label': 'v22 Type-I ELM dW pedestal exact query', 'reason': 'v22_fusion_unit_query', 'tier': 'metadata_only'},
        ])
    return _dedupe_link_dicts(base + extra)


# ---------------------------------------------------------------------------
# v23 extraction/source layer: exact-domain parsers and positive-path source
# targeting for EL branch, materials/biology, and fusion diagnostics.
# ---------------------------------------------------------------------------

def _v23_text(data: bytes, url: str, max_pages: int = 50) -> str:
    try:
        return _v22_text_from_bytes(data, url, max_pages=max_pages)
    except Exception:
        try:
            return data.decode('utf-8', errors='ignore')
        except Exception:
            return ''


def _v23_float_from_text(s: str) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', str(s).replace(',', ''))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _v23_nand_structured_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    text = _v23_text(data, url, max_pages=20)
    # 1) Try true HTML tables first.
    try:
        tables = pd.read_html(io.BytesIO(data))
    except Exception:
        tables = []
    rows: List[Dict[str, Any]] = []
    for table in tables[:12]:
        try:
            df = table.copy()
            cols = [str(c).strip().lower() for c in df.columns]
            joined_cols = ' '.join(cols)
            if not re.search(r'layer|nand|flash|die|capacity|bits|cell|density', joined_cols):
                continue
            for _, r in df.iterrows():
                row_text = ' '.join(str(x) for x in r.tolist())
                low = row_text.lower()
                if not re.search(r'nand|v-?nand|flash|layer|gb|gbit|tb|die|cell', low):
                    continue
                layers = None
                m = re.search(r'(\d{2,3})\s*(?:-|\s)?layer', low)
                if m: layers = float(m.group(1))
                cap_gb = None
                m = re.search(r'(\d+(?:\.\d+)?)\s*(tb|gb|gbit|gbits?)\b', low)
                if m:
                    cap_gb = float(m.group(1)) * (1000.0 if m.group(2).startswith('t') else 1.0)
                area = None
                m = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm\s*2|mm2|mm\^2)', low)
                if m: area = float(m.group(1))
                bits = None
                m = re.search(r'\b([2345])\s*(?:bit|bpc|bits/cell|bits per cell|tlc|qlc|mlc|slc)\b', low)
                if m: bits = float(m.group(1))
                if layers or cap_gb or area:
                    rows.append({'manufacturer': None, 'year': None, 'generation_or_product': row_text[:120], 'layers': layers, 'die_capacity_Gb': cap_gb, 'die_area_mm2': area, 'bits_per_cell': bits, 'density_Gb_per_mm2': (cap_gb/area if cap_gb and area else None), 'line_text': row_text[:600], 'source_url': url})
        except Exception:
            continue
    # 2) Fallback line parser for spec pages/PDF text.
    for line in text.splitlines():
        l = re.sub(r'\s+', ' ', line.strip())
        low = l.lower()
        if not re.search(r'3d\s*nand|v-?nand|nand|flash|layer', low):
            continue
        if not re.search(r'gb|gbit|tb|die|mm|bits?\s*/?\s*cell|tlc|qlc|mlc|slc|layer', low):
            continue
        layers = None
        m = re.search(r'(\d{2,3})\s*(?:-|\s)?layer', low)
        if m: layers = float(m.group(1))
        cap_gb = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(tb|gb|gbit|gbits?)\b', low)
        if m: cap_gb = float(m.group(1)) * (1000.0 if m.group(2).startswith('t') else 1.0)
        area = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm\s*2|mm2|mm\^2)', low)
        if m: area = float(m.group(1))
        bits = None
        if 'qlc' in low: bits = 4.0
        elif 'tlc' in low: bits = 3.0
        elif 'mlc' in low: bits = 2.0
        elif 'slc' in low: bits = 1.0
        else:
            m = re.search(r'\b([2345])\s*(?:bit|bpc|bits/cell|bits per cell)\b', low)
            if m: bits = float(m.group(1))
        manufacturer = None
        mm = re.search(r'\b(Samsung|Micron|SK\s*Hynix|Kioxia|Toshiba|Western\s*Digital|Intel|YMTC)\b', l, flags=re.I)
        if mm: manufacturer = mm.group(1)
        if layers or cap_gb or area:
            rows.append({'manufacturer': manufacturer, 'year': None, 'generation_or_product': l[:120], 'layers': layers, 'die_capacity_Gb': cap_gb, 'die_area_mm2': area, 'bits_per_cell': bits, 'density_Gb_per_mm2': (cap_gb/area if cap_gb and area else None), 'line_text': l[:600], 'source_url': url})
    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=['line_text','source_url'])
        df.attrs['source_url'] = url
        df.attrs['evidence_tier'] = 'secondary_exact_spec_text_or_html_table'
        df.attrs['parser'] = 'v23_nand_exact_parser'
        df.attrs['v23_nand_rows'] = len(df)
        df.attrs['confirmation_allowed'] = False
        df.attrs['falsification_allowed'] = False
        frames.append(df)
    return frames


def _v23_optical_interconnect_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    text = _v23_text(data, url, max_pages=35)
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        l = re.sub(r'\s+', ' ', line.strip())
        low = l.lower()
        if not re.search(r'optical|photonic|silicon photonics|interconnect|link|i/o|serdes', low):
            continue
        if not re.search(r'fj\s*/\s*bit|pj\s*/\s*bit|gb/s|gbps|tb/s|tbps|reach|bandwidth|mm|cm|meter|metre', low):
            continue
        epb = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(fJ|pJ)\s*/\s*bit', l, flags=re.I)
        if m: epb = float(m.group(1)) * (0.001 if m.group(2).lower() == 'fj' else 1.0)
        bw = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(Gb/s|Gbps|Tb/s|Tbps)', l, flags=re.I)
        if m: bw = float(m.group(1)) * (1000.0 if m.group(2).lower().startswith('t') else 1.0)
        reach = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|m)\b', l, flags=re.I)
        if m: reach = float(m.group(1)) * ({'mm':1.0,'cm':10.0,'m':1000.0}[m.group(2).lower()])
        if epb is not None or bw is not None or reach is not None:
            rows.append({'energy_pJ_per_bit': epb, 'bandwidth_Gbps': bw, 'reach_mm': reach, 'optical_vs_electrical': 'optical' if re.search(r'optical|photonic', low) else None, 'line_text': l[:700], 'source_url': url})
    if not rows: return []
    df = pd.DataFrame(rows).drop_duplicates(subset=['line_text','source_url'])
    df.attrs['source_url'] = url
    df.attrs['evidence_tier'] = 'secondary_exact_spec_text_table'
    df.attrs['parser'] = 'v23_optical_interconnect_unit_parser'
    df.attrs['v23_optical_rows'] = len(df)
    df.attrs['confirmation_allowed'] = False
    df.attrs['falsification_allowed'] = False
    return [df]


def _v23_neuromorphic_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    text = _v23_text(data, url, max_pages=35)
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        l = re.sub(r'\s+', ' ', line.strip())
        low = l.lower()
        if not re.search(r'loihi|truenorth|spinnaker|brainscales|neuromorphic', low):
            continue
        if not re.search(r'energy|power|pj|nj|uj|µj|spike|inference|accuracy|benchmark|core|neuron|synapse', low):
            continue
        energy = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*(pJ|nJ|uJ|µJ|mJ)\b', l, flags=re.I)
        if m:
            energy = float(m.group(1))*({'pj':1e-12,'nj':1e-9,'uj':1e-6,'µj':1e-6,'mj':1e-3}[m.group(2).lower()])
        acc = None
        m = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:accuracy|acc\b)?', l, flags=re.I)
        if m: acc = float(m.group(1))
        chip = None
        m = re.search(r'(Loihi\s*2?|TrueNorth|SpiNNaker|BrainScaleS)', l, flags=re.I)
        if m: chip = m.group(1)
        rows.append({'chip': chip, 'energy_J_hint': energy, 'accuracy_percent_hint': acc, 'line_text': l[:700], 'source_url': url})
    if not rows: return []
    df = pd.DataFrame(rows).drop_duplicates(subset=['line_text','source_url'])
    df.attrs['source_url'] = url
    df.attrs['evidence_tier'] = 'secondary_exact_benchmark_text_table'
    df.attrs['parser'] = 'v23_neuromorphic_benchmark_parser'
    df.attrs['v23_neuro_rows'] = len(df)
    df.attrs['confirmation_allowed'] = False
    df.attrs['falsification_allowed'] = False
    return [df]


def _v23_fusion_unit_and_figure_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    text = _v23_text(data, url, max_pages=60)
    unit_rows: List[Dict[str, Any]] = []
    figure_rows: List[Dict[str, Any]] = []
    fusion_terms = r'ELM|pedestal|RMP|H98|tau_E|tauE|q95|Wped|Pped|dW|ΔW|energy loss|frequency|DIII-D|JET|ASDEX|AUG|ITER|W7-X|tokamak|H-mode'
    unit_terms = r'MJ|kJ|\bJ\b|MW|kPa|Pa|ms|\bs\b|Hz|kHz|m\^-?3|10\^19|%|MA|T\b'
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        l = re.sub(r'\s+', ' ', line.strip())
        if len(l) < 10 or len(l) > 900:
            continue
        if re.search(fusion_terms, l, flags=re.I):
            if re.search(unit_terms, l, flags=re.I) and re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', l):
                nums = re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', l)
                units = re.findall(unit_terms, l, flags=re.I)
                unit_rows.append({'line_index': idx, 'line_text': l[:750], 'numeric_values': ';'.join(nums[:20]), 'units_found': ';'.join(units[:20]), 'source_url': url})
            if re.search(r'figure|fig\.|table|axis|versus|vs\.|correlation|scal', l, flags=re.I) and re.search(r'ELM|pedestal|Wped|Pped|dW|RMP|H98|q95', l, flags=re.I):
                figure_rows.append({'line_index': idx, 'figure_candidate_text': l[:750], 'source_url': url})
    frames: List[pd.DataFrame] = []
    if unit_rows:
        df = pd.DataFrame(unit_rows).drop_duplicates(subset=['line_text','source_url'])
        df.attrs['source_url'] = url
        df.attrs['evidence_tier'] = 'secondary_auto_pdf_text_table'
        df.attrs['parser'] = 'v23_fusion_unit_line_extractor'
        df.attrs['v23_fusion_unit_rows'] = len(df)
        df.attrs['confirmation_allowed'] = False
        df.attrs['falsification_allowed'] = False
        frames.append(df)
    if figure_rows:
        df2 = pd.DataFrame(figure_rows).drop_duplicates(subset=['figure_candidate_text','source_url'])
        df2.attrs['source_url'] = url
        df2.attrs['evidence_tier'] = 'secondary_figure_page_candidate'
        df2.attrs['parser'] = 'v23_fusion_figure_candidate_detector'
        df2.attrs['v23_fusion_figure_candidate_pages'] = len(df2)
        df2.attrs['confirmation_allowed'] = False
        df2.attrs['falsification_allowed'] = False
        frames.append(df2)
    return frames


try:
    _v23_extract_ref = extract_frames_from_artifact  # type: ignore[name-defined]
    def extract_frames_from_artifact(data: bytes, url: str, meta: Dict[str, Any], cache_dir: Path) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:  # type: ignore[override]
        frames, diag = _v23_extract_ref(data, url, meta, cache_dir)
        us = str(url)
        low = us.lower()
        ctype = str((meta or {}).get('content_type') or '').lower()
        try:
            if re.search(r'3d\s*nand|v-?nand|nand|flash|wikichip|techinsights|samsung|micron|hynix|kioxia', low):
                xs = _v23_nand_structured_frames(data, us)
                frames.extend(xs)
                diag['v23_nand_rows'] = sum(int(x.attrs.get('v23_nand_rows', len(x))) for x in xs)
                diag.setdefault('extractors_tried', []).append('v23_nand_exact_parser')
        except Exception as e:
            diag['v23_nand_error'] = repr(e)
        try:
            if re.search(r'optical|photonic|interconnect|irds|silicon.?photonics|pj|fj', us, flags=re.I):
                xs = _v23_optical_interconnect_frames(data, us)
                frames.extend(xs)
                diag['v23_optical_rows'] = sum(int(x.attrs.get('v23_optical_rows', len(x))) for x in xs)
                diag.setdefault('extractors_tried', []).append('v23_optical_interconnect_unit_parser')
        except Exception as e:
            diag['v23_optical_error'] = repr(e)
        try:
            if re.search(r'loihi|truenorth|spinnaker|brainscales|neuromorphic', us, flags=re.I):
                xs = _v23_neuromorphic_frames(data, us)
                frames.extend(xs)
                diag['v23_neuro_rows'] = sum(int(x.attrs.get('v23_neuro_rows', len(x))) for x in xs)
                diag.setdefault('extractors_tried', []).append('v23_neuromorphic_benchmark_parser')
        except Exception as e:
            diag['v23_neuro_error'] = repr(e)
        try:
            if ('pdf' in ctype or low.endswith('.pdf')) and re.search(r'ELM|pedestal|RMP|H-mode|tokamak|W7-X|DIII-D|JET|ITER|ASDEX|AUG|Loarte|Liang|fusion', us, flags=re.I):
                xs = _v23_fusion_unit_and_figure_frames(data, us)
                frames.extend(xs)
                diag['v23_fusion_unit_rows'] = sum(int(x.attrs.get('v23_fusion_unit_rows', 0)) for x in xs)
                diag['v23_fusion_figure_candidate_pages'] = sum(int(x.attrs.get('v23_fusion_figure_candidate_pages', 0)) for x in xs)
                diag.setdefault('extractors_tried', []).append('v23_fusion_unit_and_figure_extractor')
        except Exception as e:
            diag['v23_fusion_error'] = repr(e)
        return frames, diag
except Exception:
    pass


_v23_seed_ref = additional_seed_sources_v11

def additional_seed_sources_v11(test_id: str) -> List[Dict[str, Any]]:  # type: ignore[override]
    try:
        base = list(_v23_seed_ref(test_id))
    except Exception:
        base = []
    tid = str(test_id).upper()
    extra: List[Dict[str, Any]] = []
    if tid == 'T44':
        extra.extend([
            {'url':'https://en.wikichip.org/wiki/3d_nand', 'label':'v23 WikiChip 3D NAND parser target', 'reason':'v23_t44_primary_el_source', 'tier':'html_table_candidate'},
            {'url':'https://en.wikichip.org/wiki/list_of_flash_memory_cells', 'label':'v23 WikiChip flash-cell table parser target', 'reason':'v23_t44_primary_el_source', 'tier':'html_table_candidate'},
            {'url':'https://www.techinsights.com/blog/3d-nand-flash-memory-density-ranking', 'label':'v23 TechInsights NAND density parser target', 'reason':'v23_t44_primary_el_source', 'tier':'html_table_candidate'},
            {'url':'https://zenodo.org/api/records/?q=%223D%20NAND%22%20layers%20die%20area%20capacity&size=10', 'label':'v23 3D NAND die area capacity repository query', 'reason':'v23_t44_repository_query', 'tier':'metadata_only'},
        ])
    if tid == 'T45':
        extra.extend([
            {'url':'https://zenodo.org/api/records/?q=%22fJ%2Fbit%22%20%22Gb%2Fs%22%20%22optical%20interconnect%22&size=10', 'label':'v23 optical fJ/bit bandwidth query', 'reason':'v23_t45_unit_query', 'tier':'metadata_only'},
            {'url':'https://zenodo.org/api/records/?q=%22pJ%2Fbit%22%20%22silicon%20photonics%22%20reach&size=10', 'label':'v23 silicon photonics pJ/bit reach query', 'reason':'v23_t45_unit_query', 'tier':'metadata_only'},
        ])
    if tid == 'T47':
        extra.extend([
            {'url':'https://zenodo.org/api/records/?q=%22Loihi%22%20benchmark%20energy%20accuracy&size=10', 'label':'v23 Loihi benchmark energy accuracy query', 'reason':'v23_t47_exact_benchmark_query', 'tier':'metadata_only'},
            {'url':'https://zenodo.org/api/records/?q=%22TrueNorth%22%20benchmark%20energy%20accuracy&size=10', 'label':'v23 TrueNorth benchmark energy query', 'reason':'v23_t47_exact_benchmark_query', 'tier':'metadata_only'},
            {'url':'https://zenodo.org/api/records/?q=%22SpiNNaker%22%20energy%20per%20spike%20benchmark&size=10', 'label':'v23 SpiNNaker exact benchmark query', 'reason':'v23_t47_exact_benchmark_query', 'tier':'metadata_only'},
        ])
    if tid in {'T26','T27','T28','T29','T30'}:
        extra.extend([
            {'url':'https://zenodo.org/api/records/?q=%22Wped%22%20%22Pped%22%20%22ELM%22%20tokamak&size=10', 'label':'v23 Wped Pped ELM exact repository query', 'reason':'v23_fusion_secondary_query', 'tier':'metadata_only'},
            {'url':'https://zenodo.org/api/records/?q=%22tau_E%22%20%22H98%22%20%22q95%22%20tokamak&size=10', 'label':'v23 H98 tau_E q95 exact repository query', 'reason':'v23_fusion_secondary_query', 'tier':'metadata_only'},
            {'url':'https://zenodo.org/api/records/?q=%22W7-X%22%20profile%20transport%20public%20data&size=10', 'label':'v23 W7-X profile transport public query', 'reason':'v23_fusion_secondary_query', 'tier':'metadata_only'},
        ])
    if tid == 'T53':
        extra.extend([
            {'url':'https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/README.md', 'label':'v23 ProteinGym README mapping source', 'reason':'v23_t53_mapping_source', 'tier':'metadata_only'},
            {'url':'https://zenodo.org/api/records/?q=ProteinGym%20DMS%20fitness%20assay%20csv&size=10', 'label':'v23 ProteinGym DMS fitness exact query', 'reason':'v23_t53_data_query', 'tier':'metadata_only'},
        ])
    return _dedupe_link_dicts(base + extra)


# ---------------------------------------------------------------------------
# v24 exact parser upgrades: stronger EL/T44, T45, T47 and fusion numeric-line
# extraction. All secondary extractions are diagnostics only.
# ---------------------------------------------------------------------------

def _v24_text(data: bytes, url: str, max_pages: int = 80) -> str:
    # Try prior text extractor first, then PyPDF/pypdf/PyPDF2 fallback, then raw decode.
    try:
        txt = _v23_text(data, url, max_pages=max_pages)
        if txt and len(txt.strip()) > 100:
            return txt
    except Exception:
        pass
    try:
        import pypdf  # type: ignore
        import io as _io
        parts = []
        r = pypdf.PdfReader(_io.BytesIO(data))
        for p in r.pages[:max_pages]:
            try:
                parts.append(p.extract_text() or '')
            except Exception:
                pass
        txt = '\n'.join(parts)
        if txt.strip():
            return txt
    except Exception:
        pass
    try:
        import PyPDF2  # type: ignore
        import io as _io
        parts = []
        r = PyPDF2.PdfReader(_io.BytesIO(data))
        for p in r.pages[:max_pages]:
            try:
                parts.append(p.extract_text() or '')
            except Exception:
                pass
        txt = '\n'.join(parts)
        if txt.strip():
            return txt
    except Exception:
        pass
    try:
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return ''


def _v24_nand_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    rows: List[Dict[str, Any]] = []
    text = _v24_text(data, url, max_pages=30)
    def parse_line(l: str) -> Optional[Dict[str, Any]]:
        low = l.lower()
        if not re.search(r'3d\s*nand|v-?nand|nand|flash|layer', low):
            return None
        if not re.search(r'layer|gb|gbit|tb|die|mm\s*2|mm2|bits?/cell|tlc|qlc|mlc|slc', low):
            return None
        layers = None
        m = re.search(r'(?<!\d)(\d{2,3})\s*(?:-|\s)?layers?\b', low)
        if m: layers = float(m.group(1))
        cap = None
        m = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s*(tb|tbit|gb|gbit|gbits?)\b', low)
        if m: cap = float(m.group(1)) * (1000.0 if m.group(2).startswith('t') else 1.0)
        area = None
        m = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s*(?:mm\s*2|mm2|mm\^2)', low)
        if m: area = float(m.group(1))
        bits = None
        if 'qlc' in low: bits = 4.0
        elif 'tlc' in low: bits = 3.0
        elif 'mlc' in low: bits = 2.0
        elif 'slc' in low: bits = 1.0
        else:
            m = re.search(r'\b([12345])\s*(?:bits?\s*/\s*cell|bits? per cell|bpc)\b', low)
            if m: bits = float(m.group(1))
        manu = None
        m = re.search(r'\b(Samsung|Micron|SK\s*Hynix|Hynix|Kioxia|Toshiba|Western\s*Digital|SanDisk|Intel|YMTC)\b', l, flags=re.I)
        if m: manu = m.group(1)
        year = None
        m = re.search(r'\b(20\d{2}|19\d{2})\b', l)
        if m: year = float(m.group(1))
        if layers or (cap and area) or (layers and cap):
            return {'manufacturer': manu, 'year': year, 'generation_or_product': l[:160], 'layers': layers, 'die_capacity_Gb': cap, 'die_area_mm2': area, 'bits_per_cell': bits, 'density_Gb_per_mm2': (cap/area if cap and area else None), 'line_text': l[:750], 'source_url': url}
        return None
    # HTML tables first.
    try:
        tables = pd.read_html(io.BytesIO(data))
    except Exception:
        tables = []
    for tab in tables[:20]:
        # Treat each row as text; robust to wildly different column naming.
        for _, r in tab.iterrows():
            row = ' '.join(str(x) for x in r.tolist() if str(x) != 'nan')
            pr = parse_line(row)
            if pr: rows.append(pr)
    for line in text.splitlines():
        l = re.sub(r'\s+', ' ', line.strip())
        if 10 <= len(l) <= 1000:
            pr = parse_line(l)
            if pr: rows.append(pr)
    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=['line_text','source_url'])
        df.attrs.update({'source_url': url, 'evidence_tier': 'secondary_exact_spec_text_or_html_table', 'parser': 'v24_nand_exact_parser', 'v24_nand_rows': len(df), 'confirmation_allowed': False, 'falsification_allowed': False})
        frames.append(df)
    return frames


def _v24_optical_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    text = _v24_text(data, url, max_pages=60)
    rows=[]
    for line in text.splitlines():
        l = re.sub(r'\s+', ' ', line.strip())
        low = l.lower()
        if len(l) < 8 or len(l) > 1000:
            continue
        if not re.search(r'optical|photonic|silicon photonics|interconnect|link|i/o|serdes|modulator|transceiver', low):
            continue
        if not re.search(r'fj\s*/\s*bit|pj\s*/\s*bit|gb/s|gbps|tb/s|tbps|reach|bandwidth|mm|cm|\bm\b', low):
            continue
        epb=None; bw=None; reach=None; year=None
        m=re.search(r'(\d+(?:\.\d+)?)\s*(fJ|pJ)\s*/\s*bit', l, flags=re.I)
        if m: epb=float(m.group(1))*(0.001 if m.group(2).lower()=='fj' else 1.0)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(Gb/s|Gbps|Tb/s|Tbps)', l, flags=re.I)
        if m: bw=float(m.group(1))*(1000.0 if m.group(2).lower().startswith('t') else 1.0)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|m)\b', l, flags=re.I)
        if m: reach=float(m.group(1))*({'mm':1.0,'cm':10.0,'m':1000.0}[m.group(2).lower()])
        m=re.search(r'\b(20\d{2}|19\d{2})\b', l)
        if m: year=float(m.group(1))
        if epb is not None or bw is not None or reach is not None:
            rows.append({'technology': 'optical' if re.search(r'optical|photonic', low) else None, 'year': year, 'energy_pJ_per_bit': epb, 'bandwidth_Gbps': bw, 'reach_mm': reach, 'line_text': l[:800], 'source_url': url})
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['line_text','source_url'])
    df.attrs.update({'source_url': url, 'evidence_tier': 'secondary_exact_pdf_unit_text_table', 'parser': 'v24_optical_unit_line_parser', 'v24_optical_rows': len(df), 'confirmation_allowed': False, 'falsification_allowed': False})
    return [df]


def _v24_neuro_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    text = _v24_text(data, url, max_pages=60)
    rows=[]
    for line in text.splitlines():
        l=re.sub(r'\s+', ' ', line.strip())
        low=l.lower()
        if len(l)<8 or len(l)>1000: continue
        if not re.search(r'loihi|truenorth|spinnaker|brainscales|neuromorphic', low): continue
        if not re.search(r'energy|power|pj|nj|uj|µj|spike|inference|accuracy|benchmark|core|neuron|synapse|mnist|imagenet|dvs', low): continue
        energy=None; acc=None; chip=None; cores=None
        m=re.search(r'(Loihi\s*2?|TrueNorth|SpiNNaker|BrainScaleS)', l, flags=re.I)
        if m: chip=m.group(1)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(pJ|nJ|uJ|µJ|mJ)\b', l, flags=re.I)
        if m: energy=float(m.group(1))*({'pj':1e-12,'nj':1e-9,'uj':1e-6,'µj':1e-6,'mj':1e-3}[m.group(2).lower()])
        m=re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:accuracy|acc)?', l, flags=re.I)
        if m: acc=float(m.group(1))
        m=re.search(r'(\d+(?:\.\d+)?)\s*(?:cores?|neurons?|synapses?)', l, flags=re.I)
        if m: cores=float(m.group(1))
        rows.append({'chip': chip, 'energy_J_hint': energy, 'accuracy_percent_hint': acc, 'neurons_cores_topology_hint': cores, 'line_text': l[:800], 'source_url': url})
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['line_text','source_url'])
    df.attrs.update({'source_url': url, 'evidence_tier': 'secondary_exact_benchmark_text_table', 'parser': 'v24_neuromorphic_benchmark_parser', 'v24_neuro_rows': len(df), 'confirmation_allowed': False, 'falsification_allowed': False})
    return [df]


def _v24_fusion_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    text = _v24_text(data, url, max_pages=100)
    rows=[]; figs=[]
    terms = r'ELM|pedestal|RMP|H98|tau[_\s-]?E|tauE|q95|Wped|Pped|dW|ΔW|energy\s+loss|DIII-D|JET|ASDEX|AUG|ITER|W7-X|tokamak|H-mode'
    units = r'MJ|kJ|\bJ\b|MW|kPa|Pa|ms|\bs\b|Hz|kHz|m\^-?3|10\^19|%|MA|\bT\b|keV|eV'
    for i,line in enumerate(text.splitlines()):
        l=re.sub(r'\s+', ' ', line.strip())
        if len(l)<8 or len(l)>1200: continue
        if re.search(terms, l, flags=re.I):
            if re.search(units, l, flags=re.I) and re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', l):
                nums=re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', l)
                us=re.findall(units, l, flags=re.I)
                rows.append({'line_index': i, 'line_text': l[:900], 'numeric_values': ';'.join(nums[:25]), 'units_found': ';'.join(us[:25]), 'source_url': url})
            if re.search(r'figure|fig\.|table|axis|versus|vs\.|correlation|scal|plot', l, flags=re.I):
                figs.append({'line_index': i, 'figure_candidate_text': l[:900], 'source_url': url})
    out=[]
    if rows:
        df=pd.DataFrame(rows).drop_duplicates(subset=['line_text','source_url'])
        df.attrs.update({'source_url': url, 'evidence_tier': 'secondary_auto_pdf_text_table', 'parser': 'v24_fusion_numeric_line_extractor', 'v24_fusion_unit_rows': len(df), 'confirmation_allowed': False, 'falsification_allowed': False})
        out.append(df)
    if figs:
        df=pd.DataFrame(figs).drop_duplicates(subset=['figure_candidate_text','source_url'])
        df.attrs.update({'source_url': url, 'evidence_tier': 'secondary_figure_page_candidate', 'parser': 'v24_fusion_figure_candidate_detector', 'v24_fusion_figure_candidate_pages': len(df), 'confirmation_allowed': False, 'falsification_allowed': False})
        out.append(df)
    return out


try:
    _v24_extract_ref = extract_frames_from_artifact  # type: ignore[name-defined]
    def extract_frames_from_artifact(data: bytes, url: str, meta: Dict[str, Any], cache_dir: Path) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:  # type: ignore[override]
        frames, diag = _v24_extract_ref(data, url, meta, cache_dir)
        us=str(url); low=us.lower(); ctype=str((meta or {}).get('content_type') or '').lower()
        try:
            if re.search(r'3d\s*nand|v-?nand|nand|flash|wikichip|techinsights|samsung|micron|hynix|kioxia|toshiba|sandisk', low, flags=re.I):
                xs=_v24_nand_frames(data, us); frames.extend(xs)
                diag['v24_nand_rows']=sum(int(x.attrs.get('v24_nand_rows', len(x))) for x in xs)
                diag.setdefault('extractors_tried', []).append('v24_nand_exact_parser')
        except Exception as e: diag['v24_nand_error']=repr(e)
        try:
            if re.search(r'optical|photonic|interconnect|irds|silicon.?photonics|pj|fj|serdes', low, flags=re.I):
                xs=_v24_optical_frames(data, us); frames.extend(xs)
                diag['v24_optical_rows']=sum(int(x.attrs.get('v24_optical_rows', len(x))) for x in xs)
                diag.setdefault('extractors_tried', []).append('v24_optical_unit_line_parser')
        except Exception as e: diag['v24_optical_error']=repr(e)
        try:
            if re.search(r'loihi|truenorth|spinnaker|brainscales|neuromorphic', low, flags=re.I):
                xs=_v24_neuro_frames(data, us); frames.extend(xs)
                diag['v24_neuro_rows']=sum(int(x.attrs.get('v24_neuro_rows', len(x))) for x in xs)
                diag.setdefault('extractors_tried', []).append('v24_neuromorphic_exact_parser')
        except Exception as e: diag['v24_neuro_error']=repr(e)
        try:
            if ('pdf' in ctype or low.endswith('.pdf')) and re.search(r'elm|pedestal|rmp|h-mode|tokamak|w7-x|diii-d|jet|iter|asdex|aug|loarte|liang|fusion', low, flags=re.I):
                xs=_v24_fusion_frames(data, us); frames.extend(xs)
                diag['v24_fusion_unit_rows']=sum(int(x.attrs.get('v24_fusion_unit_rows', 0)) for x in xs)
                diag['v24_fusion_figure_candidate_pages']=sum(int(x.attrs.get('v24_fusion_figure_candidate_pages', 0)) for x in xs)
                diag.setdefault('extractors_tried', []).append('v24_fusion_numeric_line_and_figure_parser')
        except Exception as e: diag['v24_fusion_error']=repr(e)
        return frames, diag
except Exception:
    pass


_v24_seed_ref = additional_seed_sources_v11

def additional_seed_sources_v11(test_id: str) -> List[Dict[str, Any]]:  # type: ignore[override]
    try: base=list(_v24_seed_ref(test_id))
    except Exception: base=[]
    tid=str(test_id).upper(); extra=[]
    if tid=='T44':
        extra += [
            {'url':'https://en.wikichip.org/wiki/flash_memory', 'label':'v24 WikiChip flash memory NAND parser target', 'reason':'v24_t44_exact_source', 'tier':'html_table_candidate'},
            {'url':'https://en.wikichip.org/wiki/3d_nand', 'label':'v24 WikiChip 3D NAND parser target', 'reason':'v24_t44_exact_source', 'tier':'html_table_candidate'},
            {'url':'https://zenodo.org/api/records/?q=%223D%20NAND%22%20%22die%20area%22%20%22layers%22%20%22bits%20per%20cell%22&size=10', 'label':'v24 exact 3D NAND die/layer/cell query', 'reason':'v24_t44_exact_query', 'tier':'metadata_only'},
        ]
    if tid=='T45':
        extra += [
            {'url':'https://zenodo.org/api/records/?q=%22pJ%2Fbit%22%20%22Gb%2Fs%22%20%22reach%22%20%22optical%20interconnect%22&size=10', 'label':'v24 optical interconnect pJ-bit bandwidth reach query', 'reason':'v24_t45_unit_query', 'tier':'metadata_only'},
            {'url':'https://zenodo.org/api/records/?q=%22fJ%2Fbit%22%20%22silicon%20photonics%22%20%22Gb%2Fs%22&size=10', 'label':'v24 silicon photonics fJ-bit exact query', 'reason':'v24_t45_unit_query', 'tier':'metadata_only'},
        ]
    if tid=='T47':
        extra += [
            {'url':'https://zenodo.org/api/records/?q=%22Loihi%202%22%20%22energy%22%20%22benchmark%22%20accuracy&size=10', 'label':'v24 Loihi 2 benchmark energy exact query', 'reason':'v24_t47_benchmark_query', 'tier':'metadata_only'},
            {'url':'https://zenodo.org/api/records/?q=%22TrueNorth%22%20%22energy%20per%20inference%22%20benchmark&size=10', 'label':'v24 TrueNorth energy inference exact query', 'reason':'v24_t47_benchmark_query', 'tier':'metadata_only'},
        ]
    if tid in {'T26','T27','T28','T29','T30'}:
        extra += [
            {'url':'https://zenodo.org/api/records/?q=%22W_ELM%22%20%22Pped%22%20%22shot%22%20tokamak&size=10', 'label':'v24 W_ELM Pped shot exact query', 'reason':'v24_fusion_numeric_line_query', 'tier':'metadata_only'},
            {'url':'https://zenodo.org/api/records/?q=%22dW%22%20%22Wped%22%20%22ELM%22%20%22JET%22&size=10', 'label':'v24 JET dW Wped ELM exact query', 'reason':'v24_fusion_numeric_line_query', 'tier':'metadata_only'},
            {'url':'https://zenodo.org/api/records/?q=%22DIII-D%22%20%22Wped%22%20%22ELM%22%20%22shot%22&size=10', 'label':'v24 DIII-D Wped ELM shot exact query', 'reason':'v24_fusion_numeric_line_query', 'tier':'metadata_only'},
        ]
    return _dedupe_link_dicts(base+extra)


# ---------------------------------------------------------------------------
# v25 extractor improvements: stronger text layer, exact NAND/optical/neuro
# rows, and fusion secondary numeric-line context extraction.
# ---------------------------------------------------------------------------

def _v25_text(data: bytes, url: str, max_pages: int = 120) -> str:
    # Prefer PyMuPDF text extraction for PDFs; fall back to v24/pypdf/raw.
    try:
        import fitz  # type: ignore
        if (url or '').lower().endswith('.pdf') or data[:4] == b'%PDF':
            doc = fitz.open(stream=data, filetype='pdf')
            parts=[]
            for p in range(min(len(doc), max_pages)):
                try: parts.append(doc[p].get_text('text') or '')
                except Exception: pass
            txt='\n'.join(parts)
            if len(txt.strip())>100: return txt
    except Exception:
        pass
    try:
        return _v24_text(data,url,max_pages=max_pages)
    except Exception:
        try: return data.decode('utf-8', errors='ignore')
        except Exception: return ''


def _v25_float_from_text(x):
    try:
        if x is None: return None
        s=str(x).replace(',',' ').replace('−','-')
        m=re.search(r'[-+]?\d+(?:\.\d+)?', s)
        return float(m.group(0)) if m else None
    except Exception: return None


def _v25_nand_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    rows=[]
    # Structured HTML tables with columns like Layers of Cells / Bits per Cell.
    try: tables=pd.read_html(io.BytesIO(data))
    except Exception: tables=[]
    for tab in tables[:30]:
        cols=[str(c) for c in tab.columns]
        low=[c.lower() for c in cols]
        def pick(*pats):
            for i,c in enumerate(low):
                if any(p in c for p in pats): return cols[i]
            return None
        c_layers=pick('layers','layer count','layers of cells')
        c_bits=pick('bits per cell','bpc','cell type','cell level')
        c_cap=pick('capacity','die capacity','gb','gbit','tbit')
        c_area=pick('die area','area','mm2','mm^2')
        c_year=pick('year','date','introduced','announced')
        c_manu=pick('manufacturer','company','vendor','maker')
        c_prod=pick('product','generation','technology','part','name')
        if c_layers or c_bits or c_cap or c_area:
            for _,r in tab.iterrows():
                line=' '.join(str(v) for v in r.tolist() if str(v)!='nan')
                layers=_v25_float_from_text(r.get(c_layers)) if c_layers else None
                bits=None
                if c_bits:
                    btxt=str(r.get(c_bits)).lower()
                    if 'qlc' in btxt: bits=4.0
                    elif 'tlc' in btxt: bits=3.0
                    elif 'mlc' in btxt: bits=2.0
                    elif 'slc' in btxt: bits=1.0
                    else: bits=_v25_float_from_text(r.get(c_bits))
                cap=_v25_float_from_text(r.get(c_cap)) if c_cap else None
                area=_v25_float_from_text(r.get(c_area)) if c_area else None
                year=_v25_float_from_text(r.get(c_year)) if c_year else None
                manu=str(r.get(c_manu))[:80] if c_manu else None
                prod=str(r.get(c_prod))[:160] if c_prod else line[:160]
                if layers or (cap and area) or (layers and bits):
                    rows.append({'manufacturer':manu,'year':year,'generation_or_product':prod,'layers':layers,'die_capacity_Gb':cap,'die_area_mm2':area,'bits_per_cell':bits,'density_Gb_per_mm2':(cap/area if cap and area else None),'line_text':line[:800],'source_url':url})
    # Reuse v24 line parser if available.
    try:
        for df in _v24_nand_frames(data,url):
            for _,r in df.iterrows(): rows.append(dict(r))
    except Exception: pass
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['line_text','source_url'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_spec_text_or_html_table','parser':'v25_nand_exact_parser','v25_nand_rows':len(df),'confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v25_optical_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    rows=[]; text=_v25_text(data,url,max_pages=100)
    for line in text.splitlines():
        l=re.sub(r'\s+',' ',line.strip()); low=l.lower()
        if len(l)<8 or len(l)>1400: continue
        if not re.search(r'optical|photonic|silicon photonics|interconnect|link|i/o|serdes|modulator|transceiver|wireline',low): continue
        if not re.search(r'fj\s*/\s*bit|pj\s*/\s*bit|gb/s|gbps|tb/s|tbps|reach|bandwidth|mm|cm|\bm\b',low): continue
        epb=bw=reach=year=None
        m=re.search(r'(\d+(?:\.\d+)?)\s*(fJ|pJ)\s*/\s*bit',l,re.I)
        if m: epb=float(m.group(1))*(0.001 if m.group(2).lower()=='fj' else 1.0)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(Gb/s|Gbps|Tb/s|Tbps)',l,re.I)
        if m: bw=float(m.group(1))*(1000.0 if m.group(2).lower().startswith('t') else 1.0)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|m)\b',l,re.I)
        if m: reach=float(m.group(1))*({'mm':1,'cm':10,'m':1000}[m.group(2).lower()])
        m=re.search(r'\b(20\d{2}|19\d{2})\b',l)
        if m: year=float(m.group(1))
        rows.append({'technology':'optical' if re.search(r'optical|photonic',low) else None,'year':year,'energy_pJ_per_bit':epb,'bandwidth_Gbps':bw,'reach_mm':reach,'line_text':l[:900],'source_url':url})
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['line_text','source_url'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_pdf_unit_text_table','parser':'v25_optical_unit_line_parser','v25_optical_rows':len(df),'confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v25_neuro_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    rows=[]; text=_v25_text(data,url,max_pages=100)
    for line in text.splitlines():
        l=re.sub(r'\s+',' ',line.strip()); low=l.lower()
        if len(l)<8 or len(l)>1400: continue
        if not re.search(r'loihi|truenorth|spinnaker|brainscales|neuromorphic',low): continue
        if not re.search(r'energy|power|pj|nj|uj|µj|spike|inference|accuracy|benchmark|core|neuron|synapse|mnist|imagenet|dvs',low): continue
        rows.append({'line_text':l[:900],'source_url':url})
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['line_text','source_url'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_benchmark_text_table','parser':'v25_neuromorphic_benchmark_parser','v25_neuro_rows':len(df),'confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v25_fusion_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    text=_v25_text(data,url,max_pages=140)
    lines=[re.sub(r'\s+',' ',x.strip()) for x in text.splitlines()]
    rows=[]; figs=[]
    terms=r'ELM|pedestal|W_ELM|E_ELM|dW|ΔW|Wped|Pped|energy loss|DIII-D|JET|ITER|W7-X|ASDEX|AUG|tokamak|H-mode|RMP|q95|tau[_\s-]?E|H98'
    units=r'MJ|kJ|\bJ\b|MW|kPa|Pa|ms|Hz|kHz|%|MA|keV|eV|10\^19|m\^-?3'
    for i,l in enumerate(lines):
        if len(l)<6 or len(l)>1200: continue
        ctx=' '.join(lines[max(0,i-2):min(len(lines),i+3)])[:1800]
        if re.search(terms,ctx,re.I) and re.search(units,ctx,re.I) and re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',ctx):
            nums=re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',ctx)
            us=re.findall(units,ctx,re.I)
            rows.append({'line_index':i,'context_text':ctx,'numeric_values':';'.join(nums[:30]),'units_found':';'.join(us[:30]),'source_url':url})
        if re.search(r'fig\.?|figure|table',l,re.I) and re.search(r'ELM|pedestal|energy loss|Wped|Pped|dW|ΔW',ctx,re.I):
            figs.append({'line_index':i,'figure_candidate_text':ctx,'source_url':url})
    out=[]
    if rows:
        df=pd.DataFrame(rows).drop_duplicates(subset=['context_text','source_url'])
        df.attrs.update({'source_url':url,'evidence_tier':'secondary_auto_pdf_text_table','parser':'v25_fusion_numeric_context_extractor','v25_fusion_unit_rows':len(df),'confirmation_allowed':False,'falsification_allowed':False})
        out.append(df)
    if figs:
        df=pd.DataFrame(figs).drop_duplicates(subset=['figure_candidate_text','source_url'])
        df.attrs.update({'source_url':url,'evidence_tier':'secondary_figure_page_candidate','parser':'v25_fusion_figure_context_detector','v25_fusion_figure_candidate_pages':len(df),'confirmation_allowed':False,'falsification_allowed':False})
        out.append(df)
    return out

try:
    _v25_extract_ref = extract_frames_from_artifact  # type: ignore[name-defined]
    def extract_frames_from_artifact(data: bytes, url: str, meta: Dict[str,Any], cache_dir: Path):  # type: ignore[override]
        frames, diag = _v25_extract_ref(data,url,meta,cache_dir)
        us=str(url).lower()
        try:
            if any(x in us for x in ['nand','flash','wikichip','techinsights','samsung','micron','hynix','kioxia']):
                xs=_v25_nand_frames(data,url); frames.extend(xs); diag['v25_nand_rows']=sum(int(x.attrs.get('v25_nand_rows',len(x))) for x in xs); diag.setdefault('extractors_tried',[]).append('v25_nand_exact_parser')
        except Exception as e: diag['v25_nand_error']=repr(e)
        try:
            if any(x in us for x in ['optical','photonic','irds','interconnect','silicon-photonics','pj','fj']):
                xs=_v25_optical_frames(data,url); frames.extend(xs); diag['v25_optical_rows']=sum(int(x.attrs.get('v25_optical_rows',len(x))) for x in xs); diag.setdefault('extractors_tried',[]).append('v25_optical_unit_parser')
        except Exception as e: diag['v25_optical_error']=repr(e)
        try:
            if any(x in us for x in ['loihi','truenorth','spinnaker','brainscales','neuromorphic']):
                xs=_v25_neuro_frames(data,url); frames.extend(xs); diag['v25_neuro_rows']=sum(int(x.attrs.get('v25_neuro_rows',len(x))) for x in xs); diag.setdefault('extractors_tried',[]).append('v25_neuro_benchmark_parser')
        except Exception as e: diag['v25_neuro_error']=repr(e)
        try:
            if any(x in us for x in ['elm','pedestal','fusion','tokamak','diii','jet','iter','w7-x','asdex','rmp']):
                xs=_v25_fusion_frames(data,url); frames.extend(xs); diag['v25_fusion_unit_rows']=sum(int(x.attrs.get('v25_fusion_unit_rows',0)) for x in xs); diag['v25_fusion_figure_candidate_pages']=sum(int(x.attrs.get('v25_fusion_figure_candidate_pages',0)) for x in xs); diag.setdefault('extractors_tried',[]).append('v25_fusion_numeric_context_parser')
        except Exception as e: diag['v25_fusion_error']=repr(e)
        return frames, diag
except Exception:
    pass

try:
    _v25_seed_ref=additional_seed_sources_v11
    def additional_seed_sources_v11(test_id: str) -> List[Dict[str,Any]]:  # type: ignore[override]
        base=list(_v25_seed_ref(test_id))
        if test_id=='T44':
            base += [
                {'url':'https://en.wikichip.org/wiki/3d_nand','label':'v25 WikiChip 3D NAND exact parser target','reason':'v25_t44_exact_parser','tier':'html_table_candidate'},
                {'url':'https://en.wikichip.org/wiki/flash_memory','label':'v25 WikiChip flash memory exact parser target','reason':'v25_t44_exact_parser','tier':'html_table_candidate'},
                {'url':'https://zenodo.org/api/records/?q=%223D%20NAND%22%20%22die%20area%22%20%22layers%22%20%22bits%20per%20cell%22&size=10','label':'v25 3D NAND die/layer/cell exact query','reason':'v25_t44_query','tier':'metadata_only'},
            ]
        if test_id=='T45':
            base += [
                {'url':'https://zenodo.org/api/records/?q=%22pJ%2Fbit%22%20%22Gb%2Fs%22%20%22optical%20interconnect%22%20reach&size=10','label':'v25 optical interconnect pJ-bit reach exact query','reason':'v25_t45_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=%22fJ%2Fbit%22%20%22silicon%20photonics%22%20%22Gb%2Fs%22&size=10','label':'v25 silicon photonics fJ-bit exact query','reason':'v25_t45_query','tier':'metadata_only'},
            ]
        if test_id=='T47':
            base += [
                {'url':'https://zenodo.org/api/records/?q=%22Loihi%202%22%20energy%20benchmark%20accuracy&size=10','label':'v25 Loihi 2 benchmark exact query','reason':'v25_t47_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=TrueNorth%20energy%20inference%20benchmark%20accuracy&size=10','label':'v25 TrueNorth benchmark exact query','reason':'v25_t47_query','tier':'metadata_only'},
            ]
        if test_id in {'T26','T27','T28','T29','T30'}:
            base += [
                {'url':'https://zenodo.org/api/records/?q=%22W_ELM%22%20%22Pped%22%20shot%20tokamak&size=10','label':'v25 W_ELM Pped shot exact query','reason':'v25_fusion_context_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=%22%CE%94W%22%20%22Wped%22%20ELM%20JET&size=10','label':'v25 ΔW Wped JET exact query','reason':'v25_fusion_context_query','tier':'metadata_only'},
            ]
        return base
except Exception:
    pass


# ---------------------------------------------------------------------------
# v26 exact-row / confirm-focused extractors and seeds.
# ---------------------------------------------------------------------------

def _v26_text(data: bytes, url: str, max_pages: int = 160) -> str:
    try:
        import fitz  # type: ignore
        if (url or '').lower().endswith('.pdf') or data[:4] == b'%PDF':
            doc=fitz.open(stream=data, filetype='pdf')
            parts=[]
            for i in range(min(len(doc), max_pages)):
                try: parts.append(doc[i].get_text('text') or '')
                except Exception: pass
            txt='\n'.join(parts)
            if len(txt.strip())>100: return txt
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        import tempfile
        if (url or '').lower().endswith('.pdf') or data[:4] == b'%PDF':
            with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
                tmp.write(data); tmp.flush()
                txt=extract_text(tmp.name) or ''
                if len(txt.strip())>100: return txt
    except Exception:
        pass
    try: return _v25_text(data,url,max_pages=max_pages)
    except Exception:
        try: return data.decode('utf-8',errors='ignore')
        except Exception: return ''


def _v26_num(x):
    try:
        m=re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', str(x).replace(',',''))
        return float(m.group(0)) if m else None
    except Exception: return None


def _v26_nand_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    # Strengthen v25 by adding schema-specific normalization aliases and row provenance.
    rows=[]
    try:
        tables=pd.read_html(io.BytesIO(data))
    except Exception:
        tables=[]
    for tab in tables[:60]:
        cols=[str(c) for c in tab.columns]
        low=[c.lower() for c in cols]
        def pick(*pats):
            for i,c in enumerate(low):
                if any(p in c for p in pats): return cols[i]
            return None
        c_layers=pick('layers','layer count','layers of cells','layer')
        c_bits=pick('bits per cell','bpc','cell type','cell level','slc','mlc','tlc','qlc')
        c_cap=pick('die capacity','capacity','gbit','gb','tbit','tb')
        c_area=pick('die area','area mm','mm2','mm^2','area')
        c_year=pick('year','introduced','announced','date')
        c_manu=pick('manufacturer','company','vendor','maker')
        c_prod=pick('product','generation','technology','node','name')
        score=sum(bool(x) for x in [c_layers,c_bits,c_cap,c_area,c_year,c_manu,c_prod])
        if score<2: continue
        for _,r in tab.iterrows():
            line=' | '.join(str(v) for v in r.tolist() if str(v)!='nan')
            lowline=line.lower()
            layers=_v26_num(r.get(c_layers)) if c_layers else None
            bits=None
            if c_bits:
                btxt=str(r.get(c_bits)).lower()
                bits=4.0 if 'qlc' in btxt else 3.0 if 'tlc' in btxt else 2.0 if 'mlc' in btxt else 1.0 if 'slc' in btxt else _v26_num(r.get(c_bits))
            cap=_v26_num(r.get(c_cap)) if c_cap else None
            area=_v26_num(r.get(c_area)) if c_area else None
            year=_v26_num(r.get(c_year)) if c_year else None
            manu=str(r.get(c_manu))[:100] if c_manu else None
            prod=str(r.get(c_prod))[:180] if c_prod else line[:180]
            if not (layers or (cap and area) or 'nand' in lowline or 'v-nand' in lowline or 'layers of cells' in lowline): continue
            rows.append({'manufacturer':manu,'year':year,'generation_or_product':prod,'layers':layers,'die_capacity_Gb':cap,'die_area_mm2':area,'bits_per_cell':bits,'density_Gb_per_mm2':(cap/area if cap and area else None),'source_url':url,'provenance_line':line[:1000]})
    try:
        for df in _v25_nand_frames(data,url):
            for _,r in df.iterrows(): rows.append(dict(r))
    except Exception: pass
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','provenance_line'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_spec_table','parser':'v26_nand_exact_rows','v26_nand_rows':len(df),'generated_csv':'data/generated/t44_nand_exact_rows_v26.csv','confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v26_optical_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    text=_v26_text(data,url,max_pages=140); rows=[]
    for line in text.splitlines():
        l=re.sub(r'\s+',' ',line.strip()); low=l.lower()
        if len(l)<10 or len(l)>1600: continue
        if not re.search(r'optical|photonic|silicon photonics|interconnect|i/o|link|modulator|transceiver|wireline|serdes',low): continue
        if not re.search(r'fj\s*/\s*bit|pj\s*/\s*bit|gb/s|gbps|tb/s|tbps|reach|bandwidth|\bmm\b|\bcm\b|\bm\b',low): continue
        epb=bw=reach=year=None
        m=re.search(r'(\d+(?:\.\d+)?)\s*(fJ|pJ)\s*/\s*bit',l,re.I)
        if m: epb=float(m.group(1))*(0.001 if m.group(2).lower()=='fj' else 1.0)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(Gb/s|Gbps|Tb/s|Tbps)',l,re.I)
        if m: bw=float(m.group(1))*(1000.0 if m.group(2).lower().startswith('t') else 1.0)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|m)\b',l,re.I)
        if m: reach=float(m.group(1))*({'mm':1.0,'cm':10.0,'m':1000.0}[m.group(2).lower()])
        m=re.search(r'\b(19\d{2}|20\d{2})\b',l)
        if m: year=float(m.group(1))
        rows.append({'technology':'optical' if re.search(r'optical|photonic',low) else 'electrical_or_mixed','year':year,'energy_pJ_per_bit':epb,'bandwidth_Gbps':bw,'reach_mm':reach,'source_url':url,'provenance_line':l[:1000]})
    try:
        for df in _v25_optical_frames(data,url):
            for _,r in df.iterrows(): rows.append(dict(r))
    except Exception: pass
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','provenance_line'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_pdf_unit_text_table','parser':'v26_optical_unit_rows','v26_optical_rows':len(df),'generated_csv':'data/generated/t45_optical_interconnect_rows_v26.csv','confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v26_neuro_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    text=_v26_text(data,url,max_pages=140); rows=[]
    for line in text.splitlines():
        l=re.sub(r'\s+',' ',line.strip()); low=l.lower()
        if len(l)<10 or len(l)>1600: continue
        if not re.search(r'loihi|truenorth|spinnaker|brainscales|neuromorphic',low): continue
        if not re.search(r'energy|power|pj|nj|uj|µj|spike|inference|accuracy|benchmark|core|neuron|synapse|mnist|imagenet|dvs',low): continue
        rows.append({'source_url':url,'provenance_line':l[:1000]})
    try:
        for df in _v25_neuro_frames(data,url):
            for _,r in df.iterrows(): rows.append(dict(r))
    except Exception: pass
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','provenance_line'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_benchmark_text_table','parser':'v26_neuromorphic_rows','v26_neuro_rows':len(df),'generated_csv':'data/generated/t47_neuromorphic_rows_v26.csv','confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v26_fusion_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    text=_v26_text(data,url,max_pages=180)
    lines=[re.sub(r'\s+',' ',x.strip()) for x in text.splitlines()]
    rows=[]; figs=[]
    terms=r'ELM energy|W_ELM|E_ELM|dW|ΔW|delta W|Wped|Pped|pedestal|energy loss|DIII-D|JET|ITER|W7-X|ASDEX|AUG|tokamak|RMP|H98|q95|tau[_\s-]?E'
    units=r'MJ|kJ|\bJ\b|MW|kPa|Pa|ms|Hz|kHz|%|MA|keV|eV|10\^19|m\^-?3'
    for i,l in enumerate(lines):
        ctx=' '.join(lines[max(0,i-2):min(len(lines),i+3)])[:2200]
        if re.search(terms,ctx,re.I) and re.search(units,ctx,re.I) and re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',ctx):
            rows.append({'source_url':url,'line_index':i,'context_text':ctx,'numeric_values':';'.join(re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',ctx)[:40]),'units_found':';'.join(re.findall(units,ctx,re.I)[:40])})
        if re.search(r'fig\.?|figure|table',l,re.I) and re.search(r'ELM|pedestal|energy loss|Wped|Pped|dW|ΔW',ctx,re.I):
            figs.append({'source_url':url,'line_index':i,'figure_candidate_text':ctx})
    try:
        for df in _v25_fusion_frames(data,url):
            for _,r in df.iterrows():
                if 'context_text' in r or 'figure_candidate_text' in r: rows.append(dict(r))
    except Exception: pass
    out=[]
    if rows:
        df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','context_text'])
        df.attrs.update({'source_url':url,'evidence_tier':'secondary_auto_pdf_text_table','parser':'v26_fusion_numeric_context','v26_fusion_unit_rows':len(df),'generated_csv':'data/generated/fusion_secondary_rows_v26.csv','confirmation_allowed':False,'falsification_allowed':False})
        out.append(df)
    if figs:
        df=pd.DataFrame(figs).drop_duplicates(subset=['source_url','figure_candidate_text'])
        df.attrs.update({'source_url':url,'evidence_tier':'secondary_figure_page_candidate','parser':'v26_fusion_figure_context','v26_fusion_figure_candidate_pages':len(df),'confirmation_allowed':False,'falsification_allowed':False})
        out.append(df)
    return out

try:
    _v26_extract_ref=extract_frames_from_artifact  # type: ignore[name-defined]
    def extract_frames_from_artifact(data: bytes, url: str, meta: Dict[str,Any], cache_dir: Path):  # type: ignore[override]
        frames, diag=_v26_extract_ref(data,url,meta,cache_dir)
        us=str(url).lower()
        def add(key, xs, parser):
            frames.extend(xs); diag[key]=sum(int(x.attrs.get(key,len(x))) for x in xs); diag.setdefault('extractors_tried',[]).append(parser)
        try:
            if any(x in us for x in ['nand','flash','wikichip','techinsights','samsung','micron','hynix','kioxia']): add('v26_nand_rows', _v26_nand_frames(data,url), 'v26_nand_exact_rows')
        except Exception as e: diag['v26_nand_error']=repr(e)
        try:
            if any(x in us for x in ['optical','photonic','irds','interconnect','pj','fj','silicon-photonics']): add('v26_optical_rows', _v26_optical_frames(data,url), 'v26_optical_unit_rows')
        except Exception as e: diag['v26_optical_error']=repr(e)
        try:
            if any(x in us for x in ['loihi','truenorth','spinnaker','brainscales','neuromorphic']): add('v26_neuro_rows', _v26_neuro_frames(data,url), 'v26_neuro_rows')
        except Exception as e: diag['v26_neuro_error']=repr(e)
        try:
            if any(x in us for x in ['elm','pedestal','fusion','tokamak','diii','jet','iter','w7-x','asdex','rmp']):
                xs=_v26_fusion_frames(data,url); frames.extend(xs); diag['v26_fusion_unit_rows']=sum(int(x.attrs.get('v26_fusion_unit_rows',0)) for x in xs); diag['v26_fusion_figure_candidate_pages']=sum(int(x.attrs.get('v26_fusion_figure_candidate_pages',0)) for x in xs); diag.setdefault('extractors_tried',[]).append('v26_fusion_numeric_context')
        except Exception as e: diag['v26_fusion_error']=repr(e)
        return frames, diag
except Exception:
    pass

try:
    _v26_seed_ref=additional_seed_sources_v11
    def additional_seed_sources_v11(test_id: str) -> List[Dict[str,Any]]:  # type: ignore[override]
        base=list(_v26_seed_ref(test_id))
        if test_id=='T44':
            base += [
                {'url':'https://en.wikichip.org/wiki/3d_nand','label':'v26 WikiChip 3D NAND exact rows','reason':'v26_t44_confirm_parser','tier':'html_table_candidate'},
                {'url':'https://en.wikichip.org/wiki/flash_memory','label':'v26 WikiChip flash memory exact rows','reason':'v26_t44_confirm_parser','tier':'html_table_candidate'},
                {'url':'https://zenodo.org/api/records/?q=%223D%20NAND%22%20%22die%20area%22%20%22layers%22%20%22bits%20per%20cell%22%20manufacturer&size=10','label':'v26 3D NAND die/layer/bpc manufacturer exact query','reason':'v26_t44_confirm_query','tier':'metadata_only'},
            ]
        if test_id=='T45':
            base += [
                {'url':'https://zenodo.org/api/records/?q=%22pJ%2Fbit%22%20%22Gb%2Fs%22%20reach%20%22optical%20interconnect%22&size=10','label':'v26 optical interconnect pJ-bit reach exact query','reason':'v26_t45_confirm_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=%22fJ%2Fbit%22%20%22silicon%20photonics%22%20%22Tb%2Fs%22&size=10','label':'v26 silicon photonics fJ-bit Tb/s query','reason':'v26_t45_confirm_query','tier':'metadata_only'},
            ]
        if test_id=='T47':
            base += [
                {'url':'https://zenodo.org/api/records/?q=%22Loihi%202%22%20%22energy%20per%20inference%22%20accuracy%20benchmark&size=10','label':'v26 Loihi2 energy accuracy benchmark query','reason':'v26_t47_confirm_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=SpiNNaker%20TrueNorth%20Loihi%20energy%20benchmark%20accuracy&size=10','label':'v26 neuromorphic energy benchmark query','reason':'v26_t47_confirm_query','tier':'metadata_only'},
            ]
        if test_id in {'T26','T27','T28','T29','T30'}:
            base += [
                {'url':'https://zenodo.org/api/records/?q=%22W_ELM%22%20%22Pped%22%20%22shot%22%20tokamak&size=10','label':'v26 W_ELM Pped shot exact query','reason':'v26_fusion_confirm_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=%22ELM%20energy%22%20%22Wped%22%20%22dW%22%20JET&size=10','label':'v26 ELM Wped dW JET query','reason':'v26_fusion_confirm_query','tier':'metadata_only'},
            ]
        if test_id in {'T57','T59'}:
            base += [
                {'url':'https://www.hepdata.net/search/?q=MET%20Drell-Yan%20di-Higgs%20csv','label':'v26 HEPData exact table search only','reason':'v26_hepdata_manifest_query','tier':'metadata_only'},
            ]
        return base
except Exception:
    pass


# ---------------------------------------------------------------------------
# v27 exact-row extraction: stronger EL/NAND, optical, neuro, fusion routes.
# These extract secondary/diagnostic normalized rows; confirmation remains gated
# in tierb_runner and requires sufficient rows + controls.
# ---------------------------------------------------------------------------

def _v27_text(data: bytes, url: str, max_pages: int = 220) -> str:
    try:
        return _v26_text(data, url, max_pages=max_pages)
    except Exception:
        try:
            return data.decode('utf-8', errors='ignore')
        except Exception:
            return ''


def _v27_num(x):
    try:
        m=re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', str(x).replace(',',''))
        return float(m.group(0)) if m else None
    except Exception:
        return None


def _v27_nand_text_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    rows=[]
    # Start with v26 table parser.
    try:
        for df in _v26_nand_frames(data, url):
            for _,r in df.iterrows(): rows.append(dict(r))
    except Exception:
        pass
    text=_v27_text(data, url, max_pages=80)
    lines=[re.sub(r'\s+',' ',x.strip()) for x in text.splitlines()]
    vendors=r'Samsung|Micron|SK\s*Hynix|Hynix|Kioxia|Toshiba|Intel|Western Digital|WD|YMTC|SanDisk'
    for i,l in enumerate(lines):
        ctx=' '.join(lines[max(0,i-1):min(len(lines),i+2)])[:1600]
        low=ctx.lower()
        if not re.search(r'3d|v-nand|vertical|nand|flash', low):
            continue
        if not re.search(r'layer|\b\d{2,3}\s*L\b|Gb|Gbit|Tbit|bits? per cell|TLC|QLC|MLC|die area|mm\s*(?:2|\^2|²)', ctx, re.I):
            continue
        layers=None
        m=re.search(r'(\d{2,3})\s*(?:layers?|L)\b', ctx, re.I)
        if m: layers=float(m.group(1))
        cap=None
        m=re.search(r'(\d+(?:\.\d+)?)\s*(Tb|Tbit|Gb|Gbit)\b', ctx, re.I)
        if m: cap=float(m.group(1))*(1000.0 if m.group(2).lower().startswith('t') else 1.0)
        area=None
        m=re.search(r'(\d+(?:\.\d+)?)\s*mm\s*(?:2|\^2|²)', ctx, re.I)
        if m: area=float(m.group(1))
        bits=None
        if re.search(r'QLC', ctx, re.I): bits=4.0
        elif re.search(r'TLC', ctx, re.I): bits=3.0
        elif re.search(r'MLC', ctx, re.I): bits=2.0
        elif re.search(r'SLC', ctx, re.I): bits=1.0
        else:
            m=re.search(r'(\d+(?:\.\d+)?)\s*bits?\s*per\s*cell', ctx, re.I)
            if m: bits=float(m.group(1))
        year=None
        m=re.search(r'\b(20\d{2}|19\d{2})\b', ctx)
        if m: year=float(m.group(1))
        manu=None
        m=re.search(vendors, ctx, re.I)
        if m: manu=m.group(0)
        if layers or (cap and area) or bits:
            rows.append({'manufacturer':manu,'year':year,'generation_or_product':ctx[:180],'layers':layers,'die_capacity_Gb':cap,'die_area_mm2':area,'bits_per_cell':bits,'density_Gb_per_mm2':(cap/area if cap and area else None),'source_url':url,'provenance_line':ctx})
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','provenance_line'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_spec_table','parser':'v27_nand_exact_rows','v27_nand_rows':len(df),'generated_csv':'data/generated/t44_nand_exact_rows_v27.csv','confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v27_optical_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    rows=[]
    try:
        for df in _v26_optical_frames(data, url):
            for _,r in df.iterrows(): rows.append(dict(r))
    except Exception:
        pass
    text=_v27_text(data, url, max_pages=180)
    lines=[re.sub(r'\s+',' ',x.strip()) for x in text.splitlines()]
    for i,l in enumerate(lines):
        ctx=' '.join(lines[max(0,i-1):min(len(lines),i+2)])[:1800]
        low=ctx.lower()
        if not re.search(r'optical|photon|silicon photonics|interconnect|link|i/o|transceiver|serdes|modulator', low):
            continue
        if not re.search(r'fJ\s*/\s*bit|pJ\s*/\s*bit|Gb/s|Gbps|Tb/s|Tbps|reach|bandwidth|\bmm\b|\bcm\b|\bm\b', ctx, re.I):
            continue
        epb=bw=reach=year=None
        m=re.search(r'(\d+(?:\.\d+)?)\s*(fJ|pJ)\s*/\s*bit',ctx,re.I)
        if m: epb=float(m.group(1))*(0.001 if m.group(2).lower()=='fj' else 1.0)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(Gb/s|Gbps|Tb/s|Tbps)',ctx,re.I)
        if m: bw=float(m.group(1))*(1000.0 if m.group(2).lower().startswith('t') else 1.0)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|m)\b',ctx,re.I)
        if m: reach=float(m.group(1))*({'mm':1.0,'cm':10.0,'m':1000.0}[m.group(2).lower()])
        m=re.search(r'\b(20\d{2}|19\d{2})\b',ctx)
        if m: year=float(m.group(1))
        rows.append({'technology':'optical' if re.search(r'optical|photon',low) else 'electrical_or_mixed','year':year,'energy_pJ_per_bit':epb,'bandwidth_Gbps':bw,'reach_mm':reach,'source_url':url,'provenance_line':ctx})
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','provenance_line'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_pdf_unit_text_table','parser':'v27_optical_unit_rows','v27_optical_rows':len(df),'generated_csv':'data/generated/t45_optical_interconnect_rows_v27.csv','confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v27_neuro_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    rows=[]
    try:
        for df in _v26_neuro_frames(data, url):
            for _,r in df.iterrows(): rows.append(dict(r))
    except Exception:
        pass
    text=_v27_text(data, url, max_pages=180)
    lines=[re.sub(r'\s+',' ',x.strip()) for x in text.splitlines()]
    for i,l in enumerate(lines):
        ctx=' '.join(lines[max(0,i-1):min(len(lines),i+2)])[:1800]
        low=ctx.lower()
        if not re.search(r'loihi|loihi\s*2|truenorth|spinnaker|brainscales|neuromorphic', low):
            continue
        if not re.search(r'energy|power|pJ|nJ|uJ|µJ|spike|inference|accuracy|benchmark|core|neuron|synapse|MNIST|DVS|imagenet', ctx, re.I):
            continue
        rows.append({'source_url':url,'chip_hint':re.search(r'Loihi\s*2|Loihi|TrueNorth|SpiNNaker|BrainScaleS',ctx,re.I).group(0) if re.search(r'Loihi\s*2|Loihi|TrueNorth|SpiNNaker|BrainScaleS',ctx,re.I) else None,'provenance_line':ctx})
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','provenance_line'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_benchmark_text_table','parser':'v27_neuromorphic_rows','v27_neuro_rows':len(df),'generated_csv':'data/generated/t47_neuromorphic_rows_v27.csv','confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v27_fusion_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    rows=[]; figs=[]
    try:
        for df in _v26_fusion_frames(data, url):
            if 'v26_fusion_unit_rows' in df.attrs:
                for _,r in df.iterrows(): rows.append(dict(r))
            elif 'v26_fusion_figure_candidate_pages' in df.attrs:
                for _,r in df.iterrows(): figs.append(dict(r))
    except Exception:
        pass
    text=_v27_text(data, url, max_pages=240)
    lines=[re.sub(r'\s+',' ',x.strip()) for x in text.splitlines()]
    terms=r'ELM energy|W_ELM|E_ELM|Wped|Pped|dW|ΔW|delta W|pedestal energy|pedestal pressure|energy loss|DIII-D|JET|ITER|W7-X|ASDEX|AUG|tokamak|RMP|H98|q95|tau[_\s-]?E'
    units=r'MJ|kJ|\bJ\b|MW|kPa|Pa|ms|Hz|kHz|%|MA|keV|eV|10\^19|m\^-?3'
    for i,l in enumerate(lines):
        ctx=' '.join(lines[max(0,i-2):min(len(lines),i+3)])[:2400]
        if re.search(terms,ctx,re.I) and re.search(units,ctx,re.I) and re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',ctx):
            rows.append({'source_url':url,'line_index':i,'context_text':ctx,'numeric_values':';'.join(re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',ctx)[:60]),'units_found':';'.join(re.findall(units,ctx,re.I)[:60])})
        if re.search(r'fig\.?|figure|table', l, re.I) and re.search(r'ELM|pedestal|energy loss|Wped|Pped|dW|ΔW', ctx, re.I):
            figs.append({'source_url':url,'line_index':i,'figure_candidate_text':ctx})
    out=[]
    if rows:
        df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','context_text'])
        df.attrs.update({'source_url':url,'evidence_tier':'secondary_auto_pdf_text_table','parser':'v27_fusion_numeric_context','v27_fusion_unit_rows':len(df),'generated_csv':'data/generated/fusion_secondary_rows_v27.csv','confirmation_allowed':False,'falsification_allowed':False})
        out.append(df)
    if figs:
        df=pd.DataFrame(figs).drop_duplicates(subset=['source_url','figure_candidate_text'])
        df.attrs.update({'source_url':url,'evidence_tier':'secondary_figure_page_candidate','parser':'v27_fusion_figure_context','v27_fusion_figure_candidate_pages':len(df),'confirmation_allowed':False,'falsification_allowed':False})
        out.append(df)
    return out

try:
    _v27_extract_ref=extract_frames_from_artifact  # type: ignore[name-defined]
    def extract_frames_from_artifact(data: bytes, url: str, meta: Dict[str,Any], cache_dir: Path):  # type: ignore[override]
        frames, diag=_v27_extract_ref(data,url,meta,cache_dir)
        us=str(url).lower()
        def add(key, xs, parser):
            if not xs: return
            frames.extend(xs)
            diag[key]=sum(int(x.attrs.get(key,len(x))) for x in xs)
            diag.setdefault('extractors_tried',[]).append(parser)
        try:
            if any(x in us for x in ['nand','flash','wikichip','techinsights','samsung','micron','hynix','kioxia']):
                add('v27_nand_rows', _v27_nand_text_frames(data,url), 'v27_nand_exact_rows')
        except Exception as e: diag['v27_nand_error']=repr(e)
        try:
            if any(x in us for x in ['optical','photonic','irds','interconnect','pj','fj','silicon-photonics','transceiver','serdes']):
                add('v27_optical_rows', _v27_optical_frames(data,url), 'v27_optical_unit_rows')
        except Exception as e: diag['v27_optical_error']=repr(e)
        try:
            if any(x in us for x in ['loihi','truenorth','spinnaker','brainscales','neuromorphic']):
                add('v27_neuro_rows', _v27_neuro_frames(data,url), 'v27_neuromorphic_rows')
        except Exception as e: diag['v27_neuro_error']=repr(e)
        try:
            if any(x in us for x in ['elm','pedestal','fusion','tokamak','diii','jet','iter','w7-x','asdex','rmp','h-mode']):
                xs=_v27_fusion_frames(data,url)
                frames.extend(xs)
                diag['v27_fusion_unit_rows']=sum(int(x.attrs.get('v27_fusion_unit_rows',0)) for x in xs)
                diag['v27_fusion_figure_candidate_pages']=sum(int(x.attrs.get('v27_fusion_figure_candidate_pages',0)) for x in xs)
                if xs: diag.setdefault('extractors_tried',[]).append('v27_fusion_numeric_context')
        except Exception as e: diag['v27_fusion_error']=repr(e)
        return frames, diag
except Exception:
    pass

try:
    _v27_seed_ref=additional_seed_sources_v11
    def additional_seed_sources_v11(test_id: str):  # type: ignore[override]
        base=list(_v27_seed_ref(test_id))
        extra=[]
        if test_id=='T44':
            extra += [
                {'url':'https://en.wikichip.org/wiki/3d_nand','label':'v27 WikiChip 3D NAND exact normalized rows','reason':'v27_t44_confirm_parser','tier':'html_table_candidate'},
                {'url':'https://en.wikichip.org/wiki/flash_memory','label':'v27 WikiChip flash memory exact normalized rows','reason':'v27_t44_confirm_parser','tier':'html_table_candidate'},
                {'url':'https://en.wikichip.org/wiki/list_of_flash_memory_cells','label':'v27 WikiChip flash memory cell list','reason':'v27_t44_confirm_parser','tier':'html_table_candidate'},
                {'url':'https://zenodo.org/api/records/?q=%223D%20NAND%22%20%22die%20area%22%20%22layers%22%20%22bits%20per%20cell%22%20manufacturer&size=10','label':'v27 3D NAND exact repository query','reason':'v27_t44_confirm_query','tier':'metadata_only'},
            ]
        if test_id=='T45':
            extra += [
                {'url':'https://zenodo.org/api/records/?q=%22pJ%2Fbit%22%20%22Gb%2Fs%22%20reach%20%22optical%20interconnect%22&size=10','label':'v27 optical interconnect pJ/bit reach query','reason':'v27_t45_confirm_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=%22fJ%2Fbit%22%20%22silicon%20photonics%22%20%22Tb%2Fs%22&size=10','label':'v27 silicon photonics fJ/bit Tb/s query','reason':'v27_t45_confirm_query','tier':'metadata_only'},
            ]
        if test_id=='T47':
            extra += [
                {'url':'https://zenodo.org/api/records/?q=%22Loihi%202%22%20%22energy%20per%20inference%22%20accuracy%20benchmark&size=10','label':'v27 Loihi2 energy accuracy benchmark query','reason':'v27_t47_confirm_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=SpiNNaker%20TrueNorth%20Loihi%20energy%20benchmark%20accuracy&size=10','label':'v27 neuromorphic energy benchmark query','reason':'v27_t47_confirm_query','tier':'metadata_only'},
            ]
        if test_id in {'T26','T27','T28','T29','T30'}:
            extra += [
                {'url':'https://zenodo.org/api/records/?q=%22W_ELM%22%20%22Pped%22%20%22shot%22%20tokamak&size=10','label':'v27 W_ELM Pped shot exact query','reason':'v27_fusion_confirm_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=%22ELM%20energy%22%20%22Wped%22%20%22dW%22%20JET&size=10','label':'v27 ELM Wped dW JET query','reason':'v27_fusion_confirm_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=%22tau_E%22%20%22H98%22%20%22q95%22%20tokamak%20database&size=10','label':'v27 H98 tauE q95 database query','reason':'v27_fusion_confirm_query','tier':'metadata_only'},
            ]
        return base+extra
except Exception:
    pass


# ---------------------------------------------------------------------------
# v28 source-specific extraction layer: stronger exact parsers for confirm rows.
# This does not relax evidence gates; generated rows are diagnostic until runner
# sees enough rows and controls pass.
# ---------------------------------------------------------------------------

def _v28_text(data: bytes, url: str, max_pages: int = 260) -> str:
    try:
        return _v27_text(data, url, max_pages=max_pages)
    except Exception:
        try: return data.decode('utf-8', errors='ignore')
        except Exception: return ''


def _v28_firstnum(s):
    try:
        m=re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', str(s).replace(',',''))
        return float(m.group(0)) if m else None
    except Exception: return None


def _v28_nand_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    rows=[]
    try:
        for df in _v27_nand_text_frames(data,url):
            for _,r in df.iterrows(): rows.append(dict(r))
    except Exception: pass
    try:
        tables=pd.read_html(io.BytesIO(data))
    except Exception:
        tables=[]
    for tab in tables[:100]:
        if tab.empty: continue
        tab=tab.copy()
        # Flatten multiindex columns.
        tab.columns=[' '.join([str(x) for x in c if str(x)!='nan']) if isinstance(c, tuple) else str(c) for c in tab.columns]
        low=[c.lower() for c in tab.columns]
        def findcol(*patterns):
            for i,c in enumerate(low):
                if any(re.search(p,c,re.I) for p in patterns): return tab.columns[i]
            return None
        c_layers=findcol(r'\blayers?\b', r'layer count', r'layers of cells')
        c_cap=findcol(r'die capacity', r'capacity', r'\bgb\b', r'gbit', r'tbit', r'\btb\b')
        c_area=findcol(r'die area', r'area.*mm', r'mm\s*(?:2|\^2|²)')
        c_bits=findcol(r'bits? per cell', r'cell type', r'tlc|qlc|mlc|slc', r'level')
        c_year=findcol(r'year', r'announced', r'introduced', r'date')
        c_manu=findcol(r'manufacturer', r'company', r'vendor', r'maker')
        c_prod=findcol(r'product', r'generation', r'technology', r'node', r'name')
        score=sum(bool(x) for x in [c_layers,c_cap,c_area,c_bits,c_year,c_manu,c_prod])
        if score < 3: continue
        for _,r in tab.iterrows():
            line=' | '.join(str(v) for v in r.tolist() if str(v)!='nan')
            if not re.search(r'nand|v-nand|flash|layer|tlc|qlc|mlc|slc|gb|tb|mm', line, re.I): continue
            layers=_v28_firstnum(r.get(c_layers)) if c_layers else None
            cap=_v28_firstnum(r.get(c_cap)) if c_cap else None
            if c_cap and re.search(r'\bTb|Tbit\b', str(r.get(c_cap)), re.I) and cap is not None: cap*=1000.0
            area=_v28_firstnum(r.get(c_area)) if c_area else None
            bits=None
            if c_bits:
                b=str(r.get(c_bits))
                bits=4.0 if re.search('QLC',b,re.I) else 3.0 if re.search('TLC',b,re.I) else 2.0 if re.search('MLC',b,re.I) else 1.0 if re.search('SLC',b,re.I) else _v28_firstnum(b)
            year=_v28_firstnum(r.get(c_year)) if c_year else None
            manu=str(r.get(c_manu))[:100] if c_manu else None
            prod=str(r.get(c_prod))[:180] if c_prod else line[:180]
            if layers or (cap and area) or bits:
                rows.append({'manufacturer':manu,'year':year,'generation_or_product':prod,'layers':layers,'die_capacity_Gb':cap,'die_area_mm2':area,'bits_per_cell':bits,'density_Gb_per_mm2':(cap/area if cap and area else None),'source_url':url,'provenance_line':line[:1200]})
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','provenance_line'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_spec_table','parser':'v28_nand_exact_rows','v28_nand_rows':len(df),'generated_csv':'data/generated/t44_nand_exact_rows_v28.csv','confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v28_optical_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    rows=[]
    try:
        for df in _v27_optical_frames(data,url):
            for _,r in df.iterrows(): rows.append(dict(r))
    except Exception: pass
    text=_v28_text(data,url,max_pages=240)
    lines=[re.sub(r'\s+',' ',x.strip()) for x in text.splitlines()]
    for i,l in enumerate(lines):
        ctx=' '.join(lines[max(0,i-2):min(len(lines),i+3)])[:2200]
        if not re.search(r'optical|photonic|silicon photonics|interconnect|link|i/o|transceiver|modulator|serdes', ctx, re.I): continue
        if not re.search(r'fJ\s*/\s*bit|pJ\s*/\s*bit|Gb/s|Gbps|Tb/s|Tbps|reach|bandwidth|\bmm\b|\bcm\b|\bm\b', ctx, re.I): continue
        epb=bw=reach=year=node=None
        m=re.search(r'(\d+(?:\.\d+)?)\s*(fJ|pJ)\s*/\s*bit',ctx,re.I)
        if m: epb=float(m.group(1))*(0.001 if m.group(2).lower()=='fj' else 1.0)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(Gb/s|Gbps|Tb/s|Tbps)',ctx,re.I)
        if m: bw=float(m.group(1))*(1000.0 if m.group(2).lower().startswith('t') else 1.0)
        m=re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|m)\b',ctx,re.I)
        if m: reach=float(m.group(1))*({'mm':1.0,'cm':10.0,'m':1000.0}[m.group(2).lower()])
        m=re.search(r'\b(20\d{2}|19\d{2})\b',ctx)
        if m: year=float(m.group(1))
        m=re.search(r'(\d+(?:\.\d+)?)\s*nm\b',ctx,re.I)
        if m: node=float(m.group(1))
        rows.append({'technology':'optical' if re.search(r'optical|photonic',ctx,re.I) else 'electrical_or_mixed','year':year,'energy_pJ_per_bit':epb,'bandwidth_Gbps':bw,'reach_mm':reach,'process_node_nm':node,'optical_vs_electrical':'optical' if re.search(r'optical|photonic',ctx,re.I) else 'electrical_or_mixed','source_url':url,'provenance_line':ctx})
    if not rows: return []
    df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','provenance_line'])
    df.attrs.update({'source_url':url,'evidence_tier':'secondary_exact_pdf_unit_text_table','parser':'v28_optical_interconnect_rows','v28_optical_rows':len(df),'generated_csv':'data/generated/t45_optical_interconnect_rows_v28.csv','confirmation_allowed':False,'falsification_allowed':False})
    return [df]


def _v28_fusion_frames(data: bytes, url: str) -> List[pd.DataFrame]:
    rows=[]; figs=[]
    try:
        for df in _v27_fusion_frames(data,url):
            if 'v27_fusion_unit_rows' in df.attrs:
                for _,r in df.iterrows(): rows.append(dict(r))
            elif 'v27_fusion_figure_candidate_pages' in df.attrs:
                for _,r in df.iterrows(): figs.append(dict(r))
    except Exception: pass
    text=_v28_text(data,url,max_pages=320)
    lines=[re.sub(r'\s+',' ',x.strip()) for x in text.splitlines()]
    terms=r'ELM energy|W_ELM|E_ELM|Wped|Pped|dW|ΔW|delta W|pedestal energy|pedestal pressure|energy loss|DIII-D|JET|ITER|W7-X|ASDEX|AUG|tokamak|RMP|H98|q95|tau[_\s-]?E'
    units=r'MJ|kJ|\bJ\b|MW|kPa|Pa|ms|Hz|kHz|%|MA|keV|eV|10\^19|m\^-?3'
    for i,l in enumerate(lines):
        ctx=' '.join(lines[max(0,i-2):min(len(lines),i+3)])[:2600]
        if re.search(terms,ctx,re.I) and re.search(units,ctx,re.I) and re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',ctx):
            rows.append({'source_url':url,'line_index':i,'context_text':ctx,'numeric_values':';'.join(re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',ctx)[:80]),'units_found':';'.join(re.findall(units,ctx,re.I)[:80])})
        if re.search(r'fig\.?|figure|table',l,re.I) and re.search(r'ELM|pedestal|energy loss|Wped|Pped|dW|ΔW',ctx,re.I):
            figs.append({'source_url':url,'line_index':i,'figure_candidate_text':ctx})
    out=[]
    if rows:
        df=pd.DataFrame(rows).drop_duplicates(subset=['source_url','context_text'])
        df.attrs.update({'source_url':url,'evidence_tier':'secondary_auto_pdf_text_table','parser':'v28_fusion_numeric_context','v28_fusion_unit_rows':len(df),'generated_csv':'data/generated/fusion_secondary_rows_v28.csv','confirmation_allowed':False,'falsification_allowed':False})
        out.append(df)
    if figs:
        df=pd.DataFrame(figs).drop_duplicates(subset=['source_url','figure_candidate_text'])
        df.attrs.update({'source_url':url,'evidence_tier':'secondary_figure_page_candidate','parser':'v28_fusion_figure_context','v28_fusion_figure_candidate_pages':len(df),'confirmation_allowed':False,'falsification_allowed':False})
        out.append(df)
    return out

try:
    _v28_extract_ref=extract_frames_from_artifact  # type: ignore[name-defined]
    def extract_frames_from_artifact(data: bytes, url: str, meta: Dict[str,Any], cache_dir: Path):  # type: ignore[override]
        frames, diag=_v28_extract_ref(data,url,meta,cache_dir)
        us=str(url).lower()
        def add(key, xs, parser):
            if not xs: return
            frames.extend(xs)
            diag[key]=sum(int(x.attrs.get(key,len(x))) for x in xs)
            diag.setdefault('extractors_tried',[]).append(parser)
        try:
            if any(x in us for x in ['nand','flash','wikichip','techinsights','samsung','micron','hynix','kioxia','toshiba','western-digital']):
                add('v28_nand_rows', _v28_nand_frames(data,url), 'v28_nand_exact_rows')
        except Exception as e: diag['v28_nand_error']=repr(e)
        try:
            if any(x in us for x in ['optical','photonic','irds','interconnect','pj','fj','silicon-photonics','transceiver','serdes']):
                add('v28_optical_rows', _v28_optical_frames(data,url), 'v28_optical_interconnect_rows')
        except Exception as e: diag['v28_optical_error']=repr(e)
        try:
            if any(x in us for x in ['elm','pedestal','fusion','tokamak','diii','jet','iter','w7-x','asdex','rmp','h-mode']):
                xs=_v28_fusion_frames(data,url)
                frames.extend(xs)
                diag['v28_fusion_unit_rows']=sum(int(x.attrs.get('v28_fusion_unit_rows',0)) for x in xs)
                diag['v28_fusion_figure_candidate_pages']=sum(int(x.attrs.get('v28_fusion_figure_candidate_pages',0)) for x in xs)
                if xs: diag.setdefault('extractors_tried',[]).append('v28_fusion_numeric_context')
        except Exception as e: diag['v28_fusion_error']=repr(e)
        return frames, diag
except Exception:
    pass

try:
    _v28_seed_ref=additional_seed_sources_v11
    def additional_seed_sources_v11(test_id: str):  # type: ignore[override]
        base=list(_v28_seed_ref(test_id)); extra=[]
        if test_id=='T44':
            extra += [
                {'url':'https://en.wikichip.org/wiki/3d_nand','label':'v28 WikiChip 3D NAND exact parser','reason':'v28_t44_confirm_parser','tier':'html_table_candidate'},
                {'url':'https://en.wikichip.org/wiki/flash_memory','label':'v28 WikiChip flash memory exact parser','reason':'v28_t44_confirm_parser','tier':'html_table_candidate'},
                {'url':'https://zenodo.org/api/records/?q=%223D%20NAND%22%20%22die%20area%22%20%22layers%22%20%22bits%20per%20cell%22&size=25','label':'v28 3D NAND die-area layers query','reason':'v28_t44_confirm_query','tier':'metadata_only'},
            ]
        if test_id=='T45':
            extra += [
                {'url':'https://zenodo.org/api/records/?q=%22pJ%2Fbit%22%20%22Gb%2Fs%22%20%22optical%20interconnect%22%20reach&size=25','label':'v28 optical interconnect pJ/bit Gb/s reach query','reason':'v28_t45_confirm_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=%22fJ%2Fbit%22%20%22silicon%20photonics%22%20%22Tb%2Fs%22&size=25','label':'v28 silicon photonics fJ/bit Tb/s query','reason':'v28_t45_confirm_query','tier':'metadata_only'},
            ]
        if test_id in {'T26','T27','T28','T29','T30'}:
            extra += [
                {'url':'https://zenodo.org/api/records/?q=%22W_ELM%22%20%22Pped%22%20%22shot%22%20tokamak&size=25','label':'v28 fusion W_ELM Pped shot query','reason':'v28_fusion_confirm_query','tier':'metadata_only'},
                {'url':'https://zenodo.org/api/records/?q=%22ELM%20energy%22%20%22Wped%22%20%22dW%22%20JET&size=25','label':'v28 fusion ELM Wped dW JET query','reason':'v28_fusion_confirm_query','tier':'metadata_only'},
            ]
        return base+extra
except Exception:
    pass
