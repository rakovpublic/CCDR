#!/usr/bin/env python3
"""EL3 public node-density test.

EL3: volume -> area scaling crossover at L ~ 10-100 nm for silicon.

v2.8 improvements:
  * separates primary physical-pitch/SRAM CSV rows from secondary advertised-node placeholders;
  * exposes EL3c primary vs secondary evidence ladders so primary-only data_missing is explicit.

v2.6 improvements on top of v2.5:
  * adds explicit volume-to-area transition scoring, so a proxy cannot pass just because the small-node side is area-like;
  * reports density_kind counts and keeps foundry-only decisions strict.

v2.5 improvements:
  * keeps the v2.4 corrected small-node / large-node segment labels;
  * reports historical raw rows and historical node-median rows separately;
  * filters EL3b to foundry-standard-cell/reported-density rows only;
  * adds constrained 10-100 nm hypothesis scoring, rolling local slopes, and
    deterministic subsample robustness so one breakpoint cannot dominate the verdict.

No manual files are required. The small built-in fallback table is not accepted
blindly: a row is added only when the numeric density string is present in the
corresponding downloaded public HTML page.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from _public_test_common import (
    csv_rows_from_url,
    download_to_cache,
    float_or_none,
    piecewise_log_slope,
    read_text_any,
    source_records,
    write_json,
)
from _public_improvements import evidence_for_status

SOURCES = {
    "Barentsen tech-progress transistor counts CSV": "https://raw.githubusercontent.com/barentsen/tech-progress-data/master/data/transistor-counts/transistor-counts.csv",
    "Our World in Data transistor microprocessor CSV": "https://ourworldindata.org/grapher/transistors-per-microprocessor.csv",
    "Wikipedia 22 nm process page": "https://en.wikipedia.org/wiki/22_nm_process",
    "Wikipedia 14 nm process page": "https://en.wikipedia.org/wiki/14_nm_process",
    "Wikipedia 10 nm process page": "https://en.wikipedia.org/wiki/10_nm_process",
    "Wikipedia 7 nm process page": "https://en.wikipedia.org/wiki/7_nm_process",
    "Wikipedia 5 nm process page": "https://en.wikipedia.org/wiki/5_nm_process",
    "Wikipedia 3 nm process page": "https://en.wikipedia.org/wiki/3_nm_process",
}

# v2.7: secondary public physical-pitch/SRAM-bitcell proxy rows.
# Rows are accepted only when verification tokens occur in the downloaded process-node page.
PHYSICAL_PITCH_CANDIDATES: Dict[str, List[Dict[str, Any]]] = {
    "Wikipedia 22 nm process page": [
        {"company": "Intel", "process_name": "22nm", "advertised_node_nm": 22.0, "feature_kind": "physical_pitch_or_node_proxy", "physical_feature_nm": 22.0, "verify": ["22"]},
    ],
    "Wikipedia 14 nm process page": [
        {"company": "Intel", "process_name": "14nm", "advertised_node_nm": 14.0, "feature_kind": "physical_pitch_or_node_proxy", "physical_feature_nm": 14.0, "verify": ["14"]},
    ],
    "Wikipedia 10 nm process page": [
        {"company": "Intel", "process_name": "10nm", "advertised_node_nm": 10.0, "feature_kind": "physical_pitch_or_node_proxy", "physical_feature_nm": 10.0, "verify": ["10"]},
    ],
    "Wikipedia 7 nm process page": [
        {"company": "TSMC", "process_name": "N7", "advertised_node_nm": 7.0, "feature_kind": "physical_pitch_or_node_proxy", "physical_feature_nm": 7.0, "verify": ["7"]},
    ],
    "Wikipedia 5 nm process page": [
        {"company": "TSMC", "process_name": "N5", "advertised_node_nm": 5.0, "feature_kind": "physical_pitch_or_node_proxy", "physical_feature_nm": 5.0, "verify": ["5"]},
    ],
    "Wikipedia 3 nm process page": [
        {"company": "TSMC", "process_name": "N3", "advertised_node_nm": 3.0, "feature_kind": "physical_pitch_or_node_proxy", "physical_feature_nm": 3.0, "verify": ["3"]},
    ],
}

# Public-HTML-verified fallback rows. These values are from the downloaded process-node
# pages; code only accepts a row if the exact density string is present in the HTML.
# This avoids brittle dataframe parsing while still keeping data public-source-gated.
FOUNDRY_DENSITY_CANDIDATES: Dict[str, List[Dict[str, Any]]] = {
    "Wikipedia 22 nm process page": [
        {"company": "Intel", "process_name": "22nm", "nominal_node_nm": 22.0, "density_MTr_per_mm2": 16.5, "verify": ["16.5"]},
    ],
    "Wikipedia 14 nm process page": [
        {"company": "Samsung", "process_name": "14LPE", "nominal_node_nm": 14.0, "density_MTr_per_mm2": 32.94, "verify": ["32.94"]},
        {"company": "Samsung", "process_name": "14LPP", "nominal_node_nm": 14.0, "density_MTr_per_mm2": 54.38, "verify": ["54.38"]},
        {"company": "Samsung", "process_name": "11LPP", "nominal_node_nm": 11.0, "density_MTr_per_mm2": 28.88, "verify": ["28.88"]},
        {"company": "TSMC", "process_name": "16nm", "nominal_node_nm": 16.0, "density_MTr_per_mm2": 33.8, "verify": ["33.8"]},
        {"company": "TSMC", "process_name": "12nm", "nominal_node_nm": 12.0, "density_MTr_per_mm2": 44.67, "verify": ["44.67"]},
        {"company": "Intel", "process_name": "14nm", "nominal_node_nm": 14.0, "density_MTr_per_mm2": 37.5, "verify": ["37.5"]},
    ],
    "Wikipedia 10 nm process page": [
        {"company": "Samsung", "process_name": "10LPE", "nominal_node_nm": 10.0, "density_MTr_per_mm2": 51.82, "verify": ["51.82"]},
        {"company": "Samsung", "process_name": "10LPP", "nominal_node_nm": 10.0, "density_MTr_per_mm2": 61.18, "verify": ["61.18"]},
        {"company": "Samsung", "process_name": "8LPP", "nominal_node_nm": 8.0, "density_MTr_per_mm2": 55.75, "verify": ["55.75"]},
        {"company": "Samsung", "process_name": "8LPU", "nominal_node_nm": 8.0, "density_MTr_per_mm2": 52.51, "verify": ["52.51"]},
        {"company": "TSMC", "process_name": "10FF", "nominal_node_nm": 10.0, "density_MTr_per_mm2": 60.41, "verify": ["60.41"]},
        # Intel 10nm is sometimes given as 100.76 by Intel's weighted logic formula;
        # the page also contains 60.41 for a different comparison cell. Keep both if visible.
        {"company": "Intel", "process_name": "10nm formula", "nominal_node_nm": 10.0, "density_MTr_per_mm2": 100.76, "verify": ["100.76"]},
    ],
    "Wikipedia 7 nm process page": [
        {"company": "Samsung", "process_name": "7LPP", "nominal_node_nm": 7.0, "density_MTr_per_mm2": 97.835, "verify": ["95.08", "100.59"], "range": [95.08, 100.59]},
        {"company": "TSMC", "process_name": "N7", "nominal_node_nm": 7.0, "density_MTr_per_mm2": 93.85, "verify": ["91.2", "96.5"], "range": [91.2, 96.5]},
        {"company": "TSMC", "process_name": "N7P", "nominal_node_nm": 7.0, "density_MTr_per_mm2": 113.9, "verify": ["113.9"]},
        {"company": "TSMC", "process_name": "N7+", "nominal_node_nm": 7.0, "density_MTr_per_mm2": 114.2, "verify": ["114.2"]},
        {"company": "Intel", "process_name": "Intel 7", "nominal_node_nm": 7.0, "density_MTr_per_mm2": 89.0, "verify": ["89"]},
    ],
    "Wikipedia 5 nm process page": [
        {"company": "Samsung", "process_name": "5LPE", "nominal_node_nm": 5.0, "density_MTr_per_mm2": 126.9, "verify": ["126.9"]},
        {"company": "TSMC", "process_name": "N5", "nominal_node_nm": 5.0, "density_MTr_per_mm2": 138.2, "verify": ["138.2"]},
        {"company": "SMIC", "process_name": "N+3", "nominal_node_nm": 5.0, "density_MTr_per_mm2": 120.0, "verify": ["120"]},
        {"company": "Samsung", "process_name": "SF4E", "nominal_node_nm": 4.0, "density_MTr_per_mm2": 137.0, "verify": ["137"]},
        {"company": "TSMC", "process_name": "N4", "nominal_node_nm": 4.0, "density_MTr_per_mm2": 143.7, "verify": ["143.7"]},
        {"company": "Intel", "process_name": "Intel 4", "nominal_node_nm": 4.0, "density_MTr_per_mm2": 126.61, "verify": ["126.61"]},
    ],
    "Wikipedia 3 nm process page": [
        {"company": "Samsung", "process_name": "3GAE/SF3E", "nominal_node_nm": 3.0, "density_MTr_per_mm2": 150.0, "verify": ["150"]},
        {"company": "Samsung", "process_name": "3GAP/SF3", "nominal_node_nm": 3.0, "density_MTr_per_mm2": 190.0, "verify": ["190"]},
        {"company": "TSMC", "process_name": "N3", "nominal_node_nm": 3.0, "density_MTr_per_mm2": 197.0, "verify": ["197"]},
        {"company": "TSMC", "process_name": "N3E", "nominal_node_nm": 3.0, "density_MTr_per_mm2": 216.0, "verify": ["216"]},
        {"company": "TSMC", "process_name": "N3P", "nominal_node_nm": 3.0, "density_MTr_per_mm2": 224.0, "verify": ["224"]},
        {"company": "Intel", "process_name": "Intel 3", "nominal_node_nm": 3.0, "density_MTr_per_mm2": 143.37, "verify": ["143.37"]},
    ],
}


def html_snippet_around_tokens(html: str, tokens: List[str], radius: int = 140) -> Optional[str]:
    """Return a short provenance snippet around the first verification token."""
    if not html or not tokens:
        return None
    compact_html = html.replace(",", "")
    for token in tokens:
        idx = compact_html.find(str(token))
        if idx >= 0:
            start = max(0, idx - radius)
            end = min(len(compact_html), idx + len(str(token)) + radius)
            snippet = compact_html[start:end]
            snippet = re.sub(r"<[^>]+>", " ", snippet)
            snippet = re.sub(r"\s+", " ", snippet).strip()
            return snippet[:320]
    return None


def is_area_like_alpha(alpha: Optional[float]) -> bool:
    return alpha is not None and 1.2 <= alpha <= 2.8


def closer_to_area_than_volume(alpha: Optional[float]) -> bool:
    return alpha is not None and abs(alpha - 2.0) < abs(alpha - 3.0)


def transition_score_from_alphas(small_alpha: Optional[float], large_alpha: Optional[float]) -> Dict[str, Any]:
    """Score whether a fit resembles a volume-to-area crossover rather than only
    an area-like slope.  EL3 should not pass merely because the advanced-node
    segment has alpha≈2; the larger-node segment should be steeper or at least
    clearly not flatter.
    """
    if small_alpha is None or large_alpha is None:
        return {
            "alpha_drop_large_minus_small": None,
            "large_segment_steeper_than_small": False,
            "crossover_like_alpha_drop": False,
            "strong_volume_to_area_alpha_drop": False,
            "large_segment_volume_like": False,
        }
    drop = float(large_alpha - small_alpha)
    return {
        "alpha_drop_large_minus_small": drop,
        "large_segment_steeper_than_small": bool(drop > 0.0),
        "crossover_like_alpha_drop": bool(drop >= 0.25 and large_alpha >= 1.5),
        "strong_volume_to_area_alpha_drop": bool(drop >= 0.45 and large_alpha >= 2.2 and is_area_like_alpha(small_alpha)),
        "large_segment_volume_like": bool(2.2 <= large_alpha <= 3.6),
    }



def add_verified_physical_pitch_proxy_rows(html_texts: Dict[str, str], density_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v2.7 EL3c: physical-pitch/SRAM proxy branch.

    Public process pages rarely expose consistent CPP/MMP/SRAM values in a clean
    machine-readable table. This branch therefore uses verified advertised-node
    physical-feature placeholders only as a low-confidence physical-scale proxy,
    paired to the median foundry density at the same node when available.
    """
    by_node: Dict[float, List[Dict[str, Any]]] = {}
    for r in density_rows:
        if str(r.get("density_kind", "")).startswith("foundry_standard_cell"):
            node = float(r.get("advertised_node_nm") or r.get("node_nm") or 0)
            if node > 0:
                by_node.setdefault(node, []).append(r)
    out: List[Dict[str, Any]] = []
    for label, candidates in PHYSICAL_PITCH_CANDIDATES.items():
        html = html_texts.get(label, "").replace(",", "")
        if not html:
            continue
        for cand in candidates:
            if not all(str(tok) in html for tok in cand.get("verify", [])):
                continue
            node = float(cand["advertised_node_nm"])
            source_density_rows = by_node.get(node, [])
            if not source_density_rows:
                continue
            dens = sorted(float(r["density_MTr_per_mm2"]) for r in source_density_rows)
            med = dens[len(dens)//2] if len(dens)%2 else 0.5*(dens[len(dens)//2-1]+dens[len(dens)//2])
            out.append({
                "source": label,
                "source_url": SOURCES.get(label),
                "company": cand.get("company"),
                "processor": cand.get("process_name"),
                "node_nm": float(cand["physical_feature_nm"]),
                "advertised_node_nm": node,
                "physical_feature_nm": float(cand["physical_feature_nm"]),
                "feature_kind": cand.get("feature_kind"),
                "density_MTr_per_mm2": float(med),
                "density_kind": "physical_pitch_proxy_paired_to_foundry_density",
                "row_type": "physical_pitch_proxy",
                "confidence": "low_secondary_proxy",
                "data_role": "EL3c_physical_pitch_proxy_not_direct_measurement",
                "evidence_tokens": cand.get("verify", []),
                "evidence_snippet": html_snippet_around_tokens(html_texts.get(label, ""), [str(t) for t in cand.get("verify", [])]),
            })
    return out



def load_primary_physical_pitch_csv_rows(
    urls: List[str],
    cache: Path,
    foundry_density_rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Any]]:
    """Load optional public primary/semi-primary physical-pitch rows.

    Expected public CSV columns are flexible:
      advertised_node_nm/node_nm, physical_feature_nm/cpp_nm/mmp_nm/metal_pitch_nm,
      and optionally density_MTr_per_mm2. If density is absent, the row may be
      paired to foundry density at the same advertised node, but that is flagged.
    """
    by_node: Dict[float, List[float]] = {}
    for r in foundry_density_rows:
        if str(r.get("density_kind", "")).startswith("foundry_standard_cell"):
            node = float(r.get("advertised_node_nm") or r.get("node_nm") or 0)
            if node > 0 and r.get("density_MTr_per_mm2"):
                by_node.setdefault(node, []).append(float(r["density_MTr_per_mm2"]))

    rows_out: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []
    sources: List[Any] = []
    for i, url in enumerate(urls or []):
        rows, src = csv_rows_from_url(f"el3_primary_physical_pitch_csv_{i}", url, cache)
        sources.append(src)
        usable = 0
        for row in rows:
            lower = {str(k).strip().lower(): v for k, v in row.items()}
            def pick(*names: str) -> Optional[float]:
                for name in names:
                    if name in lower:
                        val = float_or_none(lower.get(name))
                        if val is not None:
                            return val
                return None
            advertised = pick("advertised_node_nm", "node_nm", "process_node_nm", "node")
            physical = pick("physical_feature_nm", "cpp_nm", "contacted_poly_pitch_nm", "mmp_nm", "metal_pitch_nm", "minimum_metal_pitch_nm", "gate_pitch_nm")
            density = pick("density_mtr_per_mm2", "density_mtr/mm2", "standard_cell_density_mtr_per_mm2", "transistor_density_mtr_per_mm2")
            feature_kind = "primary_physical_pitch"
            if physical is None:
                bitcell = pick("sram_bitcell_um2", "sram_cell_um2", "bitcell_um2")
                if bitcell is not None and bitcell > 0:
                    physical = (bitcell ** 0.5) * 1000.0
                    feature_kind = "sram_bitcell_sqrt_area_nm"
            if advertised is None or physical is None:
                continue
            density_source_kind = "primary_csv_density"
            if density is None:
                candidates = by_node.get(float(advertised), [])
                if candidates:
                    candidates = sorted(candidates)
                    mid = len(candidates) // 2
                    density = candidates[mid] if len(candidates) % 2 else 0.5 * (candidates[mid - 1] + candidates[mid])
                    density_source_kind = "paired_foundry_density_same_advertised_node"
            if density is None or density <= 0 or physical <= 0:
                continue
            rows_out.append({
                "source": f"public_primary_physical_pitch_csv_{i}",
                "source_url": url,
                "company": lower.get("company") or lower.get("manufacturer") or lower.get("foundry"),
                "processor": lower.get("process") or lower.get("process_name") or lower.get("node_name"),
                "node_nm": float(physical),
                "advertised_node_nm": float(advertised),
                "physical_feature_nm": float(physical),
                "feature_kind": feature_kind,
                "density_MTr_per_mm2": float(density),
                "density_kind": "primary_physical_pitch_with_density" if density_source_kind == "primary_csv_density" else "primary_physical_pitch_paired_to_foundry_density",
                "density_source_kind": density_source_kind,
                "row_type": "primary_physical_pitch_table",
                "confidence": "primary_or_user_supplied_public_csv",
                "is_primary_source": True,
                "data_role": "EL3c_primary_physical_pitch_proxy",
            })
            usable += 1
        notes.append({"url": url, "rows_seen": len(rows), "usable_rows": usable, "download_ok": bool(src and src.ok)})
    return rows_out, notes, sources

def status_with_ladder(status: str) -> Dict[str, Any]:
    return {"status": status, "evidence_ladder": evidence_for_status(status)}

def parse_node_nm(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).replace("μ", "u").replace("µ", "u").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(nm|nanometer|nanometre)", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(um|micrometer|micrometre|micro)", s)
    if m:
        return float(m.group(1)) * 1000.0
    m = re.search(r"^\s*(\d+(?:\.\d+)?)\s*$", s)
    if m:
        return float(m.group(1))
    return None


def parse_area_mm2(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def parse_transistors(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    mult = 1.0
    if re.search(r"\bbillion\b|\bB\b", s, re.I):
        mult = 1e9
    elif re.search(r"\bmillion\b|\bM\b", s, re.I):
        mult = 1e6
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) * mult if m else None


def parse_density_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).replace("\u2013", "-").replace("\u2212", "-").replace(",", "")
    vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    vals = [v for v in vals if 0.1 <= v <= 1000]
    if not vals:
        return None
    if len(vals) >= 2 and "-" in s:
        return float(sum(vals[:2]) / 2.0)
    return float(vals[0])


def load_barentsen_csv(text: str) -> List[Dict[str, Any]]:
    rows = list(csv.DictReader(io.StringIO(text)))
    out: List[Dict[str, Any]] = []
    for r in rows:
        node = parse_node_nm(r.get("process") or r.get("Process"))
        area = parse_area_mm2(r.get("area") or r.get("Area") or r.get("Area mm2"))
        trans = parse_transistors(r.get("transistors") or r.get("Transistors"))
        year = float_or_none(r.get("year") or r.get("Year"))
        processor = r.get("processor") or r.get("Processor") or r.get("Entity") or ""
        if node and area and trans and node > 0 and area > 0 and trans > 0:
            out.append(
                {
                    "source": "barentsen_tech_progress",
                    "processor": processor,
                    "year": year,
                    "node_nm": node,
                    "advertised_node_nm": node,
                    "area_mm2": area,
                    "transistors": trans,
                    "density_MTr_per_mm2": trans / area / 1e6,
                    "density_kind": "realized_full_chip_density",
                    "row_type": "historical_chip",
                    "confidence": "medium",
                    "is_primary_source": False,
                }
            )
    return out


def _flatten_col(c: Any) -> str:
    if isinstance(c, tuple):
        return " ".join(str(x) for x in c if str(x).lower() != "nan")
    return str(c)


def scrape_wikipedia_density_tables(html_texts: Dict[str, str]) -> List[Dict[str, Any]]:
    """Best-effort extraction of process-node density rows from downloaded HTML tables."""
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for label, html in html_texts.items():
        if not label.startswith("Wikipedia"):
            continue
        page_node = parse_node_nm(label)
        try:
            tables = pd.read_html(io.StringIO(html))
        except Exception:
            continue
        for ti, df in enumerate(tables):
            # Flatten awkward multi-index columns from Wikipedia tables.
            df = df.copy()
            df.columns = [_flatten_col(c) for c in df.columns]
            str_df = df.astype(str)
            for ri, row in str_df.iterrows():
                joined = " ".join(row.values).lower().replace(" ", "")
                if "transistordensity" not in joined and not ("transistor" in joined and "density" in joined):
                    continue
                process_row = None
                for back in range(max(0, int(ri) - 5), int(ri)):
                    joined_back = " ".join(str_df.iloc[back].values).lower()
                    if "process name" in joined_back:
                        process_row = str_df.iloc[back]
                        break
                for col in df.columns:
                    val = parse_density_value(row[col])
                    if val is None:
                        continue
                    if process_row is not None:
                        process = str(process_row[col])
                    else:
                        process = str(col)
                    # Skip label cells rather than numeric density cells.
                    if "density" in process.lower() or process.lower() in {"nan", "unknown"}:
                        process = str(col)
                    node = parse_node_nm(process) or page_node
                    out.append(
                        {
                            "source": label,
                            "processor": process,
                            "company": str(col),
                            "year": None,
                            "node_nm": node,
                            "advertised_node_nm": node,
                            "area_mm2": None,
                            "transistors": None,
                            "density_MTr_per_mm2": val,
                            "density_kind": "foundry_standard_cell_or_reported_transistor_density",
                            "table_index": ti,
                            "row_type": "foundry_node_table",
                            "extraction_method": "pandas_read_html",
                            "confidence": "medium_low",
                            "is_primary_source": False,
                        }
                    )
    return out


def add_verified_foundry_fallback_rows(html_texts: Dict[str, str], existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add modern foundry-node rows only when density strings are visible in downloaded HTML."""
    existing_keys = {
        (r.get("source"), r.get("processor"), round(float(r.get("node_nm") or 0), 3), round(float(r.get("density_MTr_per_mm2") or 0), 3))
        for r in existing
    }
    out: List[Dict[str, Any]] = list(existing)
    for label, candidates in FOUNDRY_DENSITY_CANDIDATES.items():
        html = html_texts.get(label, "")
        if not html:
            continue
        compact = html.replace(",", "")
        for cand in candidates:
            verify_tokens = [str(v) for v in cand.get("verify", [])]
            if not all(token in compact for token in verify_tokens):
                continue
            density = float(cand["density_MTr_per_mm2"])
            node = float(cand["nominal_node_nm"])
            key = (label, cand["process_name"], round(node, 3), round(density, 3))
            if key in existing_keys:
                continue
            existing_keys.add(key)
            out.append(
                {
                    "source": label,
                    "source_url": SOURCES.get(label),
                    "processor": cand["process_name"],
                    "company": cand["company"],
                    "year": None,
                    "node_nm": node,
                    "advertised_node_nm": node,
                    "area_mm2": None,
                    "transistors": None,
                    "density_MTr_per_mm2": density,
                    "density_kind": "foundry_standard_cell_or_reported_transistor_density",
                    "row_type": "foundry_node_table",
                    "extraction_method": "public_html_verified_fallback",
                    "density_range_MTr_per_mm2": cand.get("range"),
                    "evidence_tokens": verify_tokens,
                    "evidence_snippet": html_snippet_around_tokens(html, verify_tokens),
                    "confidence": "medium",
                    "is_primary_source": False,
                }
            )
    return out


def write_foundry_csv(rows: List[Dict[str, Any]], path: Path) -> Optional[str]:
    foundry = [r for r in rows if r.get("row_type") == "foundry_node_table"]
    if not foundry:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["source", "source_url", "company", "processor", "node_nm", "advertised_node_nm", "density_MTr_per_mm2", "density_kind", "density_range_MTr_per_mm2", "extraction_method", "confidence", "is_primary_source", "evidence_tokens", "evidence_snippet"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in foundry:
            w.writerow({c: r.get(c) for c in cols})
    return str(path)


def aggregate_density_by_node(rows: List[Dict[str, Any]], round_digits: int = 4) -> List[Dict[str, Any]]:
    """Collapse multiple rows at the same nominal node to a median-density row.

    Foundry standard-cell tables often contain many variants at the same marketing
    node (N7, N7P, N7+, Intel 7, etc.). Treating every variant as an independent
    x value biases piecewise fits and creates poorly conditioned split candidates.
    """
    groups: Dict[float, List[Dict[str, Any]]] = {}
    for r in rows:
        node = float(r.get("node_nm") or 0)
        dens = float(r.get("density_MTr_per_mm2") or 0)
        if node <= 0 or dens <= 0:
            continue
        groups.setdefault(round(node, round_digits), []).append(r)
    out: List[Dict[str, Any]] = []
    for node, vals in sorted(groups.items()):
        densities = sorted(float(v["density_MTr_per_mm2"]) for v in vals)
        mid = len(densities) // 2
        if len(densities) % 2:
            median = densities[mid]
        else:
            median = 0.5 * (densities[mid - 1] + densities[mid])
        out.append(
            {
                "source": "node_median",
                "processor": f"node_{node:g}_median",
                "company": "multiple" if len(vals) > 1 else vals[0].get("company"),
                "node_nm": float(node),
                "advertised_node_nm": float(node),
                "density_MTr_per_mm2": float(median),
                "density_kind": vals[0].get("density_kind"),
                "row_type": "node_median",
                "n_rows_collapsed": len(vals),
                "density_min_MTr_per_mm2": min(densities),
                "density_max_MTr_per_mm2": max(densities),
                "confidence": "node_median_from_public_proxy_rows",
            }
        )
    return out


def band_slope(rows: List[Dict[str, Any]], lo: float, hi: float, aggregate_by_node: bool = False) -> Dict[str, Any]:
    use_rows = aggregate_density_by_node(rows) if aggregate_by_node else rows
    sub = [r for r in use_rows if lo <= r["node_nm"] < hi and r["density_MTr_per_mm2"] > 0]
    if len(sub) < 3 or len({round(float(r["node_nm"]), 4) for r in sub}) < 3:
        return {"n": len(sub), "alpha": None}
    x = np.log10([r["node_nm"] for r in sub])
    y = np.log10([r["density_MTr_per_mm2"] for r in sub])
    slope, intercept = np.polyfit(x, y, 1)
    return {"n": len(sub), "alpha": float(-slope), "node_range_nm": [lo, hi], "aggregate_by_node": bool(aggregate_by_node)}


def fit_decision(
    rows: List[Dict[str, Any]],
    min_points: int = 5,
    aggregate_by_node: bool = False,
    observable_label: str = "unspecified",
) -> Dict[str, Any]:
    """Fit EL3 with both a global breakpoint and a constrained 10-100 nm hypothesis fit.

    v2.4 avoids a false binary pass/fail from a single global breakpoint. A proxy can be
    marked `window_compatible_proxy` if a 10-100 nm constrained fit is area-like and is
    statistically competitive with the global optimum.
    """
    fit_rows = aggregate_density_by_node(rows) if aggregate_by_node else list(rows)
    xs = [r["node_nm"] for r in fit_rows]
    ys = [r["density_MTr_per_mm2"] for r in fit_rows]
    global_fit = piecewise_log_slope(
        xs,
        ys,
        min_points=min_points,
        min_unique_x_per_segment=3,
    )
    hypothesis_fit = piecewise_log_slope(
        xs,
        ys,
        min_points=min_points,
        min_unique_x_per_segment=3,
        break_min_nm=10,
        break_max_nm=100,
    )

    def fit_flags(fit: Dict[str, Any]) -> Dict[str, Any]:
        br = fit.get("break_node_nm")
        small_alpha = fit.get("small_node_alpha", fit.get("post_alpha_small_nodes"))
        large_alpha = fit.get("large_node_alpha", fit.get("pre_alpha_large_nodes"))
        in_window = br is not None and 10 <= br <= 100
        area_like = is_area_like_alpha(small_alpha)
        closer_area = closer_to_area_than_volume(small_alpha)
        transition = transition_score_from_alphas(small_alpha, large_alpha)
        return {
            "break_in_10_100nm_window": bool(in_window),
            "small_node_slope_area_like": bool(area_like),
            "small_node_slope_closer_to_area_than_volume": bool(closer_area),
            "small_node_alpha": small_alpha,
            "large_node_alpha": large_alpha,
            **transition,
        }

    global_flags = fit_flags(global_fit)
    hypothesis_flags = fit_flags(hypothesis_fit)
    global_sse = global_fit.get("sse")
    hyp_sse = hypothesis_fit.get("sse")
    n = global_fit.get("n") or len(fit_rows)
    if global_sse is not None and hyp_sse is not None and n:
        sse_penalty_frac = float((hyp_sse - global_sse) / max(global_sse, 1e-12))
        rmse_penalty_log10 = float(math.sqrt(hyp_sse / n) - math.sqrt(global_sse / n))
    else:
        sse_penalty_frac = None
        rmse_penalty_log10 = None

    global_pass = all([
        global_flags["break_in_10_100nm_window"],
        global_flags["small_node_slope_area_like"],
        global_flags["small_node_slope_closer_to_area_than_volume"],
        global_flags["crossover_like_alpha_drop"],
    ])
    hypothesis_competitive = (
        hypothesis_flags["break_in_10_100nm_window"]
        and hypothesis_flags["small_node_slope_area_like"]
        and hypothesis_flags["small_node_slope_closer_to_area_than_volume"]
        and hypothesis_flags["crossover_like_alpha_drop"]
        and rmse_penalty_log10 is not None
        and rmse_penalty_log10 <= 0.035
    )
    if global_pass:
        status = "pass_proxy"
        interpretation = "Global best two-slope break lies in 10-100 nm, the advanced-node segment is area-like, and the larger-node segment is steeper."
    elif hypothesis_competitive:
        status = "window_compatible_proxy"
        interpretation = "Global best break is outside 10-100 nm, but the constrained 10-100 nm hypothesis fit is area-like, crossover-like, and nearly as good in log-RMSE."
    else:
        status = "fail_or_inconclusive_proxy"
        interpretation = "No stable area-like 10-100 nm crossover is supported by this observable under current public proxy rules."

    return {
        "observable_label": observable_label,
        "fit_rows_used": len(fit_rows),
        "aggregate_by_node": bool(aggregate_by_node),
        "node_median_rows": fit_rows[:30] if aggregate_by_node else None,
        "fit": global_fit,
        "global_fit": global_fit,
        "hypothesis_fit_10_100nm": hypothesis_fit,
        "hypothesis_penalty_vs_global": {
            "sse_penalty_frac": sse_penalty_frac,
            "rmse_penalty_log10": rmse_penalty_log10,
            "competitive_threshold_rmse_log10": 0.035,
        },
        "decision": {
            "status": status,
            "interpretation": interpretation,
            "global_fit_flags": global_flags,
            "hypothesis_fit_flags": hypothesis_flags,
            "break_in_10_100nm_window": bool(global_flags["break_in_10_100nm_window"]),
            "small_node_slope_area_like": bool(global_flags["small_node_slope_area_like"]),
            "small_node_slope_closer_to_area_than_volume": bool(global_flags["small_node_slope_closer_to_area_than_volume"]),
            "hypothesis_10_100nm_competitive": bool(hypothesis_competitive),
            "uses_corrected_segment_labels_v2_4": True,
        },
    }



def _decision_supportive(status: Optional[str]) -> bool:
    return status in {"pass_proxy", "window_compatible_proxy", "weak_window_compatible_proxy"}


def local_slope_profile(rows: List[Dict[str, Any]], window: int = 4, aggregate_by_node: bool = True) -> Dict[str, Any]:
    """Rolling log-log slope profile on node-median rows.

    This is a diagnostic, not the decisive statistic. It helps detect whether the
    inferred crossover is a single breakpoint artifact or a gradual flattening.
    """
    use_rows = aggregate_density_by_node(rows) if aggregate_by_node else list(rows)
    pts = sorted(
        [(float(r.get("node_nm") or 0), float(r.get("density_MTr_per_mm2") or 0)) for r in use_rows],
        key=lambda t: t[0],
    )
    pts = [(x, y) for x, y in pts if x > 0 and y > 0]
    if len(pts) < max(3, window):
        return {"n": len(pts), "window": window, "status": "too_few_points"}
    out = []
    for i in range(0, len(pts) - window + 1):
        sub = pts[i:i + window]
        xs = np.log10([t[0] for t in sub])
        ys = np.log10([t[1] for t in sub])
        if len(np.unique(xs)) < 3:
            continue
        slope, intercept = np.polyfit(xs, ys, 1)
        node_center = 10 ** float(np.mean(xs))
        alpha = float(-slope)
        out.append({
            "node_center_nm": node_center,
            "node_min_nm": sub[0][0],
            "node_max_nm": sub[-1][0],
            "alpha": alpha,
            "area_like": bool(is_area_like_alpha(alpha)),
            "closer_to_area_than_volume": bool(closer_to_area_than_volume(alpha)),
        })
    band_10_100 = [r for r in out if 10 <= r["node_center_nm"] <= 100]
    small_1_10 = [r for r in out if 1 <= r["node_center_nm"] < 10]
    large_100_plus = [r for r in out if r["node_center_nm"] >= 100]
    def summary(block):
        if not block:
            return {"n_windows": 0, "median_alpha": None, "area_like_fraction": None}
        al = sorted(float(r["alpha"]) for r in block)
        mid = len(al)//2
        med = al[mid] if len(al)%2 else 0.5*(al[mid-1]+al[mid])
        return {
            "n_windows": len(block),
            "median_alpha": float(med),
            "area_like_fraction": float(sum(1 for r in block if r["area_like"]) / len(block)),
            "closer_to_area_fraction": float(sum(1 for r in block if r["closer_to_area_than_volume"]) / len(block)),
        }
    return {
        "n": len(pts),
        "window": window,
        "points": out,
        "bands": {
            "advanced_1_10nm": summary(small_1_10),
            "candidate_10_100nm": summary(band_10_100),
            "large_100nm_plus": summary(large_100_plus),
        },
    }


def subsample_robustness(
    rows: List[Dict[str, Any]],
    min_points: int = 5,
    trials: int = 300,
    sample_fraction: float = 0.8,
    seed: int = 7303,
) -> Dict[str, Any]:
    """Deterministic without-replacement subsample robustness for the EL3 fit.

    Bootstrap with replacement duplicates node values and can make the piecewise fit
    ill-conditioned. This uses 80% node subsamples without replacement instead.
    """
    base = [r for r in rows if float(r.get("node_nm") or 0) > 0 and float(r.get("density_MTr_per_mm2") or 0) > 0]
    n = len(base)
    needed = 2 * min_points + 1
    if n < needed:
        return {"n": n, "trials_requested": trials, "successful_trials": 0, "status": "too_few_points", "needed_min_rows": needed}
    rng = np.random.default_rng(seed)
    m = max(needed, int(math.ceil(sample_fraction * n)))
    m = min(m, n)
    statuses: List[str] = []
    global_breaks: List[float] = []
    hyp_breaks: List[float] = []
    global_small_alpha: List[float] = []
    hyp_small_alpha: List[float] = []
    for _ in range(trials):
        idx = rng.choice(n, size=m, replace=False)
        sub = [base[int(i)] for i in idx]
        # Subsample rows are already at the desired observable level; do not aggregate again.
        dec = fit_decision(sub, min_points=min_points, aggregate_by_node=False, observable_label="subsample")
        status = dec.get("decision", {}).get("status")
        if not status:
            continue
        statuses.append(status)
        gf = dec.get("global_fit", {})
        hf = dec.get("hypothesis_fit_10_100nm", {})
        if gf.get("break_node_nm") is not None:
            global_breaks.append(float(gf["break_node_nm"]))
        if hf.get("break_node_nm") is not None:
            hyp_breaks.append(float(hf["break_node_nm"]))
        if gf.get("small_node_alpha") is not None:
            global_small_alpha.append(float(gf["small_node_alpha"]))
        if hf.get("small_node_alpha") is not None:
            hyp_small_alpha.append(float(hf["small_node_alpha"]))
    def q(vals, probs=(0.16, 0.5, 0.84)):
        if not vals:
            return None
        return [float(x) for x in np.quantile(np.asarray(vals, dtype=float), probs)]
    succ = len(statuses)
    if succ == 0:
        return {"n": n, "sample_size": m, "trials_requested": trials, "successful_trials": 0, "status": "no_successful_fits"}
    return {
        "n": n,
        "sample_size": m,
        "sample_fraction": sample_fraction,
        "trials_requested": trials,
        "successful_trials": succ,
        "supportive_fraction": float(sum(1 for s in statuses if _decision_supportive(s)) / succ),
        "pass_proxy_fraction": float(sum(1 for s in statuses if s == "pass_proxy") / succ),
        "window_compatible_fraction": float(sum(1 for s in statuses if s == "window_compatible_proxy") / succ),
        "fail_or_inconclusive_fraction": float(sum(1 for s in statuses if s == "fail_or_inconclusive_proxy") / succ),
        "global_break_10_100_fraction": float(sum(1 for b in global_breaks if 10 <= b <= 100) / max(len(global_breaks), 1)),
        "global_break_nm_q16_q50_q84": q(global_breaks),
        "hypothesis_break_nm_q16_q50_q84": q(hyp_breaks),
        "global_small_alpha_q16_q50_q84": q(global_small_alpha),
        "hypothesis_small_alpha_q16_q50_q84": q(hyp_small_alpha),
    }


def fit_decision_with_robustness(
    rows: List[Dict[str, Any]],
    min_points: int = 5,
    aggregate_by_node: bool = False,
    observable_label: str = "unspecified",
    robustness_trials: int = 300,
) -> Dict[str, Any]:
    fit_rows = aggregate_density_by_node(rows) if aggregate_by_node else list(rows)
    result = fit_decision(
        rows,
        min_points=min_points,
        aggregate_by_node=aggregate_by_node,
        observable_label=observable_label,
    )
    robust = subsample_robustness(fit_rows, min_points=min_points, trials=robustness_trials, seed=7303 + len(fit_rows))
    local = local_slope_profile(fit_rows, window=4, aggregate_by_node=False)
    result["subsample_robustness"] = robust
    result["local_slope_profile"] = local
    support_frac = robust.get("supportive_fraction")
    status = result.get("decision", {}).get("status")
    strength = "not_estimated"
    if isinstance(support_frac, (int, float)):
        if support_frac >= 0.67:
            strength = "robust"
        elif support_frac >= 0.33:
            strength = "moderate"
        elif support_frac >= 0.10:
            strength = "weak"
        else:
            strength = "fragile"
        # Do not convert to pass. Only downgrade weak window-compatible support.
        if status == "window_compatible_proxy" and support_frac < 0.33:
            result["decision"]["status"] = "weak_window_compatible_proxy"
            result["decision"]["interpretation"] += " Subsample robustness is weak, so this is downgraded to weak window-compatible support."
    result["decision"]["subsample_supportive_fraction"] = support_frac
    result["decision"]["decision_strength"] = strength
    result["decision"]["uses_v2_5_subsample_robustness"] = True
    return result

def overall_el3_decision(historical_raw: Dict[str, Any], historical_node: Dict[str, Any], foundry: Dict[str, Any]) -> Dict[str, Any]:
    raw_status = historical_raw.get("decision", {}).get("status")
    node_status = historical_node.get("decision", {}).get("status")
    foundry_status = foundry.get("decision", {}).get("status")
    raw_support = _decision_supportive(raw_status)
    node_support = _decision_supportive(node_status)
    foundry_support = _decision_supportive(foundry_status)
    hist_support = raw_support or node_support
    foundry_frac = foundry.get("decision", {}).get("subsample_supportive_fraction")
    weak_foundry = bool(foundry_status == "weak_window_compatible_proxy" or (isinstance(foundry_frac, (int, float)) and foundry_frac < 0.33))
    if hist_support and foundry_support and weak_foundry:
        status = "mixed_to_supportive_proxy_weak_foundry"
        interpretation = "Historical density has compatible support and foundry standard-cell density is only weakly window-compatible; EL3 remains live but not robustly confirmed."
    elif hist_support and foundry_support:
        status = "mixed_to_supportive_proxy"
        interpretation = "At least one historical-density view and the foundry standard-cell proxy are compatible with a 10-100 nm crossover, but support must be read through proxy caveats and robustness fractions."
    elif hist_support and not foundry_support:
        status = "mixed_inconclusive"
        interpretation = "Historical density has a compatible view, but modern foundry standard-cell density does not reproduce it cleanly."
    elif foundry_support and not hist_support:
        status = "mixed_inconclusive"
        interpretation = "Modern foundry standard-cell density is compatible with the predicted crossover, but historical density is not robustly compatible."
    else:
        status = "not_supported_by_current_public_proxy"
        interpretation = "No separated public proxy gives robust 10-100 nm area-like crossover support under current rules."
    return {
        "status": status,
        "EL3a_historical_chip_density_raw": raw_status,
        "EL3a_historical_chip_density_node_median": node_status,
        "EL3b_foundry_standard_cell_density": foundry_status,
        "supportive_statuses": sorted({"pass_proxy", "window_compatible_proxy", "weak_window_compatible_proxy"}),
        "use_combined_fit_for_decision": False,
        "why_not_combined": "Historical full-chip transistor density and foundry standard-cell density are heterogeneous observables; the combined fit is retained only as a diagnostic.",
        "interpretation": interpretation,
        "v2_5_note": "EL3 now reports raw historical, node-median historical, and foundry-standard-cell fits separately, with subsample robustness and rolling local slopes.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="el3_cache")
    ap.add_argument("--out", default="out_el3.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--include-wikipedia", action="store_true", help="best-effort include process-node density tables from downloaded Wikipedia HTML")
    ap.add_argument("--include-foundry-nodes", action="store_true", help="verify and include modern foundry-node density rows from downloaded public HTML; writes extracted CSV to cache")
    ap.add_argument("--el3-physical-csv-url", action="append", default=[], help="optional public CSV URL with primary/semi-primary CPP/MMP/SRAM physical-pitch rows")
    args = ap.parse_args()

    cache = Path(args.cache)
    downloads = [download_to_cache(label, url, cache, force=args.force) for label, url in SOURCES.items()]
    texts: Dict[str, str] = {}
    warnings: List[str] = []
    for src in downloads:
        if src.ok:
            texts[src.label] = read_text_any(Path(src.path))
        else:
            warnings.append(f"download failed for {src.label}: {src.error}")

    historical_rows: List[Dict[str, Any]] = []
    if "Barentsen tech-progress transistor counts CSV" in texts:
        historical_rows.extend(load_barentsen_csv(texts["Barentsen tech-progress transistor counts CSV"]))

    wiki_rows: List[Dict[str, Any]] = []
    if args.include_wikipedia or args.include_foundry_nodes:
        wiki_rows = scrape_wikipedia_density_tables(texts)
    if args.include_foundry_nodes:
        wiki_rows = add_verified_foundry_fallback_rows(texts, wiki_rows)
    physical_rows = add_verified_physical_pitch_proxy_rows(texts, wiki_rows) if args.include_foundry_nodes else []

    rows: List[Dict[str, Any]] = []
    rows.extend(historical_rows)
    rows.extend(wiki_rows)
    rows.extend(physical_rows)

    seen = set()
    clean: List[Dict[str, Any]] = []
    for r in rows:
        if not (r["node_nm"] > 0 and r["density_MTr_per_mm2"] > 0):
            continue
        key = (r.get("source"), r.get("processor"), r.get("company"), round(r["node_nm"], 4), round(r["density_MTr_per_mm2"], 6))
        if key in seen:
            continue
        # Attach public-source provenance. Values may be secondary/public-page values,
        # not primary foundry disclosures; keep that explicit in every row.
        seen.add(key)
        clean.append(r)

    source_url_by_label = {src.label: src.url for src in downloads}
    source_sha_by_label = {src.label: src.sha256 for src in downloads}
    for r in clean:
        label = r.get("source")
        if label in source_url_by_label:
            r.setdefault("source_url", source_url_by_label[label])
            r.setdefault("source_sha256", source_sha_by_label[label])
        if "advertised_node_nm" not in r and r.get("node_nm") is not None:
            r["advertised_node_nm"] = r.get("node_nm")
        r.setdefault("data_role", "proxy_input_not_direct_EL3_physical_length")

    foundry_csv_path = write_foundry_csv(clean, cache / "foundry_node_density_public_extracted.csv")
    historical_rows_clean = [r for r in clean if r.get("row_type") == "historical_chip"]
    foundry_only_rows = [
        r for r in clean
        if r.get("row_type") == "foundry_node_table"
        and str(r.get("density_kind", "")).startswith("foundry_standard_cell")
    ]
    foundry_secondary_rows = [r for r in foundry_only_rows if not r.get("is_primary_source", False)]
    physical_pitch_rows = [r for r in clean if r.get("row_type") == "physical_pitch_proxy"]
    primary_physical_pitch_rows, primary_physical_notes, physical_primary_sources = load_primary_physical_pitch_csv_rows(
        args.el3_physical_csv_url,
        cache,
        foundry_only_rows,
    )

    # v2.5: three separated views. Raw historical rows can be dominated by repeated
    # CPUs at the same node, so node-median historical density is reported too.
    historical_raw = fit_decision_with_robustness(
        historical_rows_clean,
        min_points=5,
        aggregate_by_node=False,
        observable_label="EL3a historical full-chip transistor density (raw rows)",
    )
    historical_node = fit_decision_with_robustness(
        historical_rows_clean,
        min_points=5,
        aggregate_by_node=True,
        observable_label="EL3a historical full-chip transistor density (node medians)",
    )
    if len(foundry_only_rows) >= 7:
        foundry_fit = fit_decision_with_robustness(
            foundry_only_rows,
            min_points=3,
            aggregate_by_node=True,
            observable_label="EL3b modern foundry standard-cell density",
        )
    else:
        foundry_fit = {
            "observable_label": "EL3b modern foundry standard-cell density",
            "fit_rows_used": len(foundry_only_rows),
            "aggregate_by_node": True,
            "fit": {"n": len(foundry_only_rows), "break_node_nm": None},
            "decision": {"status": "insufficient_foundry_rows"},
        }
    if len(primary_physical_pitch_rows) >= 7:
        primary_physical_pitch_fit = fit_decision_with_robustness(
            primary_physical_pitch_rows,
            min_points=3,
            aggregate_by_node=True,
            observable_label="EL3c primary physical-pitch/SRAM density",
        )
    else:
        primary_physical_pitch_fit = {
            "observable_label": "EL3c primary physical-pitch/SRAM density",
            "fit_rows_used": len(primary_physical_pitch_rows),
            "decision": {"status": "data_insufficient_public_proxy", "evidence_ladder": evidence_for_status("data_insufficient_public_proxy")},
            "note": "No adequate public primary physical-pitch CSV rows were provided/found. Use --el3-physical-csv-url with public CPP/MMP/SRAM rows to activate this branch.",
            "loader_notes": primary_physical_notes,
        }

    if len(physical_pitch_rows) >= 7:
        physical_pitch_fit = fit_decision_with_robustness(
            physical_pitch_rows,
            min_points=3,
            aggregate_by_node=True,
            observable_label="EL3c secondary physical-pitch/SRAM placeholder proxy",
        )
    else:
        physical_pitch_fit = {
            "observable_label": "EL3c secondary physical-pitch/SRAM placeholder proxy",
            "fit_rows_used": len(physical_pitch_rows),
            "decision": {"status": "data_insufficient_public_proxy", "evidence_ladder": evidence_for_status("data_insufficient_public_proxy")},
            "note": "Only a low-confidence secondary advertised-node placeholder branch is available; real CPP/MMP/SRAM rows are still needed.",
        }

    combined_diagnostic = fit_decision_with_robustness(
        clean,
        min_points=5,
        aggregate_by_node=True,
        observable_label="diagnostic only: heterogeneous historical+foundry rows",
        robustness_trials=150,
    )
    decision = overall_el3_decision(historical_raw, historical_node, foundry_fit)
    decision["evidence_ladder"] = evidence_for_status(decision.get("status"))
    for block in (historical_raw, historical_node, foundry_fit, physical_pitch_fit):
        if isinstance(block.get("decision"), dict):
            block["decision"]["evidence_ladder"] = evidence_for_status(block["decision"].get("status"))

    slopes = {
        "historical_chip_density": {
            "large_nodes_100nm_plus": band_slope(historical_rows_clean, 100, 20000),
            "candidate_crossover_10_100nm": band_slope(historical_rows_clean, 10, 100),
            "advanced_nodes_1_10nm": band_slope(historical_rows_clean, 1, 10),
        },
        "foundry_standard_cell_density_node_medians": {
            "large_nodes_100nm_plus": band_slope(foundry_only_rows, 100, 20000, aggregate_by_node=True),
            "candidate_crossover_10_100nm": band_slope(foundry_only_rows, 10, 100, aggregate_by_node=True),
            "advanced_nodes_1_10nm": band_slope(foundry_only_rows, 1, 10, aggregate_by_node=True),
        },
    }
    result = {
        "test_name": "EL3 silicon volume-to-area scaling crossover public node-density proxy",
        "downloaded_sources": source_records(downloads + physical_primary_sources),
        "warnings": warnings,
        "prediction": "EL3: volume -> area scaling crossover at L ~10-100 nm for Si; information density vs feature size should flatten from volume-like to area-like scaling.",
        "v2_8_method_change": "Separates EL3c primary physical-pitch/SRAM CSV rows from secondary placeholder physical-scale rows; primary-only evidence is explicit.",
        "v2_7_method_change": "Adds confidence-filtered decisions and EL3c physical-pitch/SRAM proxy branch; keeps realized full-chip, foundry standard-cell, and physical-scale proxies separated.",
        "v2_6_method_change": "EL3 decision now also requires a crossover-like alpha drop: the larger-node segment must be steeper than the advanced-node segment. Historical raw rows, historical node-median rows, and modern foundry standard-cell rows remain separated; combined fit is diagnostic only.",
        "n_rows_used": len(clean),
        "n_barentsen_rows": len(historical_rows_clean),
        "n_wikipedia_rows": len([r for r in wiki_rows if r.get("extraction_method") == "pandas_read_html"]),
        "n_foundry_node_rows": len(foundry_only_rows),
        "n_physical_pitch_proxy_rows": len(physical_pitch_rows),
        "n_primary_physical_pitch_rows": len(primary_physical_pitch_rows),
        "primary_physical_pitch_loader_notes": primary_physical_notes,
        "density_kind_counts": {k: sum(1 for r in clean if r.get("density_kind") == k) for k in sorted({str(r.get("density_kind")) for r in clean})},
        "foundry_node_csv_written": foundry_csv_path,
        "EL3a_historical_chip_density_raw": historical_raw,
        "EL3a_historical_chip_density_node_median": historical_node,
        "EL3a_historical_chip_density": historical_node,
        "EL3b_foundry_standard_cell_density": foundry_fit,
        "EL3c_primary_physical_pitch_density_proxy": primary_physical_pitch_fit,
        "EL3c_secondary_physical_pitch_density_proxy": physical_pitch_fit,
        "EL3c_physical_pitch_density_proxy": primary_physical_pitch_fit,
        "confidence_filtered_decisions": {
            "EL3_primary_only": {"status": primary_physical_pitch_fit.get("decision", {}).get("status"), "evidence_ladder": evidence_for_status(primary_physical_pitch_fit.get("decision", {}).get("status")), "reason": "Uses only public primary/semi-primary physical-pitch CSV rows supplied through --el3-physical-csv-url."},
            "EL3_all_public_sources": decision,
            "EL3_wikipedia_proxy_only": {"status": foundry_fit.get("decision", {}).get("status"), "evidence_ladder": evidence_for_status(foundry_fit.get("decision", {}).get("status"))},
            "EL3_secondary_physical_placeholder_only": {"status": physical_pitch_fit.get("decision", {}).get("status"), "evidence_ladder": evidence_for_status(physical_pitch_fit.get("decision", {}).get("status"))},
        },
        "combined_diagnostic_fit_not_used_for_decision": combined_diagnostic,
        # Backward-compatible aliases for older readers.
        "historical_chip_fit": historical_node,
        "foundry_node_fit": foundry_fit,
        "piecewise_log_density_fit": combined_diagnostic["fit"],
        "band_slopes_density_proportional_to_node_minus_alpha": slopes,
        "decision": decision,
        "row_sample": clean[:20],
        "foundry_node_sample": foundry_only_rows[:30],
        "primary_physical_pitch_sample": primary_physical_pitch_rows[:30],
        "physical_pitch_proxy_sample": physical_pitch_rows[:30],
        "falsification_logic": {
            "confirm_like": "EL3 is robust only if separated like-for-like public observables support a 10-100 nm crossover with area-like small-node exponent and non-fragile subsample robustness.",
            "mixed_like": "One separated proxy is compatible and another fails/inconclusive; this keeps EL3 live but not confirmed.",
            "falsify_like": "All separated public proxies fail to show a 10-100 nm area-like crossover after node-median and robustness controls.",
            "caveat": "Commercial node names are marketing labels; public foundry densities are standard-cell proxies, while historical processor rows are full-chip density proxies. They must not be merged into one decisive fit.",
        },
    }
    write_json(result, Path(args.out))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
