#!/usr/bin/env python3
"""v2.7 shared improvement helpers for CCDR public proxy tests.

The helpers are intentionally conservative: they download public artifacts, parse
machine-readable tables when available, and classify evidence on a common ladder.
"""
from __future__ import annotations

import csv
import io
import json
import re
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from _public_test_common import download_to_cache, ensure_dir, read_text_any, slugify

EVIDENCE_LADDER: Dict[str, Dict[str, Any]] = {
    "data_missing": {"level": 0, "description": "No relevant public data rows or only failed downloads."},
    "qualitative_context": {"level": 1, "description": "Downloaded literature gives context/phrases only; no numeric proxy table."},
    "partial_proxy": {"level": 2, "description": "Numeric proxy exists, but it is not the target observable or is too small/low-quality."},
    "window_compatible_proxy": {"level": 3, "description": "Proxy is directionally/window compatible, but not decisive or not robust enough."},
    "robust_proxy": {"level": 4, "description": "Like-for-like public proxy is robust under controls/subsampling."},
    "decisive_public_test": {"level": 5, "description": "Target observable is tested on adequate public rows with decisive metric."},
}

STATUS_TO_LADDER = {
    "data_insufficient_public_proxy": "data_missing",
    "data_limited": "data_missing",
    "insufficient_foundry_rows": "data_missing",
    "unconfirmed": "qualitative_context",
    "partial": "partial_proxy",
    "partial_proxy_only": "partial_proxy",
    "partial_support_TLS_not_qubit": "partial_proxy",
    "mechanism_supported": "robust_proxy",
    "proxy_pass": "window_compatible_proxy",
    "window_compatible_proxy": "window_compatible_proxy",
    "weak_window_compatible_proxy": "partial_proxy",
    "mixed_to_supportive_proxy": "window_compatible_proxy",
    "mixed_to_supportive_proxy_weak_foundry": "window_compatible_proxy",
    "mixed_inconclusive": "partial_proxy",
    "pass_proxy": "robust_proxy",
    "pass": "decisive_public_test",
    "confirm_like": "decisive_public_test",
    "fail_or_inconclusive_proxy": "partial_proxy",
    "not_supported_by_current_public_proxy": "partial_proxy",
}


def evidence_for_status(status: Any) -> Dict[str, Any]:
    key = STATUS_TO_LADDER.get(str(status), "qualitative_context")
    out = dict(EVIDENCE_LADDER[key])
    out["ladder_key"] = key
    out["status"] = status
    return out


def add_evidence_ladder(result: Dict[str, Any], status: Any) -> Dict[str, Any]:
    result["evidence_ladder"] = evidence_for_status(status)
    return result


def find_public_supplement_links(html: str, base_url: str) -> List[str]:
    """Find candidate machine-readable supplement links in downloaded HTML/text."""
    links: List[str] = []
    if not html:
        return links
    patterns = [
        r'href=["\']([^"\']+\.(?:csv|tsv|txt|dat|xlsx?|zip|json|yaml|yml)(?:\?[^"\']*)?)["\']',
        r'(https?://\S+\.(?:csv|tsv|txt|dat|xlsx?|zip|json|yaml|yml))',
    ]
    for pat in patterns:
        for m in re.finditer(pat, html, re.IGNORECASE):
            href = m.group(1).strip().rstrip(".)]")
            links.append(urllib.parse.urljoin(base_url, href))
    # Some publishers use Supplementary/Data links without a file extension.
    for m in re.finditer(r'href=["\']([^"\']*(?:supplement|supporting|mediaobjects|download|dataset|data)[^"\']*)["\']', html, re.IGNORECASE):
        href = urllib.parse.urljoin(base_url, m.group(1).strip())
        if href not in links:
            links.append(href)
    # Stable de-dupe preserving order.
    seen = set()
    out = []
    for u in links:
        if u not in seen and u.startswith(("http://", "https://")):
            seen.add(u)
            out.append(u)
    return out[:40]


def parse_csv_like_text(text: str, source_label: str) -> List[Dict[str, Any]]:
    """Parse CSV/TSV-ish content into rows; returns empty when header is absent."""
    if not text or len(text) < 10:
        return []
    sample = text[:4096]
    delim = "\t" if sample.count("\t") > sample.count(",") else ","
    try:
        rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r and any(k for k in r.keys() if k is not None):
            rr = {str(k): v for k, v in r.items() if k is not None}
            rr["__source_label"] = source_label
            out.append(rr)
    return out


def parse_table_artifact(path: Path, label: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse a downloaded public table artifact into dict rows.

    Supports csv/tsv/txt/dat, xlsx/xls through pandas, and zip files containing
    these formats. Unsupported files are returned as notes rather than errors.
    """
    notes: List[str] = []
    suffix = path.suffix.lower()
    rows: List[Dict[str, Any]] = []
    try:
        if suffix in {".csv", ".tsv", ".txt", ".dat"}:
            rows.extend(parse_csv_like_text(read_text_any(path), label))
        elif suffix in {".xlsx", ".xls"}:
            try:
                import pandas as pd  # type: ignore
                book = pd.read_excel(path, sheet_name=None)
                for sheet, df in book.items():
                    for rec in df.to_dict(orient="records"):
                        rec = {str(k): v for k, v in rec.items()}
                        rec["__source_label"] = f"{label}::{sheet}"
                        rows.append(rec)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"xlsx parse failed for {label}: {exc}")
        elif suffix == ".zip" or zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue
                    inner_suffix = Path(name).suffix.lower()
                    if inner_suffix not in {".csv", ".tsv", ".txt", ".dat", ".json"}:
                        continue
                    raw = zf.read(name)
                    text = raw.decode("utf-8", errors="replace")
                    rows.extend(parse_csv_like_text(text, f"{label}::{name}"))
        else:
            notes.append(f"unsupported table artifact extension {suffix or '<none>'} for {label}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"parse failed for {label}: {exc}")
    return rows, notes


def crawl_public_supplements(source_pages: Dict[str, str], cache_dir: Path, force: bool = False) -> Dict[str, Any]:
    """Download source pages, discover and parse public supplement tables."""
    ensure_dir(cache_dir)
    page_sources = []
    supplement_sources = []
    all_rows: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []
    for label, url in source_pages.items():
        page = download_to_cache(f"supplement_page_{slugify(label, 60)}", url, cache_dir, force=force)
        page_sources.append(page)
        if not page.ok:
            notes.append({"label": label, "url": url, "error": page.error})
            continue
        text = read_text_any(Path(page.path))
        links = find_public_supplement_links(text, url)
        notes.append({"label": label, "url": url, "candidate_links": len(links), "links_sample": links[:8]})
        for i, link in enumerate(links):
            src = download_to_cache(f"supplement_{slugify(label, 40)}_{i}", link, cache_dir, force=force)
            supplement_sources.append(src)
            if not src.ok:
                notes.append({"label": label, "supplement_url": link, "error": src.error})
                continue
            rows, parse_notes = parse_table_artifact(Path(src.path), f"{label}::{link}")
            all_rows.extend(rows)
            if parse_notes:
                notes.append({"label": label, "supplement_url": link, "parse_notes": parse_notes})
    return {
        "page_sources": page_sources,
        "supplement_sources": supplement_sources,
        "rows": all_rows,
        "notes": notes,
    }


def table_column_inventory(rows: Iterable[Dict[str, Any]], max_examples: int = 20) -> Dict[str, Any]:
    cols: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        for k in row.keys():
            cols[str(k)] = cols.get(str(k), 0) + 1
        if len(examples) < max_examples:
            examples.append({k: str(v)[:80] for k, v in row.items()})
    return {"column_counts": dict(sorted(cols.items(), key=lambda kv: (-kv[1], kv[0]))), "examples": examples}
