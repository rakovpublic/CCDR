#!/usr/bin/env python3
"""v55 public-source fusion parsers for CCDR Tier-B tests T26-T30.

These parsers deliberately extract only text-layer PDF/table rows from public papers.
They do not use OCR and they do not treat paper/figure summaries as strict confirmation
unless a machine-readable, per-shot / per-row measurement table with required columns is found.

Design goal:
- T29: best preliminary structured-public path from Stroth et al. W7-X/AUG/W7-AS comparison.
- T28: Verdoolaege/DB5.2.3 public summary/regression anchor only.
- T27: Paz-Soldan RMP-ELM public compilation summary/suggestive rows only.
- T26: Loarte/ITPA/JET/AUG/MAST ELM-loss figure/summary rows only.
- T30: derived/dependency parser, reusing T28/T29 if available.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .tierb_common import download_bytes, ensure_dir, safe_name, sha1_text, to_jsonable


PUBLIC_FUSION_SOURCES_V55: Dict[str, List[Dict[str, str]]] = {
    "T26": [
        {
            "label": "IAEA/ITPA edge pedestal / ITER implications summary with ΔWELM/Wped trends",
            "url": "https://www-pub.iaea.org/mtcd/meetings/fec2006/it_1-3.pdf",
            "kind": "pdf_figures_summary",
            "parser": "elm_loss_summary_text_parser_v55",
        },
    ],
    "T27": [
        {
            "label": "Paz-Soldan et al. 2024 Nucl. Fusion RMP-ELM suppressed operational-space survey",
            "url": "https://fusion.columbia.edu/sites/fusion.columbia.edu/files/content/papers/Paz-Soldan_2024_Nucl._Fusion_64_096004.pdf",
            "kind": "pdf_compilation_summary",
            "parser": "rmp_elm_summary_text_parser_v55",
        },
    ],
    "T28": [
        {
            "label": "Verdoolaege et al. 2021 updated ITPA global H-mode confinement database DB5.2.3-STD5",
            "url": "https://pure.mpg.de/rest/items/item_3325255_5/component/file_3357083/content",
            "kind": "pdf_summary_regression_tables",
            "parser": "db5_summary_table_text_parser_v55",
        },
        {
            "label": "OSF International Global H-Mode Confinement Database landing/API",
            "url": "https://api.osf.io/v2/nodes/drwcq/files/osfstorage/",
            "kind": "osf_file_listing_probe",
            "parser": "osf_structured_attachment_probe_v55",
        },
    ],
    "T29": [
        {
            "label": "Stroth et al. 2021 Stellarator-Tokamak Energy Confinement Comparison",
            "url": "https://pure.mpg.de/rest/items/item_3266722_3/component/file_3273290/content",
            "kind": "pdf_comparison_tables",
            "parser": "stroth_comparison_table_text_parser_v55",
        },
    ],
    "T30": [],
}


DEVICE_PATTERNS = [
    ("W7-X", r"\bW\s*7\s*[-–]?\s*X\b|\bW7-X\b"),
    ("W7-AS", r"\bW\s*7\s*[-–]?\s*AS\b|\bW7-AS\b"),
    ("AUG", r"\bAUG\b|ASDEX\s+Upgrade"),
    ("DIII-D", r"\bDIII\s*[-–]?\s*D\b"),
    ("JET", r"\bJET\b"),
    ("MAST", r"\bMAST\b"),
    ("EAST", r"\bEAST\b"),
    ("KSTAR", r"\bKSTAR\b"),
    ("ITER", r"\bITER\b"),
]


_FLOAT_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _find_devices(text: str) -> List[str]:
    out = []
    for name, pat in DEVICE_PATTERNS:
        if re.search(pat, text or "", re.I):
            out.append(name)
    return out


def _device_type(device: str, line: str = "") -> str:
    s = (device + " " + (line or "")).lower()
    if "w7" in s or "stellarator" in s:
        return "stellarator"
    if any(x.lower() in s for x in ["aug", "asdex", "diii", "jet", "mast", "east", "kstar", "tokamak"]):
        return "tokamak"
    return "unknown"


def _numbers(text: str) -> List[float]:
    vals: List[float] = []
    for m in _FLOAT_RE.finditer(text or ""):
        try:
            vals.append(float(m.group(0)))
        except Exception:
            pass
    return vals


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_pdf_text_pages(data: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return page text from a PDF using text-layer readers only."""
    meta: Dict[str, Any] = {"method": None, "page_count": 0, "errors": []}
    pages: List[Dict[str, Any]] = []
    # PyMuPDF is fast and installed in the bundle requirements.
    try:
        import fitz  # type: ignore
        doc = fitz.open(stream=data, filetype="pdf")
        for i, page in enumerate(doc):
            try:
                text = page.get_text("text") or ""
            except Exception as e:
                meta["errors"].append(f"page_{i+1}_fitz_text_failed:{type(e).__name__}:{e}")
                text = ""
            pages.append({"page": i + 1, "text": text})
        meta.update({"method": "pymupdf_text", "page_count": len(pages)})
        if any((p.get("text") or "").strip() for p in pages):
            return pages, meta
    except Exception as e:
        meta["errors"].append(f"pymupdf_failed:{type(e).__name__}:{e}")
    # Fallback: pdfplumber text extraction.
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    meta["errors"].append(f"page_{i+1}_pdfplumber_text_failed:{type(e).__name__}:{e}")
                    text = ""
                pages.append({"page": i + 1, "text": text})
        meta.update({"method": "pdfplumber_text", "page_count": len(pages)})
        return pages, meta
    except Exception as e:
        meta["errors"].append(f"pdfplumber_failed:{type(e).__name__}:{e}")
    return pages, meta


def _extract_pdf_tables(data: bytes, max_tables: int = 80) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract text-layer tables with pdfplumber; no OCR."""
    tables: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"method": "pdfplumber_extract_tables", "table_count": 0, "errors": []}
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_i, page in enumerate(pdf.pages):
                if len(tables) >= max_tables:
                    break
                try:
                    extracted = page.extract_tables() or []
                except Exception as e:
                    meta["errors"].append(f"page_{page_i+1}_tables_failed:{type(e).__name__}:{e}")
                    extracted = []
                for t_i, table in enumerate(extracted):
                    if len(tables) >= max_tables:
                        break
                    rows = []
                    for row in table or []:
                        rows.append([_norm_space(str(c)) if c is not None else "" for c in row])
                    tables.append({"page": page_i + 1, "table_index": t_i, "rows": rows})
    except Exception as e:
        meta["errors"].append(f"pdfplumber_open_failed:{type(e).__name__}:{e}")
    meta["table_count"] = len(tables)
    return tables, meta


def _context_lines(text: str, i: int, radius: int = 2) -> str:
    lines = text.splitlines()
    lo = max(0, i - radius)
    hi = min(len(lines), i + radius + 1)
    return _norm_space(" | ".join(lines[lo:hi]))


def _line_rows(
    pages: Sequence[Dict[str, Any]],
    *,
    include_regex: str,
    min_numbers: int = 1,
    source_label: str,
    source_url: str,
    test_id: str,
    row_type: str,
    max_rows: int = 300,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    inc = re.compile(include_regex, re.I)
    seen = set()
    for page in pages:
        text = page.get("text") or ""
        lines = text.splitlines()
        for i, line in enumerate(lines):
            clean = _norm_space(line)
            if not clean or not inc.search(clean):
                continue
            nums = _numbers(clean)
            if len(nums) < min_numbers:
                # keep some semantic rows even without numbers, but mark them as non-measurement.
                if not re.search(r"table|database|comparison|survey|suppression|scaling|regression", clean, re.I):
                    continue
            ctx = _context_lines(text, i, radius=2)
            key = (page.get("page"), clean[:180])
            if key in seen:
                continue
            seen.add(key)
            devices = _find_devices(ctx or clean)
            rows.append({
                "test_id": test_id,
                "row_type_v55": row_type,
                "source_label_v55": source_label,
                "source_url_v55": source_url,
                "page_v55": page.get("page"),
                "line_text_v55": clean[:2000],
                "context_v55": ctx[:3000],
                "devices_v55": "|".join(devices),
                "numeric_values_v55": "|".join(str(x) for x in nums[:25]),
                "n_numeric_values_v55": len(nums),
                "extraction_method_v55": "pdf_text_line_context_no_ocr",
                "confirm_allowed_from_row_v55": False,
            })
            if len(rows) >= max_rows:
                return rows
    return rows


def _table_rows(
    tables: Sequence[Dict[str, Any]],
    *,
    include_regex: str,
    min_numbers: int = 1,
    source_label: str,
    source_url: str,
    test_id: str,
    row_type: str,
    max_rows: int = 300,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    inc = re.compile(include_regex, re.I)
    seen = set()
    for table in tables:
        trows = table.get("rows") or []
        header = []
        # choose first non-empty row as potential header
        for cand in trows[:3]:
            if any(str(c).strip() for c in cand):
                header = [str(c) for c in cand]
                break
        for ridx, row in enumerate(trows):
            line = _norm_space(" | ".join(str(c) for c in row if c is not None))
            if not line or not inc.search(line):
                continue
            nums = _numbers(line)
            if len(nums) < min_numbers:
                continue
            devices = _find_devices(line)
            key = (table.get("page"), table.get("table_index"), ridx, line[:180])
            if key in seen:
                continue
            seen.add(key)
            rowmap = {}
            if header and len(header) == len(row) and header != row:
                rowmap = {safe_name(str(h), 40): str(v) for h, v in zip(header, row)}
            rows.append({
                "test_id": test_id,
                "row_type_v55": row_type,
                "source_label_v55": source_label,
                "source_url_v55": source_url,
                "page_v55": table.get("page"),
                "table_index_v55": table.get("table_index"),
                "table_row_index_v55": ridx,
                "header_guess_v55": "|".join(header)[:2000],
                "row_text_v55": line[:3000],
                "row_json_v55": json.dumps(to_jsonable(rowmap), sort_keys=True)[:5000],
                "devices_v55": "|".join(devices),
                "numeric_values_v55": "|".join(str(x) for x in nums[:25]),
                "n_numeric_values_v55": len(nums),
                "extraction_method_v55": "pdfplumber_table_text_no_ocr",
                "confirm_allowed_from_row_v55": False,
            })
            if len(rows) >= max_rows:
                return rows
    return rows


def _source_download(cache_dir: Path, source: Dict[str, str], timeout: int, force: bool) -> Tuple[Optional[bytes], Dict[str, Any]]:
    url = source.get("url") or ""
    data, meta = download_bytes(url, cache_dir, timeout=timeout, force=force)
    meta = dict(meta or {})
    meta.update({"source_label_v55": source.get("label"), "source_kind_v55": source.get("kind"), "parser_v55": source.get("parser")})
    return data, meta


def _parse_json_or_text_listing(data: bytes, source: Dict[str, str], test_id: str) -> List[Dict[str, Any]]:
    """Probe OSF/API listings for true attached structured files. Metadata wrappers are not evidence."""
    text = data.decode("utf-8", errors="replace") if data else ""
    rows: List[Dict[str, Any]] = []
    try:
        obj = json.loads(text)
    except Exception:
        obj = None
    def walk(x: Any):
        if isinstance(x, dict):
            name = str(x.get("name") or x.get("attributes", {}).get("name") or x.get("links", {}).get("download") or "")
            url = str(x.get("download") or x.get("links", {}).get("download") or x.get("href") or "")
            blob = json.dumps(x, sort_keys=True)[:4000]
            if re.search(r"\.(csv|tsv|tab|xlsx?|h5|hdf5|dat|txt)(\?|$)", name + " " + url, re.I):
                rows.append({
                    "test_id": test_id,
                    "row_type_v55": "structured_attachment_candidate",
                    "source_label_v55": source.get("label"),
                    "source_url_v55": source.get("url"),
                    "candidate_name_v55": name,
                    "candidate_url_v55": url,
                    "candidate_json_v55": blob,
                    "extraction_method_v55": "json_api_structured_attachment_probe",
                    "confirm_allowed_from_row_v55": False,
                })
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x[:500]:
                walk(v)
    if obj is not None:
        walk(obj)
    else:
        for m in re.finditer(r"https?://\S+", text):
            url = m.group(0).rstrip('"\',)>]')
            if re.search(r"\.(csv|tsv|tab|xlsx?|h5|hdf5|dat|txt)(\?|$)", url, re.I):
                rows.append({
                    "test_id": test_id,
                    "row_type_v55": "structured_attachment_candidate",
                    "source_label_v55": source.get("label"),
                    "source_url_v55": source.get("url"),
                    "candidate_url_v55": url,
                    "extraction_method_v55": "text_structured_link_probe",
                    "confirm_allowed_from_row_v55": False,
                })
    return rows


def parse_t26(cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    downloads: List[Dict[str, Any]] = []
    for source in PUBLIC_FUSION_SOURCES_V55["T26"]:
        data, meta = _source_download(cache_dir / "v55_fusion_public_sources" / "T26", source, timeout, force)
        downloads.append(meta)
        if not data:
            continue
        pages, text_meta = _extract_pdf_text_pages(data)
        tables, table_meta = _extract_pdf_tables(data, max_tables=max_tables)
        include = r"ELM|W\s*[_-]?\s*ELM|WELM|dW|ΔW|pedestal|W\s*[_-]?\s*ped|Wped|P\s*[_-]?\s*ped|fluence|loss|collisionality|ITER|DIII|JET|AUG|ASDEX|MAST"
        rows.extend(_table_rows(tables, include_regex=include, min_numbers=1, source_label=source["label"], source_url=source["url"], test_id="T26", row_type="elm_loss_pdf_table_or_summary_v55"))
        rows.extend(_line_rows(pages, include_regex=include, min_numbers=1, source_label=source["label"], source_url=source["url"], test_id="T26", row_type="elm_loss_pdf_line_or_figure_context_v55"))
        meta["text_extract_meta_v55"] = text_meta
        meta["table_extract_meta_v55"] = table_meta
    # Strict per-shot gate: must have named ELM + pedestal + shot/device + volume/proxy columns in a machine-readable row. PDF summary rows fail by policy.
    return {
        "test_id": "T26",
        "parser_status_v55": "summary_or_figure_rows_extracted_nonconfirm" if rows else "no_public_rows_extracted_or_download_failed",
        "rows": rows,
        "downloads": downloads,
        "n_rows_v55": len(rows),
        "strict_confirm_ready_v55": False,
        "preliminary_status_v55": "partial_trend_support_only" if rows else "blocked",
        "policy_v55": "Text-layer PDF table/line extraction is partial only; rigorous T26 confirmation requires public per-shot E_ELM/W_ELM + Pped/Wped + volume/proxy + device/shot rows.",
    }


def parse_t27(cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    downloads: List[Dict[str, Any]] = []
    for source in PUBLIC_FUSION_SOURCES_V55["T27"]:
        data, meta = _source_download(cache_dir / "v55_fusion_public_sources" / "T27", source, timeout, force)
        downloads.append(meta)
        if not data:
            continue
        pages, text_meta = _extract_pdf_text_pages(data)
        tables, table_meta = _extract_pdf_tables(data, max_tables=max_tables)
        include = r"RMP|resonant magnetic perturb|ELM|suppression|mitigation|frequency|phasing|coil|current|n\s*=|toroidal mode|DIII|AUG|ASDEX|EAST|KSTAR|JET"
        rows.extend(_table_rows(tables, include_regex=include, min_numbers=1, source_label=source["label"], source_url=source["url"], test_id="T27", row_type="rmp_elm_pdf_table_or_summary_v55"))
        rows.extend(_line_rows(pages, include_regex=include, min_numbers=1, source_label=source["label"], source_url=source["url"], test_id="T27", row_type="rmp_elm_pdf_line_or_figure_context_v55"))
        meta["text_extract_meta_v55"] = text_meta
        meta["table_extract_meta_v55"] = table_meta
    return {
        "test_id": "T27",
        "parser_status_v55": "suggestive_compilation_rows_extracted_nonconfirm" if rows else "no_public_rows_extracted_or_download_failed",
        "rows": rows,
        "downloads": downloads,
        "n_rows_v55": len(rows),
        "strict_confirm_ready_v55": False,
        "preliminary_status_v55": "suggestive_public_compilation" if rows else "blocked",
        "policy_v55": "RMP/ELM text/table extraction is suggestive only; confirmation requires raw per-discharge RMP amplitude/current/phasing + ELM-frequency rows with baseline/no-RMP controls.",
    }


def parse_t28(cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    attachments: List[Dict[str, Any]] = []
    downloads: List[Dict[str, Any]] = []
    for source in PUBLIC_FUSION_SOURCES_V55["T28"]:
        data, meta = _source_download(cache_dir / "v55_fusion_public_sources" / "T28", source, timeout, force)
        downloads.append(meta)
        if not data:
            continue
        if source.get("kind") == "osf_file_listing_probe":
            attachments.extend(_parse_json_or_text_listing(data, source, "T28"))
            continue
        pages, text_meta = _extract_pdf_text_pages(data)
        tables, table_meta = _extract_pdf_tables(data, max_tables=max_tables)
        include = r"DB5|H[- ]?mode|confinement|tau|τ|H98|IPB98|q95|density|n\s*e|P\s*(heat|loss|aux)|power|regression|database|time\s*slices|ITER|tokamak"
        rows.extend(_table_rows(tables, include_regex=include, min_numbers=1, source_label=source["label"], source_url=source["url"], test_id="T28", row_type="db5_summary_or_regression_pdf_table_v55"))
        rows.extend(_line_rows(pages, include_regex=include, min_numbers=1, source_label=source["label"], source_url=source["url"], test_id="T28", row_type="db5_summary_or_regression_pdf_line_v55"))
        meta["text_extract_meta_v55"] = text_meta
        meta["table_extract_meta_v55"] = table_meta
    true_attachment_candidates = [r for r in attachments if re.search(r"\.(csv|tsv|tab|xlsx?|h5|hdf5|dat)(\?|$)", (r.get("candidate_name_v55", "") + " " + r.get("candidate_url_v55", "")), re.I)]
    return {
        "test_id": "T28",
        "parser_status_v55": "db5_summary_rows_extracted_full_rows_not_public" if rows else "no_db5_summary_rows_extracted_or_download_failed",
        "rows": rows,
        "attachment_candidates": attachments,
        "downloads": downloads,
        "n_rows_v55": len(rows),
        "n_structured_attachment_candidates_v55": len(true_attachment_candidates),
        "strict_confirm_ready_v55": False,
        "preliminary_status_v55": "strong_summary_anchor_only" if rows else "blocked",
        "policy_v55": "Verdoolaege/DB5 summary/regression rows are ingredient support only; full DB5.2.3 per-timeslice row table is required for T28 confirmation.",
    }


def parse_t29(cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    downloads: List[Dict[str, Any]] = []
    for source in PUBLIC_FUSION_SOURCES_V55["T29"]:
        data, meta = _source_download(cache_dir / "v55_fusion_public_sources" / "T29", source, timeout, force)
        downloads.append(meta)
        if not data:
            continue
        pages, text_meta = _extract_pdf_text_pages(data)
        tables, table_meta = _extract_pdf_tables(data, max_tables=max_tables)
        include = r"W\s*7\s*[-–]?\s*X|W\s*7\s*[-–]?\s*AS|ASDEX|AUG|stellarator|tokamak|confinement|transport|χ|chi|tau|τ|density|temperature|power|database|comparison|ISS04|IPB98|H98|neoclassical|turbulent"
        rows.extend(_table_rows(tables, include_regex=include, min_numbers=1, source_label=source["label"], source_url=source["url"], test_id="T29", row_type="stroth_comparison_pdf_table_v55"))
        rows.extend(_line_rows(pages, include_regex=include, min_numbers=1, source_label=source["label"], source_url=source["url"], test_id="T29", row_type="stroth_comparison_pdf_line_v55"))
        meta["text_extract_meta_v55"] = text_meta
        meta["table_extract_meta_v55"] = table_meta
    # derive normalized device summary rows from raw text/table rows
    norm_rows: List[Dict[str, Any]] = []
    for r in rows:
        text = " ".join(str(r.get(k, "")) for k in ["row_text_v55", "line_text_v55", "context_v55", "header_guess_v55"])
        devices = _find_devices(text)
        if not devices and re.search(r"stellarator|tokamak", text, re.I):
            devices = ["generic_stellarator_or_tokamak"]
        nums = _numbers(text)
        for dev in devices[:4]:
            norm_rows.append({
                "test_id": "T29",
                "device_v55": dev,
                "device_type_v55": _device_type(dev, text),
                "source_label_v55": r.get("source_label_v55"),
                "source_url_v55": r.get("source_url_v55"),
                "page_v55": r.get("page_v55"),
                "raw_row_type_v55": r.get("row_type_v55"),
                "candidate_transport_or_confinement_text_v55": _norm_space(text)[:2500],
                "numeric_values_v55": "|".join(str(x) for x in nums[:25]),
                "n_numeric_values_v55": len(nums),
                "extraction_method_v55": "derived_device_row_from_stroth_pdf_text_or_table",
                "confirm_allowed_from_row_v55": False,
                "preliminary_allowed_from_row_v55": True,
            })
    # De-duplicate by device/page/raw text prefix
    seen = set(); dedup = []
    for r in norm_rows:
        key = (r.get("device_v55"), r.get("page_v55"), (r.get("candidate_transport_or_confinement_text_v55") or "")[:160])
        if key in seen:
            continue
        seen.add(key); dedup.append(r)
    devices = sorted(set(r.get("device_v55") for r in dedup if r.get("device_v55")))
    has_w7x = any(str(d).upper().replace(" ", "") == "W7-X" for d in devices)
    has_tokamak = any(_device_type(str(d)) == "tokamak" for d in devices) or any(str(d) in {"AUG", "DIII-D", "JET"} for d in devices)
    preliminary_ready = len(dedup) >= 3 and has_w7x and has_tokamak
    return {
        "test_id": "T29",
        "parser_status_v55": "preliminary_structured_public_rows_extracted" if preliminary_ready else ("candidate_public_rows_extracted_underpowered" if dedup else "no_public_rows_extracted_or_download_failed"),
        "rows": rows,
        "normalized_rows": dedup,
        "downloads": downloads,
        "n_rows_v55": len(rows),
        "n_normalized_rows_v55": len(dedup),
        "devices_found_v55": devices,
        "strict_confirm_ready_v55": False,
        "preliminary_public_test_ready_v55": bool(preliminary_ready),
        "preliminary_status_v55": "strongest_fusion_preliminary_path" if preliminary_ready else "needs_more_extracted_rows_or_device_coverage",
        "policy_v55": "Stroth public comparison rows can support a preliminary T29 run only; strict confirmation requires raw/profile-level public structured rows and controls.",
    }


def parse_t30_from_t28_t29(cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:
    # T30 has no independent public parser. It summarizes whether T28/T29 parsers produced reusable anchors.
    t28 = parse_t28(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    t29 = parse_t29(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    reusable = []
    if t28.get("n_rows_v55", 0):
        reusable.append("T28_DB5_summary_regression_anchor")
    if t29.get("n_normalized_rows_v55", 0):
        reusable.append("T29_Stroth_preliminary_comparison_rows")
    return {
        "test_id": "T30",
        "parser_status_v55": "derived_dependency_anchors_available_nonconfirm" if reusable else "no_independent_public_parser",
        "rows": [],
        "normalized_rows": [],
        "downloads": [],
        "n_rows_v55": 0,
        "strict_confirm_ready_v55": False,
        "preliminary_public_test_ready_v55": False,
        "reusable_anchor_sources_v55": reusable,
        "policy_v55": "T30 remains a secondary diagnostic; it may reuse exact T28/T29 rows if a future release exposes row-level curvature/residual inputs.",
        "t28_summary_v55": {k: t28.get(k) for k in ["parser_status_v55", "n_rows_v55", "n_structured_attachment_candidates_v55"]},
        "t29_summary_v55": {k: t29.get(k) for k in ["parser_status_v55", "n_normalized_rows_v55", "devices_found_v55", "preliminary_public_test_ready_v55"]},
    }


def parse_fusion_public_source(test_id: str, cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:
    tid = test_id.upper()
    if tid == "T26":
        return parse_t26(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    if tid == "T27":
        return parse_t27(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    if tid == "T28":
        return parse_t28(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    if tid == "T29":
        return parse_t29(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    if tid == "T30":
        return parse_t30_from_t28_t29(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    return {"test_id": tid, "parser_status_v55": "not_a_fusion_public_parser_test", "rows": [], "strict_confirm_ready_v55": False}

# ---------------------------------------------------------------------------
# v56 extraction upgrades: add PyMuPDF block-level extraction and conservative
# source-anchor fallbacks. These do not create strict confirmations.
# ---------------------------------------------------------------------------

def _extract_pdf_blocks_v56(data: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    meta = {"method": "pymupdf_blocks_v56", "block_count": 0, "errors": []}
    try:
        import fitz  # type: ignore
        doc = fitz.open(stream=data, filetype="pdf")
        for page_i, page in enumerate(doc):
            try:
                for b_i, b in enumerate(page.get_text("blocks") or []):
                    txt = _norm_space(str(b[4] if len(b) > 4 else ""))
                    if txt:
                        blocks.append({"page": page_i+1, "block_index": b_i, "text": txt})
            except Exception as e:
                meta["errors"].append(f"page_{page_i+1}_blocks_failed:{type(e).__name__}:{e}")
    except Exception as e:
        meta["errors"].append(f"pymupdf_blocks_failed:{type(e).__name__}:{e}")
    meta["block_count"] = len(blocks)
    return blocks, meta


def _block_rows_v56(blocks: Sequence[Dict[str, Any]], *, include_regex: str, min_numbers: int, source_label: str, source_url: str, test_id: str, row_type: str, max_rows: int = 300) -> List[Dict[str, Any]]:
    rows=[]; inc=re.compile(include_regex, re.I); seen=set()
    for b in blocks:
        txt=_norm_space(str(b.get('text') or ''))
        if not txt or not inc.search(txt):
            continue
        nums=_numbers(txt)
        if len(nums) < min_numbers and not re.search(r'table|database|comparison|survey|regression|scaling', txt, re.I):
            continue
        key=(b.get('page'), b.get('block_index'), txt[:180])
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            'test_id': test_id,
            'row_type_v55': row_type,
            'source_label_v55': source_label,
            'source_url_v55': source_url,
            'page_v55': b.get('page'),
            'block_index_v56': b.get('block_index'),
            'line_text_v55': txt[:2000],
            'context_v55': txt[:3000],
            'devices_v55': '|'.join(_find_devices(txt)),
            'numeric_values_v55': '|'.join(str(x) for x in nums[:25]),
            'n_numeric_values_v55': len(nums),
            'extraction_method_v55': 'pymupdf_block_text_no_ocr_v56',
            'confirm_allowed_from_row_v55': False,
        })
        if len(rows) >= max_rows:
            break
    return rows


def _source_anchor_rows_v56(test_id: str) -> List[Dict[str, Any]]:
    rows=[]
    for src in PUBLIC_FUSION_SOURCES_V55.get(test_id, []):
        label=src.get('label','')
        devices='|'.join(_find_devices(label))
        rows.append({
            'test_id': test_id,
            'row_type_v55': 'expected_source_anchor_no_measurement_rows_v56',
            'source_label_v55': label,
            'source_url_v55': src.get('url'),
            'devices_v55': devices,
            'line_text_v55': label,
            'context_v55': 'Known public source anchor; no physical row was extracted from text/table layer in this run.',
            'n_numeric_values_v55': 0,
            'numeric_values_v55': '',
            'extraction_method_v55': 'source_manifest_anchor_v56',
            'confirm_allowed_from_row_v55': False,
        })
    return rows

# Keep references to v55 implementations.
_parse_t26_v55 = parse_t26
_parse_t27_v55 = parse_t27
_parse_t28_v55 = parse_t28
_parse_t29_v55 = parse_t29


def parse_t29(cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:  # type: ignore[override]
    res = _parse_t29_v55(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    rows = list(res.get('rows') or [])
    downloads = list(res.get('downloads') or [])
    # If table/line extraction got nothing, retry with block extraction on cached/downloaded PDFs.
    if not rows:
        for source in PUBLIC_FUSION_SOURCES_V55['T29']:
            data, meta = _source_download(cache_dir / 'v56_fusion_public_sources' / 'T29', source, timeout, force)
            downloads.append(meta)
            if not data:
                continue
            blocks, bm = _extract_pdf_blocks_v56(data)
            include = r"W\s*7\s*[-–]?\s*X|W\s*7\s*[-–]?\s*AS|ASDEX|AUG|stellarator|tokamak|confinement|transport|χ|chi|tau|τ|density|temperature|power|database|comparison|ISS04|IPB98|H98|neoclassical|turbulent"
            rows.extend(_block_rows_v56(blocks, include_regex=include, min_numbers=1, source_label=source['label'], source_url=source['url'], test_id='T29', row_type='stroth_comparison_pdf_block_v56'))
            meta['block_extract_meta_v56'] = bm
    source_anchor_only = False
    if not rows:
        rows = _source_anchor_rows_v56('T29')
        source_anchor_only = True
    norm_rows=[]
    if not source_anchor_only:
        for r in rows:
            text=' '.join(str(r.get(k,'')) for k in ['row_text_v55','line_text_v55','context_v55','header_guess_v55'])
            devices=_find_devices(text)
            nums=_numbers(text)
            for dev in devices[:4]:
                norm_rows.append({'test_id':'T29','device_v55':dev,'device_type_v55':_device_type(dev,text),'source_label_v55':r.get('source_label_v55'),'source_url_v55':r.get('source_url_v55'),'page_v55':r.get('page_v55'),'raw_row_type_v55':r.get('row_type_v55'),'candidate_transport_or_confinement_text_v55':_norm_space(text)[:2500],'numeric_values_v55':'|'.join(str(x) for x in nums[:25]),'n_numeric_values_v55':len(nums),'extraction_method_v55':'derived_device_row_from_stroth_pdf_text_or_table_v56','confirm_allowed_from_row_v55':False,'preliminary_allowed_from_row_v55':True})
    seen=set(); dedup=[]
    for r in norm_rows:
        key=(r.get('device_v55'), r.get('page_v55'), (r.get('candidate_transport_or_confinement_text_v55') or '')[:160])
        if key in seen: continue
        seen.add(key); dedup.append(r)
    devices=sorted(set(r.get('device_v55') for r in dedup if r.get('device_v55')))
    has_w7x=any(str(d).upper().replace(' ','')=='W7-X' for d in devices)
    has_tok=any(_device_type(str(d))=='tokamak' for d in devices) or any(str(d) in {'AUG','DIII-D','JET'} for d in devices)
    preliminary_ready=bool((not source_anchor_only) and len(dedup)>=3 and has_w7x and has_tok)
    res.update({'parser_status_v55':'preliminary_structured_public_rows_extracted_v56' if preliminary_ready else ('source_anchor_only_no_pdf_rows_v56' if source_anchor_only else 'candidate_public_rows_extracted_underpowered_v56'), 'rows':rows, 'normalized_rows':dedup, 'downloads':downloads, 'n_rows_v55':0 if source_anchor_only else len(rows), 'n_normalized_rows_v55':len(dedup), 'devices_found_v55':devices, 'preliminary_public_test_ready_v55':preliminary_ready, 'preliminary_status_v55':'strongest_fusion_preliminary_path_v56' if preliminary_ready else ('source_anchor_only_not_preliminary' if source_anchor_only else 'needs_more_extracted_rows_or_device_coverage'), 'policy_v55':'v56 block extraction added; source anchors are diagnostics only and never confirmations.'})
    return res


def _parse_with_anchor_fallback_v56(base_func, test_id: str, cache_dir: Path, timeout: int, force: bool, max_tables: int) -> Dict[str, Any]:
    res=base_func(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    if not (res.get('rows') or res.get('attachment_candidates')):
        anchors=_source_anchor_rows_v56(test_id)
        res.update({'rows':anchors,'n_rows_v55':0,'parser_status_v55':f'source_anchor_only_no_public_rows_extracted_v56','preliminary_status_v55':'source_anchor_only_not_evidence','policy_v55':str(res.get('policy_v55','')) + ' v56 source anchors are diagnostic only.'})
    return res


def parse_t26(cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:  # type: ignore[override]
    return _parse_with_anchor_fallback_v56(_parse_t26_v55, 'T26', cache_dir, timeout, force, max_tables)


def parse_t27(cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:  # type: ignore[override]
    return _parse_with_anchor_fallback_v56(_parse_t27_v55, 'T27', cache_dir, timeout, force, max_tables)


def parse_t28(cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:  # type: ignore[override]
    return _parse_with_anchor_fallback_v56(_parse_t28_v55, 'T28', cache_dir, timeout, force, max_tables)


def parse_fusion_public_source(test_id: str, cache_dir: Path, timeout: int = 45, force: bool = False, max_tables: int = 80) -> Dict[str, Any]:  # type: ignore[override]
    tid=test_id.upper()
    if tid=='T26': return parse_t26(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    if tid=='T27': return parse_t27(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    if tid=='T28': return parse_t28(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    if tid=='T29': return parse_t29(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    if tid=='T30': return parse_t30_from_t28_t29(cache_dir, timeout=timeout, force=force, max_tables=max_tables)
    return {'test_id': tid, 'parser_status_v55': 'not_a_fusion_public_parser_test', 'rows': [], 'strict_confirm_ready_v55': False}
