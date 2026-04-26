#!/usr/bin/env python3
"""
MAT5 public-literature proxy: Kibble-Zurek-like grain-size/cooling-rate scaling.

Prediction proxy:
    xi ~ tau_Q^{nu/(1+nu*z)}, tau_Q ~ 1/cooling_rate.
3D Ising reference values:
    nu=0.63, z=2.02, alpha=nu/(1+nu*z)=~0.277.

v6 improvements:
  - Adds alternative XML/PMC URLs where publisher HTML blocks script access.
  - Adds generic HTML table extraction for columns such as cooling rate, grain size,
    secondary dendrite arm spacing, cell size, precipitate/phase size, etc.
  - Adds conservative parallel-list extraction from prose: cooling rates [..] paired
    with grain sizes [..].
  - Keeps strict evidence gates: no decisive confirm/falsify unless >=2 same-composition
    groups have >=5 distinct cooling rates each and >=10 total clean points.
  - Rejects generic inequality/threshold prose (e.g. "below 200 um", ">1 K/s")
    and additive/composition prose that is not a controlled cooling-rate series.
  - Discovers and parses public supplementary/source-data CSV/XLSX/ZIP links from
    open articles, using the same same-composition evidence gates.
  - Adds near-miss and leave-one-out stability diagnostics for promising n=4 groups.
"""
from __future__ import annotations

import math
import pathlib
import re
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
from typing import Any, Dict, List, Optional, Tuple

from ccdr_lit_public_utils import (
    common_arg_parser,
    download,
    download_text,
    floats_from_text,
    html_tables_from_path,
    json_dumps,
    path_to_text,
    power_law_fit,
    source_record,
    utc_now,
    write_json,
)

SOURCES = [
    {
        "key": "al5zr_mdpi_2023",
        "label": "Li et al. 2023 Al-5Zr master alloy cooling-rate/grain-size",
        "url": "https://www.mdpi.com/2075-4701/13/4/749",
        "alt_urls": ["https://www.mdpi.com/2075-4701/13/4/749/xml"],
        "material_hint": "Al-5Zr",
    },
    {
        "key": "mg_mdpi_2021",
        "label": "Jamel et al. 2021 Mg alloy solidification-rate/grain-size",
        "url": "https://www.mdpi.com/2075-4701/11/8/1264",
        "alt_urls": ["https://www.mdpi.com/2075-4701/11/8/1264/xml"],
        "material_hint": "Mg alloy",
    },
    {
        "key": "alcu_pmc_2025",
        "label": "Cao et al. 2025 Al-13Cu in-situ grain refinement PMC HTML",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12423412/",
        "material_hint": "Al-13Cu",
    },
    {
        "key": "processes_steel_2022",
        "label": "Huang et al. 2022 cooling rate vs austenitic grain size",
        "url": "https://www.mdpi.com/2227-9717/10/6/1101",
        "alt_urls": ["https://www.mdpi.com/2227-9717/10/6/1101/xml"],
        "material_hint": "steel/austenite",
    },
    {
        "key": "sgbm_pmc_2022",
        "label": "Liu et al. 2022 solidification grain-boundary migration cooling-rate table PMC HTML",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9547067/",
        "material_hint": "solidification alloy SGBM",
    },
    {
        "key": "aa7075_pmc_2019",
        "label": "Zuo et al. 2019 AA7075 cooling-rate grain-size open PMC HTML",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6650472/",
        "material_hint": "AA7075",
    },
    {
        "key": "alsi_pmc_2022",
        "label": "Shen et al. 2022 Al-Si alloy cooling-rate microstructure PMC HTML",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8779305/",
        "material_hint": "Al-Si alloy",
    },
    {
        "key": "aa7050_pmc_2024",
        "label": "Gao et al. 2024 AA7050 heating/cooling-rate microstructure PMC HTML",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10817534/",
        "material_hint": "AA7050",
    },
]

ISING_NU = 0.63
ISING_Z = 2.02
ISING_ALPHA = ISING_NU / (1.0 + ISING_NU * ISING_Z)
DIAGNOSTIC_MIN_ROWS_PER_GROUP = 3
DECISIVE_MIN_ROWS_PER_GROUP = 5
DECISIVE_MIN_GROUPS = 2
DECISIVE_MIN_TOTAL_POINTS = 10

COOL_RE = re.compile(r"cool(?:ing)?\s*(?:rate)?|quench|solidification\s*rate|growth\s*rate", re.I)
SIZE_RE = re.compile(r"grain\s*(?:size|diameter)|cell\s*size|dendrite|SDAS|DAS|arm\s*spacing|phase\s*size|particle\s*size|precipitate\s*size|spacing", re.I)
MICRO_UNIT_RE = re.compile(r"(?:u|µ|μ)m|microm|nm|nanom", re.I)
COMPOSITION_RE = re.compile(r"alloy|composition|sample|specimen|material|wt\.?%|at\.?%", re.I)


def plausible_xi_um(xi_um: float, context: str) -> bool:
    """Reject alloy-name/code artifacts such as AA7075 -> 7075 um."""
    if not math.isfinite(xi_um) or xi_um <= 0:
        return False
    if xi_um > 5000:
        return False
    if re.search(r"\bAA\s*7075\b|\b7075\b", context, re.I) and abs(xi_um - 7075.0) < 1e-6:
        return False
    return True


def group_quality(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rates = sorted({round(float(r["cooling_rate_K_per_s"]), 8) for r in rows})
    xis = sorted({round(float(r["xi_um"]), 8) for r in rows})
    materials = sorted({str(r.get("material", "")) for r in rows})
    return {
        "n_unique_cooling_rates": len(rates),
        "n_unique_xi": len(xis),
        "n_material_labels": len(materials),
        "material_labels_sample": materials[:8],
        "rate_values_sample": rates[:8],
        "composition_confounded": len(rates) < max(3, min(5, len(rows) // 2)),
    }


def clean_text(text: str) -> str:
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("μ", "u").replace("µ", "u")
    text = re.sub(r"K\s*[·.]?\s*s\s*[-−]?1", "K/s", text, flags=re.I)
    text = re.sub(r"°C\s*/\s*s|°C\s*s\s*[-−]?1", "K/s", text, flags=re.I)
    return re.sub(r"\s+", " ", text)


def add_point(points: List[Dict[str, Any]], *, source_key: str, material: str, cooling_rate: float, xi_um: float, quantity: str, context: str, confidence: str = "source_specific") -> None:
    if not (math.isfinite(cooling_rate) and math.isfinite(xi_um) and cooling_rate > 0 and xi_um > 0):
        return
    # Sanity ranges: literature cooling rates can be very high, but grain/cell sizes must be plausible.
    if not plausible_xi_um(xi_um, context) or cooling_rate > 1e12:
        return
    # Generic prose extraction must not treat thresholds, forecasts, or additive/composition
    # effects as controlled cooling-rate measurements. Source-specific and table rows are kept.
    if confidence.startswith("generic"):
        ctx = clean_text(context)
        if re.search(r"\b(?:below|less\s+than|under|above|greater\s+than|higher\s+than|at\s+least|at\s+most)\b|[<>]", ctx, re.I):
            return
        if re.search(r"\b(?:addition|added|additions|wt\s*%|particles?|promising|industrial\s+conditions|typically\s+involve|achieving\s+practical)\b", ctx, re.I):
            return
    points.append({
        "source_key": source_key,
        "material": material,
        "cooling_rate_K_per_s": float(cooling_rate),
        "tau_Q_proxy_s_per_K": float(1.0 / cooling_rate),
        "xi_um": float(xi_um),
        "quantity": quantity,
        "confidence": confidence,
        "context": clean_text(context)[:420],
    })


def dedupe_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[Tuple[str, str, str, float, float], Dict[str, Any]] = {}
    for p in points:
        key = (p["source_key"], p["material"], p["quantity"], round(p["cooling_rate_K_per_s"], 8), round(p["xi_um"], 8))
        seen[key] = p
    return list(seen.values())


def maybe_to_um(value: float, context: str) -> float:
    c = context.lower().replace("μ", "u").replace("µ", "u")
    if re.search(r"\bnm\b|nanom", c) and not re.search(r"(?:u|µ|μ)m|microm", c):
        return value / 1000.0
    return value


def extract_rate_size_from_text(context: str) -> List[Tuple[float, float, str]]:
    """Return (cooling_rate, xi_um, tag) pairs from one local text context."""
    c = clean_text(context)
    out: List[Tuple[float, float, str]] = []
    # direct local pairs, either order
    pats = [
        r"(?:cool(?:ing)?\s*rate|solidification\s*rate)[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)\s*K/s.{0,220}?(?:grain\s*size|grain\s*diameter|cell\s*size|SDAS|DAS|arm\s*spacing|phase\s*size|particle\s*size)[^0-9]{0,80}([0-9]+(?:\.[0-9]+)?)\s*((?:u|µ|μ)m|nm|nanom)?",
        r"(?:grain\s*size|grain\s*diameter|cell\s*size|SDAS|DAS|arm\s*spacing|phase\s*size|particle\s*size)[^0-9]{0,80}([0-9]+(?:\.[0-9]+)?)\s*((?:u|µ|μ)m|nm|nanom)?.{0,220}?(?:cool(?:ing)?\s*rate|solidification\s*rate)[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)\s*K/s",
    ]
    for m in re.finditer(pats[0], c, re.I):
        rate = float(m.group(1)); size = maybe_to_um(float(m.group(2)), m.group(0) + " " + (m.group(3) or ""))
        out.append((rate, size, "local_pair_rate_first"))
    for m in re.finditer(pats[1], c, re.I):
        size = maybe_to_um(float(m.group(1)), m.group(0) + " " + (m.group(2) or "")); rate = float(m.group(3))
        out.append((rate, size, "local_pair_size_first"))

    # Parallel list: cooling rates are 0.2, 0.5, 1.0 K/s ... grain sizes are 248, 221, 217 um
    m = re.search(r"cool(?:ing)?\s*rates?[^.;]{0,220}?((?:[0-9]+(?:\.[0-9]+)?(?:\s*(?:,|and|to|-)\s*)?){2,})\s*K/s[^.;]{0,360}?(?:grain\s*sizes?|average\s*grains?|grain\s*diameters?|cell\s*sizes?|SDAS|DAS)[^.;]{0,220}?((?:[0-9]+(?:\.[0-9]+)?(?:\s*(?:,|and|to|-)\s*)?){2,})\s*((?:u|µ|μ)m|nm|nanom)?", c, re.I)
    if not m:
        m = re.search(r"(?:grain\s*sizes?|average\s*grains?|grain\s*diameters?|cell\s*sizes?|SDAS|DAS)[^.;]{0,220}?((?:[0-9]+(?:\.[0-9]+)?(?:\s*(?:,|and|to|-)\s*)?){2,})\s*((?:u|µ|μ)m|nm|nanom)?[^.;]{0,360}?cool(?:ing)?\s*rates?[^.;]{0,220}?((?:[0-9]+(?:\.[0-9]+)?(?:\s*(?:,|and|to|-)\s*)?){2,})\s*K/s", c, re.I)
        if m:
            sizes = floats_from_text(m.group(1)); rates = floats_from_text(m.group(3)); unit_context = m.group(0) + " " + (m.group(2) or "")
            if len(rates) == len(sizes) and len(rates) >= 3:
                out.extend((r, maybe_to_um(x, unit_context), "parallel_lists_size_first") for r, x in zip(rates, sizes))
    else:
        rates = floats_from_text(m.group(1)); sizes = floats_from_text(m.group(2)); unit_context = m.group(0) + " " + (m.group(3) or "")
        if len(rates) == len(sizes) and len(rates) >= 3:
            out.extend((r, maybe_to_um(x, unit_context), "parallel_lists_rate_first") for r, x in zip(rates, sizes))
    return out



def discover_data_links(article_path: pathlib.Path, base_url: str) -> List[str]:
    """Find public supplementary/source-data links in an open article HTML page."""
    try:
        raw = article_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        raw = article_path.read_bytes().decode("utf-8", errors="replace")
    links: List[str] = []
    for m in re.finditer(r"href=[\"']([^\"']+(?:\.xlsx|\.xls|\.csv|\.tsv|\.txt|\.dat|\.zip|source-data[^\"']*|supplementary[^\"']*))[\"']", raw, re.I):
        links.append(urljoin(base_url, m.group(1)))
    for m in re.finditer(r"https?://[^\s\"']+(?:\.xlsx|\.xls|\.csv|\.tsv|\.txt|\.dat|\.zip)", raw, re.I):
        links.append(m.group(0).rstrip("),.;"))
    seen: List[str] = []
    for u in links:
        if u not in seen:
            seen.append(u)
    return seen[:20]


def xlsx_sheets(path: pathlib.Path) -> Dict[str, List[List[str]]]:
    """Minimal stdlib XLSX reader, enough for supplementary numeric tables."""
    out: Dict[str, List[List[str]]] = {}
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    try:
        with zipfile.ZipFile(path) as zf:
            shared: List[str] = []
            if "xl/sharedStrings.xml" in zf.namelist():
                root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in root.findall("a:si", ns):
                    shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
            wb = ET.fromstring(zf.read("xl/workbook.xml"))
            rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            rid_to_target = {rel.attrib.get("Id"): rel.attrib.get("Target") for rel in rels_root}
            for sheet in wb.findall(".//a:sheet", ns):
                name = sheet.attrib.get("name", "sheet")
                rid = sheet.attrib.get("{%s}id" % ns["r"])
                target = rid_to_target.get(rid, "")
                if not target:
                    continue
                candidates = ["xl/" + target.lstrip("/"), "xl/worksheets/" + pathlib.Path(target).name, target.lstrip("/")]
                sheet_path = next((c for c in candidates if c in zf.namelist()), None)
                if not sheet_path:
                    continue
                sroot = ET.fromstring(zf.read(sheet_path))
                rows: List[List[str]] = []
                for row in sroot.findall(".//a:sheetData/a:row", ns):
                    vals: List[str] = []
                    last_col = -1
                    for c in row.findall("a:c", ns):
                        ref = c.attrib.get("r", "")
                        col_letters = re.sub(r"\d+", "", ref)
                        col_idx = 0
                        for ch in col_letters:
                            col_idx = col_idx * 26 + (ord(ch.upper()) - 64)
                        col_idx -= 1
                        while last_col + 1 < col_idx:
                            vals.append(""); last_col += 1
                        vtag = c.find("a:v", ns)
                        val = vtag.text if vtag is not None and vtag.text is not None else ""
                        if c.attrib.get("t") == "s":
                            try:
                                val = shared[int(val)]
                            except Exception:
                                pass
                        elif c.attrib.get("t") == "inlineStr":
                            texts = [t.text or "" for t in c.findall(".//a:t", ns)]
                            val = "".join(texts)
                        vals.append(str(val)); last_col = col_idx
                    if any(v.strip() for v in vals):
                        rows.append(vals)
                if rows:
                    out[name] = rows
    except Exception:
        return {}
    return out


def table_rows_to_points(headers: List[str], rows: List[List[str]], source_key: str, material_hint: str, label: str, confidence: str) -> List[Dict[str, Any]]:
    pts: List[Dict[str, Any]] = []
    headers = [clean_text(h) for h in headers]
    table_text = " ".join([label] + headers + [" ".join(r) for r in rows[:5]])
    if not (COOL_RE.search(table_text) and SIZE_RE.search(table_text)):
        return pts
    cool_cols = [i for i, h in enumerate(headers) if COOL_RE.search(h) or re.search(r"K/s|K\s*s-1|°C/s", h, re.I)]
    size_cols = [i for i, h in enumerate(headers) if SIZE_RE.search(h) or MICRO_UNIT_RE.search(h)]
    for row in rows:
        row_ctx = f"{confidence} {label} headers={headers} row={row}"
        for ci in cool_cols:
            if ci >= len(row):
                continue
            rates = floats_from_text(row[ci])
            if not rates:
                continue
            rate = rates[0]
            if rate <= 0:
                continue
            for si in size_cols:
                if si >= len(row) or si == ci:
                    continue
                vals = floats_from_text(row[si])
                if not vals:
                    continue
                size = maybe_to_um(vals[0], headers[si] + " " + row[si])
                quantity = headers[si] or "microstructure length"
                mat = material_hint
                for j, h in enumerate(headers):
                    if j < len(row) and j not in {ci, si} and COMPOSITION_RE.search(h) and row[j].strip() and not floats_from_text(row[j]):
                        mat = f"{material_hint} / {clean_text(row[j])}"
                        break
                if mat == material_hint and row and not floats_from_text(row[0]) and len(row[0]) < 80:
                    mat = f"{material_hint} / {clean_text(row[0])}"
                add_point(pts, source_key=source_key, material=mat, cooling_rate=rate, xi_um=size, quantity=quantity, context=row_ctx, confidence=confidence)
    return dedupe_points(pts)


def extract_points_from_downloaded_data(path: pathlib.Path, source_key: str, material_hint: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pts: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()
    note: Dict[str, Any] = {"path": str(path), "kind": suffix}
    try:
        if suffix in {".xlsx", ".xls"} or path.read_bytes()[:2] == b"PK":
            sheets = xlsx_sheets(path)
            if sheets:
                note["sheets"] = list(sheets.keys())[:20]
                for name, rows in sheets.items():
                    for hi, header in enumerate(rows[:30]):
                        pts.extend(table_rows_to_points(header, rows[hi+1:], source_key, material_hint, f"sheet {name} header_row {hi}", "source_data_table"))
            elif suffix == ".zip" or zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as zf:
                    note["zip_members_sample"] = zf.namelist()[:30]
                    tmpdir = pathlib.Path(str(path) + "_unzipped")
                    tmpdir.mkdir(exist_ok=True)
                    for name in zf.namelist():
                        if not re.search(r"\.(xlsx|xls|csv|tsv|txt|dat)$", name, re.I):
                            continue
                        target = tmpdir / pathlib.Path(name).name
                        target.write_bytes(zf.read(name))
                        got, _ = extract_points_from_downloaded_data(target, source_key, material_hint)
                        pts.extend(got)
            else:
                note["warning"] = "PK-like file but not parseable as xlsx/zip"
        elif suffix in {".csv", ".tsv", ".txt", ".dat"}:
            txt = path_to_text(path)
            sep = "\t" if "\t" in txt[:2000] else (";" if ";" in txt[:2000] else ",")
            rows = [re.split(sep, line) for line in txt.splitlines() if line.strip()]
            for hi, header in enumerate(rows[:30]):
                pts.extend(table_rows_to_points(header, rows[hi+1:], source_key, material_hint, f"delimited {path.name} header_row {hi}", "source_data_table"))
    except Exception as exc:
        note["error"] = str(exc)
    pts = dedupe_points(pts)
    note["points"] = len(pts)
    return pts, note

def parse_html_tables_for_points(path: pathlib.Path, source_key: str, material_hint: str) -> List[Dict[str, Any]]:
    pts: List[Dict[str, Any]] = []
    for tbl in html_tables_from_path(path):
        caption = clean_text(tbl.get("caption", ""))
        headers = [clean_text(h) for h in tbl.get("headers", [])]
        rows = tbl.get("rows", [])
        table_text = " ".join([caption] + headers + [" ".join(r) for r in rows[:5]])
        if not (COOL_RE.search(table_text) and SIZE_RE.search(table_text)):
            continue
        cool_cols = [i for i, h in enumerate(headers) if COOL_RE.search(h) or re.search(r"K/s|K\s*s-1|°C/s", h, re.I)]
        size_cols = [i for i, h in enumerate(headers) if SIZE_RE.search(h) or MICRO_UNIT_RE.search(h)]
        for ri, row in enumerate(rows):
            row_ctx = f"table {tbl.get('index')} {caption} headers={headers} row={row}"
            # Header-based extraction.
            for ci in cool_cols:
                if ci >= len(row):
                    continue
                rates = floats_from_text(row[ci])
                if not rates:
                    continue
                for si in size_cols:
                    if si == ci or si >= len(row):
                        continue
                    sizes = floats_from_text(row[si])
                    if not sizes:
                        continue
                    rate = rates[0]
                    size = maybe_to_um(sizes[0], headers[si] + " " + row[si])
                    quantity = headers[si] or "table microstructure size"
                    mat = material_hint
                    comp_cols = [j for j, h in enumerate(headers) if COMPOSITION_RE.search(h)]
                    if comp_cols:
                        j = comp_cols[0]
                        if j < len(row) and row[j].strip() and len(row[j]) < 120:
                            mat = f"{material_hint} / {clean_text(row[j])}"
                    elif headers and len(row) >= 1 and not floats_from_text(row[0]) and len(row[0]) < 80:
                        mat = f"{material_hint} / {row[0]}"
                    add_point(pts, source_key=source_key, material=mat, cooling_rate=rate, xi_um=size, quantity=quantity, context=row_ctx, confidence="html_table")
            # Text fallback inside table row.
            for rate, size, tag in extract_rate_size_from_text(" ".join(row)):
                add_point(pts, source_key=source_key, material=material_hint, cooling_rate=rate, xi_um=size, quantity="table-row microstructure size", context=row_ctx, confidence=f"html_table_{tag}")
    return dedupe_points(pts)


# Source-specific parsers retained from v2.
def parse_al5zr_mdpi(text: str) -> List[Dict[str, Any]]:
    t = clean_text(text)
    pts: List[Dict[str, Any]] = []
    m_rates_920 = re.search(r"remelting temperature is 920 .*?cooling rates.*?([\d.]+)\s*K/s.*?([\d.]+)\s*K/s.*?([\d.]+)\s*K/s", t, re.I)
    m_rates_1320 = re.search(r"remelting temperature is 1320 .*?cooling rates.*?([\d.]+)\s*K/s.*?([\d.]+)\s*K/s.*?([\d.]+)\s*K/s", t, re.I)
    m_sizes = re.search(r"average sizes of grains are\s*([\d.]+),\s*([\d.]+),\s*and\s*([\d.]+)\s*um.*?920\s*(?:°C|C).*?and that are\s*([\d.]+),\s*([\d.]+),\s*and\s*([\d.]+)\s*um.*?1320\s*(?:°C|C)", t, re.I)
    if m_rates_920 and m_rates_1320 and m_sizes:
        rates_920 = [float(m_rates_920.group(i)) for i in (1, 2, 3)]
        rates_1320 = [float(m_rates_1320.group(i)) for i in (1, 2, 3)]
        sizes = [float(m_sizes.group(i)) for i in range(1, 7)]
        for r, x in zip(rates_920, sizes[:3]):
            add_point(pts, source_key="al5zr_mdpi_2023", material="Al-5Zr remelted 920C", cooling_rate=r, xi_um=x, quantity="average grain size", context=m_sizes.group(0))
        for r, x in zip(rates_1320, sizes[3:]):
            add_point(pts, source_key="al5zr_mdpi_2023", material="Al-5Zr remelted 1320C", cooling_rate=r, xi_um=x, quantity="average grain size", context=m_sizes.group(0))
    return dedupe_points(pts)


def parse_alcu_pmc(text: str) -> List[Dict[str, Any]]:
    t = clean_text(text)
    pts: List[Dict[str, Any]] = []
    m = re.search(r"average grain size declined from\s*([\d.]+)\s*um\s*\(?\s*([\d.]+)\s*K/s\)?.{0,120}?to\s*([\d.]+)\s*um\s*\(?\s*([\d.]+)\s*K/s\)?.{0,120}?to\s*([\d.]+)\s*um\s*at\s*([\d.]+)\s*K/s", t, re.I)
    if m:
        triples = [(float(m.group(2)), float(m.group(1))), (float(m.group(4)), float(m.group(3))), (float(m.group(6)), float(m.group(5)))]
        for r, x in triples:
            add_point(pts, source_key="alcu_pmc_2025", material="Al-13Cu", cooling_rate=r, xi_um=x, quantity="average grain size", context=m.group(0))
    return dedupe_points(pts)


def parse_generic_cooling_grain_text(text: str, source_key: str, material_hint: str) -> List[Dict[str, Any]]:
    t = clean_text(text)
    pts: List[Dict[str, Any]] = []
    # Use sentence-like chunks plus larger local windows because PMC table text often loses punctuation.
    chunks = re.split(r"(?<=[.!?;])\s+", t)
    for chunk in chunks:
        if len(chunk) > 1200:
            # split long table-ish blobs around cooling mentions
            starts = [m.start() for m in re.finditer(r"cool(?:ing)?\s*rate|solidification\s*rate", chunk, re.I)]
            subchunks = [chunk[max(0, s - 250): s + 700] for s in starts[:20]]
        else:
            subchunks = [chunk]
        for sub in subchunks:
            if not (COOL_RE.search(sub) and SIZE_RE.search(sub)):
                continue
            for rate, size, tag in extract_rate_size_from_text(sub):
                add_point(pts, source_key=source_key, material=material_hint, cooling_rate=rate, xi_um=size, quantity="average microstructure length", context=sub, confidence=f"generic_{tag}")
    return dedupe_points(pts)


SOURCE_SPECIFIC = {
    "al5zr_mdpi_2023": parse_al5zr_mdpi,
    "alcu_pmc_2025": parse_alcu_pmc,
}


def group_key(p: Dict[str, Any]) -> str:
    return f"{p['source_key']}::{p['material']}::{p['quantity']}"


def analyze_optima(by_group: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    opt: Dict[str, Any] = {}
    for group, rows in by_group.items():
        rows_s = sorted(rows, key=lambda r: r["cooling_rate_K_per_s"])
        if len(rows_s) < 3:
            continue
        gbd = [1.0 / (r["xi_um"] ** 2) for r in rows_s]
        idx = min(range(len(gbd)), key=lambda i: gbd[i])
        opt[group] = {
            "n_rows": len(rows_s),
            "cooling_rate_K_per_s_at_min_gb_density_proxy": rows_s[idx]["cooling_rate_K_per_s"],
            "xi_um_at_min_gb_density_proxy": rows_s[idx]["xi_um"],
            "gb_density_proxy_1_over_xi2": gbd[idx],
            "is_interior_minimum": bool(0 < idx < len(rows_s) - 1),
            "edge_or_interior": "interior" if 0 < idx < len(rows_s) - 1 else "edge_monotonic_or_incomplete",
            "note": "Interior minimum is required for an optimum-cooling claim; edge minimum only shows monotonic trend or incomplete cooling-rate span.",
        }
    return opt



def leave_one_out_exponents(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Small-n robustness diagnostic; does not affect confirm/falsify gates."""
    if len(rows) < 4:
        return {"available": False, "reason": "need at least 4 rows"}
    vals: List[float] = []
    rows_s = sorted(rows, key=lambda r: r["tau_Q_proxy_s_per_K"])
    for i in range(len(rows_s)):
        sub = rows_s[:i] + rows_s[i+1:]
        fit = power_law_fit([r["tau_Q_proxy_s_per_K"] for r in sub], [r["xi_um"] for r in sub])
        if "exponent" in fit and math.isfinite(float(fit["exponent"])):
            vals.append(float(fit["exponent"]))
    if not vals:
        return {"available": False, "reason": "no finite leave-one-out fits"}
    return {
        "available": True,
        "min_exponent": min(vals),
        "max_exponent": max(vals),
        "mean_exponent": sum(vals)/len(vals),
        "max_abs_frac_error_vs_ising": max(abs(v - ISING_ALPHA) / ISING_ALPHA for v in vals),
        "n_leave_one_out_fits": len(vals),
    }

def analyze(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for p in dedupe_points(points):
        by_group.setdefault(group_key(p), []).append(p)

    fits: Dict[str, Any] = {}
    for group, rows in by_group.items():
        if len(rows) < DIAGNOSTIC_MIN_ROWS_PER_GROUP:
            continue
        rows_s = sorted(rows, key=lambda r: r["tau_Q_proxy_s_per_K"])
        fit = power_law_fit([r["tau_Q_proxy_s_per_K"] for r in rows_s], [r["xi_um"] for r in rows_s])
        fit["alpha_expected_3d_ising"] = ISING_ALPHA
        fit["abs_frac_error_vs_ising"] = abs(fit.get("exponent", float("nan")) - ISING_ALPHA) / ISING_ALPHA if "exponent" in fit else float("nan")
        fit["n_rows"] = len(rows_s)
        fit["leave_one_out_stability"] = leave_one_out_exponents(rows_s)
        q = group_quality(rows_s)
        fit.update(q)
        decisive_ok = (
            len(rows_s) >= DECISIVE_MIN_ROWS_PER_GROUP
            and q["n_unique_cooling_rates"] >= DECISIVE_MIN_ROWS_PER_GROUP
            and not q["composition_confounded"]
        )
        fit["evidence_class"] = "decisive_candidate" if decisive_ok else ("composition_confounded_or_too_few_rates" if len(rows_s) >= DECISIVE_MIN_ROWS_PER_GROUP else "diagnostic_only")
        fits[group] = fit

    pooled = power_law_fit([p["tau_Q_proxy_s_per_K"] for p in points], [p["xi_um"] for p in points]) if len(points) >= 3 else {"status": "insufficient", "n": len(points)}
    if "exponent" in pooled:
        pooled["alpha_expected_3d_ising"] = ISING_ALPHA
        pooled["abs_frac_error_vs_ising"] = abs(pooled["exponent"] - ISING_ALPHA) / ISING_ALPHA
        pooled["warning"] = "Mixed-material pooled fit is diagnostic only and must not confirm/falsify MAT5."

    diagnostic_fits = [v for v in fits.values() if math.isfinite(float(v.get("abs_frac_error_vs_ising", float("nan"))))]
    decisive_fits = [v for v in diagnostic_fits if v.get("evidence_class") == "decisive_candidate"]
    within_15_decisive = [v for v in decisive_fits if v["abs_frac_error_vs_ising"] <= 0.15]
    within_15_diagnostic = [v for v in diagnostic_fits if v["abs_frac_error_vs_ising"] <= 0.15]
    outside_30_diagnostic = [v for v in diagnostic_fits if v["abs_frac_error_vs_ising"] > 0.30]

    optima = analyze_optima(by_group)
    interior_optima = [v for v in optima.values() if v.get("is_interior_minimum")]

    decisive_ready = len(decisive_fits) >= DECISIVE_MIN_GROUPS and len(points) >= DECISIVE_MIN_TOTAL_POINTS
    if decisive_ready:
        if len(within_15_decisive) >= math.ceil(0.5 * len(decisive_fits)):
            status = "confirm_like" if interior_optima else "partial_exponent_only"
        elif not within_15_decisive:
            status = "falsify_like"
        else:
            status = "partial_mixed"
    else:
        if diagnostic_fits and outside_30_diagnostic and not within_15_diagnostic:
            status = "data_limited_negative"
        elif within_15_diagnostic:
            status = "partial_supportive"
        elif diagnostic_fits:
            status = "data_limited"
        else:
            status = "data_limited"

    near_miss = [
        {"group": g, "n_rows": v.get("n_rows"), "n_unique_cooling_rates": v.get("n_unique_cooling_rates"), "exponent": v.get("exponent"), "abs_frac_error_vs_ising": v.get("abs_frac_error_vs_ising"), "evidence_class": v.get("evidence_class"), "leave_one_out_stability": v.get("leave_one_out_stability")}
        for g, v in fits.items()
        if v.get("n_unique_cooling_rates", 0) >= 4 or (math.isfinite(float(v.get("abs_frac_error_vs_ising", float("nan")))) and v.get("abs_frac_error_vs_ising") <= 0.20)
    ]

    return {
        "status": status,
        "n_points": len(points),
        "n_groups_total": len(by_group),
        "n_groups_with_diagnostic_fit": len(diagnostic_fits),
        "n_groups_with_decisive_fit": len(decisive_fits),
        "n_groups_with_exponent_within_15pct": len(within_15_diagnostic),
        "alpha_expected_3d_ising": ISING_ALPHA,
        "fits_by_material": fits,
        "pooled_fit_warning_mixed_materials": pooled,
        "gb_density_optimum_tests_by_group": optima,
        "n_interior_gb_density_minima": len(interior_optima),
        "near_miss_groups": near_miss[:10],
        "decisive_requirements": {
            "diagnostic_min_rows_per_group": DIAGNOSTIC_MIN_ROWS_PER_GROUP,
            "decisive_min_rows_per_group": DECISIVE_MIN_ROWS_PER_GROUP,
            "decisive_min_groups": DECISIVE_MIN_GROUPS,
            "decisive_min_total_points": DECISIVE_MIN_TOTAL_POINTS,
            "needs_interior_gb_density_minimum_for_full_confirm": True,
        },
        "metric_interpretation": "Confirm/falsify is gated to >=2 same-composition groups with >=5 distinct cooling rates. Composition series at fixed cooling rate are diagnostic only; edge GB-density minima still do not prove an optimum cooling rate.",
    }


def download_first(src: Dict[str, Any], cache: str, force: bool, timeout: int = 25) -> Tuple[pathlib.Path, str, str, List[str]]:
    errors: List[str] = []
    for url in [src["url"]] + list(src.get("alt_urls", [])):
        try:
            path, text = download_text(url, cache, force=force, timeout=timeout)
            return path, text, url, errors
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def main() -> None:
    parser = common_arg_parser(__doc__ or "MAT5 public test")
    args = parser.parse_args()

    downloaded = []
    points: List[Dict[str, Any]] = []
    extraction_notes = []
    for src in SOURCES:
        try:
            path, text, used_url, prior_errors = download_first(src, args.cache, args.force, timeout=25)
            downloaded.append(source_record(src["label"], used_url, path, notes=("fallback used after: " + " | ".join(prior_errors)) if prior_errors else None))
            mat_hint = src.get("material_hint", src["key"])
            pts: List[Dict[str, Any]] = []
            pts.extend(parse_html_tables_for_points(path, src["key"], mat_hint))
            if src["key"] in SOURCE_SPECIFIC:
                pts.extend(SOURCE_SPECIFIC[src["key"]](text))
            pts.extend(parse_generic_cooling_grain_text(text, src["key"], mat_hint))
            data_links = discover_data_links(path, used_url)
            data_notes = []
            for du in data_links[:12]:
                try:
                    dpath = download(du, args.cache, force=args.force, timeout=25)
                    downloaded.append(source_record(src["label"] + " discovered/source data", du, dpath))
                    got, dnote = extract_points_from_downloaded_data(dpath, src["key"], mat_hint)
                    pts.extend(got)
                    data_notes.append(dnote)
                except Exception as exc:
                    data_notes.append({"url": du, "error": str(exc)})
            pts = dedupe_points(pts)
            points.extend(pts)
            extraction_notes.append({"source_key": src["key"], "used_url": used_url, "text_chars": len(text), "points": len(pts), "table_count": len(html_tables_from_path(path)), "data_links_seen": data_links[:12], "data_notes": data_notes[:12]})
        except Exception as exc:
            downloaded.append(source_record(src["label"], src["url"], notes=f"download/extract failed: {exc}"))
            extraction_notes.append({"source_key": src["key"], "error": str(exc)})

    points = dedupe_points(points)
    result = {
        "prediction_id": "MAT5",
        "test_name": "Kibble-Zurek cooling-rate exponent and GB-density proxy from public literature",
        "generated_utc": utc_now(),
        "analysis": analyze(points),
        "extracted_points": points,
        "downloaded_sources": downloaded,
        "extraction_notes": extraction_notes,
        "falsification_logic": {
            "confirm_like": "At least two same-composition public datasets with >=5 distinct cooling rates each fit alpha within 10-15% of 3D Ising Kibble-Zurek alpha≈0.277 and show an interior GB-density minimum.",
            "partial_exponent_only": "Exponent criterion passes but no interior GB-density optimum is shown.",
            "data_limited_negative": "Diagnostic 3-point fits lean away from the target exponent, but evidence is too thin for decisive falsification.",
            "falsify_like": "Adequate same-material open data give exponents far outside the 10-15% window or no consistent optimum.",
            "data_limited": "Open literature text supports cooling-rate scaling, but extracted data are too few/mixed to be decisive.",
        },
    }
    write_json(result, args.outdir, "test02_mat5_kibble_zurek_gb_density.json")
    print(json_dumps(result))
    if args.strict and result["analysis"]["status"] in {"data_limited", "data_limited_negative", "partial_supportive", "partial_exponent_only", "partial_mixed"}:
        raise SystemExit("MAT5 extraction is not decisive")


if __name__ == "__main__":
    main()
