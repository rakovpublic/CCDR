#!/usr/bin/env python3
"""
MAT9 public-literature proxy: moire/twisted-lattice phonon tunability.

Prediction target:
    Twist angle tunes a phononic bandgap or characteristic phonon-band frequency.
Metric:
    Clean numeric angle-bandgap/frequency pairs should show strong tunability with twist angle.

v6 improvements:
  - Adds PMC/Nature-Communications sources likely to expose Source Data links.
  - Discovers supplementary/source-data links from article HTML and parses CSV/XLSX with
    stdlib-only XLSX support when possible.
  - Accepts cm^-1 Raman shifts as a frequency proxy and converts to GHz.
  - Keeps strict filters against LaTeX/source-code/affiliation garbage.
  - Adds row-wise source-data fallback for sheets where angle/frequency labels are
    embedded in cells rather than clean headers.
  - Parses inline-string XLSX cells, TSV/JSON-like source-data files, and records
    source-data blockers so failed supplements are easier to fix.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from ccdr_lit_public_utils import (
    common_arg_parser,
    download,
    download_arxiv_source,
    download_text,
    floats_from_text,
    html_tables_from_path,
    json_dumps,
    path_to_text,
    power_law_fit,
    source_record,
    spearman,
    utc_now,
    write_json,
)

SOURCES = [
    {
        "key": "lamparski_1912_12568",
        "label": "Lamparski et al. tBLG phonon spectra vs 692 twist angles arXiv",
        "url": "https://arxiv.org/abs/1912.12568",
        "arxiv_id": "1912.12568",
    },
    {
        "key": "liu_2112_13240",
        "label": "Liu et al. magic-angle tBLG phonons arXiv/PMC",
        "url": "https://arxiv.org/abs/2112.13240",
        "arxiv_id": "2112.13240",
        "alt_urls": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC9562463/"],
    },
    {
        "key": "xie_2305_16640",
        "label": "Xie & Liu 2023 lattice distortions/moire phonons in MATBG arXiv",
        "url": "https://arxiv.org/abs/2305.16640",
        "arxiv_id": "2305.16640",
    },
    {
        "key": "liao_natcomm_2020_mos2",
        "label": "Liao et al. 2020 twist-angle controlled MoS2 homostructures PMC/Nature source-data search",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7195481/",
        "alt_urls": ["https://www.nature.com/articles/s41467-020-16056-4"],
        # Candidate static links are tried only if article HTML discovery does not find them.
        "candidate_source_data_urls": [
            "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-16056-4/MediaObjects/41467_2020_16056_MOESM3_ESM.zip",
            *[f"https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-16056-4/MediaObjects/41467_2020_16056_MOESM{i}_ESM.xlsx" for i in range(1, 9)],
        ],
    },
    {
        "key": "qin_pmc_2024_phonon_polarizer",
        "label": "Qin et al. 2024 moire pattern controlled phonon polarizer PMC HTML",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11180428/",
    },
    {
        "key": "apl_2022_tml",
        "label": "Phononic twisted moire lattice with quasicrystalline patterns AIP HTML",
        "url": "https://pubs.aip.org/aip/apl/article/121/14/142202/2834360/Phononic-twisted-moire-lattice-with",
    },
]

PHYSICS_WORDS = re.compile(r"\b(phonon|phononic|acoustic|raman|band\s*gap|bandgap|gap|mode|frequency|wavenumber|eigenfrequency|dispersion|moire|moir[eé]|twist)", re.I)
REQUIRED_FREQ_WORDS = re.compile(r"\b(frequency|frequencies|eigenfrequency|omega|wavenumber|raman\s*shift|phonon|phononic|acoustic|band\s*gap|bandgap|gap|mode|THz|GHz|MHz|cm\s*[-^−]?\s*1)", re.I)
ANGLE_WORDS = re.compile(r"\b(twist|twisted|angle|theta|degree|degrees|deg)\b|°", re.I)
BAD_CONTEXT = re.compile(
    r"\\(?:documentclass|begin|end|includegraphics|caption|label|ref|cite|affiliation|author|title|bibliography|bibitem)|"
    r"\b(affiliation|university|department|school|institute|shanghai|china|corresponding|email|figure width|textwidth|includegraphics|copyright|received|accepted|published)\b",
    re.I,
)
UNITS = r"GHz|MHz|THz|cm\s*[-^−]?\s*1|cm-1|cm−1"


def clean_text(text: str) -> str:
    text = text.replace("−", "-").replace("μ", "u").replace("µ", "u")
    text = text.replace("\\,", " ").replace("~", " ")
    return re.sub(r"\s+", " ", text)


def freq_to_ghz(value: float, unit: str) -> Optional[float]:
    u = unit.lower().replace(" ", "").replace("−", "-")
    if u == "ghz":
        ghz = value
    elif u == "mhz":
        ghz = value / 1000.0
    elif u == "thz":
        if value > 300:  # catches affiliation/address artifacts such as 200031 THz
            return None
        ghz = value * 1000.0
    elif u in {"cm-1", "cm^-1"}:
        if value > 10000:
            return None
        ghz = value * 29.9792458
    else:
        return None
    if not (1e-4 <= ghz <= 3e5):
        return None
    return float(ghz)


def context_is_clean(context: str) -> Tuple[bool, str]:
    c = clean_text(context)
    if BAD_CONTEXT.search(c):
        return False, "bad_context_latex_author_affiliation_or_figure_markup"
    if not PHYSICS_WORDS.search(c):
        return False, "missing_physics_words"
    if not REQUIRED_FREQ_WORDS.search(c):
        return False, "missing_frequency_or_gap_words"
    if not ANGLE_WORDS.search(c):
        return False, "missing_angle_words"
    if c.count("\\") >= 3 and not re.search(r"tabular|theta|angle|twist|THz|GHz|cm", c, re.I):
        return False, "too_much_tex_markup"
    return True, "ok"


def add_pair(pairs: List[Dict[str, Any]], rejected: List[Dict[str, Any]], *, source_key: str, angle: float, freq: float, unit: str, context: str, quantity: str = "frequency_or_bandgap", confidence: str = "text_regex") -> None:
    ok, reason = context_is_clean(context)
    ghz = freq_to_ghz(freq, unit)
    angle_ok = (0.0 < angle <= 60.0)
    if not ok or ghz is None or not angle_ok:
        if ghz is None:
            rej_reason = "frequency_unit_or_range_rejected"
        elif not angle_ok:
            rej_reason = "angle_range_rejected_or_zero_baseline"
        else:
            rej_reason = reason
        rejected.append({
            "source_key": source_key,
            "twist_angle_deg": angle,
            "original_frequency": freq,
            "original_unit": unit,
            "reason": rej_reason,
            "context": clean_text(context)[:260],
        })
        return
    pairs.append({
        "source_key": source_key,
        "twist_angle_deg": float(angle),
        "frequency_GHz": float(ghz),
        "original_frequency": float(freq),
        "original_unit": unit,
        "quantity": quantity,
        "confidence": confidence,
        "context": clean_text(context)[:360],
    })


def discover_data_links(article_path: pathlib.Path, base_url: str) -> List[str]:
    raw = article_path.read_text(encoding="utf-8", errors="replace")
    links: List[str] = []
    for m in re.finditer(r"href=[\"']([^\"']+(?:\.xlsx|\.xls|\.csv|\.tsv|\.txt|\.dat|\.json|\.zip|source-data[^\"']*))[\"']", raw, re.I):
        links.append(urljoin(base_url, m.group(1)))
    # Also catch visible static-content links.
    for m in re.finditer(r"https?://[^\s\"']+(?:\.xlsx|\.xls|\.csv|\.tsv|\.txt|\.dat|\.json|\.zip)", raw, re.I):
        links.append(m.group(0).rstrip("),.;"))
    seen: List[str] = []
    for u in links:
        if u not in seen:
            seen.append(u)
    return seen[:20]


def xlsx_sheets(path: pathlib.Path) -> Dict[str, List[List[str]]]:
    """Minimal stdlib XLSX reader for numeric source-data workbooks."""
    out: Dict[str, List[List[str]]] = {}
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    try:
        with zipfile.ZipFile(path) as zf:
            shared: List[str] = []
            if "xl/sharedStrings.xml" in zf.namelist():
                root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in root.findall("a:si", ns):
                    texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                    shared.append("".join(texts))
            wb = ET.fromstring(zf.read("xl/workbook.xml"))
            rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            rid_to_target = {rel.attrib.get("Id"): rel.attrib.get("Target") for rel in rels_root}
            for sheet in wb.findall(".//a:sheet", ns):
                name = sheet.attrib.get("name", "sheet")
                rid = sheet.attrib.get("{%s}id" % ns["r"])
                target = rid_to_target.get(rid, "")
                if not target:
                    continue
                sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
                if sheet_path not in zf.namelist():
                    sheet_path = "xl/worksheets/" + pathlib.Path(target).name
                if sheet_path not in zf.namelist():
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
                            vals.append("")
                            last_col += 1
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
                        vals.append(str(val))
                        last_col = col_idx
                    if any(v.strip() for v in vals):
                        rows.append(vals)
                if rows:
                    out[name] = rows
    except Exception:
        return {}
    return out


def extract_pairs_from_table_rows(rows: List[List[str]], source_key: str, label: str, rejected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    if not rows:
        return pairs
    # Try each row as header; source-data sheets often have notes before real headers.
    for hi, header in enumerate(rows[:20]):
        headers = [clean_text(h) for h in header]
        if not any(re.search(r"angle|theta|twist", h, re.I) for h in headers):
            continue
        freq_cols = []
        angle_cols = []
        for i, h in enumerate(headers):
            hl = h.lower()
            if re.search(r"angle|theta|twist", h, re.I):
                angle_cols.append(i)
            if re.search(r"frequency|freq|wavenumber|raman\s*shift|cm\s*[-^−]?\s*1|cm-1|thz|ghz|mode|gap", h, re.I) and not re.search(r"intensity|ratio|amplitude|width|fwhm|wavelength|laser|energy", h, re.I):
                freq_cols.append(i)
            # Mode names such as FA1g in sheets titled Raman characterizations can be peak-position columns.
            if re.search(r"FA1g|A1g|E2g|LB|shear|breathing", h, re.I) and re.search(r"raman|phonon|mode|shift|cm", label, re.I):
                freq_cols.append(i)
        if not angle_cols or not freq_cols:
            continue
        for row in rows[hi + 1:]:
            for ai in angle_cols:
                if ai >= len(row):
                    continue
                av = floats_from_text(row[ai])
                if not av:
                    continue
                angle = av[0]
                if not (0.0 < angle <= 60.0):
                    continue
                for fi in freq_cols:
                    if fi >= len(row) or fi == ai:
                        continue
                    fv = floats_from_text(row[fi])
                    if not fv:
                        continue
                    unit_context = headers[fi] + " " + row[fi] + " " + label
                    unit = "cm-1" if re.search(r"cm\s*[-^−]?\s*1|raman|A1g|E2g|FA1g|wavenumber", unit_context, re.I) else ("THz" if re.search(r"THz", unit_context, re.I) else ("GHz" if re.search(r"GHz", unit_context, re.I) else "cm-1"))
                    context = f"source-data table {label}; headers={headers}; row={row}"
                    add_pair(pairs, rejected, source_key=source_key, angle=angle, freq=fv[0], unit=unit, context=context, quantity=headers[fi] or "source-data phonon frequency", confidence="source_data_table")
    # Row-wise source-data fallback: some supplementary sheets have no clean header row,
    # but each row/caption contains strings like "twist angle 3.5 deg" and
    # "Raman shift 405 cm-1". Keep the same context/range filters as add_pair.
    for row in rows[:5000]:
        row_text = clean_text(" ".join(str(c) for c in row if str(c).strip()))
        if len(row_text) < 8 or len(row_text) > 800:
            continue
        context = f"source-data row {label}; row={row_text}"
        angle_matches = list(re.finditer(r"(?:twist\s*)?(?:angle|theta|θ)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:°|deg|degree|degrees)", row_text, re.I))
        if not angle_matches:
            # fallback: cells like "3.5°" or "3.5 deg" without the word angle
            angle_matches = list(re.finditer(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:°|deg|degree|degrees)\b", row_text, re.I))
        freq_matches = list(re.finditer(rf"([0-9]+(?:\.[0-9]+)?)\s*({UNITS})", row_text, re.I))
        # If unit is only in a nearby header/label, use cm-1 for Raman-like rows.
        if not freq_matches and re.search(r"raman|wavenumber|cm\s*[-^−]?\s*1|A1g|E2g|phonon|mode|gap", label + " " + row_text, re.I):
            numbers = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", row_text)]
            angle_nums = [float(m.group(1)) for m in angle_matches]
            for angle in angle_nums[:3]:
                for num in numbers:
                    if abs(num - angle) < 1e-9:
                        continue
                    if 1.0 <= num <= 1000.0:
                        add_pair(pairs, rejected, source_key=source_key, angle=angle, freq=num, unit="cm-1", context=context, quantity="row-wise source-data phonon frequency", confidence="source_data_row_fallback")
                        break
        else:
            for am in angle_matches[:3]:
                angle = float(am.group(1))
                for fm in freq_matches[:5]:
                    freq = float(fm.group(1)); unit = fm.group(2)
                    # Avoid reusing the same numeric token as both angle and frequency.
                    if abs(freq - angle) < 1e-9 and fm.start() == am.start():
                        continue
                    add_pair(pairs, rejected, source_key=source_key, angle=angle, freq=freq, unit=unit, context=context, quantity="row-wise source-data phonon frequency", confidence="source_data_row_fallback")

    # dedupe
    dedup: Dict[Tuple[str, float, float, str], Dict[str, Any]] = {}
    for p in pairs:
        dedup[(p["source_key"], round(p["twist_angle_deg"], 5), round(p["frequency_GHz"], 5), p.get("quantity", ""))] = p
    return list(dedup.values())


def extract_pairs_from_downloaded_data(path: pathlib.Path, source_key: str, rejected: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    note: Dict[str, Any] = {"path": str(path), "kind": pathlib.Path(path).suffix.lower()}
    suffix = pathlib.Path(path).suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"} or path.read_bytes()[:2] == b"PK":
            sheets = xlsx_sheets(path)
            note["sheets"] = list(sheets.keys())
            for name, rows in sheets.items():
                pairs.extend(extract_pairs_from_table_rows(rows, source_key, name, rejected))
        elif suffix in {".csv", ".tsv", ".txt", ".dat"}:
            text = path_to_text(path)
            rows = [re.split(r",|\t|;", line) for line in text.splitlines() if line.strip()]
            pairs.extend(extract_pairs_from_table_rows(rows, source_key, path.name, rejected))
        elif suffix == ".json":
            text = path_to_text(path)
            try:
                obj = json.loads(text)
                flat_rows: List[List[str]] = []
                def walk(o: Any, prefix: str = "") -> None:
                    if isinstance(o, dict):
                        if any(re.search(r"angle|theta|twist|frequency|raman|gap|phonon", str(k), re.I) for k in o):
                            flat_rows.append([f"{k}: {v}" for k, v in o.items()])
                        for v in o.values():
                            walk(v, prefix)
                    elif isinstance(o, list):
                        for v in o[:10000]:
                            walk(v, prefix)
                walk(obj)
                pairs.extend(extract_pairs_from_table_rows(flat_rows, source_key, path.name + " json", rejected))
            except Exception:
                rows = [re.split(r",|\t|;", line) for line in text.splitlines() if line.strip()]
                pairs.extend(extract_pairs_from_table_rows(rows, source_key, path.name + " json-text", rejected))
        elif suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                note["zip_members_sample"] = zf.namelist()[:20]
                tmpdir = pathlib.Path(str(path) + "_unzipped")
                tmpdir.mkdir(exist_ok=True)
                for name in zf.namelist():
                    if not re.search(r"\.(xlsx|xls|csv|tsv|txt|dat|json)$", name, re.I):
                        continue
                    target = tmpdir / pathlib.Path(name).name
                    target.write_bytes(zf.read(name))
                    got, _ = extract_pairs_from_downloaded_data(target, source_key, rejected)
                    pairs.extend(got)
    except Exception as exc:
        note["error"] = str(exc)
    note["pairs"] = len(pairs)
    return pairs, note


def extract_angle_frequency_pairs(text: str, source_key: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    t = clean_text(text)
    pairs: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    pat_a = rf"((?:twist(?:ing|ed)?\s*)?(?:angle|theta)[^.;]{{0,140}}?([\d.]+)\s*(?:degree|degrees|deg|°)[^.;]{{0,300}}?(?:frequency|frequencies|wavenumber|raman\s*shift|band\s*gap|bandgap|gap|mode|phonon|acoustic)[^.;]{{0,150}}?([\d.]+)\s*({UNITS}))"
    for m in re.finditer(pat_a, t, re.I):
        add_pair(pairs, rejected, source_key=source_key, angle=float(m.group(2)), freq=float(m.group(3)), unit=m.group(4), context=m.group(1))
    pat_b = rf"((?:frequency|frequencies|wavenumber|raman\s*shift|band\s*gap|bandgap|gap|mode|phonon|acoustic)[^.;]{{0,160}}?([\d.]+)\s*({UNITS})[^.;]{{0,250}}?(?:at|for|with|near|around)[^.;]{{0,100}}?([\d.]+)\s*(?:degree|degrees|deg|°))"
    for m in re.finditer(pat_b, t, re.I):
        add_pair(pairs, rejected, source_key=source_key, angle=float(m.group(4)), freq=float(m.group(2)), unit=m.group(3), context=m.group(1))
    # HTML table/text table fallback already handled separately; keep line fallback very strict.
    for raw in re.split(r"\n|\\\\", text):
        line = clean_text(raw)
        if len(line) > 500:
            continue
        if not (ANGLE_WORDS.search(line) and REQUIRED_FREQ_WORDS.search(line) and re.search(UNITS, line, re.I)):
            continue
        ok, _ = context_is_clean(line)
        if not ok:
            continue
        nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", line)]
        unit_m = re.search(UNITS, line, re.I)
        if len(nums) >= 2 and unit_m:
            angle_candidates = [x for x in nums if 0.0 < x <= 60.0]
            if not angle_candidates:
                continue
            angle = angle_candidates[0]
            later = [x for x in nums if abs(x - angle) > 1e-12 and x > 0]
            if later:
                add_pair(pairs, rejected, source_key=source_key, angle=angle, freq=later[0], unit=unit_m.group(0), context=line)
    dedup: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
    for p in pairs:
        dedup[(p["source_key"], round(p["twist_angle_deg"], 4), round(p["frequency_GHz"], 6))] = p
    return list(dedup.values()), rejected[:50]


def extract_qualitative_flags(text: str, source_key: str) -> Dict[str, Any]:
    t = clean_text(text)
    return {
        "source_key": source_key,
        "mentions_twist_angle_dependence": bool(re.search(r"twist angle[- ]dependent|function of twist angle|twist angle.*depend|angle.*tunable|tun(?:e|able).*twist|twist.*phonon", t, re.I)),
        "mentions_bandgap": bool(re.search(r"phononic bandgap|phonon bandgap|band gap|bandgap|miniphonon bands separated by gaps", t, re.I)),
        "mentions_692_angles": bool(re.search(r"692\s+twist", t, re.I)),
        "mentions_small_angle_regime": bool(re.search(r"small twist angles|small-angle|magic-angle|below.*degree", t, re.I)),
        "mentions_source_data": bool(re.search(r"source data|supplementary data|xlsx|csv", t, re.I)),
    }


def analyze(pairs: List[Dict[str, Any]], flags: List[Dict[str, Any]], rejected: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(pairs) >= 4:
        angles = [float(p["twist_angle_deg"]) for p in pairs]
        freqs = [float(p["frequency_GHz"]) for p in pairs]
        freq_ratio = max(freqs) / min(freqs) if min(freqs) > 0 else float("inf")
        if freq_ratio > 1e5:
            return {"status": "extraction_failed", "n_pairs": len(pairs), "reason": "Clean-row guard failed: extracted frequencies span more than 1e5, suggesting unit/context contamination.", "frequency_range_GHz": [float(min(freqs)), float(max(freqs))], "angle_range_deg": [float(min(angles)), float(max(angles))], "n_rejected_candidates": len(rejected)}
        x = [math.sin(math.radians(a) / 2.0) for a in angles]
        fit = power_law_fit(x, freqs)
        corr = spearman(angles, freqs)
        strength = max(abs(corr) if math.isfinite(corr) else 0.0, abs(float(fit.get("pearson_log", 0.0))) if fit.get("pearson_log") is not None else 0.0)
        table_rows = sum(1 for p in pairs if p.get("confidence") in {"source_data_table", "source_data_row_fallback"})
        status = "confirm_like" if strength >= 0.6 and table_rows >= 4 else "partial_numeric"
        return {"status": status, "n_pairs": len(pairs), "n_source_data_table_pairs": table_rows, "spearman_angle_vs_frequency": corr, "power_law_freq_vs_sin_theta_over_2": fit, "frequency_range_GHz": [float(min(freqs)), float(max(freqs))], "angle_range_deg": [float(min(angles)), float(max(angles))], "n_rejected_candidates": len(rejected), "interpretation": "Numeric result is used only after strict context/range filtering; full confirm requires >=4 clean source-data/table rows and strong monotonic/log trend."}
    any_qual = any(f.get("mentions_twist_angle_dependence") or f.get("mentions_692_angles") for f in flags)
    any_bandgap = any(f.get("mentions_bandgap") for f in flags)
    blocked = sum(1 for r in rejected if "403" in str(r) or "download failed" in str(r))
    return {"status": "partial_qualitative" if any_qual else "data_limited", "n_pairs": len(pairs), "qualitative_twist_dependence_seen": any_qual, "qualitative_bandgap_seen": any_bandgap, "flags": flags, "n_rejected_candidates": len(rejected), "n_source_data_blockers_or_rejections": blocked, "reason": "Open sources discuss twist-angle phonon tunability, but no trustworthy numeric angle-frequency/bandgap table was extracted."}


def download_first(src: Dict[str, Any], cache: str, force: bool) -> Tuple[pathlib.Path, str, str, List[str]]:
    errs: List[str] = []
    for url in [src["url"]] + list(src.get("alt_urls", [])):
        try:
            path, text = download_text(url, cache, force=force, timeout=25)
            return path, text, url, errs
        except Exception as exc:
            errs.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errs))


def main() -> None:
    parser = common_arg_parser(__doc__ or "MAT9 public test")
    args = parser.parse_args()

    downloaded = []
    pairs: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = []
    extraction_notes = []

    for src in SOURCES:
        try:
            path, text, used_url, prior_errors = download_first(src, args.cache, args.force)
            downloaded.append(source_record(src["label"], used_url, path, notes=("fallback used after: " + " | ".join(prior_errors)) if prior_errors else None))
            source_text = text
            # arXiv source files: text-like only, with strict filters.
            if src.get("arxiv_id"):
                try:
                    ep_path, files = download_arxiv_source(src["arxiv_id"], args.cache, force=args.force)
                    downloaded.append(source_record(src["label"] + " source e-print", f"https://arxiv.org/e-print/{src['arxiv_id']}", ep_path))
                    source_text += "\n" + "\n".join(files.values())
                    extraction_notes.append({"source_key": src["key"], "arxiv_source_files": len(files)})
                except Exception as exc:
                    extraction_notes.append({"source_key": src["key"], "arxiv_source_error": str(exc)})
            # HTML tables in the article itself.
            for tbl in html_tables_from_path(path):
                table_rows = [tbl.get("headers", [])] + tbl.get("rows", [])
                got = extract_pairs_from_table_rows(table_rows, src["key"], f"html table {tbl.get('index')} {tbl.get('caption','')}", rejected)
                pairs.extend(got)
            # Discover and parse source-data links.
            discovered_links = discover_data_links(path, used_url)
            data_links = []
            for du in discovered_links + list(src.get("candidate_source_data_urls", [])):
                if du not in data_links:
                    data_links.append(du)
            data_notes = []
            for du in data_links[:12]:
                try:
                    dpath = download(du, args.cache, force=args.force, timeout=25)
                    downloaded.append(source_record(src["label"] + " discovered/source data", du, dpath))
                    got, note = extract_pairs_from_downloaded_data(dpath, src["key"], rejected)
                    pairs.extend(got)
                    data_notes.append(note)
                except Exception as exc:
                    data_notes.append({"url": du, "error": str(exc)})
            got_pairs, got_rejected = extract_angle_frequency_pairs(source_text, src["key"])
            got_flags = extract_qualitative_flags(source_text, src["key"])
            pairs.extend(got_pairs)
            rejected.extend(got_rejected)
            flags.append(got_flags)
            extraction_notes.append({"source_key": src["key"], "used_url": used_url, "text_chars": len(source_text), "accepted_pairs_text_or_tables": len(got_pairs), "rejected_candidates_sampled": len(got_rejected), "data_links_seen": data_links[:12], "data_notes": data_notes[:12], "flags": got_flags})
        except Exception as exc:
            downloaded.append(source_record(src["label"], src["url"], notes=f"download/extract failed: {exc}"))
            extraction_notes.append({"source_key": src["key"], "error": str(exc)})

    dedup: Dict[Tuple[str, float, float, str], Dict[str, Any]] = {}
    for p in pairs:
        dedup[(p["source_key"], round(p["twist_angle_deg"], 5), round(p["frequency_GHz"], 5), p.get("quantity", ""))] = p
    pairs = list(dedup.values())

    result = {
        "prediction_id": "MAT9",
        "test_name": "Moiré twist-angle phonon/bandgap tunability from public literature",
        "generated_utc": utc_now(),
        "analysis": analyze(pairs, flags, rejected),
        "extracted_angle_frequency_pairs": pairs,
        "rejected_candidate_examples": rejected[:30],
        "downloaded_sources": downloaded,
        "extraction_notes": extraction_notes,
        "falsification_logic": {
            "confirm_like": "At least 4 clean public source-data/table angle-bandgap/frequency rows show strong twist-angle tunability after context/range filtering.",
            "partial_numeric": "Clean numeric rows exist but are too few, too weakly correlated, or mostly prose-extracted rather than source-data table rows.",
            "falsify_like": "Adequate clean public numeric rows show no dependence of phononic bandgap/frequency on twist angle.",
            "partial_qualitative": "Open papers support twist-angle phonon tunability qualitatively, but source data are in plots or unparseable tables.",
            "extraction_failed": "Candidate numeric rows were rejected as source-code/affiliation/figure markup or nonsensical units/ranges.",
        },
    }
    write_json(result, args.outdir, "test05_mat9_moire_twist_phonon_bandgap.json")
    print(json_dumps(result))
    if args.strict and result["analysis"]["status"] in {"partial", "partial_qualitative", "partial_numeric", "data_limited", "extraction_failed"}:
        raise SystemExit("MAT9 extraction is not decisive")


if __name__ == "__main__":
    main()
