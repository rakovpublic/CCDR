#!/usr/bin/env python3
"""
MAT4 public-literature proxy: thermoelectric gain near engineered 30 degree grain boundaries.

Prediction target:
    ZT_optimized / ZT_random ~= 1.2 +/- 0.1 for Bi2Te3-class material at theta~30 deg GB.

v6 improvements:
  - Uses the PMC mirror of the Tan/Sci. Reports article in addition to Nature.
  - Parses HTML tables and legacy .doc supplementary strings when available.
  - Adds robust voltage-series parsing: if the paper exposes paired 0/10/20/30/40 V
    ZT values, the baseline and optimized ZT become independent rows.
  - Still refuses decisive confirm/falsify unless optimized and baseline ZT are parsed
    independently from the same angle/voltage-controlled study.
  - Can derive a non-decisive baseline estimate from explicit percent-gain language,
    but derived baselines never count as independent confirm/falsify evidence.
"""
from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

from ccdr_lit_public_utils import (
    binary_strings,
    common_arg_parser,
    download,
    download_text,
    floats_from_text,
    html_tables_from_path,
    json_dumps,
    path_to_text,
    source_record,
    utc_now,
    write_json,
)

SOURCES = [
    {
        "key": "tan_sbrep_2020_sb2te3_nature",
        "study_key": "tan_sbrep_2020_sb2te3",
        "label": "Tan et al. 2020 Scientific Reports Sb2Te3 angular intraplanar GB open HTML",
        "url": "https://www.nature.com/articles/s41598-020-63062-z",
        "supp_urls": ["https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-020-63062-z/MediaObjects/41598_2020_63062_MOESM1_ESM.doc"],
    },
    {
        "key": "tan_sbrep_2020_sb2te3_pmc",
        "study_key": "tan_sbrep_2020_sb2te3",
        "label": "Tan et al. 2020 Scientific Reports Sb2Te3 angular intraplanar GB PMC mirror",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7136274/",
    },
    {
        "key": "poudel_2008_nanostructured_bi_sb_te",
        "study_key": "poudel_2008_nanostructured_bi_sb_te",
        "label": "Poudel et al. 2008 nanostructured BiSbTe PDF mirror, baseline context only",
        "url": "https://scispace.com/pdf/high-thermoelectric-performance-of-nanostructured-bismuth-4evuadx0tz.pdf",
        "same_study_denominator": False,
    },
]

VOLTAGES = [0.0, 10.0, 20.0, 30.0, 40.0]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("−", "-").replace("–", "-").replace("—", "-").replace("μ", "u").replace("µ", "u")).strip()


def split_chunks(text: str) -> List[str]:
    t = clean_text(text)
    parts = re.split(r"(?<=[.!?;])\s+", t)
    # add sliding windows around ZT mentions because figure captions/text can be long
    starts = [m.start() for m in re.finditer(r"\bZ\s*T\b|\bZT\b|figure of merit", t, re.I)]
    windows = [t[max(0, s - 600): s + 900] for s in starts[:100]]
    return [p for p in parts + windows if len(p) > 10]


def voltage_present(s: str, voltage: float) -> bool:
    return bool(re.search(rf"\b{int(voltage)}\s*V\b", s, re.I))


def add_or_update(rows: List[Dict[str, Any]], voltage: float, **kwargs: Any) -> Dict[str, Any]:
    row = next((r for r in rows if abs(float(r.get("voltage_V", -999)) - voltage) < 1e-9), None)
    if row is None:
        row = {"voltage_V": float(voltage)}
        rows.append(row)
    for k, v in kwargs.items():
        if v is not None:
            row[k] = v
    return row


def parse_zt_value_near_label(text: str) -> Optional[float]:
    pats = [
        r"(?:Z\s*T|ZT|figure\s*of\s*merit)\s*(?:value)?[^0-9]{0,80}(?:of|=|as high as|reaches?|reached|is|was)?\s*([0-9]+(?:\.[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:for|as)?\s*(?:Z\s*T|ZT)",
    ]
    for pat in pats:
        m = re.search(pat, text, re.I)
        if m:
            val = float(m.group(1))
            if 0.05 <= val <= 5.0:
                return val
    return None



def is_clean_scientific_context(text: str, *, require_zt: bool = False) -> bool:
    """Reject legacy-DOC/binary-string garbage before a numeric value can become evidence."""
    if not text:
        return False
    c = clean_text(text)
    if "\ufffd" in c or "�" in c or any(ord(ch) < 32 and ch not in "\n\t\r" for ch in c):
        return False
    punct = sum(1 for ch in c if ch in "{}[]@`^~%$#\\|<>_")
    letters = sum(1 for ch in c if ch.isalpha())
    if len(c) < 20 or letters < 8:
        return False
    if punct / max(len(c), 1) > 0.08:
        return False
    if require_zt and not re.search(r"(?:\bZ\s*T\b|\bZT\b|figure\s*of\s*merit|thermoelectric)", c, re.I):
        return False
    if not re.search(r"\b(sample|film|deposited|deposition|voltage|electric\s*field|temperature|room\s*temperature|thermoelectric|conductivity|seebeck|power\s*factor|grain|boundary|baseline|random|pristine)\b", c, re.I):
        return False
    return True



def derive_baseline_from_percent_gain(chunks: List[str], zt30: Optional[float]) -> List[Dict[str, Any]]:
    """Return non-decisive baseline estimates from explicit same-paragraph gain language.

    Example accepted form: "ZT ... under 30 V ... 1.75, which is 20% higher than
    the 0 V/baseline sample". These rows are useful audit information but are never
    treated as independently parsed denominator evidence.
    """
    if zt30 is None or zt30 <= 0:
        return []
    out: List[Dict[str, Any]] = []
    baseline_words = r"(?:0\s*V|zero\s*voltage|without\s+(?:external\s+|assisted\s+)?voltage|baseline|random|pristine|unoptimized)"
    gain_words = r"(?:higher|increase[ds]?|increased|improve[ds]?|improved|enhance[ds]?|enhanced|larger|greater)"
    for chunk in chunks:
        c = clean_text(chunk)
        if not (voltage_present(c, 30) and re.search(r"(?:\bZ\s*T\b|\bZT\b|figure\s*of\s*merit)", c, re.I)):
            continue
        if not is_clean_scientific_context(c, require_zt=True):
            continue
        patterns = [
            rf"{gain_words}[^.;]{{0,100}}?by\s*([0-9]+(?:\.[0-9]+)?)\s*%[^.;]{{0,140}}?{baseline_words}",
            rf"([0-9]+(?:\.[0-9]+)?)\s*%[^.;]{{0,80}}?{gain_words}[^.;]{{0,140}}?{baseline_words}",
            rf"{gain_words}[^.;]{{0,140}}?{baseline_words}[^.;]{{0,100}}?by\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        ]
        for pat in patterns:
            m = re.search(pat, c, re.I)
            if not m:
                continue
            pct = float(m.group(1))
            if 1.0 <= pct <= 500.0:
                baseline = zt30 / (1.0 + pct / 100.0)
                if 0.05 <= baseline <= 5.0:
                    out.append({
                        "voltage_V": 0.0,
                        "ZT_300K": float(baseline),
                        "zt_context": c[:520],
                        "zt_parse_mode": "derived_baseline_from_explicit_percent_gain",
                        "derived_from_percent_gain": pct,
                        "non_decisive": True,
                    })
                    break
    return out[:3]


def clean_zt_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for r in rows:
        if "ZT_300K" not in r:
            clean.append(r)
            continue
        ctx = str(r.get("zt_context") or r.get("zt_context_30V") or r.get("zt_context_0V") or "")
        mode = r.get("zt_parse_mode")
        if not is_clean_scientific_context(ctx, require_zt=True):
            rr = dict(r)
            rr["reject_reason"] = "garbled_or_non_scientific_zt_context"
            rejected.append(rr)
            continue
        if abs(float(r.get("voltage_V", -999)) - 0.0) < 1e-9 and mode == "explicit_voltage_zt_pair":
            if not re.search(r"\b(0\s*V|zero\s*voltage|without\s+(?:assisted\s+)?voltage|baseline|random|pristine|unoptimized)\b", ctx, re.I):
                rr = dict(r); rr["reject_reason"] = "weak_0v_baseline_context"; rejected.append(rr); continue
            if not re.search(r"(?:\bZ\s*T\b|\bZT\b|figure\s*of\s*merit)", ctx, re.I):
                rr = dict(r); rr["reject_reason"] = "no_explicit_zt_token_for_0v_baseline"; rejected.append(rr); continue
        clean.append(r)
    return clean, rejected

def parse_voltage_series_values(text: str, quantity_label: str = "ZT") -> List[Dict[str, Any]]:
    """Extract explicit voltage series like 0/10/20/30/40 V paired with ZT values."""
    t = clean_text(text)
    rows: List[Dict[str, Any]] = []
    volt_pat = r"0\s*V\s*,\s*10\s*V\s*,\s*20\s*V\s*,\s*30\s*V\s*,\s*(?:and\s*)?40\s*V"
    # Very permissive voltage-series catch for decimal ZT lists.
    val_list = r"((?:[0-9]+(?:\.[0-9]+)?(?:\s*,?\s*(?:and\s*)?)?){5,})"
    for m in re.finditer(r"(?:Z\s*T|ZT|figure\s*of\s*merit)[^.;]{0,260}?" + volt_pat + r"[^.;]{0,220}?(?:values?|are|were|of|:)\s*" + val_list, t, re.I):
        vals = [v for v in floats_from_text(m.group(1)) if 0.05 <= v <= 5.0]
        if len(vals) >= 5:
            for v, zt in zip(VOLTAGES, vals[:5]):
                rows.append({"voltage_V": v, "ZT_300K": float(zt), "zt_context": m.group(0)[:520], "zt_parse_mode": "voltage_series_values_after_labels"})

    # series where voltages are named before values
    for m in re.finditer(volt_pat + r"[^.]{0,500}?(?:Z\s*T|ZT|figure\s*of\s*merit)[^.]{0,180}?(?:values?|are|were|of|:)\s*([^.;]{0,260})", t, re.I):
        vals = [v for v in floats_from_text(m.group(1)) if 0.05 <= v <= 5.0]
        if len(vals) >= 5:
            for v, zt in zip(VOLTAGES, vals[:5]):
                rows.append({"voltage_V": v, "ZT_300K": float(zt), "zt_context": m.group(0)[:520], "zt_parse_mode": "voltage_series_values_after_labels"})
    # values before explicit 0/10/20/30/40 list, uncommon but possible
    for m in re.finditer(r"(?:Z\s*T|ZT|figure\s*of\s*merit)[^.]{0,180}?(?:values?|are|were|of|:)\s*([^.;]{0,260}?)\s*(?:for|under|at|of)[^.]{0,80}?" + volt_pat, t, re.I):
        vals = [v for v in floats_from_text(m.group(1)) if 0.05 <= v <= 5.0]
        if len(vals) >= 5:
            for v, zt in zip(VOLTAGES, vals[:5]):
                rows.append({"voltage_V": v, "ZT_300K": float(zt), "zt_context": m.group(0)[:520], "zt_parse_mode": "voltage_series_values_before_labels"})

    # ZT values for samples deposited under 0/10/20/30/40 V are ...
    for m in re.finditer(r"(?:Z\s*T|ZT|figure\s*of\s*merit)[^.]{0,220}?(?:for|under|at|of)[^.]{0,160}?" + volt_pat + r"[^.]{0,160}?(?:values?|are|were|of|:)\s*([^.;]{0,260})", t, re.I):
        vals = [v for v in floats_from_text(m.group(1)) if 0.05 <= v <= 5.0]
        if len(vals) >= 5:
            for v, zt in zip(VOLTAGES, vals[:5]):
                rows.append({"voltage_V": v, "ZT_300K": float(zt), "zt_context": m.group(0)[:520], "zt_parse_mode": "voltage_series_values_after_labels"})

    # pair style: 0 V: ZT=..., 10 V: ZT=...
    for m in re.finditer(r"\b(0|10|20|30|40)\s*V\b[^.;]{0,120}?(?:Z\s*T|ZT)[^0-9]{0,80}([0-9]+(?:\.[0-9]+)?)", t, re.I):
        val = float(m.group(2))
        if 0.05 <= val <= 5.0:
            rows.append({"voltage_V": float(m.group(1)), "ZT_300K": val, "zt_context": m.group(0)[:300], "zt_parse_mode": "explicit_voltage_zt_pair"})
    return rows


def parse_voltage_series_sigma(text: str) -> List[Dict[str, Any]]:
    t = clean_text(text)
    rows: List[Dict[str, Any]] = []
    m = re.search(r"0\s*V,\s*10\s*V,\s*20\s*V,\s*30\s*V,\s*and\s*40\s*V[^.]{0,260}?sigma[^.]{0,120}?values?\s*of\s*([^.;]{0,260})", t, re.I)
    if m:
        vals = []
        for sm in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*10\s*\^?\s*\{?([0-9]+)\}?", m.group(1), re.I):
            vals.append(float(sm.group(1)) * (10 ** int(sm.group(2))))
        if len(vals) >= 5:
            for v, sig in zip(VOLTAGES, vals[:5]):
                rows.append({"voltage_V": v, "sigma_S_per_m_max": sig, "sigma_context": m.group(0)[:420]})
    return rows


def parse_html_tables_for_zt(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tbl in html_tables_from_path(path):
        caption = clean_text(tbl.get("caption", ""))
        headers = [clean_text(h) for h in tbl.get("headers", [])]
        body = tbl.get("rows", [])
        joined = " ".join([caption] + headers + [" ".join(r) for r in body[:10]])
        if not (re.search(r"\bZ\s*T\b|\bZT\b|figure\s*of\s*merit", joined, re.I) and re.search(r"\bV\b|voltage|electric\s*field", joined, re.I)):
            continue
        volt_cols = [i for i, h in enumerate(headers) if re.search(r"voltage|electric\s*field|\bV\b", h, re.I)]
        zt_cols = [i for i, h in enumerate(headers) if re.search(r"\bZ\s*T\b|\bZT\b|figure\s*of\s*merit", h, re.I)]
        for row in body:
            row_ctx = f"table {tbl.get('index')} {caption} headers={headers} row={row}"
            if volt_cols and zt_cols:
                for vi in volt_cols:
                    if vi >= len(row):
                        continue
                    vs = [v for v in floats_from_text(row[vi]) if v in VOLTAGES or 0 <= v <= 40]
                    if not vs:
                        continue
                    for zi in zt_cols:
                        if zi >= len(row):
                            continue
                        zts = [z for z in floats_from_text(row[zi]) if 0.05 <= z <= 5.0]
                        if zts:
                            rows.append({"voltage_V": float(vs[0]), "ZT_300K": float(zts[0]), "zt_context": row_ctx[:520], "zt_parse_mode": "html_table_voltage_zt"})
            # Row fallback.
            rows.extend(parse_voltage_series_values(" ".join(row)))
    return rows


def parse_tan_like_source(text: str, path: Optional[pathlib.Path], source_key: str) -> Dict[str, Any]:
    t = clean_text(text)
    chunks = split_chunks(t)
    rows: List[Dict[str, Any]] = []

    # GB-angle mapping from prose.
    for voltage, angle in [(20.0, 20.0), (30.0, 30.0), (40.0, 40.0)]:
        if re.search(rf"{int(voltage)}\s*V[^.;]{{0,240}}?angle[^.;]{{0,80}}?~?\s*{int(angle)}\s*(?:°|degree|deg)|angle[^.;]{{0,80}}?~?\s*{int(angle)}\s*(?:°|degree|deg)[^.;]{{0,240}}?{int(voltage)}\s*V", t, re.I):
            add_or_update(rows, voltage, gb_angle_deg=angle, angle_context=f"{int(voltage)} V mapped to ~{int(angle)} degree GB angle from article prose")

    # HTML tables first; explicit text series second; single optimized sentence last.
    if path is not None:
        for r in parse_html_tables_for_zt(path):
            add_or_update(rows, float(r["voltage_V"]), ZT_300K=r.get("ZT_300K"), zt_context=r.get("zt_context"), zt_parse_mode=r.get("zt_parse_mode"))
    for r in parse_voltage_series_values(t):
        add_or_update(rows, float(r["voltage_V"]), ZT_300K=r.get("ZT_300K"), zt_context=r.get("zt_context"), zt_parse_mode=r.get("zt_parse_mode"))
    for r in parse_voltage_series_sigma(t):
        add_or_update(rows, float(r["voltage_V"]), sigma_S_per_m_max=r.get("sigma_S_per_m_max"), sigma_context=r.get("sigma_context"))

    # Optimized 30 V ZT sentence. This cannot create a baseline.
    for chunk in chunks:
        if not voltage_present(chunk, 30):
            continue
        if not re.search(r"\bZ\s*T\b|\bZT\b|figure\s*of\s*merit", chunk, re.I):
            continue
        zt = parse_zt_value_near_label(chunk)
        if zt is not None:
            add_or_update(rows, 30.0, gb_angle_deg=30.0, ZT_300K=zt, zt_context_30V=chunk[:520], zt_parse_mode="optimized_30V_sentence")
            break

    # Independent baseline single sentence. Reject chunks mentioning 30 V/~30°.
    baseline_candidates: List[Dict[str, Any]] = []
    baseline_patterns = [
        r"\b0\s*V\b[^.;]{0,220}?\b(?:Z\s*T|ZT|figure\s*of\s*merit)\b[^.;]{0,80}?(?:=|of|is|was|reaches?|reached)\s*([0-9]+(?:\.[0-9]+)?)",
        r"\b(?:random|disordered|baseline|pristine|unoptimized|without assisted voltage|without external voltage|no assisted voltage)\b[^.;]{0,260}?\b(?:Z\s*T|ZT|figure\s*of\s*merit)\b[^.;]{0,80}?(?:=|of|is|was|reaches?|reached)\s*([0-9]+(?:\.[0-9]+)?)",
        r"\b(?:Z\s*T|ZT|figure\s*of\s*merit)\b[^.;]{0,80}?(?:=|of|is|was|reaches?|reached)\s*([0-9]+(?:\.[0-9]+)?)[^.;]{0,220}?\b(?:0\s*V|random|disordered|baseline|pristine|unoptimized|no assisted voltage)\b",
    ]
    for chunk in chunks:
        if voltage_present(chunk, 30) or re.search(r"~\s*30\s*(?:°|degree|deg)|30\s*(?:°|degree|deg)", chunk, re.I):
            continue
        for pat in baseline_patterns:
            m = re.search(pat, chunk, re.I)
            if m:
                val = float(m.group(1))
                if 0.05 <= val <= 5.0:
                    baseline_candidates.append({"ZT_300K": val, "context": chunk[:520], "parse_mode": "independent_baseline_sentence"})
                break
    # Only use baseline sentence if no better voltage-series/table baseline exists.
    if baseline_candidates and not any(abs(float(r.get("voltage_V", -999)) - 0.0) < 1e-9 and r.get("ZT_300K") for r in rows):
        cand = baseline_candidates[0]
        add_or_update(rows, 0.0, gb_angle_deg=None, ZT_300K=cand["ZT_300K"], zt_context=cand["context"], zt_parse_mode=cand["parse_mode"])

    # Non-decisive baseline estimates from explicit percentage-gain statements.
    row30_for_est = next((r for r in rows if abs(float(r.get("voltage_V", -999)) - 30.0) < 1e-9 and r.get("ZT_300K")), None)
    if row30_for_est is not None and not any(abs(float(r.get("voltage_V", -999)) - 0.0) < 1e-9 and r.get("ZT_300K") for r in rows):
        rows.extend(derive_baseline_from_percent_gain(chunks, float(row30_for_est["ZT_300K"])))

    # Normalize provenance fields and reject garbled DOC/binary false ZT rows.
    rows, rejected_zt_rows = clean_zt_rows(rows)
    for r in rows:
        r.setdefault("source_key", source_key)
        if "zt_context" in r and abs(float(r.get("voltage_V", -999)) - 30.0) < 1e-9:
            r.setdefault("zt_context_30V", r["zt_context"])
        if "zt_context" in r and abs(float(r.get("voltage_V", -999)) - 0.0) < 1e-9:
            r.setdefault("zt_context_0V", r["zt_context"])

    return {
        "rows": sorted(rows, key=lambda r: r.get("voltage_V", -1)),
        "baseline_candidates_seen": baseline_candidates[:5],
        "rejected_zt_rows": rejected_zt_rows[:8],
        "notes": [
            "Sb2Te3 is a Bi2Te3-family proxy, not exact Bi2Te3.",
            "Exact MAT4 status requires independent same-study baseline/random ZT and 30-degree optimized ZT.",
            "Voltage-series/table parsing is accepted as independent; a lone 30 V optimized sentence is only partial_supportive.",
        ],
    }


def parse_poudel_baseline(text: str) -> Dict[str, Any]:
    t = clean_text(text)
    rows = []
    for pat, label in [
        (r"conventional\s+Bi2Te3[^.]{0,220}?peak\s*Z\s*T\s*of\s*about\s*([\d.]+)", "conventional Bi2Te3-based context"),
        (r"peak\s*Z\s*T\s*of\s*([\d.]+)\s*at\s*100", "nanostructured BiSbTe context"),
    ]:
        m = re.search(pat, t, re.I)
        if m:
            rows.append({"material": label, "ZT_peak": float(m.group(1)), "angle_controlled": False})
    return {"context_rows": rows, "note": "Not used as denominator for same-study MAT4 ratio."}


def study_rows(extractions: Dict[str, Any], study_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, ex in extractions.items():
        if key.startswith(study_key) or key == study_key:
            rows.extend(ex.get("rows", []))
    # dedupe by voltage, prefer rows with ZT from table/series over lone optimized sentence
    priority = {"html_table_voltage_zt": 3, "voltage_series_values_after_labels": 3, "voltage_series_values_before_labels": 3, "explicit_voltage_zt_pair": 3, "independent_baseline_sentence": 2, "optimized_30V_sentence": 1, None: 0}
    best: Dict[float, Dict[str, Any]] = {}
    for r in rows:
        v = float(r.get("voltage_V", -999))
        old = best.get(v)
        if old is None or priority.get(r.get("zt_parse_mode"), 0) >= priority.get(old.get("zt_parse_mode"), 0):
            merged = dict(old or {})
            merged.update(r)
            best[v] = merged
    return sorted(best.values(), key=lambda r: r.get("voltage_V", -1))


def analyze(extractions: Dict[str, Any]) -> Dict[str, Any]:
    rows = study_rows(extractions, "tan_sbrep_2020_sb2te3")
    row30 = next((r for r in rows if abs(float(r.get("voltage_V", -999)) - 30.0) < 1e-9), None)
    row0 = next((r for r in rows if abs(float(r.get("voltage_V", -999)) - 0.0) < 1e-9), None)
    zt30 = row30.get("ZT_300K") if row30 else None
    zt0 = row0.get("ZT_300K") if row0 else None
    zt30_context = (row30 or {}).get("zt_context_30V") or (row30 or {}).get("zt_context")
    zt0_context = (row0 or {}).get("zt_context_0V") or (row0 or {}).get("zt_context")
    zt0_mode = (row0 or {}).get("zt_parse_mode")
    sigma30 = (row30 or {}).get("sigma_S_per_m_max")
    sigma0 = (row0 or {}).get("sigma_S_per_m_max")

    ratio = None
    status = "data_limited"
    metric = "exact ZT ratio not extractable"
    warnings: List[str] = []
    trusted_baseline_modes = {"html_table_voltage_zt", "voltage_series_values_after_labels", "voltage_series_values_before_labels", "independent_baseline_sentence"}
    independent_baseline = bool(
        zt0 is not None
        and zt0_context
        and zt0_mode in trusted_baseline_modes
        and is_clean_scientific_context(str(zt0_context), require_zt=True)
        and not re.search(r"\b30\s*V\b|~\s*30\s*(?:°|degree|deg)|30\s*(?:°|degree|deg)", str(zt0_context), re.I)
    )
    derived_baseline = bool(zt0 is not None and zt0_mode == "derived_baseline_from_explicit_percent_gain")

    if zt30 is not None and zt0 is not None and zt0 > 0 and independent_baseline:
        ratio = float(zt30 / zt0)
        status = "confirm_like" if 1.1 <= ratio <= 1.3 else "falsify_like"
        metric = "direct independent same-study ZT_30deg_or_30V / ZT_0V_or_random"
    elif zt30 is not None and derived_baseline and zt0 and zt0 > 0:
        ratio = float(zt30 / zt0)
        status = "partial_estimated_ratio"
        metric = "optimized 30-degree/30V ZT parsed; denominator estimated from explicit percent-gain language, not independent baseline"
    elif zt30 is not None:
        status = "partial_supportive"
        metric = "optimized 30-degree/30V ZT parsed; independent baseline/random ZT not parsed"
    elif sigma30 and sigma0:
        status = "partial_directional"
        metric = "conductivity ratio parsed, but ZT ratio missing"
    sigma_ratio = float(sigma30 / sigma0) if sigma30 and sigma0 else None
    target_baseline_window = [float(zt30) / 1.3, float(zt30) / 1.1] if zt30 else None
    missing_decisive_inputs = [] if independent_baseline else [
        "independent 0V/random/pristine ZT from same study",
        "clean voltage/angle-vs-ZT table or source-data row",
    ]

    return {
        "status": status,
        "zt_30deg_or_30V": zt30,
        "zt_random_or_0V": zt0 if independent_baseline else None,
        "zt_baseline_estimated_from_percent_gain": zt0 if derived_baseline and not independent_baseline else None,
        "zt_ratio": ratio,
        "baseline_independently_parsed": independent_baseline,
        "baseline_estimated_not_independent": derived_baseline and not independent_baseline,
        "sigma_30V_over_0V": sigma_ratio,
        "metric": metric,
        "target_ratio_window": [1.1, 1.3],
        "target_baseline_ZT_window_if_30V_is_correct": target_baseline_window,
        "missing_decisive_inputs": missing_decisive_inputs,
        "warnings": warnings,
        "rows_used_from_tan_study": rows,
        "interpretation": "Counts as confirm/falsify only if both optimized and baseline ZT are independently parsed from a same-study angle/voltage-controlled source.",
    }


def supplement_text(path: pathlib.Path) -> str:
    try:
        txt = path_to_text(path)
    except Exception:
        txt = ""
    try:
        bs = binary_strings(path)
    except Exception:
        bs = ""
    return clean_text(txt + "\n" + bs)


def main() -> None:
    parser = common_arg_parser(__doc__ or "MAT4 public test")
    args = parser.parse_args()

    downloaded = []
    extractions: Dict[str, Any] = {}
    for src in SOURCES:
        try:
            path, text = download_text(src["url"], args.cache, force=args.force, timeout=25)
            downloaded.append(source_record(src["label"], src["url"], path))
            if src.get("study_key") == "tan_sbrep_2020_sb2te3":
                combined_text = text
                for supp_url in src.get("supp_urls", []):
                    try:
                        supp = download(supp_url, args.cache, force=args.force)
                        downloaded.append(source_record(src["label"] + " supplement", supp_url, supp))
                        combined_text += "\n" + supplement_text(supp)
                    except Exception as exc:
                        downloaded.append(source_record(src["label"] + " supplement", supp_url, notes=f"supplement download failed: {exc}"))
                extractions[src["key"]] = parse_tan_like_source(combined_text, path, src["key"])
            else:
                extractions[src["key"]] = parse_poudel_baseline(text)
        except Exception as exc:
            downloaded.append(source_record(src["label"], src["url"], notes=f"download/extract failed: {exc}"))

    result = {
        "prediction_id": "MAT4",
        "test_name": "30-degree grain-boundary ZT gain in Bi2Te3-family thermoelectrics from public literature",
        "generated_utc": utc_now(),
        "analysis": analyze(extractions),
        "extractions": extractions,
        "downloaded_sources": downloaded,
        "falsification_logic": {
            "confirm_like": "A same-study Bi2Te3/Bi2Te3-family source independently exposes ZT_30deg and random/baseline ZT with ratio 1.1-1.3.",
            "falsify_like": "Adequate same-study independent angle-controlled ZT data expose a ratio outside the 1.1-1.3 window.",
            "partial_supportive": "Open source confirms enhanced 30-degree/30V engineered state but baseline ZT or exact Bi2Te3 condition is not cleanly extractable.",
            "partial_estimated_ratio": "ZT ratio is estimated from explicit percentage-gain prose; useful but non-decisive because denominator is not independently parsed.",
            "partial_directional": "Transport directionality is parsed, but exact ZT ratio is unavailable.",
            "data_limited": "Neither optimized nor baseline ZT could be extracted from public text.",
        },
    }
    write_json(result, args.outdir, "test03_mat4_bi2te3_gb_zt_gain.json")
    print(json_dumps(result))
    if args.strict and result["analysis"]["status"] in {"partial_supportive", "partial_estimated_ratio", "partial_directional", "data_limited"}:
        raise SystemExit("MAT4 exact ZT-ratio extraction is not decisive")


if __name__ == "__main__":
    main()
