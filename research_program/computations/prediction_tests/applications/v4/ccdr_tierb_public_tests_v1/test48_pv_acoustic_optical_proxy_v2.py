#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

from _tierb_v5_quality_helpers import download_bytes, ensure_dir, find_col, linear_fit_metrics, now_utc_iso, numeric_series

NREL_PAGE_CANDIDATES = [
    "https://www.nrel.gov/pv/cell-efficiency.html",
    "https://www.nrel.gov/pv/interactive-cell-efficiency.html",
    "https://www.nrel.gov/pv/cell-efficiency-data.html",
]

DIRECT_CANDIDATES = [
    "https://www.nrel.gov/media/docs/libraries/pv/cell-efficiency-data-table.xlsx",
    "https://www.nrel.gov/media/docs/libraries/pv/cell-efficiency-data-table.csv",
]

YEAR = [r"year", r"date"]
EFF = [r"efficiency", r"eff", r"percent", r"%"]
MAT = [r"material", r"technology", r"cell.*type", r"classification", r"family"]
AREA = [r"area", r"cm2", r"aperture"]
CELL = [r"cell", r"device", r"submodule", r"module"]

MATERIAL_PROXY = {
    "si": {"mass_contrast": 1.0, "symmetry": 0.7},
    "silicon": {"mass_contrast": 1.0, "symmetry": 0.7},
    "gaas": {"mass_contrast": 1.5, "symmetry": 0.8},
    "iii-v": {"mass_contrast": 1.7, "symmetry": 0.8},
    "perovskite": {"mass_contrast": 2.5, "symmetry": 0.6},
    "cigs": {"mass_contrast": 3.0, "symmetry": 0.55},
    "cdte": {"mass_contrast": 3.2, "symmetry": 0.55},
    "organic": {"mass_contrast": 1.2, "symmetry": 0.35},
    "dye": {"mass_contrast": 1.2, "symmetry": 0.35},
    "multi-junction": {"mass_contrast": 2.0, "symmetry": 0.75},
    "multijunction": {"mass_contrast": 2.0, "symmetry": 0.75},
}


def html_data_links(html: str, base_url: str):
    links = set()
    # script/link anchors
    for m in re.finditer(r"(?:src|href)=[\"']([^\"']+)[\"']", html, flags=re.I):
        u = urljoin(base_url, m.group(1))
        if re.search(r"cell|efficien|chart|research|data|pv", u, re.I):
            links.add(u)
    # explicit embedded asset URLs
    for m in re.finditer(r"https?://[^\"'\s<>]+", html):
        u = m.group(0)
        if re.search(r"cell|efficien|chart|research|data|pv", u, re.I):
            links.add(u)
    return sorted(links)


def parse_tables_from_html(html: str):
    if pd is None:
        return []
    try:
        return [(f"html_table_{i}", df) for i, df in enumerate(pd.read_html(io.StringIO(html)))]
    except Exception:
        return []


def parse_embedded_json_tables(text: str):
    if pd is None:
        return []
    out = []
    # Conservative extraction: find JSON-looking arrays containing efficiency/year fields.
    for m in re.finditer(r"\[[\s\S]{100,200000}?\]", text):
        s = m.group(0)
        if not re.search(r"year|efficien|cell|technology|material", s, re.I):
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                df = pd.DataFrame(obj)
                if len(df) >= 20 and len(df.columns) >= 3:
                    out.append(("embedded_json_array", df))
        except Exception:
            continue
    return out


def parse_asset(blob: bytes, name: str):
    if pd is None:
        return []
    low = name.lower()
    out = []
    try:
        if low.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(io.BytesIO(blob))
            for sh in xls.sheet_names:
                df = xls.parse(sh)
                out.append((f"{name}:{sh}", df))
        elif low.endswith(".csv"):
            out.append((name, pd.read_csv(io.BytesIO(blob))))
        elif low.endswith((".json", ".js")):
            text = blob.decode("utf-8", errors="ignore")
            out += parse_embedded_json_tables(text)
        elif "html" in low:
            text = blob.decode("utf-8", errors="ignore")
            out += parse_tables_from_html(text)
            out += parse_embedded_json_tables(text)
    except Exception:
        pass
    return out


def table_quality(df):
    if df is None or len(df) < 20:
        return None
    cols = {
        "year": find_col(df, YEAR),
        "efficiency": find_col(df, EFF),
        "material": find_col(df, MAT),
        "area": find_col(df, AREA),
        "cell_type": find_col(df, CELL),
    }
    score = sum(1 for v in cols.values() if v)
    return {"n_rows": int(len(df)), "n_cols": int(len(df.columns)), "score": score, "matched_columns": cols, "columns_preview": [str(c) for c in list(df.columns)[:30]]}


def proxy_from_material(s: str):
    t = str(s).lower()
    for k, v in MATERIAL_PROXY.items():
        if k in t:
            return v
    return {"mass_contrast": 1.0, "symmetry": 0.4}


def analyze_pv(df, source_name):
    q = table_quality(df)
    if not q:
        return {"ok": False, "reason": "bad_table"}
    cols = q["matched_columns"]
    if not (cols["year"] and cols["efficiency"] and cols["material"]):
        return {"ok": False, "reason": "missing_required_columns", "quality": q}
    d = df.copy()
    y_eff = pd.to_numeric(d[cols["efficiency"]], errors="coerce")
    year = pd.to_numeric(d[cols["year"]], errors="coerce")
    mat = d[cols["material"]].astype(str)
    proxies = mat.apply(proxy_from_material)
    mass = np.array([p["mass_contrast"] for p in proxies], dtype=float)
    sym = np.array([p["symmetry"] for p in proxies], dtype=float)
    area = pd.to_numeric(d[cols["area"]], errors="coerce") if cols["area"] else pd.Series(np.ones(len(d)), index=d.index)
    mask = np.isfinite(y_eff) & np.isfinite(year)
    n = int(mask.sum())
    if n < 100:
        return {"ok": False, "reason": "too_few_candidate_rows", "candidate_rows_count": n, "quality": q}
    yy = y_eff[mask].astype(float)
    base = {"intercept": np.ones(n), "year_centered": (year[mask].astype(float) - np.nanmedian(year[mask].astype(float))).to_numpy()}
    test = dict(base)
    test["log_area"] = np.log(np.clip(area[mask].astype(float).to_numpy(), 1e-9, None))
    test["mass_contrast_proxy"] = mass[mask.to_numpy()]
    test["symmetry_proxy"] = sym[mask.to_numpy()]
    test["acoustic_optical_proxy"] = mass[mask.to_numpy()] * sym[mask.to_numpy()]
    fit_base = linear_fit_metrics(yy, base)
    fit_test = linear_fit_metrics(yy, test)
    out = {"ok": True, "source_table": source_name, "candidate_rows_count": n, "quality": q, "base_fit": fit_base, "acoustic_optical_proxy_fit": fit_test, "used_columns": cols}
    if fit_base.get("ok") and fit_test.get("ok"):
        out["delta_aic_proxy_minus_base"] = fit_test["aic"] - fit_base["aic"]
        out["rms_reduction_fraction"] = (fit_base["rms"] - fit_test["rms"]) / max(fit_base["rms"], 1e-30)
        beta = fit_test.get("beta", {})
        out["support_like"] = bool(out["delta_aic_proxy_minus_base"] < 0 and beta.get("acoustic_optical_proxy", 0.0) > 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_t48_pv_proxy_v2")
    ap.add_argument("--url", action="append", default=[])
    args = ap.parse_args()
    outdir = ensure_dir(args.outdir)
    sources = list(dict.fromkeys(args.url + DIRECT_CANDIDATES + NREL_PAGE_CANDIDATES))
    downloaded = []
    asset_links = []
    tables = []
    for url in sources:
        blob, meta = download_bytes(url)
        downloaded.append({"url": url, "meta": meta})
        if not blob:
            continue
        ctype = (meta.get("content_type") or "").lower()
        name = url.split("/")[-1] or "page.html"
        if "html" in ctype or name.endswith(".html") or b"<html" in blob[:1000].lower():
            html = blob.decode("utf-8", errors="ignore")
            asset_links += html_data_links(html, url)
            tables += parse_tables_from_html(html)
            tables += parse_embedded_json_tables(html)
        tables += parse_asset(blob, name)
    # Follow discovered JS/data links, prioritizing likely data assets.
    ranked = sorted(set(asset_links), key=lambda u: (not re.search(r"data|csv|xlsx|json", u, re.I), len(u)))[:80]
    for url in ranked:
        blob, meta = download_bytes(url)
        downloaded.append({"url": url, "meta": meta, "stage": "asset"})
        if not blob:
            continue
        name = url.split("/")[-1] or "asset"
        tables += parse_asset(blob, name)
    summaries = []
    analyses = []
    for name, df in tables:
        q = table_quality(df)
        if q:
            summaries.append({"source_table": name, **q})
            analyses.append(analyze_pv(df, name))
    best = None
    good = [a for a in analyses if a.get("ok")]
    if good:
        best = sorted(good, key=lambda a: a.get("delta_aic_proxy_minus_base", 1e99))[0]
    result = {
        "schema": "ccdr-tierb-result-v1",
        "quality_patch_version": "v6_items_5_6_8_t48_additive",
        "generated_utc": now_utc_iso(),
        "test_id": "T48",
        "test_name": "photovoltaic acoustic-optical proxy",
        "prediction_ids": ["EN?"],
        "prediction_names": ["material symmetry/mass-contrast residual proxy for PV efficiency"],
        "source_strategy": "NREL PV page parsed as web app: direct candidates, script/link asset discovery, embedded JSON, HTML tables",
        "downloaded_sources": downloaded[:120],
        "discovered_asset_links_sample": ranked[:40],
        "tables_count": len(tables),
        "table_summaries": summaries[:40],
        "candidate_rows_count": max([a.get("candidate_rows_count", 0) for a in analyses] or [0]),
        "analysis_results": analyses[:20],
        "best_result": best,
        "readiness_status": "model_fit_done" if best else ("candidate_table_found_missing_required_columns" if summaries else "source_found_no_usable_table"),
        "evidence_status": "confirm_like" if best and best.get("support_like") else ("null_like" if best else "data_limited"),
        "support_like": None if best is None else bool(best.get("support_like")),
        "falsification_logic": {
            "caveat": "Run the residual model only if candidate_rows_count >= 100 and year/material/efficiency columns exist; area is used when present.",
            "confirm_like": "After year/area controls, acoustic-optical proxy has positive coefficient and improves AIC/RMS.",
            "falsify_like": "Adequate NREL rows exist and acoustic-optical proxy is absent/reversed or penalized by AIC/BIC."
        },
    }
    outpath = outdir / "t48_pv_acoustic_optical_proxy_v2_result.json"
    outpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
