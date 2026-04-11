#!/usr/bin/env python3
"""
Test 07: Pre-registered dark-matter tower scan on public direct-detection curves.

This script prefers public machine-readable HEPData table links and extracts only
actual table downloads rather than generic record resources. It then parses CSV or
JSON table payloads into numeric mass-cross-section curves and scans for residual
structure relative to a smooth envelope.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from _common_public_data import (
    _hepdata_try_record_json,
    download_text,
    save_json,
    smooth_curve_peak_scan,
)


HEPDATA_RECORD_URLS = [
    "https://www.hepdata.net/record/155182?format=json",
    "https://www.hepdata.net/record/ins3085605?version=2&format=json",
]


def _collect_table_resources(record_json: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    tables = record_json.get("data_tables") or record_json.get("tables") or []
    if not isinstance(tables, list):
        return out
    for item in tables:
        if not isinstance(item, dict):
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        name = str(item.get("name") or item.get("processed_name") or item.get("id") or "table")
        desc = str(item.get("description") or name)
        if isinstance(data.get("csv"), str):
            out.append({"name": name, "description": desc, "url": data["csv"], "format": "csv"})
        if isinstance(data.get("json"), str):
            out.append({"name": name, "description": desc, "url": data["json"], "format": "json"})
    # Deduplicate by URL while preserving order.
    dedup: Dict[str, Dict[str, str]] = {}
    for row in out:
        dedup.setdefault(row["url"], row)
    return list(dedup.values())


def fetch_candidate_tables() -> List[Dict[str, str]]:
    resources: List[Dict[str, str]] = []
    for i, record_url in enumerate(HEPDATA_RECORD_URLS, start=1):
        rec = _hepdata_try_record_json([record_url], cache_name=f"direct_detection_record_{i}.json")
        resources.extend(_collect_table_resources(rec))
    # Prefer SI / cross-section style tables first.
    def score(r: Dict[str, str]) -> Tuple[int, str]:
        txt = (r.get("name", "") + " " + r.get("description", "")).lower()
        s = 0
        if "cross section" in txt or "cross-section" in txt:
            s += 10
        if "si" in txt or "spin-independent" in txt:
            s += 8
        if "wimp" in txt:
            s += 4
        if "efficiency" in txt or "data points" in txt or txt.strip() == "data":
            s -= 10
        return (-s, r["url"])
    resources.sort(key=score)
    return resources


def _cache_name_for_url(url: str, suffix: str) -> str:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    return f"hepdata_{digest}.{suffix}"


def _parse_hepdata_json_table(text: str) -> pd.DataFrame | None:
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    indep = obj.get("independent_variables")
    dep = obj.get("dependent_variables")
    if not isinstance(indep, list) or not isinstance(dep, list) or not indep or not dep:
        return None

    def extract_numeric_values(var: Dict[str, Any]) -> List[float]:
        vals: List[float] = []
        for row in var.get("values", []) if isinstance(var.get("values"), list) else []:
            val = np.nan
            if isinstance(row, dict):
                if "value" in row:
                    try:
                        val = float(row["value"])
                    except Exception:
                        val = np.nan
                elif "low" in row and "high" in row:
                    try:
                        val = 0.5 * (float(row["low"]) + float(row["high"]))
                    except Exception:
                        val = np.nan
            vals.append(val)
        return vals

    x = extract_numeric_values(indep[0])
    best_df: pd.DataFrame | None = None
    best_rows = 0
    for depvar in dep:
        y = extract_numeric_values(depvar)
        n = min(len(x), len(y))
        if n < 5:
            continue
        df = pd.DataFrame({"x": x[:n], "y": y[:n]})
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df[(df["x"] > 0) & (df["y"] > 0)]
        if len(df) > best_rows:
            best_df = df
            best_rows = len(df)
    return best_df


def _parse_tabular_text(text: str) -> pd.DataFrame | None:
    stripped_lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    cleaned = "\n".join(stripped_lines)
    if not cleaned.strip():
        return None
    best_df: pd.DataFrame | None = None
    best_rows = 0
    for sep in [",", "\t", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(cleaned), sep=sep, engine="python")
        except Exception:
            continue
        if df.empty:
            continue
        # Coerce every column; HEPData CSV often mixes strings in the header rows.
        num = df.apply(pd.to_numeric, errors="coerce")
        # Keep columns that contain at least a few numeric values.
        num = num.loc[:, num.notna().sum(axis=0) >= 5]
        if num.shape[1] < 2:
            continue
        # Choose the two columns with the most numeric support.
        keep = list(num.notna().sum(axis=0).sort_values(ascending=False).index[:2])
        cand = num[keep].copy()
        cand.columns = ["x", "y"]
        cand = cand.replace([np.inf, -np.inf], np.nan).dropna()
        cand = cand[(cand["x"] > 0) & (cand["y"] > 0)]
        if len(cand) > best_rows:
            best_df = cand
            best_rows = len(cand)
    # Last-resort manual extraction: two numeric tokens per line.
    if best_df is None:
        rows = []
        for ln in stripped_lines:
            toks = [t.strip() for t in ln.replace(";", ",").split(",")]
            nums = []
            for tok in toks:
                try:
                    nums.append(float(tok))
                except Exception:
                    pass
            if len(nums) >= 2:
                rows.append((nums[0], nums[1]))
        if len(rows) >= 5:
            best_df = pd.DataFrame(rows, columns=["x", "y"])
    return best_df


def load_any_numeric_table(resource: Dict[str, str]) -> pd.DataFrame | None:
    url = resource["url"]
    fmt = resource.get("format", "csv")
    suffix = "json" if fmt == "json" else "csv"
    try:
        txt = download_text([url], cache_name=_cache_name_for_url(url, suffix), timeout=180)
    except Exception:
        return None
    if fmt == "json":
        df = _parse_hepdata_json_table(txt)
        if df is not None and len(df) >= 5:
            return df
    df = _parse_tabular_text(txt)
    if df is not None and len(df) >= 5:
        return df
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test07_dm_tower_preregistered_scan"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    resources = fetch_candidate_tables()
    numeric_tables = []
    attempted = []
    for resource in resources:
        attempted.append(resource)
        df = load_any_numeric_table(resource)
        if df is not None and len(df) >= 10:
            numeric_tables.append((resource, df))

    curve_summaries = []
    peak_summaries = []
    for resource, df in numeric_tables[:12]:
        x = np.asarray(df["x"], dtype=float)
        y = np.asarray(df["y"], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]
        if len(x) < 20:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        scan = smooth_curve_peak_scan(x, y, window=15, polyorder=3)
        curve_summaries.append({
            "source": resource["url"],
            "name": resource.get("name", "table"),
            "format": resource.get("format", "unknown"),
            "n_points": int(len(x)),
            "mass_min": float(np.min(x)),
            "mass_max": float(np.max(x)),
            "n_peaks": int(scan["n_peaks"]),
        })
        peak_summaries.append({"source": resource["url"], "name": resource.get("name", "table"), **scan})

    summary = {
        "test_name": "Dark-matter tower pre-registered scan",
        "n_candidate_resources": int(len(resources)),
        "n_numeric_tables_loaded": int(len(curve_summaries)),
        "curve_summaries": curve_summaries,
        "peak_scans": peak_summaries,
        "falsification_logic": {
            "confirm_like": "Once public sensitivity reaches the predicted mass range, multiple residual peaks with roughly geometric spacing should appear in the same scan family.",
            "falsify_like": "Sensitivity reaches the predicted second-peak region and no second peak appears.",
        },
        "notes": [
            "This is a preregistered curve-based proxy because public event-level likelihoods are not uniformly available.",
            f"Attempted {len(attempted)} public machine-readable direct-detection table resources.",
            "HEPData record JSON is used to extract direct table CSV/JSON links instead of generic record resources.",
            "Null results at masses far below the predicted second peak do not count as a hard falsification.",
        ],
    }
    save_json(args.outdir / "test07_dm_tower_preregistered_scan_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
