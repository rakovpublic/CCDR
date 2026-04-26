from __future__ import annotations

"""
FR3 public-data gate / numeric runner.

Prediction: E_ELM ~ P_ped * V_ped * (DeltaP/P_ped)^2.

Default public MAST shot API does not expose ELM energy, so it remains a gate.
If a public numeric table is available, set FR3_NUMERIC_URLS or FR3_ELM_TABLE_URL
(comma/semicolon separated URLs). The script will download CSV/XLSX/TXT files,
find E_ELM / pedestal pressure / volume-width / delta-p columns, and run the
log-log slope test automatically.
"""

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from _common_public_data import DEFAULT_HEADERS, TIMEOUT, cached_download, ensure_dir, fit_linear, json_dump, structured_report

ROOT = Path(__file__).resolve().parent
OUT = ensure_dir(ROOT / "outputs" / "fr3")
API = "https://mastapp.site/json/shots"

REQUIRED = {
    "elm_energy": re.compile(r"(^|[_\s-])(e_?elm|elm.*(energy|fluence)|divertor.*fluence|energy.*elm)([_\s-]|$)", re.I),
    "pedestal_pressure": re.compile(r"p(ed)?(estal)?(_|\s|-)*(press|p)|pressure.*ped|p_?ped", re.I),
    "pedestal_volume": re.compile(r"v(ed)?|v_?ped|volume|ped.*width|width.*ped|delta.*ped", re.I),
    "delta_p": re.compile(r"delta.*p|d_?p|dp|pressure.*drop|ped.*drop", re.I),
}
PROXY = {
    "stored_energy": re.compile(r"generic_(max_)?(total_)?energy$|generic_energy_max|generic_dt_total_energy", re.I),
    "volume": re.compile(r"generic_max_plasma_vol$|generic_plasma_vol", re.I),
    "power_loss": re.compile(r"generic_max_power_loss$|nbi_power_max_power|generic_max_plasma_power_loss", re.I),
    "q95": re.compile(r"generic_min_q95$|generic_q95", re.I),
}


def _find_numeric(df: pd.DataFrame, pat: re.Pattern, min_count: int = 10) -> str | None:
    for c in df.columns:
        name = str(c).strip().replace("\ufeff", "")
        if not name:
            continue
        if pat.search(name):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= min_count:
                return str(c)
    return None


def _read_table_url(url: str, idx: int) -> tuple[str, pd.DataFrame | None, str | None]:
    suffix = Path(urlparse(url).path).suffix.lower() or ".dat"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(urlparse(url).path).name or f"fr3_table_{idx}{suffix}")
    path = cached_download(url, OUT / "external_tables" / safe)
    try:
        if suffix in {".xlsx", ".xls"}:
            return str(path), pd.read_excel(path), None
        for sep in [",", "\t", ";", r"\s+|,|;|\t"]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python")
                if len(df.columns) >= 3 and len(df) >= 5:
                    return str(path), df, None
            except Exception:
                pass
        return str(path), None, "unparseable_table"
    except Exception as e:
        return str(path), None, f"{type(e).__name__}: {e}"


def _numeric_fit_from_df(df: pd.DataFrame, source: str) -> dict:
    cols = {k: _find_numeric(df, p, min_count=5) for k, p in REQUIRED.items()}
    missing = [k for k, v in cols.items() if not v]
    if missing:
        return {"source": source, "status": "missing_required_columns", "columns_found": cols, "missing": missing}

    e = pd.to_numeric(df[cols["elm_energy"]], errors="coerce")
    p = pd.to_numeric(df[cols["pedestal_pressure"]], errors="coerce")
    v = pd.to_numeric(df[cols["pedestal_volume"]], errors="coerce")
    dp = pd.to_numeric(df[cols["delta_p"]], errors="coerce")
    target = p * v * (dp / p) ** 2
    mask = e.notna() & target.notna() & (e > 0) & (target > 0)
    if mask.sum() < 8:
        return {"source": source, "status": "insufficient_valid_rows", "columns_found": cols, "n_valid": int(mask.sum())}

    lx = np.log(target[mask].to_numpy(float))
    ly = np.log(e[mask].to_numpy(float))
    fit = fit_linear(lx, ly)
    slope = fit.params["a"]
    verdict = "support_like_slope_near_one" if abs(slope - 1.0) <= 0.35 and fit.r2 > 0.25 else "no_support_slope_not_near_one"
    return {
        "source": source,
        "status": "ok",
        "columns_found": cols,
        "n_used": int(mask.sum()),
        "loglog_slope": float(slope),
        "r2": fit.r2,
        "rmse": fit.rmse,
        "verdict": verdict,
        "criterion": "support-like if log(E_ELM) vs log(P_ped*V*(DeltaP/P)^2) slope is within ±0.35 of 1 and R²>0.25",
    }


def _external_numeric_tests() -> list[dict]:
    raw = os.environ.get("FR3_NUMERIC_URLS") or os.environ.get("FR3_ELM_TABLE_URL") or ""
    urls = [u.strip() for u in re.split(r"[,;\n]+", raw) if u.strip()]
    out = []
    for i, url in enumerate(urls):
        source, df, err = _read_table_url(url, i)
        if df is None:
            out.append({"source_url": url, "downloaded_path": source, "status": "unparseable", "error": err})
        else:
            r = _numeric_fit_from_df(df, source)
            r["source_url"] = url
            out.append(r)
    return out


def _mast_gate() -> dict:
    try:
        r = requests.get(API, headers=DEFAULT_HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", data if isinstance(data, list) else [])
    except Exception as e:
        return structured_report("FR3", "source_unavailable", source=API, reason=repr(e), verdict="no_physics_verdict")

    keys = set(); rows = []
    for it in items[:500]:
        if isinstance(it, dict):
            keys.update(map(str, it.keys())); rows.append(it)
    found = {n: [k for k in sorted(keys) if p.search(k)] for n, p in REQUIRED.items()}
    missing = [n for n, v in found.items() if not v]

    proxy_result = None
    df = pd.DataFrame(rows)
    pcols = {n: _find_numeric(df, p) for n, p in PROXY.items()}
    if pcols.get("stored_energy") and pcols.get("volume"):
        e = pd.to_numeric(df[pcols["stored_energy"]], errors="coerce")
        v = pd.to_numeric(df[pcols["volume"]], errors="coerce")
        m = e.notna() & v.notna() & (e > 0) & (v > 0)
        if m.sum() >= 30:
            slope = float(np.polyfit(np.log(v[m]), np.log(e[m]), 1)[0])
            proxy_result = {
                "proxy_name": "stored_energy_vs_plasma_volume_not_ELM",
                "columns": pcols,
                "n_used": int(m.sum()),
                "loglog_slope_energy_vs_volume": slope,
                "spearman_energy_volume": float(v[m].corr(e[m], method="spearman")),
                "interpretation": "diagnostic only: public MAST shot API lacks ELM-energy columns, so this cannot confirm FR3.",
            }

    return structured_report(
        "FR3",
        "not_executable_missing_public_columns" if missing else "ready_for_numeric_fit",
        source=API,
        n_rows_seen=len(items),
        available_keys_sample=sorted(keys)[:140],
        required_fields_found=found,
        missing_required=missing,
        proxy_energy_density_audit=proxy_result,
        verdict="no_physics_verdict" if missing else "can_fit_prediction",
        next_numeric_test="fit log(E_ELM) ~ log(P_ped*V_ped*(DeltaP/P_ped)^2) with slope≈1 and compare against null scalings",
        note="MAST shot API is a data-gate only source. Provide FR3_NUMERIC_URLS / FR3_ELM_TABLE_URL for a real numeric FR3 fit.",
    )


def main() -> None:
    external = _external_numeric_tests()
    if external:
        ok = [r for r in external if r.get("status") == "ok"]
        support = [r for r in ok if str(r.get("verdict", "")).startswith("support")]
        report = structured_report(
            "FR3", "ok_external_numeric" if ok else "external_tables_no_fit",
            external_numeric_tests=external,
            verdict="support_like" if support else ("no_support_or_no_fit" if ok else "no_physics_verdict"),
        )
    else:
        report = _mast_gate()
    json_dump(report, OUT / "fr3_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
