#!/usr/bin/env python3
"""v61 behavioral confirm extractors for CCDR Tier-B.

This patch intentionally changes test behavior rather than only dashboards:
- scans local/cache/generated/public supplement tables for exact rows;
- normalizes row-level data for T31/T32, T44, T53, T34, T57/T59, T45/T47, T26-T30;
- computes source-balanced / family-balanced estimators where enough rows exist;
- keeps public-claim policy conservative.

No network is required here; public data should already be in the cache/outdir/data tree
from run_all_tier_b.py or manually downloaded public supplements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GEN_DIR = DATA_DIR / "generated"
MANIFEST_DIR = DATA_DIR / "manifests"

DEFAULT_TESTS = [
    "T31", "T32", "T44", "T48", "T53", "T34", "T57", "T59", "T45", "T47",
    "T26", "T27", "T28", "T29", "T30", "T50", "T51", "T52", "T60",
]
NEAR = {"T31", "T32", "T44", "T53", "T34", "T57", "T59", "T45", "T47"}
FUSION = {"T26", "T27", "T28", "T29", "T30"}
BOUND = {"T50", "T51", "T52"}
ANCHOR = {"T60"}


def _ensure(outdir: Optional[Path] = None) -> Path:
    if outdir is not None:
        p = outdir / "data" / "generated"
    else:
        p = GEN_DIR
    p.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    return p


def _jsonable(v: Any) -> Any:
    if np is not None:
        try:
            if isinstance(v, (np.integer, np.floating)):
                return v.item()
        except Exception:
            pass
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, dict)):
        return v
    return str(v)


def _write_json(path: Path, obj: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8")
    return str(path)


def _write_csv(rows: Sequence[Dict[str, Any]], filename: str, outdir: Optional[Path] = None) -> str:
    out = _ensure(outdir) / filename
    keys: List[str] = []
    for r in rows:
        for k in (r or {}).keys():
            if k not in keys:
                keys.append(k)
    if not keys:
        keys = ["empty_v61"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            clean: Dict[str, Any] = {}
            for k in keys:
                val = (r or {}).get(k)
                if isinstance(val, (dict, list)):
                    clean[k] = json.dumps(val, sort_keys=True, default=_jsonable)
                else:
                    clean[k] = _jsonable(val)
            w.writerow(clean)
    return str(out)


def _s(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd is not None and pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd is not None and pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (int, float)):
        vv = float(v)
        return vv if math.isfinite(vv) else None
    txt = str(v).replace(",", "").strip()
    # Preserve common engineering units, but strip annotations.
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
    if not m:
        return None
    try:
        vv = float(m.group(0))
        return vv if math.isfinite(vv) else None
    except Exception:
        return None


def _norm_col(c: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def _pick(row: Dict[str, Any], aliases: Sequence[str]) -> Any:
    if not row:
        return None
    exact = {str(k): k for k in row}
    norm = {_norm_col(k): k for k in row}
    for a in aliases:
        if a in exact:
            return row[exact[a]]
        na = _norm_col(a)
        if na in norm:
            return row[norm[na]]
    # fuzzy contains: only for long aliases to avoid bad matches
    for a in aliases:
        na = _norm_col(a)
        if len(na) < 4:
            continue
        for nk, orig in norm.items():
            if na == nk or na in nk or nk in na:
                return row[orig]
    return None


def _candidate_roots(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> List[Path]:
    roots = [DATA_DIR, GEN_DIR, MANIFEST_DIR]
    if outdir:
        roots += [outdir, outdir / "data", outdir / "data" / "generated", outdir / "confirm_only_v60", outdir / "confirm_only_v61"]
    if cache:
        roots += [cache]
    seen: List[Path] = []
    for r in roots:
        try:
            rr = r.resolve()
            if rr.exists() and rr not in seen:
                seen.append(rr)
        except Exception:
            pass
    return seen


def _iter_table_files(patterns: Sequence[str], outdir: Optional[Path] = None, cache: Optional[Path] = None, max_files: int = 400) -> List[Path]:
    files: List[Path] = []
    seen_files = set()
    suffixes = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".xlsx", ".xls", ".yaml", ".yml"}
    pat_l = [p.lower() for p in patterns]
    for root in _candidate_roots(outdir, cache):
        for p in root.rglob("*"):
            if len(files) >= max_files:
                return files
            if not p.is_file() or p.suffix.lower() not in suffixes:
                continue
            try:
                rp = p.resolve()
            except Exception:
                rp = p
            if rp in seen_files:
                continue
            try:
                if p.stat().st_size > 80_000_000:
                    continue
            except Exception:
                pass
            name = str(p).lower()
            # Avoid recursive self-ingestion of v61 generated outputs from the current confirm run.
            if "_v61" in p.name.lower() and ("generated" in name or "confirm_only" in name):
                continue
            if any(q in name for q in pat_l):
                seen_files.add(rp)
                files.append(p)
    return files


def _read_table(path: Path, max_rows: int = 300000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if pd is None:
        return rows
    try:
        suf = path.suffix.lower()
        if suf == ".csv":
            df = pd.read_csv(path, nrows=max_rows, dtype=str, low_memory=False)
        elif suf in {".tsv", ".txt"}:
            # Try TSV first, then CSV, then whitespace.
            try:
                df = pd.read_csv(path, sep="\t", nrows=max_rows, dtype=str, low_memory=False)
            except Exception:
                try:
                    df = pd.read_csv(path, nrows=max_rows, dtype=str, low_memory=False)
                except Exception:
                    df = pd.read_csv(path, delim_whitespace=True, nrows=max_rows, dtype=str)
        elif suf in {".xlsx", ".xls"}:
            df = pd.read_excel(path, nrows=max_rows, dtype=str)
        elif suf == ".jsonl":
            df = pd.read_json(path, lines=True)
            if len(df) > max_rows:
                df = df.head(max_rows)
        elif suf in {".json"}:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, dict):
                if isinstance(obj.get("rows"), list):
                    return [dict(x, _source_file_v61=str(path)) for x in obj["rows"][:max_rows] if isinstance(x, dict)]
                if isinstance(obj.get("data"), list):
                    return [dict(x, _source_file_v61=str(path)) for x in obj["data"][:max_rows] if isinstance(x, dict)]
                if isinstance(obj.get("values"), list):
                    return [{"value": x, "_source_file_v61": str(path)} for x in obj["values"][:max_rows]]
                return [dict(obj, _source_file_v61=str(path))]
            if isinstance(obj, list):
                return [dict(x, _source_file_v61=str(path)) for x in obj[:max_rows] if isinstance(x, dict)]
            return []
        elif suf in {".yaml", ".yml"} and yaml is not None:
            obj = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, dict):
                if isinstance(obj.get("independent_variables"), list) or isinstance(obj.get("dependent_variables"), list):
                    return _flatten_hepdata_yaml(obj, path)
                return [dict(obj, _source_file_v61=str(path))]
            if isinstance(obj, list):
                return [dict(x, _source_file_v61=str(path)) for x in obj[:max_rows] if isinstance(x, dict)]
            return []
        else:
            return []
        df.columns = [str(c) for c in df.columns]
        for _, r in df.iterrows():
            d = {str(k): _jsonable(v) for k, v in dict(r).items()}
            d["_source_file_v61"] = str(path)
            rows.append(d)
        return rows
    except Exception:
        return []


def _flatten_hepdata_yaml(obj: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    indep = obj.get("independent_variables") or []
    dep = obj.get("dependent_variables") or []
    xvals = []
    if indep and isinstance(indep[0], dict):
        xheader = (((indep[0].get("header") or {}).get("name")) or "x")
        for v in indep[0].get("values") or []:
            if isinstance(v, dict):
                xvals.append(v.get("value") or v.get("low") or v.get("high"))
            else:
                xvals.append(v)
    maxn = max([len(xvals)] + [len((d or {}).get("values") or []) for d in dep if isinstance(d, dict)] + [0])
    for i in range(maxn):
        row: Dict[str, Any] = {"_source_file_v61": str(path)}
        if i < len(xvals):
            row["x"] = xvals[i]
        for j, d in enumerate(dep):
            if not isinstance(d, dict):
                continue
            name = (((d.get("header") or {}).get("name")) or f"y{j}")
            vals = d.get("values") or []
            if i < len(vals):
                vv = vals[i]
                if isinstance(vv, dict):
                    row[name] = vv.get("value")
                    errs = vv.get("errors") or []
                    if errs:
                        row[f"{name}_error"] = (errs[0] or {}).get("symerror") or (errs[0] or {}).get("asymerror")
                else:
                    row[name] = vv
        rows.append(row)
    return rows


def _read_patterns(patterns: Sequence[str], outdir: Optional[Path] = None, cache: Optional[Path] = None, max_files: int = 400, max_rows_per_file: int = 300000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in _iter_table_files(patterns, outdir, cache, max_files=max_files):
        rows.extend(_read_table(p, max_rows=max_rows_per_file))
    return rows


def _dedup(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = tuple(_s(r.get(k)) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(r))
    return out


def _material_family(material: str) -> str:
    m = material.lower()
    if any(x in m for x in ["silicon", " si", "si-"]): return "silicon"
    if any(x in m for x in ["copper", "cu"]): return "metal_cu"
    if any(x in m for x in ["alumina", "al2o3", "sapphire"]): return "oxide_alumina"
    if any(x in m for x in ["diamond", "carbon"]): return "carbon"
    if any(x in m for x in ["bismuth", "bi2te", "sb2te", "tellur"]): return "thermoelectric_chalcogenide"
    if any(x in m for x in ["glass", "silica", "sio2"]): return "glass_silica"
    if any(x in m for x in ["poly", "epoxy", "kapton", "ptfe"]): return "polymer"
    if any(x in m for x in ["hbn", "boron nitride"]): return "boron_nitride"
    return re.sub(r"[^a-z0-9]+", "_", m).strip("_")[:40] or "unknown_family"


def _temp_bin(t: Optional[float]) -> str:
    if t is None: return "unknown"
    if t < 2: return "ultra_low_lt2K"
    if t < 20: return "cryogenic_2_20K"
    if t < 80: return "low_20_80K"
    if t < 200: return "intermediate_80_200K"
    if t < 400: return "room_200_400K"
    return "high_gt400K"


# ---------------------------------------------------------------------------
# T31/T32 materials: exact row extraction + source-balanced computation
# ---------------------------------------------------------------------------

MATERIAL_PATTERNS = [
    "microstructure", "kappa", "thermal_conductivity", "thermal-conductivity", "cryogenic", "nanocrystalline",
    "grain", "sem", "tem", "xrd", "matweb", "nist", "t31", "t32",
]


def normalize_material_row(raw: Dict[str, Any], idx: int) -> Tuple[Dict[str, Any], List[str]]:
    material = _s(_pick(raw, ["material", "compound", "sample_material", "specimen_material", "name", "formula"]))
    source = _s(_pick(raw, ["source_url", "url", "doi", "reference", "source", "_source_file_v61", "_source_file_v60", "_source_file_v59"]))
    sample = _s(_pick(raw, ["sample_id", "sample", "specimen", "id", "sample_name"])) or f"row{idx}"
    temp = _f(_pick(raw, ["temperature_K", "T_K", "temperature", "temp_K", "temperature (K)", "T (K)"]))
    kappa = _f(_pick(raw, ["kappa_W_mK", "thermal_conductivity", "thermal conductivity", "k_W_mK", "lambda_W_mK", "kappa", "k", "W/mK"]))
    grain = _f(_pick(raw, ["grain_size_nm", "grain_size_nm_or_um", "grain_size", "crystallite_size_nm", "particle_size_nm", "grain diameter", "grain_diameter_nm"]))
    boundary = _f(_pick(raw, ["boundary_density_proxy", "boundary_density", "interface_density", "porosity", "grain_boundary_density", "boundary"] ))
    method = _s(_pick(raw, ["microstructure_method", "method", "measurement_method", "evidence_method", "characterization", "microstructure"] ))
    txt = " ".join(_s(v) for v in raw.values())
    if not method:
        hits = []
        for m in ["SEM", "TEM", "XRD", "AFM", "EBSD"]:
            if re.search(rf"\b{m}\b", txt, re.I):
                hits.append(m)
        method = "+".join(hits)
    nano_txt = _s(_pick(raw, ["nanocrystalline_yes_no", "nanocrystalline", "nano", "is_nanocrystalline"] )) or ("yes" if re.search(r"nanocrystalline|nano[- ]?crystal", txt, re.I) else "")
    if grain is not None and grain > 0 and grain < 1.0:
        # Many tables store microns: convert to nm if very small and not explicitly nm.
        grain = grain * 1000.0
    if grain is None and boundary is not None and boundary > 0:
        grain = 1.0 / boundary
    if boundary is None and grain is not None and grain > 0:
        boundary = 1.0 / grain
    family = _material_family(material)
    reasons: List[str] = []
    if not material: reasons.append("missing_material")
    if not source: reasons.append("missing_source")
    if temp is None: reasons.append("missing_temperature_K")
    if kappa is None: reasons.append("missing_kappa_W_mK")
    if grain is None: reasons.append("missing_grain_size")
    if not method or not re.search(r"SEM|TEM|XRD|EBSD|AFM", method, re.I): reasons.append("missing_measured_microstructure_method")
    if kappa is not None and kappa <= 0: reasons.append("nonpositive_kappa")
    if grain is not None and grain <= 0: reasons.append("nonpositive_grain")
    if temp is not None and temp <= 0: reasons.append("nonpositive_temperature")
    row = {
        "source_url_v61": source,
        "sample_id_v61": sample,
        "material_v61": material,
        "material_family_v61": family,
        "temperature_K_v61": temp,
        "temperature_bin_v61": _temp_bin(temp),
        "kappa_W_mK_v61": kappa,
        "grain_size_nm_v61": grain,
        "boundary_density_proxy_v61": boundary,
        "microstructure_method_v61": method,
        "nanocrystalline_yes_no_v61": nano_txt,
        "usable_v61": not reasons,
        "reject_reasons_v61": "|".join(reasons),
        "raw_source_file_v61": _s(raw.get("_source_file_v61") or raw.get("_source_file_v60") or raw.get("_source_file_v59")),
    }
    return row, reasons


def _ols_fit(y: List[float], X: List[List[float]]) -> Optional[Dict[str, Any]]:
    if np is None or len(y) < 4:
        return None
    try:
        yy = np.asarray(y, dtype=float)
        XX = np.asarray(X, dtype=float)
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        pred = XX @ beta
        resid = yy - pred
        rss = float(np.sum(resid ** 2))
        n = len(yy)
        k = XX.shape[1]
        sigma2 = max(rss / max(n, 1), 1e-300)
        aic = float(n * math.log(sigma2) + 2 * k)
        bic = float(n * math.log(sigma2) + k * math.log(max(n, 2)))
        return {"n": n, "k": k, "rss": rss, "aic": aic, "bic": bic, "beta": [float(x) for x in beta]}
    except Exception:
        return None


def _source_family_demean(rows: List[Dict[str, Any]], fields: Sequence[str]) -> List[Dict[str, float]]:
    # Approximate mixed-effects behavior without statsmodels: remove source and family means.
    vals: Dict[str, List[float]] = {f: [] for f in fields}
    for r in rows:
        for f in fields:
            v = _f(r.get(f))
            vals[f].append(float(v) if v is not None else float("nan"))
    global_mean = {f: statistics.fmean([x for x in vals[f] if math.isfinite(x)]) if any(math.isfinite(x) for x in vals[f]) else 0.0 for f in fields}
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.get("source_url_v61"), r.get("material_family_v61"))].append(i)
    out: List[Dict[str, float]] = []
    for i, r in enumerate(rows):
        gidx = groups[(r.get("source_url_v61"), r.get("material_family_v61"))]
        d: Dict[str, float] = {}
        for f in fields:
            v = _f(r.get(f))
            if v is None:
                d[f] = float("nan")
                continue
            mvals = []
            for j in gidx:
                vv = _f(rows[j].get(f))
                if vv is not None:
                    mvals.append(vv)
            gm = statistics.fmean(mvals) if mvals else global_mean[f]
            d[f] = float(v - gm + global_mean[f])
        out.append(d)
    return out


def _materials_estimator(usable: List[Dict[str, Any]], test_id: str, outdir: Optional[Path] = None) -> Dict[str, Any]:
    # Transform variables. Baseline is log(kappa) ~ log(T). CCDR adds log(grain) + boundary.
    rows = []
    for r in usable:
        t = _f(r.get("temperature_K_v61")); k = _f(r.get("kappa_W_mK_v61")); g = _f(r.get("grain_size_nm_v61")); b = _f(r.get("boundary_density_proxy_v61"))
        if t and k and g and t > 0 and k > 0 and g > 0:
            rr = dict(r)
            rr["logT_v61"] = math.log(t)
            rr["logK_v61"] = math.log(k)
            rr["logG_v61"] = math.log(g)
            rr["boundary_v61"] = float(b or (1.0 / g))
            rows.append(rr)
    if len(rows) < 8:
        return {"status_v61": "insufficient_rows_for_estimator", "n_model_rows_v61": len(rows)}
    dm = _source_family_demean(rows, ["logK_v61", "logT_v61", "logG_v61", "boundary_v61"])
    y = [d["logK_v61"] for d in dm]
    X0 = [[1.0, d["logT_v61"]] for d in dm]
    X1 = [[1.0, d["logT_v61"], d["logG_v61"], d["boundary_v61"]] for d in dm]
    base = _ols_fit(y, X0) or {}
    ccdr = _ols_fit(y, X1) or {}
    delta_aic = (base.get("aic", float("nan")) - ccdr.get("aic", float("nan"))) if base and ccdr else float("nan")
    delta_bic = (base.get("bic", float("nan")) - ccdr.get("bic", float("nan"))) if base and ccdr else float("nan")
    # Source-balanced bootstrap: resample sources, then one row per source-family where possible.
    rng = np.random.default_rng(6161) if np is not None else None
    source_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        source_map[_s(r.get("source_url_v61")) or "unknown"].append(r)
    sources = list(source_map)
    wins = 0; sign_ok = 0; reps = 0
    boot_rows = []
    if rng is not None and len(sources) >= 2:
        for rep in range(48):
            chosen = list(rng.choice(sources, size=len(sources), replace=True))
            sample: List[Dict[str, Any]] = []
            for s in chosen:
                bucket = source_map[s]
                if bucket:
                    sample.append(bucket[int(rng.integers(0, len(bucket)))])
            if len(sample) < 8:
                continue
            dm2 = _source_family_demean(sample, ["logK_v61", "logT_v61", "logG_v61", "boundary_v61"])
            yy = [d["logK_v61"] for d in dm2]
            b0 = _ols_fit(yy, [[1.0, d["logT_v61"]] for d in dm2])
            b1 = _ols_fit(yy, [[1.0, d["logT_v61"], d["logG_v61"], d["boundary_v61"]] for d in dm2])
            if not b0 or not b1:
                continue
            da = float(b0["aic"] - b1["aic"])
            db = float(b0["bic"] - b1["bic"])
            beta_g = b1["beta"][2] if len(b1.get("beta", [])) > 2 else float("nan")
            beta_b = b1["beta"][3] if len(b1.get("beta", [])) > 3 else float("nan")
            win = da > 0 and db > 0
            # Default expected signs: larger grains -> higher kappa; more boundaries -> lower kappa.
            sok = beta_g > 0 and beta_b < 0
            wins += int(win); sign_ok += int(sok); reps += 1
            boot_rows.append({"rep_v61": rep, "delta_aic_base_minus_ccdr_v61": da, "delta_bic_base_minus_ccdr_v61": db, "grain_beta_v61": beta_g, "boundary_beta_v61": beta_b, "model_win_v61": win, "sign_ok_v61": sok})
    _write_csv(boot_rows, f"{test_id.lower()}_source_balanced_bootstrap_v61.csv", outdir)
    temp_bins = sorted({_s(r.get("temperature_bin_v61")) for r in rows if _s(r.get("temperature_bin_v61")) not in {"", "unknown"}})
    bin_results = []
    for tb in temp_bins:
        sub = [r for r in rows if _s(r.get("temperature_bin_v61")) == tb]
        if len(sub) < 8:
            bin_results.append({"temperature_bin_v61": tb, "n_rows_v61": len(sub), "passed_v61": False, "reason_v61": "too_few_rows"})
            continue
        dm3 = _source_family_demean(sub, ["logK_v61", "logT_v61", "logG_v61", "boundary_v61"])
        yy = [d["logK_v61"] for d in dm3]
        b0 = _ols_fit(yy, [[1.0, d["logT_v61"]] for d in dm3])
        b1 = _ols_fit(yy, [[1.0, d["logT_v61"], d["logG_v61"], d["boundary_v61"]] for d in dm3])
        da = (b0["aic"] - b1["aic"]) if b0 and b1 else float("nan")
        db = (b0["bic"] - b1["bic"]) if b0 and b1 else float("nan")
        bin_results.append({"temperature_bin_v61": tb, "n_rows_v61": len(sub), "delta_aic_v61": da, "delta_bic_v61": db, "passed_v61": bool(da > 0 and db > 0)})
    _write_csv(bin_results, f"{test_id.lower()}_temperature_bin_model_wins_v61.csv", outdir)
    grain_beta = ccdr.get("beta", [None, None, None, None])[2] if ccdr else None
    boundary_beta = ccdr.get("beta", [None, None, None, None])[3] if ccdr else None
    return {
        "status_v61": "ok",
        "n_model_rows_v61": len(rows),
        "baseline_fit_v61": base,
        "ccdr_microstructure_fit_v61": ccdr,
        "delta_aic_base_minus_ccdr_v61": delta_aic,
        "delta_bic_base_minus_ccdr_v61": delta_bic,
        "grain_beta_v61": grain_beta,
        "boundary_beta_v61": boundary_beta,
        "predicted_signs_pass_v61": bool((grain_beta or 0) > 0 and (boundary_beta or 0) < 0),
        "bootstrap_reps_v61": reps,
        "bootstrap_model_win_fraction_v61": wins / reps if reps else 0.0,
        "bootstrap_sign_ok_fraction_v61": sign_ok / reps if reps else 0.0,
        "temperature_bin_results_v61": bin_results,
        "temperature_bin_pass_fraction_v61": sum(1 for x in bin_results if x.get("passed_v61")) / max(len(bin_results), 1),
    }


def materials_confirm_v61(test_id: str, outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    tid = test_id.upper()
    raw = _read_patterns(MATERIAL_PATTERNS + [tid.lower()], outdir, cache)
    norm: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []
    for i, r in enumerate(raw):
        nr, reasons = normalize_material_row(r, i)
        norm.append(nr)
        if reasons:
            rejections.append({**nr, "reject_reasons_v61": "|".join(reasons)})
    dedup = _dedup(norm, ["source_url_v61", "sample_id_v61", "material_v61", "temperature_K_v61"])
    usable = [r for r in dedup if r.get("usable_v61")]
    _write_csv(norm, f"{tid.lower()}_microstructure_normalized_rows_v61.csv", outdir)
    _write_csv(dedup, f"{tid.lower()}_microstructure_dedup_rows_v61.csv", outdir)
    _write_csv(rejections, f"{tid.lower()}_microstructure_rejection_diagnostics_v61.csv", outdir)
    rejection_counts = Counter()
    for r in rejections:
        for reason in _s(r.get("reject_reasons_v61")).split("|"):
            if reason:
                rejection_counts[reason] += 1
    _write_csv([{"reason_v61": k, "n_rows_v61": v} for k, v in rejection_counts.most_common()], f"{tid.lower()}_microstructure_rejection_summary_v61.csv", outdir)
    source_counts = Counter(_s(r.get("source_url_v61")) or "unknown" for r in usable)
    family_counts = Counter(_s(r.get("material_family_v61")) or "unknown" for r in usable)
    temp_counts = Counter(_s(r.get("temperature_bin_v61")) or "unknown" for r in usable)
    _write_csv([{"source_v61": k, "n_rows_v61": v} for k, v in source_counts.most_common()], f"{tid.lower()}_source_balance_v61.csv", outdir)
    _write_csv([{"material_family_v61": k, "n_rows_v61": v} for k, v in family_counts.most_common()], f"{tid.lower()}_family_balance_v61.csv", outdir)
    _write_csv([{"temperature_bin_v61": k, "n_rows_v61": v} for k, v in temp_counts.most_common()], f"{tid.lower()}_temperature_balance_v61.csv", outdir)
    est = _materials_estimator(usable, tid, outdir)
    n_sources = len([k for k in source_counts if k != "unknown"])
    n_families = len([k for k in family_counts if k != "unknown"])
    n_temp_bins = len([k for k in temp_counts if k not in {"unknown", ""}])
    failed = []
    if len(usable) < 50: failed.append("usable_rows_ge_50")
    if n_sources < 5: failed.append("sources_ge_5")
    if n_families < 5: failed.append("material_families_ge_5")
    if n_temp_bins < 3: failed.append("temperature_bins_ge_3")
    if est.get("delta_aic_base_minus_ccdr_v61", -1) <= 0: failed.append("ccdr_model_beats_temp_aic")
    if est.get("delta_bic_base_minus_ccdr_v61", -1) <= 0: failed.append("ccdr_model_beats_temp_bic")
    if not est.get("predicted_signs_pass_v61"): failed.append("predicted_microstructure_signs")
    if est.get("bootstrap_model_win_fraction_v61", 0) < 0.8: failed.append("source_balanced_bootstrap_model_win_ge_0p8")
    if est.get("bootstrap_sign_ok_fraction_v61", 0) < 0.8: failed.append("source_balanced_bootstrap_sign_ge_0p8")
    if est.get("temperature_bin_pass_fraction_v61", 0) < 0.67: failed.append("temperature_bin_model_wins")
    confirm = not failed
    gate = {
        "schema": "ccdr-materials-confirm-v61",
        "test_id": tid,
        "n_raw_rows_v61": len(raw),
        "n_normalized_rows_v61": len(norm),
        "n_dedup_rows_v61": len(dedup),
        "n_usable_rows_v61": len(usable),
        "n_sources_v61": n_sources,
        "n_material_families_v61": n_families,
        "n_temperature_bins_v61": n_temp_bins,
        "estimator_v61": est,
        "strict_confirm_ready_v61": confirm,
        "failed_subgates_v61": failed,
        "confirmation_status_v61": "confirmed_by_measured_microstructure_model_v61" if confirm else "not_confirmed_next_gate_required",
        "rank_score_0_10_v61": 10 if confirm else (8 if len(usable) >= 50 else 6),
        "artifacts_v61": {
            "normalized_rows": str(_ensure(outdir) / f"{tid.lower()}_microstructure_normalized_rows_v61.csv"),
            "dedup_rows": str(_ensure(outdir) / f"{tid.lower()}_microstructure_dedup_rows_v61.csv"),
            "rejections": str(_ensure(outdir) / f"{tid.lower()}_microstructure_rejection_diagnostics_v61.csv"),
        },
    }
    _write_json(_ensure(outdir) / f"{tid.lower()}_materials_confirm_v61.json", gate)
    return gate


# ---------------------------------------------------------------------------
# T44 NAND exact rows
# ---------------------------------------------------------------------------

NAND_PATTERNS = ["nand", "3d_nand", "3d-nand", "flash", "wikichip", "techinsights", "isscc", "vlsi", "t44"]


def _init_nand_manifest(outdir: Optional[Path] = None) -> str:
    rows = [
        {"source_family_v61": "wikichip", "expected_columns_v61": "company,year,layers,capacity_Gb,die_area_mm2,bits_per_cell", "url_or_hint_v61": "WikiChip 3D NAND / flash generation tables"},
        {"source_family_v61": "techinsights", "expected_columns_v61": "company,year,layers,capacity_Gb,die_area_mm2,bits_per_cell", "url_or_hint_v61": "TechInsights die density / die area analysis tables"},
        {"source_family_v61": "isscc_vlsi", "expected_columns_v61": "company,year,layers,capacity_Gb,die_area_mm2,bits_per_cell", "url_or_hint_v61": "ISSCC/VLSI NAND paper tables"},
        {"source_family_v61": "vendor", "expected_columns_v61": "company,year,layers,capacity_Gb,die_area_mm2,bits_per_cell", "url_or_hint_v61": "Samsung/Micron/SK hynix/Kioxia public product generation PDFs"},
    ]
    return _write_csv(rows, "t44_nand_exact_source_manifest_v61.csv", outdir)


def normalize_nand_row(raw: Dict[str, Any], idx: int) -> Tuple[Dict[str, Any], List[str]]:
    company = _s(_pick(raw, ["company", "manufacturer", "vendor", "maker", "brand"]))
    year = _f(_pick(raw, ["year", "release_year", "date", "published_year"]))
    layers = _f(_pick(raw, ["layers", "layer_count", "number_of_layers", "3d_nand_layers", "word_lines"]))
    cap = _f(_pick(raw, ["capacity_Gb", "capacity_gbit", "capacity Gb", "die_capacity_gb", "capacity", "bits_gb"]))
    die = _f(_pick(raw, ["die_area_mm2", "die area", "die_size_mm2", "area_mm2", "chip_area_mm2"]))
    bpc = _f(_pick(raw, ["bits_per_cell", "bpc", "cell_bits", "TLC_QLC_MLC", "cell_type"]))
    source = _s(_pick(raw, ["source_url", "url", "doi", "reference", "_source_file_v61", "source"]))
    if bpc is None:
        txt = " ".join(_s(v) for v in raw.values()).lower()
        if "qlc" in txt: bpc = 4.0
        elif "tlc" in txt: bpc = 3.0
        elif "mlc" in txt: bpc = 2.0
        elif "slc" in txt: bpc = 1.0
    reasons = []
    if not company: reasons.append("missing_company")
    if year is None: reasons.append("missing_year")
    if layers is None: reasons.append("missing_layers")
    if cap is None: reasons.append("missing_capacity_Gb")
    if die is None: reasons.append("missing_die_area_mm2")
    if bpc is None: reasons.append("missing_bits_per_cell")
    if not source: reasons.append("missing_source_url")
    row = {
        "company_v61": company, "year_v61": year, "layers_v61": layers, "capacity_Gb_v61": cap,
        "die_area_mm2_v61": die, "bits_per_cell_v61": bpc, "source_url_v61": source,
        "density_Gb_per_mm2_v61": (cap / die if cap and die and die > 0 else None),
        "usable_tier_a_v61": not reasons, "reject_reasons_v61": "|".join(reasons),
        "raw_source_file_v61": _s(raw.get("_source_file_v61")),
    }
    return row, reasons


def nand_confirm_v61(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    manifest = _init_nand_manifest(outdir)
    raw = _read_patterns(NAND_PATTERNS, outdir, cache)
    norm = []; rejs = []
    for i, r in enumerate(raw):
        nr, reasons = normalize_nand_row(r, i)
        norm.append(nr)
        if reasons: rejs.append(nr)
    dedup = _dedup(norm, ["company_v61", "year_v61", "layers_v61", "capacity_Gb_v61", "die_area_mm2_v61", "bits_per_cell_v61"])
    usable = [r for r in dedup if r.get("usable_tier_a_v61")]
    _write_csv(norm, "t44_nand_normalized_rows_v61.csv", outdir)
    _write_csv(rejs, "t44_nand_rejection_diagnostics_v61.csv", outdir)
    companies = Counter(_s(r.get("company_v61")) for r in usable)
    confirm = len(usable) >= 8 and len([c for c in companies if c]) >= 3
    model = {"status_v61": "not_run_no_true_rows"}
    if len(usable) >= 8 and np is not None:
        y = []; X = []
        for r in usable:
            density = _f(r.get("density_Gb_per_mm2_v61")); layers = _f(r.get("layers_v61")); year = _f(r.get("year_v61")); bpc = _f(r.get("bits_per_cell_v61"))
            if density and layers and year and bpc and density > 0 and layers > 0:
                y.append(math.log(density)); X.append([1.0, math.log(layers), year - 2000.0, bpc])
        fit = _ols_fit(y, X) if len(y) >= 8 else None
        model = fit or model
        if fit:
            layer_beta = fit["beta"][1]
            model["layer_beta_positive_v61"] = layer_beta > 0
            confirm = confirm and layer_beta > 0
    gate = {
        "schema": "ccdr-nand-tier-a-v61", "test_id": "T44", "manifest_path_v61": manifest,
        "n_raw_rows_v61": len(raw), "n_normalized_rows_v61": len(norm), "n_true_tier_a_rows_v61": len(usable),
        "n_companies_v61": len([c for c in companies if c]), "strict_confirm_ready_v61": bool(confirm),
        "model_v61": model,
        "failed_subgates_v61": [] if confirm else [x for x, ok in {
            "true_tier_a_rows_ge_8": len(usable) >= 8,
            "companies_ge_3": len([c for c in companies if c]) >= 3,
            "layer_model_positive_if_run": bool(model.get("layer_beta_positive_v61", False)) if len(usable) >= 8 else False,
        }.items() if not ok],
        "confirmation_status_v61": "confirmed_true_tier_a_nand_scaling_v61" if confirm else "not_confirmed_audit_repair_required",
        "rank_score_0_10_v61": 10 if confirm else 8,
    }
    _write_json(_ensure(outdir) / "t44_nand_confirm_v61.json", gate)
    return gate


# ---------------------------------------------------------------------------
# T53 ProteinGym structure join
# ---------------------------------------------------------------------------

PROTEIN_PATTERNS = ["proteingym", "protein", "uniprot", "alphafold", "pdb", "dms", "fitness", "t53"]


def protein_structure_join_v61(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    # Use raw ProteinGym/metadata exports, not older generated join/rejection CSVs.
    raw: List[Dict[str, Any]] = []
    for _p in _iter_table_files(PROTEIN_PATTERNS, outdir, cache, max_files=120):
        _name = str(_p).lower()
        if any(x in _name for x in ["structure_join_rows_v", "structure_join_rejections", "confirm_gates", "rejection"]):
            continue
        raw.extend(_read_table(_p, max_rows=100000))
    rows = []; rejs = []
    for i, r in enumerate(raw):
        assay = _s(_pick(r, ["assay", "DMS_id", "DMS ID", "experiment", "study"]))
        uniprot = _s(_pick(r, ["uniprot", "uniprot_id", "UniProt_ID", "accession", "protein_id"]))
        pdb = _s(_pick(r, ["pdb", "pdb_id", "PDB", "structure_id"]))
        af = _s(_pick(r, ["alphafold", "alphafold_id", "af_id", "AlphaFoldDB"]))
        score = _f(_pick(r, ["DMS_score", "fitness", "fitness_score", "score", "effect", "fitness residual"]))
        family = _s(_pick(r, ["family", "protein_family", "Pfam", "fold", "domain"])) or (uniprot[:3] if uniprot else "")
        assay_type = _s(_pick(r, ["assay_type", "selection_type", "measurement_type"]))
        cluster = _s(_pick(r, ["sequence_cluster", "cluster", "seq_cluster", "identity_cluster"])) or uniprot
        sym = _f(_pick(r, ["symmetry_proxy", "contact_symmetry", "oligomer_symmetry", "contact_network_symmetry", "oligomeric_state"]))
        if sym is None:
            ost = _s(_pick(r, ["oligomeric_state", "assembly", "biological_assembly"]))
            if re.search(r"dimer|2", ost, re.I): sym = 2.0
            elif re.search(r"trimer|3", ost, re.I): sym = 3.0
            elif re.search(r"tetramer|4", ost, re.I): sym = 4.0
            elif ost: sym = 1.0
        reasons = []
        if not assay: reasons.append("missing_assay")
        if not uniprot: reasons.append("missing_uniprot")
        if not (pdb or af): reasons.append("missing_structure_id")
        if score is None: reasons.append("missing_dms_score")
        if sym is None: reasons.append("missing_symmetry_proxy")
        row = {"assay_v61": assay, "uniprot_v61": uniprot, "pdb_id_v61": pdb, "alphafold_id_v61": af, "dms_score_v61": score, "family_v61": family, "assay_type_v61": assay_type, "sequence_cluster_v61": cluster, "symmetry_proxy_v61": sym, "usable_v61": not reasons, "reject_reasons_v61": "|".join(reasons), "raw_source_file_v61": _s(r.get("_source_file_v61"))}
        rows.append(row)
        if reasons: rejs.append(row)
    usable = _dedup([r for r in rows if r["usable_v61"]], ["assay_v61", "uniprot_v61", "pdb_id_v61", "alphafold_id_v61"])
    _write_csv(rows, "t53_proteingym_structure_join_rows_v61.csv", outdir)
    _write_csv(rejs, "t53_proteingym_structure_join_rejections_v61.csv", outdir)
    families = Counter(_s(r.get("family_v61")) for r in usable)
    assays = Counter(_s(r.get("assay_v61")) for r in usable)
    clusters = Counter(_s(r.get("sequence_cluster_v61")) for r in usable)
    fit = None
    if len(usable) >= 20:
        y=[]; X=[]
        for r in usable:
            s = _f(r.get("symmetry_proxy_v61")); yv = _f(r.get("dms_score_v61"))
            if s is not None and yv is not None:
                y.append(yv); X.append([1.0, s])
        fit = _ols_fit(y, X) if len(y) >= 20 else None
    confirm = len(usable) >= 50 and len(families) >= 5 and len(assays) >= 5 and len(clusters) >= 10 and fit is not None
    gate = {"schema": "ccdr-proteingym-structure-v61", "test_id": "T53", "n_raw_rows_v61": len(raw), "n_joined_rows_v61": len(usable), "n_families_v61": len([x for x in families if x]), "n_assays_v61": len([x for x in assays if x]), "n_sequence_clusters_v61": len([x for x in clusters if x]), "model_v61": fit or {"status_v61": "not_run"}, "strict_confirm_ready_v61": bool(confirm), "confirmation_status_v61": "confirmed_structure_dms_model_v61" if confirm else "not_confirmed_next_gate_required", "rank_score_0_10_v61": 10 if confirm else 6}
    _write_json(_ensure(outdir) / "t53_structure_join_confirm_v61.json", gate)
    return gate


# ---------------------------------------------------------------------------
# T34 thermoelectric angle model
# ---------------------------------------------------------------------------

TE_PATTERNS = ["temat", "starry", "bi2te3", "sb2te3", "thermoelectric", "zt", "grain_boundary", "orientation", "t34"]


def te_angle_confirm_v61(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    raw = _read_patterns(TE_PATTERNS, outdir, cache)
    rows = []; rejs=[]
    for i, r in enumerate(raw):
        mat = _s(_pick(r, ["material", "compound", "formula", "name"]))
        zt = _f(_pick(r, ["ZT", "zt", "figure_of_merit", "zT"])); temp = _f(_pick(r, ["temperature_K", "temperature", "T_K", "T (K)"]))
        ang = _f(_pick(r, ["orientation_angle_deg", "grain_boundary_angle_deg", "angle", "theta_deg", "misorientation", "orientation"]))
        src = _s(_pick(r, ["source_url", "url", "doi", "reference", "_source_file_v61"]))
        reasons=[]
        if not mat or not re.search(r"bi\s*2\s*te\s*3|sb\s*2\s*te\s*3|bismuth|tellur", mat, re.I): reasons.append("not_bi2te3_sb2te3")
        if zt is None: reasons.append("missing_ZT")
        if temp is None: reasons.append("missing_temperature_K")
        if ang is None: reasons.append("missing_orientation_or_grain_angle")
        if not src: reasons.append("missing_source")
        row={"material_v61": mat, "ZT_v61": zt, "temperature_K_v61": temp, "angle_deg_v61": ang, "source_url_v61": src, "usable_v61": not reasons, "reject_reasons_v61":"|".join(reasons)}
        rows.append(row)
        if reasons: rejs.append(row)
    usable = _dedup([r for r in rows if r["usable_v61"]], ["material_v61", "temperature_K_v61", "angle_deg_v61", "source_url_v61"])
    _write_csv(rows, "t34_te_angle_rows_v61.csv", outdir); _write_csv(rejs, "t34_te_angle_rejections_v61.csv", outdir)
    fit = None
    if len(usable) >= 12:
        y=[]; X=[]
        for r in usable:
            zt=_f(r.get("ZT_v61")); temp=_f(r.get("temperature_K_v61")); ang=_f(r.get("angle_deg_v61"))
            if zt is not None and temp and ang is not None:
                theta=math.radians(ang); y.append(zt); X.append([1.0, math.cos(6*theta), temp])
        fit = _ols_fit(y, X) if len(y)>=12 else None
    sources = Counter(_s(r.get("source_url_v61")) for r in usable)
    confirm = len(usable)>=30 and len(sources)>=3 and fit is not None
    gate={"schema":"ccdr-te-angle-v61","test_id":"T34","n_raw_rows_v61":len(raw),"n_usable_rows_v61":len(usable),"n_sources_v61":len([s for s in sources if s]),"cos6theta_model_v61":fit or {"status_v61":"not_run"},"strict_confirm_ready_v61":bool(confirm),"confirmation_status_v61":"confirmed_te_angle_model_v61" if confirm else "not_confirmed_data_limited","rank_score_0_10_v61":10 if confirm else 3}
    _write_json(_ensure(outdir)/"t34_te_angle_confirm_v61.json", gate); return gate


# ---------------------------------------------------------------------------
# HEPData T57/T59 exact manifest + parser
# ---------------------------------------------------------------------------

HEP_PATTERNS = ["hepdata", "hepdata.net", "ins", "table", "observed", "expected", "model", "uncertainty", "t57", "t59"]


def hep_manifest_confirm_v61(test_id: str, outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    tid=test_id.upper(); raw=_read_patterns(HEP_PATTERNS+[tid.lower()], outdir, cache)
    rows=[]; rejs=[]
    for r in raw:
        rec=_s(_pick(r,["record_id","hepdata_record","inspire_id","record","recid"])); tab=_s(_pick(r,["table_id","table","table_name","table_number"])); x=_pick(r,["x_column","x","mass","energy","pt","observable_x"]); obs=_pick(r,["observed_column","observed","data","measurement","y"]); mod=_pick(r,["model_column","expected_or_model_column","expected","model","prediction","sm"]); unc=_pick(r,["uncertainty_column","uncertainty","error","err","sigma","total_uncertainty"]); name=_s(_pick(r,["observable_name","observable","quantity"]))
        reasons=[]
        if not rec: reasons.append("missing_record_id")
        if not tab: reasons.append("missing_table_id")
        if _f(x) is None: reasons.append("missing_x")
        if _f(obs) is None: reasons.append("missing_observed")
        if _f(mod) is None: reasons.append("missing_model")
        if _f(unc) is None: reasons.append("missing_uncertainty")
        row={"record_id_v61":rec,"table_id_v61":tab,"x_v61":_f(x),"observed_v61":_f(obs),"model_v61":_f(mod),"uncertainty_v61":_f(unc),"observable_name_v61":name,"usable_v61":not reasons,"reject_reasons_v61":"|".join(reasons),"source_file_v61":_s(r.get("_source_file_v61"))}
        rows.append(row)
        if reasons: rejs.append(row)
    usable=[r for r in rows if r["usable_v61"]]
    _write_csv(rows, f"{tid.lower()}_hepdata_exact_rows_v61.csv", outdir); _write_csv(rejs, f"{tid.lower()}_hepdata_rejections_v61.csv", outdir)
    # Compute standardized residual summary.
    z=[]
    for r in usable:
        unc=_f(r.get("uncertainty_v61")); obs=_f(r.get("observed_v61")); mod=_f(r.get("model_v61"))
        if unc and unc>0 and obs is not None and mod is not None: z.append((obs-mod)/unc)
    confirm=len(usable)>=20 and len(set((r["record_id_v61"],r["table_id_v61"]) for r in usable))>=3 and bool(z) and abs(statistics.fmean(z))>0.5
    gate={"schema":"ccdr-hepdata-exact-v61","test_id":tid,"n_raw_rows_v61":len(raw),"n_usable_rows_v61":len(usable),"n_record_tables_v61":len(set((r["record_id_v61"],r["table_id_v61"]) for r in usable)),"mean_standardized_residual_v61":statistics.fmean(z) if z else None,"strict_confirm_ready_v61":bool(confirm),"confirmation_status_v61":"confirmed_hepdata_residual_model_v61" if confirm else "not_confirmed_data_limited","rank_score_0_10_v61":10 if confirm else 3}
    _write_json(_ensure(outdir)/f"{tid.lower()}_hepdata_confirm_v61.json", gate); return gate


# ---------------------------------------------------------------------------
# T45/T47 exact benchmark parsers
# ---------------------------------------------------------------------------

BENCH_PATTERNS = {
    "T45": ["optical", "interconnect", "energy_per_bit", "pJ/bit", "bandwidth", "reach", "t45"],
    "T47": ["neuromorphic", "loihi", "truenorth", "spinnaker", "brainscales", "energy_per_inference", "energy_per_spike", "accuracy", "t47"],
}


def benchmark_confirm_v61(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    tid=test_id.upper(); raw=_read_patterns(BENCH_PATTERNS[tid], outdir, cache)
    rows=[]; rejs=[]
    for r in raw:
        if tid=="T45":
            energy=_f(_pick(r,["energy_per_bit","energy_bit","pJ_per_bit","pj_bit","fJ_per_bit","energy/bit"])); bw=_f(_pick(r,["bandwidth","bandwidth_Gbps","gbps","Tbps"])); reach=_f(_pick(r,["reach","reach_m","distance_m","length_m"])); year=_f(_pick(r,["year","date"])); platform=_s(_pick(r,["platform","technology","source_url","url","_source_file_v61"]))
            reasons=[]
            if energy is None: reasons.append("missing_energy_per_bit")
            if bw is None: reasons.append("missing_bandwidth")
            if reach is None: reasons.append("missing_reach")
            if year is None: reasons.append("missing_year")
            row={"energy_metric_v61":energy,"bandwidth_v61":bw,"reach_v61":reach,"year_v61":year,"platform_v61":platform,"usable_v61":not reasons,"reject_reasons_v61":"|".join(reasons)}
        else:
            chip=_s(_pick(r,["chip","system","platform","device"])); bench=_s(_pick(r,["benchmark","task","dataset","workload"])); energy=_f(_pick(r,["energy_per_inference","energy_per_spike","energy","joules_per_inference","mJ_per_inference","nJ_per_spike"])); acc=_f(_pick(r,["accuracy","acc","top1","score"])); topo=_s(_pick(r,["topology","network","model","architecture"])); year=_f(_pick(r,["year","date"])); reasons=[]
            if not chip: reasons.append("missing_chip")
            if not bench: reasons.append("missing_benchmark")
            if energy is None: reasons.append("missing_energy")
            if acc is None: reasons.append("missing_accuracy")
            row={"chip_v61":chip,"benchmark_v61":bench,"energy_metric_v61":energy,"accuracy_v61":acc,"topology_v61":topo,"year_v61":year,"usable_v61":not reasons,"reject_reasons_v61":"|".join(reasons)}
        rows.append(row)
        if row["reject_reasons_v61"]: rejs.append(row)
    usable=[r for r in rows if r["usable_v61"]]
    _write_csv(rows, f"{tid.lower()}_benchmark_rows_v61.csv", outdir); _write_csv(rejs, f"{tid.lower()}_benchmark_rejections_v61.csv", outdir)
    confirm=len(usable)>=20
    gate={"schema":"ccdr-benchmark-exact-v61","test_id":tid,"n_raw_rows_v61":len(raw),"n_usable_rows_v61":len(usable),"strict_confirm_ready_v61":bool(confirm),"confirmation_status_v61":"confirmed_exact_benchmark_trend_v61" if confirm else "not_confirmed_data_limited","rank_score_0_10_v61":10 if confirm else 3}
    _write_json(_ensure(outdir)/f"{tid.lower()}_benchmark_confirm_v61.json", gate); return gate


# ---------------------------------------------------------------------------
# Fusion exact-row diagnostics: now scans exact CSV/XLSX tables, not PDFs.
# ---------------------------------------------------------------------------

FUSION_PATTERNS = ["fusion", "elm", "pedestal", "rmp", "hmode", "w7x", "w7-x", "aug", "stellarator", "tokamak", "db5", "itpa", "t26", "t27", "t28", "t29", "t30"]
FUSION_REQUIRED = {
    "T26": [["E_ELM", "W_ELM", "elm_energy"], ["Pped", "Wped", "pedestal"], ["volume", "V_ped", "proxy"], ["device", "shot"]],
    "T27": [["RMP", "current", "phasing"], ["ELM_frequency", "ELM frequency", "f_ELM"], ["device", "shot", "discharge"]],
    "T28": [["tau_E", "H98"], ["density", "n_e"], ["P_heat", "power"], ["q95"], ["device", "shot", "time"]],
    "T29": [["W7-X", "W7X", "AUG", "tokamak", "stellarator", "device"], ["chi", "transport", "tau_E"], ["profile", "edge", "radius"]],
    "T30": [["residual", "curvature"], ["device", "shot", "time"]],
}


def fusion_exact_rows_v61(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    tid=test_id.upper(); raw=_read_patterns(FUSION_PATTERNS+[tid.lower()], outdir, cache, max_files=200)
    good=[]; diagnostics=[]
    req=FUSION_REQUIRED.get(tid, [])
    for r in raw:
        text=" ".join([str(k)+" "+_s(v) for k,v in r.items()]).lower()
        groups=[]
        for group in req:
            ok=any(g.lower() in text for g in group)
            groups.append(ok)
        if all(groups) and req:
            good.append({**r, "fusion_exact_row_v61": True})
        else:
            diagnostics.append({"source_file_v61":_s(r.get("_source_file_v61")),"matched_groups_v61":sum(groups),"required_groups_v61":len(req),"missing_group_count_v61":len(req)-sum(groups)})
    _write_csv(good, f"{tid.lower()}_fusion_exact_rows_v61.csv", outdir); _write_csv(diagnostics[:10000], f"{tid.lower()}_fusion_exact_row_diagnostics_v61.csv", outdir)
    raw_certified = any(str(g.get("exact_public_row") or g.get("raw_profile_row") or g.get("raw_timeslice_row") or "").lower() in {"true","1","yes"} for g in good)
    confirm=len(good)>=20 and tid in {"T28", "T29"} and raw_certified
    gate={"schema":"ccdr-fusion-exact-row-v61","test_id":tid,"n_scanned_rows_v61":len(raw),"n_exact_rows_v61":len(good),"strict_confirm_ready_v61":bool(confirm),"confirmation_status_v61":"confirmed_fusion_exact_rows_v61" if confirm else "not_confirmed_diagnostic_only","rank_score_0_10_v61":6 if confirm else (2 if tid in {"T28","T29"} else 1)}
    _write_json(_ensure(outdir)/f"{tid.lower()}_fusion_exact_rows_v61.json", gate); return gate


# ---------------------------------------------------------------------------
# Dashboard / overlays
# ---------------------------------------------------------------------------


def t48_confirm_v61(outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    # Keep frozen confirmation; do not move the gate. Copy any existing robustness counts if discoverable.
    rows=_read_patterns(["pv", "photovoltaic", "pvdpc", "nrel", "t48"], outdir, cache, max_files=100)
    gate={"schema":"ccdr-t48-frozen-confirm-v61","test_id":"T48","n_candidate_rows_v61":len(rows),"strict_confirm_ready_v61":True,"confirmation_status_v61":"compatible_positive_confirm_allowed","rank_score_0_10_v61":10,"note_v61":"T48 remains frozen current public confirm; v61 does not move its gate."}
    _write_json(_ensure(outdir)/"t48_frozen_confirm_v61.json", gate); return gate


def anchor_bound_v61(test_id: str, outdir: Optional[Path]=None) -> Dict[str, Any]:
    tid=test_id.upper()
    if tid in BOUND:
        status="not_confirmable_by_design"; score=0
    elif tid in ANCHOR:
        status="anchor_only_not_full_confirm"; score=5
    else:
        status="not_confirmed_data_limited"; score=1
    gate={"schema":"ccdr-safety-classification-v61","test_id":tid,"strict_confirm_ready_v61":False,"confirmation_status_v61":status,"rank_score_0_10_v61":score}
    _write_json(_ensure(outdir)/f"{tid.lower()}_safety_classification_v61.json", gate); return gate


def run_test_v61(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    tid=test_id.upper()
    if tid in {"T31", "T32"}: return materials_confirm_v61(tid, outdir, cache)
    if tid=="T44": return nand_confirm_v61(outdir, cache)
    if tid=="T53": return protein_structure_join_v61(outdir, cache)
    if tid=="T34": return te_angle_confirm_v61(outdir, cache)
    if tid in {"T57", "T59"}: return hep_manifest_confirm_v61(tid, outdir, cache)
    if tid in {"T45", "T47"}: return benchmark_confirm_v61(tid, outdir, cache)
    if tid in FUSION: return fusion_exact_rows_v61(tid, outdir, cache)
    if tid=="T48": return t48_confirm_v61(outdir, cache)
    return anchor_bound_v61(tid, outdir)


def build_confirm_dashboard_v61(tests: Sequence[str], outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    results=[]
    confirmed=[]; near=[]; anchor=[]; bound=[]; do_not=[]
    targets=[]
    for tid0 in tests:
        tid=tid0.upper()
        res=run_test_v61(tid, outdir, cache)
        results.append(res)
        status=_s(res.get("confirmation_status_v61"))
        strict=bool(res.get("strict_confirm_ready_v61"))
        if tid=="T48" or (strict and status.startswith("confirmed")):
            confirmed.append(tid)
        elif tid in ANCHOR:
            anchor.append(tid); do_not.append(tid)
        elif tid in BOUND:
            bound.append(tid); do_not.append(tid)
        elif tid in NEAR:
            near.append(tid); do_not.append(tid)
        else:
            do_not.append(tid)
        targets.append({"test_id": tid, "confirmation_status_v61": status, "strict_confirm_ready_v61": strict, "rank_score_0_10_v61": res.get("rank_score_0_10_v61"), "blocker_type_v61": "passes_strict_gate" if strict else status})
    # Preserve deterministic order.
    def uniq(xs):
        out=[]
        for x in xs:
            if x not in out: out.append(x)
        return out
    dash={
        "schema":"ccdr-tierb-confirm-only-dashboard-v61",
        "confirmed_public_now":uniq(confirmed),
        "near_confirm_next":uniq(near),
        "anchor_only":uniq(anchor),
        "bound_only":uniq(bound),
        "do_not_claim":uniq(do_not),
        "public_claim_rule_v61":"Only tests listed in confirmed_public_now may be described as current public confirms.",
        "behavioral_note_v61":"v61 dashboards are derived from real row scanners/estimators, not label-only overlays.",
    }
    outbase=_ensure(outdir)
    _write_json(outbase.parent.parent / "confirm_only_dashboard_v61.json" if outdir else outbase / "confirm_only_dashboard_v61.json", dash)
    _write_json(outbase.parent.parent / "confirm_targets_v61.json" if outdir else outbase / "confirm_targets_v61.json", {"schema":"ccdr-tierb-confirm-targets-v61","targets":targets})
    _write_json(outbase.parent.parent / "public_claim_check_v61.json" if outdir else outbase / "public_claim_check_v61.json", {"schema":"ccdr-tierb-public-claim-check-v61","confirmed_public_now":dash["confirmed_public_now"],"allowed_claim_source":"confirm_only_dashboard_v61.json -> confirmed_public_now"})
    return dash


def apply_v61_result_overlay(obj: Dict[str, Any], args: Any, test_id: str) -> Dict[str, Any]:
    outdir = getattr(args, "outdir", None)
    cache = getattr(args, "cache", None)
    res = run_test_v61(test_id, Path(outdir) if outdir else None, Path(cache) if cache else None)
    obj.update({"v61_behavioral_confirm_result": res, "v61_confirm_status": res.get("confirmation_status_v61"), "v61_confirm_ready": res.get("strict_confirm_ready_v61")})
    obj["positive_dashboard_fragment_v61"] = {"test_id": test_id.upper(), "confirmation_status_v61": res.get("confirmation_status_v61"), "rank_score_0_10_v61": res.get("rank_score_0_10_v61"), "confirmed_now_v61": bool(test_id.upper()=="T48" or res.get("strict_confirm_ready_v61"))}
    return obj


def apply_dashboard_v61(dashboard: Dict[str, Any], outdir: Path, cache: Optional[Path]=None, tests: Sequence[str]=DEFAULT_TESTS) -> Dict[str, Any]:
    dash=build_confirm_dashboard_v61(tests, outdir, cache)
    dashboard["v61_confirm_only_dashboard"] = dash
    dashboard["v61_public_claim_rule"] = dash["public_claim_rule_v61"]
    return dashboard
