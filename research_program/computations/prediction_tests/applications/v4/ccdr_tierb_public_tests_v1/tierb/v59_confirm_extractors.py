#!/usr/bin/env python3
"""v59 Tier-B confirm-extraction patch.

This layer implements the next 10 confirm-focused improvements over v58:
1) strict T31/T32 measured microstructure parser with source/sample de-duplication,
2) row-by-row rejection CSVs,
3) grouped bootstrap + material/source/temp-bin jackknife outputs,
4) curated T44 exact NAND source manifest,
5) T44 parser hard-refuses rows missing die area or bits/cell,
6) T53 ProteinGym->structure join gate with model/FDR placeholders,
7) T34 exact thermoelectric export parser/gate,
8) T57/T59 exact HEPData manifest loader,
9) T45/T47 exact benchmark table loaders,
10) fusion T26-T30 kept diagnostic until exact row tables are available.

It is conservative by design: only T48 remains a public confirm unless this
module explicitly places another test in confirm_only_dashboard_v59.confirmed_public_now.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from . import v58_confirm_focus as v58

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GEN_DIR = DATA_DIR / "generated"


def _ensure() -> Path:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    return GEN_DIR


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


def _write_csv(rows: Sequence[Dict[str, Any]], filename: str) -> str:
    out = _ensure() / filename
    keys: List[str] = []
    for r in rows:
        for k in (r or {}).keys():
            if k not in keys:
                keys.append(k)
    if not keys:
        keys = ["empty_v59"]
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
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
    if not m:
        return None
    try:
        vv = float(m.group(0))
        return vv if math.isfinite(vv) else None
    except Exception:
        return None


def _pick(row: Dict[str, Any], names: Sequence[str]) -> Any:
    lower = {str(k).lower(): k for k in row.keys()}
    for name in names:
        if name in row:
            return row[name]
        lk = name.lower()
        if lk in lower:
            return row[lower[lk]]
    for name in names:
        key = name.lower().replace("_", "")
        for k in row.keys():
            if key and key in str(k).lower().replace("_", ""):
                return row[k]
    return None


def _read_patterns(patterns: Sequence[str]) -> List[Dict[str, Any]]:
    try:
        rows = v58._read_patterns(patterns)  # type: ignore[attr-defined]
        if rows:
            return rows
    except Exception:
        pass
    if pd is None:
        return []
    out: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    roots = [GEN_DIR, DATA_DIR, ROOT]
    max_files = 80
    max_rows_per_file = 50000
    max_file_bytes = 50000000
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            for p in root.rglob(pat):
                if len(seen) >= max_files:
                    return out
                if p in seen or not p.is_file():
                    continue
                try:
                    if p.stat().st_size > max_file_bytes:
                        continue
                except Exception:
                    pass
                seen.add(p)
                try:
                    if p.suffix.lower() == ".csv":
                        df = pd.read_csv(p, nrows=max_rows_per_file, dtype=str)
                    elif p.suffix.lower() == ".tsv":
                        df = pd.read_csv(p, sep="\t", nrows=max_rows_per_file, dtype=str)
                    elif p.suffix.lower() == ".json":
                        obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                        vals = obj if isinstance(obj, list) else (obj.get("rows") or obj.get("data") or obj.get("targets") or []) if isinstance(obj, dict) else []
                        df = pd.DataFrame(vals[:max_rows_per_file])
                    elif p.suffix.lower() == ".jsonl":
                        vals = [json.loads(line) for line in p.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
                        df = pd.DataFrame(vals[:max_rows_per_file])
                    else:
                        continue
                    for _, rr in df.iterrows():
                        d = dict(rr)
                        d["_source_file_v59"] = str(p)
                        out.append(d)
                except Exception:
                    continue
    return out


def _hash_row(row: Dict[str, Any], keys: Sequence[str]) -> str:
    txt = "|".join(_s(row.get(k)) for k in keys)
    return hashlib.sha1(txt.encode("utf-8", errors="ignore")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# T31/T32 measured microstructure parser/model gate
# ---------------------------------------------------------------------------

def _material_family(material: str) -> str:
    m = material.lower()
    if any(x in m for x in ["silicon", "si", "silica", "sio"]): return "silicon_or_silica"
    if any(x in m for x in ["diamond", "carbon", "graphene", "graphite"]): return "carbon"
    if any(x in m for x in ["alumina", "aluminum", "aluminium", "al2o3"]): return "alumina"
    if any(x in m for x in ["titania", "tio2", "oxide"]): return "oxide"
    if any(x in m for x in ["bismuth", "telluride", "bi2te3", "sb2te3"]): return "thermoelectric_telluride"
    if any(x in m for x in ["poly", "epoxy", "pmma"]): return "polymer"
    return re.sub(r"[^a-z0-9]+", "_", m)[:40] or "unknown"


def _normalize_material_row(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    material = _s(_pick(raw, ["material", "material_v57", "compound", "sample_material", "name"]))
    source = _s(_pick(raw, ["source_url", "source_reference", "source_reference_v57", "url", "doi", "reference", "_source_file_v59", "_source_file_v58", "_source_file_v57"]))
    sample = _s(_pick(raw, ["sample_id", "sample_id_v57", "specimen", "sample", "id"]))
    temp = _f(_pick(raw, ["temperature_K", "temperature_K_v57", "T_K", "temperature", "temp_K"]))
    kappa = _f(_pick(raw, ["kappa_W_mK", "kappa_W_mK_v57", "thermal_conductivity", "k_W_mK", "lambda_W_mK"]))
    grain = _f(_pick(raw, ["grain_size_nm_or_um", "grain_size_nm", "grain_size_nm_v57", "grain_nm", "grain_size", "crystallite_size_nm"]))
    boundary = _f(_pick(raw, ["boundary_density_proxy", "boundary_density_1_per_nm", "boundary_density_1_per_nm_v57", "boundary", "porosity", "interface_density"]))
    method = _s(_pick(raw, ["microstructure_method", "measurement_method", "measurement_method_v57", "method", "evidence_method"]))
    nano = _s(_pick(raw, ["nanocrystalline_yes_no", "nanocrystalline", "nano", "is_nanocrystalline"]))
    measured_flag = _s(_pick(raw, ["measured_microstructure", "measured_microstructure_v57", "SEM", "TEM", "XRD"]))
    txt = " ".join(_s(v) for v in raw.values())
    if not method:
        if re.search(r"\bSEM\b", txt, re.I): method = "SEM"
        elif re.search(r"\bTEM\b", txt, re.I): method = "TEM"
        elif re.search(r"\bXRD\b|x[- ]?ray", txt, re.I): method = "XRD"
    if grain is not None and "um" in " ".join(str(k).lower()+str(v).lower() for k,v in raw.items() if "grain" in str(k).lower()):
        # If field name/value says microns and value is not huge, convert to nm.
        if grain < 1000:
            grain = grain * 1000.0
    if boundary is None and grain and grain > 0:
        boundary = 1.0 / grain
    reasons: List[str] = []
    if not material: reasons.append("missing_material")
    if not source: reasons.append("missing_source_url_or_reference")
    if temp is None or temp <= 0: reasons.append("missing_or_bad_temperature_K")
    if kappa is None or kappa <= 0: reasons.append("missing_or_bad_kappa_W_mK")
    if grain is None or grain <= 0: reasons.append("missing_or_bad_grain_size")
    evidence = bool(re.search(r"\b(SEM|TEM|XRD|diffraction|microscopy|grain|nanocrystalline|crystallite)\b", method + " " + measured_flag + " " + txt, re.I))
    if not evidence: reasons.append("missing_measured_microstructure_evidence")
    temp_bin = "unknown"
    if temp is not None:
        if temp < 20: temp_bin = "cryogenic_lt20K"
        elif temp < 100: temp_bin = "low_20_100K"
        elif temp < 250: temp_bin = "mid_100_250K"
        else: temp_bin = "room_or_high_ge250K"
    fam = _s(_pick(raw, ["material_family", "material_family_v57", "family"])) or _material_family(material)
    out = {
        "raw_index_v59": idx,
        "source_url_v59": source,
        "sample_id_v59": sample or f"sample_{idx}",
        "material_v59": material,
        "material_family_v59": fam,
        "temperature_K_v59": temp,
        "temperature_bin_v59": temp_bin,
        "kappa_W_mK_v59": kappa,
        "grain_size_nm_v59": grain,
        "boundary_density_proxy_v59": boundary,
        "microstructure_method_v59": method,
        "nanocrystalline_yes_no_v59": nano,
        "usable_v59": not reasons,
        "reject_reasons_v59": "|".join(reasons),
        "row_hash_v59": "",
    }
    out["row_hash_v59"] = _hash_row(out, ["source_url_v59", "sample_id_v59", "material_v59", "temperature_K_v59", "kappa_W_mK_v59", "grain_size_nm_v59"])
    return out


def _fit_linear(rows: List[Dict[str, Any]], with_micro: bool) -> Dict[str, Any]:
    if np is None or len(rows) < (5 if with_micro else 3):
        return {"available": False, "reason": "numpy_missing_or_too_few_rows"}
    X: List[List[float]] = []
    y: List[float] = []
    for r in rows:
        t = _f(r.get("temperature_K_v59")); k = _f(r.get("kappa_W_mK_v59")); g = _f(r.get("grain_size_nm_v59")); b = _f(r.get("boundary_density_proxy_v59"))
        if not t or not k or t <= 0 or k <= 0:
            continue
        row = [1.0, math.log(max(t, 1e-9))]
        if with_micro:
            if not g or g <= 0: continue
            row += [math.log(max(g, 1e-9)), float(b or 0.0)]
        X.append(row); y.append(math.log(max(k, 1e-12)))
    n = len(y); p = len(X[0]) if X else 0
    if n <= p or n < 3:
        return {"available": False, "reason": "too_few_valid_fit_rows", "n": n, "p": p}
    A = np.asarray(X, dtype=float); yy = np.asarray(y, dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
        resid = yy - A.dot(beta)
        rss = float(np.sum(resid ** 2))
        rss = max(rss, 1e-12)
        aic = float(n * math.log(rss / n) + 2 * p)
        bic = float(n * math.log(rss / n) + p * math.log(n))
        return {"available": True, "n": n, "p": p, "beta": [float(x) for x in beta], "rss": rss, "aic": aic, "bic": bic}
    except Exception as e:
        return {"available": False, "reason": f"fit_error:{type(e).__name__}:{e}"}


def _group_jackknife(rows: List[Dict[str, Any]], group_key: str) -> Dict[str, Any]:
    vals = sorted({_s(r.get(group_key)) for r in rows if _s(r.get(group_key))})
    out: List[Dict[str, Any]] = []
    pass_count = 0
    for val in vals[:50]:
        sub = [r for r in rows if _s(r.get(group_key)) != val]
        base = _fit_linear(sub, False); micro = _fit_linear(sub, True)
        ok = bool(base.get("available") and micro.get("available") and (micro.get("aic", 1e99) < base.get("aic", -1e99)) and (micro.get("bic", 1e99) < base.get("bic", -1e99)))
        pass_count += int(ok)
        out.append({"jackknife_v59": group_key, "left_out_v59": val, "n_rows_v59": len(sub), "pass_v59": ok, "delta_aic_micro_minus_temp_v59": (micro.get("aic") if micro.get("available") else None) - (base.get("aic") if base.get("available") else 0) if base.get("available") and micro.get("available") else None, "delta_bic_micro_minus_temp_v59": (micro.get("bic") if micro.get("available") else None) - (base.get("bic") if base.get("available") else 0) if base.get("available") and micro.get("available") else None})
    return {"groups": len(vals), "pass_fraction": pass_count / len(vals) if vals else 0.0, "rows": out}


def _bootstrap_materials(rows: List[Dict[str, Any]], n_boot: int = 200) -> Dict[str, Any]:
    if np is None or len(rows) < 10:
        return {"available": False, "reason": "numpy_missing_or_too_few_rows"}
    sources = sorted({_s(r.get("source_url_v59")) for r in rows if _s(r.get("source_url_v59"))})
    if len(sources) < 3:
        return {"available": False, "reason": "too_few_source_groups"}
    rng = np.random.default_rng(5901)
    signs: List[int] = []
    wins = 0
    for _ in range(n_boot):
        chosen = list(rng.choice(sources, size=len(sources), replace=True))
        sub: List[Dict[str, Any]] = []
        for src in chosen:
            ss = [r for r in rows if _s(r.get("source_url_v59")) == src]
            sub.extend(ss)
        base = _fit_linear(sub, False); micro = _fit_linear(sub, True)
        if base.get("available") and micro.get("available"):
            beta = micro.get("beta") or []
            grain_beta = float(beta[2]) if len(beta) > 2 else 0.0
            signs.append(1 if grain_beta > 0 else -1)
            if micro.get("aic", 1e99) < base.get("aic", -1e99) and micro.get("bic", 1e99) < base.get("bic", -1e99):
                wins += 1
    pos = sum(1 for x in signs if x > 0)
    return {"available": bool(signs), "n_boot": len(signs), "grain_sign_positive_fraction": pos / len(signs) if signs else None, "model_win_fraction": wins / len(signs) if signs else None}


def materials_confirm_v59(test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    base = v58.materials_confirm_v58(tid)
    raw = _read_patterns([
        f"{tid.lower()}*microstructure*dedup*_v*.csv", f"{tid.lower()}*microstructure*normalized*_v*.csv", f"{tid.lower()}*materials*rows*.csv",
        "*measured_microstructure*.csv", "*grain_size_known*.csv", "*microstructure_manifest*.csv",
    ])
    norm = [_normalize_material_row(r, i) for i, r in enumerate(raw)]
    # Hard de-duplication by source/sample/material/temperature, keeping first usable row preferentially.
    dedup_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for r in norm:
        key = (_s(r.get("source_url_v59")), _s(r.get("sample_id_v59")), _s(r.get("material_v59")), _s(r.get("temperature_K_v59")))
        if key not in dedup_map or (not dedup_map[key].get("usable_v59") and r.get("usable_v59")):
            dedup_map[key] = r
    dedup = list(dedup_map.values())
    usable = [r for r in dedup if r.get("usable_v59")]
    rejects = [r for r in norm if not r.get("usable_v59")]
    temp_fit = _fit_linear(usable, False)
    micro_fit = _fit_linear(usable, True)
    delta_aic = delta_bic = None
    if temp_fit.get("available") and micro_fit.get("available"):
        delta_aic = float(micro_fit["aic"] - temp_fit["aic"])
        delta_bic = float(micro_fit["bic"] - temp_fit["bic"])
    source_jk = _group_jackknife(usable, "source_url_v59")
    material_jk = _group_jackknife(usable, "material_family_v59")
    temp_jk = _group_jackknife(usable, "temperature_bin_v59")
    boot = _bootstrap_materials(usable)
    sources = {_s(r.get("source_url_v59")) for r in usable if _s(r.get("source_url_v59"))}
    fams = {_s(r.get("material_family_v59")) for r in usable if _s(r.get("material_family_v59"))}
    bins = {_s(r.get("temperature_bin_v59")) for r in usable if _s(r.get("temperature_bin_v59")) and _s(r.get("temperature_bin_v59")) != "unknown"}
    failed: List[str] = []
    if len(usable) < 20: failed.append(">=20_dedup_measured_microstructure_rows")
    if len(sources) < 5: failed.append(">=5_independent_sources")
    if len(fams) < 5: failed.append(">=5_material_families")
    if len(bins) < 3: failed.append(">=3_temperature_bins")
    if delta_aic is None or delta_bic is None or not (delta_aic < 0 and delta_bic < 0): failed.append("microstructure_model_beats_temperature_only_AIC_BIC")
    if boot.get("available") and (boot.get("grain_sign_positive_fraction") or 0) < 0.80: failed.append("bootstrap_grain_sign_fraction_ge_0p80")
    elif not boot.get("available"): failed.append("bootstrap_available")
    if source_jk["groups"] >= 5 and source_jk["pass_fraction"] < 0.80: failed.append("source_jackknife_pass_fraction_ge_0p80")
    if material_jk["groups"] >= 5 and material_jk["pass_fraction"] < 0.80: failed.append("material_jackknife_pass_fraction_ge_0p80")
    if temp_jk["groups"] >= 3 and temp_jk["pass_fraction"] < 0.80: failed.append("temp_bin_jackknife_pass_fraction_ge_0p80")
    strict = not failed
    score = 10 if strict else (9 if len(usable) >= 20 and len(sources) >= 5 else 7 if len(usable) >= 10 else 4)
    gate = {
        "schema": "ccdr-materials-confirm-gates-v59", "test_id": tid,
        "n_raw_rows_v59": len(raw), "n_normalized_rows_v59": len(norm), "n_dedup_rows_v59": len(dedup), "n_usable_rows_v59": len(usable),
        "n_sources_v59": len(sources), "n_material_families_v59": len(fams), "n_temperature_bins_v59": len(bins),
        "temperature_only_fit_v59": temp_fit, "microstructure_fit_v59": micro_fit, "delta_aic_micro_minus_temp_v59": delta_aic, "delta_bic_micro_minus_temp_v59": delta_bic,
        "bootstrap_v59": boot, "source_jackknife_pass_fraction_v59": source_jk["pass_fraction"], "material_jackknife_pass_fraction_v59": material_jk["pass_fraction"], "temperature_bin_jackknife_pass_fraction_v59": temp_jk["pass_fraction"],
        "strict_confirm_ready_v59": strict, "failed_subgates_v59": failed,
    }
    _write_csv(norm, f"{tid.lower()}_microstructure_normalized_rows_v59.csv")
    _write_csv(dedup, f"{tid.lower()}_microstructure_dedup_rows_v59.csv")
    _write_csv(rejects, f"{tid.lower()}_microstructure_rejection_diagnostics_v59.csv")
    jk_rows = source_jk["rows"] + material_jk["rows"] + temp_jk["rows"]
    _write_csv(jk_rows, f"{tid.lower()}_microstructure_grouped_jackknife_v59.csv")
    _write_csv([gate], f"{tid.lower()}_microstructure_confirm_gates_v59.csv")
    _write_json(_ensure() / f"{tid.lower()}_microstructure_confirm_gates_v59.json", gate)
    return {"schema": "ccdr-materials-confirm-v59", "test_id": tid, "status": "strict_materials_confirm_ready_v59" if strict else "materials_confirm_gates_pending_v59", "strict_confirm_ready_v59": strict, "score_0_10_v59": score, "failed_subgates_v59": failed, "gate": gate, "base_v58": base, "artifacts": {"normalized": f"data/generated/{tid.lower()}_microstructure_normalized_rows_v59.csv", "dedup": f"data/generated/{tid.lower()}_microstructure_dedup_rows_v59.csv", "rejections": f"data/generated/{tid.lower()}_microstructure_rejection_diagnostics_v59.csv", "jackknife": f"data/generated/{tid.lower()}_microstructure_grouped_jackknife_v59.csv", "gates": f"data/generated/{tid.lower()}_microstructure_confirm_gates_v59.json"}}


# ---------------------------------------------------------------------------
# Exact-source gates for T44/T53/T34/T57/T59/T45/T47/fusion
# ---------------------------------------------------------------------------

def t44_nand_exact_v59() -> Dict[str, Any]:
    base = v58.t44_nand_tier_a_v58()
    manifest = [
        {"source_family_v59": "WikiChip", "expected_columns_v59": "company|year|layers|capacity_Gb|die_area_mm2|bits_per_cell|source_url", "role_v59": "open_web_reference_audit"},
        {"source_family_v59": "TechInsights", "expected_columns_v59": "company|year|layers|capacity_Gb|die_area_mm2|bits_per_cell|source_url", "role_v59": "die_photo_or_teardown_exact_area"},
        {"source_family_v59": "ISSCC/IEDM vendor paper", "expected_columns_v59": "company|year|layers|capacity_Gb|die_area_mm2|bits_per_cell|source_url", "role_v59": "official_benchmark_or_paper"},
    ]
    raw = _read_patterns(["t44*nand*rows*.csv", "*nand_exact*.csv", "*nand*tier*a*.csv", "*electronics*spec*.csv"])
    rows: List[Dict[str, Any]] = []
    rejs: List[Dict[str, Any]] = []
    for i, r in enumerate(raw):
        company = _s(_pick(r, ["company", "manufacturer", "vendor"]))
        year = _f(_pick(r, ["year", "date", "publication_year"]))
        layers = _f(_pick(r, ["layers", "n_layers", "layer_count"]))
        cap = _f(_pick(r, ["capacity_Gb", "capacity_gb", "capacity", "Gb", "bits_Gb"]))
        area = _f(_pick(r, ["die_area_mm2", "die_area", "area_mm2"]))
        bpc = _f(_pick(r, ["bits_per_cell", "bpc", "cell_bits"]))
        url = _s(_pick(r, ["source_url", "url", "doi", "reference", "_source_file_v59", "_source_file_v58"]))
        reasons = []
        if not company: reasons.append("missing_company")
        if year is None: reasons.append("missing_year")
        if layers is None: reasons.append("missing_layers")
        if cap is None: reasons.append("missing_capacity_Gb")
        if area is None: reasons.append("missing_die_area_mm2")
        if bpc is None: reasons.append("missing_bits_per_cell")
        if not url: reasons.append("missing_source_url")
        row = {"raw_index_v59": i, "company_v59": company, "year_v59": year, "layers_v59": layers, "capacity_Gb_v59": cap, "die_area_mm2_v59": area, "bits_per_cell_v59": bpc, "source_url_v59": url, "usable_tier_a_v59": not reasons, "reject_reasons_v59": "|".join(reasons), "derived_die_area_policy_v59": "if_derived_then_audit_only_not_confirm"}
        rows.append(row)
        if reasons: rejs.append(row)
    usable = [r for r in rows if r["usable_tier_a_v59"]]
    companies = {_s(r.get("company_v59")) for r in usable if _s(r.get("company_v59"))}
    failed = []
    if len(usable) < 8: failed.append(">=8_true_tier_a_rows")
    if len(companies) < 3: failed.append(">=3_companies")
    failed.append("manufacturer_year_jackknife_model_required")
    gate = {"schema": "ccdr-t44-nand-exact-v59", "test_id": "T44", "n_raw_rows_v59": len(raw), "usable_tier_a_rows_v59": len(usable), "n_companies_v59": len(companies), "strict_confirm_ready_v59": False, "failed_subgates_v59": failed}
    _write_csv(manifest, "t44_nand_exact_source_manifest_v59.csv")
    _write_csv(rows, "t44_nand_exact_rows_v59.csv")
    _write_csv(rejs, "t44_nand_exact_rejection_diagnostics_v59.csv")
    _write_json(_ensure() / "t44_nand_exact_gate_v59.json", gate)
    return {"schema": "ccdr-t44-nand-exact-v59", "status": "t44_true_tier_a_audit_repair_required_v59", "strict_confirm_ready_v59": False, "score_0_10_v59": 8 if len(usable) else 5, "gate": gate, "base_v58": base}


def t53_structure_join_v59() -> Dict[str, Any]:
    base = v58.t53_structure_join_v58()
    raw = _read_patterns(["t53*proteingym*structure*rows*.csv", "t53*proteingym*enriched*.csv", "*ProteinGym*.csv", "*alphafold*.csv", "*uniprot*.csv"])
    rows: List[Dict[str, Any]] = []
    rejs: List[Dict[str, Any]] = []
    for i, r in enumerate(raw):
        assay = _s(_pick(r, ["assay", "assay_name", "ProteinGym_assay", "DMS_id"]))
        uniprot = _s(_pick(r, ["uniprot", "UniProt", "uniprot_accession", "accession"]))
        pdb = _s(_pick(r, ["pdb_id", "PDB", "alphafold_id", "AlphaFold", "structure_id"]))
        family = _s(_pick(r, ["family", "protein_family", "gene_family"]))
        outcome = _f(_pick(r, ["DMS_outcome", "fitness", "score", "effect", "DMS_score"]))
        sym = _f(_pick(r, ["symmetry_proxy", "contact_network_proxy", "oligomeric_state_numeric", "contacts", "symmetry_score"]))
        cluster = _s(_pick(r, ["sequence_cluster", "cluster", "sequence_identity_cluster"]))
        reasons = []
        if not assay: reasons.append("missing_ProteinGym_assay")
        if not uniprot: reasons.append("missing_UniProt")
        if not pdb: reasons.append("missing_PDB_or_AlphaFold")
        if not family: reasons.append("missing_family")
        if outcome is None: reasons.append("missing_DMS_outcome")
        if sym is None: reasons.append("missing_symmetry_or_contact_proxy")
        if not cluster: reasons.append("missing_sequence_cluster")
        row = {"raw_index_v59": i, "assay_v59": assay, "uniprot_v59": uniprot, "structure_id_v59": pdb, "family_v59": family, "DMS_outcome_v59": outcome, "symmetry_proxy_v59": sym, "sequence_cluster_v59": cluster, "usable_join_row_v59": not reasons, "reject_reasons_v59": "|".join(reasons)}
        rows.append(row); rejs.extend([row] if reasons else [])
    usable = [r for r in rows if r["usable_join_row_v59"]]
    fams = {_s(r.get("family_v59")) for r in usable if _s(r.get("family_v59"))}
    assays = {_s(r.get("assay_v59")) for r in usable if _s(r.get("assay_v59"))}
    clusters = {_s(r.get("sequence_cluster_v59")) for r in usable if _s(r.get("sequence_cluster_v59"))}
    failed = []
    if len(usable) < 50: failed.append(">=50_ProteinGym_structure_join_rows")
    if len(fams) < 5: failed.append(">=5_families")
    if len(assays) < 2: failed.append(">=2_assays")
    if len(clusters) < 10: failed.append(">=10_sequence_clusters")
    failed += ["family_assay_sequence_jackknife", "BH_FDR_or_bootstrap_significance"]
    gate = {"schema": "ccdr-t53-structure-join-model-v59", "test_id": "T53", "usable_join_rows_v59": len(usable), "n_families_v59": len(fams), "n_assays_v59": len(assays), "n_sequence_clusters_v59": len(clusters), "strict_confirm_ready_v59": False, "failed_subgates_v59": failed}
    _write_csv(rows, "t53_proteingym_structure_join_rows_v59.csv")
    _write_csv(rejs, "t53_proteingym_structure_join_rejections_v59.csv")
    _write_json(_ensure() / "t53_proteingym_structure_model_gate_v59.json", gate)
    return {"schema": "ccdr-t53-structure-join-v59", "status": "t53_structure_join_model_pending_v59", "strict_confirm_ready_v59": False, "score_0_10_v59": 8 if len(usable) >= 50 else 6, "gate": gate, "base_v58": base}


def passthrough_exact_gate_v59(test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    if tid == "T34":
        base = v58.t34_te_exact_rows_v58(); required = ["material", "ZT", "temperature_K", "orientation_angle_deg", "grain_boundary_angle_deg", "source_url"]; fn = "t34_exact_te_source_contract_v59.csv"; score = int(base.get("score_0_10_v58") or 3); blocker = "exact_te_rows_missing_or_model_pending"
    elif tid in {"T57", "T59"}:
        base = v58.hepdata_manifest_gate_v58(tid); required = ["record_id", "table_id", "x_column", "observed_column", "model_column", "uncertainty_column", "observable_name"]; fn = f"{tid.lower()}_hepdata_manifest_contract_v59.csv"; score = int(base.get("score_0_10_v58") or 3); blocker = "exact_HEPData_manifest_and_model_required"
    elif tid in {"T45", "T47"}:
        base = v58.benchmark_gate_v58(tid); required = ["source_url", "year", "benchmark", "energy", "performance", "accuracy_or_reach"] ; fn = f"{tid.lower()}_benchmark_contract_v59.csv"; score = int(base.get("score_0_10_v58") or 3); blocker = "exact_benchmark_table_required"
    elif tid in {"T26", "T27", "T28", "T29", "T30"}:
        base = v58.fusion_contract_v58(tid); required = ["exact_physical_row_table", "named_columns", "device_or_source", "shot_or_timeslice_if_applicable", "source_url"]; fn = f"{tid.lower()}_fusion_diagnostic_contract_v59.csv"; score = int(base.get("score_0_10_v58") or 1); blocker = "raw_fusion_row_table_required_for_confirm"
    else:
        base = {}; required = []; fn = f"{tid.lower()}_contract_v59.csv"; score = 1; blocker = "data_limited"
    contract = {"schema": "ccdr-exact-source-contract-v59", "test_id": tid, "required_columns_v59": "|".join(required), "confirm_policy_v59": "diagnostic_only_until_exact_rows_pass" if tid in {"T26","T27","T28","T29","T30"} else "exact_rows_required", "strict_confirm_ready_v59": False, "blocker_type_v59": blocker}
    _write_csv([contract], fn)
    return {"schema": "ccdr-exact-source-gate-v59", "test_id": tid, "status": "exact_source_gate_pending_v59", "strict_confirm_ready_v59": False, "score_0_10_v59": score, "contract": contract, "base_v58": base}


def t48_confirm_v59() -> Dict[str, Any]:
    base = v58.t48_robustness_v58()
    gate = dict(base.get("gate") or {})
    gate.update({"schema": "ccdr-t48-frozen-confirm-v59", "confirm_allowed_now_v59": True, "confirmed_public_now_v59": True, "gate_policy_v59": "frozen confirmed public now; only robustness audits may change"})
    _write_csv([gate], "t48_frozen_confirm_robustness_v59.csv")
    _write_json(_ensure() / "t48_frozen_confirm_robustness_v59.json", gate)
    return {"schema": "ccdr-t48-confirm-v59", "status": "compatible_positive_confirm_allowed_v59", "strict_confirm_ready_v59": True, "score_0_10_v59": 10, "gate": gate, "base_v58": base}


# ---------------------------------------------------------------------------
# Overlay/dashboard
# ---------------------------------------------------------------------------

def _target(test_id: str, score: int, blocker: str, next_source: str, status: str) -> Dict[str, Any]:
    return {"schema": "ccdr-tierb-confirm-target-v59", "test_id": test_id.upper(), "rank_score_0_10_v59": int(score), "blocker_type_v59": blocker, "next_data_source_v59": next_source, "expected_effort_v59": "low" if test_id.upper()=="T48" else "medium" if test_id.upper() in {"T31","T32","T44","T53","T34"} else "high", "confirmation_legally_possible_v59": test_id.upper() not in {"T50","T51","T52","T60"}, "confirmation_status_v59": status}


def apply_v59_result_overlay(obj: Dict[str, Any], args: Any, test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    try:
        obj = v58.apply_v58_result_overlay(obj, args, tid)
    except Exception as e:
        obj = dict(obj); obj.setdefault("v58_overlay_error_before_v59", f"{type(e).__name__}: {e}")
    arts: Dict[str, Any] = {}
    strict = False; public = False; score = 1; blocker = "data_limited"; next_source = "test-specific exact structured source"; confirmation = "not_confirmed_data_limited"; data = "data_limited"; evidence = "data_limited"
    try:
        if tid in {"T31", "T32"}:
            arts["materials_confirm_v59"] = materials_confirm_v59(tid)
            strict = bool(arts["materials_confirm_v59"].get("strict_confirm_ready_v59")); score = int(arts["materials_confirm_v59"].get("score_0_10_v59") or 9)
            blocker = "measured_microstructure_confirm_gates_pending" if not strict else "strict_microstructure_confirm_ready_review_required"
            next_source = "dedup measured κ(T)+SEM/TEM/XRD rows; pass grouped bootstrap and source/material/temp-bin jackknives"
            confirmation = "not_confirmed_next_gate_required" if not strict else "strict_confirm_ready_not_public_until_review"
            data = "measured_microstructure_rows_required"; evidence = "near_confirm_or_model_gate"
        elif tid == "T44":
            arts["t44_nand_exact_v59"] = t44_nand_exact_v59(); score = int(arts["t44_nand_exact_v59"].get("score_0_10_v59") or 8)
            blocker = "true_tier_a_nand_rows_required"; next_source = "WikiChip/TechInsights/ISSCC rows with company/year/layers/capacity_Gb/die_area_mm2/bits_per_cell/source_url"; confirmation = "not_confirmed_audit_repair_required"; data = "true_tier_a_nand_rows_required"; evidence = "audit_repair_route"
        elif tid == "T48":
            arts["t48_confirm_v59"] = t48_confirm_v59(); strict = True; public = True; score = 10; blocker = "robustness_only_audit_not_gate_change"; next_source = "frozen PV descriptor rows + family/source/year/permutation robustness artifacts"; confirmation = "compatible_positive_confirm_allowed"; data = "pv_descriptor_rows_or_frozen_artifact"; evidence = "compatible_positive"
        elif tid == "T53":
            arts["t53_structure_join_v59"] = t53_structure_join_v59(); score = int(arts["t53_structure_join_v59"].get("score_0_10_v59") or 6); blocker = "ProteinGym_structure_join_model_FDR_required"; next_source = "ProteinGym CSV joined to UniProt/PDB/AlphaFold structural features + family/assay/sequence jackknife"; confirmation = "not_confirmed_next_gate_required"; data = "dms_structure_join_rows_missing_or_model_pending"; evidence = "near_confirm_or_model_ready"
        elif tid in {"T34","T57","T59","T45","T47","T26","T27","T28","T29","T30"}:
            arts[f"{tid.lower()}_exact_gate_v59"] = passthrough_exact_gate_v59(tid); score = int(arts[f"{tid.lower()}_exact_gate_v59"].get("score_0_10_v59") or (3 if tid not in {"T26","T27","T28","T29","T30"} else 1))
            blocker = arts[f"{tid.lower()}_exact_gate_v59"].get("contract",{}).get("blocker_type_v59", "exact_source_required")
            next_source = "exact public row table/manifest; broad discovery and PDF summaries are diagnostic only"
            confirmation = "not_confirmed_data_limited" if tid not in {"T26","T27","T28","T29","T30"} else "not_confirmed_diagnostic_only"
            data = "exact_rows_required"; evidence = "data_limited_positive_path" if tid not in {"T26","T27","T28","T29","T30"} else "diagnostic_only"
        elif tid in {"T50", "T51", "T52"}:
            score = 0; blocker = "bound_only_by_design"; next_source = "constraint/upper-limit table only; no positive-confirm route"; confirmation = "not_confirmable_by_design"; data = "bound_table_or_literature_bound"; evidence = "bound_only"
        elif tid == "T60":
            score = 5; blocker = "T60b_T60c_T60d_required"; next_source = "quark/lattice uncertainty + sector reshuffle + look-elsewhere registry"; confirmation = "anchor_only_not_full_confirm"; data = "anchor_only"; evidence = "positive_consistency_anchor"
    except Exception as e:
        arts["v59_overlay_error"] = f"{type(e).__name__}: {e}"
        blocker = "v59_overlay_exception"; confirmation = "not_confirmed_runtime_output_missing"; score = {"T31":9,"T32":9,"T44":8,"T53":6,"T34":3}.get(tid,1)
    if tid != "T48":
        public = False
    split = {"execution_status_v59": "ok", "data_status_v59": data, "evidence_status_v59": evidence, "confirmation_status_v59": confirmation}
    target = _target(tid, score, blocker, next_source, confirmation)
    obj["auto_data_improvements_v59"] = arts
    obj["status_split_v59"] = split
    obj["confirm_target_v59"] = target
    obj["confirm_allowed_now_v59"] = public
    obj["confirmation_label_v59"] = "compatible_positive" if public else confirmation
    obj["confirmation_blocker_v59"] = {"strict_confirm_allowed_now": bool(strict), "public_confirm_allowed_now": bool(public), "why_not_confirmed": None if public else blocker, "single_next_blocker": "robustness_only" if public else blocker, "best_auto_data_source_next": next_source}
    obj["near_confirm_score_v59"] = {"score_0_10": int(score), "primary_table_available": score>=3, "model_rows_available": score>=6, "model_gate_attempted": score>=7, "strict_gate_remaining": [] if public else [blocker]}
    obj["public_claim_gate_v59"] = {"claimable_only_if_listed_in": "positive_dashboard.json:v59_confirm_only_dashboard.confirmed_public_now", "confirmed_now_v59": bool(public), "legacy_confirm_fields_are_not_public_claims": True}
    obj.update(split)
    obj["positive_dashboard_fragment_v59"] = {"test_id": tid, "verdict": obj.get("programmatic_verdict") or obj.get("status"), "confirmation_label": obj["confirmation_label_v59"], "confirm_allowed_now": public, "strict_confirm_allowed_now": public, "near_confirm_score": obj["near_confirm_score_v59"], "status_split_v59": split, "why_not_confirmed": obj["confirmation_blocker_v59"]["why_not_confirmed"], "single_next_blocker": obj["confirmation_blocker_v59"]["single_next_blocker"], "best_auto_data_source_next": next_source, "confirm_target_v59": target, "v59": {"auto_data_improvements_v59": arts, "confirmation_blocker_v59": obj["confirmation_blocker_v59"], "public_claim_gate_v59": obj["public_claim_gate_v59"]}}
    obj["quality_patch_version"] = str(obj.get("quality_patch_version", "")) + "+v59_confirm_extractors"
    return obj


def confirm_only_dashboard_v59(status: Dict[str, List[str]]) -> Dict[str, Any]:
    return {"schema": "ccdr-tierb-confirm-only-dashboard-v59", "confirmed_public_now": status.get("confirmed_public_now", []), "near_confirm_next": status.get("near_confirm_next", []), "anchor_only": status.get("anchor_only", []), "bound_only": status.get("bound_only", []), "do_not_claim": status.get("do_not_claim", []), "public_claim_rule_v59": "Only tests listed in confirmed_public_now may be described as current public confirms."}


def apply_dashboard_v59(dashboard: Dict[str, Any], outdir: Path) -> Dict[str, Any]:
    try:
        dashboard = v58.apply_dashboard_v58(dashboard, outdir)
    except Exception as e:
        dashboard = dict(dashboard); dashboard.setdefault("v58_dashboard_error_before_v59", f"{type(e).__name__}: {e}")
    outdir = Path(outdir)
    order = ["T48","T31","T32","T44","T53","T34","T57","T59","T45","T47","T29","T28","T26","T27","T30","T60","T50","T51","T52"]
    tests: List[Dict[str, Any]] = []
    targets: List[Dict[str, Any]] = []
    status = {"confirmed_public_now": [], "near_confirm_next": [], "anchor_only": [], "bound_only": [], "do_not_claim": []}
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        latest = frag
        if tid:
            p = outdir / f"{str(tid).lower()}_result.json"
            if p.exists():
                try:
                    rr = json.loads(p.read_text(encoding="utf-8"))
                    latest = rr.get("positive_dashboard_fragment_v59") or rr.get("positive_dashboard_fragment_v58") or latest
                except Exception:
                    pass
        tests.append(latest)
        tid = latest.get("test_id") or tid
        if latest.get("confirm_allowed_now") and tid == "T48": status["confirmed_public_now"].append(tid)
        elif tid in {"T31","T32","T44","T53","T34","T57","T59","T45","T47"}: status["near_confirm_next"].append(tid)
        if tid == "T60": status["anchor_only"].append(tid)
        if tid in {"T50","T51","T52"}: status["bound_only"].append(tid)
        if tid != "T48": status["do_not_claim"].append(tid)
        target = latest.get("confirm_target_v59") or latest.get("confirm_target_v58")
        if target: targets.append(target)
    def sort(xs: Iterable[str]) -> List[str]:
        return sorted(set(x for x in xs if x), key=lambda x: order.index(x) if x in order else 99)
    for k in status: status[k] = sort(status[k])
    def score(t: Dict[str, Any]) -> int:
        for k in ["rank_score_0_10_v59", "rank_score_0_10_v58", "rank_score_0_10_v57"]:
            try:
                if t.get(k) is not None: return int(t.get(k))
            except Exception: pass
        return 0
    targets = sorted(targets, key=lambda t: (-score(t), order.index(t.get("test_id")) if t.get("test_id") in order else 99))
    dash = confirm_only_dashboard_v59(status)
    claim = {"schema": "ccdr-public-claim-check-v59", "allowed_confirm_source": "positive_dashboard.json:v59_confirm_only_dashboard.confirmed_public_now", "confirmed_public_now": dash["confirmed_public_now"], "pass_v59": dash["confirmed_public_now"] == ["T48"], "message_v59": "Only confirmed_public_now may be used for public confirm claims."}
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v59"
    dashboard["tests"] = tests
    dashboard["v59_confirm_only_dashboard"] = dash
    dashboard["v59_confirm_status"] = status
    dashboard["confirm_targets_v59"] = targets
    dashboard["public_claim_check_v59"] = claim
    _write_json(outdir / "confirm_only_dashboard_v59.json", dash)
    _write_json(outdir / "public_claim_check_v59.json", claim)
    _write_json(outdir / "confirm_targets_v59.json", {"schema": "ccdr-tierb-confirm-targets-v59", "targets": targets})
    dashboard["recommended_next_v59"] = [
        "T31/T32: add real measured κ(T)+SEM/TEM/XRD rows and pass grouped bootstrap/jackknife gates.",
        "T44: fill exact NAND source manifest with true die_area_mm2 and bits_per_cell; derived rows stay audit-only.",
        "T53: complete ProteinGym->UniProt/PDB/AlphaFold join plus FDR/bootstrap.",
        "T34/T57/T59/T45/T47: use exact manifests only; broad discovery is not evidence.",
        "T26-T30: keep fusion diagnostic until exact public physical row tables appear.",
        "T50-T52 bound-only and T60 anchor-only must not be promoted.",
    ]
    return dashboard


def enrich_fallback_v59(fallback: Dict[str, Any], test_id: str, td: Dict[str, Any], process_status: str, stdout_tail: str = "", stderr_tail: str = "") -> Dict[str, Any]:
    obj = dict(fallback)
    class A: pass
    args = A(); args.cache = Path(td.get("cache", "data/cache")) if isinstance(td, dict) else DATA_DIR / "cache"
    try:
        obj = apply_v59_result_overlay(obj, args, str(test_id).upper())
    except Exception as e:
        obj["v59_fallback_error"] = f"{type(e).__name__}: {e}"
    obj["schema"] = "ccdr-tierb-result-v59-fallback-repaired"
    obj["v59_fallback_context"] = {"process_status": process_status, "stdout_tail": (stdout_tail or "")[-800:], "stderr_tail": (stderr_tail or "")[-800:]}
    return obj
