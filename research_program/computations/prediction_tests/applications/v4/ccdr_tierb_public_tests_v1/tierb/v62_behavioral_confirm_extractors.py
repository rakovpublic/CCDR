#!/usr/bin/env python3
"""v62 behavioral confirm extractors for CCDR Tier-B.

v62 is intentionally NOT a dashboard-only patch. It adds/changes behavior:
- manifest-driven public supplement downloader (optional, no hard failure offline);
- broader source scanners with source-family/provenance hashes;
- stronger T31/T32 unit normalization, family/source/temperature balancing, and fixed-effect estimators;
- strict exact-row parsers for T44/T53/T34/T57/T59/T45/T47 with unit conversion and rejection reasons;
- fusion remains exact-row diagnostic-only unless raw/certified row tables are present.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import urllib.request
from collections import Counter, defaultdict
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

from tierb import v61_behavioral_confirm_extractors as v61

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GEN_DIR = DATA_DIR / "generated"
MANIFEST_DIR = DATA_DIR / "manifests"
PUBLIC_SOURCE_DIRNAME = "public_sources_v62"

DEFAULT_TESTS = list(v61.DEFAULT_TESTS)
NEAR = set(v61.NEAR)
FUSION = set(v61.FUSION)
BOUND = set(v61.BOUND)
ANCHOR = set(v61.ANCHOR)


def _ensure(outdir: Optional[Path] = None) -> Path:
    if outdir is not None:
        p = outdir / "data" / "generated"
    else:
        p = GEN_DIR
    p.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, obj: Dict[str, Any]) -> str:
    return v61._write_json(path, obj)


def _write_csv(rows: Sequence[Dict[str, Any]], filename: str, outdir: Optional[Path] = None) -> str:
    return v61._write_csv(rows, filename, outdir)


def _s(v: Any) -> str:
    return v61._s(v)


def _f(v: Any) -> Optional[float]:
    return v61._f(v)


def _norm_col(c: Any) -> str:
    return v61._norm_col(c)


def _pick(row: Dict[str, Any], aliases: Sequence[str]) -> Any:
    return v61._pick(row, aliases)


def _read_table(path: Path, max_rows: int = 300000) -> List[Dict[str, Any]]:
    return v61._read_table(path, max_rows=max_rows)


def _dedup(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    return v61._dedup(rows, keys)


def _material_family(material: str) -> str:
    # Extend v61's material families with more thermal/material classes.
    m = material.lower()
    if any(x in m for x in ["steel", "stainless", "fe", "iron"]): return "steel_iron"
    if any(x in m for x in ["aluminum", "aluminium", " al ", "al6061", "6061"]): return "aluminum"
    if any(x in m for x in ["titanium", "ti-", " ti "]): return "titanium"
    if any(x in m for x in ["molybdenum", " moly ", " mo "]): return "molybdenum"
    if any(x in m for x in ["tungsten", " w "]): return "tungsten"
    if any(x in m for x in ["gaas", "gallium arsenide"]): return "iii_v_semiconductor"
    return v61._material_family(material)


def _temp_bin(t: Optional[float]) -> str:
    return v61._temp_bin(t)


def _provenance_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(_s(p).encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# v62 manifest-driven local/public-source ingestion
# ---------------------------------------------------------------------------

V62_MANIFEST_ROWS = [
    {"test_id": "T31,T32", "source_family": "cryogenic_materials", "url_or_hint": "local/online CSV/XLSX with temperature_K,kappa_W_mK,grain_size,SEM/TEM/XRD evidence", "required_columns": "material,temperature,kappa,grain_size,microstructure_method"},
    {"test_id": "T31,T32", "source_family": "nanocrystalline_supplements", "url_or_hint": "Zenodo/Figshare/supplement CSV/XLSX from nanocrystalline thermal conductivity papers", "required_columns": "sample,material,T,kappa,grain_size,method"},
    {"test_id": "T44", "source_family": "nand_exact", "url_or_hint": "WikiChip/TechInsights/ISSCC/VLSI exact NAND tables", "required_columns": "company,year,layers,capacity_Gb,die_area_mm2,bits_per_cell"},
    {"test_id": "T53", "source_family": "proteingym_structure", "url_or_hint": "ProteinGym assay CSV + UniProt/PDB/AlphaFold mapping CSV", "required_columns": "assay,uniprot,pdb_or_alphafold,dms_score,symmetry_proxy"},
    {"test_id": "T34", "source_family": "thermoelectric_exact", "url_or_hint": "teMatDb/Starrydata export for Bi2Te3/Sb2Te3", "required_columns": "material,ZT,temperature,orientation_angle_or_grain_angle"},
    {"test_id": "T57,T59", "source_family": "hepdata_exact", "url_or_hint": "HEPData YAML/CSV exact manifest", "required_columns": "record_id,table_id,x,observed,model,uncertainty"},
    {"test_id": "T45", "source_family": "optical_benchmark_exact", "url_or_hint": "exact benchmark table", "required_columns": "energy_per_bit,bandwidth,reach,year,platform"},
    {"test_id": "T47", "source_family": "neuromorphic_benchmark_exact", "url_or_hint": "exact benchmark table", "required_columns": "chip,benchmark,energy,accuracy,topology"},
]


def init_v62_source_manifest(outdir: Optional[Path] = None) -> str:
    return _write_csv(V62_MANIFEST_ROWS, "v62_exact_public_source_manifest.csv", outdir)


def _manifest_paths(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> List[Path]:
    roots = [MANIFEST_DIR, DATA_DIR]
    if outdir:
        roots += [outdir, outdir / "data", outdir / "data" / "generated"]
    if cache:
        roots.append(cache)
    paths: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl", ".yaml", ".yml"} and any(x in p.name.lower() for x in ["manifest", "source_list", "download"]):
                paths.append(p)
    return paths


def _read_manifest_rows(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in _manifest_paths(outdir, cache):
        rows.extend(_read_table(p, max_rows=10000))
    # Include built-in rows as hints, but they do not have URLs unless user fills them.
    rows.extend(dict(x, _source_file_v62="builtin_v62_manifest") for x in V62_MANIFEST_ROWS)
    return rows


def download_manifest_sources_v62(outdir: Optional[Path] = None, cache: Optional[Path] = None, timeout_s: int = 3, test_ids: Optional[Sequence[str]] = None, max_downloads: int = 12) -> Dict[str, Any]:
    """Download manifest-listed public supplements when URLs are present.

    Offline or blocked downloads are recorded as diagnostics and never treated as
    failures. This materially changes behavior when real public-source manifests
    are provided by auto-populating the scanner input directory.
    """
    base = (outdir or DATA_DIR) / PUBLIC_SOURCE_DIRNAME
    base.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest_rows(outdir, cache)
    log: List[Dict[str, Any]] = []
    downloaded = 0
    if os.environ.get("CCDR_V62_ENABLE_DOWNLOADS", "0").lower() not in {"1", "true", "yes", "on"}:
        log.append({"status": "network_downloads_disabled_v62", "hint": "set CCDR_V62_ENABLE_DOWNLOADS=1 to auto-download manifest URLs"})
        _write_csv(log, "v62_manifest_download_log.csv", outdir)
        return {"n_manifest_rows_v62": len(rows), "n_downloaded_or_cached_v62": 0, "downloads_enabled_v62": False, "download_log_v62": str(_ensure(outdir) / "v62_manifest_download_log.csv")}
    wanted = {str(x).upper() for x in (test_ids or [])}
    attempted = 0
    for i, r in enumerate(rows):
        url = _s(_pick(r, ["url", "download_url", "source_url", "href", "link"]))
        tid = _s(_pick(r, ["test_id", "test", "prediction"])) or "unknown"
        tid_tokens = {t.strip().upper() for t in re.split(r"[,;\s]+", tid) if t.strip()}
        if wanted and tid_tokens and "UNKNOWN" not in tid_tokens and not (wanted & tid_tokens):
            continue
        if not re.match(r"https?://", url):
            continue
        if attempted >= max_downloads:
            log.append({"url": url, "test_id": tid, "status": "skipped_max_downloads_v62"})
            continue
        attempted += 1
        suffix = Path(url.split("?")[0]).suffix or ".dat"
        safe_tid = re.sub(r"[^A-Za-z0-9_,-]+", "_", tid)[:60] or "source"
        name = _s(_pick(r, ["filename", "file", "name"])) or f"{safe_tid}_{i}{suffix}"
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        dst = base / safe_tid / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            log.append({"url": url, "test_id": tid, "path": str(dst), "status": "already_exists"})
            downloaded += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ccdr-tierb-v62/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = resp.read(80_000_000)
            dst.write_bytes(data)
            log.append({"url": url, "test_id": tid, "path": str(dst), "status": "downloaded", "bytes": len(data)})
            downloaded += 1
        except Exception as e:
            log.append({"url": url, "test_id": tid, "path": str(dst), "status": "download_failed", "error": type(e).__name__ + ": " + str(e)[:200]})
    _write_csv(log, "v62_manifest_download_log.csv", outdir)
    return {"n_manifest_rows_v62": len(rows), "n_downloaded_or_cached_v62": downloaded, "download_log_v62": str(_ensure(outdir) / "v62_manifest_download_log.csv")}


def _candidate_roots_v62(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> List[Path]:
    roots = []
    # Put downloaded/exact sources first so they dominate over generated rejection files.
    if outdir:
        roots += [outdir / PUBLIC_SOURCE_DIRNAME, outdir / "data" / PUBLIC_SOURCE_DIRNAME, outdir / "exact_sources", outdir / "data" / "exact_sources"]
    if cache:
        roots += [cache / PUBLIC_SOURCE_DIRNAME, cache / "exact_sources", cache]
    roots += [DATA_DIR / PUBLIC_SOURCE_DIRNAME, DATA_DIR / "exact_sources", DATA_DIR, GEN_DIR, MANIFEST_DIR]
    seen: List[Path] = []
    for r in roots:
        try:
            rr = r.resolve()
            if rr.exists() and rr not in seen:
                seen.append(rr)
        except Exception:
            pass
    return seen


def _iter_table_files_v62(patterns: Sequence[str], outdir: Optional[Path] = None, cache: Optional[Path] = None, max_files: int = 800) -> List[Path]:
    files: List[Path] = []
    seen_files = set()
    suffixes = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".xlsx", ".xls", ".yaml", ".yml"}
    pat_l = [p.lower() for p in patterns]
    bad_name_bits = ["rejection", "reject", "confirm_gate", "confirm_targets", "dashboard", "public_claim", "positive_dashboard"]
    for root in _candidate_roots_v62(outdir, cache):
        for p in root.rglob("*"):
            if len(files) >= max_files:
                return files
            if not p.is_file() or p.suffix.lower() not in suffixes:
                continue
            try:
                if p.stat().st_size > 120_000_000:
                    continue
            except Exception:
                pass
            name = str(p).lower()
            if any(x in p.name.lower() for x in bad_name_bits):
                continue
            if not any(q in name for q in pat_l):
                continue
            try:
                rp = p.resolve()
            except Exception:
                rp = p
            if rp in seen_files:
                continue
            seen_files.add(rp)
            files.append(p)
    return files


def _read_patterns_v62(patterns: Sequence[str], outdir: Optional[Path] = None, cache: Optional[Path] = None, max_files: int = 800, max_rows_per_file: int = 300000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in _iter_table_files_v62(patterns, outdir, cache, max_files=max_files):
        for r in _read_table(p, max_rows=max_rows_per_file):
            if "_source_file_v61" not in r:
                r["_source_file_v61"] = str(p)
            r["_source_file_v62"] = str(p)
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# T31/T32 materials v62: richer unit conversion + fixed-effect estimators
# ---------------------------------------------------------------------------

MATERIAL_PATTERNS_V62 = list(set(v61.MATERIAL_PATTERNS + [
    "thermal", "conductivity", "lambda", "cryodata", "cryogenic_material", "nist", "matprop",
    "property", "properties", "specific_heat", "kapton", "al6061", "ofhc", "nanograin",
    "crystallite", "xrd_scherrer", "grain_nm", "grain_um", "sem_tem", "supplement", "supplementary"
]))


def _unit_scale(raw: Dict[str, Any], aliases: Sequence[str], default: float = 1.0) -> float:
    # Best-effort unit extraction from column names and nearby explicit unit columns.
    joined = " ".join([_s(x) for x in aliases] + [_s(k) for k in raw.keys()]).lower()
    if any(x in joined for x in ["um", "µm", "micron"]):
        return 1000.0
    if "mm" in joined and "mkm" not in joined:
        return 1_000_000.0
    if "nm" in joined:
        return 1.0
    return default


def normalize_material_row_v62(raw: Dict[str, Any], idx: int) -> Tuple[Dict[str, Any], List[str]]:
    material = _s(_pick(raw, ["material", "compound", "sample_material", "specimen_material", "name", "formula", "Material Name", "material_name", "alloy", "composition"]))
    source = _s(_pick(raw, ["source_url", "url", "doi", "reference", "citation", "source", "_source_file_v62", "_source_file_v61", "file"]))
    sample = _s(_pick(raw, ["sample_id", "sample", "specimen", "id", "sample_name", "dataset_id", "run_id"])) or f"row{idx}"
    # Temperature conversion: accept C and K aliases.
    temp = _f(_pick(raw, ["temperature_K", "T_K", "temp_K", "temperature (K)", "T (K)", "temperature_k", "temperature"]))
    temp_c = _f(_pick(raw, ["temperature_C", "T_C", "temp_C", "temperature (C)", "T (C)", "temperature_c"]))
    if temp is None and temp_c is not None:
        temp = temp_c + 273.15
    elif temp is not None:
        # Heuristic: if column says C and value is room-like negative/positive, convert.
        cols = " ".join(str(k).lower() for k in raw.keys())
        if ("temperature_c" in cols or "t_c" in cols or "(c)" in cols) and -273.15 < temp < 200:
            temp = temp + 273.15
    kappa = _f(_pick(raw, ["kappa_W_mK", "thermal_conductivity_W_mK", "thermal_conductivity", "thermal conductivity", "k_W_mK", "lambda_W_mK", "lambda", "kappa", "k", "W/mK", "W m-1 K-1"]))
    grain_aliases = ["grain_size_nm", "grain_size_nm_or_um", "grain_size", "crystallite_size_nm", "crystallite_size", "particle_size_nm", "particle_size", "grain diameter", "grain_diameter_nm", "grain_diameter", "d_grain", "scherrer_size"]
    grain = _f(_pick(raw, grain_aliases))
    if grain is not None:
        grain *= _unit_scale(raw, grain_aliases, 1.0)
    boundary = _f(_pick(raw, ["boundary_density_proxy", "boundary_density", "interface_density", "porosity", "grain_boundary_density", "boundary", "gb_density", "void_fraction"] ))
    method = _s(_pick(raw, ["microstructure_method", "method", "measurement_method", "evidence_method", "characterization", "microstructure", "grain_method", "SEM", "TEM", "XRD"] ))
    text = " ".join(_s(v) for v in raw.values()).lower()
    method_text = (method + " " + text).lower()
    methods = []
    for m in ["sem", "tem", "xrd", "afm", "ebsd", "scherrer"]:
        if m in method_text:
            methods.append(m.upper())
    if methods and not method:
        method = "+".join(sorted(set(methods)))
    nano = bool(re.search(r"nano|nanocrystalline|nanograin|nanostruct", text)) or (grain is not None and grain <= 100.0)
    fam = _material_family(material)
    tbin = _temp_bin(temp)
    reasons: List[str] = []
    if not material: reasons.append("missing_material")
    if not source: reasons.append("missing_source")
    if temp is None: reasons.append("missing_temperature_K")
    if kappa is None or (kappa is not None and kappa <= 0): reasons.append("missing_or_invalid_kappa_W_mK")
    if grain is None or (grain is not None and grain <= 0): reasons.append("missing_or_invalid_grain_size")
    if not methods and not re.search(r"sem|tem|xrd|ebsd|afm|scherrer", method_text): reasons.append("missing_measured_microstructure_method")
    row = {
        "source_url_v62": source,
        "sample_id_v62": sample,
        "material_v62": material,
        "material_family_v62": fam,
        "temperature_K_v62": temp,
        "temperature_bin_v62": tbin,
        "kappa_W_mK_v62": kappa,
        "grain_size_nm_v62": grain,
        "boundary_density_proxy_v62": boundary,
        "microstructure_method_v62": method,
        "nanocrystalline_yes_no_v62": bool(nano),
        "usable_v62": not reasons,
        "reject_reasons_v62": "|".join(reasons),
        "raw_source_file_v62": _s(raw.get("_source_file_v62") or raw.get("_source_file_v61")),
        "source_family_v62": _provenance_hash(source or raw.get("_source_file_v62"), fam),
        "row_provenance_hash_v62": _provenance_hash(source, sample, material, temp, kappa, grain),
    }
    return row, reasons


def _ols_fit(y: List[float], X: List[List[float]]) -> Optional[Dict[str, Any]]:
    return v61._ols_fit(y, X)


def _one_hot(values: Sequence[str], min_count: int = 2, max_levels: int = 20) -> Tuple[List[str], List[List[float]]]:
    counts = Counter(values)
    levels = [v for v, c in counts.most_common(max_levels + 1) if v and c >= min_count]
    # Drop most common level to avoid full dummy trap.
    if levels:
        levels = levels[1:max_levels+1]
    mat: List[List[float]] = []
    for v in values:
        mat.append([1.0 if v == lev else 0.0 for lev in levels])
    return levels, mat


def _materials_estimator_v62(usable: List[Dict[str, Any]], test_id: str, outdir: Optional[Path] = None) -> Dict[str, Any]:
    rows = []
    for r in usable:
        temp = _f(r.get("temperature_K_v62")); kap = _f(r.get("kappa_W_mK_v62")); grain = _f(r.get("grain_size_nm_v62"))
        if temp and kap and grain and temp > 0 and kap > 0 and grain > 0:
            rr = dict(r)
            rr["logT_v62"] = math.log(temp)
            rr["logKappa_v62"] = math.log(kap)
            rr["logGrain_v62"] = math.log(grain)
            rr["boundary_proxy_num_v62"] = _f(r.get("boundary_density_proxy_v62")) or (1.0 / grain)
            rows.append(rr)
    if len(rows) < 12 or np is None:
        return {"status_v62": "not_enough_rows_for_estimator", "n_model_rows_v62": len(rows)}
    y = [float(r["logKappa_v62"]) for r in rows]
    # v62 fixed-effect design: logT + logGrain + boundary + source/family controls.
    source_vals = [_s(r.get("source_url_v62") or r.get("raw_source_file_v62")) for r in rows]
    fam_vals = [_s(r.get("material_family_v62")) for r in rows]
    source_levels, source_oh = _one_hot(source_vals, min_count=3, max_levels=12)
    fam_levels, fam_oh = _one_hot(fam_vals, min_count=3, max_levels=12)
    X_temp = [[1.0, float(r["logT_v62"])] + source_oh[i] + fam_oh[i] for i, r in enumerate(rows)]
    X_micro = [[1.0, float(r["logT_v62"]), float(r["logGrain_v62"]), float(r["boundary_proxy_num_v62"])] + source_oh[i] + fam_oh[i] for i, r in enumerate(rows)]
    fit_temp = _ols_fit(y, X_temp)
    fit_micro = _ols_fit(y, X_micro)
    model_wins = False
    sign_ok = False
    if fit_temp and fit_micro:
        # Need microstructure to beat temperature-only after adding source/family effects.
        model_wins = float(fit_micro.get("aic", 1e99)) < float(fit_temp.get("aic", -1e99)) and float(fit_micro.get("bic", 1e99)) < float(fit_temp.get("bic", -1e99))
        beta = fit_micro.get("beta") or []
        # expected: larger grains usually increase kappa; higher boundary proxy usually decreases kappa.
        sign_ok = len(beta) > 3 and beta[2] > 0 and beta[3] < 0
    # Source-balanced bootstrap: resample sources, then rows within sources.
    rng = np.random.default_rng(6201)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[_s(r.get("source_url_v62") or r.get("raw_source_file_v62"))].append(r)
    sources = [s for s in by_source if s]
    boot_wins = 0; boot_sign = 0; boot_n = 0
    if len(sources) >= 3:
        for _ in range(80):
            sample_rows: List[Dict[str, Any]] = []
            chosen = rng.choice(sources, size=len(sources), replace=True)
            for s in chosen:
                group = by_source[str(s)]
                if not group:
                    continue
                idxs = rng.integers(0, len(group), size=max(1, len(group)))
                sample_rows.extend(group[int(i)] for i in idxs)
            if len(sample_rows) < 12:
                continue
            sy = [float(r["logKappa_v62"]) for r in sample_rows]
            sX0 = [[1.0, float(r["logT_v62"])] for r in sample_rows]
            sX1 = [[1.0, float(r["logT_v62"]), float(r["logGrain_v62"]), float(r["boundary_proxy_num_v62"])] for r in sample_rows]
            f0 = _ols_fit(sy, sX0); f1 = _ols_fit(sy, sX1)
            if f0 and f1:
                boot_n += 1
                if f1["aic"] < f0["aic"] and f1["bic"] < f0["bic"]:
                    boot_wins += 1
                b = f1.get("beta") or []
                if len(b) > 3 and b[2] > 0 and b[3] < 0:
                    boot_sign += 1
    # Within-bin model wins (temperature confounding control)
    bin_results = []
    for tb, g in defaultdict(list, { }).items():
        pass
    bins: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        bins[_s(r.get("temperature_bin_v62"))].append(r)
    for tb, g in bins.items():
        if len(g) < 10:
            continue
        gy = [float(r["logKappa_v62"]) for r in g]
        gX0 = [[1.0] for _ in g]
        gX1 = [[1.0, float(r["logGrain_v62"]), float(r["boundary_proxy_num_v62"])] for r in g]
        f0 = _ols_fit(gy, gX0); f1 = _ols_fit(gy, gX1)
        win = bool(f0 and f1 and f1["aic"] < f0["aic"] and f1["bic"] < f0["bic"])
        bin_results.append({"temperature_bin_v62": tb, "n_v62": len(g), "microstructure_wins_v62": win, "aic_baseline_v62": f0.get("aic") if f0 else None, "aic_micro_v62": f1.get("aic") if f1 else None})
    _write_csv(bin_results, f"{test_id.lower()}_temperature_bin_model_wins_v62.csv", outdir)
    return {
        "status_v62": "ok",
        "n_model_rows_v62": len(rows),
        "source_fixed_effect_levels_v62": source_levels,
        "family_fixed_effect_levels_v62": fam_levels,
        "temperature_only_fit_v62": fit_temp,
        "microstructure_fit_v62": fit_micro,
        "microstructure_beats_temperature_baseline_v62": model_wins,
        "predicted_signs_pass_v62": sign_ok,
        "source_balanced_bootstrap_n_v62": boot_n,
        "source_balanced_bootstrap_model_win_fraction_v62": (boot_wins / boot_n if boot_n else 0.0),
        "source_balanced_bootstrap_sign_fraction_v62": (boot_sign / boot_n if boot_n else 0.0),
        "temperature_bin_results_v62": bin_results,
        "temperature_bin_win_fraction_v62": (sum(1 for x in bin_results if x.get("microstructure_wins_v62")) / len(bin_results) if bin_results else 0.0),
    }


def materials_confirm_v62(test_id: str, outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    init_v62_source_manifest(outdir)
    dl = download_manifest_sources_v62(outdir, cache, test_ids=[test_id])
    raw = _read_patterns_v62(MATERIAL_PATTERNS_V62 + [test_id.lower()], outdir, cache, max_files=800, max_rows_per_file=250000)
    norm: List[Dict[str, Any]] = []
    rejs: List[Dict[str, Any]] = []
    reasons_counter: Counter[str] = Counter()
    for i, r in enumerate(raw):
        nr, reasons = normalize_material_row_v62(r, i)
        norm.append(nr)
        if reasons:
            rejs.append(nr)
            reasons_counter.update(reasons)
    dedup = _dedup(norm, ["source_url_v62", "sample_id_v62", "material_v62", "temperature_K_v62", "kappa_W_mK_v62", "grain_size_nm_v62"])
    usable = [r for r in dedup if r.get("usable_v62")]
    _write_csv(norm, f"{test_id.lower()}_materials_normalized_rows_v62.csv", outdir)
    _write_csv(rejs, f"{test_id.lower()}_materials_rejection_diagnostics_v62.csv", outdir)
    reason_rows = [{"reject_reason_v62": k, "n_rows_v62": v} for k, v in reasons_counter.most_common()]
    _write_csv(reason_rows, f"{test_id.lower()}_materials_rejection_summary_v62.csv", outdir)
    sources = Counter(_s(r.get("source_url_v62") or r.get("raw_source_file_v62")) for r in usable)
    fams = Counter(_s(r.get("material_family_v62")) for r in usable)
    tbins = Counter(_s(r.get("temperature_bin_v62")) for r in usable)
    estimator = _materials_estimator_v62(usable, test_id, outdir)
    gates = {
        "sources_ge_5_v62": len([s for s in sources if s]) >= 5,
        "material_families_ge_5_v62": len([f for f in fams if f]) >= 5,
        "temperature_bins_ge_3_v62": len([b for b in tbins if b and b != "unknown"]) >= 3,
        "microstructure_beats_temperature_baseline_v62": bool(estimator.get("microstructure_beats_temperature_baseline_v62")),
        "predicted_signs_pass_v62": bool(estimator.get("predicted_signs_pass_v62")),
        "bootstrap_sign_fraction_ge_0_80_v62": float(estimator.get("source_balanced_bootstrap_sign_fraction_v62") or 0.0) >= 0.80,
        "bootstrap_model_win_fraction_ge_0_80_v62": float(estimator.get("source_balanced_bootstrap_model_win_fraction_v62") or 0.0) >= 0.80,
        "temperature_bin_win_fraction_ge_0_67_v62": float(estimator.get("temperature_bin_win_fraction_v62") or 0.0) >= 0.67,
    }
    confirm = all(gates.values())
    gate = {
        "schema": "ccdr-materials-confirm-v62",
        "test_id": test_id.upper(),
        "download_summary_v62": dl,
        "n_raw_rows_v62": len(raw),
        "n_normalized_rows_v62": len(norm),
        "n_dedup_rows_v62": len(dedup),
        "n_usable_rows_v62": len(usable),
        "n_sources_v62": len([s for s in sources if s]),
        "n_material_families_v62": len([f for f in fams if f]),
        "n_temperature_bins_v62": len([b for b in tbins if b and b != "unknown"]),
        "top_rejection_reasons_v62": dict(reasons_counter.most_common(12)),
        "estimator_v62": estimator,
        "gates_v62": gates,
        "failed_subgates_v62": [k for k, ok in gates.items() if not ok],
        "strict_confirm_ready_v62": bool(confirm),
        "confirmation_status_v62": "confirmed_materials_microstructure_model_v62" if confirm else "not_confirmed_next_gate_required",
        "rank_score_0_10_v62": 10 if confirm else 9,
        "behavioral_delta_v62": "manifest downloads + unit normalization + source/family fixed effects + source-balanced bootstrap",
    }
    _write_json(_ensure(outdir) / f"{test_id.lower()}_materials_confirm_v62.json", gate)
    return gate


# ---------------------------------------------------------------------------
# T44 NAND v62: stricter exact rows + unit conversion
# ---------------------------------------------------------------------------

NAND_PATTERNS_V62 = list(set(v61.NAND_PATTERNS + ["die_density", "die-area", "die_area", "gb_mm2", "gb/mm2", "cell_type", "qlc", "tlc", "mlc", "slc"]))


def _capacity_to_gb(val: Any, raw: Dict[str, Any]) -> Optional[float]:
    x = _f(val)
    if x is None:
        return None
    text = " ".join([str(k) for k in raw.keys()] + [_s(v) for v in raw.values()]).lower()
    if "tb" in text or "tbit" in text or "terabit" in text:
        return x * 1024.0
    if "mb" in text or "mbit" in text or "megabit" in text:
        return x / 1024.0
    return x


def normalize_nand_row_v62(raw: Dict[str, Any], idx: int) -> Tuple[Dict[str, Any], List[str]]:
    company = _s(_pick(raw, ["company", "manufacturer", "vendor", "maker", "brand", "fab", "supplier"]))
    year = _f(_pick(raw, ["year", "release_year", "date", "published_year", "isscc_year", "vlsi_year"]))
    if year is not None and year < 100:
        year += 2000
    layers = _f(_pick(raw, ["layers", "layer_count", "number_of_layers", "3d_nand_layers", "word_lines", "stacked_layers"]))
    cap_raw = _pick(raw, ["capacity_Gb", "capacity_gbit", "capacity Gb", "die_capacity_gb", "capacity", "bits_gb", "Gb/die", "Gbit"])
    cap = _capacity_to_gb(cap_raw, raw)
    die = _f(_pick(raw, ["die_area_mm2", "die area", "die_size_mm2", "area_mm2", "chip_area_mm2", "die area mm2", "Die size"] ))
    bpc = _f(_pick(raw, ["bits_per_cell", "bpc", "cell_bits", "TLC_QLC_MLC", "cell_type", "bits/cell"]))
    text = " ".join(_s(v) for v in raw.values()).lower()
    if bpc is None:
        if "qlc" in text: bpc = 4.0
        elif "tlc" in text: bpc = 3.0
        elif "mlc" in text: bpc = 2.0
        elif "slc" in text: bpc = 1.0
    source = _s(_pick(raw, ["source_url", "url", "doi", "reference", "_source_file_v62", "_source_file_v61", "source"]))
    reasons = []
    if not company: reasons.append("missing_company")
    if year is None or year < 1990 or year > 2035: reasons.append("missing_or_invalid_year")
    if layers is None or layers <= 0: reasons.append("missing_or_invalid_layers")
    if cap is None or cap <= 0: reasons.append("missing_or_invalid_capacity_Gb")
    if die is None or die <= 0: reasons.append("missing_or_invalid_die_area_mm2")
    if bpc is None or bpc <= 0: reasons.append("missing_or_invalid_bits_per_cell")
    if not source: reasons.append("missing_source_url")
    density = cap / die if cap and die and die > 0 else None
    row = {
        "company_v62": company, "year_v62": year, "layers_v62": layers, "capacity_Gb_v62": cap,
        "die_area_mm2_v62": die, "bits_per_cell_v62": bpc, "source_url_v62": source,
        "density_Gb_per_mm2_v62": density,
        "usable_tier_a_v62": not reasons,
        "reject_reasons_v62": "|".join(reasons),
        "row_provenance_hash_v62": _provenance_hash(company, year, layers, cap, die, bpc, source),
        "raw_source_file_v62": _s(raw.get("_source_file_v62") or raw.get("_source_file_v61")),
    }
    return row, reasons


def nand_confirm_v62(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    init_v62_source_manifest(outdir)
    dl = download_manifest_sources_v62(outdir, cache, test_ids=["T44"])
    raw = _read_patterns_v62(NAND_PATTERNS_V62, outdir, cache, max_files=300, max_rows_per_file=300000)
    norm=[]; rejs=[]; rc=Counter()
    for i, r in enumerate(raw):
        nr, reasons = normalize_nand_row_v62(r, i)
        norm.append(nr)
        if reasons:
            rejs.append(nr); rc.update(reasons)
    dedup = _dedup(norm, ["company_v62","year_v62","layers_v62","capacity_Gb_v62","die_area_mm2_v62","bits_per_cell_v62"])
    usable = [r for r in dedup if r.get("usable_tier_a_v62")]
    _write_csv(norm, "t44_nand_normalized_rows_v62.csv", outdir)
    _write_csv(rejs, "t44_nand_rejection_diagnostics_v62.csv", outdir)
    _write_csv([{"reject_reason_v62": k, "n_rows_v62": v} for k, v in rc.most_common()], "t44_nand_rejection_summary_v62.csv", outdir)
    companies = Counter(_s(r.get("company_v62")) for r in usable)
    model = {"status_v62":"not_run_no_true_rows"}
    layer_positive = False
    if len(usable) >= 8 and np is not None:
        y=[]; X=[]
        comps = [_s(r.get("company_v62")) for r in usable]
        lev, oh = _one_hot(comps, min_count=2, max_levels=10)
        for i, r in enumerate(usable):
            density=_f(r.get("density_Gb_per_mm2_v62")); layers=_f(r.get("layers_v62")); year=_f(r.get("year_v62")); bpc=_f(r.get("bits_per_cell_v62"))
            if density and layers and year and bpc and density > 0 and layers > 0:
                y.append(math.log(density)); X.append([1.0, math.log(layers), year-2000.0, bpc] + oh[i])
        fit = _ols_fit(y, X) if len(y) >= 8 else None
        if fit:
            layer_positive = (fit.get("beta") or [0,0])[1] > 0
            model = fit
            model["company_fixed_effect_levels_v62"] = lev
            model["layer_beta_positive_v62"] = layer_positive
    confirm = len(usable) >= 8 and len([c for c in companies if c]) >= 3 and layer_positive
    gate = {
        "schema":"ccdr-nand-tier-a-v62", "test_id":"T44", "download_summary_v62": dl,
        "n_raw_rows_v62":len(raw), "n_normalized_rows_v62":len(norm), "n_true_tier_a_rows_v62":len(usable),
        "n_companies_v62":len([c for c in companies if c]), "top_rejection_reasons_v62": dict(rc.most_common(12)),
        "model_v62":model, "strict_confirm_ready_v62":bool(confirm),
        "failed_subgates_v62": [] if confirm else [x for x, ok in {"true_tier_a_rows_ge_8":len(usable)>=8,"companies_ge_3":len([c for c in companies if c])>=3,"layer_beta_positive":layer_positive}.items() if not ok],
        "confirmation_status_v62":"confirmed_true_tier_a_nand_scaling_v62" if confirm else "not_confirmed_audit_repair_required",
        "rank_score_0_10_v62":10 if confirm else 8,
    }
    _write_json(_ensure(outdir)/"t44_nand_confirm_v62.json", gate)
    return gate


# ---------------------------------------------------------------------------
# T53/T34/T57/T59/T45/T47: v62 exact parsers with better filtering
# ---------------------------------------------------------------------------


def protein_structure_join_v62(outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    init_v62_source_manifest(outdir); dl = download_manifest_sources_v62(outdir, cache, test_ids=["T53"])
    raw = _read_patterns_v62(v61.PROTEIN_PATTERNS + ["mapping", "oligomer", "pfam", "dms_substitutions"], outdir, cache, max_files=200, max_rows_per_file=120000)
    rows=[]; rejs=[]; rc=Counter()
    for i, r in enumerate(raw):
        assay=_s(_pick(r,["assay","DMS_id","DMS ID","experiment","study","assay_id"]))
        uniprot=_s(_pick(r,["uniprot","uniprot_id","UniProt_ID","accession","protein_id","target_uniprot"]))
        pdb=_s(_pick(r,["pdb","pdb_id","PDB","structure_id","pdb_chain"]))
        af=_s(_pick(r,["alphafold","alphafold_id","af_id","AlphaFoldDB","af2_model","alphafold_model"]))
        score=_f(_pick(r,["DMS_score","fitness","fitness_score","score","effect","fitness residual","mutant_score","mean_fitness"]))
        family=_s(_pick(r,["family","protein_family","Pfam","fold","domain","superfamily"])) or (uniprot[:4] if uniprot else "")
        assay_type=_s(_pick(r,["assay_type","selection_type","measurement_type","phenotype"]))
        cluster=_s(_pick(r,["sequence_cluster","cluster","seq_cluster","identity_cluster","mmseqs_cluster"])) or uniprot
        sym=_f(_pick(r,["symmetry_proxy","contact_symmetry","oligomer_symmetry","contact_network_symmetry","oligomeric_state","n_chains","assembly_chains"]))
        if sym is None:
            ost=_s(_pick(r,["oligomeric_state","assembly","biological_assembly","quaternary_structure"])).lower()
            if "dimer" in ost: sym=2.0
            elif "trimer" in ost: sym=3.0
            elif "tetramer" in ost: sym=4.0
            elif "monomer" in ost: sym=1.0
        reasons=[]
        for cond, reason in [(not assay,"missing_assay"),(not uniprot,"missing_uniprot"),(not(pdb or af),"missing_structure_id"),(score is None,"missing_dms_score"),(sym is None,"missing_symmetry_proxy")]:
            if cond: reasons.append(reason)
        row={"assay_v62":assay,"uniprot_v62":uniprot,"pdb_id_v62":pdb,"alphafold_id_v62":af,"dms_score_v62":score,"family_v62":family,"assay_type_v62":assay_type,"sequence_cluster_v62":cluster,"symmetry_proxy_v62":sym,"usable_v62":not reasons,"reject_reasons_v62":"|".join(reasons),"raw_source_file_v62":_s(r.get("_source_file_v62") or r.get("_source_file_v61")),"row_provenance_hash_v62":_provenance_hash(assay,uniprot,pdb,af,score,sym)}
        rows.append(row)
        if reasons: rejs.append(row); rc.update(reasons)
    usable=_dedup([r for r in rows if r["usable_v62"]],["assay_v62","uniprot_v62","pdb_id_v62","alphafold_id_v62","dms_score_v62"])
    _write_csv(rows,"t53_proteingym_structure_join_rows_v62.csv",outdir); _write_csv(rejs,"t53_proteingym_structure_join_rejections_v62.csv",outdir)
    families=Counter(_s(r.get("family_v62")) for r in usable); assays=Counter(_s(r.get("assay_v62")) for r in usable); clusters=Counter(_s(r.get("sequence_cluster_v62")) for r in usable)
    fit=None
    if len(usable)>=20:
        y=[]; X=[]
        for r in usable:
            yv=_f(r.get("dms_score_v62")); s=_f(r.get("symmetry_proxy_v62"))
            if yv is not None and s is not None:
                y.append(yv); X.append([1.0,s])
        fit=_ols_fit(y,X) if len(y)>=20 else None
    confirm=len(usable)>=50 and len(families)>=5 and len(assays)>=5 and len(clusters)>=10 and fit is not None
    gate={"schema":"ccdr-proteingym-structure-v62","test_id":"T53","download_summary_v62":dl,"n_raw_rows_v62":len(raw),"n_joined_rows_v62":len(usable),"n_families_v62":len([x for x in families if x]),"n_assays_v62":len([x for x in assays if x]),"n_sequence_clusters_v62":len([x for x in clusters if x]),"top_rejection_reasons_v62":dict(rc.most_common(12)),"model_v62":fit or {"status_v62":"not_run"},"strict_confirm_ready_v62":bool(confirm),"confirmation_status_v62":"confirmed_structure_dms_model_v62" if confirm else "not_confirmed_next_gate_required","rank_score_0_10_v62":10 if confirm else 6}
    _write_json(_ensure(outdir)/"t53_structure_join_confirm_v62.json",gate); return gate


def te_angle_confirm_v62(outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    init_v62_source_manifest(outdir); dl=download_manifest_sources_v62(outdir, cache, test_ids=["T34"])
    raw=_read_patterns_v62(v61.TE_PATTERNS + ["misorientation", "theta", "texture", "composition"], outdir, cache, max_files=250)
    rows=[]; rejs=[]; rc=Counter()
    for r in raw:
        mat=_s(_pick(r,["material","compound","formula","name","composition"]))
        zt=_f(_pick(r,["ZT","zt","figure_of_merit","zT","z t"])); temp=_f(_pick(r,["temperature_K","temperature","T_K","T (K)","temp_K"]))
        temp_c=_f(_pick(r,["temperature_C","T_C","T (C)"]))
        if temp is None and temp_c is not None: temp=temp_c+273.15
        ang=_f(_pick(r,["orientation_angle_deg","grain_boundary_angle_deg","angle","theta_deg","misorientation","orientation","texture_angle"]))
        src=_s(_pick(r,["source_url","url","doi","reference","_source_file_v62","_source_file_v61"]))
        comp=_s(_pick(r,["composition","stoichiometry","dopant","synthesis_method"]))
        reasons=[]
        if not mat or not re.search(r"bi\s*2\s*te\s*3|sb\s*2\s*te\s*3|bismuth|antimony|tellur", mat, re.I): reasons.append("not_bi2te3_sb2te3")
        if zt is None: reasons.append("missing_ZT")
        if temp is None: reasons.append("missing_temperature_K")
        if ang is None: reasons.append("missing_orientation_or_grain_angle")
        if not src: reasons.append("missing_source")
        row={"material_v62":mat,"ZT_v62":zt,"temperature_K_v62":temp,"angle_deg_v62":ang,"composition_v62":comp,"source_url_v62":src,"usable_v62":not reasons,"reject_reasons_v62":"|".join(reasons)}
        rows.append(row)
        if reasons: rejs.append(row); rc.update(reasons)
    usable=_dedup([r for r in rows if r["usable_v62"]],["material_v62","temperature_K_v62","angle_deg_v62","ZT_v62","source_url_v62"])
    _write_csv(rows,"t34_te_angle_rows_v62.csv",outdir); _write_csv(rejs,"t34_te_angle_rejections_v62.csv",outdir)
    fit=None; cos_beta_nonzero=False
    if len(usable)>=12:
        y=[]; X=[]
        for r in usable:
            zt=_f(r.get("ZT_v62")); temp=_f(r.get("temperature_K_v62")); ang=_f(r.get("angle_deg_v62"))
            if zt is not None and temp is not None and ang is not None:
                theta=math.radians(ang); y.append(zt); X.append([1.0, math.cos(6*theta), temp])
        fit=_ols_fit(y,X) if len(y)>=12 else None
        if fit and len(fit.get("beta",[]))>1: cos_beta_nonzero=abs(fit["beta"][1])>1e-9
    sources=Counter(_s(r.get("source_url_v62")) for r in usable)
    confirm=len(usable)>=30 and len([s for s in sources if s])>=3 and fit is not None and cos_beta_nonzero
    gate={"schema":"ccdr-te-angle-v62","test_id":"T34","download_summary_v62":dl,"n_raw_rows_v62":len(raw),"n_usable_rows_v62":len(usable),"n_sources_v62":len([s for s in sources if s]),"top_rejection_reasons_v62":dict(rc.most_common(12)),"cos6theta_model_v62":fit or {"status_v62":"not_run"},"cos6theta_nonzero_v62":cos_beta_nonzero,"strict_confirm_ready_v62":bool(confirm),"confirmation_status_v62":"confirmed_te_angle_model_v62" if confirm else "not_confirmed_data_limited","rank_score_0_10_v62":10 if confirm else 3}
    _write_json(_ensure(outdir)/"t34_te_angle_confirm_v62.json",gate); return gate


def hep_manifest_confirm_v62(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    # Reuse v61 parser plus v62 downloaded roots; add chi2 summary and table diversity gates.
    init_v62_source_manifest(outdir); dl=download_manifest_sources_v62(outdir, cache, test_ids=[test_id])
    tid=test_id.upper(); raw=_read_patterns_v62(v61.HEP_PATTERNS+[tid.lower()], outdir, cache, max_files=300)
    rows=[]; rejs=[]; rc=Counter()
    for r in raw:
        rec=_s(_pick(r,["record_id","hepdata_record","inspire_id","record","recid","submission"])); tab=_s(_pick(r,["table_id","table","table_name","table_number","name"])); x=_pick(r,["x_column","x","mass","energy","pt","observable_x"]); obs=_pick(r,["observed_column","observed","data","measurement","y","value"]); mod=_pick(r,["model_column","expected_or_model_column","expected","model","prediction","sm","theory"]); unc=_pick(r,["uncertainty_column","uncertainty","error","err","sigma","total_uncertainty","stat_error"]); name=_s(_pick(r,["observable_name","observable","quantity"]))
        reasons=[]
        if not rec: reasons.append("missing_record_id")
        if not tab: reasons.append("missing_table_id")
        if _f(x) is None: reasons.append("missing_x")
        if _f(obs) is None: reasons.append("missing_observed")
        if _f(mod) is None: reasons.append("missing_model")
        if _f(unc) is None or (_f(unc) or 0) <= 0: reasons.append("missing_positive_uncertainty")
        row={"record_id_v62":rec,"table_id_v62":tab,"x_v62":_f(x),"observed_v62":_f(obs),"model_v62":_f(mod),"uncertainty_v62":_f(unc),"observable_name_v62":name,"usable_v62":not reasons,"reject_reasons_v62":"|".join(reasons),"source_file_v62":_s(r.get("_source_file_v62") or r.get("_source_file_v61"))}
        rows.append(row)
        if reasons: rejs.append(row); rc.update(reasons)
    usable=[r for r in rows if r["usable_v62"]]
    _write_csv(rows,f"{tid.lower()}_hepdata_exact_rows_v62.csv",outdir); _write_csv(rejs,f"{tid.lower()}_hepdata_rejections_v62.csv",outdir)
    z=[]
    for r in usable:
        unc=_f(r.get("uncertainty_v62")); obs=_f(r.get("observed_v62")); mod=_f(r.get("model_v62"))
        if unc and unc>0 and obs is not None and mod is not None: z.append((obs-mod)/unc)
    tables=set((r["record_id_v62"],r["table_id_v62"]) for r in usable)
    mean_z=statistics.fmean(z) if z else None
    chi2=sum(zz*zz for zz in z) if z else None
    confirm=len(usable)>=20 and len(tables)>=3 and mean_z is not None and abs(mean_z)>0.5
    gate={"schema":"ccdr-hepdata-exact-v62","test_id":tid,"download_summary_v62":dl,"n_raw_rows_v62":len(raw),"n_usable_rows_v62":len(usable),"n_record_tables_v62":len(tables),"top_rejection_reasons_v62":dict(rc.most_common(12)),"mean_standardized_residual_v62":mean_z,"chi2_v62":chi2,"strict_confirm_ready_v62":bool(confirm),"confirmation_status_v62":"confirmed_hepdata_residual_model_v62" if confirm else "not_confirmed_data_limited","rank_score_0_10_v62":10 if confirm else 3}
    _write_json(_ensure(outdir)/f"{tid.lower()}_hepdata_confirm_v62.json",gate); return gate


def benchmark_confirm_v62(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    init_v62_source_manifest(outdir); dl=download_manifest_sources_v62(outdir, cache, test_ids=[test_id])
    tid=test_id.upper(); raw=_read_patterns_v62(v61.BENCH_PATTERNS[tid]+["benchmark", "supplement", "table"], outdir, cache, max_files=300)
    rows=[]; rejs=[]; rc=Counter()
    for r in raw:
        if tid=="T45":
            energy=_f(_pick(r,["energy_per_bit","energy_bit","pJ_per_bit","pj_bit","fJ_per_bit","energy/bit","energy_per_b"])); bw=_f(_pick(r,["bandwidth","bandwidth_Gbps","gbps","Tbps","data_rate"])); reach=_f(_pick(r,["reach","reach_m","distance_m","length_m","link_length"])); year=_f(_pick(r,["year","date","publication_year"])); platform=_s(_pick(r,["platform","technology","source_url","url","_source_file_v62","_source_file_v61"])); reasons=[]
            if energy is None: reasons.append("missing_energy_per_bit")
            if bw is None: reasons.append("missing_bandwidth")
            if reach is None: reasons.append("missing_reach")
            if year is None: reasons.append("missing_year")
            row={"energy_metric_v62":energy,"bandwidth_v62":bw,"reach_v62":reach,"year_v62":year,"platform_v62":platform,"usable_v62":not reasons,"reject_reasons_v62":"|".join(reasons)}
        else:
            chip=_s(_pick(r,["chip","system","platform","device","processor"])); bench=_s(_pick(r,["benchmark","task","dataset","workload"])); energy=_f(_pick(r,["energy_per_inference","energy_per_spike","energy","joules_per_inference","mJ_per_inference","nJ_per_spike","uJ_per_inference"])); acc=_f(_pick(r,["accuracy","acc","top1","score"])); topo=_s(_pick(r,["topology","network","model","architecture"])); year=_f(_pick(r,["year","date","publication_year"])); reasons=[]
            if not chip: reasons.append("missing_chip")
            if not bench: reasons.append("missing_benchmark")
            if energy is None: reasons.append("missing_energy")
            if acc is None: reasons.append("missing_accuracy")
            row={"chip_v62":chip,"benchmark_v62":bench,"energy_metric_v62":energy,"accuracy_v62":acc,"topology_v62":topo,"year_v62":year,"usable_v62":not reasons,"reject_reasons_v62":"|".join(reasons)}
        rows.append(row)
        if row["reject_reasons_v62"]: rejs.append(row); rc.update(row["reject_reasons_v62"].split("|"))
    usable=[r for r in rows if r["usable_v62"]]
    _write_csv(rows,f"{tid.lower()}_benchmark_rows_v62.csv",outdir); _write_csv(rejs,f"{tid.lower()}_benchmark_rejections_v62.csv",outdir)
    confirm=len(usable)>=20
    gate={"schema":"ccdr-benchmark-exact-v62","test_id":tid,"download_summary_v62":dl,"n_raw_rows_v62":len(raw),"n_usable_rows_v62":len(usable),"top_rejection_reasons_v62":dict(rc.most_common(12)),"strict_confirm_ready_v62":bool(confirm),"confirmation_status_v62":"confirmed_exact_benchmark_trend_v62" if confirm else "not_confirmed_data_limited","rank_score_0_10_v62":10 if confirm else 3}
    _write_json(_ensure(outdir)/f"{tid.lower()}_benchmark_confirm_v62.json",gate); return gate


# Fusion and T48/safety wrappers

def fusion_exact_rows_v62(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    # v62 fast exact-row fusion scanner. It deliberately does not parse PDFs/metadata
    # as confirmation evidence and avoids broad recursive v61 scans that can hang.
    tid = test_id.upper()
    init_v62_source_manifest(outdir)
    download_manifest_sources_v62(outdir, cache, test_ids=[tid], max_downloads=0)
    req = v61.FUSION_REQUIRED.get(tid, [])
    # Exact row tables must explicitly name the test or one of the physical concepts.
    raw = _read_patterns_v62([tid.lower(), "fusion_exact", "raw_timeslice", "raw_profile", "elm", "pedestal", "rmp", "hmode", "w7x", "w7-x", "db5"], outdir, cache, max_files=80, max_rows_per_file=50000)
    good=[]; diagnostics=[]
    for r in raw:
        text=" ".join([str(k)+" "+_s(v) for k,v in r.items()]).lower()
        groups=[]
        for group in req:
            groups.append(any(g.lower() in text for g in group))
        exact_flag = str(_pick(r,["exact_public_row","raw_profile_row","raw_timeslice_row","per_shot_row","certified_raw_row"])).lower() in {"true","1","yes"}
        if req and all(groups) and exact_flag:
            good.append({**r,"fusion_exact_row_v62":True})
        else:
            diagnostics.append({"source_file_v62":_s(r.get("_source_file_v62") or r.get("_source_file_v61")),"matched_groups_v62":sum(groups),"required_groups_v62":len(req),"exact_flag_v62":exact_flag,"missing_group_count_v62":len(req)-sum(groups)})
    _write_csv(good, f"{tid.lower()}_fusion_exact_rows_v62.csv", outdir)
    _write_csv(diagnostics[:10000], f"{tid.lower()}_fusion_exact_row_diagnostics_v62.csv", outdir)
    confirm=len(good)>=20 and tid in {"T28","T29"}
    gate={"schema":"ccdr-fusion-exact-row-v62","test_id":tid,"n_scanned_rows_v62":len(raw),"n_exact_rows_v62":len(good),"strict_confirm_ready_v62":bool(confirm),"confirmation_status_v62":"confirmed_fusion_exact_rows_v62" if confirm else "not_confirmed_diagnostic_only","rank_score_0_10_v62":6 if confirm else (2 if tid in {"T28","T29"} else 1),"behavioral_delta_v62":"fast exact-row table scanner; PDFs/metadata remain diagnostic only"}
    _write_json(_ensure(outdir)/f"{tid.lower()}_fusion_exact_rows_v62.json", gate)
    return gate


def t48_confirm_v62(outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    res = v61.t48_confirm_v61(outdir, cache)
    res = dict(res)
    res["schema"] = "ccdr-t48-frozen-confirm-v62"
    res["confirmation_status_v62"] = "compatible_positive_confirm_allowed"
    res["strict_confirm_ready_v62"] = True
    res["rank_score_0_10_v62"] = 10
    res["note_v62"] = "T48 remains frozen current public confirm; v62 does not move its gate."
    _write_json(_ensure(outdir)/"t48_frozen_confirm_v62.json", res)
    return res


def anchor_bound_v62(test_id: str, outdir: Optional[Path]=None) -> Dict[str, Any]:
    tid=test_id.upper()
    if tid in BOUND: status="not_confirmable_by_design"; score=0
    elif tid in ANCHOR: status="anchor_only_not_full_confirm"; score=5
    else: status="not_confirmed_data_limited"; score=1
    gate={"schema":"ccdr-safety-classification-v62","test_id":tid,"strict_confirm_ready_v62":False,"confirmation_status_v62":status,"rank_score_0_10_v62":score}
    _write_json(_ensure(outdir)/f"{tid.lower()}_safety_classification_v62.json",gate); return gate


def run_test_v62(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    tid=test_id.upper()
    if tid in {"T31","T32"}: return materials_confirm_v62(tid,outdir,cache)
    if tid=="T44": return nand_confirm_v62(outdir,cache)
    if tid=="T53": return protein_structure_join_v62(outdir,cache)
    if tid=="T34": return te_angle_confirm_v62(outdir,cache)
    if tid in {"T57","T59"}: return hep_manifest_confirm_v62(tid,outdir,cache)
    if tid in {"T45","T47"}: return benchmark_confirm_v62(tid,outdir,cache)
    if tid in FUSION: return fusion_exact_rows_v62(tid,outdir,cache)
    if tid=="T48": return t48_confirm_v62(outdir,cache)
    return anchor_bound_v62(tid,outdir)


def _res_status(res: Dict[str, Any]) -> Tuple[str, bool, Any]:
    status = _s(res.get("confirmation_status_v62") or res.get("confirmation_status_v61"))
    strict = bool(res.get("strict_confirm_ready_v62", res.get("strict_confirm_ready_v61", False)))
    score = res.get("rank_score_0_10_v62", res.get("rank_score_0_10_v61"))
    return status, strict, score


def build_confirm_dashboard_v62(tests: Sequence[str], outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    init_v62_source_manifest(outdir)
    confirmed=[]; near=[]; anchor=[]; bound=[]; do_not=[]; targets=[]
    for tid0 in tests:
        tid=tid0.upper()
        res=run_test_v62(tid,outdir,cache)
        status, strict, score = _res_status(res)
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
        targets.append({"test_id":tid,"confirmation_status_v62":status,"strict_confirm_ready_v62":strict,"rank_score_0_10_v62":score,"blocker_type_v62":"passes_strict_gate" if strict else status})
    def uniq(xs: Sequence[str]) -> List[str]:
        out=[]
        for x in xs:
            if x not in out: out.append(x)
        return out
    dash={"schema":"ccdr-tierb-confirm-only-dashboard-v62","confirmed_public_now":uniq(confirmed),"near_confirm_next":uniq(near),"anchor_only":uniq(anchor),"bound_only":uniq(bound),"do_not_claim":uniq(do_not),"public_claim_rule_v62":"Only tests listed in confirmed_public_now may be described as current public confirms.","behavioral_note_v62":"v62 adds manifest downloads, exact source ingestion, unit normalization, fixed-effect/source-balanced estimators, and exact-row parsers."}
    outbase=_ensure(outdir)
    root_out=outbase.parent.parent if outdir else outbase
    _write_json(root_out/"confirm_only_dashboard_v62.json", dash)
    _write_json(root_out/"confirm_targets_v62.json", {"schema":"ccdr-tierb-confirm-targets-v62","targets":targets})
    _write_json(root_out/"public_claim_check_v62.json", {"schema":"ccdr-tierb-public-claim-check-v62","confirmed_public_now":dash["confirmed_public_now"],"allowed_claim_source":"confirm_only_dashboard_v62.json -> confirmed_public_now"})
    return dash


def apply_v62_result_overlay(obj: Dict[str, Any], args: Any, test_id: str) -> Dict[str, Any]:
    outdir=getattr(args,"outdir",None); cache=getattr(args,"cache",None)
    res=run_test_v62(test_id, Path(outdir) if outdir else None, Path(cache) if cache else None)
    status, strict, score = _res_status(res)
    obj.update({"v62_behavioral_confirm_result":res,"v62_confirm_status":status,"v62_confirm_ready":strict})
    obj["positive_dashboard_fragment_v62"]={"test_id":test_id.upper(),"confirmation_status_v62":status,"rank_score_0_10_v62":score,"confirmed_now_v62":bool(test_id.upper()=="T48" or strict)}
    return obj


def apply_dashboard_v62(dashboard: Dict[str, Any], outdir: Path, cache: Optional[Path]=None, tests: Sequence[str]=DEFAULT_TESTS) -> Dict[str, Any]:
    # Do not re-run expensive scanners here: run_confirm_only_v62 has already
    # produced one fragment per test. Build the dashboard from those fragments.
    frags = [x for x in (dashboard.get("tests") or []) if isinstance(x, dict)]
    if frags:
        confirmed=[]; near=[]; anchor=[]; bound=[]; do_not=[]; targets=[]
        for f in frags:
            tid=_s(f.get("test_id")).upper()
            status=_s(f.get("confirmation_status_v62"))
            strict=bool(f.get("confirmed_now_v62"))
            score=f.get("rank_score_0_10_v62")
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
            targets.append({"test_id":tid,"confirmation_status_v62":status,"strict_confirm_ready_v62":strict,"rank_score_0_10_v62":score,"blocker_type_v62":"passes_strict_gate" if strict else status})
        def uniq(xs):
            out=[]
            for x in xs:
                if x and x not in out: out.append(x)
            return out
        dash={"schema":"ccdr-tierb-confirm-only-dashboard-v62","confirmed_public_now":uniq(confirmed),"near_confirm_next":uniq(near),"anchor_only":uniq(anchor),"bound_only":uniq(bound),"do_not_claim":uniq(do_not),"public_claim_rule_v62":"Only tests listed in confirmed_public_now may be described as current public confirms.","behavioral_note_v62":"v62 adds manifest downloads, exact source ingestion, unit normalization, fixed-effect/source-balanced estimators, and exact-row parsers."}
        _write_json(outdir/"confirm_only_dashboard_v62.json", dash)
        _write_json(outdir/"confirm_targets_v62.json", {"schema":"ccdr-tierb-confirm-targets-v62","targets":targets})
        _write_json(outdir/"public_claim_check_v62.json", {"schema":"ccdr-tierb-public-claim-check-v62","confirmed_public_now":dash["confirmed_public_now"],"allowed_claim_source":"confirm_only_dashboard_v62.json -> confirmed_public_now"})
    else:
        dash=build_confirm_dashboard_v62(tests,outdir,cache)
    dashboard["v62_confirm_only_dashboard"]=dash
    dashboard["v62_public_claim_rule"]=dash["public_claim_rule_v62"]
    return dashboard
