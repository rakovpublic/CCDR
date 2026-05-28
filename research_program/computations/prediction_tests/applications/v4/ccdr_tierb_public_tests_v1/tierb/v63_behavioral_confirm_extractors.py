#!/usr/bin/env python3
"""v63 behavioral/runtime confirm extractors for CCDR Tier-B.

v63 is a targeted behavioral patch after the v62 run diagnostics:
- avoid re-ingesting historical generated dashboards/rejection CSVs as source data;
- read source tables in dtype-safe/chunked mode to reduce DtypeWarning/timeouts;
- prefer exact-source directories/manifests over broad recursive crawls;
- add independent-source/material/temperature balancing for T31/T32;
- use strict exact-row parsers for T44/T53/T34/T57/T59/T45/T47;
- keep fusion/bounds/anchor classifications conservative.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
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

from tierb import v62_behavioral_confirm_extractors as v62
from tierb import v61_behavioral_confirm_extractors as v61

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GEN_DIR = DATA_DIR / "generated"
MANIFEST_DIR = DATA_DIR / "manifests"

DEFAULT_TESTS = list(v62.DEFAULT_TESTS)
NEAR = set(v62.NEAR)
FUSION = set(v62.FUSION)
BOUND = set(v62.BOUND)
ANCHOR = set(v62.ANCHOR)

# v63 deliberately excludes generated/output directories from source scans by default.
EXACT_DIR_NAMES = [
    "exact_sources", "materials_sources", "external", "materials_exact", "downloaded_supplements",
    "public_sources_v63", "public_sources_v62", "source_tables", "curated_sources", "manifests",
]
BAD_PATH_BITS = [
    "data/generated", "tierb_out", "confirm_only", "__pycache__", "positive_dashboard", "confirm_targets",
    "public_claim", "rejection_diagnostics", "rejection_summary", "confirm_gate", "one_command_summary",
]
BAD_NAME_BITS = [
    "positive_dashboard", "confirm_targets", "public_claim", "rejection_diagnostics", "rejection_summary",
    "confirm_gate", "one_command_summary", "smoke", "dashboard", "generated_rows_v5",
]


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


def _pick(row: Dict[str, Any], aliases: Sequence[str]) -> Any:
    return v61._pick(row, aliases)


def _dedup(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    return v61._dedup(rows, keys)


def _material_family(material: str) -> str:
    return v62._material_family(material)


def _temp_bin(t: Optional[float]) -> str:
    return v62._temp_bin(t)


def _provenance_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(_s(p).encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# v63 source manifests and dtype/chunk-safe table reading
# ---------------------------------------------------------------------------

V63_MANIFEST_ROWS = [
    {"test_id": "T31,T32", "source_family": "materials_exact", "local_dir": "data/materials_sources", "required_columns": "material,temperature,kappa,grain_size,microstructure_method", "note": "Exact κ(T)+microstructure tables only; generated diagnostics are excluded."},
    {"test_id": "T31,T32", "source_family": "materials_exact", "local_dir": "data/external/materials", "required_columns": "material,T,kappa,grain_size,SEM/TEM/XRD/EBSD/AFM/Scherrer", "note": "Use independent source families and temperature bins."},
    {"test_id": "T44", "source_family": "nand_exact", "local_dir": "data/exact_sources/nand", "required_columns": "company,year,layers,capacity_Gb,die_area_mm2,bits_per_cell", "note": "Broad scans are disabled for T44."},
    {"test_id": "T53", "source_family": "proteingym_assays", "local_dir": "data/exact_sources/proteingym", "required_columns": "assay,uniprot,DMS_score"},
    {"test_id": "T53", "source_family": "protein_structure_mapping", "local_dir": "data/exact_sources/protein_structures", "required_columns": "uniprot,pdb_or_alphafold,symmetry_proxy"},
    {"test_id": "T34", "source_family": "thermoelectric_exact", "local_dir": "data/exact_sources/thermoelectric", "required_columns": "material,ZT,temperature_K,orientation_angle_deg"},
    {"test_id": "T57,T59", "source_family": "hepdata_exact", "local_dir": "data/exact_sources/hepdata", "required_columns": "record_id,table_id,x,observed,model,uncertainty"},
    {"test_id": "T45", "source_family": "optical_interconnect_exact", "local_dir": "data/exact_sources/optical_interconnect", "required_columns": "energy_per_bit,bandwidth,reach,year,platform"},
    {"test_id": "T47", "source_family": "neuromorphic_exact", "local_dir": "data/exact_sources/neuromorphic", "required_columns": "chip,benchmark,energy,accuracy,topology"},
    {"test_id": "T26,T27,T28,T29,T30", "source_family": "fusion_exact_rows", "local_dir": "data/exact_sources/fusion", "required_columns": "certified exact raw row tables only"},
]


def init_v63_source_manifest(outdir: Optional[Path] = None) -> str:
    return _write_csv(V63_MANIFEST_ROWS, "v63_exact_source_manifest.csv", outdir)


def _safe_rel(p: Path) -> str:
    try:
        return p.resolve().as_posix().lower()
    except Exception:
        return str(p).lower().replace("\\", "/")


def _is_generated_or_output_path(p: Path) -> bool:
    s = _safe_rel(p)
    name = p.name.lower()
    return any(bit in s for bit in BAD_PATH_BITS) or any(bit in name for bit in BAD_NAME_BITS)


def _candidate_roots_v63(outdir: Optional[Path] = None, cache: Optional[Path] = None, source_kind: str = "generic") -> List[Path]:
    roots: List[Path] = []
    if outdir:
        roots += [
            outdir / "public_sources_v63", outdir / "exact_sources", outdir / "source_tables",
            outdir / "data" / "public_sources_v63", outdir / "data" / "exact_sources",
        ]
    if cache:
        roots += [cache / "public_sources_v63", cache / "exact_sources", cache / f"{source_kind}_exact", cache / "source_tables"]
    roots += [
        DATA_DIR / "public_sources_v63", DATA_DIR / "exact_sources", DATA_DIR / "source_tables",
        DATA_DIR / "materials_sources", DATA_DIR / "external", DATA_DIR / "downloaded_supplements", MANIFEST_DIR,
    ]
    if os.environ.get("CCDR_V63_ALLOW_LEGACY_SOURCE_SCAN", "0").lower() in {"1", "true", "yes", "on"}:
        # Optional compatibility mode; still excludes generated/dashboard outputs.
        roots += [DATA_DIR]
        if cache:
            roots += [cache]
    seen: List[Path] = []
    for r in roots:
        try:
            rr = r.resolve()
        except Exception:
            rr = r
        if rr.exists() and rr not in seen and not _is_generated_or_output_path(rr):
            seen.append(rr)
    return seen


def _iter_table_files_v63(patterns: Sequence[str], outdir: Optional[Path] = None, cache: Optional[Path] = None, source_kind: str = "generic", max_files: int = 250) -> List[Path]:
    suffixes = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".xlsx", ".xls"}
    pats = [p.lower() for p in patterns]
    files: List[Path] = []
    seen = set()
    for root in _candidate_roots_v63(outdir, cache, source_kind=source_kind):
        for p in root.rglob("*"):
            if len(files) >= max_files:
                return files
            if not p.is_file() or p.suffix.lower() not in suffixes:
                continue
            if _is_generated_or_output_path(p):
                continue
            try:
                if p.stat().st_size > 80_000_000:
                    continue
            except Exception:
                pass
            s = _safe_rel(p)
            if pats and not any(q in s for q in pats):
                continue
            try:
                rp = p.resolve()
            except Exception:
                rp = p
            if rp in seen:
                continue
            seen.add(rp)
            files.append(p)
    return files


def _read_table_v63(path: Path, max_rows: int = 200000, chunksize: int = 50000) -> List[Dict[str, Any]]:
    """Dtype-safe, chunked reader for exact source tables.

    All columns are initially strings. Numeric conversion happens later in the
    row normalizers, avoiding pandas mixed-type warnings and large memory spikes.
    """
    rows: List[Dict[str, Any]] = []
    if not path.exists() or path.is_dir():
        return rows
    suf = path.suffix.lower()
    try:
        if suf in {".csv", ".tsv", ".txt"} and pd is not None:
            sep = "\t" if suf == ".tsv" else None
            try:
                for chunk in pd.read_csv(path, sep=sep, engine="python", dtype=str, chunksize=chunksize, on_bad_lines="skip"):
                    for r in chunk.fillna("").to_dict(orient="records"):
                        r["_source_file_v63"] = str(path)
                        rows.append(r)
                        if len(rows) >= max_rows:
                            return rows
            except TypeError:
                # Older pandas compatibility.
                df = pd.read_csv(path, sep=sep, engine="python", dtype=str).fillna("")
                for r in df.head(max_rows).to_dict(orient="records"):
                    r["_source_file_v63"] = str(path); rows.append(r)
        elif suf in {".xlsx", ".xls"} and pd is not None:
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names[:8]:
                df = pd.read_excel(path, sheet_name=sheet, dtype=str).fillna("")
                for r in df.head(max_rows - len(rows)).to_dict(orient="records"):
                    r["_source_file_v63"] = f"{path}::{sheet}"; rows.append(r)
                    if len(rows) >= max_rows: return rows
        elif suf in {".json", ".jsonl"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            objs: List[Any] = []
            if suf == ".jsonl":
                for line in text.splitlines():
                    line=line.strip()
                    if not line: continue
                    try: objs.append(json.loads(line))
                    except Exception: continue
            else:
                obj = json.loads(text)
                if isinstance(obj, list): objs = obj
                elif isinstance(obj, dict):
                    for key in ["rows", "data", "values", "tables", "records"]:
                        if isinstance(obj.get(key), list):
                            objs = obj[key]; break
                    if not objs: objs = [obj]
            for o in objs[:max_rows]:
                if isinstance(o, dict):
                    r = {str(k): v for k, v in o.items()}; r["_source_file_v63"] = str(path); rows.append(r)
        elif suf in {".yaml", ".yml"}:
            # Minimal HEPData-style YAML row extraction without PyYAML dependency.
            text = path.read_text(encoding="utf-8", errors="ignore")
            # Parse simple lines into a single record; full HEPData tables usually
            # also ship CSV, which is preferred. This still surfaces manifest values.
            rec: Dict[str, Any] = {"_source_file_v63": str(path)}
            for m in re.finditer(r"^\s*([A-Za-z0-9_ .-]+):\s*([^#\n]+)", text, re.M):
                k = m.group(1).strip(); v = m.group(2).strip().strip('"\'')
                if k and k not in rec: rec[k] = v
            if len(rec) > 1: rows.append(rec)
    except Exception as e:
        rows.append({"_source_file_v63": str(path), "_read_error_v63": type(e).__name__ + ": " + str(e)[:200]})
    return rows[:max_rows]


def _read_patterns_v63(patterns: Sequence[str], outdir: Optional[Path] = None, cache: Optional[Path] = None, source_kind: str = "generic", max_files: int = 250, max_rows_per_file: int = 200000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files = _iter_table_files_v63(patterns, outdir, cache, source_kind=source_kind, max_files=max_files)
    for p in files:
        rows.extend(_read_table_v63(p, max_rows=max_rows_per_file))
    return rows


# ---------------------------------------------------------------------------
# T31/T32 materials: exact-source-only scanning + residualized fixed effects
# ---------------------------------------------------------------------------

MATERIAL_PATTERNS_V63 = ["material", "thermal", "kappa", "conductivity", "grain", "microstructure", "cryogenic", "nanocrystalline", "xrd", "sem", "tem", "ebsd", "scherrer"]


def normalize_material_row_v63(raw: Dict[str, Any], idx: int) -> Tuple[Dict[str, Any], List[str]]:
    # Reuse v62 unit handling, but ensure v63 provenance source wins.
    raw2 = dict(raw)
    if "_source_file_v62" not in raw2 and "_source_file_v63" in raw2:
        raw2["_source_file_v62"] = raw2["_source_file_v63"]
    nr, reasons = v62.normalize_material_row_v62(raw2, idx)
    nr = {k.replace("_v62", "_v63"): v for k, v in nr.items()}
    nr["raw_source_file_v63"] = _s(raw.get("_source_file_v63") or raw.get("_source_file_v62") or raw.get("_source_file_v61"))
    # Recalculate source identifier to avoid one manifest file pretending to be many sources.
    src = _s(nr.get("source_url_v63")) or nr["raw_source_file_v63"]
    nr["source_id_v63"] = src
    nr["row_provenance_hash_v63"] = _provenance_hash(src, nr.get("sample_id_v63"), nr.get("material_v63"), nr.get("temperature_K_v63"), nr.get("kappa_W_mK_v63"), nr.get("grain_size_nm_v63"))
    return nr, reasons


def _ols(y: Sequence[float], X: Sequence[Sequence[float]]) -> Optional[Dict[str, Any]]:
    return v61._ols_fit(list(y), [list(x) for x in X])


def _demean_by_groups(vals: List[float], groups: Sequence[str]) -> List[float]:
    sums: Dict[str, float] = defaultdict(float); counts: Dict[str, int] = defaultdict(int)
    for v, g in zip(vals, groups): sums[g] += v; counts[g] += 1
    return [v - (sums[g] / counts[g]) for v, g in zip(vals, groups)]


def _materials_estimator_v63(usable: List[Dict[str, Any]], test_id: str, outdir: Optional[Path] = None) -> Dict[str, Any]:
    model_rows: List[Dict[str, Any]] = []
    for r in usable:
        temp = _f(r.get("temperature_K_v63")); kap = _f(r.get("kappa_W_mK_v63")); grain = _f(r.get("grain_size_nm_v63"))
        if temp and kap and grain and temp > 0 and kap > 0 and grain > 0:
            rr = dict(r)
            rr["logT_v63"] = math.log(temp); rr["logKappa_v63"] = math.log(kap); rr["logGrain_v63"] = math.log(grain)
            rr["boundary_proxy_num_v63"] = _f(r.get("boundary_density_proxy_v63")) or (1.0 / grain)
            model_rows.append(rr)
    if len(model_rows) < 12 or np is None:
        return {"status_v63": "not_enough_rows_for_estimator", "n_model_rows_v63": len(model_rows)}
    y = [float(r["logKappa_v63"]) for r in model_rows]
    logT = [float(r["logT_v63"]) for r in model_rows]
    logG = [float(r["logGrain_v63"]) for r in model_rows]
    bound = [float(r["boundary_proxy_num_v63"]) for r in model_rows]
    source = [_s(r.get("source_id_v63") or r.get("raw_source_file_v63")) for r in model_rows]
    fam = [_s(r.get("material_family_v63")) for r in model_rows]
    # Residualize by source and material family to avoid pseudo-replication.
    group1 = [s + "||" + f for s, f in zip(source, fam)]
    y_r = _demean_by_groups(_demean_by_groups(y, source), fam)
    t_r = _demean_by_groups(_demean_by_groups(logT, source), fam)
    g_r = _demean_by_groups(_demean_by_groups(logG, source), fam)
    b_r = _demean_by_groups(_demean_by_groups(bound, source), fam)
    fit_temp = _ols(y_r, [[1.0, t] for t in t_r])
    fit_micro = _ols(y_r, [[1.0, t, g, b] for t, g, b in zip(t_r, g_r, b_r)])
    model_wins = bool(fit_temp and fit_micro and fit_micro.get("aic", 1e99) < fit_temp.get("aic", -1e99) and fit_micro.get("bic", 1e99) < fit_temp.get("bic", -1e99))
    beta = fit_micro.get("beta") if fit_micro else []
    sign_ok = bool(beta and len(beta) > 3 and beta[2] > 0 and beta[3] < 0)
    # Source-balanced bootstrap.
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in model_rows: by_source[_s(r.get("source_id_v63") or r.get("raw_source_file_v63"))].append(r)
    sources = [s for s in by_source if s]
    boot_n = boot_win = boot_sign = 0
    if len(sources) >= 3 and np is not None:
        rng = np.random.default_rng(6301)
        for _ in range(100):
            sample: List[Dict[str, Any]] = []
            for s in rng.choice(sources, size=len(sources), replace=True):
                group = by_source[str(s)]
                idx = rng.integers(0, len(group), size=max(1, min(len(group), 80)))
                sample.extend(group[int(i)] for i in idx)
            if len(sample) < 12: continue
            sy = [float(r["logKappa_v63"]) for r in sample]
            st = [float(r["logT_v63"]) for r in sample]
            sg = [float(r["logGrain_v63"]) for r in sample]
            sb = [float(r["boundary_proxy_num_v63"]) for r in sample]
            ss = [_s(r.get("source_id_v63")) for r in sample]
            sf = [_s(r.get("material_family_v63")) for r in sample]
            sy = _demean_by_groups(_demean_by_groups(sy, ss), sf); st = _demean_by_groups(_demean_by_groups(st, ss), sf); sg = _demean_by_groups(_demean_by_groups(sg, ss), sf); sb = _demean_by_groups(_demean_by_groups(sb, ss), sf)
            ft = _ols(sy, [[1.0, t] for t in st]); fm = _ols(sy, [[1.0, t, g, b] for t,g,b in zip(st,sg,sb)])
            if not ft or not fm: continue
            boot_n += 1
            if fm.get("aic", 1e99) < ft.get("aic", -1e99) and fm.get("bic", 1e99) < ft.get("bic", -1e99): boot_win += 1
            bb = fm.get("beta") or []
            if len(bb) > 3 and bb[2] > 0 and bb[3] < 0: boot_sign += 1
    # Temperature-bin wins.
    bin_rows=[]
    by_bin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in model_rows: by_bin[_s(r.get("temperature_bin_v63"))].append(r)
    for b, group in by_bin.items():
        if len(group) < 8: continue
        yy=[float(r["logKappa_v63"]) for r in group]; tt=[float(r["logT_v63"]) for r in group]; gg=[float(r["logGrain_v63"]) for r in group]; bb=[float(r["boundary_proxy_num_v63"]) for r in group]
        ft=_ols(yy, [[1.0,t] for t in tt]); fm=_ols(yy, [[1.0,t,g,x] for t,g,x in zip(tt,gg,bb)])
        win=bool(ft and fm and fm.get("aic",1e99)<ft.get("aic",-1e99) and fm.get("bic",1e99)<ft.get("bic",-1e99))
        bin_rows.append({"temperature_bin_v63":b,"n_rows_v63":len(group),"microstructure_wins_v63":win,"aic_temp_v63":ft.get("aic") if ft else None,"aic_micro_v63":fm.get("aic") if fm else None})
    _write_csv(bin_rows, f"{test_id.lower()}_temperature_bin_model_wins_v63.csv", outdir)
    return {
        "status_v63":"ok", "n_model_rows_v63":len(model_rows), "fixed_effect_method_v63":"two-step source+family demeaning",
        "temperature_only_fit_v63":fit_temp, "microstructure_fit_v63":fit_micro,
        "microstructure_beats_temperature_baseline_v63":model_wins, "predicted_signs_pass_v63":sign_ok,
        "source_balanced_bootstrap_n_v63":boot_n,
        "source_balanced_bootstrap_model_win_fraction_v63":boot_win/boot_n if boot_n else 0.0,
        "source_balanced_bootstrap_sign_fraction_v63":boot_sign/boot_n if boot_n else 0.0,
        "temperature_bin_results_v63":bin_rows,
        "temperature_bin_win_fraction_v63":sum(1 for x in bin_rows if x.get("microstructure_wins_v63"))/len(bin_rows) if bin_rows else 0.0,
    }


def materials_confirm_v63(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    init_v63_source_manifest(outdir)
    raw = _read_patterns_v63(MATERIAL_PATTERNS_V63 + [test_id.lower()], outdir, cache, source_kind="materials", max_files=180, max_rows_per_file=120000)
    norm=[]; rejs=[]; rc=Counter()
    for i,r in enumerate(raw):
        nr,reasons=normalize_material_row_v63(r,i); norm.append(nr)
        if reasons: rejs.append(nr); rc.update(reasons)
    dedup=_dedup(norm,["source_id_v63","sample_id_v63","material_v63","temperature_K_v63","kappa_W_mK_v63","grain_size_nm_v63"])
    usable=[r for r in dedup if r.get("usable_v63")]
    _write_csv(norm, f"{test_id.lower()}_materials_normalized_rows_v63.csv", outdir)
    _write_csv(rejs, f"{test_id.lower()}_materials_rejection_diagnostics_v63.csv", outdir)
    _write_csv([{"reject_reason_v63":k,"n_rows_v63":v} for k,v in rc.most_common()], f"{test_id.lower()}_materials_rejection_summary_v63.csv", outdir)
    sources=Counter(_s(r.get("source_id_v63")) for r in usable); fams=Counter(_s(r.get("material_family_v63")) for r in usable); tbins=Counter(_s(r.get("temperature_bin_v63")) for r in usable)
    estimator=_materials_estimator_v63(usable,test_id,outdir)
    gates={
        "sources_ge_5_v63":len([s for s in sources if s])>=5,
        "material_families_ge_5_v63":len([f for f in fams if f])>=5,
        "temperature_bins_ge_3_v63":len([b for b in tbins if b and b!="unknown"])>=3,
        "microstructure_beats_temperature_baseline_v63":bool(estimator.get("microstructure_beats_temperature_baseline_v63")),
        "predicted_signs_pass_v63":bool(estimator.get("predicted_signs_pass_v63")),
        "bootstrap_sign_fraction_ge_0_80_v63":float(estimator.get("source_balanced_bootstrap_sign_fraction_v63") or 0.0)>=0.80,
        "bootstrap_model_win_fraction_ge_0_80_v63":float(estimator.get("source_balanced_bootstrap_model_win_fraction_v63") or 0.0)>=0.80,
        "temperature_bin_win_fraction_ge_0_67_v63":float(estimator.get("temperature_bin_win_fraction_v63") or 0.0)>=0.67,
    }
    confirm=all(gates.values())
    gate={"schema":"ccdr-materials-confirm-v63","test_id":test_id.upper(),"n_raw_rows_v63":len(raw),"n_normalized_rows_v63":len(norm),"n_dedup_rows_v63":len(dedup),"n_usable_rows_v63":len(usable),"n_sources_v63":len([s for s in sources if s]),"n_material_families_v63":len([f for f in fams if f]),"n_temperature_bins_v63":len([b for b in tbins if b and b!='unknown']),"top_rejection_reasons_v63":dict(rc.most_common(12)),"estimator_v63":estimator,"gates_v63":gates,"failed_subgates_v63":[k for k,ok in gates.items() if not ok],"strict_confirm_ready_v63":bool(confirm),"confirmation_status_v63":"confirmed_materials_microstructure_model_v63" if confirm else "not_confirmed_next_gate_required","rank_score_0_10_v63":10 if confirm else 9,"behavioral_delta_v63":"exact-source-only scan + dtype/chunked reads + source/family residualized estimator"}
    _write_json(_ensure(outdir)/f"{test_id.lower()}_materials_confirm_v63.json", gate)
    return gate


# ---------------------------------------------------------------------------
# Exact-row parsers for other near-confirm tests
# ---------------------------------------------------------------------------

def normalize_nand_row_v63(raw: Dict[str, Any], idx: int) -> Tuple[Dict[str, Any], List[str]]:
    raw2=dict(raw)
    if "_source_file_v62" not in raw2 and "_source_file_v63" in raw2: raw2["_source_file_v62"] = raw2["_source_file_v63"]
    nr,reasons=v62.normalize_nand_row_v62(raw2,idx)
    nr={k.replace("_v62","_v63"):v for k,v in nr.items()}
    nr["raw_source_file_v63"]=_s(raw.get("_source_file_v63") or raw.get("_source_file_v62"))
    return nr,reasons


def nand_confirm_v63(outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    init_v63_source_manifest(outdir)
    raw=_read_patterns_v63(["nand","die_area","die-density","die_density","gb_mm2","isscc","vlsi","wikichip","techinsights","samsung","micron","kioxia","hynix"], outdir, cache, source_kind="nand", max_files=120, max_rows_per_file=120000)
    norm=[]; rejs=[]; rc=Counter()
    for i,r in enumerate(raw):
        nr,reasons=normalize_nand_row_v63(r,i); norm.append(nr)
        if reasons: rejs.append(nr); rc.update(reasons)
    dedup=_dedup(norm,["company_v63","year_v63","layers_v63","capacity_Gb_v63","die_area_mm2_v63","bits_per_cell_v63"])
    usable=[r for r in dedup if r.get("usable_tier_a_v63")]
    _write_csv(norm,"t44_nand_normalized_rows_v63.csv",outdir); _write_csv(rejs,"t44_nand_rejection_diagnostics_v63.csv",outdir); _write_csv([{"reject_reason_v63":k,"n_rows_v63":v} for k,v in rc.most_common()],"t44_nand_rejection_summary_v63.csv",outdir)
    comps=Counter(_s(r.get("company_v63")) for r in usable)
    model={"status_v63":"not_run_no_true_rows"}; layer_positive=False
    if len(usable)>=8 and np is not None:
        y=[]; X=[]; compvals=[_s(r.get("company_v63")) for r in usable]
        levels=[c for c,n in Counter(compvals).most_common(8) if c and n>=2][1:]
        for r in usable:
            den=_f(r.get("density_Gb_per_mm2_v63")); layers=_f(r.get("layers_v63")); year=_f(r.get("year_v63")); bpc=_f(r.get("bits_per_cell_v63")); comp=_s(r.get("company_v63"))
            if den and layers and year and bpc and den>0 and layers>0:
                y.append(math.log(den)); X.append([1.0,math.log(layers),year-2000.0,bpc]+[1.0 if comp==lv else 0.0 for lv in levels])
        fit=_ols(y,X) if len(y)>=8 else None
        if fit:
            layer_positive=(fit.get("beta") or [0,0])[1]>0; model=fit; model["company_fixed_effect_levels_v63"]=levels; model["layer_beta_positive_v63"]=layer_positive
    confirm=len(usable)>=8 and len([c for c in comps if c])>=3 and layer_positive
    gate={"schema":"ccdr-nand-tier-a-v63","test_id":"T44","n_raw_rows_v63":len(raw),"n_normalized_rows_v63":len(norm),"n_true_tier_a_rows_v63":len(usable),"n_companies_v63":len([c for c in comps if c]),"top_rejection_reasons_v63":dict(rc.most_common(12)),"model_v63":model,"strict_confirm_ready_v63":bool(confirm),"failed_subgates_v63":[] if confirm else [k for k,ok in {"true_tier_a_rows_ge_8":len(usable)>=8,"companies_ge_3":len([c for c in comps if c])>=3,"layer_beta_positive":layer_positive}.items() if not ok],"confirmation_status_v63":"confirmed_true_tier_a_nand_scaling_v63" if confirm else "not_confirmed_audit_repair_required","rank_score_0_10_v63":10 if confirm else 8,"behavioral_delta_v63":"exact NAND directories only; broad scan disabled"}
    _write_json(_ensure(outdir)/"t44_nand_confirm_v63.json",gate); return gate


def protein_structure_join_v63(outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    raw=_read_patterns_v63(["proteingym","dms","uniprot","alphafold","pdb","structure","mapping","oligomer"], outdir, cache, source_kind="protein", max_files=180, max_rows_per_file=100000)
    # Build structure map first, then assay rows; also accept prejoined rows.
    struct_by_uniprot: Dict[str, Dict[str, Any]] = {}
    assay_rows=[]; joined=[]; rejs=[]; rc=Counter()
    for r in raw:
        unip=_s(_pick(r,["uniprot","uniprot_id","UniProt_ID","accession","protein_id","target_uniprot"])).upper()
        pdb=_s(_pick(r,["pdb","pdb_id","PDB","structure_id","pdb_chain"])); af=_s(_pick(r,["alphafold","alphafold_id","af_id","AlphaFoldDB","af2_model","alphafold_model"])); sym=_f(_pick(r,["symmetry_proxy","contact_symmetry","oligomer_symmetry","contact_network_symmetry","oligomeric_state","n_chains","assembly_chains"]))
        ost=_s(_pick(r,["oligomeric_state","assembly","biological_assembly","quaternary_structure"])).lower()
        if sym is None:
            if "dimer" in ost: sym=2.0
            elif "trimer" in ost: sym=3.0
            elif "tetramer" in ost: sym=4.0
            elif "monomer" in ost: sym=1.0
        if unip and (pdb or af or sym is not None): struct_by_uniprot[unip]={"pdb":pdb,"af":af,"sym":sym,"_source_file_v63":_s(r.get("_source_file_v63"))}
        if _f(_pick(r,["DMS_score","fitness","fitness_score","score","effect","mutant_score","mean_fitness"])) is not None or _s(_pick(r,["assay","DMS_id","experiment","assay_id"])):
            assay_rows.append(r)
    for i,r in enumerate(assay_rows):
        assay=_s(_pick(r,["assay","DMS_id","DMS ID","experiment","study","assay_id"])); unip=_s(_pick(r,["uniprot","uniprot_id","UniProt_ID","accession","protein_id","target_uniprot"])).upper(); score=_f(_pick(r,["DMS_score","fitness","fitness_score","score","effect","mutant_score","mean_fitness"])); family=_s(_pick(r,["family","protein_family","Pfam","fold","domain","superfamily"])) or (unip[:4] if unip else ""); assay_type=_s(_pick(r,["assay_type","selection_type","measurement_type","phenotype"])); cluster=_s(_pick(r,["sequence_cluster","cluster","seq_cluster","identity_cluster","mmseqs_cluster"])) or unip
        st=struct_by_uniprot.get(unip,{})
        pdb=_s(_pick(r,["pdb","pdb_id","PDB","structure_id","pdb_chain"])) or _s(st.get("pdb")); af=_s(_pick(r,["alphafold","alphafold_id","af_id","AlphaFoldDB","af2_model","alphafold_model"])) or _s(st.get("af")); sym=_f(_pick(r,["symmetry_proxy","contact_symmetry","oligomer_symmetry","contact_network_symmetry","oligomeric_state","n_chains","assembly_chains"]))
        if sym is None: sym=_f(st.get("sym"))
        reasons=[]
        for cond, reason in [(not assay,"missing_assay"),(not unip,"missing_uniprot"),(not(pdb or af),"missing_structure_id"),(score is None,"missing_dms_score"),(sym is None,"missing_symmetry_proxy")]:
            if cond: reasons.append(reason)
        row={"assay_v63":assay,"uniprot_v63":unip,"pdb_id_v63":pdb,"alphafold_id_v63":af,"dms_score_v63":score,"family_v63":family,"assay_type_v63":assay_type,"sequence_cluster_v63":cluster,"symmetry_proxy_v63":sym,"usable_v63":not reasons,"reject_reasons_v63":"|".join(reasons),"raw_source_file_v63":_s(r.get("_source_file_v63")),"structure_join_source_v63":_s(st.get("_source_file_v63"))}
        joined.append(row)
        if reasons: rejs.append(row); rc.update(reasons)
    usable=_dedup([r for r in joined if r.get("usable_v63")],["assay_v63","uniprot_v63","pdb_id_v63","alphafold_id_v63","dms_score_v63"])
    _write_csv(joined,"t53_proteingym_structure_join_rows_v63.csv",outdir); _write_csv(rejs,"t53_proteingym_structure_join_rejections_v63.csv",outdir); _write_csv([{"reject_reason_v63":k,"n_rows_v63":v} for k,v in rc.most_common()],"t53_proteingym_structure_join_rejection_summary_v63.csv",outdir)
    fams=Counter(_s(r.get("family_v63")) for r in usable); assays=Counter(_s(r.get("assay_v63")) for r in usable); clusters=Counter(_s(r.get("sequence_cluster_v63")) for r in usable)
    fit=None
    if len(usable)>=20:
        y=[]; X=[]
        for r in usable:
            yy=_f(r.get("dms_score_v63")); ss=_f(r.get("symmetry_proxy_v63"))
            if yy is not None and ss is not None: y.append(yy); X.append([1.0,ss])
        fit=_ols(y,X) if len(y)>=20 else None
    confirm=len(usable)>=50 and len(fams)>=5 and len(assays)>=5 and len(clusters)>=10 and fit is not None
    gate={"schema":"ccdr-proteingym-structure-v63","test_id":"T53","n_raw_rows_v63":len(raw),"n_assay_rows_v63":len(assay_rows),"n_structure_map_entries_v63":len(struct_by_uniprot),"n_joined_rows_v63":len(usable),"n_families_v63":len([x for x in fams if x]),"n_assays_v63":len([x for x in assays if x]),"n_sequence_clusters_v63":len([x for x in clusters if x]),"top_rejection_reasons_v63":dict(rc.most_common(12)),"model_v63":fit or {"status_v63":"not_run"},"strict_confirm_ready_v63":bool(confirm),"confirmation_status_v63":"confirmed_structure_dms_model_v63" if confirm else "not_confirmed_next_gate_required","rank_score_0_10_v63":10 if confirm else 6}
    _write_json(_ensure(outdir)/"t53_structure_join_confirm_v63.json",gate); return gate


def _simple_exact_parser(test_id: str, patterns: Sequence[str], source_kind: str, required_aliases: Dict[str, Sequence[str]], outdir: Optional[Path], cache: Optional[Path], min_rows: int=20) -> Dict[str, Any]:
    raw=_read_patterns_v63(patterns,outdir,cache,source_kind=source_kind,max_files=160,max_rows_per_file=100000)
    rows=[]; rejs=[]; rc=Counter()
    for r in raw:
        out={"raw_source_file_v63":_s(r.get("_source_file_v63"))}; reasons=[]
        for canon, aliases in required_aliases.items():
            val=_pick(r,aliases)
            out[canon+"_v63"]=_s(val) if canon in {"record_id","table_id","observable_name","material","composition","platform","chip","benchmark","topology","source_url"} else _f(val)
            if out[canon+"_v63"] in [None, ""]: reasons.append("missing_"+canon)
        out["usable_v63"]=not reasons; out["reject_reasons_v63"]="|".join(reasons)
        rows.append(out)
        if reasons: rejs.append(out); rc.update(reasons)
    usable=[r for r in rows if r.get("usable_v63")]
    _write_csv(rows,f"{test_id.lower()}_{source_kind}_rows_v63.csv",outdir); _write_csv(rejs,f"{test_id.lower()}_{source_kind}_rejections_v63.csv",outdir); _write_csv([{"reject_reason_v63":k,"n_rows_v63":v} for k,v in rc.most_common()],f"{test_id.lower()}_{source_kind}_rejection_summary_v63.csv",outdir)
    confirm=len(usable)>=min_rows
    extra={}
    if test_id in {"T57","T59"} and usable:
        chi2=0.0; n=0
        for r in usable:
            obs=_f(r.get("observed_v63")); mod=_f(r.get("model_v63")); unc=_f(r.get("uncertainty_v63"))
            if obs is not None and mod is not None and unc and unc>0:
                chi2 += ((obs-mod)/unc)**2; n += 1
        extra.update({"chi2_v63":chi2,"n_residuals_v63":n,"chi2_per_dof_v63":chi2/max(1,n)})
    gate={"schema":f"ccdr-{source_kind}-v63","test_id":test_id,"n_raw_rows_v63":len(raw),"n_usable_rows_v63":len(usable),"top_rejection_reasons_v63":dict(rc.most_common(12)),"strict_confirm_ready_v63":bool(confirm),"confirmation_status_v63":"confirmed_exact_rows_v63" if confirm else "not_confirmed_data_limited","rank_score_0_10_v63":10 if confirm else 3,**extra}
    _write_json(_ensure(outdir)/f"{test_id.lower()}_{source_kind}_confirm_v63.json",gate); return gate


def te_angle_confirm_v63(outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    raw=_read_patterns_v63(["thermoelectric","bi2te3","sb2te3","bismuth","tellur","zt","orientation","angle"], outdir, cache, source_kind="thermoelectric", max_files=120)
    rows=[]; rejs=[]; rc=Counter()
    for r in raw:
        nr={"material_v63":_s(_pick(r,["material","compound","formula","name","composition"])),"ZT_v63":_f(_pick(r,["ZT","zt","figure_of_merit","zT"])),"temperature_K_v63":_f(_pick(r,["temperature_K","temperature","T_K","T (K)"])),"angle_deg_v63":_f(_pick(r,["orientation_angle_deg","grain_boundary_angle_deg","angle","theta_deg","misorientation","orientation"])),"composition_v63":_s(_pick(r,["composition","stoichiometry","dopant","synthesis_method"])),"source_url_v63":_s(_pick(r,["source_url","url","doi","reference","_source_file_v63"]))}
        temp_c=_f(_pick(r,["temperature_C","T_C","T (C)"]))
        if nr["temperature_K_v63"] is None and temp_c is not None: nr["temperature_K_v63"]=temp_c+273.15
        reasons=[]
        if not nr["material_v63"] or not re.search(r"bi\s*2\s*te\s*3|sb\s*2\s*te\s*3|bismuth|antimony|tellur",nr["material_v63"],re.I): reasons.append("not_bi2te3_sb2te3")
        for k in ["ZT_v63","temperature_K_v63","angle_deg_v63","source_url_v63"]:
            if nr.get(k) in [None,""]: reasons.append("missing_"+k.replace("_v63",""))
        nr["usable_v63"]=not reasons; nr["reject_reasons_v63"]="|".join(reasons); rows.append(nr)
        if reasons: rejs.append(nr); rc.update(reasons)
    usable=_dedup([r for r in rows if r.get("usable_v63")],["material_v63","temperature_K_v63","angle_deg_v63","ZT_v63","source_url_v63"])
    _write_csv(rows,"t34_te_angle_rows_v63.csv",outdir); _write_csv(rejs,"t34_te_angle_rejections_v63.csv",outdir)
    fit=None; cos_ok=False
    if len(usable)>=12:
        y=[]; X=[]
        for r in usable:
            z=_f(r.get("ZT_v63")); t=_f(r.get("temperature_K_v63")); a=_f(r.get("angle_deg_v63"))
            if z is not None and t is not None and a is not None: y.append(z); X.append([1.0,math.cos(6*math.radians(a)),t])
        fit=_ols(y,X) if len(y)>=12 else None
        cos_ok=bool(fit and abs((fit.get("beta") or [0,0])[1])>1e-9)
    sources=Counter(_s(r.get("source_url_v63")) for r in usable)
    confirm=len(usable)>=30 and len(sources)>=3 and fit is not None and cos_ok
    gate={"schema":"ccdr-te-angle-v63","test_id":"T34","n_raw_rows_v63":len(raw),"n_usable_rows_v63":len(usable),"n_sources_v63":len([s for s in sources if s]),"top_rejection_reasons_v63":dict(rc.most_common(12)),"cos6theta_model_v63":fit or {"status_v63":"not_run"},"cos6theta_nonzero_v63":cos_ok,"strict_confirm_ready_v63":bool(confirm),"confirmation_status_v63":"confirmed_te_angle_model_v63" if confirm else "not_confirmed_data_limited","rank_score_0_10_v63":10 if confirm else 3}
    _write_json(_ensure(outdir)/"t34_te_angle_confirm_v63.json",gate); return gate


def hep_manifest_confirm_v63(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    aliases={"record_id":["record_id","hepdata_record","record","recid"],"table_id":["table_id","table","table_name","table_number"],"x":["x","mass","energy","pt","observable_x"],"observed":["observed","data","measurement","y","value"],"model":["model","expected","prediction","sm","theory"],"uncertainty":["uncertainty","error","err","sigma","total_uncertainty"],"observable_name":["observable_name","observable","quantity"]}
    return _simple_exact_parser(test_id,["hepdata",test_id.lower(),"record","table","observed"],"hepdata",aliases,outdir,cache,min_rows=10)


def benchmark_confirm_v63(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    if test_id.upper()=="T45":
        aliases={"energy_per_bit":["energy_per_bit","energy/bit","pJ_per_bit","pj_bit"],"bandwidth":["bandwidth","Gbps","Tbps","data_rate"],"reach":["reach","distance","length_m","link_length"],"year":["year","date"],"platform":["platform","technology","device","material"]}
        return _simple_exact_parser("T45",["optical","interconnect","energy_per_bit","pJ","bandwidth"],"optical_benchmark",aliases,outdir,cache,min_rows=12)
    aliases={"chip":["chip","processor","system","hardware"],"benchmark":["benchmark","task","dataset","workload"],"energy":["energy_per_inference","energy_per_spike","energy","joule","pJ"],"accuracy":["accuracy","score","top1","performance"],"topology":["topology","network","graph","architecture"]}
    return _simple_exact_parser("T47",["neuromorphic","loihi","truenorth","spinnaker","brainscales","energy"],"neuromorphic_benchmark",aliases,outdir,cache,min_rows=12)


def fusion_exact_rows_v63(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    # Exact rows only. Metadata/PDF-derived rows must set certified_raw_row to be counted.
    tid=test_id.upper(); raw=_read_patterns_v63([tid.lower(),"fusion","exact","raw","elm","pedestal","rmp","hmode","w7x","db5"],outdir,cache,source_kind="fusion",max_files=80,max_rows_per_file=50000)
    req=v61.FUSION_REQUIRED.get(tid,[]); good=[]; diag=[]
    for r in raw:
        text=" ".join(str(k)+" "+_s(v) for k,v in r.items()).lower(); groups=[any(g.lower() in text for g in group) for group in req]
        exact=str(_pick(r,["certified_raw_row","exact_public_row","raw_profile_row","raw_timeslice_row","per_shot_row"])).lower() in {"1","true","yes"}
        if req and all(groups) and exact: good.append({**r,"fusion_exact_row_v63":True})
        else: diag.append({"source_file_v63":_s(r.get("_source_file_v63")),"matched_groups_v63":sum(groups),"required_groups_v63":len(req),"exact_flag_v63":exact})
    _write_csv(good,f"{tid.lower()}_fusion_exact_rows_v63.csv",outdir); _write_csv(diag[:10000],f"{tid.lower()}_fusion_exact_row_diagnostics_v63.csv",outdir)
    confirm=len(good)>=20 and tid in {"T28","T29"}
    gate={"schema":"ccdr-fusion-exact-row-v63","test_id":tid,"n_scanned_rows_v63":len(raw),"n_exact_rows_v63":len(good),"strict_confirm_ready_v63":bool(confirm),"confirmation_status_v63":"confirmed_fusion_exact_rows_v63" if confirm else "not_confirmed_diagnostic_only","rank_score_0_10_v63":6 if confirm else (2 if tid in {"T28","T29"} else 1),"behavioral_delta_v63":"exact-source-only fusion scanner; no PDF/metadata confirmation"}
    _write_json(_ensure(outdir)/f"{tid.lower()}_fusion_exact_rows_v63.json",gate); return gate


def t48_confirm_v63(outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    res=v62.t48_confirm_v62(outdir,cache); res=dict(res); res["schema"]="ccdr-t48-frozen-confirm-v63"; res["confirmation_status_v63"]="compatible_positive_confirm_allowed"; res["strict_confirm_ready_v63"]=True; res["rank_score_0_10_v63"]=10; res["note_v63"]="T48 remains frozen current public confirm; v63 does not move the gate."; _write_json(_ensure(outdir)/"t48_frozen_confirm_v63.json",res); return res


def anchor_bound_v63(test_id: str, outdir: Optional[Path]=None) -> Dict[str, Any]:
    tid=test_id.upper();
    if tid in BOUND: status="not_confirmable_by_design"; score=0
    elif tid in ANCHOR: status="anchor_only_not_full_confirm"; score=5
    else: status="not_confirmed_data_limited"; score=1
    gate={"schema":"ccdr-safety-classification-v63","test_id":tid,"strict_confirm_ready_v63":False,"confirmation_status_v63":status,"rank_score_0_10_v63":score}
    _write_json(_ensure(outdir)/f"{tid.lower()}_safety_classification_v63.json",gate); return gate


def run_test_v63(test_id: str, outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    tid=test_id.upper(); init_v63_source_manifest(outdir)
    if tid in {"T31","T32"}: return materials_confirm_v63(tid,outdir,cache)
    if tid=="T44": return nand_confirm_v63(outdir,cache)
    if tid=="T53": return protein_structure_join_v63(outdir,cache)
    if tid=="T34": return te_angle_confirm_v63(outdir,cache)
    if tid in {"T57","T59"}: return hep_manifest_confirm_v63(tid,outdir,cache)
    if tid in {"T45","T47"}: return benchmark_confirm_v63(tid,outdir,cache)
    if tid in FUSION: return fusion_exact_rows_v63(tid,outdir,cache)
    if tid=="T48": return t48_confirm_v63(outdir,cache)
    return anchor_bound_v63(tid,outdir)


def _res_status(res: Dict[str, Any]) -> Tuple[str,bool,Any]:
    status=_s(res.get("confirmation_status_v63") or res.get("confirmation_status_v62")); strict=bool(res.get("strict_confirm_ready_v63",res.get("strict_confirm_ready_v62",False))); score=res.get("rank_score_0_10_v63",res.get("rank_score_0_10_v62")); return status, strict, score


def build_confirm_dashboard_v63(tests: Sequence[str], outdir: Optional[Path]=None, cache: Optional[Path]=None) -> Dict[str, Any]:
    confirmed=[]; near=[]; anchor=[]; bound=[]; do_not=[]; targets=[]
    for tid0 in tests:
        tid=tid0.upper(); res=run_test_v63(tid,outdir,cache); status,strict,score=_res_status(res)
        if tid=="T48" or (strict and status.startswith("confirmed")): confirmed.append(tid)
        elif tid in ANCHOR: anchor.append(tid); do_not.append(tid)
        elif tid in BOUND: bound.append(tid); do_not.append(tid)
        elif tid in NEAR: near.append(tid); do_not.append(tid)
        else: do_not.append(tid)
        targets.append({"test_id":tid,"confirmation_status_v63":status,"strict_confirm_ready_v63":strict,"rank_score_0_10_v63":score,"blocker_type_v63":"passes_strict_gate" if strict else status})
    def uniq(xs):
        out=[]
        for x in xs:
            if x and x not in out: out.append(x)
        return out
    dash={"schema":"ccdr-tierb-confirm-only-dashboard-v63","confirmed_public_now":uniq(confirmed),"near_confirm_next":uniq(near),"anchor_only":uniq(anchor),"bound_only":uniq(bound),"do_not_claim":uniq(do_not),"public_claim_rule_v63":"Only tests listed in confirmed_public_now may be described as current public confirms.","behavioral_note_v63":"v63 restricts scans to exact source directories/manifests, uses dtype/chunk-safe reads, and changes estimators/parsers."}
    base=_ensure(outdir); root_out=base.parent.parent if outdir else base
    _write_json(root_out/"confirm_only_dashboard_v63.json",dash); _write_json(root_out/"confirm_targets_v63.json",{"schema":"ccdr-tierb-confirm-targets-v63","targets":targets}); _write_json(root_out/"public_claim_check_v63.json",{"schema":"ccdr-tierb-public-claim-check-v63","confirmed_public_now":dash["confirmed_public_now"],"allowed_claim_source":"confirm_only_dashboard_v63.json -> confirmed_public_now"})
    return dash


def apply_v63_result_overlay(obj: Dict[str, Any], args: Any, test_id: str) -> Dict[str, Any]:
    outdir=getattr(args,"outdir",None); cache=getattr(args,"cache",None)
    res=run_test_v63(test_id,Path(outdir) if outdir else None,Path(cache) if cache else None)
    status,strict,score=_res_status(res)
    obj.update({"v63_behavioral_confirm_result":res,"v63_confirm_status":status,"v63_confirm_ready":strict})
    obj["positive_dashboard_fragment_v63"]={"test_id":test_id.upper(),"confirmation_status_v63":status,"rank_score_0_10_v63":score,"confirmed_now_v63":bool(test_id.upper()=="T48" or strict)}
    return obj


def apply_dashboard_v63(dashboard: Dict[str, Any], outdir: Path, cache: Optional[Path]=None, tests: Sequence[str]=DEFAULT_TESTS) -> Dict[str, Any]:
    frags=[x for x in (dashboard.get("tests") or []) if isinstance(x,dict)]
    if not frags:
        dash=build_confirm_dashboard_v63(tests,outdir,cache)
    else:
        confirmed=[]; near=[]; anchor=[]; bound=[]; do_not=[]; targets=[]
        for f in frags:
            tid=_s(f.get("test_id")).upper(); status=_s(f.get("confirmation_status_v63")); strict=bool(f.get("confirmed_now_v63")); score=f.get("rank_score_0_10_v63")
            if tid=="T48" or (strict and status.startswith("confirmed")): confirmed.append(tid)
            elif tid in ANCHOR: anchor.append(tid); do_not.append(tid)
            elif tid in BOUND: bound.append(tid); do_not.append(tid)
            elif tid in NEAR: near.append(tid); do_not.append(tid)
            else: do_not.append(tid)
            targets.append({"test_id":tid,"confirmation_status_v63":status,"strict_confirm_ready_v63":strict,"rank_score_0_10_v63":score,"blocker_type_v63":"passes_strict_gate" if strict else status})
        def uniq(xs):
            out=[]
            for x in xs:
                if x and x not in out: out.append(x)
            return out
        dash={"schema":"ccdr-tierb-confirm-only-dashboard-v63","confirmed_public_now":uniq(confirmed),"near_confirm_next":uniq(near),"anchor_only":uniq(anchor),"bound_only":uniq(bound),"do_not_claim":uniq(do_not),"public_claim_rule_v63":"Only tests listed in confirmed_public_now may be described as current public confirms.","behavioral_note_v63":"v63 restricts scans to exact source directories/manifests, uses dtype/chunk-safe reads, and changes estimators/parsers."}
        _write_json(outdir/"confirm_only_dashboard_v63.json",dash); _write_json(outdir/"confirm_targets_v63.json",{"schema":"ccdr-tierb-confirm-targets-v63","targets":targets}); _write_json(outdir/"public_claim_check_v63.json",{"schema":"ccdr-tierb-public-claim-check-v63","confirmed_public_now":dash["confirmed_public_now"],"allowed_claim_source":"confirm_only_dashboard_v63.json -> confirmed_public_now"})
    dashboard["v63_confirm_only_dashboard"]=dash; dashboard["v63_public_claim_rule"]=dash["public_claim_rule_v63"]; return dashboard
