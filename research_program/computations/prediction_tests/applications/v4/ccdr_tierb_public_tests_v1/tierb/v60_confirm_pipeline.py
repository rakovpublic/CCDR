#!/usr/bin/env python3
"""v60 Tier-B one-command confirm pipeline and stronger exact-row gates.

This layer implements the next requested improvements over v59:
1. one-command run_all + confirm-only wrapper support,
2. T31/T32 measured microstructure source/sample dedup + rejection summaries,
3. T31/T32 source/family/temp-bin balance gates,
4. T44 exact NAND manifest + strict fixture/row gate,
5. T53 ProteinGym->UniProt/PDB/AlphaFold join attempt + FDR/bootstrap gate,
6. T34 exact thermoelectric ZT-angle parser gate,
7. T57/T59 exact HEPData manifest parser gate,
8. T45/T47 exact benchmark row parser gate,
9. T26-T30 fusion locked diagnostic-only unless exact row tables exist,
10. final v60 public-claim dashboard/checker.

The default public claim remains conservative: T48 only, unless a test passes
its explicit v60 strict/public gate. This prevents readiness/anchors/bounds from
being inflated into confirmations.
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from . import v59_confirm_extractors as v59

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GEN_DIR = DATA_DIR / "generated"
MANIFEST_DIR = DATA_DIR / "manifests"

CONFIRM_ONLY_DEFAULT_TESTS = [
    "T31", "T32", "T44", "T48", "T53", "T34", "T57", "T59", "T45", "T47",
    "T26", "T27", "T28", "T29", "T30", "T50", "T51", "T52", "T60",
]
ORDER = ["T48", "T31", "T32", "T44", "T53", "T34", "T57", "T59", "T45", "T47", "T29", "T28", "T26", "T27", "T30", "T60", "T50", "T51", "T52"]
NEAR_CONFIRM = {"T31", "T32", "T44", "T53", "T34", "T57", "T59", "T45", "T47"}
FUSION = {"T26", "T27", "T28", "T29", "T30"}
BOUND = {"T50", "T51", "T52"}
ANCHOR = {"T60"}


def _ensure() -> Path:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    return GEN_DIR


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
    return v59._pick(row, names)  # type: ignore[attr-defined]


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, dict)):
        return v
    return str(v)


def _write_json(path: Path, obj: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8")
    return str(path)


def _write_csv(rows: Sequence[Dict[str, Any]], filename: str, root: Optional[Path] = None) -> str:
    out = (root or _ensure()) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in (r or {}).keys():
            if k not in keys:
                keys.append(k)
    if not keys:
        keys = ["empty_v60"]
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


def _read_patterns(patterns: Sequence[str]) -> List[Dict[str, Any]]:
    # v59 already searches ROOT/DATA/GEN and supports csv/tsv/json/jsonl.
    return v59._read_patterns(patterns)  # type: ignore[attr-defined]


def _read_csv_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or pd is None:
        return []
    try:
        return [dict(r) for _, r in pd.read_csv(path).iterrows()]
    except Exception:
        return []


def _counter_rows(counter: Counter, key_name: str, value_name: str = "n_rows_v60") -> List[Dict[str, Any]]:
    return [{key_name: k, value_name: v} for k, v in counter.most_common()]


# ---------------------------------------------------------------------------
# T31/T32 material gates
# ---------------------------------------------------------------------------

def _reject_summary(rows: Sequence[Dict[str, Any]], reason_col: str = "reject_reasons_v59") -> List[Dict[str, Any]]:
    c: Counter = Counter()
    for r in rows:
        reasons = _s(r.get(reason_col))
        if not reasons:
            continue
        for reason in re.split(r"[|,;]+", reasons):
            rr = reason.strip()
            if rr:
                c[rr] += 1
    return [{"reject_reason_v60": k, "n_rows_v60": v} for k, v in c.most_common()] or [{"reject_reason_v60": "none_recorded", "n_rows_v60": 0}]


def materials_confirm_v60(test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    base = v59.materials_confirm_v59(tid)
    dedup_path = GEN_DIR / f"{tid.lower()}_microstructure_dedup_rows_v59.csv"
    rej_path = GEN_DIR / f"{tid.lower()}_microstructure_rejection_diagnostics_v59.csv"
    dedup_rows = _read_csv_if_exists(dedup_path)
    rej_rows = _read_csv_if_exists(rej_path)
    usable = [r for r in dedup_rows if str(r.get("usable_v59", "")).lower() in {"true", "1", "yes"} or r.get("usable_v59") is True]

    source_counts = Counter(_s(r.get("source_url_v59")) or "unknown_source" for r in usable)
    family_counts = Counter(_s(r.get("material_family_v59")) or "unknown_family" for r in usable)
    temp_counts = Counter(_s(r.get("temperature_bin_v59")) or "unknown_temp_bin" for r in usable)
    n = len(usable)
    max_source_fraction = max(source_counts.values()) / n if n and source_counts else 1.0
    max_family_fraction = max(family_counts.values()) / n if n and family_counts else 1.0
    unknown_temp_fraction = temp_counts.get("unknown", 0) / n if n else 1.0

    gate = dict(base.get("gate") or {})
    failed = list(gate.get("failed_subgates_v59") or [])
    if max_source_fraction > 0.40:
        failed.append("source_dominance_fraction_le_0p40")
    if max_family_fraction > 0.50:
        failed.append("material_family_dominance_fraction_le_0p50")
    if unknown_temp_fraction > 0.10:
        failed.append("unknown_temperature_bin_fraction_le_0p10")

    strict = bool(base.get("strict_confirm_ready_v59")) and not failed
    score = 10 if strict else int(base.get("score_0_10_v59") or 9)
    if n == 0:
        score = 4
    elif n < 20:
        score = min(score, 7)

    rejection_summary = _reject_summary(rej_rows)
    balance_rows = (
        [{"balance_dimension_v60": "source", "value_v60": k, "n_rows_v60": v, "fraction_v60": v / n if n else 0} for k, v in source_counts.most_common()] +
        [{"balance_dimension_v60": "material_family", "value_v60": k, "n_rows_v60": v, "fraction_v60": v / n if n else 0} for k, v in family_counts.most_common()] +
        [{"balance_dimension_v60": "temperature_bin", "value_v60": k, "n_rows_v60": v, "fraction_v60": v / n if n else 0} for k, v in temp_counts.most_common()]
    )
    _write_csv(rejection_summary, f"{tid.lower()}_microstructure_rejection_summary_v60.csv")
    _write_csv(balance_rows, f"{tid.lower()}_microstructure_balance_gates_v60.csv")
    gate_v60 = {
        "schema": "ccdr-materials-confirm-gates-v60",
        "test_id": tid,
        "n_usable_rows_v60": n,
        "n_sources_v60": len([x for x in source_counts if x and x != "unknown_source"]),
        "n_material_families_v60": len([x for x in family_counts if x and x != "unknown_family"]),
        "n_temperature_bins_v60": len([x for x in temp_counts if x and x not in {"unknown", "unknown_temp_bin"}]),
        "max_source_fraction_v60": max_source_fraction,
        "max_material_family_fraction_v60": max_family_fraction,
        "unknown_temperature_bin_fraction_v60": unknown_temp_fraction,
        "base_v59_strict_confirm_ready": bool(base.get("strict_confirm_ready_v59")),
        "strict_confirm_ready_v60": strict,
        "failed_subgates_v60": sorted(set(failed)),
        "rejection_summary_csv_v60": f"data/generated/{tid.lower()}_microstructure_rejection_summary_v60.csv",
        "balance_gates_csv_v60": f"data/generated/{tid.lower()}_microstructure_balance_gates_v60.csv",
    }
    _write_json(GEN_DIR / f"{tid.lower()}_microstructure_confirm_gates_v60.json", gate_v60)
    _write_csv([gate_v60], f"{tid.lower()}_microstructure_confirm_gates_v60.csv")
    return {
        "schema": "ccdr-materials-confirm-v60",
        "test_id": tid,
        "status": "strict_materials_confirm_ready_v60" if strict else "materials_confirm_gates_pending_v60",
        "strict_confirm_ready_v60": strict,
        "score_0_10_v60": score,
        "failed_subgates_v60": sorted(set(failed)),
        "gate_v60": gate_v60,
        "base_v59": base,
    }


# ---------------------------------------------------------------------------
# Exact row/manifests for T44/T53/T34/T57/T59/T45/T47
# ---------------------------------------------------------------------------

def t44_nand_exact_v60() -> Dict[str, Any]:
    base = v59.t44_nand_exact_v59()
    manifest = [
        {
            "source_family_v60": "WikiChip",
            "example_lookup_v60": "3D NAND die or ISSCC device page",
            "required_columns_v60": "company|year|layers|capacity_Gb|die_area_mm2|bits_per_cell|source_url",
            "confirmation_policy_v60": "reference_only unless die_area_mm2 and bits_per_cell are explicit",
        },
        {
            "source_family_v60": "TechInsights",
            "example_lookup_v60": "memory teardown/die photo report",
            "required_columns_v60": "company|year|layers|capacity_Gb|die_area_mm2|bits_per_cell|source_url",
            "confirmation_policy_v60": "preferred Tier-A when die area is measured",
        },
        {
            "source_family_v60": "ISSCC/IEDM/vendor paper",
            "example_lookup_v60": "paper table with die size, layers, Gb and cell type",
            "required_columns_v60": "company|year|layers|capacity_Gb|die_area_mm2|bits_per_cell|source_url",
            "confirmation_policy_v60": "official table accepted",
        },
    ]
    fixture = [{"company": "", "year": "", "layers": "", "capacity_Gb": "", "die_area_mm2": "", "bits_per_cell": "", "source_url": "", "notes": "fill with explicit measured/published values only"}]
    _write_csv(manifest, "t44_nand_exact_source_manifest_v60.csv", MANIFEST_DIR)
    _write_csv(fixture, "t44_nand_exact_fixture_v60.csv", MANIFEST_DIR)
    # Reuse v59 parsed rows but add hard public gate.
    rows = _read_csv_if_exists(GEN_DIR / "t44_nand_exact_rows_v59.csv")
    usable = [r for r in rows if str(r.get("usable_tier_a_v59", "")).lower() in {"true", "1", "yes"} or r.get("usable_tier_a_v59") is True]
    companies = {(_s(r.get("company_v59")) or _s(r.get("company"))) for r in usable if (_s(r.get("company_v59")) or _s(r.get("company")))}
    failed: List[str] = []
    if len(usable) < 8:
        failed.append(">=8_true_tier_a_rows")
    if len(companies) < 3:
        failed.append(">=3_companies")
    failed.append("manufacturer_year_jackknife_model_required")
    strict = False  # never auto-confirm until model has run; avoids resurrecting old frozen T44.
    gate = {
        "schema": "ccdr-t44-nand-exact-gate-v60",
        "test_id": "T44",
        "usable_tier_a_rows_v60": len(usable),
        "n_companies_v60": len(companies),
        "strict_confirm_ready_v60": strict,
        "failed_subgates_v60": failed,
        "manifest_csv_v60": "data/manifests/t44_nand_exact_source_manifest_v60.csv",
        "fixture_csv_v60": "data/manifests/t44_nand_exact_fixture_v60.csv",
    }
    _write_json(GEN_DIR / "t44_nand_exact_gate_v60.json", gate)
    _write_csv([gate], "t44_nand_exact_gate_v60.csv")
    return {"schema": "ccdr-t44-nand-exact-v60", "status": "t44_true_tier_a_audit_repair_required_v60", "strict_confirm_ready_v60": strict, "score_0_10_v60": 8 if usable else 5, "gate_v60": gate, "base_v59": base}


def _normalize_t53_row(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    assay = _s(_pick(raw, ["assay", "assay_name", "ProteinGym_assay", "DMS_id", "DMS_id_x", "DMS_id_y"]))
    uniprot = _s(_pick(raw, ["uniprot", "UniProt", "uniprot_accession", "accession", "target_seq", "protein_name"]))
    pdb = _s(_pick(raw, ["pdb_id", "PDB", "alphafold_id", "AlphaFold", "structure_id", "alphafold_model"]))
    family = _s(_pick(raw, ["family", "protein_family", "gene_family", "taxon", "organism"]))
    outcome = _f(_pick(raw, ["DMS_outcome", "fitness", "score", "effect", "DMS_score", "fitness_score"]))
    sym = _f(_pick(raw, ["symmetry_proxy", "contact_network_proxy", "oligomeric_state_numeric", "contacts", "symmetry_score", "n_contacts"]))
    assay_type = _s(_pick(raw, ["assay_type", "selection_type", "DMS_type"])) or "unknown_assay_type"
    cluster = _s(_pick(raw, ["sequence_cluster", "cluster", "sequence_identity_cluster", "family_cluster"]))
    reasons = []
    if not assay: reasons.append("missing_ProteinGym_assay")
    if not uniprot: reasons.append("missing_UniProt")
    if not pdb: reasons.append("missing_PDB_or_AlphaFold")
    if not family: reasons.append("missing_family")
    if outcome is None: reasons.append("missing_DMS_outcome")
    if sym is None: reasons.append("missing_symmetry_or_contact_proxy")
    if not cluster: reasons.append("missing_sequence_cluster")
    return {"raw_index_v60": idx, "assay_v60": assay, "uniprot_v60": uniprot, "structure_id_v60": pdb, "family_v60": family, "assay_type_v60": assay_type, "DMS_outcome_v60": outcome, "symmetry_proxy_v60": sym, "sequence_cluster_v60": cluster, "usable_join_row_v60": not reasons, "reject_reasons_v60": "|".join(reasons)}


def t53_structure_join_v60() -> Dict[str, Any]:
    base = v59.t53_structure_join_v59()
    raw = _read_patterns(["t53*proteingym*structure*rows*.csv", "t53*proteingym*enriched*.csv", "*ProteinGym*.csv", "*proteingym*.csv", "*alphafold*.csv", "*uniprot*.csv", "*pdb*.csv"])
    rows = [_normalize_t53_row(r, i) for i, r in enumerate(raw)]
    usable = [r for r in rows if r["usable_join_row_v60"]]
    rejs = [r for r in rows if not r["usable_join_row_v60"]]
    fams = {_s(r.get("family_v60")) for r in usable if _s(r.get("family_v60"))}
    assays = {_s(r.get("assay_v60")) for r in usable if _s(r.get("assay_v60"))}
    assay_types = {_s(r.get("assay_type_v60")) for r in usable if _s(r.get("assay_type_v60")) and _s(r.get("assay_type_v60")) != "unknown_assay_type"}
    clusters = {_s(r.get("sequence_cluster_v60")) for r in usable if _s(r.get("sequence_cluster_v60"))}
    failed = []
    if len(usable) < 50: failed.append(">=50_ProteinGym_structure_join_rows")
    if len(fams) < 5: failed.append(">=5_families")
    if len(assays) < 2: failed.append(">=2_assays")
    if len(clusters) < 10: failed.append(">=10_sequence_clusters")
    if len(assay_types) < 2: failed.append(">=2_assay_types")
    failed += ["family_assay_sequence_jackknife", "BH_FDR_or_bootstrap_significance"]
    _write_csv(rows, "t53_proteingym_structure_join_rows_v60.csv")
    _write_csv(rejs, "t53_proteingym_structure_join_rejections_v60.csv")
    _write_csv(_reject_summary(rejs, "reject_reasons_v60"), "t53_structure_join_rejection_summary_v60.csv")
    gate = {"schema": "ccdr-t53-structure-join-model-v60", "test_id": "T53", "raw_rows_v60": len(raw), "usable_join_rows_v60": len(usable), "n_families_v60": len(fams), "n_assays_v60": len(assays), "n_assay_types_v60": len(assay_types), "n_sequence_clusters_v60": len(clusters), "strict_confirm_ready_v60": False, "failed_subgates_v60": failed}
    _write_json(GEN_DIR / "t53_proteingym_structure_model_gate_v60.json", gate)
    _write_csv([gate], "t53_proteingym_structure_model_gate_v60.csv")
    return {"schema": "ccdr-t53-structure-join-v60", "status": "t53_structure_join_model_pending_v60", "strict_confirm_ready_v60": False, "score_0_10_v60": 8 if len(usable) >= 50 else 6, "gate_v60": gate, "base_v59": base}


def _generic_exact_rows(test_id: str, required: Sequence[str], patterns: Sequence[str], min_rows: int = 5, min_sources: int = 2) -> Dict[str, Any]:
    tid = test_id.upper()
    raw = _read_patterns(patterns)
    rows: List[Dict[str, Any]] = []
    rejs: List[Dict[str, Any]] = []
    for i, r in enumerate(raw):
        out: Dict[str, Any] = {"raw_index_v60": i}
        reasons: List[str] = []
        for col in required:
            val = _pick(r, [col, col.lower(), col.upper(), col.replace("_", " ")])
            if col.endswith("_K") or col.endswith("_deg") or col in {"ZT", "year", "energy", "performance", "accuracy_or_reach"}:
                vv = _f(val)
                out[col + "_v60"] = vv
                if vv is None:
                    reasons.append(f"missing_{col}")
            else:
                ss = _s(val)
                out[col + "_v60"] = ss
                if not ss:
                    reasons.append(f"missing_{col}")
        out["usable_exact_row_v60"] = not reasons
        out["reject_reasons_v60"] = "|".join(reasons)
        rows.append(out)
        if reasons:
            rejs.append(out)
    usable = [r for r in rows if r["usable_exact_row_v60"]]
    sources = {_s(r.get("source_url_v60")) for r in usable if _s(r.get("source_url_v60"))}
    failed: List[str] = []
    if len(usable) < min_rows:
        failed.append(f">={min_rows}_usable_exact_rows")
    if len(sources) < min_sources:
        failed.append(f">={min_sources}_source_urls")
    failed.append("test_specific_model_and_jackknife_required")
    _write_csv(rows, f"{tid.lower()}_exact_rows_v60.csv")
    _write_csv(rejs, f"{tid.lower()}_exact_rejection_diagnostics_v60.csv")
    _write_csv(_reject_summary(rejs, "reject_reasons_v60"), f"{tid.lower()}_exact_rejection_summary_v60.csv")
    gate = {"schema": "ccdr-exact-row-gate-v60", "test_id": tid, "required_columns_v60": "|".join(required), "n_raw_rows_v60": len(raw), "n_usable_rows_v60": len(usable), "n_source_urls_v60": len(sources), "strict_confirm_ready_v60": False, "failed_subgates_v60": failed}
    _write_json(GEN_DIR / f"{tid.lower()}_exact_row_gate_v60.json", gate)
    _write_csv([gate], f"{tid.lower()}_exact_row_gate_v60.csv")
    return {"schema": "ccdr-exact-row-gate-v60", "test_id": tid, "status": "exact_rows_pending_v60", "strict_confirm_ready_v60": False, "score_0_10_v60": 3 if len(usable) else 2, "gate_v60": gate}


def exact_gate_v60(test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    if tid == "T34":
        return _generic_exact_rows(tid, ["material", "ZT", "temperature_K", "orientation_angle_deg", "grain_boundary_angle_deg", "source_url"], ["t34*exact*.csv", "*teMatDb*.csv", "*Starrydata*.csv", "*Bi2Te3*.csv", "*Sb2Te3*.csv"], min_rows=20, min_sources=2)
    if tid in {"T57", "T59"}:
        return _generic_exact_rows(tid, ["record_id", "table_id", "x_column", "observed_column", "model_column", "uncertainty_column", "observable_name", "source_url"], [f"{tid.lower()}*hepdata*manifest*.csv", "*hepdata*manifest*.csv", "*HEPData*.csv"], min_rows=3, min_sources=1)
    if tid == "T45":
        return _generic_exact_rows(tid, ["source_url", "year", "benchmark", "energy", "performance", "accuracy_or_reach"], ["t45*benchmark*.csv", "*optical*interconnect*.csv", "*energy_bit*.csv"], min_rows=10, min_sources=2)
    if tid == "T47":
        return _generic_exact_rows(tid, ["source_url", "year", "benchmark", "energy", "performance", "accuracy_or_reach"], ["t47*benchmark*.csv", "*neuromorphic*.csv", "*loihi*.csv", "*spinnaker*.csv"], min_rows=10, min_sources=2)
    return {"schema": "ccdr-exact-row-gate-v60", "test_id": tid, "status": "no_exact_gate_defined_v60", "strict_confirm_ready_v60": False, "score_0_10_v60": 1}


def fusion_diagnostic_v60(test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    required = {
        "T26": "per-shot E_ELM/W_ELM + Pped/Wped + volume/proxy + device/shot rows",
        "T27": "per-discharge RMP current/phasing + ELM-frequency/energy rows",
        "T28": "public DB5-style per-timeslice H-mode confinement rows",
        "T29": "raw W7-X/AUG transport/profile rows, not paper-level averages",
        "T30": "residual/curvature row table tied to T28/T29 inputs",
    }.get(tid, "exact physical row table")
    gate = {"schema": "ccdr-fusion-diagnostic-contract-v60", "test_id": tid, "diagnostic_only_v60": True, "strict_confirm_ready_v60": False, "required_for_confirm_v60": required, "policy_v60": "PDFs, figures, and summary regressions are support/diagnostic only; they cannot confirm."}
    _write_json(GEN_DIR / f"{tid.lower()}_fusion_diagnostic_contract_v60.json", gate)
    _write_csv([gate], f"{tid.lower()}_fusion_diagnostic_contract_v60.csv")
    return {"schema": "ccdr-fusion-diagnostic-v60", "test_id": tid, "status": "fusion_diagnostic_only_v60", "strict_confirm_ready_v60": False, "score_0_10_v60": 1, "gate_v60": gate}


def t48_confirm_v60() -> Dict[str, Any]:
    base = v59.t48_confirm_v59()
    gate = dict(base.get("gate") or {})
    gate.update({"schema": "ccdr-t48-frozen-confirm-v60", "confirm_allowed_now_v60": True, "confirmed_public_now_v60": True, "gate_policy_v60": "frozen public confirm; v60 adds only robustness/checker artifacts"})
    _write_json(GEN_DIR / "t48_frozen_confirm_robustness_v60.json", gate)
    _write_csv([gate], "t48_frozen_confirm_robustness_v60.csv")
    return {"schema": "ccdr-t48-confirm-v60", "status": "compatible_positive_confirm_allowed_v60", "strict_confirm_ready_v60": True, "score_0_10_v60": 10, "gate_v60": gate, "base_v59": base}


def _target(test_id: str, score: int, blocker: str, next_source: str, status: str) -> Dict[str, Any]:
    tid = test_id.upper()
    return {"schema": "ccdr-tierb-confirm-target-v60", "test_id": tid, "rank_score_0_10_v60": int(score), "blocker_type_v60": blocker, "next_data_source_v60": next_source, "expected_effort_v60": "low" if tid == "T48" else "medium" if tid in {"T31", "T32", "T44", "T53", "T34"} else "high", "confirmation_legally_possible_v60": tid not in BOUND and tid not in ANCHOR, "confirmation_status_v60": status}


def apply_v60_result_overlay(obj: Dict[str, Any], args: Any, test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    try:
        obj = v59.apply_v59_result_overlay(obj, args, tid)
    except Exception as e:
        obj = dict(obj)
        obj.setdefault("v59_overlay_error_before_v60", f"{type(e).__name__}: {e}")
    arts: Dict[str, Any] = {}
    score = 1
    public = False
    strict = False
    blocker = "data_limited"
    next_source = "test-specific exact structured source"
    confirmation = "not_confirmed_data_limited"
    data = "data_limited"
    evidence = "data_limited"
    try:
        if tid in {"T31", "T32"}:
            arts["materials_confirm_v60"] = materials_confirm_v60(tid)
            strict = bool(arts["materials_confirm_v60"].get("strict_confirm_ready_v60"))
            score = int(arts["materials_confirm_v60"].get("score_0_10_v60") or 9)
            blocker = "measured_microstructure_confirm_gates_pending" if not strict else "strict_microstructure_confirm_ready_review_required"
            next_source = "add strict measured κ(T)+SEM/TEM/XRD rows; pass balance, bootstrap, AIC/BIC and jackknife gates"
            confirmation = "not_confirmed_next_gate_required" if not strict else "strict_confirm_ready_pending_manual_review"
            data = "measured_microstructure_rows_required"
            evidence = "near_confirm_or_model_gate"
        elif tid == "T44":
            arts["t44_nand_exact_v60"] = t44_nand_exact_v60()
            score = int(arts["t44_nand_exact_v60"].get("score_0_10_v60") or 8)
            blocker = "true_tier_a_nand_rows_required"
            next_source = "fill data/manifests/t44_nand_exact_fixture_v60.csv with real WikiChip/TechInsights/ISSCC values"
            confirmation = "not_confirmed_audit_repair_required"
            data = "true_tier_a_nand_rows_required"
            evidence = "audit_repair_route"
        elif tid == "T48":
            arts["t48_confirm_v60"] = t48_confirm_v60()
            strict = public = True
            score = 10
            blocker = "robustness_only_audit_not_gate_change"
            next_source = "frozen PV descriptor rows + family/source/year/permutation robustness artifacts"
            confirmation = "compatible_positive_confirm_allowed"
            data = "pv_descriptor_rows_or_frozen_artifact"
            evidence = "compatible_positive"
        elif tid == "T53":
            arts["t53_structure_join_v60"] = t53_structure_join_v60()
            score = int(arts["t53_structure_join_v60"].get("score_0_10_v60") or 6)
            blocker = "ProteinGym_structure_join_model_FDR_required"
            next_source = "complete ProteinGym->UniProt/PDB/AlphaFold join plus family/assay/sequence jackknife"
            confirmation = "not_confirmed_next_gate_required"
            data = "dms_structure_join_rows_missing_or_model_pending"
            evidence = "near_confirm_or_model_ready"
        elif tid in {"T34", "T57", "T59", "T45", "T47"}:
            arts[f"{tid.lower()}_exact_gate_v60"] = exact_gate_v60(tid)
            score = int(arts[f"{tid.lower()}_exact_gate_v60"].get("score_0_10_v60") or 3)
            blocker = "exact_row_table_or_manifest_required"
            next_source = "provide exact public row table/manifest; broad discovery is not evidence"
            confirmation = "not_confirmed_data_limited"
            data = "exact_rows_required"
            evidence = "data_limited_positive_path"
        elif tid in FUSION:
            arts[f"{tid.lower()}_fusion_diagnostic_v60"] = fusion_diagnostic_v60(tid)
            score = 1
            blocker = "raw_fusion_row_table_required_for_confirm"
            next_source = "exact fusion measurement rows, not PDF summaries"
            confirmation = "not_confirmed_diagnostic_only"
            data = "fusion_exact_rows_required"
            evidence = "diagnostic_only"
        elif tid in BOUND:
            score = 0
            blocker = "bound_only_by_design"
            next_source = "constraint/upper-limit table only; no positive-confirm route"
            confirmation = "not_confirmable_by_design"
            data = "bound_table_or_literature_bound"
            evidence = "bound_only"
        elif tid in ANCHOR:
            score = 5
            blocker = "T60b_T60c_T60d_required"
            next_source = "quark/lattice uncertainty + sector reshuffle + look-elsewhere registry"
            confirmation = "anchor_only_not_full_confirm"
            data = "anchor_only"
            evidence = "positive_consistency_anchor"
    except Exception as e:
        arts["v60_overlay_error"] = f"{type(e).__name__}: {e}"
        blocker = "v60_overlay_exception"
        confirmation = "not_confirmed_runtime_output_missing"
        score = {"T31": 9, "T32": 9, "T44": 8, "T53": 6, "T34": 3}.get(tid, 1)
    if tid != "T48":
        public = False
    split = {"execution_status_v60": "ok", "data_status_v60": data, "evidence_status_v60": evidence, "confirmation_status_v60": confirmation}
    target = _target(tid, score, blocker, next_source, confirmation)
    obj["auto_data_improvements_v60"] = arts
    obj["status_split_v60"] = split
    obj["confirm_target_v60"] = target
    obj["confirm_allowed_now_v60"] = bool(public)
    obj["confirmation_label_v60"] = "compatible_positive" if public else confirmation
    obj["confirmation_blocker_v60"] = {"strict_confirm_allowed_now": bool(strict), "public_confirm_allowed_now": bool(public), "why_not_confirmed": None if public else blocker, "single_next_blocker": "robustness_only" if public else blocker, "best_auto_data_source_next": next_source}
    obj["near_confirm_score_v60"] = {"score_0_10": int(score), "primary_table_available": score >= 3, "model_rows_available": score >= 6, "model_gate_attempted": score >= 7, "strict_gate_remaining": [] if public else [blocker]}
    obj["public_claim_gate_v60"] = {"claimable_only_if_listed_in": "positive_dashboard.json:v60_confirm_only_dashboard.confirmed_public_now", "confirmed_now_v60": bool(public), "legacy_confirm_fields_are_not_public_claims": True}
    obj.update(split)
    obj["positive_dashboard_fragment_v60"] = {"test_id": tid, "verdict": obj.get("programmatic_verdict") or obj.get("status"), "confirmation_label": obj["confirmation_label_v60"], "confirm_allowed_now": bool(public), "strict_confirm_allowed_now": bool(public), "near_confirm_score": obj["near_confirm_score_v60"], "status_split_v60": split, "why_not_confirmed": obj["confirmation_blocker_v60"]["why_not_confirmed"], "single_next_blocker": obj["confirmation_blocker_v60"]["single_next_blocker"], "best_auto_data_source_next": next_source, "confirm_target_v60": target, "v60": {"auto_data_improvements_v60": arts, "confirmation_blocker_v60": obj["confirmation_blocker_v60"], "public_claim_gate_v60": obj["public_claim_gate_v60"]}}
    obj["quality_patch_version"] = str(obj.get("quality_patch_version", "")) + "+v60_confirm_pipeline"
    return obj


def confirm_only_dashboard_v60(status: Dict[str, List[str]]) -> Dict[str, Any]:
    return {"schema": "ccdr-tierb-confirm-only-dashboard-v60", "confirmed_public_now": status.get("confirmed_public_now", []), "near_confirm_next": status.get("near_confirm_next", []), "anchor_only": status.get("anchor_only", []), "bound_only": status.get("bound_only", []), "do_not_claim": status.get("do_not_claim", []), "public_claim_rule_v60": "Only tests listed in confirmed_public_now may be described as current public confirms."}


def _sort_ids(xs: Iterable[str]) -> List[str]:
    return sorted(set(x for x in xs if x), key=lambda x: ORDER.index(x) if x in ORDER else 99)


def apply_dashboard_v60(dashboard: Dict[str, Any], outdir: Path) -> Dict[str, Any]:
    try:
        dashboard = v59.apply_dashboard_v59(dashboard, outdir)
    except Exception as e:
        dashboard = dict(dashboard)
        dashboard.setdefault("v59_dashboard_error_before_v60", f"{type(e).__name__}: {e}")
    outdir = Path(outdir)
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
                    latest = rr.get("positive_dashboard_fragment_v60") or rr.get("positive_dashboard_fragment_v59") or rr.get("positive_dashboard_fragment_v58") or latest
                except Exception:
                    pass
        tests.append(latest)
        tid = latest.get("test_id") or tid
        if latest.get("confirm_allowed_now") and tid == "T48":
            status["confirmed_public_now"].append(tid)
        elif tid in NEAR_CONFIRM:
            status["near_confirm_next"].append(tid)
        if tid in ANCHOR:
            status["anchor_only"].append(tid)
        if tid in BOUND:
            status["bound_only"].append(tid)
        if tid != "T48":
            status["do_not_claim"].append(tid)
        target = latest.get("confirm_target_v60") or latest.get("confirm_target_v59") or latest.get("confirm_target_v58")
        if target:
            targets.append(target)
    for k in status:
        status[k] = _sort_ids(status[k])
    def score(t: Dict[str, Any]) -> int:
        for k in ["rank_score_0_10_v60", "rank_score_0_10_v59", "rank_score_0_10_v58", "rank_score_0_10_v57"]:
            try:
                if t.get(k) is not None:
                    return int(t.get(k))
            except Exception:
                pass
        return 0
    targets = sorted(targets, key=lambda t: (-score(t), ORDER.index(t.get("test_id")) if t.get("test_id") in ORDER else 99))
    dash = confirm_only_dashboard_v60(status)
    claim = {"schema": "ccdr-public-claim-check-v60", "allowed_confirm_source": "positive_dashboard.json:v60_confirm_only_dashboard.confirmed_public_now", "confirmed_public_now": dash["confirmed_public_now"], "pass_v60": dash["confirmed_public_now"] == ["T48"], "message_v60": "Only confirmed_public_now may be used for public confirm claims."}
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v60"
    dashboard["tests"] = tests
    dashboard["v60_confirm_only_dashboard"] = dash
    dashboard["v60_confirm_status"] = status
    dashboard["confirm_targets_v60"] = targets
    dashboard["public_claim_check_v60"] = claim
    _write_json(outdir / "confirm_only_dashboard_v60.json", dash)
    _write_json(outdir / "public_claim_check_v60.json", claim)
    _write_json(outdir / "confirm_targets_v60.json", {"schema": "ccdr-tierb-confirm-targets-v60", "targets": targets})
    final = {"schema": "ccdr-tierb-final-dashboard-v60", "confirmed_public_now": dash["confirmed_public_now"], "near_confirm_next": dash["near_confirm_next"], "anchor_only": dash["anchor_only"], "bound_only": dash["bound_only"], "public_claim_check": claim, "recommended_next_v60": ["T31/T32: fill strict measured κ(T)+SEM/TEM/XRD rows and pass balance/bootstrap/jackknife gates.", "T44: fill exact NAND fixture/manifest with true die_area_mm2 and bits_per_cell.", "T53: complete ProteinGym->UniProt/PDB/AlphaFold joined rows and FDR/bootstrap.", "T34/T57/T59/T45/T47: use exact row manifests only; broad discovery remains diagnostic.", "T26-T30: keep fusion diagnostic until exact public row tables appear."]}
    _write_json(outdir / "final_dashboard_v60.json", final)
    dashboard["final_dashboard_v60"] = final
    return dashboard


def enrich_fallback_v60(fallback: Dict[str, Any], test_id: str, td: Dict[str, Any], process_status: str, stdout_tail: str = "", stderr_tail: str = "") -> Dict[str, Any]:
    obj = dict(fallback)
    class A: pass
    args = A(); args.cache = Path(td.get("cache", "data/cache")) if isinstance(td, dict) else DATA_DIR / "cache"
    try:
        obj = apply_v60_result_overlay(obj, args, str(test_id).upper())
    except Exception as e:
        obj["v60_fallback_error"] = f"{type(e).__name__}: {e}"
    obj["schema"] = "ccdr-tierb-result-v60-fallback-repaired"
    obj["v60_fallback_context"] = {"process_status": process_status, "stdout_tail": (stdout_tail or "")[-800:], "stderr_tail": (stderr_tail or "")[-800:]}
    return obj
