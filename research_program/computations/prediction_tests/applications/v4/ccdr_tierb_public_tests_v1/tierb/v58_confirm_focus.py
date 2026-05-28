#!/usr/bin/env python3
"""v58 Tier-B confirm-focus patch.

Implements the 10 confirm-oriented improvements requested after the v57 report:
strict T31/T32 material loaders, exact T44/T53/T34/T57/T59/T45/T47 gates,
non-confirm fusion contracts, strict bound/anchor policy, and a confirm-only
public-claim dashboard.

This module is intentionally conservative: it can make a test *ready* only when
strict row/model gates pass, and public claims are only taken from
positive_dashboard.json -> v58_confirm_only_dashboard.confirmed_public_now.
"""
from __future__ import annotations

import csv
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

from . import v57_confirm_repairs as v57

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
    for row in rows:
        for k in (row or {}).keys():
            if k not in keys:
                keys.append(k)
    if not keys:
        keys = ["empty_v58"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean: Dict[str, Any] = {}
            for k in keys:
                vv = (row or {}).get(k)
                if isinstance(vv, (list, dict)):
                    clean[k] = json.dumps(vv, sort_keys=True, default=_jsonable)
                else:
                    clean[k] = _jsonable(vv)
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
        val = float(v)
        return val if math.isfinite(val) else None
    txt = str(v).replace(",", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
    if not m:
        return None
    try:
        val = float(m.group(0))
        return val if math.isfinite(val) else None
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
        nn = name.lower().replace("_", "")
        for k in row.keys():
            kk = str(k).lower().replace("_", "")
            if nn and nn in kk:
                return row[k]
    return None


def _read_patterns(patterns: Sequence[str]) -> List[Dict[str, Any]]:
    # Reuse v57 reader, then fall back to a slightly broader local reader.
    rows = v57._read_patterns(patterns)  # type: ignore[attr-defined]
    if rows or pd is None:
        return rows
    out: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    max_files = 80
    max_rows_per_file = 50000
    max_file_bytes = 50000000
    for root in [GEN_DIR, DATA_DIR, ROOT]:
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
                    if p.suffix.lower() in {".csv", ".tsv"}:
                        df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",", nrows=max_rows_per_file, dtype=str)
                    elif p.suffix.lower() == ".json":
                        obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                        vals = obj if isinstance(obj, list) else obj.get("rows", obj.get("data", [])) if isinstance(obj, dict) else []
                        df = pd.DataFrame(vals[:max_rows_per_file])
                    elif p.suffix.lower() == ".jsonl":
                        vals = [json.loads(line) for line in p.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
                        df = pd.DataFrame(vals[:max_rows_per_file])
                    else:
                        continue
                    for _, rr in df.iterrows():
                        d = dict(rr)
                        d["_source_file_v58"] = str(p)
                        out.append(d)
                except Exception:
                    continue
    return out


def _status(test_id: str, strict: bool, data: str, evidence: str, confirmation: str) -> Dict[str, str]:
    tid = test_id.upper()
    if tid in {"T50", "T51", "T52"}:
        return {
            "execution_status_v58": "ok",
            "data_status_v58": "bound_table_or_literature_bound",
            "evidence_status_v58": "bound_only",
            "confirmation_status_v58": "not_confirmable_by_design",
        }
    if tid == "T60":
        return {
            "execution_status_v58": "ok",
            "data_status_v58": "anchor_only",
            "evidence_status_v58": "positive_consistency_anchor",
            "confirmation_status_v58": "anchor_only_not_full_confirm",
        }
    return {
        "execution_status_v58": "ok",
        "data_status_v58": data,
        "evidence_status_v58": evidence,
        "confirmation_status_v58": "compatible_positive_confirm_allowed" if (strict and tid == "T48") else ("strict_confirm_ready_not_public_claim_until_review" if strict else confirmation),
    }


def _target(test_id: str, score: int, blocker: str, next_source: str, confirmation_status: str) -> Dict[str, Any]:
    tid = test_id.upper()
    return {
        "schema": "ccdr-tierb-confirm-target-v58",
        "test_id": tid,
        "rank_score_0_10_v58": int(score),
        "blocker_type_v58": blocker,
        "next_data_source_v58": next_source,
        "expected_effort_v58": "low" if tid == "T48" else ("medium" if tid in {"T31", "T32", "T44", "T53", "T34", "T29"} else "high"),
        "confirmation_legally_possible_v58": tid not in {"T50", "T51", "T52", "T60"},
        "confirmation_status_v58": confirmation_status,
    }


# 1/2. Strict T31/T32 materials loader + explicit row rejection diagnostics.
def materials_confirm_v58(test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    base = v57.materials_confirm_v57(tid)
    model = dict(base.get("model") or {})
    # v58 explicit source contract + concise gate summary.  The v57 function already
    # writes normalized/dedup/rejection/jackknife artifacts; v58 adds a strict contract.
    required_columns = [
        "source_url", "sample_id", "material", "material_family", "temperature_K", "kappa_W_mK",
        "grain_size_nm_or_um", "microstructure_method", "nanocrystalline_yes_no", "boundary_density_proxy",
    ]
    failed = list(base.get("failed_subgates") or model.get("failed_subgates_v57") or [])
    # Enforce the exact contract in v58 naming.
    exact_contract = {
        "test_id": tid,
        "schema": "ccdr-materials-strict-contract-v58",
        "required_columns_v58": required_columns,
        "n_raw_rows_v58": int(model.get("n_raw_rows_v57") or 0),
        "n_dedup_rows_v58": int(model.get("n_dedup_rows_v57") or 0),
        "n_usable_rows_v58": int(model.get("n_usable_rows_v57") or 0),
        "n_source_groups_v58": int(model.get("n_source_groups_v57") or 0),
        "n_material_families_v58": int(model.get("n_material_families_v57") or 0),
        "n_temperature_bins_v58": int(model.get("n_temperature_bins_v57") or 0),
        "strict_confirm_ready_v58": bool(base.get("strict_confirm_ready_v57")) and not failed,
        "failed_subgates_v58": "|".join(failed),
        "public_claim_policy_v58": "near-confirm until strict gates pass and dashboard lists it under confirmed_public_now",
    }
    if exact_contract["n_usable_rows_v58"] < 20 and ">=20_dedup_measured_microstructure_rows" not in failed:
        failed.append(">=20_dedup_measured_microstructure_rows")
    if exact_contract["n_source_groups_v58"] < 5 and ">=5_independent_source_groups" not in failed:
        failed.append(">=5_independent_source_groups")
    if exact_contract["n_material_families_v58"] < 5 and ">=5_material_families" not in failed:
        failed.append(">=5_material_families")
    if exact_contract["n_temperature_bins_v58"] < 3 and ">=3_temperature_bins" not in failed:
        failed.append(">=3_temperature_bins")
    exact_contract["failed_subgates_v58"] = "|".join(failed)
    exact_contract["strict_confirm_ready_v58"] = bool(base.get("strict_confirm_ready_v57")) and not failed
    score = 10 if exact_contract["strict_confirm_ready_v58"] else (9 if exact_contract["n_usable_rows_v58"] >= 20 else 7 if exact_contract["n_usable_rows_v58"] >= 10 else 4)
    exact_contract["score_0_10_v58"] = score
    _write_csv([exact_contract], f"{tid.lower()}_materials_strict_contract_v58.csv")
    _write_json(_ensure() / f"{tid.lower()}_materials_strict_contract_v58.json", exact_contract)
    return {
        "schema": "ccdr-materials-confirm-v58",
        "test_id": tid,
        "status": "strict_materials_confirm_ready_v58" if exact_contract["strict_confirm_ready_v58"] else "materials_confirm_gates_pending_v58",
        "strict_confirm_ready_v58": bool(exact_contract["strict_confirm_ready_v58"]),
        "score_0_10_v58": score,
        "failed_subgates_v58": failed,
        "contract": exact_contract,
        "base_v57": base,
        "artifacts": {
            "strict_contract_json": f"data/generated/{tid.lower()}_materials_strict_contract_v58.json",
            "strict_contract_csv": f"data/generated/{tid.lower()}_materials_strict_contract_v58.csv",
            "v57_dedup_rows": (base.get("artifacts") or {}).get("dedup_rows"),
            "v57_rejection_diagnostics": (base.get("artifacts") or {}).get("rejection_diagnostics"),
            "v57_jackknife": (base.get("artifacts") or {}).get("jackknife"),
        },
    }


# 3. T44 true Tier-A NAND exact parser/audit.
def t44_nand_tier_a_v58() -> Dict[str, Any]:
    base = v57.nand_tier_a_v57()
    summary = dict(base.get("summary") or {})
    required = ["company", "year", "layers", "capacity_Gb", "die_area_mm2", "bits_per_cell", "source_url"]
    contract = {
        "schema": "ccdr-t44-nand-tier-a-contract-v58",
        "test_id": "T44",
        "required_columns_v58": required,
        "n_raw_rows_v58": int(summary.get("n_raw_rows_v57") or 0),
        "usable_tier_a_rows_v58": int(summary.get("usable_tier_a_rows_v57") or 0),
        "n_companies_v58": int(summary.get("n_companies_v57") or 0),
        "strict_confirm_ready_v58": bool(base.get("strict_confirm_ready_v57")),
        "failed_subgates_v58": "|".join(base.get("failed_subgates") or []),
        "derived_die_area_policy_v58": "audit_only_never_confirm",
    }
    _write_csv([contract], "t44_true_tier_a_contract_v58.csv")
    _write_json(_ensure() / "t44_true_tier_a_contract_v58.json", contract)
    return {
        "schema": "ccdr-t44-nand-tier-a-v58",
        "status": "t44_true_tier_a_confirm_ready_v58" if contract["strict_confirm_ready_v58"] else "t44_true_tier_a_audit_repair_required_v58",
        "strict_confirm_ready_v58": bool(contract["strict_confirm_ready_v58"]),
        "score_0_10_v58": 10 if contract["strict_confirm_ready_v58"] else 8,
        "contract": contract,
        "base_v57": base,
        "artifacts": {"contract": "data/generated/t44_true_tier_a_contract_v58.json", "base_rows": (base.get("artifacts") or {}).get("rows"), "base_rejections": (base.get("artifacts") or {}).get("rejections")},
    }


# 4. T53 ProteinGym structure join gate.
def t53_structure_join_v58() -> Dict[str, Any]:
    base = v57.t53_structure_join_v57()
    summary = dict(base.get("summary") or {})
    strict_rows = int(summary.get("usable_join_rows_v57") or 0)
    families = int(summary.get("n_families_v57") or 0)
    assays = int(summary.get("n_assays_v57") or 0)
    clusters = int(summary.get("n_sequence_clusters_v57") or 0)
    failed = []
    if strict_rows < 50: failed.append(">=50_ProteinGym_structure_join_rows")
    if families < 5: failed.append(">=5_families")
    if assays < 2: failed.append(">=2_assay_types")
    if clusters < 10: failed.append(">=10_sequence_clusters")
    failed.extend(["family_assay_sequence_jackknife_model", "BH_FDR_or_bootstrap_significance"])
    gate = {
        "schema": "ccdr-t53-proteingym-structure-model-gate-v58",
        "test_id": "T53",
        "usable_join_rows_v58": strict_rows,
        "n_families_v58": families,
        "n_assays_v58": assays,
        "n_sequence_clusters_v58": clusters,
        "required_model_gates_v58": ["family_jackknife", "assay_jackknife", "sequence_cluster_jackknife", "BH_FDR_or_bootstrap"],
        "strict_confirm_ready_v58": False,
        "failed_subgates_v58": "|".join(failed),
    }
    _write_csv([gate], "t53_proteingym_structure_model_gate_v58.csv")
    _write_json(_ensure() / "t53_proteingym_structure_model_gate_v58.json", gate)
    return {"schema": "ccdr-t53-structure-join-v58", "status": "t53_structure_join_model_pending_v58", "strict_confirm_ready_v58": False, "score_0_10_v58": 8 if strict_rows >= 50 else 6, "failed_subgates_v58": failed, "gate": gate, "base_v57": base}


# 5. T48 frozen confirm + robustness only.
def t48_robustness_v58() -> Dict[str, Any]:
    base = v57.t48_robustness_v57()
    summary = dict(base.get("summary") or {})
    gate = {
        "schema": "ccdr-t48-frozen-confirm-robustness-v58",
        "test_id": "T48",
        "confirm_allowed_now_v58": True,
        "confirmed_public_now_v58": True,
        "gate_policy_v58": "frozen compatible-positive; robustness-only audit, no moving gate",
        "n_usable_rows_v58": int(summary.get("n_usable_rows_v57") or summary.get("n_rows_v57") or 0),
        "descriptor_corr_v58": summary.get("descriptor_corr_v57"),
        "permutation_p_two_sided_v58": summary.get("permutation_p_two_sided_v57"),
        "n_jackknife_checks_v58": int(summary.get("n_jackknife_checks_v57") or 0),
    }
    _write_csv([gate], "t48_frozen_confirm_robustness_v58.csv")
    _write_json(_ensure() / "t48_frozen_confirm_robustness_v58.json", gate)
    return {"schema": "ccdr-t48-robustness-v58", "status": "compatible_positive_confirm_allowed_v58", "strict_confirm_ready_v58": True, "score_0_10_v58": 10, "gate": gate, "base_v57": base}


# 6. T34 exact thermoelectric rows.
def t34_te_exact_rows_v58() -> Dict[str, Any]:
    raw = _read_patterns(["t34*te*rows_v*.csv", "t34*thermo*rows_v*.csv", "*teMatDb*.csv", "*Starrydata*.csv", "*Bi2Te3*.csv", "*Sb2Te3*.csv"])
    rows: List[Dict[str, Any]] = []
    reject: List[Dict[str, Any]] = []
    for i, r in enumerate(raw):
        material = _s(_pick(r, ["material", "compound", "formula", "name"]))
        zt = _f(_pick(r, ["ZT", "zt", "zT", "figure_of_merit"]))
        temp = _f(_pick(r, ["temperature_K", "T_K", "temperature", "temp_K"]))
        angle = _f(_pick(r, ["orientation_angle_deg", "orientation", "angle_deg", "grain_boundary_angle_deg", "theta_deg"]))
        gb_angle = _f(_pick(r, ["grain_boundary_angle_deg", "gb_angle", "grain_angle_deg"]))
        url = _s(_pick(r, ["source_url", "url", "doi", "reference", "_source_file_v58", "_source_file_v57"]))
        txt = " ".join(_s(v) for v in r.values())
        target_mat = bool(re.search(r"Bi\s*2\s*Te\s*3|Bi2Te3|Sb\s*2\s*Te\s*3|Sb2Te3|bismuth telluride|antimony telluride", material + " " + txt, re.I))
        reasons = []
        if not target_mat: reasons.append("not_Bi2Te3_or_Sb2Te3")
        if zt is None: reasons.append("missing_ZT")
        if temp is None: reasons.append("missing_temperature_K")
        if angle is None and gb_angle is None: reasons.append("missing_orientation_or_grain_boundary_angle")
        if not url: reasons.append("missing_source_url")
        row = {"raw_index_v58": i, "material_v58": material, "ZT_v58": zt, "temperature_K_v58": temp, "orientation_angle_deg_v58": angle, "grain_boundary_angle_deg_v58": gb_angle, "source_url_v58": url, "usable_exact_te_row_v58": not reasons, "reject_reasons_v58": "|".join(reasons)}
        rows.append(row)
        if reasons: reject.append(row)
    usable = [r for r in rows if r["usable_exact_te_row_v58"]]
    failed = []
    if len(usable) < 20: failed.append(">=20_exact_ZT_temperature_angle_rows")
    mats = {r["material_v58"] for r in usable if r.get("material_v58")}
    if len(mats) < 2: failed.append("Bi2Te3_and_Sb2Te3_or_multiple_materials")
    gate = {"schema": "ccdr-t34-exact-te-rows-v58", "test_id": "T34", "n_raw_rows_v58": len(raw), "usable_exact_rows_v58": len(usable), "n_materials_v58": len(mats), "strict_confirm_ready_v58": False, "failed_subgates_v58": "|".join(failed + ["orientation_model_and_jackknife_required"])}
    _write_csv(rows, "t34_exact_te_orientation_zt_rows_v58.csv")
    _write_csv(reject, "t34_exact_te_orientation_zt_rejections_v58.csv")
    _write_json(_ensure() / "t34_exact_te_orientation_zt_gate_v58.json", gate)
    return {"schema": "ccdr-t34-te-exact-v58", "status": "t34_exact_te_rows_pending_v58", "strict_confirm_ready_v58": False, "score_0_10_v58": 5 if len(usable) >= 20 else 3, "gate": gate, "artifacts": {"rows": "data/generated/t34_exact_te_orientation_zt_rows_v58.csv", "rejections": "data/generated/t34_exact_te_orientation_zt_rejections_v58.csv"}}


# 7. T57/T59 exact HEPData manifest.
def hepdata_manifest_gate_v58(test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    raw = _read_patterns([f"{tid.lower()}*hepdata*manifest*.csv", f"{tid.lower()}*registry*.csv", "hepdata_exact_registry*.csv", "*hepdata*record*table*column*.csv"])
    rows: List[Dict[str, Any]] = []
    reject: List[Dict[str, Any]] = []
    for i, r in enumerate(raw):
        record = _s(_pick(r, ["record_id", "record", "hepdata_record", "inspire_id"]))
        table = _s(_pick(r, ["table_id", "table", "table_name"]))
        xcol = _s(_pick(r, ["x_column", "x", "bin_column", "independent_variable"]))
        obs = _s(_pick(r, ["observed_column", "observed", "data_column", "y_column"]))
        model = _s(_pick(r, ["model_column", "expected_column", "prediction_column", "theory_column"]))
        unc = _s(_pick(r, ["uncertainty_column", "error_column", "stat_error", "total_error"]))
        observable = _s(_pick(r, ["observable_name", "observable", "quantity"]))
        reasons = []
        for name, val in [("record_id", record), ("table_id", table), ("x_column", xcol), ("observed_column", obs), ("model_column", model), ("uncertainty_column", unc), ("observable_name", observable)]:
            if not val: reasons.append(f"missing_{name}")
        row = {"raw_index_v58": i, "record_id_v58": record, "table_id_v58": table, "x_column_v58": xcol, "observed_column_v58": obs, "model_column_v58": model, "uncertainty_column_v58": unc, "observable_name_v58": observable, "usable_manifest_row_v58": not reasons, "reject_reasons_v58": "|".join(reasons)}
        rows.append(row)
        if reasons: reject.append(row)
    usable = [r for r in rows if r["usable_manifest_row_v58"]]
    gate = {"schema": "ccdr-hepdata-exact-manifest-v58", "test_id": tid, "n_manifest_rows_v58": len(rows), "usable_manifest_rows_v58": len(usable), "strict_confirm_ready_v58": False, "failed_subgates_v58": "official_YAML_JSON_parse_and_model_residual_required" if usable else "exact_HEPData_manifest_required"}
    _write_csv(rows, f"{tid.lower()}_hepdata_exact_manifest_audit_v58.csv")
    _write_csv(reject, f"{tid.lower()}_hepdata_exact_manifest_rejections_v58.csv")
    _write_json(_ensure() / f"{tid.lower()}_hepdata_exact_manifest_gate_v58.json", gate)
    return {"schema": "ccdr-hepdata-manifest-gate-v58", "test_id": tid, "status": "hepdata_manifest_rows_available_model_pending_v58" if usable else "hepdata_exact_manifest_missing_v58", "strict_confirm_ready_v58": False, "score_0_10_v58": 5 if usable else 3, "gate": gate}


# 8. T45/T47 exact benchmark gates.
def benchmark_gate_v58(test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    if tid == "T45":
        raw = _read_patterns(["t45*benchmark*.csv", "*optical*interconnect*.csv", "*energy*bit*.csv"])
        required = ["energy_per_bit", "bandwidth", "reach", "year", "source_url"]
    else:
        raw = _read_patterns(["t47*benchmark*.csv", "*neuromorphic*benchmark*.csv", "*loihi*.csv", "*spinnaker*.csv", "*truenorth*.csv"])
        required = ["chip_or_system", "benchmark", "energy_per_inference_or_spike", "accuracy", "source_url"]
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(raw):
        vals = {name: _s(_pick(r, [name, name.replace("_", " "), name.replace("_", "")])) for name in required}
        # numeric-friendly aliases
        if tid == "T45":
            vals["energy_per_bit"] = vals["energy_per_bit"] or _s(_pick(r, ["pJ_per_bit", "fJ_per_bit", "energy_bit", "energy/bit"]))
            vals["bandwidth"] = vals["bandwidth"] or _s(_pick(r, ["Gbps", "bandwidth_Gbps", "data_rate"]))
        else:
            vals["energy_per_inference_or_spike"] = vals["energy_per_inference_or_spike"] or _s(_pick(r, ["energy_per_inference", "energy_per_spike", "J_per_inference", "nJ_per_inference"]))
        reasons = [f"missing_{k}" for k, v in vals.items() if not v]
        rows.append({"raw_index_v58": i, **{f"{k}_v58": v for k, v in vals.items()}, "usable_benchmark_row_v58": not reasons, "reject_reasons_v58": "|".join(reasons)})
    usable = [r for r in rows if r["usable_benchmark_row_v58"]]
    gate = {"schema": "ccdr-exact-benchmark-gate-v58", "test_id": tid, "required_columns_v58": required, "n_raw_rows_v58": len(raw), "usable_benchmark_rows_v58": len(usable), "strict_confirm_ready_v58": False, "failed_subgates_v58": ">=20_exact_benchmark_rows_and_baseline_model" if len(usable) < 20 else "baseline_model_and_jackknife_required"}
    _write_csv(rows, f"{tid.lower()}_exact_benchmark_rows_v58.csv")
    _write_json(_ensure() / f"{tid.lower()}_exact_benchmark_gate_v58.json", gate)
    return {"schema": "ccdr-exact-benchmark-v58", "test_id": tid, "status": "exact_benchmark_rows_pending_v58", "strict_confirm_ready_v58": False, "score_0_10_v58": 5 if len(usable) >= 20 else 3, "gate": gate}


# 9. Fusion strict non-confirm policy.
def fusion_contract_v58(test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    req = {
        "T26": ["shot_or_discharge", "device", "E_ELM_or_W_ELM", "Pped_or_Wped", "volume_or_proxy"],
        "T27": ["discharge", "device", "RMP_current_or_amplitude", "phasing", "toroidal_mode", "ELM_frequency_response"],
        "T28": ["timeslice", "device", "tau_E", "H98", "n_e", "P_heat", "q95"],
        "T29": ["device", "configuration", "radius_or_profile_coordinate", "chi_or_transport", "density_or_temperature_profile"],
        "T30": ["row_id", "curvature_or_residual", "device_or_profile", "uncertainty"],
    }.get(tid, ["exact_public_row_table"])
    gate = {"schema": "ccdr-fusion-row-contract-v58", "test_id": tid, "required_columns_v58": req, "strict_confirm_ready_v58": False, "confirmation_status_v58": "not_confirmed_data_limited", "policy_v58": "PDF/text/summary extraction is diagnostic only; strict confirmation requires exact public physical row tables"}
    _write_csv([gate], f"{tid.lower()}_fusion_row_contract_v58.csv")
    _write_json(_ensure() / f"{tid.lower()}_fusion_row_contract_v58.json", gate)
    return {"schema": "ccdr-fusion-contract-v58", "test_id": tid, "status": "fusion_exact_row_table_required_v58", "strict_confirm_ready_v58": False, "score_0_10_v58": {"T29": 2, "T28": 2, "T27": 1, "T26": 1, "T30": 1}.get(tid, 1), "gate": gate}


# 10. Confirm-only dashboard export.
def confirm_only_dashboard_v58(status: Dict[str, List[str]]) -> Dict[str, Any]:
    return {
        "schema": "ccdr-tierb-confirm-only-dashboard-v58",
        "confirmed_public_now": status.get("confirmed_public_now", []),
        "near_confirm_next": status.get("near_confirm_next", []),
        "anchor_only": status.get("anchor_only", []),
        "bound_only": status.get("bound_only", []),
        "do_not_claim": status.get("do_not_claim", []),
        "public_claim_rule_v58": "Only tests listed in confirmed_public_now may be described as current public confirms.",
    }


def apply_v58_result_overlay(obj: Dict[str, Any], args: Any, test_id: str) -> Dict[str, Any]:
    tid = test_id.upper()
    arts: Dict[str, Any] = dict(obj.get("auto_data_improvements_v57") or {})
    strict = False
    score = 1
    blocker = "exact_structured_public_source_required"
    next_source = "test-specific exact structured source"
    data = "data_limited"
    evidence = "not_confirmed"
    confirmation = "not_confirmed_data_limited"
    try:
        if tid in {"T31", "T32"}:
            arts["materials_confirm_v58"] = materials_confirm_v58(tid)
            strict = bool(arts["materials_confirm_v58"].get("strict_confirm_ready_v58"))
            score = int(arts["materials_confirm_v58"].get("score_0_10_v58") or 9)
            blocker = "measured_microstructure_confirm_gates_pending" if not strict else "strict_microstructure_confirm_ready"
            next_source = "dedup measured κ(T)+SEM/TEM/XRD rows with source/material/temp-bin jackknives"
            data = "measured_microstructure_rows_available_or_required"
            evidence = "near_confirm_or_confirm_ready"
            confirmation = "not_confirmed_next_gate_required"
        elif tid == "T44":
            arts["nand_tier_a_v58"] = t44_nand_tier_a_v58()
            strict = bool(arts["nand_tier_a_v58"].get("strict_confirm_ready_v58"))
            score = int(arts["nand_tier_a_v58"].get("score_0_10_v58") or 8)
            blocker = "true_tier_a_nand_rows_required" if not strict else "true_tier_a_nand_ready_review_required"
            next_source = "company/year/layers/capacity_Gb/die_area_mm2/bits_per_cell/source_url NAND rows"
            data = "true_tier_a_nand_rows_required"
            evidence = "audit_repair_route"
            confirmation = "not_confirmed_audit_repair_required"
        elif tid == "T48":
            arts["t48_robustness_v58"] = t48_robustness_v58()
            strict = True
            score = 10
            blocker = "robustness_only_audit_not_gate_change"
            next_source = "frozen PV descriptor rows plus family/source/year/permutation robustness artifacts"
            data = "pv_descriptor_rows_or_frozen_artifact"
            evidence = "compatible_positive"
            confirmation = "compatible_positive_confirm_allowed"
        elif tid == "T53":
            arts["t53_structure_join_v58"] = t53_structure_join_v58()
            strict = False
            score = int(arts["t53_structure_join_v58"].get("score_0_10_v58") or 6)
            blocker = "ProteinGym_structure_join_model_FDR_required"
            next_source = "ProteinGym CSV joined to UniProt/PDB/AlphaFold structural features + family/assay/sequence jackknife"
            data = "dms_structure_join_rows_missing_or_model_pending"
            evidence = "near_confirm_or_model_ready"
            confirmation = "not_confirmed_next_gate_required"
        elif tid == "T34":
            arts["t34_te_exact_rows_v58"] = t34_te_exact_rows_v58()
            strict = False
            score = int(arts["t34_te_exact_rows_v58"].get("score_0_10_v58") or 3)
            blocker = "exact_te_rows_missing_or_model_pending"
            next_source = "exact teMatDb/Starrydata Bi2Te3/Sb2Te3 ZT+temperature+angle export"
            data = "exact_te_rows_required"
            evidence = "near_confirm_data_limited"
            confirmation = "not_confirmed_data_limited"
        elif tid in {"T57", "T59"}:
            arts[f"{tid.lower()}_hepdata_manifest_v58"] = hepdata_manifest_gate_v58(tid)
            strict = False
            score = int(arts[f"{tid.lower()}_hepdata_manifest_v58"].get("score_0_10_v58") or 3)
            blocker = "exact_HEPData_manifest_and_model_required"
            next_source = "exact HEPData record/table/column manifest with observed/model/uncertainty columns"
            data = "hepdata_exact_manifest_required"
            evidence = "data_limited_positive_path"
            confirmation = "not_confirmed_data_limited"
        elif tid in {"T45", "T47"}:
            arts[f"{tid.lower()}_benchmark_v58"] = benchmark_gate_v58(tid)
            strict = False
            score = int(arts[f"{tid.lower()}_benchmark_v58"].get("score_0_10_v58") or 3)
            blocker = "exact_benchmark_table_required"
            next_source = "exact public benchmark supplement/table"
            data = "exact_benchmark_rows_required"
            evidence = "data_limited_positive_path"
            confirmation = "not_confirmed_data_limited"
        elif tid in {"T26", "T27", "T28", "T29", "T30"}:
            arts[f"{tid.lower()}_fusion_contract_v58"] = fusion_contract_v58(tid)
            strict = False
            score = int(arts[f"{tid.lower()}_fusion_contract_v58"].get("score_0_10_v58") or 1)
            blocker = "raw_fusion_row_table_required_for_confirm"
            next_source = "exact fusion physical row table; PDF summaries are diagnostic only"
            data = "fusion_exact_rows_required"
            evidence = "diagnostic_or_preliminary_only"
            confirmation = "not_confirmed_data_limited"
        elif tid in {"T50", "T51", "T52"}:
            strict = False; score = 0; blocker = "bound_only_by_design"; next_source = "constraint/upper-limit table only; no positive-confirm route"; data = "bound_table_or_literature_bound"; evidence = "bound_only"; confirmation = "not_confirmable_by_design"
        elif tid == "T60":
            strict = False; score = 5; blocker = "T60b_T60c_T60d_required"; next_source = "quark/lattice uncertainty + sector reshuffle + look-elsewhere registry"; data = "anchor_only"; evidence = "positive_consistency_anchor"; confirmation = "anchor_only_not_full_confirm"
        else:
            prev = obj.get("confirm_target_v57") or obj.get("confirm_target_v56") or obj.get("confirm_target_v55") or obj.get("confirm_target_v53") or {}
            for k in ["rank_score_0_10_v57", "rank_score_0_10_v56", "rank_score_0_10_v55", "rank_score_0_10_v53"]:
                if isinstance(prev, dict) and prev.get(k) is not None:
                    score = int(prev.get(k) or 1); break
            blocker = str(prev.get("blocker_type_v57") or prev.get("blocker_type_v56") or prev.get("blocker_type_v55") or blocker)
            next_source = str(prev.get("next_data_source_v57") or prev.get("next_data_source_v56") or prev.get("next_data_source_v55") or next_source)
    except Exception as e:
        arts["v58_overlay_error"] = f"{type(e).__name__}: {e}"
        strict = False
        blocker = "v58_overlay_exception"
        confirmation = "not_confirmed_runtime_output_missing"
        score = {"T31": 9, "T32": 9, "T44": 8, "T53": 6, "T34": 3}.get(tid, 1)
    if tid in {"T50", "T51", "T52", "T60"}:
        strict = False
    if tid != "T48":
        # v58 keeps non-T48 strict passes as review-required, not public confirmed.
        public_confirm = False
    else:
        public_confirm = bool(strict)
    status = _status(tid, strict, data, evidence, confirmation)
    target = _target(tid, score, blocker, next_source, status["confirmation_status_v58"])
    obj["auto_data_improvements_v58"] = arts
    obj["confirm_allowed_now_v58"] = public_confirm
    obj["confirmation_label_v58"] = "compatible_positive" if public_confirm else status["confirmation_status_v58"]
    obj["status_split_v58"] = status
    obj["confirm_target_v58"] = target
    obj["confirmation_blocker_v58"] = {"strict_confirm_allowed_now": bool(strict), "public_confirm_allowed_now": public_confirm, "why_not_confirmed": None if public_confirm else blocker, "single_next_blocker": "robustness_only" if public_confirm else blocker, "best_auto_data_source_next": next_source}
    obj["near_confirm_score_v58"] = {"score_0_10": int(score), "primary_table_available": score >= 3, "model_rows_available": score >= 6, "model_gate_attempted": score >= 7, "strict_gate_remaining": [] if public_confirm else [blocker]}
    obj.update(status)
    obj["public_claim_gate_v58"] = {"claimable_only_if_listed_in": "positive_dashboard.json:v58_confirm_only_dashboard.confirmed_public_now", "confirmed_now_v58": public_confirm, "legacy_confirm_fields_are_not_public_claims": True}
    obj["positive_dashboard_fragment_v58"] = {"test_id": tid, "verdict": obj.get("programmatic_verdict") or obj.get("status"), "confirmation_label": obj["confirmation_label_v58"], "confirm_allowed_now": public_confirm, "strict_confirm_allowed_now": public_confirm, "near_confirm_score": obj["near_confirm_score_v58"], "status_split_v58": status, "why_not_confirmed": obj["confirmation_blocker_v58"]["why_not_confirmed"], "single_next_blocker": obj["confirmation_blocker_v58"]["single_next_blocker"], "best_auto_data_source_next": next_source, "confirm_target_v58": target, "v58": {"auto_data_improvements_v58": arts, "confirmation_blocker_v58": obj["confirmation_blocker_v58"], "near_confirm_score_v58": obj["near_confirm_score_v58"], "status_split_v58": status, "public_claim_gate_v58": obj["public_claim_gate_v58"]}}
    obj["quality_patch_version"] = str(obj.get("quality_patch_version", "")) + "+v58_confirm_focus"
    return obj


def enrich_fallback_v58(fallback: Dict[str, Any], test_id: str, td: Dict[str, Any], process_status: str, stdout_tail: str = "", stderr_tail: str = "") -> Dict[str, Any]:
    obj = dict(fallback)
    class A:
        pass
    args = A(); args.cache = Path(td.get("cache", "data/cache")) if isinstance(td, dict) else DATA_DIR / "cache"
    try:
        obj = apply_v58_result_overlay(obj, args, str(test_id).upper())
    except Exception as e:
        obj["v58_fallback_error"] = f"{type(e).__name__}: {e}"
    obj["schema"] = "ccdr-tierb-result-v58-fallback-repaired"
    obj.setdefault("status", "data_limited_runtime_output_repaired_v58")
    obj["v58_fallback_context"] = {"process_status": process_status, "stdout_tail": (stdout_tail or "")[-800:], "stderr_tail": (stderr_tail or "")[-800:]}
    return obj


def apply_dashboard_v58(dashboard: Dict[str, Any], outdir: Path) -> Dict[str, Any]:
    outdir = Path(outdir)
    dashboard = dict(dashboard)
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v58"
    order = ["T48", "T31", "T32", "T44", "T53", "T34", "T29", "T57", "T59", "T45", "T47", "T28", "T26", "T27", "T30", "T60", "T50", "T51", "T52"]
    tests: List[Dict[str, Any]] = []
    targets: List[Dict[str, Any]] = []
    counts = {"execution": {}, "data": {}, "evidence": {}, "confirmation": {}}
    status_lists = {"confirmed_public_now": [], "near_confirm_next": [], "anchor_only": [], "bound_only": [], "do_not_claim": [], "fusion_row_required": [], "exact_manifest_required": []}
    def inc(bucket: str, key: Any) -> None:
        kk = str(key or "unknown")
        counts[bucket][kk] = counts[bucket].get(kk, 0) + 1
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        latest = frag
        if tid:
            rf = outdir / f"{str(tid).lower()}_result.json"
            if rf.exists():
                try:
                    rr = json.loads(rf.read_text(encoding="utf-8"))
                    latest = rr.get("positive_dashboard_fragment_v58") or rr.get("positive_dashboard_fragment_v57") or rr.get("positive_dashboard_fragment_v56") or frag
                except Exception:
                    latest = frag
        tests.append(latest)
        split = latest.get("status_split_v58") or latest.get("status_split_v57") or latest.get("status_split_v56") or {}
        inc("execution", split.get("execution_status_v58") or split.get("execution_status_v57") or split.get("execution_status_v56"))
        inc("data", split.get("data_status_v58") or split.get("data_status_v57") or split.get("data_status_v56"))
        inc("evidence", split.get("evidence_status_v58") or split.get("evidence_status_v57") or split.get("evidence_status_v56"))
        inc("confirmation", split.get("confirmation_status_v58") or split.get("confirmation_status_v57") or split.get("confirmation_status_v56"))
        target = latest.get("confirm_target_v58") or latest.get("confirm_target_v57") or latest.get("confirm_target_v56")
        if target:
            targets.append(target)
        if tid == "T48" and latest.get("confirm_allowed_now"):
            status_lists["confirmed_public_now"].append(tid)
        elif tid in {"T31", "T32", "T44", "T53", "T34", "T57", "T59", "T45", "T47"}:
            status_lists["near_confirm_next"].append(tid)
        if tid == "T60": status_lists["anchor_only"].append(tid)
        if tid in {"T50", "T51", "T52"}: status_lists["bound_only"].append(tid)
        if tid in {"T26", "T27", "T28", "T29", "T30"}: status_lists["fusion_row_required"].append(tid)
        if tid in {"T34", "T45", "T47", "T57", "T59"}: status_lists["exact_manifest_required"].append(tid)
        if tid != "T48": status_lists["do_not_claim"].append(tid)
    def sort(vals: Iterable[str]) -> List[str]:
        return sorted(set(x for x in vals if x), key=lambda x: order.index(x) if x in order else 99)
    for k in status_lists:
        status_lists[k] = sort(status_lists[k])
    def score(d: Dict[str, Any]) -> int:
        for key in ["rank_score_0_10_v58", "rank_score_0_10_v57", "rank_score_0_10_v56"]:
            if isinstance(d, dict) and d.get(key) is not None:
                try: return int(d.get(key) or 0)
                except Exception: return 0
        return 0
    targets = sorted(targets, key=lambda d: (-score(d), order.index(d.get("test_id")) if isinstance(d, dict) and d.get("test_id") in order else 99))
    confirm_only = confirm_only_dashboard_v58(status_lists)
    dashboard["tests"] = tests
    dashboard["v58_confirm_only_dashboard"] = confirm_only
    dashboard["v58_confirm_status"] = status_lists
    dashboard["status_split_counts_v58"] = counts
    dashboard["confirm_targets_v58"] = targets
    claim_check = {"schema": "ccdr-public-claim-check-v58", "allowed_confirm_source": "positive_dashboard.json:v58_confirm_only_dashboard.confirmed_public_now", "confirmed_public_now": confirm_only["confirmed_public_now"], "pass_v58": confirm_only["confirmed_public_now"] == ["T48"], "message_v58": "Only confirmed_public_now may be used for public confirm claims."}
    dashboard["public_claim_check_v58"] = claim_check
    _write_json(outdir / "public_claim_check_v58.json", claim_check)
    _write_json(outdir / "confirm_targets_v58.json", {"schema": "ccdr-tierb-confirm-targets-v58", "targets": targets})
    _write_json(outdir / "confirm_only_dashboard_v58.json", confirm_only)
    dashboard["recommended_next_v58"] = [
        "Use only v58_confirm_only_dashboard.confirmed_public_now for public confirm claims; currently T48 only.",
        "T31/T32: load strict measured κ(T)+SEM/TEM/XRD rows and pass source/material/temp-bin jackknife gates.",
        "T44: require true Tier-A NAND rows; derived die-area rows remain audit-only.",
        "T53: complete ProteinGym -> UniProt/PDB/AlphaFold model with FDR/bootstrap and family/assay/sequence jackknives.",
        "T34/T45/T47/T57/T59: exact manifests/tables only; broad discovery is diagnostic.",
        "T26-T30: fusion PDFs/summaries are diagnostic only; strict confirm requires exact physical row tables.",
        "T50-T52 bound-only and T60 anchor-only must not be promoted.",
    ]
    return dashboard
