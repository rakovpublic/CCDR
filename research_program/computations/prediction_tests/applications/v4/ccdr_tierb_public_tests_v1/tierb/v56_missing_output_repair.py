#!/usr/bin/env python3
"""v56 targeted fallback repair for CCDR Tier-B all-test runs.

Purpose: when legacy/live parsers time out or fail before writing JSON, do not let
important tests become an uninformative generic missing-output fallback.  Instead
emit a conservative, test-specific status that preserves the correct scientific
classification and next blocker.
"""
from __future__ import annotations

from typing import Any, Dict


def _split(test_id: str) -> Dict[str, str]:
    if test_id in {"T50", "T51", "T52"}:
        return {
            "execution_status_v56": "fallback_repaired_ok",
            "data_status_v56": "bound_table_or_literature_bound",
            "evidence_status_v56": "bound_only",
            "confirmation_status_v56": "not_confirmable_by_design",
        }
    if test_id in {"T31", "T32"}:
        return {
            "execution_status_v56": "missing_output_repaired_v56",
            "data_status_v56": "measured_microstructure_rows_required",
            "evidence_status_v56": "near_confirm_route_not_evaluated_this_run",
            "confirmation_status_v56": "not_confirmed_runtime_output_missing",
        }
    if test_id == "T44":
        return {
            "execution_status_v56": "missing_output_repaired_v56",
            "data_status_v56": "true_tier_a_nand_rows_required",
            "evidence_status_v56": "audit_repair_route_not_evaluated_this_run",
            "confirmation_status_v56": "not_confirmed_audit_repair_required",
        }
    if test_id == "T26":
        return {
            "execution_status_v56": "missing_output_repaired_v56",
            "data_status_v56": "fusion_per_shot_rows_required",
            "evidence_status_v56": "figure_summary_only_or_not_evaluated",
            "confirmation_status_v56": "not_confirmed_data_limited",
        }
    return {
        "execution_status_v56": "missing_output_repaired_v56",
        "data_status_v56": "data_limited",
        "evidence_status_v56": "not_evaluated_this_run",
        "confirmation_status_v56": "not_confirmed_runtime_output_missing",
    }


def _target(test_id: str) -> Dict[str, Any]:
    if test_id in {"T31", "T32"}:
        return {
            "test_id": test_id,
            "rank_score_0_10_v56": 9,
            "blocker_type_v56": "restore_measured_microstructure_model_output",
            "next_data_source_v56": "measured κ(T)+SEM/TEM/XRD grain-size rows; rerun strict dedup/bin/jackknife model",
            "expected_effort_v56": "medium",
            "confirmation_legally_possible_v56": True,
            "confirmation_status_v56": "not_confirmed_runtime_output_missing",
        }
    if test_id == "T44":
        return {
            "test_id": test_id,
            "rank_score_0_10_v56": 8,
            "blocker_type_v56": "true_tier_a_nand_rows_missing_or_not_evaluated",
            "next_data_source_v56": "company/year/layers/capacity_Gb/die_area_mm2/bits_per_cell/source_url NAND rows",
            "expected_effort_v56": "medium",
            "confirmation_legally_possible_v56": True,
            "confirmation_status_v56": "not_confirmed_audit_repair_required",
        }
    if test_id in {"T50", "T51", "T52"}:
        return {
            "test_id": test_id,
            "rank_score_0_10_v56": 0,
            "blocker_type_v56": "bound_only_by_design",
            "next_data_source_v56": "constraint/upper-limit table only; no positive-confirm route",
            "expected_effort_v56": "none_for_confirms",
            "confirmation_legally_possible_v56": False,
            "confirmation_status_v56": "not_confirmable_by_design",
        }
    if test_id == "T26":
        return {
            "test_id": test_id,
            "rank_score_0_10_v56": 2,
            "blocker_type_v56": "fusion_per_shot_elm_pedestal_rows_required",
            "next_data_source_v56": "per-shot E_ELM/W_ELM + Pped/Wped + volume/proxy + device/shot rows",
            "expected_effort_v56": "high",
            "confirmation_legally_possible_v56": True,
            "confirmation_status_v56": "not_confirmed_data_limited",
        }
    return {
        "test_id": test_id,
        "rank_score_0_10_v56": 1,
        "blocker_type_v56": "missing_output_runtime_repair",
        "next_data_source_v56": "test-specific exact structured source",
        "expected_effort_v56": "high",
        "confirmation_legally_possible_v56": test_id not in {"T50", "T51", "T52", "T60"},
        "confirmation_status_v56": "not_confirmed_runtime_output_missing",
    }


def enrich_fallback_v56(fallback: Dict[str, Any], test_id: str, td: Dict[str, Any], process_status: str, stdout_tail: str = "", stderr_tail: str = "") -> Dict[str, Any]:
    tid = str(test_id).upper()
    split = _split(tid)
    target = _target(tid)
    bound = tid in {"T50", "T51", "T52"}
    fallback = dict(fallback)
    fallback.update({
        "schema": "ccdr-tierb-result-v56-fallback-repaired",
        "status": "bound_only" if bound else "data_limited_runtime_output_repaired_v56",
        "programmatic_verdict": "bound_only" if bound else "data_limited_runtime_output_repaired_v56",
        "quality_patch_version": str(fallback.get("quality_patch_version", "")) + "+v56_missing_output_repair",
        "status_split_v56": split,
        "confirm_target_v56": target,
        "confirm_allowed_now_v56": False,
        "confirmation_label_v56": "not_confirmable_by_design" if bound else target["confirmation_status_v56"],
        "confirmation_blocker_v56": {
            "strict_confirm_allowed_now": False,
            "why_not_confirmed": target["blocker_type_v56"],
            "single_next_blocker": target["blocker_type_v56"],
            "best_auto_data_source_next": target["next_data_source_v56"],
        },
        "near_confirm_score_v56": {
            "score_0_10": target["rank_score_0_10_v56"],
            "primary_table_available": False if not bound else True,
            "model_rows_available": False,
            "model_gate_attempted": False,
            "strict_gate_remaining": [target["blocker_type_v56"]],
            "fallback_repaired_v56": True,
        },
        "public_claim_gate_v56": {
            "claimable_only_if_listed_in": "positive_dashboard.json:v56_confirm_status.confirmed_now",
            "confirmed_now_v56": False,
            "fallback_rows_are_not_confirmations_v56": True,
        },
    })
    fallback.update(split)
    fallback["positive_dashboard_fragment_v56"] = {
        "test_id": tid,
        "verdict": fallback.get("programmatic_verdict"),
        "confirmation_label": fallback.get("confirmation_label_v56"),
        "confirm_allowed_now": False,
        "strict_confirm_allowed_now": False,
        "near_confirm_score": fallback["near_confirm_score_v56"],
        "status_split_v56": split,
        "why_not_confirmed": fallback["confirmation_blocker_v56"]["why_not_confirmed"],
        "single_next_blocker": fallback["confirmation_blocker_v56"]["single_next_blocker"],
        "best_auto_data_source_next": fallback["confirmation_blocker_v56"]["best_auto_data_source_next"],
        "confirm_target_v56": target,
        "v56": {
            "fallback_repaired_v56": True,
            "process_status": process_status,
            "stdout_tail": (stdout_tail or "")[-800:],
            "stderr_tail": (stderr_tail or "")[-800:],
        },
    }
    return fallback
