#!/usr/bin/env python3
"""Run all CCDR v7.5 Tier-B public-data tests T26-T60.

Every test downloads public data/sources into --cache. No manual input files are required.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tierb.tierb_catalog import TESTS


def main():
    p = argparse.ArgumentParser(description="Run all Tier-B CCDR public-data tests")
    p.add_argument("--cache", type=Path, default=Path("tierb_cache"))
    p.add_argument("--outdir", type=Path, default=Path("tierb_out"))
    p.add_argument("--max-papers", type=int, default=25)
    p.add_argument("--max-tables", type=int, default=80)
    p.add_argument("--timeout", type=int, default=45)
    p.add_argument("--script-timeout", type=int, default=900, help="Per-test subprocess wall-clock timeout in seconds")
    p.add_argument("--mode", choices=["scientific", "discovery"], default="scientific")
    p.add_argument("--manifest-only", action="store_true", default=True)
    p.add_argument("--allow-broad-discovery", action="store_true")
    p.add_argument("--max-bytes", type=int, default=50_000_000)
    p.add_argument("--header-rows", type=int, default=50)
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", nargs="*", help="Optional list, e.g. T31 T32 T46")
    p.add_argument("--confirm-candidates", action="store_true", help="Deprecated in v33: accepted for compatibility, but no longer splits the run; all tests run unless --only is supplied")
    p.add_argument("--primary-table-hunt", action="store_true", help="Deprecated in v33: accepted for compatibility, but no longer splits the run; all tests run unless --only is supplied")
    p.add_argument("--continue-on-error", action="store_true", default=True)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    selected = [t.upper() for t in (args.only or TESTS.keys())]
    if (args.confirm_candidates or args.primary_table_hunt) and not args.only:
        print("v33: --confirm-candidates/--primary-table-hunt are deprecated no-op selectors; running ALL tests T26-T60. Use --only only for debugging.", flush=True)
        selected = [t.upper() for t in TESTS.keys()]
    summary = []
    for tid in selected:
        td = TESTS[tid]
        script_matches = sorted((ROOT / "tests").glob(f"test{tid[1:]}_*.py"))
        if not script_matches:
            summary.append({"test_id": tid, "status": "missing_script"})
            continue
        script = script_matches[0]
        cmd = [sys.executable, str(script), "--cache", str(args.cache), "--outdir", str(args.outdir), "--max-papers", str(args.max_papers), "--max-tables", str(args.max_tables), "--timeout", str(args.timeout), "--mode", args.mode, "--max-bytes", str(args.max_bytes), "--header-rows", str(args.header_rows)]
        if args.manifest_only:
            cmd.append("--manifest-only")
        if args.allow_broad_discovery:
            cmd.append("--allow-broad-discovery")
        if args.force:
            cmd.append("--force")
        print(f"=== Running {tid}: {td['name']} ===", flush=True)
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.script_timeout)
            status = "ok" if proc.returncode == 0 else "process_error"
        except subprocess.TimeoutExpired as e:
            class _P:
                returncode = 124
                stdout = (e.stdout or "") if isinstance(e.stdout, str) else ((e.stdout or b"").decode("utf-8", errors="replace"))
                stderr = (e.stderr or "") if isinstance(e.stderr, str) else ((e.stderr or b"").decode("utf-8", errors="replace"))
            proc = _P()
            status = "process_timeout"
        result_file = args.outdir / f"{tid.lower()}_result.json"
        parsed_status = None
        if result_file.exists():
            try:
                parsed = json.loads(result_file.read_text(encoding="utf-8"))
                parsed_status = parsed.get("status")
            except Exception:
                pass
        else:
            # v35 safety: every selected test must emit a JSON block so dashboards do not silently omit T34 or timed-out tests.
            fallback = {
                "schema": "ccdr-tierb-result-v38-fallback",
                "test_id": tid,
                "test_name": td.get("name"),
                "prediction_ids": td.get("predictions", []),
                "prediction_names": td.get("prediction_names", []),
                "status": "data_limited_missing_output_fallback",
                "programmatic_verdict": "data_limited_missing_output_fallback",
                "confirm_allowed_now_v35": False,
                "confirm_allowed_now_v36": False,
                "confirm_allowed_now_v37": False,
                "confirm_allowed_now_v38": False,
                "confirmation_blocker_v35": {
                    "strict_confirm_allowed_now": False,
                    "why_not_confirmed": "test did not produce a result JSON in this all-tests run",
                    "single_next_blocker": "debug script runtime/parser and ensure emit_result() is called on all paths",
                    "best_auto_data_source_next": "test-specific structured public table source"
                },
                "near_confirm_score_v35": {
                    "score_0_10": 0,
                    "primary_table_available": False,
                    "model_rows_available": False,
                    "model_gate_attempted": False,
                    "strict_gate_remaining": ["missing_result_json"]
                },
                "confirmation_blocker_v36": {
                    "strict_confirm_allowed_now": False,
                    "why_not_confirmed": "test did not produce a result JSON in this all-tests run",
                    "single_next_blocker": "debug script runtime/parser and ensure emit_result() is called on all paths",
                    "best_auto_data_source_next": "test-specific structured public table source"
                },
                "near_confirm_score_v36": {
                    "score_0_10": 0,
                    "primary_table_available": False,
                    "model_rows_available": False,
                    "model_gate_attempted": False,
                    "strict_gate_remaining": ["missing_result_json"]
                },
                "confirmation_blocker_v37": {
                    "strict_confirm_allowed_now": False,
                    "why_not_confirmed": "test did not produce a result JSON in this all-tests run",
                    "single_next_blocker": "debug script runtime/parser and ensure emit_result() is called on all paths",
                    "best_auto_data_source_next": "test-specific structured public table source"
                },
                "near_confirm_score_v37": {
                    "score_0_10": 0,
                    "primary_table_available": False,
                    "model_rows_available": False,
                    "model_gate_attempted": False,
                    "strict_gate_remaining": ["missing_result_json"]
                },
                "confirmation_blocker_v38": {
                    "strict_confirm_allowed_now": False,
                    "why_not_confirmed": "test did not produce a result JSON in this all-tests run",
                    "single_next_blocker": "debug script runtime/parser and ensure emit_result() is called on all paths",
                    "best_auto_data_source_next": "test-specific structured public table source"
                },
                "near_confirm_score_v38": {
                    "score_0_10": 0,
                    "primary_table_available": False,
                    "model_rows_available": False,
                    "model_gate_attempted": False,
                    "strict_gate_remaining": ["missing_result_json"]
                },
                "positive_dashboard_fragment_v38": {
                    "test_id": tid,
                    "verdict": "data_limited_missing_output_fallback",
                    "confirm_allowed_now": False,
                    "strict_confirm_allowed_now": False,
                    "near_confirm_score": {
                        "score_0_10": 0,
                        "primary_table_available": False,
                        "model_rows_available": False,
                        "model_gate_attempted": False,
                        "strict_gate_remaining": ["missing_result_json"]
                    },
                    "why_not_confirmed": "test did not produce a result JSON in this all-tests run",
                    "single_next_blocker": "debug script runtime/parser and ensure emit_result() is called on all paths",
                    "best_auto_data_source_next": "test-specific structured public table source",
                    "v38": {"fallback_restored_output": True}
                },
                "positive_dashboard_fragment_v35": {
                    "test_id": tid,
                    "verdict": "data_limited_missing_output_fallback",
                    "confirm_allowed_now": False,
                    "strict_confirm_allowed_now": False,
                    "near_confirm_score": {
                        "score_0_10": 0,
                        "primary_table_available": False,
                        "model_rows_available": False,
                        "model_gate_attempted": False,
                        "strict_gate_remaining": ["missing_result_json"]
                    },
                    "why_not_confirmed": "test did not produce a result JSON in this all-tests run",
                    "single_next_blocker": "debug script runtime/parser and ensure emit_result() is called on all paths",
                    "best_auto_data_source_next": "test-specific structured public table source",
                    "v35": {"fallback_restored_output": True}
                },
                "positive_dashboard_fragment_v36": {
                    "test_id": tid,
                    "verdict": "data_limited_missing_output_fallback",
                    "confirm_allowed_now": False,
                    "strict_confirm_allowed_now": False,
                    "near_confirm_score": {
                        "score_0_10": 0,
                        "primary_table_available": False,
                        "model_rows_available": False,
                        "model_gate_attempted": False,
                        "strict_gate_remaining": ["missing_result_json"]
                    },
                    "why_not_confirmed": "test did not produce a result JSON in this all-tests run",
                    "single_next_blocker": "debug script runtime/parser and ensure emit_result() is called on all paths",
                    "best_auto_data_source_next": "test-specific structured public table source",
                    "v36": {"fallback_restored_output": True}
                },
                "positive_dashboard_fragment_v37": {
                    "test_id": tid,
                    "verdict": "data_limited_missing_output_fallback",
                    "confirm_allowed_now": False,
                    "strict_confirm_allowed_now": False,
                    "near_confirm_score": {
                        "score_0_10": 0,
                        "primary_table_available": False,
                        "model_rows_available": False,
                        "model_gate_attempted": False,
                        "strict_gate_remaining": ["missing_result_json"]
                    },
                    "why_not_confirmed": "test did not produce a result JSON in this all-tests run",
                    "single_next_blocker": "debug script runtime/parser and ensure emit_result() is called on all paths",
                    "best_auto_data_source_next": "test-specific structured public table source",
                    "v37": {"fallback_restored_output": True}
                },
                "process_status": status,
                "stdout_tail": proc.stdout[-1200:],
                "stderr_tail": proc.stderr[-1200:],
            }
            try:
                from tierb.v56_missing_output_repair import enrich_fallback_v56
                fallback = enrich_fallback_v56(fallback, tid, td, status, proc.stdout, proc.stderr)
            except Exception as _v56_e:
                fallback.setdefault("v56_missing_output_repair_error", f"{type(_v56_e).__name__}: {_v56_e}")
            try:
                from tierb.v57_confirm_repairs import enrich_fallback_v57
                fallback = enrich_fallback_v57(fallback, tid, td, status, proc.stdout, proc.stderr)
            except Exception as _v57_e:
                fallback.setdefault("v57_missing_output_repair_error", f"{type(_v57_e).__name__}: {_v57_e}")
            try:
                from tierb.v58_confirm_focus import enrich_fallback_v58
                fallback = enrich_fallback_v58(fallback, tid, td, status, proc.stdout, proc.stderr)
            except Exception as _v58_e:
                fallback.setdefault("v58_missing_output_repair_error", f"{type(_v58_e).__name__}: {_v58_e}")
            try:
                from tierb.v59_confirm_extractors import enrich_fallback_v59
                fallback = enrich_fallback_v59(fallback, tid, td, status, proc.stdout, proc.stderr)
            except Exception as _v59_e:
                fallback.setdefault("v59_missing_output_repair_error", f"{type(_v59_e).__name__}: {_v59_e}")
            try:
                result_file.write_text(json.dumps(fallback, indent=2, sort_keys=True), encoding="utf-8")
                parsed_status = fallback["status"]
            except Exception:
                pass
        summary.append({"test_id": tid, "name": td["name"], "process_status": status, "result_status": parsed_status, "script": str(script), "stdout_tail": proc.stdout[-1200:], "stderr_tail": proc.stderr[-1200:]})
        if proc.returncode != 0 and not args.continue_on_error:
            break
    out = {"schema": "ccdr-tierb-batch-summary-v1", "n_tests": len(summary), "summary": summary}
    (args.outdir / "tier_b_batch_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    # v20 positive dashboard: aggregate per-result programmatic verdicts and positive-path fragments.
    dashboard = {
        "schema": "ccdr-tierb-positive-dashboard-v22",
        "anchors": [],
        "physical_positive_leads": [],
        "readiness_positive": [],
        "descriptor_paths_ready": [],
        "engineering_paths_ready": [],
        "broad_nulls_with_open_narrow_path": [],
        "data_limited_positive_paths": [],
        "bound_only": [],
        "tests": [],
    }
    for item in summary:
        tid = item.get("test_id")
        result_file = args.outdir / f"{str(tid).lower()}_result.json"
        if not result_file.exists():
            continue
        try:
            r = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        verdict = r.get("programmatic_verdict") or (r.get("programmatic_verdict_v19") or {}).get("verdict") or r.get("status")
        frag = (r.get("positive_dashboard_fragment_v51") or r.get("positive_dashboard_fragment_v50") or r.get("positive_dashboard_fragment_v49") or r.get("positive_dashboard_fragment_v48") or r.get("positive_dashboard_fragment_v47") or r.get("positive_dashboard_fragment_v46") or r.get("positive_dashboard_fragment_v45") or r.get("positive_dashboard_fragment_v44") or r.get("positive_dashboard_fragment_v43") or r.get("positive_dashboard_fragment_v42") or r.get("positive_dashboard_fragment_v41") or r.get("positive_dashboard_fragment_v40") or r.get("positive_dashboard_fragment_v39") or r.get("positive_dashboard_fragment_v38") or r.get("positive_dashboard_fragment_v37") or r.get("positive_dashboard_fragment_v36") or r.get("positive_dashboard_fragment_v35") or r.get("positive_dashboard_fragment_v34") or r.get("positive_dashboard_fragment_v33") or r.get("positive_dashboard_fragment_v32") or r.get("positive_dashboard_fragment_v30") or r.get("positive_dashboard_fragment_v29") or r.get("positive_dashboard_fragment_v28") or r.get("positive_dashboard_fragment_v27") or r.get("positive_dashboard_fragment_v26") or r.get("positive_dashboard_fragment_v25") or r.get("positive_dashboard_fragment_v24") or r.get("positive_dashboard_fragment_v23") or r.get("positive_dashboard_fragment_v22") or r.get("positive_dashboard_fragment_v21") or r.get("positive_dashboard_fragment_v20") or {"test_id": tid, "verdict": verdict})
        dashboard["tests"].append(frag)
        if verdict == "positive_consistency_anchor":
            dashboard["anchors"].append(tid)
        elif verdict == "positive_physical_lead":
            dashboard["physical_positive_leads"].append(tid)
        elif verdict == "readiness_positive":
            dashboard["readiness_positive"].append(tid)
        elif verdict == "null_coarse_proxy_descriptor_model_ready":
            dashboard["descriptor_paths_ready"].append(tid)
        elif verdict == "null_current_benchmark_design_search_ready":
            dashboard["engineering_paths_ready"].append(tid)
        elif verdict == "null_broad_claim_open_narrow_claim":
            dashboard["broad_nulls_with_open_narrow_path"].append(tid)
        elif verdict == "data_limited_positive_path_ready":
            dashboard["data_limited_positive_paths"].append(tid)
        elif verdict == "bound_only":
            dashboard["bound_only"].append(tid)
    # v21 positive dashboard enrichment, especially EL branch tracking.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v21"
    dashboard.setdefault("EL_branch", {
        "synthetic_positive_or_open": [],
        "best_data_positive_path": [],
        "second_best_data_path": [],
        "early_path": [],
        "current_nulls": [],
        "recommended_next": "T44 WikiChip/TechInsights NAND exact parser",
    })
    dashboard.setdefault("materials_flagship", [])
    dashboard.setdefault("biology_model_ready", [])
    dashboard.setdefault("fusion_secondary_diagnostics", [])
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        el = frag.get("el_branch") or {}
        if tid == "T44" or el.get("label") == "EL1_EL3_3D_NAND_flagship":
            if tid not in dashboard["EL_branch"]["best_data_positive_path"]:
                dashboard["EL_branch"]["best_data_positive_path"].append(tid)
        if tid == "T45" or el.get("label") == "EL8_optical_interconnect":
            if tid not in dashboard["EL_branch"]["second_best_data_path"]:
                dashboard["EL_branch"]["second_best_data_path"].append(tid)
        if tid == "T47" or el.get("label") == "EL_neuromorphic_energy_topology":
            if tid not in dashboard["EL_branch"]["early_path"]:
                dashboard["EL_branch"]["early_path"].append(tid)
        if tid == "T46":
            if tid not in dashboard["EL_branch"]["synthetic_positive_or_open"]:
                dashboard["EL_branch"]["synthetic_positive_or_open"].append(tid)
            if tid not in dashboard["EL_branch"]["current_nulls"]:
                dashboard["EL_branch"]["current_nulls"].append("T46a")
        if tid in {"T31", "T32"}:
            dashboard["materials_flagship"].append(tid)
        if tid == "T53":
            dashboard["biology_model_ready"].append(tid)
        if tid in {"T26", "T27", "T28", "T29", "T30"}:
            dashboard["fusion_secondary_diagnostics"].append(tid)

    # v22 positive dashboard enrichment: EL branch, materials, T53, fusion, PV, optimization.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v22"
    dashboard.setdefault("recommended_next", [])
    dashboard.setdefault("EL_branch", {
        "synthetic_positive_or_open": [],
        "best_data_positive_path": [],
        "second_best_data_path": [],
        "early_path": [],
        "current_nulls": [],
        "recommended_next": "T44 WikiChip/TechInsights NAND exact parser",
    })
    dashboard.setdefault("positive_path_priorities_v22", [])
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        v22 = frag.get("v22") or {}
        if tid == "T44":
            dashboard["positive_path_priorities_v22"].append({"rank": 1, "test_id": tid, "path": "EL1/EL3 3D NAND exact parser", "status": (v22.get("t44_nand") or {}).get("status")})
        if tid == "T53":
            dashboard["positive_path_priorities_v22"].append({"rank": 2, "test_id": tid, "path": "biology UniProt/PDB enrichment residual model", "status": (v22.get("t53_enrichment") or {}).get("status")})
        if tid == "T31":
            dashboard["positive_path_priorities_v22"].append({"rank": 3, "test_id": tid, "path": "grain/nano materials flagship", "status": (v22.get("materials_score") or {}).get("status")})
        if tid == "T45":
            dashboard["positive_path_priorities_v22"].append({"rank": 4, "test_id": tid, "path": "EL8 optical interconnect unit parser", "status": (v22.get("t45_optical") or {}).get("status")})
        if tid == "T46":
            dashboard["positive_path_priorities_v22"].append({"rank": 5, "test_id": tid, "path": "T46b optimized CDT-hybrid code search", "status": (v22.get("t46b_search") or {}).get("status")})
        if tid in {"T26", "T27", "T28", "T29", "T30"}:
            dashboard["positive_path_priorities_v22"].append({"rank": 6, "test_id": tid, "path": "fusion secondary unit-line / figure diagnostics", "status": (v22.get("fusion_diag") or {}).get("status")})
    dashboard["recommended_next"] = [
        "Implement/inspect T44 NAND exact rows first; this is the strongest EL data-positive route.",
        "Run T53 enrichment join to turn readiness-positive into model-positive.",
        "Grow T31 grain-size-known rows to >=10 decisive rows.",
        "Use T45 PDF/unit extraction for pJ/bit and bandwidth/reach rows.",
        "Run T46b 50-seed optimization with real decoders after proxy search.",
        "Keep fusion secondary diagnostics active but non-decisive until primary event/profile tables exist.",
    ]

    # v23 positive dashboard enrichment: fixed materials scorer, EL squeeze, biology model attempt, fusion secondary diagnostics.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v23"
    dashboard.setdefault("v23_priority_paths", [])
    dashboard.setdefault("EL_branch", {
        "synthetic_positive_or_open": [],
        "best_data_positive_path": [],
        "second_best_data_path": [],
        "early_path": [],
        "current_nulls": [],
        "recommended_next": "T44 WikiChip/TechInsights NAND exact parser",
    })
    dashboard.setdefault("positive_warnings", [])
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        v23 = frag.get("v23") or {}
        verdict = frag.get("verdict")
        if tid == "T60":
            dashboard["v23_priority_paths"].append({"rank": 1, "test_id": tid, "path": "Koide charged-lepton anchor + random/sector nulls", "status": (v23.get("t60_nulls") or {}).get("T60a")})
        if tid == "T31":
            ms = v23.get("materials_score") or {}
            dashboard["v23_priority_paths"].append({"rank": 2, "test_id": tid, "path": "grain/nano materials flagship", "status": ms.get("status"), "score": ms.get("score")})
            if (ms.get("status") or "").startswith("strong"):
                dashboard.setdefault("physical_positive_leads", []).append(tid) if tid not in dashboard.get("physical_positive_leads", []) else None
        if tid == "T53":
            dashboard["v23_priority_paths"].append({"rank": 3, "test_id": tid, "path": "ProteinGym first residual model + UniProt/PDB enrichment", "status": (v23.get("t53_model") or {}).get("status")})
        if tid == "T44":
            dashboard["v23_priority_paths"].append({"rank": 4, "test_id": tid, "path": "EL1/EL3 3D NAND exact parser", "status": (v23.get("t44_nand") or {}).get("status")})
            if tid not in dashboard["EL_branch"].setdefault("best_data_positive_path", []): dashboard["EL_branch"]["best_data_positive_path"].append(tid)
        if tid == "T45":
            dashboard["v23_priority_paths"].append({"rank": 5, "test_id": tid, "path": "EL8 optical interconnect unit extractor", "status": (v23.get("t45_optical") or {}).get("status")})
            if tid not in dashboard["EL_branch"].setdefault("second_best_data_path", []): dashboard["EL_branch"]["second_best_data_path"].append(tid)
        if tid == "T47":
            dashboard["v23_priority_paths"].append({"rank": 6, "test_id": tid, "path": "EL neuromorphic exact benchmark targeting", "status": (v23.get("t47_neuro") or {}).get("status")})
            if tid not in dashboard["EL_branch"].setdefault("early_path", []): dashboard["EL_branch"]["early_path"].append(tid)
        if tid == "T46":
            dashboard["v23_priority_paths"].append({"rank": 7, "test_id": tid, "path": "EL6 T46b 100-seed CDT-hybrid design search", "status": (v23.get("t46b_search") or {}).get("status")})
            if tid not in dashboard["EL_branch"].setdefault("synthetic_positive_or_open", []): dashboard["EL_branch"]["synthetic_positive_or_open"].append(tid)
            if "T46a" not in dashboard["EL_branch"].setdefault("current_nulls", []): dashboard["EL_branch"]["current_nulls"].append("T46a")
        if tid == "T48":
            dashboard["v23_priority_paths"].append({"rank": 8, "test_id": tid, "path": "T48b descriptor PV primary path", "status": "descriptor_model_ready"})
        if tid in {"T26","T27","T28","T29","T30"}:
            dashboard["v23_priority_paths"].append({"rank": 9, "test_id": tid, "path": "fusion secondary unit-line/figure diagnostics", "status": (v23.get("fusion_diag") or {}).get("status")})
    dashboard["recommended_next"] = [
        "Fix/monitor T31/T32 materials through materials_positive_score_v23; keep v20 fallback keys active.",
        "Prioritize T44 NAND exact parser rows and model density_Gb_per_mm2 ~ layers + year + bits/cell + manufacturer effects.",
        "Run T53 first residual model with ProteinGym outcome/join mapping before full PDB symmetry enrichment.",
        "Use T45 unit-line extractor on exact optical-interconnect PDFs and ignore landing/rate-limit pages as evidence.",
        "Target T47 exact benchmark supplements for Loihi/TrueNorth/SpiNNaker; keep hardware-guide HTML discovery-only.",
        "Keep fusion alive as plausible_secondary_diagnostic only until a primary event/profile table is found.",
        "Add T60 random-triplet and sector-reshuffling nulls around the T60a consistency anchor.",
    ]


    # v24 positive dashboard enrichment: result-oriented positive paths and EL/fusion squeeze.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v25"
    dashboard.setdefault("v24_priority_paths", [])
    dashboard.setdefault("EL_branch", {
        "synthetic_positive_or_open": [],
        "best_data_positive_path": [],
        "second_best_data_path": [],
        "early_path": [],
        "current_nulls": [],
        "recommended_next": "T44 WikiChip/TechInsights NAND exact parser and model fit",
    })
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        v24 = frag.get("v24") or {}
        if tid == "T60":
            dashboard["v24_priority_paths"].append({"rank": 1, "test_id": tid, "path": "Koide charged-lepton anchor + computed random triplet null", "status": (v24.get("t60_null") or {}).get("status")})
        if tid == "T31":
            ms = v24.get("materials_score") or {}
            dashboard["v24_priority_paths"].append({"rank": 2, "test_id": tid, "path": "grain/nano materials flagship", "status": ms.get("status"), "score": ms.get("score")})
        if tid == "T53":
            dashboard["v24_priority_paths"].append({"rank": 3, "test_id": tid, "path": "ProteinGym first residual model replay + PDB/UniProt enrichment", "status": (v24.get("t53_model") or {}).get("status")})
        if tid == "T44":
            dashboard["v24_priority_paths"].append({"rank": 4, "test_id": tid, "path": "EL1/EL3 3D NAND exact parser and layer model", "status": (v24.get("t44_nand") or {}).get("status")})
            if tid not in dashboard["EL_branch"].setdefault("best_data_positive_path", []): dashboard["EL_branch"]["best_data_positive_path"].append(tid)
        if tid == "T45":
            dashboard["v24_priority_paths"].append({"rank": 5, "test_id": tid, "path": "EL8 optical-interconnect unit extraction and model", "status": (v24.get("t45_optical") or {}).get("status")})
            if tid not in dashboard["EL_branch"].setdefault("second_best_data_path", []): dashboard["EL_branch"]["second_best_data_path"].append(tid)
        if tid == "T47":
            dashboard["v24_priority_paths"].append({"rank": 6, "test_id": tid, "path": "neuromorphic exact benchmark targeting", "status": (v24.get("t47_neuro") or {}).get("status")})
            if tid not in dashboard["EL_branch"].setdefault("early_path", []): dashboard["EL_branch"]["early_path"].append(tid)
        if tid == "T46":
            dashboard["v24_priority_paths"].append({"rank": 7, "test_id": tid, "path": "EL6 T46b real optimizer/decoder next", "status": "design_search_open"})
            if tid not in dashboard["EL_branch"].setdefault("synthetic_positive_or_open", []): dashboard["EL_branch"]["synthetic_positive_or_open"].append(tid)
            if "T46a" not in dashboard["EL_branch"].setdefault("current_nulls", []): dashboard["EL_branch"]["current_nulls"].append("T46a")
        if tid == "T48":
            dashboard["v24_priority_paths"].append({"rank": 8, "test_id": tid, "path": "T48b descriptor PV model", "status": "descriptor_model_ready"})
        if tid in {"T26", "T27", "T28", "T29", "T30"}:
            dashboard["v24_priority_paths"].append({"rank": 9, "test_id": tid, "path": "fusion secondary numeric-line/figure diagnostics", "status": (v24.get("fusion_diag") or {}).get("status")})
    dashboard["recommended_next"] = [
        "Run/inspect T44 exact NAND normalized rows first; fit density_Gb_per_mm2 ~ layers + year + bits/cell + manufacturer effects.",
        "Run T53 ProteinGym first residual model and then add PDB/UniProt symmetry/contact enrichment.",
        "Expand T31 grain/nano decisive rows to >=10 and rerun source/material-family jackknife.",
        "Run T48b descriptor model on parsed NREL/PVDPC rows and report FDR family q-values.",
        "Use T45 unit-line extraction on exact optical-interconnect PDFs; normalize pJ/bit, bandwidth, reach.",
        "Keep fusion secondary diagnostics active; success is plausible_secondary_diagnostic, not confirmation, until primary event/profile tables appear.",
        "Compute T60 random-triplet and sector-reshuffling nulls around the T60a consistency anchor.",
    ]

    # v25 positive dashboard enrichment.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v25"
    dashboard["v25_priority_order"] = [
        "T60a Koide anchor with random-triplet null",
        "T31 grain/nano materials flagship",
        "T44 EL1/EL3 NAND exact parser/model",
        "T53 OrganismalFitness residual model + PDB/UniProt enrichment",
        "T48b PV descriptor model",
        "T45 EL8 optical interconnect unit extractor",
        "T46b optimized coding search",
        "T47 neuromorphic exact benchmark rows",
        "Fusion secondary numeric-line diagnostics"
    ]
    dashboard["v25_recommended_next"] = [
        "Run T44 parser first and fit layer-vs-year baseline when N>=20.",
        "Use OrganismalFitness outcome in T53 and add PDB/UniProt symmetry/contact proxy.",
        "Grow T31 grain/nano rows to >=10 decisive rows and >=10-20 fits.",
        "Run T48b descriptor model with BH-FDR family q-values.",
        "Extract T45 pJ/bit + bandwidth + reach rows from exact PDFs.",
        "Export fusion_secondary_rows.csv from exact PDF numeric context rows; keep non-decisive."
    ]
    # v26 confirm-focused dashboard: separates actual confirms from near-confirm paths.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v26"
    dashboard["v26_confirm_status"] = {
        "confirmed_consistency_anchors": [],
        "near_confirm_physical_leads": [],
        "EL_confirm_candidates": [],
        "descriptor_model_confirm_candidates": [],
        "readiness_to_confirm_candidates": [],
        "bound_only": [],
        "fusion_secondary_only": [],
        "data_limited_exact_table_needed": []
    }
    dashboard["v26_priority_order"] = [
        "T44 EL1/EL3 NAND exact rows/model",
        "T31 grain/nano materials expansion",
        "T48b PV descriptor confirm model",
        "T53 OrganismalFitness + PDB/UniProt model",
        "T45 EL8 optical unit rows",
        "T60a Koide consistency anchor nulls",
        "T46b optimizer engineering search",
        "Fusion secondary rows only"
    ]
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        v26 = frag.get("v26") or {}
        verdict = frag.get("verdict")
        if tid == "T60": dashboard["v26_confirm_status"]["confirmed_consistency_anchors"].append(tid)
        if tid == "T31": dashboard["v26_confirm_status"]["near_confirm_physical_leads"].append(tid)
        if tid in {"T44","T45","T46","T47"}: dashboard["v26_confirm_status"]["EL_confirm_candidates"].append({"test_id": tid, "verdict": verdict})
        if tid == "T48": dashboard["v26_confirm_status"]["descriptor_model_confirm_candidates"].append(tid)
        if tid == "T53": dashboard["v26_confirm_status"]["readiness_to_confirm_candidates"].append(tid)
        if tid in {"T50","T51","T52"}: dashboard["v26_confirm_status"]["bound_only"].append(tid)
        if tid in {"T26","T27","T28","T29","T30"}: dashboard["v26_confirm_status"]["fusion_secondary_only"].append(tid)
        if tid in {"T54","T57","T59"}: dashboard["v26_confirm_status"]["data_limited_exact_table_needed"].append(tid)
    dashboard["v26_recommended_next"] = [
        "Run T44 exact NAND normalized-row parser first; confirm target is N>=20 and layer model beats year-only/manufacturer baseline.",
        "Grow T31 grain/nano decisive rows to >=10 and rerun material-family/source jackknives.",
        "Run T48b descriptor PV model with BH-FDR family q-values and tandem/concentrator exclusions.",
        "Use OrganismalFitness in T53 and add PDB/UniProt symmetry/contact proxy before confirm language.",
        "Extract T45 pJ/bit + bandwidth + reach rows from exact optical-interconnect PDFs.",
        "Keep fusion as secondary diagnostic; export fusion_secondary_rows.csv and do not mark confirm without primary event/profile tables.",
        "Surface T60 random-triplet + sector/look-elsewhere nulls in every T60 run."
    ]
    # v27 confirm-execution dashboard: prioritizes executable confirm gates.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v27"
    dashboard["v27_confirm_execution_status"] = {
        "confirmed_consistency_anchors": [],
        "near_confirm_physical_leads": [],
        "EL_confirm_execution_candidates": [],
        "descriptor_model_confirm_candidates": [],
        "readiness_to_confirm_candidates": [],
        "bound_only": [],
        "fusion_secondary_only": [],
        "exact_table_needed": []
    }
    dashboard["v27_priority_order"] = [
        "T44 EL1/EL3: generate normalized NAND rows and run layer-vs-year model",
        "T31/T32 materials: expand grain/nano rows and run jackknives",
        "T48b PV: run descriptor model with family BH-FDR",
        "T60: keep Koide anchor and run sector/look-elsewhere gates",
        "T53: OrganismalFitness + PDB/UniProt symmetry/contact model",
        "T45 EL8: extract optical pJ/bit + bandwidth + reach rows",
        "T46b: real optimizer with BP/min-sum and matched baselines",
        "Fusion: export secondary numeric context rows; primary confirm only from event/profile tables"
    ]
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        verdict = frag.get("verdict")
        if tid == "T60": dashboard["v27_confirm_execution_status"]["confirmed_consistency_anchors"].append(tid)
        elif tid in {"T31","T32"}: dashboard["v27_confirm_execution_status"]["near_confirm_physical_leads"].append({"test_id": tid, "verdict": verdict})
        elif tid in {"T44","T45","T46","T47"}: dashboard["v27_confirm_execution_status"]["EL_confirm_execution_candidates"].append({"test_id": tid, "verdict": verdict})
        elif tid == "T48": dashboard["v27_confirm_execution_status"]["descriptor_model_confirm_candidates"].append(tid)
        elif tid == "T53": dashboard["v27_confirm_execution_status"]["readiness_to_confirm_candidates"].append(tid)
        elif tid in {"T50","T51","T52"}: dashboard["v27_confirm_execution_status"]["bound_only"].append(tid)
        elif tid in {"T26","T27","T28","T29","T30"}: dashboard["v27_confirm_execution_status"]["fusion_secondary_only"].append(tid)
        elif tid in {"T54","T57","T59"}: dashboard["v27_confirm_execution_status"]["exact_table_needed"].append(tid)
    dashboard["v27_recommended_next"] = [
        "Implement/inspect data/generated/t44_nand_exact_rows_v27.csv first; confirm target N>=20 and layer model beats year-only/manufacturer baseline.",
        "Grow grain_size_known_manifest_v27.csv and rerun T31/T32 with source/material-family and temperature-window jackknives.",
        "Run T48b descriptor PV model now; emit family coefficients, q-values, and tandem/concentrator exclusion.",
        "Run T60 sector-reshuffle/look-elsewhere gates plus quark/lattice parser.",
        "Switch T53 to OrganismalFitness and join PDB/UniProt symmetry/contact proxy.",
        "Extract T45 exact optical-interconnect pJ/bit + bandwidth + reach rows.",
        "Keep fusion secondary-only until primary event/profile tables pass all contract groups."
    ]

    # v28 confirm-execution dashboard: tracks actual generated-row/model gates.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v28"
    dashboard["v28_confirm_status"] = {
        "confirmed_consistency_anchors": [],
        "near_confirm_physical_leads": [],
        "EL_confirm_candidates": [],
        "descriptor_model_confirm_candidates": [],
        "readiness_to_confirm_candidates": [],
        "bound_only": [],
        "fusion_secondary_only": [],
        "exact_table_needed": []
    }
    dashboard["v28_priority_order"] = [
        "T44 EL1/EL3: data/generated/t44_nand_exact_rows_v28.csv and layer-vs-year model",
        "T48b PV: run descriptor model with family BH-FDR q-values",
        "T31/T32: expand grain_size_known_manifest_v28.csv and run jackknives",
        "T60: keep Koide anchor, random-triplet null, sector/look-elsewhere gates",
        "T53: OrganismalFitness + PDB/UniProt symmetry/contact model",
        "T45 EL8: data/generated/t45_optical_interconnect_rows_v28.csv",
        "Fusion: data/generated/fusion_secondary_rows_v28.csv; primary confirm only from event/profile tables"
    ]
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        verdict = frag.get("verdict")
        if tid == "T60": dashboard["v28_confirm_status"]["confirmed_consistency_anchors"].append(tid)
        elif tid in {"T31","T32"}: dashboard["v28_confirm_status"]["near_confirm_physical_leads"].append({"test_id": tid, "verdict": verdict})
        elif tid in {"T44","T45","T46","T47"}: dashboard["v28_confirm_status"]["EL_confirm_candidates"].append({"test_id": tid, "verdict": verdict})
        elif tid == "T48": dashboard["v28_confirm_status"]["descriptor_model_confirm_candidates"].append(tid)
        elif tid == "T53": dashboard["v28_confirm_status"]["readiness_to_confirm_candidates"].append(tid)
        elif tid in {"T50","T51","T52"}: dashboard["v28_confirm_status"]["bound_only"].append(tid)
        elif tid in {"T26","T27","T28","T29","T30"}: dashboard["v28_confirm_status"]["fusion_secondary_only"].append(tid)
        elif tid in {"T54","T57","T59"}: dashboard["v28_confirm_status"]["exact_table_needed"].append(tid)
    dashboard["v28_recommended_next"] = [
        "Implement/inspect T44 NAND exact rows first; confirmation requires N>=20 and layer model beating year-only/manufacturer baseline.",
        "Run T48b descriptor model now; row base is the largest potential confirm path.",
        "Expand T31 grain/nano manifest to >=10 decisive rows and rerun T32b fixed-exponent comparison.",
        "Run T60 quark/lattice parser plus sector reshuffling and look-elsewhere gates.",
        "Switch T53 to OrganismalFitness/DMS_score and join PDB/UniProt symmetry/contact proxy.",
        "Extract T45 optical pJ/bit + bandwidth + reach rows from exact PDFs.",
        "Keep fusion secondary-only until primary event/profile tables pass all FR contract groups."
    ]
    # v29 confirm-execution dashboard: prioritizes rows/models that can actually
    # move tests from positive-path to confirmation candidate.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v29"
    dashboard["v29_confirm_status"] = {
        "confirmed_consistency_anchors": [],
        "near_confirm_physical_leads": [],
        "EL_confirm_candidates": [],
        "descriptor_model_confirm_candidates": [],
        "readiness_to_confirm_candidates": [],
        "bound_only": [],
        "fusion_secondary_only": [],
        "exact_table_needed": []
    }
    dashboard["v29_priority_order"] = [
        "T44 EL1/EL3: extract normalized NAND rows and run layer-vs-year/manufacturer model",
        "T48b PV: execute descriptor model with family BH-FDR q-values",
        "T31/T32: expand grain-size manifest and run nanostructure-only exponent/model gates",
        "T60: charged-lepton anchor plus random-triplet, sector-reshuffle, look-elsewhere, and T60b quark/lattice gates",
        "T53: OrganismalFitness/DMS outcome with PDB/UniProt symmetry-contact proxy",
        "T45 EL8: extract optical pJ/bit + bandwidth + reach rows",
        "Fusion: export secondary numeric context rows only; primary confirmation requires event/profile tables"
    ]
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        verdict = frag.get("verdict")
        if tid == "T60": dashboard["v29_confirm_status"]["confirmed_consistency_anchors"].append(tid)
        elif tid in {"T31","T32"}: dashboard["v29_confirm_status"]["near_confirm_physical_leads"].append({"test_id": tid, "verdict": verdict})
        elif tid in {"T44","T45","T46","T47"}: dashboard["v29_confirm_status"]["EL_confirm_candidates"].append({"test_id": tid, "verdict": verdict})
        elif tid == "T48": dashboard["v29_confirm_status"]["descriptor_model_confirm_candidates"].append(tid)
        elif tid == "T53": dashboard["v29_confirm_status"]["readiness_to_confirm_candidates"].append(tid)
        elif tid in {"T50","T51","T52"}: dashboard["v29_confirm_status"]["bound_only"].append(tid)
        elif tid in {"T26","T27","T28","T29","T30"}: dashboard["v29_confirm_status"]["fusion_secondary_only"].append(tid)
        elif tid in {"T54","T57","T59"}: dashboard["v29_confirm_status"]["exact_table_needed"].append(tid)
    dashboard["v29_recommended_next"] = [
        "Inspect data/generated/t44_nand_exact_rows_v29.csv; confirmation target is N>=20 with positive layer coefficient and AIC/BIC improvement.",
        "Inspect data/generated/t48b_pv_descriptor_rows_v29.csv; run family coefficients and BH-FDR q-values.",
        "Grow data/generated/grain_size_known_manifest_v29.csv to >=10 decisive rows and rerun T31/T32.",
        "Add data/generated/t60_quark_lattice_masses_v29.csv to unlock T60b sector gate.",
        "Add data/generated/t53_proteingym_enriched_rows_v29.csv with OrganismalFitness and PDB/UniProt proxy columns.",
        "Inspect data/generated/t45_optical_interconnect_rows_v29.csv; target 8-15 exact rows.",
        "Inspect data/generated/fusion_secondary_rows_v29.csv; keep it non-decisive unless primary tables appear."
    ]


    # v30 confirm-priority dashboard: implements the 10 prioritized improvements
    # and separates strict confirms from confirm candidates, readiness routes,
    # bounds, demoted tests, and fusion secondary diagnostics.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v30"
    dashboard["v30_confirm_status"] = {
        "strict_confirm_allowed_now": [],
        "confirmed_consistency_anchors": [],
        "physical_or_engineering_confirm_candidates": [],
        "near_confirm_physical_leads": [],
        "readiness_positive": [],
        "descriptor_model_candidates": [],
        "bound_only": [],
        "fusion_priority_T28_T30": [],
        "fusion_secondary_only": [],
        "exact_table_needed_or_demoted": []
    }
    dashboard["v30_priority_order"] = [
        "1. T53 ProteinGym/DMS outcome + PDB/UniProt symmetry-contact residual gate",
        "2. T31/T32 measured grain/nanocrystalline-only material gates",
        "3. T44 exact 3D-NAND layer/capacity/die-area/bits-cell model",
        "4. T48b-only PV descriptor/FDR route; T48a remains null control",
        "5. T57/T59 exact HEPData CSV/YAML manifests only",
        "6. T50-T52 hard bound-only role",
        "7. Demote exact-table-missing tests until primary tables exist",
        "8. Fusion focus: T28/T30 H-mode DB first; T26/T27 secondary until ELM/RMP event rows exist",
        "9. T60a anchor with blocking T60b/T60c/T60d gates",
        "10. Global confirm_allowed_now dashboard gate"
    ]
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        verdict = frag.get("verdict")
        v30 = frag.get("v30") or {}
        if frag.get("confirm_allowed_now"):
            dashboard["v30_confirm_status"]["strict_confirm_allowed_now"].append({"test_id": tid, "verdict": verdict})
        if tid == "T60":
            dashboard["v30_confirm_status"]["confirmed_consistency_anchors"].append(tid)
        elif tid in {"T31", "T32"}:
            dashboard["v30_confirm_status"]["near_confirm_physical_leads"].append({"test_id": tid, "verdict": verdict, "gate": v30.get("materials_confirm_execution_v30")})
        elif tid in {"T44", "T45", "T46", "T47"}:
            dashboard["v30_confirm_status"]["physical_or_engineering_confirm_candidates"].append({"test_id": tid, "verdict": verdict, "confirm_allowed_now": frag.get("confirm_allowed_now")})
        elif tid == "T48":
            dashboard["v30_confirm_status"]["descriptor_model_candidates"].append({"test_id": tid, "verdict": verdict, "confirm_allowed_now": frag.get("confirm_allowed_now")})
        elif tid == "T53":
            dashboard["v30_confirm_status"]["readiness_positive"].append({"test_id": tid, "verdict": verdict, "confirm_allowed_now": frag.get("confirm_allowed_now")})
        elif tid in {"T50", "T51", "T52"}:
            dashboard["v30_confirm_status"]["bound_only"].append(tid)
        elif tid in {"T28", "T30"}:
            dashboard["v30_confirm_status"]["fusion_priority_T28_T30"].append({"test_id": tid, "verdict": verdict})
        elif tid in {"T26", "T27", "T29"}:
            dashboard["v30_confirm_status"]["fusion_secondary_only"].append({"test_id": tid, "verdict": verdict})
        elif tid in {"T33","T34","T35","T36","T37","T38","T39","T40","T41","T42","T43","T49","T54","T55","T56","T57","T58","T59"}:
            dashboard["v30_confirm_status"]["exact_table_needed_or_demoted"].append({"test_id": tid, "verdict": verdict})
    dashboard["v30_recommended_next"] = [
        "v33 supersedes manual-fill workflow: generated CSV artifacts are cache/audit outputs written by scripts, not required user input.",
        "Run all tests together; use --only only for debugging one failing test.",
        "Treat T50-T52 as bounds only in reports; never promote them to confirmations.",
        "For fusion, prioritize automatic T28/T30 OSF/ITPA/H-mode primary-data discovery before T26/T27 ELM/RMP PDF diagnostics.",
        "For T60, keep T60a positive consistency-anchor language only until quark/lattice and look-elsewhere gates pass.",
        "Use v33_confirm_status.strict_confirm_allowed_now as the only automated list allowed to use confirm language."
    ]



    # v32 confirm + primary-table-hunt dashboard. Implements requested items
    # 1-6 and 8-10, with item 7 demotion explicitly excluded/replaced by
    # active primary-table hunting.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v32"
    dashboard["v32_confirm_status"] = {
        "strict_confirm_allowed_now": [],
        "anchors_consistency_only": [],
        "active_confirmation_candidates": [],
        "primary_table_hunts": [],
        "bounds_only": [],
        "fusion_T28_T30_priority": [],
        "restored_tests": [],
        "source_hunts_with_qualifying_tables": []
    }
    dashboard["v32_priority_order"] = [
        "1. T53 automatic ProteinGym/DMS source locator; no user-filled CSV required",
        "2. T31/T32 automatic measured grain/nano material source hunt",
        "3. T44 automatic exact 3D-NAND source hunt",
        "4. T48b PV descriptor route only; T48a remains null control",
        "5. T57/T59 HEPData record/table CSV/YAML download gates",
        "6. Fusion T28/T30 H-mode DB first; T26/T27/T29 secondary until primary rows exist",
        "7. Exact-table-missing tests get active primary-table hunts, not demotion",
        "8. T50-T52 kept bound-only",
        "9. T34 restored into dashboard/output expectations",
        "10. v33 removes separated run modes; next/default run executes all tests"
    ]
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        v32 = frag.get("v32") or {}
        if frag.get("confirm_allowed_now"):
            dashboard["v32_confirm_status"]["strict_confirm_allowed_now"].append({"test_id": tid, "verdict": frag.get("verdict")})
        if tid == "T60":
            dashboard["v32_confirm_status"]["anchors_consistency_only"].append(tid)
        if tid in {"T53","T31","T32","T44","T48","T57","T59"}:
            dashboard["v32_confirm_status"]["active_confirmation_candidates"].append({"test_id": tid, "verdict": frag.get("verdict"), "confirm_allowed_now": frag.get("confirm_allowed_now")})
        if tid in {"T33","T34","T35","T36","T37","T38","T39","T40","T41","T42","T43","T49","T54","T55","T56","T58"}:
            dashboard["v32_confirm_status"]["primary_table_hunts"].append({"test_id": tid, "verdict": frag.get("verdict"), "confirm_allowed_now": frag.get("confirm_allowed_now")})
        if tid in {"T50","T51","T52"}:
            dashboard["v32_confirm_status"]["bounds_only"].append(tid)
        if tid in {"T28","T30"}:
            dashboard["v32_confirm_status"]["fusion_T28_T30_priority"].append({"test_id": tid, "verdict": frag.get("verdict")})
        if tid == "T34":
            dashboard["v32_confirm_status"]["restored_tests"].append(tid)
        for gate in v32.values():
            if isinstance(gate, dict):
                hunt = gate.get("hunt") or gate.get("source_hunt") or gate
                if isinstance(hunt, dict) and int(hunt.get("qualifying_table_count") or 0) > 0:
                    dashboard["v32_confirm_status"]["source_hunts_with_qualifying_tables"].append({"test_id": tid, "qualifying_table_count": hunt.get("qualifying_table_count")})
    dashboard["v32_recommended_next"] = [
        "v33 default: python run_all_tier_b.py --cache tierb_cache_v34 --outdir tierb_out_v34 --script-timeout 900",
        "Do not use split confirm/primary runs except --only for debugging.",
        "Inspect data/generated/*_primary_table_hunt_audit_v33.csv and *_auto_rows_v33.csv after the all-test run.",
        "Do not call anything confirmed unless v33_confirm_status.strict_confirm_allowed_now is non-empty."
    ]


    # v33: no user-filled CSV workflow, no separated run modes. All tests run
    # together by default; generated CSVs are public-data cache/audit outputs only.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v33"
    dashboard["v33_run_policy"] = {
        "default_selection": "all_tests_T26_to_T60",
        "confirm_candidates_flag": "deprecated_no_op_selector",
        "primary_table_hunt_flag": "deprecated_no_op_selector",
        "only_flag": "debug_only",
        "manual_or_user_filled_csv_required": False
    }
    dashboard["v33_confirm_status"] = {
        "strict_confirm_allowed_now": [],
        "consistency_anchors_not_full_confirms": [],
        "auto_source_hunts": [],
        "fusion_primary_data_search": [],
        "bounds_only": []
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        if frag.get("confirm_allowed_now"):
            dashboard["v33_confirm_status"]["strict_confirm_allowed_now"].append({"test_id": tid, "verdict": frag.get("verdict")})
        if tid == "T60":
            dashboard["v33_confirm_status"]["consistency_anchors_not_full_confirms"].append(tid)
        if tid in {"T26","T27","T28","T29","T30"}:
            dashboard["v33_confirm_status"]["fusion_primary_data_search"].append({"test_id": tid, "verdict": frag.get("verdict"), "confirm_allowed_now": frag.get("confirm_allowed_now")})
        if tid in {"T50","T51","T52"}:
            dashboard["v33_confirm_status"]["bounds_only"].append(tid)
        v33 = (frag.get("v33") or {})
        if v33:
            dashboard["v33_confirm_status"]["auto_source_hunts"].append({"test_id": tid, "v33_keys": sorted(v33.keys())})
    dashboard["v33_recommended_next"] = [
        "Run all tests: python run_all_tier_b.py --cache tierb_cache_v34 --outdir tierb_out_v34 --script-timeout 900 --max-papers 40 --max-tables 120",
        "Use --only only to debug a failing single test, not for the scientific run.",
        "Generated CSVs in data/generated are auto-written cache/audit artifacts; do not hand-fill them.",
        "For fusion, inspect fusion_primary_data_search_v33 and data/generated/t2*_primary_table_hunt_audit_v33.csv.",
        "Only v33_confirm_status.strict_confirm_allowed_now may be described as confirmed."
    ]



    # v34: automatic source loaders/miners for all nine requested improvements.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v34"
    dashboard["v34_run_policy"] = {
        "default_selection": "all_tests_T26_to_T60",
        "manual_or_user_filled_csv_required": False,
        "generated_csv_role": "auto-generated cache/audit outputs only",
        "confirmation_rule": "Only v34_confirm_status.strict_confirm_allowed_now may be called confirmed."
    }
    dashboard["v34_confirm_status"] = {
        "strict_confirm_allowed_now": [],
        "not_confirmed_blockers": [],
        "consistency_anchors_not_full_confirms": [],
        "bounds_only": [],
        "split_branch_tests": [],
        "auto_loaders_present": [],
        "fusion_primary_data_search": [],
        "hepdata_fallbacks": []
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        if frag.get("strict_confirm_allowed_now") or frag.get("confirm_allowed_now"):
            dashboard["v34_confirm_status"]["strict_confirm_allowed_now"].append({"test_id": tid, "verdict": frag.get("verdict")})
        else:
            dashboard["v34_confirm_status"]["not_confirmed_blockers"].append({
                "test_id": tid,
                "why_not_confirmed": frag.get("why_not_confirmed"),
                "single_next_blocker": frag.get("single_next_blocker"),
                "best_auto_data_source_next": frag.get("best_auto_data_source_next"),
            })
        if tid == "T60":
            dashboard["v34_confirm_status"]["consistency_anchors_not_full_confirms"].append(tid)
        if tid in {"T50", "T51", "T52"}:
            dashboard["v34_confirm_status"]["bounds_only"].append(tid)
        v34 = frag.get("v34") or {}
        if (v34.get("split_branch_policy_v34") or None):
            dashboard["v34_confirm_status"]["split_branch_tests"].append(tid)
        auto = (v34.get("auto_data_improvements_v34") or {}) if isinstance(v34, dict) else {}
        if auto:
            dashboard["v34_confirm_status"]["auto_loaders_present"].append({"test_id": tid, "loaders": sorted(auto.keys())})
        if tid in {"T26", "T27", "T28", "T29", "T30"}:
            dashboard["v34_confirm_status"]["fusion_primary_data_search"].append({"test_id": tid, "loader": "fusion_recursive_primary_data_funnel_v34" in auto})
        if tid in {"T57", "T59"}:
            dashboard["v34_confirm_status"]["hepdata_fallbacks"].append({"test_id": tid, "loader": "hepdata_json_yaml_fallback_v34" in auto})
    dashboard["v34_implemented_improvements"] = [
        "1. T53 ProteinGym/DMS + UniProt/PDB/RCSB auto-join audit rows",
        "2. T31/T32 CMB-S4 reference/microstructure miner for grain/nano branch",
        "3. T44 NAND fallback parser for Wikipedia/vendor/press-style public tables",
        "4. T48b NREL/NLR loader fix with descriptor-model-only route; T48a remains null control",
        "5. Fusion OSF/Zenodo recursive structured-file funnel with DB5.2.3 aliases",
        "6. T57/T59 HEPData JSON/YAML/original fallback before browser CSV URLs",
        "7. T50-T52 hard bound-only rule preserved",
        "8. Split branch policies: T32a/T32b and T40a/T40b",
        "9. Per-test why_not_confirmed/single_next_blocker/best_auto_data_source_next dashboard fields"
    ]
    dashboard["v34_recommended_next"] = [
        "Run all tests: python run_all_tier_b.py --cache tierb_cache_v34 --outdir tierb_out_v34 --script-timeout 900 --max-papers 40 --max-tables 120",
        "Inspect positive_dashboard.json -> v34_confirm_status.not_confirmed_blockers for one next blocker per test.",
        "Inspect data/generated/*_v34.csv audit/cache artifacts. They are generated by scripts, not user-filled inputs.",
        "Use --only only for debugging a failed test, not for scientific reporting."
    ]

    # v35 near-confirm dashboard: rank next-confirm opportunities and expose strict confirmation list.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v35"
    dashboard["v35_confirm_status"] = {
        "strict_confirm_allowed_now": [],
        "near_confirm_ranked": [],
        "bound_only": [],
        "missing_output_restored": [],
        "recommended_next": []
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        if frag.get("strict_confirm_allowed_now") or frag.get("confirm_allowed_now"):
            dashboard["v35_confirm_status"]["strict_confirm_allowed_now"].append({"test_id": tid, "verdict": frag.get("verdict")})
        ncs = frag.get("near_confirm_score") or {}
        if isinstance(ncs, dict):
            dashboard["v35_confirm_status"]["near_confirm_ranked"].append({
                "test_id": tid,
                "score_0_10": ncs.get("score_0_10", 0),
                "verdict": frag.get("verdict"),
                "single_next_blocker": frag.get("single_next_blocker"),
                "best_auto_data_source_next": frag.get("best_auto_data_source_next"),
            })
        if tid in {"T50", "T51", "T52"}:
            dashboard["v35_confirm_status"]["bound_only"].append(tid)
        if (frag.get("v35") or {}).get("fallback_restored_output"):
            dashboard["v35_confirm_status"]["missing_output_restored"].append(tid)
    dashboard["v35_confirm_status"]["near_confirm_ranked"] = sorted(
        dashboard["v35_confirm_status"]["near_confirm_ranked"],
        key=lambda x: (x.get("score_0_10") or 0),
        reverse=True
    )
    dashboard["v35_confirm_status"]["recommended_next"] = [
        "T44: normalize NAND rows and run layer-vs-year/manufacturer model to strict N>=20 gate",
        "T48b: parse NREL/NLR XLSX rows and run PV descriptor residual model",
        "T53: run ProteinGym residual model with symmetry/contact proxy and family/assay jackknife",
        "T31/T32: increase measured microstructure rows and run narrow grain/nano exponent gate",
        "T57/T59: use official HEPData record/submission/table YAML/JSON endpoints",
        "T28/T30: keep fusion funnel structured-file-only and reject metadata/search API frames",
        "T34: restored fallback JSON if script output is missing; debug T34 until real result JSON appears",
        "T50-T52: keep bound-only; do not attempt positive confirmation",
        "T60: implement quark/lattice, sector reshuffle, and look-elsewhere gates"
    ]


    # v36: T34 real structured runner + near-confirm hardening dashboard.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v36"
    dashboard["v36_confirm_status"] = {
        "strict_confirm_allowed_now": [],
        "near_confirm_ranked": [],
        "bound_only": [],
        "t34_status": [],
        "recommended_next": []
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        if frag.get("strict_confirm_allowed_now") or frag.get("confirm_allowed_now"):
            dashboard["v36_confirm_status"]["strict_confirm_allowed_now"].append({"test_id": tid, "verdict": frag.get("verdict")})
        ncs = frag.get("near_confirm_score") or {}
        if isinstance(ncs, dict):
            dashboard["v36_confirm_status"]["near_confirm_ranked"].append({
                "test_id": tid,
                "score_0_10": ncs.get("score_0_10", 0),
                "verdict": frag.get("verdict"),
                "single_next_blocker": frag.get("single_next_blocker"),
                "best_auto_data_source_next": frag.get("best_auto_data_source_next"),
            })
        if tid in {"T50", "T51", "T52"}:
            dashboard["v36_confirm_status"]["bound_only"].append(tid)
        if tid == "T34":
            dashboard["v36_confirm_status"]["t34_status"].append({
                "test_id": tid,
                "verdict": frag.get("verdict"),
                "score_0_10": (ncs or {}).get("score_0_10") if isinstance(ncs, dict) else None,
                "single_next_blocker": frag.get("single_next_blocker"),
            })
    dashboard["v36_confirm_status"]["near_confirm_ranked"] = sorted(
        dashboard["v36_confirm_status"]["near_confirm_ranked"],
        key=lambda x: (x.get("score_0_10") or 0),
        reverse=True
    )
    dashboard["v36_confirm_status"]["recommended_next"] = [
        "T34: use v36 structured thermoelectric runner; seek exact orientation/grain-angle ZT exports from teMatDb/Starrydata/Bi2Te3 supplements",
        "T44: increase complete NAND rows to N>=20 and run manufacturer jackknife",
        "T48b: ensure NREL/NLR XLSX rows write and run descriptor residual model",
        "T53: strengthen symmetry/contact proxy and family/assay jackknife",
        "T31/T32: increase measured microstructure rows; keep broad T32 as null/pressure control",
        "T28/T30: keep fusion structured-file-only; reject OSF/Zenodo metadata frames",
        "T57/T59: use official HEPData record/submission/table YAML/JSON, not browser CSV URLs",
        "T50-T52: keep bound-only",
        "T60: complete quark/lattice, sector reshuffle, and look-elsewhere gates"
    ]

    # v37: confirm-hardening dashboard. Only v37_confirm_status.strict_confirm_allowed_now may be called confirmed.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v37"
    dashboard["v37_confirm_status"] = {
        "strict_confirm_allowed_now": [],
        "near_confirm_ranked": [],
        "bound_only": [],
        "recommended_next": [],
        "policy": "No confirmation language unless listed in strict_confirm_allowed_now; near-confirm scores are prioritization only."
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        if frag.get("strict_confirm_allowed_now") or frag.get("confirm_allowed_now"):
            dashboard["v37_confirm_status"]["strict_confirm_allowed_now"].append({"test_id": tid, "verdict": frag.get("verdict")})
        ncs = frag.get("near_confirm_score") or {}
        if isinstance(ncs, dict):
            dashboard["v37_confirm_status"]["near_confirm_ranked"].append({
                "test_id": tid,
                "score_0_10": ncs.get("score_0_10", 0),
                "verdict": frag.get("verdict"),
                "single_next_blocker": frag.get("single_next_blocker"),
                "best_auto_data_source_next": frag.get("best_auto_data_source_next"),
            })
        if tid in {"T50", "T51", "T52"}:
            dashboard["v37_confirm_status"]["bound_only"].append(tid)
    dashboard["v37_confirm_status"]["near_confirm_ranked"] = sorted(
        dashboard["v37_confirm_status"]["near_confirm_ranked"],
        key=lambda x: (x.get("score_0_10") or 0),
        reverse=True
    )
    dashboard["v37_confirm_status"]["recommended_next"] = [
        "T48b: descriptor model hardening with family FDR and tandem/concentrator exclusion",
        "T44: expand normalized NAND layer/year/capacity rows and manufacturer jackknife",
        "T53: symmetry/contact proxy + family/assay jackknife on ProteinGym rows",
        "T31/T32: mine decisive microstructure metadata and join to kappa(T)",
        "T34: parse exact thermoelectric exports with orientation/grain-angle + ZT",
        "T57/T59: fill exact HEPData registry rows and parse official YAML/JSON",
        "T26-T30: reject metadata wrappers and resolve only structured fusion data files",
        "T45/T47: exact benchmark table parsers for optical/neuromorphic rows",
        "T60: complete quark/lattice, sector reshuffle, and look-elsewhere gates"
    ]

    # v38 dashboard summary: strict confirmation list and near-confirm ranking.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v38"
    dashboard["v38_confirm_status"] = {
        "strict_confirm_allowed_now": [t.get("test_id") for t in dashboard.get("tests", []) if t.get("strict_confirm_allowed_now")],
        "n_strict_confirmed": sum(1 for t in dashboard.get("tests", []) if t.get("strict_confirm_allowed_now")),
    }
    dashboard["v38_near_confirm_ranking"] = sorted([
        {"test_id": t.get("test_id"), "score": ((t.get("near_confirm_score") or {}).get("score_0_10") or 0), "single_next_blocker": t.get("single_next_blocker"), "why_not_confirmed": t.get("why_not_confirmed")}
        for t in dashboard.get("tests", [])
    ], key=lambda x: x.get("score") or 0, reverse=True)[:12]
    dashboard["recommended_next"] = [
        "T48b: use the v38 final gate; promote only if sign + AIC/BIC + family BH-FDR + tandem/concentrator exclusion pass.",
        "T44: expand NAND normalized rows to at least 20 and require manufacturer jackknife stability.",
        "T53: replace proxy-only structure flags with true PDB/AlphaFold symmetry/contact features and rerun jackknife.",
        "T31/T32: join microstructure metadata to kappa(T) rows before any confirmation language.",
        "T34: recover headers in teMatDb/Starrydata exports and require orientation/grain-angle ZT rows.",
        "Fusion: only score downloaded structured measurement files, never OSF/Zenodo metadata wrappers.",
        "T57/T59: use exact HEPData record/table/column registry and official YAML/JSON.",
        "T45/T47: use exact benchmark tables; keep as lower priority than T44/T48/T53.",
        "T60: keep T60a as anchor only until T60b/T60c/T60d pass.",
    ]

    # v39 dashboard: separate confirmed vs near-confirm vs anchor/bound/data-limited.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v39"
    confirmed_now = []
    near_next = []
    positive_anchor_only = []
    bound_only_v39 = []
    data_limited_or_open = []
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        ncs = frag.get("near_confirm_score") or {}
        if frag.get("strict_confirm_allowed_now"):
            confirmed_now.append({"test_id": tid, "label": label or "strict_confirmed", "verdict": frag.get("verdict")})
        elif tid == "T60" or label == "positive_anchor_only":
            positive_anchor_only.append(tid)
        elif tid in {"T50", "T51", "T52"}:
            bound_only_v39.append(tid)
        elif isinstance(ncs, dict) and (ncs.get("score_0_10") or 0) >= 4:
            near_next.append({
                "test_id": tid,
                "score_0_10": ncs.get("score_0_10"),
                "single_next_blocker": frag.get("single_next_blocker"),
                "best_auto_data_source_next": frag.get("best_auto_data_source_next"),
            })
        else:
            data_limited_or_open.append({"test_id": tid, "verdict": frag.get("verdict")})
    near_next = sorted(near_next, key=lambda x: x.get("score_0_10") or 0, reverse=True)
    dashboard["v39_confirm_status"] = {
        "confirmed_now": confirmed_now,
        "strict_confirm_allowed_now": [x["test_id"] for x in confirmed_now],
        "near_confirm_next": near_next[:12],
        "positive_anchor_only": positive_anchor_only,
        "bound_only": bound_only_v39,
        "data_limited_or_open_count": len(data_limited_or_open),
        "policy": "T48b compatible_positive is frozen if v38 final gate remains pass; T50-T52 are never positive confirms; T60a remains anchor-only until full null suite passes.",
    }
    dashboard["recommended_next_v39"] = [
        "Freeze T48b as compatible_positive and run robustness-only checks rather than moving the goalpost.",
        "T44: expand NAND normalized rows to >=20 and run manufacturer jackknife.",
        "T31/T32: normalize temperature_K/kappa_W_mK in joined microstructure rows and run source/material jackknife.",
        "T53: use PDB/AlphaFold/UniProt symmetry_order, oligomeric_state, contact_network_regularity, assay/family/sequence jackknife.",
        "T34: recover teMatDb/Starrydata headers into orientation/grain-angle ZT rows.",
        "T57/T59: fill exact HEPData record/table/column registry and parse official YAML/JSON.",
        "T45/T47: exact optical and neuromorphic benchmark parsers; lower priority than T44/T48/T53.",
        "Fusion: lower priority until exact public measurement attachments are available.",
        "T60: keep anchor-only until T60b/T60c/T60d pass with uncertainty-aware inputs.",
    ]
    # v40 dashboard: preserve confirmed T48b and rank next-confirm targets with current fragments.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v40"
    confirmed_now_v40 = []
    near_next_v40 = []
    positive_anchor_only_v40 = []
    bound_only_v40 = []
    data_limited_v40 = []
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        ncs = frag.get("near_confirm_score") or {}
        if frag.get("strict_confirm_allowed_now"):
            confirmed_now_v40.append({"test_id": tid, "label": label or "strict_confirmed", "verdict": frag.get("verdict")})
        elif tid == "T60" or label == "positive_anchor_only":
            positive_anchor_only_v40.append(tid)
        elif tid in {"T50", "T51", "T52"}:
            bound_only_v40.append(tid)
        elif isinstance(ncs, dict) and (ncs.get("score_0_10") or 0) >= 4:
            near_next_v40.append({
                "test_id": tid,
                "score_0_10": ncs.get("score_0_10"),
                "single_next_blocker": frag.get("single_next_blocker"),
                "best_auto_data_source_next": frag.get("best_auto_data_source_next"),
            })
        else:
            data_limited_v40.append({"test_id": tid, "verdict": frag.get("verdict")})
    near_next_v40 = sorted(near_next_v40, key=lambda x: x.get("score_0_10") or 0, reverse=True)
    dashboard["v40_confirm_status"] = {
        "confirmed_now": confirmed_now_v40,
        "strict_confirm_allowed_now": [x["test_id"] for x in confirmed_now_v40],
        "near_confirm_next": near_next_v40[:12],
        "positive_anchor_only": positive_anchor_only_v40,
        "bound_only": bound_only_v40,
        "data_limited_or_open_count": len(data_limited_v40),
        "policy": "T48b compatible_positive is preserved/frozen; T50-T52 are bound-only; T60a remains anchor-only until full null suite passes.",
    }
    dashboard["recommended_next_v40"] = [
        "T48b: preserve compatible_positive and run robustness only: leave-one-family-out, certification-source jackknife, year-block jackknife.",
        "T44: use Tier A/B NAND rows only; require positive layer coefficient and manufacturer jackknife.",
        "T31/T32: normalize temperature_K/kappa_W_mK and run narrow grain/nano branch only.",
        "T53: add real PDB/AlphaFold symmetry/contact proxies and DMS outcome jackknife.",
        "T34: recover exact thermoelectric headers and map Bi2Te3/Sb2Te3 orientation/ZT rows.",
        "T57/T59: fill exact HEPData record/table/column registry and parse official YAML/JSON.",
        "T45/T47: exact benchmark tables only; lower priority than T44/T31/T53.",
        "Fusion: exact public measurement attachments only; no metadata/PDF scoring.",
        "T60: keep anchor-only until T60b/T60c/T60d pass.",
    ]

    # v41 dashboard: preserve confirmed branches and prioritize next confirmations.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v41"
    confirmed_now_v41 = []
    near_next_v41 = []
    positive_anchor_only_v41 = []
    bound_only_v41 = []
    data_limited_v41 = []
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        ncs = frag.get("near_confirm_score") or {}
        if frag.get("strict_confirm_allowed_now"):
            confirmed_now_v41.append({"test_id": tid, "label": label or "strict_confirmed", "verdict": frag.get("verdict"), "robustness_next": frag.get("single_next_blocker")})
        elif tid == "T60" or label == "positive_anchor_only":
            positive_anchor_only_v41.append(tid)
        elif tid in {"T50", "T51", "T52"}:
            bound_only_v41.append(tid)
        elif isinstance(ncs, dict) and (ncs.get("score_0_10") or 0) >= 4:
            near_next_v41.append({
                "test_id": tid,
                "score_0_10": ncs.get("score_0_10"),
                "single_next_blocker": frag.get("single_next_blocker"),
                "best_auto_data_source_next": frag.get("best_auto_data_source_next"),
            })
        else:
            data_limited_v41.append({"test_id": tid, "verdict": frag.get("verdict")})
    near_next_v41 = sorted(near_next_v41, key=lambda x: x.get("score_0_10") or 0, reverse=True)
    dashboard["v41_confirm_status"] = {
        "confirmed_now": confirmed_now_v41,
        "strict_confirm_allowed_now": [x["test_id"] for x in confirmed_now_v41],
        "near_confirm_next": near_next_v41[:12],
        "positive_anchor_only": positive_anchor_only_v41,
        "bound_only": bound_only_v41,
        "data_limited_or_open_count": len(data_limited_v41),
        "policy": "T48b and T44 are preserved when prior strict gates remain passed; T50-T52 are bound-only; T60a is anchor-only until the full null suite passes.",
    }
    dashboard["recommended_next_v41"] = [
        "T48b/T44: preserve confirms and run robustness dashboards only; do not move gates.",
        "T53: complete final DMS/PDB/AlphaFold structure-contact proxy model with family/assay/sequence jackknife.",
        "T31/T32: normalize temperature_K/kappa_W_mK and test only the narrow grain/nano branch.",
        "T34: map exact thermoelectric exports into Bi2Te3/Sb2Te3 orientation/ZT rows.",
        "T57/T59: exact HEPData registry only, with record/table/column names and YAML/JSON parse.",
        "T45/T47: exact public benchmark rows only; lower ROI than T53/T31/T34.",
        "Fusion: exact structured public measurement attachments only; no metadata/PDF scoring.",
        "T60: keep anchor-only until T60b/T60c/T60d pass with uncertainty-aware inputs.",
    ]



    # v42 dashboard: preserve current strict positives and focus next-confirm loaders.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v42"
    confirmed_now_v42 = []
    near_next_v42 = []
    positive_anchor_only_v42 = []
    bound_only_v42 = []
    data_limited_v42 = []
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        ncs = frag.get("near_confirm_score") or {}
        if frag.get("strict_confirm_allowed_now"):
            confirmed_now_v42.append({"test_id": tid, "label": label or "strict_confirmed", "verdict": frag.get("verdict"), "robustness_next": frag.get("single_next_blocker")})
        elif tid == "T60" or label == "positive_anchor_only":
            positive_anchor_only_v42.append(tid)
        elif tid in {"T50", "T51", "T52"}:
            bound_only_v42.append(tid)
        elif isinstance(ncs, dict) and (ncs.get("score_0_10") or 0) >= 4:
            near_next_v42.append({
                "test_id": tid,
                "score_0_10": ncs.get("score_0_10"),
                "single_next_blocker": frag.get("single_next_blocker"),
                "best_auto_data_source_next": frag.get("best_auto_data_source_next"),
            })
        else:
            data_limited_v42.append({"test_id": tid, "verdict": frag.get("verdict")})
    near_next_v42 = sorted(near_next_v42, key=lambda x: x.get("score_0_10") or 0, reverse=True)
    dashboard["v42_confirm_status"] = {
        "confirmed_now": confirmed_now_v42,
        "strict_confirm_allowed_now": [x["test_id"] for x in confirmed_now_v42],
        "near_confirm_next": near_next_v42[:12],
        "positive_anchor_only": positive_anchor_only_v42,
        "bound_only": bound_only_v42,
        "data_limited_or_open_count": len(data_limited_v42),
        "policy": "T48b and T44 stay frozen positives; v42 adds robustness dashboards plus final loaders for T53, T31/T32, T34, HEPData, T45/T47, fusion exact attachments, and T60 null suite.",
    }
    dashboard["recommended_next_v42"] = [
        "T48b/T44: preserve frozen confirms; run robustness dashboards only, no moving gates.",
        "T53: final DMS/PDB/AlphaFold proxy model with family/assay/sequence jackknife.",
        "T31/T32: strict narrow grain/nano κ(T)+microstructure model only.",
        "T34: exact thermoelectric Bi2Te3/Sb2Te3 orientation/ZT mapping and cos(6θ) model.",
        "T57/T59: explicit HEPData record/table/column registry and official YAML/JSON fetch order.",
        "T45: exact optical energy/bit + bandwidth + reach benchmark rows.",
        "T47: exact neuromorphic energy/inference + accuracy benchmark rows.",
        "Fusion: exact structured measurement attachments only; metadata/schema/PDF remain rejected.",
        "T60: preserve anchor-only until T60b/T60c/T60d full null suite passes.",
    ]


    # v43 dashboard: preserve current strict positives and report robustness / next-confirm targets.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v43"
    confirmed_now_v43 = []
    near_next_v43 = []
    positive_anchor_only_v43 = []
    bound_only_v43 = []
    data_limited_v43 = []
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        ncs = frag.get("near_confirm_score") or {}
        if frag.get("strict_confirm_allowed_now"):
            confirmed_now_v43.append({"test_id": tid, "label": label or "strict_confirmed", "verdict": frag.get("verdict"), "robustness_next": frag.get("single_next_blocker")})
        elif tid == "T60" or label == "positive_anchor_only":
            positive_anchor_only_v43.append(tid)
        elif tid in {"T50", "T51", "T52"}:
            bound_only_v43.append(tid)
        elif isinstance(ncs, dict) and (ncs.get("score_0_10") or 0) >= 4:
            near_next_v43.append({
                "test_id": tid,
                "score_0_10": ncs.get("score_0_10"),
                "single_next_blocker": frag.get("single_next_blocker"),
                "best_auto_data_source_next": frag.get("best_auto_data_source_next"),
            })
        else:
            data_limited_v43.append({"test_id": tid, "verdict": frag.get("verdict")})
    near_next_v43 = sorted(near_next_v43, key=lambda x: x.get("score_0_10") or 0, reverse=True)
    dashboard["v43_confirm_status"] = {
        "confirmed_now": confirmed_now_v43,
        "strict_confirm_allowed_now": [x["test_id"] for x in confirmed_now_v43],
        "near_confirm_next": near_next_v43[:12],
        "positive_anchor_only": positive_anchor_only_v43,
        "bound_only": bound_only_v43,
        "data_limited_or_open_count": len(data_limited_v43),
        "policy": "T48b and T44 stay frozen positives; v43 adds robustness artifacts and stricter next-confirm loaders for T53, T31/T32, T34, HEPData, T45/T47, fusion exact attachments, and T60 null suite.",
    }
    dashboard["recommended_next_v43"] = [
        "T48b/T44: preserve frozen confirms; run robustness dashboards only, no moving gates.",
        "T53: final DMS/PDB/AlphaFold proxy model with family/assay/sequence jackknife.",
        "T31/T32: strict narrow grain/nano κ(T)+microstructure model only.",
        "T34: exact thermoelectric Bi2Te3/Sb2Te3 orientation/ZT mapping and cos(6θ) model.",
        "T57/T59: explicit HEPData record/table/column registry and official YAML/JSON fetch order.",
        "T45: exact optical energy/bit + bandwidth + reach benchmark rows.",
        "T47: exact neuromorphic energy/inference + accuracy benchmark rows.",
        "Fusion: exact structured measurement attachments only; metadata/schema/PDF remain rejected.",
        "T60: preserve anchor-only until T60b/T60c/T60d full null suite passes.",
    ]


    # v44 dashboard: preserve confirmed branches and report robustness / next-confirm targets.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v44"
    confirmed_now_v44 = []
    near_next_v44 = []
    positive_anchor_only_v44 = []
    bound_only_v44 = []
    data_limited_v44 = []
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        ncs = frag.get("near_confirm_score") or {}
        if frag.get("strict_confirm_allowed_now"):
            confirmed_now_v44.append({"test_id": tid, "label": label or "strict_confirmed", "verdict": frag.get("verdict"), "robustness_next": frag.get("single_next_blocker")})
        elif tid == "T60" or label == "positive_anchor_only":
            positive_anchor_only_v44.append(tid)
        elif tid in {"T50", "T51", "T52"}:
            bound_only_v44.append(tid)
        elif isinstance(ncs, dict) and (ncs.get("score_0_10") or 0) >= 4:
            near_next_v44.append({
                "test_id": tid,
                "score_0_10": ncs.get("score_0_10"),
                "single_next_blocker": frag.get("single_next_blocker"),
                "best_auto_data_source_next": frag.get("best_auto_data_source_next"),
            })
        else:
            data_limited_v44.append({"test_id": tid, "verdict": frag.get("verdict")})
    near_next_v44 = sorted(near_next_v44, key=lambda x: x.get("score_0_10") or 0, reverse=True)
    dashboard["v44_confirm_status"] = {
        "confirmed_now": confirmed_now_v44,
        "strict_confirm_allowed_now": [x["test_id"] for x in confirmed_now_v44],
        "near_confirm_next": near_next_v44[:12],
        "positive_anchor_only": positive_anchor_only_v44,
        "bound_only": bound_only_v44,
        "data_limited_or_open_count": len(data_limited_v44),
        "policy": "T48b and T44 stay frozen positives; v44 adds robustness-only dashboards plus stricter next-confirm loaders for T53, T31/T32, T34, HEPData, T45/T47, fusion exact attachments, and T60 null suite.",
    }
    dashboard["recommended_next_v44"] = [
        "T48b/T44: preserve frozen confirms; run robustness dashboards only, no moving gates.",
        "T53: final DMS/PDB/AlphaFold proxy model with bootstrap or BH-FDR and family/assay/sequence jackknife.",
        "T31/T32: strict narrow grain/nano κ(T)+microstructure model only.",
        "T34: exact thermoelectric Bi2Te3/Sb2Te3 orientation/ZT mapping and cos(6θ) model.",
        "T57/T59: explicit HEPData record/table/column registry and official YAML/JSON fetch order.",
        "T45: exact optical energy/bit + bandwidth + reach benchmark rows.",
        "T47: exact neuromorphic energy/inference + accuracy benchmark rows.",
        "Fusion: exact structured measurement attachments only; metadata/schema/PDF remain rejected.",
        "T60: preserve anchor-only until T60b/T60c/T60d full null suite passes.",
    ]



    # v45 dashboard: confirmed vs near-confirm split. This layer is confirm-preserving:
    # T48b/T44 stay frozen; all other branches are ranked for next work.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v45"
    v45_status = {
        "confirmed_now": [],
        "strict_confirm_allowed_now": [],
        "near_confirm_next": [],
        "positive_anchor_only": [],
        "bound_only": [],
        "data_limited_positive_paths": [],
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        strict = bool(frag.get("strict_confirm_allowed_now"))
        score = ((frag.get("near_confirm_score") or {}).get("score_0_10") or 0)
        if strict:
            name = "T48b" if tid == "T48" else tid
            if name not in v45_status["confirmed_now"]:
                v45_status["confirmed_now"].append(name)
            if name not in v45_status["strict_confirm_allowed_now"]:
                v45_status["strict_confirm_allowed_now"].append(name)
        elif label == "positive_anchor_only" or tid == "T60":
            if "T60a" not in v45_status["positive_anchor_only"]:
                v45_status["positive_anchor_only"].append("T60a")
        elif tid in {"T50", "T51", "T52"}:
            if tid not in v45_status["bound_only"]:
                v45_status["bound_only"].append(tid)
        elif tid in {"T53", "T31", "T32", "T34"} or score >= 6:
            if tid not in v45_status["near_confirm_next"]:
                v45_status["near_confirm_next"].append(tid)
        elif tid in {"T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59"}:
            if tid not in v45_status["data_limited_positive_paths"]:
                v45_status["data_limited_positive_paths"].append(tid)
    # Stable order for report readability.
    order = ["T48b", "T44", "T53", "T31", "T32", "T34", "T60a", "T50", "T51", "T52"]
    for k in list(v45_status.keys()):
        v45_status[k] = sorted(set(v45_status[k]), key=lambda x: order.index(x) if x in order else 99)
    dashboard["v45_confirm_status"] = v45_status
    dashboard["recommended_next_v45"] = [
        "Do robustness-only auditing for frozen confirms T48b and T44; do not move their gates.",
        "Prioritize T53 final DMS/PDB/AlphaFold proxy model for the next independent confirmation.",
        "Fix strict temperature_K/kappa_W_mK/grain-size normalization for T31/T32 measured grain/nano branch.",
        "Fix T34 exact thermoelectric exports for Bi2Te3/Sb2Te3 orientation/ZT rows.",
        "Keep fusion low priority until exact structured measurement attachments are known.",
        "Use exact HEPData record/table/column registries for T57/T59; no broad search.",
        "Run T45/T47 only on exact benchmark tables; keep T50-T52 bound-only and T60a anchor-only.",
    ]


    # v46 dashboard: preserve frozen confirms and rank next-confirm targets with v46 artifacts.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v46"
    v46_status = {
        "confirmed_now": [],
        "strict_confirm_allowed_now": [],
        "near_confirm_next": [],
        "positive_anchor_only": [],
        "bound_only": [],
        "data_limited_positive_paths": [],
        "robustness_only_confirmed": [],
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        strict = bool(frag.get("strict_confirm_allowed_now"))
        score = ((frag.get("near_confirm_score") or {}).get("score_0_10") or 0)
        if strict:
            name = "T48b" if tid == "T48" else tid
            if name not in v46_status["confirmed_now"]:
                v46_status["confirmed_now"].append(name)
            if name not in v46_status["strict_confirm_allowed_now"]:
                v46_status["strict_confirm_allowed_now"].append(name)
            if name in {"T48b", "T44"} and name not in v46_status["robustness_only_confirmed"]:
                v46_status["robustness_only_confirmed"].append(name)
        elif label == "positive_anchor_only" or tid == "T60":
            if "T60a" not in v46_status["positive_anchor_only"]:
                v46_status["positive_anchor_only"].append("T60a")
        elif tid in {"T50", "T51", "T52"}:
            if tid not in v46_status["bound_only"]:
                v46_status["bound_only"].append(tid)
        elif tid in {"T53", "T31", "T32", "T34"} or score >= 6:
            if tid not in v46_status["near_confirm_next"]:
                v46_status["near_confirm_next"].append(tid)
        elif tid in {"T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59"}:
            if tid not in v46_status["data_limited_positive_paths"]:
                v46_status["data_limited_positive_paths"].append(tid)
    order = ["T48b", "T44", "T53", "T31", "T32", "T34", "T60a", "T50", "T51", "T52"]
    for k in list(v46_status.keys()):
        v46_status[k] = sorted(set(v46_status[k]), key=lambda x: order.index(x) if x in order else 99)
    dashboard["v46_confirm_status"] = v46_status
    dashboard["recommended_next_v46"] = [
        "T48b/T44: robustness-only dashboards; frozen confirms, do not move gates.",
        "T53: final DMS/PDB/AlphaFold model with bootstrap CI or BH-FDR and family/assay/sequence jackknife.",
        "T31/T32: strict grain/nano kappa(T)+microstructure parser and model; broad T^0.5 remains pressure/control.",
        "T34: exact thermoelectric Bi2Te3/Sb2Te3 export parser and cos(6theta) model.",
        "T57/T59: exact HEPData record/table/column registry only.",
        "T45/T47: exact public benchmark rows only, after T53/T31/T34.",
        "Fusion: exact structured measurement attachments only; metadata/schema/PDF rejected.",
        "T50-T52: bound-only forever; T60a anchor-only until T60b/T60c/T60d pass.",
    ]


    # v47 dashboard: confirm-preserving layer with robustness-only frozen positives and ranked next-confirm targets.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v47"
    v47_status = {
        "confirmed_now": [],
        "strict_confirm_allowed_now": [],
        "near_confirm_next": [],
        "positive_anchor_only": [],
        "bound_only": [],
        "data_limited_positive_paths": [],
        "robustness_only_confirmed": [],
        "tier_a_audit_needed": [],
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        strict = bool(frag.get("strict_confirm_allowed_now"))
        score = ((frag.get("near_confirm_score") or {}).get("score_0_10") or 0)
        if strict:
            name = "T48b" if tid == "T48" else tid
            if name not in v47_status["confirmed_now"]:
                v47_status["confirmed_now"].append(name)
            if name not in v47_status["strict_confirm_allowed_now"]:
                v47_status["strict_confirm_allowed_now"].append(name)
            if name in {"T48b", "T44"} and name not in v47_status["robustness_only_confirmed"]:
                v47_status["robustness_only_confirmed"].append(name)
            if name == "T44" and name not in v47_status["tier_a_audit_needed"]:
                v47_status["tier_a_audit_needed"].append(name)
        elif label == "positive_anchor_only" or tid == "T60":
            if "T60a" not in v47_status["positive_anchor_only"]:
                v47_status["positive_anchor_only"].append("T60a")
        elif tid in {"T50", "T51", "T52"}:
            if tid not in v47_status["bound_only"]:
                v47_status["bound_only"].append(tid)
        elif tid in {"T53", "T31", "T32", "T34"} or score >= 6:
            if tid not in v47_status["near_confirm_next"]:
                v47_status["near_confirm_next"].append(tid)
        elif tid in {"T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59"}:
            if tid not in v47_status["data_limited_positive_paths"]:
                v47_status["data_limited_positive_paths"].append(tid)
    order = ["T48b", "T44", "T53", "T31", "T32", "T34", "T60a", "T50", "T51", "T52"]
    for k in list(v47_status.keys()):
        v47_status[k] = sorted(set(v47_status[k]), key=lambda x: order.index(x) if x in order else 99)
    dashboard["v47_confirm_status"] = v47_status
    dashboard["recommended_next_v47"] = [
        "T48b/T44: frozen confirms; robustness dashboards only, no moving gates.",
        "T44: recover real Tier-A NAND rows and independent source-domain audit.",
        "T53: final DMS/PDB/AlphaFold model with bootstrap CI/BH-FDR and family/assay/sequence jackknife.",
        "T31/T32: strict grain/nano kappa(T)+microstructure parser; broad T^0.5 remains pressure/control.",
        "T34: exact thermoelectric Bi2Te3/Sb2Te3 export parser and cos(6theta) model.",
        "T57/T59: exact HEPData record/table/column registry only.",
        "T45/T47: exact public benchmark rows only, after T53/T31/T34.",
        "Fusion: exact structured measurement attachments only; metadata/schema/PDF rejected.",
        "T50-T52: bound-only forever; T60a anchor-only until T60b/T60c/T60d pass.",
    ]


    # v48 dashboard: confirm-preserving layer with stronger fallback surfacing and next-confirm targeting.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v48"
    v48_status = {
        "confirmed_now": [],
        "strict_confirm_allowed_now": [],
        "near_confirm_next": [],
        "positive_anchor_only": [],
        "bound_only": [],
        "data_limited_positive_paths": [],
        "robustness_only_confirmed": [],
        "tier_a_audit_needed": [],
        "missing_output_or_timeout": [],
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        strict = bool(frag.get("strict_confirm_allowed_now"))
        score = ((frag.get("near_confirm_score") or {}).get("score_0_10") or 0)
        verdict = str(frag.get("verdict") or "")
        if strict:
            name = "T48b" if tid == "T48" else tid
            if name not in v48_status["confirmed_now"]: v48_status["confirmed_now"].append(name)
            if name not in v48_status["strict_confirm_allowed_now"]: v48_status["strict_confirm_allowed_now"].append(name)
            if name in {"T48b", "T44"} and name not in v48_status["robustness_only_confirmed"]: v48_status["robustness_only_confirmed"].append(name)
            if name == "T44" and name not in v48_status["tier_a_audit_needed"]: v48_status["tier_a_audit_needed"].append(name)
        elif label == "positive_anchor_only" or tid == "T60":
            if "T60a" not in v48_status["positive_anchor_only"]: v48_status["positive_anchor_only"].append("T60a")
        elif tid in {"T50", "T51", "T52"}:
            if tid not in v48_status["bound_only"]: v48_status["bound_only"].append(tid)
        elif "missing_output" in verdict or "fallback" in verdict:
            if tid not in v48_status["missing_output_or_timeout"]: v48_status["missing_output_or_timeout"].append(tid)
        elif tid in {"T53", "T31", "T32", "T34"} or score >= 6:
            if tid not in v48_status["near_confirm_next"]: v48_status["near_confirm_next"].append(tid)
        elif tid in {"T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59"}:
            if tid not in v48_status["data_limited_positive_paths"]: v48_status["data_limited_positive_paths"].append(tid)
    order = ["T48b", "T44", "T53", "T31", "T32", "T34", "T60a", "T50", "T51", "T52"]
    for k in list(v48_status.keys()):
        v48_status[k] = sorted(set(v48_status[k]), key=lambda x: order.index(x) if x in order else 99)
    dashboard["v48_confirm_status"] = v48_status
    dashboard["recommended_next_v48"] = [
        "T48b/T44: frozen confirms; robustness dashboards only, no gate changes.",
        "T44: recover real Tier-A NAND rows with complete company/year/layers/capacity/die_area/bits_per_cell/source_url fields.",
        "T53: complete ProteinGym DMS outcome to UniProt/PDB/AlphaFold structural proxy join and run bootstrap/BH-FDR + family/assay/sequence jackknife.",
        "T31/T32: strict measured grain/nano kappa(T)+microstructure parser; broad T^0.5 remains pressure/control.",
        "T34: exact teMatDb/Starrydata Bi2Te3/Sb2Te3 export parser and cos(6theta) model.",
        "T57/T59: exact HEPData record/table/column registry only; no broad discovery.",
        "T45/T47: exact public benchmark tables only after T53/T31/T34.",
        "Fusion: exact structured measurement attachments only; missing-output/timeouts are surfaced separately.",
        "T50-T52: bound-only forever; T60a anchor-only until T60b/T60c/T60d pass.",
    ]



    # v49 dashboard: confirm-preserving layer plus T31/T32 third-confirm targeting.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v49"
    v49_status = {
        "confirmed_now": [],
        "strict_confirm_allowed_now": [],
        "near_confirm_next": [],
        "positive_anchor_only": [],
        "bound_only": [],
        "data_limited_positive_paths": [],
        "robustness_only_confirmed": [],
        "tier_a_audit_needed": [],
        "missing_output_or_timeout": [],
        "third_confirm_priority": [],
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        strict = bool(frag.get("strict_confirm_allowed_now"))
        score = ((frag.get("near_confirm_score") or {}).get("score_0_10") or 0)
        verdict = str(frag.get("verdict") or "")
        if strict:
            name = "T48b" if tid == "T48" else tid
            if name not in v49_status["confirmed_now"]: v49_status["confirmed_now"].append(name)
            if name not in v49_status["strict_confirm_allowed_now"]: v49_status["strict_confirm_allowed_now"].append(name)
            if name in {"T48b", "T44"} and name not in v49_status["robustness_only_confirmed"]: v49_status["robustness_only_confirmed"].append(name)
            if name == "T44" and name not in v49_status["tier_a_audit_needed"]: v49_status["tier_a_audit_needed"].append(name)
        elif label == "positive_anchor_only" or tid == "T60":
            if "T60a" not in v49_status["positive_anchor_only"]: v49_status["positive_anchor_only"].append("T60a")
        elif tid in {"T50", "T51", "T52"}:
            if tid not in v49_status["bound_only"]: v49_status["bound_only"].append(tid)
        elif "missing_output" in verdict or "fallback" in verdict or "timeout" in verdict:
            if tid not in v49_status["missing_output_or_timeout"]: v49_status["missing_output_or_timeout"].append(tid)
        elif tid in {"T31", "T32"} or score >= 8:
            if tid not in v49_status["third_confirm_priority"]: v49_status["third_confirm_priority"].append(tid)
            if tid not in v49_status["near_confirm_next"]: v49_status["near_confirm_next"].append(tid)
        elif tid in {"T53", "T34"} or score >= 6:
            if tid not in v49_status["near_confirm_next"]: v49_status["near_confirm_next"].append(tid)
        elif tid in {"T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59"}:
            if tid not in v49_status["data_limited_positive_paths"]: v49_status["data_limited_positive_paths"].append(tid)
    order = ["T48b", "T44", "T31", "T32", "T53", "T34", "T60a", "T50", "T51", "T52"]
    for k in list(v49_status.keys()):
        v49_status[k] = sorted(set(v49_status[k]), key=lambda x: order.index(x) if x in order else 99)
    dashboard["v49_confirm_status"] = v49_status
    dashboard["recommended_next_v49"] = [
        "T31/T32: make the measured grain/nano kappa(T)+microstructure model pass temperature-baseline and material/source/temperature jackknife gates; this is the fastest Confirm #3 route.",
        "T48b/T44: frozen confirms; robustness dashboards only, no gate changes.",
        "T44: recover real Tier-A NAND rows with complete company/year/layers/capacity/die_area/bits_per_cell/source_url fields.",
        "T53: complete ProteinGym DMS outcome to UniProt/PDB/AlphaFold structural proxy join and run bootstrap/BH-FDR plus family/assay/sequence jackknife.",
        "T34: exact teMatDb/Starrydata Bi2Te3/Sb2Te3 export parser and cos(6theta) model.",
        "T57/T59: exact HEPData record/table/column registry only; no broad discovery.",
        "T45/T47: exact public benchmark tables only after T31/T32/T53/T34.",
        "Fusion: exact structured measurement attachments only; missing-output/timeouts are surfaced separately.",
        "T50-T52: bound-only forever; T60a anchor-only until T60b/T60c/T60d pass.",
    ]


    # v50 dashboard: confirm-preserving layer plus T31/T32 adaptive Confirm #3 targeting.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v50"
    v50_status = {
        "confirmed_now": [],
        "strict_confirm_allowed_now": [],
        "near_confirm_next": [],
        "positive_anchor_only": [],
        "bound_only": [],
        "data_limited_positive_paths": [],
        "robustness_only_confirmed": [],
        "tier_a_audit_needed": [],
        "missing_output_or_timeout": [],
        "third_confirm_priority": [],
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        strict = bool(frag.get("strict_confirm_allowed_now"))
        score = ((frag.get("near_confirm_score") or {}).get("score_0_10") or 0)
        verdict = str(frag.get("verdict") or "")
        if strict:
            name = "T48b" if tid == "T48" else tid
            if name not in v50_status["confirmed_now"]: v50_status["confirmed_now"].append(name)
            if name not in v50_status["strict_confirm_allowed_now"]: v50_status["strict_confirm_allowed_now"].append(name)
            if name in {"T48b", "T44"} and name not in v50_status["robustness_only_confirmed"]: v50_status["robustness_only_confirmed"].append(name)
            if name == "T44" and name not in v50_status["tier_a_audit_needed"]: v50_status["tier_a_audit_needed"].append(name)
        elif label == "positive_anchor_only" or tid == "T60":
            if "T60a" not in v50_status["positive_anchor_only"]: v50_status["positive_anchor_only"].append("T60a")
        elif tid in {"T50", "T51", "T52"}:
            if tid not in v50_status["bound_only"]: v50_status["bound_only"].append(tid)
        elif "missing_output" in verdict or "fallback" in verdict or "timeout" in verdict:
            if tid not in v50_status["missing_output_or_timeout"]: v50_status["missing_output_or_timeout"].append(tid)
        elif tid in {"T31", "T32"} or score >= 8:
            if tid not in v50_status["third_confirm_priority"]: v50_status["third_confirm_priority"].append(tid)
            if tid not in v50_status["near_confirm_next"]: v50_status["near_confirm_next"].append(tid)
        elif tid in {"T53", "T34"} or score >= 6:
            if tid not in v50_status["near_confirm_next"]: v50_status["near_confirm_next"].append(tid)
        elif tid in {"T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59"}:
            if tid not in v50_status["data_limited_positive_paths"]: v50_status["data_limited_positive_paths"].append(tid)
    order = ["T48b", "T44", "T31", "T32", "T53", "T34", "T60a", "T50", "T51", "T52"]
    for k in list(v50_status.keys()):
        v50_status[k] = sorted(set(v50_status[k]), key=lambda x: order.index(x) if x in order else 99)
    dashboard["v50_confirm_status"] = v50_status
    dashboard["recommended_next_v50"] = [
        "T31/T32: adaptive grain-size and boundary-density kappa(T)+microstructure models; Confirm #3 only if AIC/BIC, sign, bootstrap, material/source/temperature jackknives all pass.",
        "T48b/T44: frozen confirms; robustness dashboards only, no gate changes.",
        "T44: recover real Tier-A NAND rows with company/year/layers/capacity/die_area/bits_per_cell/source_url; derived die-area rows are audit-only.",
        "T53: complete ProteinGym DMS outcome to UniProt/PDB/AlphaFold structural proxy join and run bootstrap/BH-FDR plus family/assay/sequence jackknife.",
        "T34: exact teMatDb/Starrydata Bi2Te3/Sb2Te3 export parser and cos(6theta) model.",
        "T57/T59: exact HEPData record/table/column registry only; no broad discovery.",
        "T45/T47: exact public benchmark tables only after T31/T32/T53/T34.",
        "Fusion: exact structured measurement attachments only; missing-output/timeouts are surfaced separately.",
        "T50-T52: bound-only forever; T60a anchor-only until T60b/T60c/T60d pass.",
    ]



    # v51 dashboard: measured-only T31/T32 confirm targeting plus frozen-confirm audits.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v51"
    v51_status = {
        "confirmed_now": [],
        "strict_confirm_allowed_now": [],
        "near_confirm_next": [],
        "measured_microstructure_priority": [],
        "positive_anchor_only": [],
        "bound_only": [],
        "data_limited_positive_paths": [],
        "robustness_only_confirmed": [],
        "tier_a_audit_needed": [],
        "missing_output_or_timeout": [],
    }
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        label = frag.get("confirmation_label")
        strict = bool(frag.get("strict_confirm_allowed_now"))
        score = ((frag.get("near_confirm_score") or {}).get("score_0_10") or 0)
        verdict = str(frag.get("verdict") or "")
        if strict:
            name = "T48b" if tid == "T48" else tid
            if name not in v51_status["confirmed_now"]: v51_status["confirmed_now"].append(name)
            if name not in v51_status["strict_confirm_allowed_now"]: v51_status["strict_confirm_allowed_now"].append(name)
            if name in {"T48b", "T44"} and name not in v51_status["robustness_only_confirmed"]: v51_status["robustness_only_confirmed"].append(name)
            if name == "T44" and name not in v51_status["tier_a_audit_needed"]: v51_status["tier_a_audit_needed"].append(name)
        elif label == "positive_anchor_only" or tid == "T60":
            if "T60a" not in v51_status["positive_anchor_only"]: v51_status["positive_anchor_only"].append("T60a")
        elif tid in {"T50", "T51", "T52"}:
            if tid not in v51_status["bound_only"]: v51_status["bound_only"].append(tid)
        elif "missing_output" in verdict or "fallback" in verdict or "timeout" in verdict:
            if tid not in v51_status["missing_output_or_timeout"]: v51_status["missing_output_or_timeout"].append(tid)
        elif tid in {"T31", "T32"}:
            if tid not in v51_status["measured_microstructure_priority"]: v51_status["measured_microstructure_priority"].append(tid)
            if tid not in v51_status["near_confirm_next"]: v51_status["near_confirm_next"].append(tid)
        elif tid in {"T53", "T34"} or score >= 6:
            if tid not in v51_status["near_confirm_next"]: v51_status["near_confirm_next"].append(tid)
        elif tid in {"T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59"}:
            if tid not in v51_status["data_limited_positive_paths"]: v51_status["data_limited_positive_paths"].append(tid)
    order = ["T48b", "T44", "T31", "T32", "T53", "T34", "T60a", "T50", "T51", "T52"]
    for k in list(v51_status.keys()):
        v51_status[k] = sorted(set(v51_status[k]), key=lambda x: order.index(x) if x in order else 99)
    dashboard["v51_confirm_status"] = v51_status
    dashboard["recommended_next_v51"] = [
        "T31/T32: measured microstructure only; proxy-only rows cannot confirm. Require grain/boundary-density model to beat temperature baseline with grouped bootstrap and material/source/temperature jackknife.",
        "T44: frozen confirm; true Tier-A audit only with company/year/layers/capacity/die_area/bits_per_cell/source_url.",
        "T48b: frozen compatible-positive; robustness-only leave-one-family/source/year descriptor audit.",
        "T53: real ProteinGym DMS to UniProt/PDB/AlphaFold join; then bootstrap/BH-FDR and family/assay/sequence jackknife.",
        "T34: exact teMatDb/Starrydata Bi2Te3/Sb2Te3 ZT+temperature+angle rows; no generic thermoelectric pages.",
        "Fusion T26-T30: exact structured measurement attachments only; metadata/schema/search/PDF rejected.",
        "T57/T59: exact HEPData record/table/column manifests only.",
        "T45/T47: exact benchmark tables only; generic hardware pages do not score.",
        "T50-T52: bound-only forever; T60a anchor-only until T60b/T60c/T60d pass.",
    ]


    # v52 dashboard: safe confirmation claims + explicit confirm target ranking.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v52"
    v52_status = {
        "confirmed_now": [],
        "strict_confirm_allowed_now": [],
        "compatible_positive_now": [],
        "audit_conflicts_demoted": [],
        "near_confirm_next": [],
        "measured_microstructure_priority": [],
        "positive_anchor_only": [],
        "bound_only": [],
        "data_limited_positive_paths": [],
        "source_contracts_needed": [],
        "missing_output_or_timeout": [],
    }
    confirm_targets_v52 = []
    status_split_counts_v52 = {"execution": {}, "data": {}, "evidence": {}, "confirmation": {}}
    def _inc(bucket, key):
        key = str(key or "unknown")
        status_split_counts_v52[bucket][key] = status_split_counts_v52[bucket].get(key, 0) + 1
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        if not tid:
            continue
        # Prefer v52 fragments if present in result JSON; otherwise this is an older fragment and will be treated conservatively.
        result_file = args.outdir / f"{str(tid).lower()}_result.json"
        latest = frag
        if result_file.exists():
            try:
                r = json.loads(result_file.read_text(encoding="utf-8"))
                latest = r.get("positive_dashboard_fragment_v52") or r.get("positive_dashboard_fragment_v51") or frag
            except Exception:
                latest = frag
        split = latest.get("status_split_v52") or {}
        _inc("execution", split.get("execution_status_v52"))
        _inc("data", split.get("data_status_v52"))
        _inc("evidence", split.get("evidence_status_v52"))
        _inc("confirmation", split.get("confirmation_status_v52"))
        conflicts = latest.get("confirmation_conflicts_v52") or []
        strict = bool(latest.get("strict_confirm_allowed_now"))
        label = str(latest.get("confirmation_label") or "")
        verdict = str(latest.get("verdict") or "")
        score = int(((latest.get("near_confirm_score") or {}).get("score_0_10") or 0))
        if latest.get("confirm_target_v52"):
            confirm_targets_v52.append(latest["confirm_target_v52"])
        if conflicts:
            v52_status["audit_conflicts_demoted"].append({"test_id": tid, "conflicts": conflicts})
        elif strict and tid == "T48":
            v52_status["confirmed_now"].append("T48b")
            v52_status["strict_confirm_allowed_now"].append("T48b")
            v52_status["compatible_positive_now"].append("T48b")
        elif strict:
            v52_status["confirmed_now"].append(tid)
            v52_status["strict_confirm_allowed_now"].append(tid)
        elif tid == "T60" or label == "positive_anchor_only":
            v52_status["positive_anchor_only"].append("T60a")
        elif tid in {"T50", "T51", "T52"}:
            v52_status["bound_only"].append(tid)
        elif "missing_output" in verdict or "fallback" in verdict or "timeout" in verdict:
            v52_status["missing_output_or_timeout"].append(tid)
        elif tid in {"T31", "T32"}:
            v52_status["measured_microstructure_priority"].append(tid)
            v52_status["near_confirm_next"].append(tid)
        elif tid in {"T53", "T34"} or score >= 6:
            v52_status["near_confirm_next"].append(tid)
        elif tid in {"T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59"}:
            v52_status["data_limited_positive_paths"].append(tid)
            v52_status["source_contracts_needed"].append(tid)
    order = ["T48b", "T44", "T31", "T32", "T53", "T34", "T60a", "T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59", "T50", "T51", "T52"]
    for k, vals in list(v52_status.items()):
        if not vals:
            continue
        if isinstance(vals[0], dict):
            v52_status[k] = sorted(vals, key=lambda d: order.index(d.get("test_id")) if d.get("test_id") in order else 99)
        else:
            v52_status[k] = sorted(set(vals), key=lambda x: order.index(x) if x in order else 99)
    confirm_targets_v52 = sorted(confirm_targets_v52, key=lambda d: (-int(d.get("rank_score_0_10_v52") or 0), order.index(d.get("test_id")) if d.get("test_id") in order else 99))
    dashboard["v52_confirm_status"] = v52_status
    dashboard["status_split_counts_v52"] = status_split_counts_v52
    dashboard["confirm_targets_v52"] = confirm_targets_v52
    (args.outdir / "confirm_targets_v52.json").write_text(json.dumps({"schema":"ccdr-tierb-confirm-targets-v52", "targets": confirm_targets_v52}, indent=2, sort_keys=True), encoding="utf-8")
    dashboard["recommended_next_v52"] = [
        "Use only v52_confirm_status.confirmed_now for public confirm claims; v52 demotes audit conflicts such as T44 zero true Tier-A rows.",
        "T48b remains the strongest compatible-positive, but publication-grade status requires descriptor model + absorber-family/source/year jackknife + permutation null artifacts.",
        "T31/T32 remain the highest physical near-confirm path: measured κ(T)+microstructure rows only, with grouped bootstrap and material/source/temperature jackknives.",
        "T53 needs ProteinGym -> UniProt/PDB/AlphaFold joined rows and family/assay/sequence jackknives before confirm language.",
        "T34/T45/T47/T57/T59 should run only from exact manifests; broad discovery is diagnostic only.",
        "Fusion T26-T30 should report source-contract failures by missing required column group until exact measurement attachments exist.",
        "T50-T52 are bound-only; T60a is an anchor only until quark/lattice, sector-reshuffle, and look-elsewhere gates pass.",
    ]



    # v53 dashboard: claim-safe confirm list after row-recovery/model-gate improvements.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v53"
    v53_status = {
        "confirmed_now": [],
        "strict_confirm_allowed_now": [],
        "compatible_positive_now": [],
        "audit_conflicts_demoted": [],
        "near_confirm_next": [],
        "measured_microstructure_priority": [],
        "positive_anchor_only": [],
        "bound_only": [],
        "data_limited_positive_paths": [],
        "source_contracts_needed": [],
        "publication_grade_pending": [],
        "missing_output_or_timeout": [],
    }
    confirm_targets_v53 = []
    status_split_counts_v53 = {"execution": {}, "data": {}, "evidence": {}, "confirmation": {}}
    def _inc53(bucket, key):
        key = str(key or "unknown")
        status_split_counts_v53[bucket][key] = status_split_counts_v53[bucket].get(key, 0) + 1
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        if not tid:
            continue
        result_file = args.outdir / f"{str(tid).lower()}_result.json"
        latest = frag
        if result_file.exists():
            try:
                r = json.loads(result_file.read_text(encoding="utf-8"))
                latest = r.get("positive_dashboard_fragment_v53") or r.get("positive_dashboard_fragment_v52") or r.get("positive_dashboard_fragment_v51") or frag
            except Exception:
                latest = frag
        split = latest.get("status_split_v53") or latest.get("status_split_v52") or {}
        _inc53("execution", split.get("execution_status_v53") or split.get("execution_status_v52"))
        _inc53("data", split.get("data_status_v53") or split.get("data_status_v52"))
        _inc53("evidence", split.get("evidence_status_v53") or split.get("evidence_status_v52"))
        _inc53("confirmation", split.get("confirmation_status_v53") or split.get("confirmation_status_v52"))
        conflicts = latest.get("confirmation_conflicts_v53") or latest.get("confirmation_conflicts_v52") or []
        strict = bool(latest.get("strict_confirm_allowed_now"))
        label = str(latest.get("confirmation_label") or "")
        verdict = str(latest.get("verdict") or "")
        score = int(((latest.get("near_confirm_score") or {}).get("score_0_10") or 0))
        target = latest.get("confirm_target_v53") or latest.get("confirm_target_v52")
        if target:
            confirm_targets_v53.append(target)
        if conflicts:
            v53_status["audit_conflicts_demoted"].append({"test_id": tid, "conflicts": conflicts})
        elif strict and tid == "T48":
            v53_status["confirmed_now"].append("T48b")
            v53_status["strict_confirm_allowed_now"].append("T48b")
            v53_status["compatible_positive_now"].append("T48b")
            # publication-grade may still be a robustness target even though compatible-positive is preserved.
            try:
                r = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
                art = (((r.get("auto_data_improvements_v53") or {}).get("t48b_pv_row_recovery_publication_v53") or {}).get("model") or {})
                if not art.get("publication_grade_ready_v53"):
                    v53_status["publication_grade_pending"].append("T48b")
            except Exception:
                pass
        elif strict:
            v53_status["confirmed_now"].append(tid)
            v53_status["strict_confirm_allowed_now"].append(tid)
        elif tid == "T60" or label == "positive_anchor_only":
            v53_status["positive_anchor_only"].append("T60a")
        elif tid in {"T50", "T51", "T52"}:
            v53_status["bound_only"].append(tid)
        elif "missing_output" in verdict or "fallback" in verdict or "timeout" in verdict:
            v53_status["missing_output_or_timeout"].append(tid)
        elif tid in {"T31", "T32"}:
            v53_status["measured_microstructure_priority"].append(tid)
            v53_status["near_confirm_next"].append(tid)
        elif tid in {"T53", "T34"} or score >= 6:
            v53_status["near_confirm_next"].append(tid)
        elif tid in {"T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59"}:
            v53_status["data_limited_positive_paths"].append(tid)
            v53_status["source_contracts_needed"].append(tid)
    order = ["T48b", "T44", "T31", "T32", "T53", "T34", "T60a", "T26", "T27", "T28", "T29", "T30", "T45", "T47", "T57", "T59", "T50", "T51", "T52"]
    for k, vals in list(v53_status.items()):
        if not vals:
            continue
        if isinstance(vals[0], dict):
            v53_status[k] = sorted(vals, key=lambda d: order.index(d.get("test_id")) if d.get("test_id") in order else 99)
        else:
            v53_status[k] = sorted(set(vals), key=lambda x: order.index(x) if x in order else 99)
    def _target_score_v53(d):
        return int(d.get("rank_score_0_10_v53") if d.get("rank_score_0_10_v53") is not None else d.get("rank_score_0_10_v52") or 0)
    confirm_targets_v53 = sorted(confirm_targets_v53, key=lambda d: (-_target_score_v53(d), order.index(d.get("test_id")) if d.get("test_id") in order else 99))
    dashboard["v53_confirm_status"] = v53_status
    dashboard["status_split_counts_v53"] = status_split_counts_v53
    dashboard["confirm_targets_v53"] = confirm_targets_v53
    (args.outdir / "confirm_targets_v53.json").write_text(json.dumps({"schema":"ccdr-tierb-confirm-targets-v53", "targets": confirm_targets_v53}, indent=2, sort_keys=True), encoding="utf-8")
    dashboard["recommended_next_v53"] = [
        "Use only v53_confirm_status.confirmed_now for public confirm claims; legacy v51/v52 confirm fields are audit inputs, not claim gates.",
        "T48b compatible-positive is preserved; publication-grade layer now separately recovers PV descriptor rows and audits absorber-family/source/year robustness.",
        "T44 stays demoted until true Tier-A NAND rows include company/year/layers/capacity/die_area/bits_per_cell/source_url; derived die-area rows are audit-only.",
        "T31/T32 confirmation now requires dedup measured microstructure rows, >=5 source groups, >=5 material families, >=3 temperature bins, and temperature-bin residual gates.",
        "T53 confirmation now requires ProteinGym outcome joined to UniProt/PDB/AlphaFold features plus family/assay/sequence jackknives.",
        "T34 requires exact Bi2Te3/Sb2Te3 ZT+temperature+orientation/grain-angle export rows; generic thermoelectric pages remain rejected.",
        "T57/T59 require exact HEPData record/table/column manifests; broad HEP search URLs are discovery-only.",
        "Fusion T26-T30 now writes missing required column-group diagnostics until exact public measurement attachments exist.",
        "T50-T52 remain bound-only; T60a remains anchor-only until quark/lattice, sector-reshuffle, and look-elsewhere gates pass."
    ]


    # v54 dashboard: targeted fusion source triage from expected-source manifest.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v54"
    v54_status = {
        "confirmed_now": list((dashboard.get("v53_confirm_status") or {}).get("confirmed_now") or []),
        "fusion_preliminary_public_anchor": [],
        "fusion_suggestive_only": [],
        "fusion_blocked_figures_only": [],
        "fusion_summary_anchor_rows_blocked": [],
        "fusion_exact_attachment_still_required": [],
        "source_contracts_needed": list((dashboard.get("v53_confirm_status") or {}).get("source_contracts_needed") or []),
    }
    confirm_targets_v54 = []
    status_split_counts_v54 = {"execution": {}, "data": {}, "evidence": {}, "confirmation": {}}
    def _inc54(bucket, key):
        key = str(key or "unknown")
        status_split_counts_v54[bucket][key] = status_split_counts_v54[bucket].get(key, 0) + 1
    fusion_order = ["T29", "T28", "T27", "T26", "T30"]
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        if not tid:
            continue
        result_file = args.outdir / f"{str(tid).lower()}_result.json"
        latest = frag
        if result_file.exists():
            try:
                r = json.loads(result_file.read_text(encoding="utf-8"))
                latest = r.get("positive_dashboard_fragment_v54") or r.get("positive_dashboard_fragment_v53") or r.get("positive_dashboard_fragment_v52") or frag
            except Exception:
                latest = frag
        split = latest.get("status_split_v54") or latest.get("status_split_v53") or latest.get("status_split_v52") or {}
        _inc54("execution", split.get("execution_status_v54") or split.get("execution_status_v53") or split.get("execution_status_v52"))
        _inc54("data", split.get("data_status_v54") or split.get("data_status_v53") or split.get("data_status_v52"))
        _inc54("evidence", split.get("evidence_status_v54") or split.get("evidence_status_v53") or split.get("evidence_status_v52"))
        _inc54("confirmation", split.get("confirmation_status_v54") or split.get("confirmation_status_v53") or split.get("confirmation_status_v52"))
        target = latest.get("confirm_target_v54") or latest.get("confirm_target_v53") or latest.get("confirm_target_v52")
        if target:
            confirm_targets_v54.append(target)
        if tid in {"T26", "T27", "T28", "T29", "T30"}:
            v54_status["fusion_exact_attachment_still_required"].append(tid)
            ev = str(split.get("evidence_status_v54") or "")
            if tid == "T29" or "preliminary_public_anchor" in ev:
                v54_status["fusion_preliminary_public_anchor"].append(tid)
            elif tid == "T28" or "summary_anchor" in ev:
                v54_status["fusion_summary_anchor_rows_blocked"].append(tid)
            elif tid == "T27" or "suggestive" in ev:
                v54_status["fusion_suggestive_only"].append(tid)
            elif tid == "T26" or "figure" in ev:
                v54_status["fusion_blocked_figures_only"].append(tid)
    def _sort54(vals):
        return sorted(set(vals), key=lambda x: fusion_order.index(x) if x in fusion_order else 99)
    for k, vals in list(v54_status.items()):
        if isinstance(vals, list) and (not vals or isinstance(vals[0], str)):
            v54_status[k] = _sort54(vals)
    def _target_score_v54(d):
        return int(d.get("rank_score_0_10_v54") if d.get("rank_score_0_10_v54") is not None else d.get("rank_score_0_10_v53") if d.get("rank_score_0_10_v53") is not None else d.get("rank_score_0_10_v52") or 0)
    full_order = ["T48", "T48b", "T44", "T31", "T32", "T53", "T34", "T29", "T28", "T27", "T26", "T30", "T45", "T47", "T57", "T59", "T60", "T60a", "T50", "T51", "T52"]
    confirm_targets_v54 = sorted(confirm_targets_v54, key=lambda d: (-_target_score_v54(d), full_order.index(d.get("test_id")) if d.get("test_id") in full_order else 99))
    dashboard["v54_confirm_status"] = v54_status
    dashboard["status_split_counts_v54"] = status_split_counts_v54
    dashboard["confirm_targets_v54"] = confirm_targets_v54
    (args.outdir / "confirm_targets_v54.json").write_text(json.dumps({"schema":"ccdr-tierb-confirm-targets-v54", "targets": confirm_targets_v54}, indent=2, sort_keys=True), encoding="utf-8")
    dashboard["recommended_next_v54"] = [
        "Fusion T29 is now the strongest preliminary public fusion path: extract Stroth 2021 W7-X/AUG/W7-AS comparison tables, but keep it preliminary until raw/structured rows pass controls.",
        "Fusion T28 has the strongest public summary anchor through Verdoolaege 2021 DB5.2.3-STD5; use summary/regression tables as ingredient support only unless exact row files are downloaded.",
        "Fusion T27 can use Paz-Soldan 2024 as suggestive RMP-ELM compilation; no raw per-discharge table means no confirm language.",
        "Fusion T26 remains blocked for rigorous confirmation; digitized figures from Loarte/ITPA/JET/AUG/MAST papers are partial trend diagnostics only.",
        "Fusion T30 stays low-priority and should inherit exact rows from T28/T29 if those become available.",
        "Public confirm claims still use only v54_confirm_status.confirmed_now; fusion paper anchors are not confirmations.",
    ]

    # v55 dashboard overlay: public-source fusion parsers (PDF/text table extraction)
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v55"
    v55_status = {
        "confirmed_now": list((dashboard.get("v54_confirm_status") or {}).get("confirmed_now") or []),
        "fusion_preliminary_public_parser_ready": [],
        "fusion_parser_rows_extracted_nonconfirm": [],
        "fusion_summary_or_suggestive_only": [],
        "fusion_raw_rows_still_required": [],
        "source_contracts_needed": list((dashboard.get("v54_confirm_status") or {}).get("source_contracts_needed") or []),
    }
    confirm_targets_v55 = []
    status_split_counts_v55 = {"execution": {}, "data": {}, "evidence": {}, "confirmation": {}}
    def _inc55(bucket, key):
        key = str(key or "unknown")
        status_split_counts_v55[bucket][key] = status_split_counts_v55[bucket].get(key, 0) + 1
    fusion_order55 = ["T29", "T28", "T27", "T26", "T30"]
    new_tests = []
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        latest = frag
        if tid:
            result_file = args.outdir / f"{str(tid).lower()}_result.json"
            if result_file.exists():
                try:
                    r = json.loads(result_file.read_text(encoding="utf-8"))
                    latest = r.get("positive_dashboard_fragment_v55") or r.get("positive_dashboard_fragment_v54") or r.get("positive_dashboard_fragment_v53") or r.get("positive_dashboard_fragment_v52") or frag
                except Exception:
                    latest = frag
        new_tests.append(latest)
        split = latest.get("status_split_v55") or latest.get("status_split_v54") or latest.get("status_split_v53") or latest.get("status_split_v52") or {}
        _inc55("execution", split.get("execution_status_v55") or split.get("execution_status_v54") or split.get("execution_status_v53") or split.get("execution_status_v52"))
        _inc55("data", split.get("data_status_v55") or split.get("data_status_v54") or split.get("data_status_v53") or split.get("data_status_v52"))
        _inc55("evidence", split.get("evidence_status_v55") or split.get("evidence_status_v54") or split.get("evidence_status_v53") or split.get("evidence_status_v52"))
        _inc55("confirmation", split.get("confirmation_status_v55") or split.get("confirmation_status_v54") or split.get("confirmation_status_v53") or split.get("confirmation_status_v52"))
        target = latest.get("confirm_target_v55") or latest.get("confirm_target_v54") or latest.get("confirm_target_v53") or latest.get("confirm_target_v52")
        if target:
            confirm_targets_v55.append(target)
        if tid in {"T26", "T27", "T28", "T29", "T30"}:
            v55_status["fusion_raw_rows_still_required"].append(tid)
            parser = (((latest.get("v55") or {}).get("fusion_public_source_parser_v55")) or {})
            if parser.get("preliminary_public_test_ready_v55"):
                v55_status["fusion_preliminary_public_parser_ready"].append(tid)
            elif parser.get("n_rows_v55") or parser.get("n_normalized_rows_v55"):
                v55_status["fusion_parser_rows_extracted_nonconfirm"].append(tid)
            ev = str(split.get("evidence_status_v55") or "")
            if "summary" in ev or "suggestive" in ev or "figure" in ev:
                v55_status["fusion_summary_or_suggestive_only"].append(tid)
    dashboard["tests"] = new_tests
    def _sort55(vals):
        return sorted(set(vals), key=lambda x: fusion_order55.index(x) if x in fusion_order55 else 99)
    for k, vals in list(v55_status.items()):
        if isinstance(vals, list) and (not vals or isinstance(vals[0], str)):
            v55_status[k] = _sort55(vals)
    def _target_score_v55(d):
        for key in ["rank_score_0_10_v55", "rank_score_0_10_v54", "rank_score_0_10_v53", "rank_score_0_10_v52"]:
            if d.get(key) is not None:
                try: return int(d.get(key) or 0)
                except Exception: return 0
        return 0
    full_order55 = ["T48", "T48b", "T44", "T31", "T32", "T53", "T34", "T29", "T28", "T27", "T26", "T30", "T45", "T47", "T57", "T59", "T60", "T60a", "T50", "T51", "T52"]
    confirm_targets_v55 = sorted(confirm_targets_v55, key=lambda d: (-_target_score_v55(d), full_order55.index(d.get("test_id")) if d.get("test_id") in full_order55 else 99))
    dashboard["v55_confirm_status"] = v55_status
    dashboard["status_split_counts_v55"] = status_split_counts_v55
    dashboard["confirm_targets_v55"] = confirm_targets_v55
    (args.outdir / "confirm_targets_v55.json").write_text(json.dumps({"schema":"ccdr-tierb-confirm-targets-v55", "targets": confirm_targets_v55}, indent=2, sort_keys=True), encoding="utf-8")
    dashboard["recommended_next_v55"] = [
        "Fusion T29 now has a real public-source parser for Stroth 2021 PDF text/tables; use it for a preliminary structured-public test only, not strict confirmation.",
        "Fusion T28 now parses Verdoolaege/DB5.2.3 summary and regression rows plus probes OSF for structured attachments; full DB5 per-timeslice rows are still required for confirmation.",
        "Fusion T27 now parses Paz-Soldan RMP-ELM text/table summaries; treat extracted rows as suggestive only until raw per-discharge rows exist.",
        "Fusion T26 now parses ELM-loss/pedestal summary or figure-context rows; treat as partial trend support only until public per-shot ELM/pedestal rows exist.",
        "Fusion T30 remains a derived secondary diagnostic depending on future exact T28/T29 row releases.",
        "Public confirm claims still use only v55_confirm_status.confirmed_now; parsed PDF rows do not override the strict raw-row gate.",
    ]

    # v56 dashboard overlay: missing-output repair + confirm-target hardening.
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v56"
    v56_status = {
        "confirmed_now": [],
        "near_confirm_routes": [],
        "runtime_repaired": [],
        "bound_only": [],
        "anchor_only": [],
        "fusion_preliminary_only": [],
        "source_contracts_needed": [],
    }
    confirm_targets_v56 = []
    status_split_counts_v56 = {"execution": {}, "data": {}, "evidence": {}, "confirmation": {}}
    def _inc56(bucket, key):
        key = str(key or "unknown")
        status_split_counts_v56[bucket][key] = status_split_counts_v56[bucket].get(key, 0) + 1
    new_tests_v56 = []
    for frag in dashboard.get("tests", []):
        tid = frag.get("test_id")
        latest = frag
        if tid:
            rf = args.outdir / f"{str(tid).lower()}_result.json"
            if rf.exists():
                try:
                    rr = json.loads(rf.read_text(encoding="utf-8"))
                    latest = rr.get("positive_dashboard_fragment_v56") or rr.get("positive_dashboard_fragment_v55") or rr.get("positive_dashboard_fragment_v54") or rr.get("positive_dashboard_fragment_v53") or rr.get("positive_dashboard_fragment_v52") or frag
                except Exception:
                    latest = frag
        new_tests_v56.append(latest)
        split = latest.get("status_split_v56") or latest.get("status_split_v55") or latest.get("status_split_v54") or latest.get("status_split_v53") or latest.get("status_split_v52") or {}
        _inc56("execution", split.get("execution_status_v56") or split.get("execution_status_v55") or split.get("execution_status_v54") or split.get("execution_status_v53") or split.get("execution_status_v52"))
        _inc56("data", split.get("data_status_v56") or split.get("data_status_v55") or split.get("data_status_v54") or split.get("data_status_v53") or split.get("data_status_v52"))
        _inc56("evidence", split.get("evidence_status_v56") or split.get("evidence_status_v55") or split.get("evidence_status_v54") or split.get("evidence_status_v53") or split.get("evidence_status_v52"))
        _inc56("confirmation", split.get("confirmation_status_v56") or split.get("confirmation_status_v55") or split.get("confirmation_status_v54") or split.get("confirmation_status_v53") or split.get("confirmation_status_v52"))
        target = latest.get("confirm_target_v56") or latest.get("confirm_target_v55") or latest.get("confirm_target_v54") or latest.get("confirm_target_v53") or latest.get("confirm_target_v52")
        if target:
            confirm_targets_v56.append(target)
        label = str(latest.get("confirmation_label") or "")
        if latest.get("confirm_allowed_now") and "positive" in label:
            v56_status["confirmed_now"].append(tid)
        conf = str(split.get("confirmation_status_v56") or split.get("confirmation_status_v55") or "")
        ev = str(split.get("evidence_status_v56") or split.get("evidence_status_v55") or "")
        if tid in {"T31", "T32", "T44", "T53", "T34", "T29"} and not latest.get("confirm_allowed_now"):
            v56_status["near_confirm_routes"].append(tid)
        if "missing_output_repaired" in str(split.get("execution_status_v56")) or ((latest.get("v56") or {}).get("fallback_repaired_v56")):
            v56_status["runtime_repaired"].append(tid)
        if tid in {"T50", "T51", "T52"}:
            v56_status["bound_only"].append(tid)
        if tid == "T60":
            v56_status["anchor_only"].append(tid)
        if tid in {"T26", "T27", "T28", "T29", "T30"} and ("preliminary" in conf or "summary" in ev or "suggestive" in ev):
            v56_status["fusion_preliminary_only"].append(tid)
        if not latest.get("confirm_allowed_now") and target:
            v56_status["source_contracts_needed"].append(tid)
    dashboard["tests"] = new_tests_v56
    order56 = ["T48", "T31", "T32", "T53", "T44", "T29", "T34", "T28", "T27", "T26", "T30", "T45", "T47", "T57", "T59", "T60", "T50", "T51", "T52"]
    def _sort56(vals):
        return sorted(set(x for x in vals if x), key=lambda x: order56.index(x) if x in order56 else 99)
    for k, vals in list(v56_status.items()):
        if isinstance(vals, list):
            v56_status[k] = _sort56(vals)
    def _score56(d):
        for key in ["rank_score_0_10_v56", "rank_score_0_10_v55", "rank_score_0_10_v54", "rank_score_0_10_v53", "rank_score_0_10_v52"]:
            if isinstance(d, dict) and d.get(key) is not None:
                try: return int(d.get(key) or 0)
                except Exception: return 0
        return 0
    confirm_targets_v56 = sorted(confirm_targets_v56, key=lambda d: (-_score56(d), order56.index(d.get("test_id")) if isinstance(d, dict) and d.get("test_id") in order56 else 99))
    dashboard["v56_confirm_status"] = v56_status
    dashboard["status_split_counts_v56"] = status_split_counts_v56
    dashboard["confirm_targets_v56"] = confirm_targets_v56
    dashboard["recommended_next_v56"] = [
        "Use only v56_confirm_status.confirmed_now for public confirm claims; currently this should preserve T48 unless a later run adds another strict v56-confirmed test.",
        "T31/T32: fix/monitor missing-output and require dedup measured κ(T)+SEM/TEM/XRD rows, >=5 sources, >=5 material families, >=3 temperature bins, sign/bootstrap/jackknife gates.",
        "T53: implement real ProteinGym -> UniProt/PDB/AlphaFold join rows and family/assay/sequence jackknife before confirmation language.",
        "T44: no frozen strict claim; require true Tier-A NAND rows with company/year/layers/capacity/die_area/bits_per_cell/source_url.",
        "T29: improve Stroth table extraction for preliminary evidence only; strict fusion confirmation still requires raw public profile/transport rows.",
        "T50-T52: keep bound-only and never rank as positive confirm targets.",
    ]
    (args.outdir / "confirm_targets_v56.json").write_text(json.dumps({"schema":"ccdr-tierb-confirm-targets-v56", "targets": confirm_targets_v56}, indent=2, sort_keys=True), encoding="utf-8")

    # v57 dashboard overlay: strict public-claim checker + confirm-target repairs.
    try:
        from tierb.v57_confirm_repairs import apply_dashboard_v57
        dashboard = apply_dashboard_v57(dashboard, args.outdir)
    except Exception as _v57_dash_e:
        dashboard.setdefault("v57_dashboard_error", f"{type(_v57_dash_e).__name__}: {_v57_dash_e}")

    # v58 dashboard overlay: confirm-only public claims + exact source contracts.
    try:
        from tierb.v58_confirm_focus import apply_dashboard_v58
        dashboard = apply_dashboard_v58(dashboard, args.outdir)
    except Exception as _v58_dash_e:
        dashboard.setdefault("v58_dashboard_error", f"{type(_v58_dash_e).__name__}: {_v58_dash_e}")

    # v59 dashboard overlay: concrete confirm extractors + stricter public claim checker.
    try:
        from tierb.v59_confirm_extractors import apply_dashboard_v59
        dashboard = apply_dashboard_v59(dashboard, args.outdir)
    except Exception as _v59_dash_e:
        dashboard.setdefault("v59_dashboard_error", f"{type(_v59_dash_e).__name__}: {_v59_dash_e}")

    (args.outdir / "positive_dashboard.json").write_text(json.dumps(dashboard, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
