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
    p.add_argument("--continue-on-error", action="store_true", default=True)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    selected = [t.upper() for t in (args.only or TESTS.keys())]
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
        frag = r.get("positive_dashboard_fragment_v21") or r.get("positive_dashboard_fragment_v20") or {"test_id": tid, "verdict": verdict}
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
    dashboard["schema"] = "ccdr-tierb-positive-dashboard-v24"
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

    (args.outdir / "positive_dashboard.json").write_text(json.dumps(dashboard, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
