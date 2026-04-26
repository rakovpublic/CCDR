#!/usr/bin/env python3
"""Run all CCDR engineering public-data proxy tests.

v2.2 improvement: flatten nested decisions in out_summary.json so combined tests
such as FR3/FR6 no longer show decision=null.

v2.3 improvement: expose EL3a/EL3b split statuses in the summary.
v2.5 improvement: expose EL3 raw/node-median/foundry robustness fractions.
v2.6 improvement: expose EL3 alpha-drop transition flags and FR8 proxy status.
v2.7 improvement: expose uniform evidence ladder plus FR8/EL3 split branches.
v2.8 improvement: expose FR3 full-vs-reduced ladders, FR8 decisive/context levels, and EL3 primary-vs-secondary physical-pitch branches.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

TESTS = [
    [sys.executable, "test_fr3_fr6_literature_proxy.py", "--out", "out_fr3_fr6.json"],
    [sys.executable, "test_fr8_eta_s_collisionality_proxy.py", "--out", "out_fr8.json"],
    [sys.executable, "test_qc5_phononic_crystal_t1_public.py", "--out", "out_qc5.json"],
    [sys.executable, "test_el3_si_area_crossover_public.py", "--include-wikipedia", "--include-foundry-nodes", "--out", "out_el3.json"],
]


def _scalar_decision(value: Any) -> Any:
    """Return compact status-like value for arbitrary nested decision objects."""
    if isinstance(value, dict):
        for key in ("status", "decision", "overall_status"):
            if key in value:
                return value[key]
        return value
    return value


def summarize_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract useful per-prediction status from a test JSON result."""
    out: Dict[str, Any] = {}
    if "test_name" in data:
        out["test_name"] = data.get("test_name")

    # Standard one-test JSON shape.
    if "decision" in data:
        out["decision"] = data.get("decision")
        out["status"] = _scalar_decision(data.get("decision"))
        if isinstance(data.get("decision"), dict) and isinstance(data["decision"].get("evidence_ladder"), dict):
            out["evidence_level"] = data["decision"]["evidence_ladder"].get("level")
            out["evidence_ladder_key"] = data["decision"]["evidence_ladder"].get("ladder_key")
    if isinstance(data.get("evidence_ladder"), dict):
        out["evidence_level"] = data["evidence_ladder"].get("level")
        out["evidence_ladder_key"] = data["evidence_ladder"].get("ladder_key")

    # Combined FR3/FR6 script shape.
    for pred in ("FR3", "FR6"):
        block = data.get(pred)
        if isinstance(block, dict):
            pred_status = block.get("decision") or block.get("status")
            # FR3 also nests strict_machine_readable_result.status.
            if pred_status is None and isinstance(block.get("strict_machine_readable_result"), dict):
                pred_status = block["strict_machine_readable_result"].get("status")
            out[pred] = pred_status
            if pred == "FR3":
                if isinstance(block.get("FR3_full_evidence_ladder"), dict):
                    out["FR3_full_evidence_level"] = block["FR3_full_evidence_ladder"].get("level")
                    out["FR3_full_ladder_key"] = block["FR3_full_evidence_ladder"].get("ladder_key")
                if isinstance(block.get("FR3b_reduced_proxy_evidence_ladder"), dict):
                    out["FR3b_reduced_proxy_evidence_level"] = block["FR3b_reduced_proxy_evidence_ladder"].get("level")
                    out["FR3b_reduced_proxy_ladder_key"] = block["FR3b_reduced_proxy_evidence_ladder"].get("ladder_key")

    # QC5 v2.2 sub-decisions.
    if isinstance(data.get("decision"), dict):
        dec = data["decision"]
        for key in ("QC5a_TLS_mechanism", "QC5b_transmon_T1"):
            if isinstance(dec.get(key), dict):
                out[key] = dec[key].get("status")
        # EL3 v2.5 split decisions.
        for key in ("EL3a_historical_chip_density_raw", "EL3a_historical_chip_density_node_median", "EL3a_historical_chip_density", "EL3b_foundry_standard_cell_density", "EL3c_primary_physical_pitch_density_proxy", "EL3c_secondary_physical_pitch_density_proxy", "EL3c_physical_pitch_density_proxy"):
            if key in dec:
                out[key] = dec.get(key)
    for key in ("EL3a_historical_chip_density_raw", "EL3a_historical_chip_density_node_median", "EL3a_historical_chip_density", "EL3b_foundry_standard_cell_density", "EL3c_primary_physical_pitch_density_proxy", "EL3c_secondary_physical_pitch_density_proxy", "EL3c_physical_pitch_density_proxy"):
        block = data.get(key)
        if isinstance(block, dict) and isinstance(block.get("decision"), dict):
            out[key] = block["decision"].get("status")
            if "subsample_supportive_fraction" in block["decision"]:
                out[f"{key}_subsample_supportive_fraction"] = block["decision"].get("subsample_supportive_fraction")
            flags = block["decision"].get("hypothesis_fit_flags") or {}
            if isinstance(flags, dict) and "alpha_drop_large_minus_small" in flags:
                out[f"{key}_hypothesis_alpha_drop"] = flags.get("alpha_drop_large_minus_small")
                out[f"{key}_hypothesis_crossover_like"] = flags.get("crossover_like_alpha_drop")

    if isinstance(data.get("fusion_edge_proxy"), dict):
        out["fusion_edge_proxy_status"] = data["fusion_edge_proxy"].get("status")
    for key in ("FR8a_QGP_eta_over_s_context", "FR8b_fusion_transport_proxy", "FR8c_true_edge_eta_or_MKSS", "EL3c_primary_physical_pitch_density_proxy", "EL3c_secondary_physical_pitch_density_proxy", "EL3c_physical_pitch_density_proxy"):
        block = data.get(key)
        if isinstance(block, dict):
            dec = block.get("decision") if isinstance(block.get("decision"), dict) else block
            out[key] = dec.get("status")
            ladder = dec.get("evidence_ladder")
            if isinstance(ladder, dict):
                out[f"{key}_evidence_level"] = ladder.get("level")
    for key in ("FR8_decisive_evidence_level", "FR8_context_evidence_level"):
        if key in data:
            out[key] = data.get(key)
    if isinstance(data.get("FR8_split_evidence_summary"), dict):
        out["FR8_split_evidence_levels"] = data["FR8_split_evidence_summary"].get("FR8_split_evidence_levels")
    if isinstance(data.get("confidence_filtered_decisions"), dict):
        for key, value in data["confidence_filtered_decisions"].items():
            if isinstance(value, dict):
                out[f"{key}_status"] = value.get("status")
                ladder = value.get("evidence_ladder")
                if isinstance(ladder, dict):
                    out[f"{key}_evidence_level"] = ladder.get("level")
    if isinstance(data.get("knolker_table2_extraction"), dict):
        out["knolker_table2_extraction_method"] = data["knolker_table2_extraction"].get("extraction_method")

    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    summary = []
    for cmd in TESTS:
        print("\n=== Running", " ".join(cmd[1:]), "===", flush=True)
        proc = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        item: Dict[str, Any] = {"script": cmd[1], "returncode": proc.returncode}
        out_arg = cmd[cmd.index("--out") + 1] if "--out" in cmd else None
        if out_arg and (root / out_arg).exists():
            try:
                data = json.loads((root / out_arg).read_text(encoding="utf-8"))
                item.update(summarize_json(data))
            except Exception as exc:  # noqa: BLE001 - summary should never mask test output
                item["json_error"] = str(exc)
        summary.append(item)
    (root / "out_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
