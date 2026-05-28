#!/usr/bin/env python3
"""v64 exact-data-pack behavioral confirm extractors for CCDR Tier-B.

This patch builds on v63 and is intentionally not a dashboard-only layer.
It changes how the near-confirm tests ingest and compute on evidence:

* material tests use exact source packs, source-pack summaries, small curated
  set mode, stronger family/source/temperature balancing, and explicit
  missing-schema diagnostics;
* NAND, ProteinGym, thermoelectric, HEPData, optical, and neuromorphic tests
  now use concrete exact-source pack locations and parsers that count only
  filled physical rows, never generated dashboards or metadata;
* ProteinGym uses a two-stage assay→UniProt→structure join cache;
* all exact-source pack templates are created automatically, but template rows
  are marked non-evidence and never count as confirmation.
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

from tierb import v63_behavioral_confirm_extractors as v63
from tierb import v62_behavioral_confirm_extractors as v62
from tierb import v61_behavioral_confirm_extractors as v61
from tierb.tierb_catalog import TESTS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GEN_DIR = DATA_DIR / "generated"
MANIFEST_DIR = DATA_DIR / "manifests"

ALL_TIERB_TESTS = [f"T{i}" for i in range(26, 61)]
DEFAULT_TESTS = list(ALL_TIERB_TESTS)
NEAR = set(v63.NEAR)
FUSION = set(v63.FUSION)
BOUND = set(v63.BOUND)
ANCHOR = set(v63.ANCHOR)
CONFIRMED_PUBLIC = {"T48"}
SYNTHETIC_OR_ENGINEERING = {"T46"}

EXACT_PACKS: Dict[str, Dict[str, Any]] = {
    "materials": {
        "dirs": ["data/exact_sources/materials", "data/materials_sources", "data/external/materials", "data/downloaded_supplements/materials", "cache/materials_exact"],
        "template": "t31_t32_materials_exact_template.csv",
        "columns": ["source_url", "source_label", "sample_id", "material", "material_family", "temperature_K", "kappa_W_mK", "grain_size_nm", "microstructure_method", "nanocrystalline_yes_no", "boundary_density_proxy", "measurement_method", "notes"],
        "description": "Exact measured κ(T)+microstructure rows. Generated diagnostics are never evidence.",
    },
    "materials_family_packs": {
        "dirs": ["data/exact_sources/materials/families"],
        "template": "family_pack_template.csv",
        "columns": ["family_name", "source_url", "sample_id", "material", "temperature_K", "kappa_W_mK", "grain_size_nm", "microstructure_method"],
        "description": "Optional per-family packs: silicon/semiconductor, oxide/ceramic, carbon, metal/alloy, thermoelectric.",
    },
    "nand": {
        "dirs": ["data/exact_sources/nand"],
        "template": "t44_nand_tier_a_template.csv",
        "columns": ["company", "year", "layers", "capacity_Gb", "die_area_mm2", "bits_per_cell", "source_url", "product_or_paper", "notes"],
        "description": "True Tier-A NAND rows only; inferred die area is rejected.",
    },
    "proteingym": {
        "dirs": ["data/exact_sources/proteingym"],
        "template": "proteingym_assays_template.csv",
        "columns": ["assay_id", "uniprot", "protein_name", "family", "assay_type", "sequence_cluster", "variant", "dms_score", "fitness_residual", "source_url"],
        "description": "ProteinGym/DMS assay rows; joined to structure maps by UniProt.",
    },
    "protein_structures": {
        "dirs": ["data/exact_sources/protein_structures"],
        "template": "protein_structure_features_template.csv",
        "columns": ["uniprot", "pdb_id", "alphafold_id", "oligomeric_state", "symmetry_proxy", "contact_network_proxy", "fold_class", "source_url"],
        "description": "UniProt→PDB/AlphaFold structure feature rows.",
    },
    "thermoelectric": {
        "dirs": ["data/exact_sources/thermoelectric"],
        "template": "t34_thermoelectric_angle_template.csv",
        "columns": ["material", "composition", "ZT", "temperature_K", "orientation_angle_deg", "grain_boundary_angle_deg", "source_url", "source_label"],
        "description": "Bi2Te3/Sb2Te3 ZT+angle rows for cos(6θ) model.",
    },
    "hepdata": {
        "dirs": ["data/exact_sources/hepdata"],
        "template": "t57_t59_hepdata_manifest_template.csv",
        "columns": ["record_id", "table_id", "x_column", "observed_column", "model_column", "uncertainty_column", "observable_name", "local_table", "source_url"],
        "description": "Frozen exact HEPData manifest plus local CSV/YAML table paths.",
    },
    "optical_interconnect": {
        "dirs": ["data/exact_sources/optical_interconnect"],
        "template": "t45_optical_interconnect_template.csv",
        "columns": ["platform", "year", "energy_per_bit_pJ", "bandwidth_Gbps", "reach_m", "source_url", "benchmark"],
        "description": "Exact optical interconnect benchmark rows.",
    },
    "neuromorphic": {
        "dirs": ["data/exact_sources/neuromorphic"],
        "template": "t47_neuromorphic_benchmark_template.csv",
        "columns": ["chip", "benchmark", "energy_per_inference_or_spike_pJ", "accuracy", "topology", "year", "source_url"],
        "description": "Exact neuromorphic benchmark rows.",
    },
    "fusion": {
        "dirs": ["data/exact_sources/fusion"],
        "template": "fusion_exact_rows_template.csv",
        "columns": ["test_id", "certified_raw_row", "device", "shot", "time_or_slice", "quantity", "value", "unit", "source_url"],
        "description": "Fusion exact physical row tables only; PDF summaries do not count.",
    },
    "ldpc_external_benchmark": {
        "dirs": ["data/exact_sources/ldpc_external_benchmark"],
        "template": "t46_external_public_benchmark_template.csv",
        "columns": ["task_id", "benchmark", "metric_name", "model_score", "baseline_score", "uncertainty", "heldout_split", "source_url", "source_label", "external_public_yes_no", "notes"],
        "description": "External public LDPC/burst-channel benchmark rows for T46; synthetic engineering rows do not count.",
    },
}

PACK_TESTS_V64: Dict[str, List[str]] = {
    "materials": ["T31", "T32"],
    "materials_family_packs": ["T31", "T32"],
    "nand": ["T44"],
    "proteingym": ["T53"],
    "protein_structures": ["T53"],
    "thermoelectric": ["T34"],
    "hepdata": ["T57", "T59"],
    "optical_interconnect": ["T45"],
    "neuromorphic": ["T47"],
    "fusion": ["T26", "T27", "T28", "T29", "T30"],
    "ldpc_external_benchmark": ["T46"],
}

PACK_MINIMUM_GATES_V64: Dict[str, Dict[str, Any]] = {
    "materials": {"min_sources": 5, "min_families": 5, "min_temperature_bins": 3, "min_usable_rows": 50},
    "materials_family_packs": {"min_sources": 5, "min_families": 5, "min_temperature_bins": 3, "min_usable_rows": 50},
    "nand": {"min_companies": 3, "min_usable_rows": 8},
    "proteingym": {"min_families": 3, "min_assays": 3, "min_sequence_clusters": 10, "min_usable_join_rows": 50},
    "protein_structures": {"min_structure_rows": 50},
    "thermoelectric": {"min_usable_rows": 30},
    "hepdata": {"min_records": 3, "min_tables": 3, "min_residual_rows": 20},
    "optical_interconnect": {"min_sources": 3, "min_usable_rows": 20},
    "neuromorphic": {"min_sources": 3, "min_usable_rows": 20},
    "fusion": {"min_exact_rows": 20},
    "ldpc_external_benchmark": {"min_sources": 2, "min_usable_rows": 5, "min_baseline_comparisons": 5},
}

PACK_DEDUP_KEYS_V64: Dict[str, List[str]] = {
    "materials": ["source_url", "sample_id", "material", "temperature_K"],
    "materials_family_packs": ["family_name", "source_url", "sample_id", "temperature_K"],
    "nand": ["company", "year", "layers", "capacity_Gb", "die_area_mm2", "bits_per_cell", "source_url"],
    "proteingym": ["assay_id", "uniprot", "variant"],
    "protein_structures": ["uniprot", "pdb_id", "alphafold_id"],
    "thermoelectric": ["material", "composition", "temperature_K", "orientation_angle_deg", "grain_boundary_angle_deg", "source_url"],
    "hepdata": ["record_id", "table_id", "observed_column", "model_column", "uncertainty_column"],
    "optical_interconnect": ["platform", "year", "benchmark", "source_url"],
    "neuromorphic": ["chip", "benchmark", "year", "source_url"],
    "fusion": ["test_id", "device", "shot", "time_or_slice", "quantity", "source_url"],
    "ldpc_external_benchmark": ["task_id", "benchmark", "metric_name", "heldout_split", "source_url"],
}

T46_EXTERNAL_BENCHMARK_PACK_V64 = "ldpc_external_benchmark"

MATERIAL_FAMILY_TEMPLATES_V64 = [
    "silicon_semiconductor",
    "oxide_ceramic",
    "carbon",
    "metal_alloy",
    "thermoelectric",
]

TEST_REQUIRED_PACKS_V64: Dict[str, List[str]] = defaultdict(list)
for _pack, _tests in PACK_TESTS_V64.items():
    for _tid in _tests:
        if _pack not in TEST_REQUIRED_PACKS_V64[_tid]:
            TEST_REQUIRED_PACKS_V64[_tid].append(_pack)

NEXT_SOURCE_V64: Dict[str, str] = {
    "T26": "Certified fusion per-shot ELM energy rows with pedestal pressure/volume/drop columns.",
    "T27": "Certified fusion RMP/helicity rows with ELM frequency and coil/phasing columns.",
    "T28": "Exact ITPA/H-mode rows with tau_E, density, and transport columns.",
    "T29": "Exact stellarator/tokamak edge transport rows with device and diffusivity/heat-flux columns.",
    "T30": "Exact confinement residual rows with density plus shaping/curvature columns.",
    "T31": "Filled exact materials packs with measured kappa(T), grain size, source URL, and SEM/TEM/XRD/EBSD microstructure method.",
    "T32": "Filled exact materials packs with measured low-temperature kappa(T), grain size, source URL, and microstructure method.",
    "T34": "Exact Bi2Te3/Sb2Te3 ZT plus orientation/grain-boundary angle rows.",
    "T44": "True Tier-A NAND rows: company, year, layers, capacity, die area, bits/cell, source URL.",
    "T45": "Exact optical-interconnect benchmark rows with energy/bit, bandwidth, reach, platform, year, and source URL.",
    "T46": "External public LDPC/burst-channel benchmark rows with task, metric, model score, baseline score, held-out split, and source URL.",
    "T47": "Exact neuromorphic benchmark rows with chip, task, energy, accuracy, topology, year, and source URL.",
    "T48": "Robustness/provenance audit only; current public confirm is already frozen by the v64 gate.",
    "T50": "Bound-table evidence only; this test is not confirmable by design.",
    "T51": "Bound-table evidence only; this test is not confirmable by design.",
    "T52": "Bound-table evidence only; this test is not confirmable by design.",
    "T53": "ProteinGym assay rows joined to UniProt/PDB/AlphaFold structure-feature rows.",
    "T57": "Frozen HEPData manifest plus local tables with observed/model/uncertainty residual columns.",
    "T59": "Frozen HEPData manifest plus local tables with observed/model/uncertainty residual columns.",
    "T60": "Separate charged-lepton/public-constant anchor from any quark/lattice sector claim.",
}


def _ensure(outdir: Optional[Path] = None) -> Path:
    if outdir is not None:
        base = outdir / "data" / "generated"
    else:
        base = GEN_DIR
    base.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    return base


def _write_json(path: Path, obj: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(path)


def _write_csv(rows: Sequence[Dict[str, Any]], filename: str, outdir: Optional[Path] = None) -> str:
    return v61._write_csv(list(rows), filename, outdir)


def _root_out(outdir: Optional[Path] = None) -> Path:
    base = _ensure(outdir)
    return base.parent.parent if outdir is not None else base


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else None
    except Exception:
        return None
    return None


def _s(v: Any) -> str:
    return v61._s(v)


def _f(v: Any) -> Optional[float]:
    return v61._f(v)


def _pick(row: Dict[str, Any], aliases: Sequence[str]) -> Any:
    return v61._pick(row, aliases)


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
# Exact source pack scaffolding and manifest diagnostics. Template rows are
# deliberately never evidence; they are only schemas for user/public rows.
# ---------------------------------------------------------------------------

def _root_from_rel(rel: str, outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Path:
    if rel.startswith("cache/") and cache is not None:
        return cache / rel[len("cache/"):]
    if outdir is not None and rel.startswith("outdir/"):
        return outdir / rel[len("outdir/"):]
    return ROOT / rel


def _pack_checklist_text_v64(pack: str, spec: Dict[str, Any]) -> str:
    affected = ", ".join(PACK_TESTS_V64.get(pack, [])) or "none"
    columns = "\n".join(f"- {c}" for c in spec.get("columns", []))
    gates = "\n".join(f"- {k}: {v}" for k, v in PACK_MINIMUM_GATES_V64.get(pack, {}).items()) or "- no numeric gate"
    dedup = ", ".join(PACK_DEDUP_KEYS_V64.get(pack, [])) or "source_url plus physical row identity"
    return (
        f"# CCDR v64 source-pack checklist: {pack}\n\n"
        f"Purpose: {spec.get('description', '')}\n\n"
        f"Affected tests: {affected}\n\n"
        "Required columns:\n"
        f"{columns}\n\n"
        f"Suggested duplicate key: {dedup}\n\n"
        "Minimum gate before a public confirm can be claimed:\n"
        f"{gates}\n\n"
        "Accepted evidence:\n"
        "- Public or locally archived exact source rows with the required physical columns filled.\n"
        "- A source_url or equivalent source label for every usable row.\n"
        "- Non-template CSV, TSV, JSON, JSONL, YAML, XLSX, or XLS files placed in this pack directory.\n\n"
        "Rejected rows:\n"
        "- Header-only template rows.\n"
        "- Rows copied from data/generated, tierb_out, dashboards, manifests, or other derived outputs.\n"
        "- Rows with only metadata, inferred quantities, screenshots, prose summaries, or missing source labels.\n"
        "- Synthetic rows, placeholder values, or hand-estimated values.\n\n"
        "Before rerun:\n"
        "1. Keep the template file for schema reference, but add evidence in a separate non-template file.\n"
        "2. Re-run the confirm-only check and inspect source_pack_status_v64.json.\n"
        "3. Claim a public confirm only when confirm_only_dashboard_v64.json lists the test in confirmed_public_now.\n"
    )


def _pack_schema_v64(pack: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "ccdr-v64-exact-source-pack-schema",
        "pack": pack,
        "affected_tests_v64": PACK_TESTS_V64.get(pack, []),
        "description_v64": spec.get("description"),
        "required_columns_v64": spec.get("columns", []),
        "minimum_gate_v64": PACK_MINIMUM_GATES_V64.get(pack, {}),
        "dedup_key_v64": PACK_DEDUP_KEYS_V64.get(pack, []),
        "accepted_evidence_v64": [
            "filled exact public/source rows",
            "source_url or source_label on every usable row",
            "physical columns named in required_columns_v64",
        ],
        "rejected_evidence_v64": [
            "template/header-only files",
            "generated artifacts and dashboards",
            "metadata-only rows",
            "synthetic, placeholder, estimated, inferred, or derived rows where exact values are required",
        ],
    }


def _example_row_v64(spec: Dict[str, Any], rejected: bool = False) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for col in spec.get("columns", []):
        key = col.lower()
        if "source_url" == key:
            row[col] = "https://example.invalid/source-do-not-use"
        elif "source_label" == key:
            row[col] = "EXAMPLE ONLY - NOT EVIDENCE"
        elif "yes_no" in key:
            row[col] = "yes"
        elif "year" == key:
            row[col] = "2024"
        elif "temperature" in key:
            row[col] = "300"
        elif any(x in key for x in ["score", "baseline", "uncertainty", "accuracy", "energy", "bandwidth", "reach", "capacity", "area", "layers", "bits", "zt", "kappa", "grain", "value"]):
            row[col] = "1.0"
        elif "external_public" in key:
            row[col] = "yes"
        elif "certified_raw_row" in key:
            row[col] = "yes"
        elif "notes" in key:
            row[col] = "example_only_do_not_count"
        else:
            row[col] = f"example_{col}"
    if rejected:
        for col in list(row.keys()):
            if col.lower() in {"source_url", "die_area_mm2", "bits_per_cell", "observed_column", "model_column"}:
                row[col] = ""
        if "notes" in row:
            row["notes"] = "rejected_example_missing_required_exact_fields"
    return row


def _write_template_csv_v64(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_pack_support_files_v64(pack: str, spec: Dict[str, Any], directory: Path) -> Dict[str, str]:
    schema_path = directory / "SCHEMA_v64.json"
    if not schema_path.exists():
        _write_json(schema_path, _pack_schema_v64(pack, spec))
    accepted = directory / "ACCEPTED_ROW_EXAMPLE_TEMPLATE_v64.csv"
    rejected = directory / "REJECTED_ROW_EXAMPLE_TEMPLATE_v64.csv"
    if not accepted.exists():
        _write_template_csv_v64(accepted, spec["columns"], [_example_row_v64(spec, rejected=False)])
    if not rejected.exists():
        _write_template_csv_v64(rejected, spec["columns"], [_example_row_v64(spec, rejected=True)])
    if pack == "materials_family_packs":
        for family in MATERIAL_FAMILY_TEMPLATES_V64:
            family_path = directory / f"{family}_family_pack_template_v64.csv"
            if not family_path.exists():
                row = _example_row_v64(spec, rejected=False)
                row["family_name"] = family
                if "source_label" in row:
                    row["source_label"] = "EXAMPLE ONLY - NOT EVIDENCE"
                _write_template_csv_v64(family_path, spec["columns"], [row])
    if pack == "hepdata":
        table_template = directory / "LOCAL_TABLE_WITH_RESIDUAL_COLUMNS_TEMPLATE_v64.csv"
        if not table_template.exists():
            _write_template_csv_v64(
                table_template,
                ["x", "observed", "model", "uncertainty", "notes"],
                [{"x": "1.0", "observed": "1.0", "model": "1.0", "uncertainty": "0.1", "notes": "example_only_do_not_count"}],
            )
    return {
        "schema": str(schema_path),
        "accepted_example_template": str(accepted),
        "rejected_example_template": str(rejected),
    }


def init_v64_source_packs(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    """Create exact-source pack directories and schema templates.

    This function does not fabricate evidence. It writes headers and README
    files so future runs can ingest exact public/source rows from a known place.
    """
    created: List[Dict[str, Any]] = []
    for pack, spec in EXACT_PACKS.items():
        for rel in spec["dirs"]:
            d = _root_from_rel(rel, outdir, cache)
            d.mkdir(parents=True, exist_ok=True)
            readme = d / "README_v64_exact_source_pack.md"
            if not readme.exists():
                readme.write_text(
                    f"# CCDR v64 exact source pack: {pack}\n\n"
                    f"{spec['description']}\n\n"
                    "Rows here are counted only when physical required columns are filled. "
                    "Template/header-only files are not evidence.\n",
                    encoding="utf-8",
                )
            templ = d / spec["template"]
            if not templ.exists():
                with templ.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(spec["columns"])
            checklist = d / "CHECKLIST_v64.md"
            if not checklist.exists():
                checklist.write_text(_pack_checklist_text_v64(pack, spec), encoding="utf-8")
            support = _write_pack_support_files_v64(pack, spec, d)
            created.append({"pack": pack, "dir": str(d), "template": str(templ), "checklist": str(checklist), "support_files": support, "columns": spec["columns"]})
    rows = []
    for x in created:
        rows.append({"pack": x["pack"], "dir": x["dir"], "template": x["template"], "checklist": x["checklist"], "schema": x["support_files"].get("schema"), "columns": ";".join(x["columns"])})
    _write_csv(rows, "v64_exact_source_pack_manifest.csv", outdir)
    obj = {"schema": "ccdr-v64-exact-source-pack-manifest", "packs": created, "note": "Templates are not evidence; filled exact public rows are required."}
    _write_json(_ensure(outdir) / "v64_exact_source_pack_manifest.json", obj)
    return obj


def _pack_dirs(pack: str, outdir: Optional[Path] = None, cache: Optional[Path] = None) -> List[Path]:
    spec = EXACT_PACKS[pack]
    dirs: List[Path] = []
    for rel in spec["dirs"]:
        p = _root_from_rel(rel, outdir, cache)
        if p.exists() and p not in dirs:
            dirs.append(p)
    # Also allow user to point at an external v64 pack root.
    extra = os.environ.get("CCDR_V64_SOURCE_PACK_ROOT")
    if extra:
        ep = Path(extra)
        if ep.exists():
            for sub in [pack, pack.replace("_", "-"), spec["dirs"][0].split("/")[-1]]:
                sp = ep / sub
                if sp.exists() and sp not in dirs:
                    dirs.append(sp)
            if ep not in dirs:
                dirs.append(ep)
    return dirs


def _is_template_or_readme(p: Path) -> bool:
    name = p.name.lower()
    if ".quarantine_v72." in name or ".mixed_invalid_backup_v72." in name:
        return True
    if name.endswith(".bak") or "backup_v72" in name or "quarantine" in name:
        return True
    if name.startswith("readme"):
        return True
    if name.startswith("schema"):
        return True
    if name.startswith("checklist"):
        return True
    if "example" in name:
        return True
    if "template" in name:
        return True
    return False


def _ignored_stale_auto_paths_v72(outdir: Optional[Path] = None) -> set[str]:
    path = _ensure(outdir) / "stale_auto_rows_ignore_v72.json"
    try:
        if not path.exists():
            return set()
        obj = json.loads(path.read_text(encoding="utf-8"))
        paths = obj.get("ignored_source_files_v72", []) if isinstance(obj, dict) else []
        return {str(Path(x).resolve()) for x in paths if x}
    except Exception:
        return set()


def _iter_pack_files(pack: str, outdir: Optional[Path] = None, cache: Optional[Path] = None, max_files: int = 250) -> List[Path]:
    suffixes = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".xlsx", ".xls"}
    files: List[Path] = []
    seen = set()
    ignored_stale_auto = _ignored_stale_auto_paths_v72(outdir)
    for root in _pack_dirs(pack, outdir, cache):
        for p in root.rglob("*"):
            if len(files) >= max_files:
                return files
            if not p.is_file() or p.suffix.lower() not in suffixes:
                continue
            if _is_template_or_readme(p) or v63._is_generated_or_output_path(p):
                continue
            try:
                if p.stat().st_size > int(os.environ.get("CCDR_V64_MAX_SOURCE_FILE_BYTES", "80000000")):
                    continue
            except Exception:
                pass
            rp = str(p.resolve())
            if rp in ignored_stale_auto:
                continue
            if rp in seen:
                continue
            seen.add(rp)
            files.append(p)
    return files


def _read_pack_rows(pack: str, outdir: Optional[Path] = None, cache: Optional[Path] = None, max_files: int = 250, max_rows_per_file: int = 200000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files = _iter_pack_files(pack, outdir, cache, max_files=max_files)
    for p in files:
        rr = v63._read_table_v63(p, max_rows=max_rows_per_file)
        for r in rr:
            r["_v64_pack"] = pack
            r["_source_file_v64"] = r.get("_source_file_v63") or str(p)
        rows.extend(rr)
    return rows


def _pack_validation_problems_v74(outdir: Optional[Path]) -> Dict[str, List[Dict[str, Any]]]:
    report_path = _root_out(outdir) / "v64_source_pack_validation.json"
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for pack_report in report.get("pack_results", []) if isinstance(report, dict) else []:
        if not isinstance(pack_report, dict):
            continue
        pack = _s(pack_report.get("pack"))
        problems = pack_report.get("problems_v64") or []
        if pack and problems:
            out[pack] = [p for p in problems if isinstance(p, dict)]
    return out


def _preconfirm_validation_block_v74(test_id: str, outdir: Optional[Path]) -> Optional[Dict[str, Any]]:
    tid = test_id.upper()
    required_packs = TEST_REQUIRED_PACKS_V64.get(tid, [])
    if not required_packs:
        return None
    problems = _pack_validation_problems_v74(outdir)
    blocked = {pack: problems.get(pack, []) for pack in required_packs if problems.get(pack)}
    if not blocked:
        return None
    gate = {
        "schema": "ccdr-preconfirm-validation-block-v74",
        "test_id": tid,
        "strict_confirm_ready_v64": False,
        "strict_public_confirm_v64": False,
        "confirmation_status_v64": "not_confirmed_source_pack_validation_failed_v74",
        "blocked_packs_v74": sorted(blocked),
        "pack_problem_counts_v74": {pack: len(rows) for pack, rows in sorted(blocked.items())},
        "top_pack_problems_v74": {pack: rows[:10] for pack, rows in sorted(blocked.items())},
        "required_action_v74": "Quarantine or repair invalid public-source rows before running confirm gates.",
    }
    wrote = False
    for pack in required_packs:
        for path in _gate_paths_for_pack_v64(pack, outdir):
            if tid.lower() in path.name.lower():
                _write_json(path, gate)
                wrote = True
    if not wrote:
        _write_json(_ensure(outdir) / f"{tid.lower()}_preconfirm_validation_block_v74.json", gate)
    return gate


def _source_pack_summary(pack: str, rows: Sequence[Dict[str, Any]], required: Sequence[str], outdir: Optional[Path] = None) -> Dict[str, Any]:
    files = sorted({_s(r.get("_source_file_v64") or r.get("_source_file_v63")) for r in rows if _s(r.get("_source_file_v64") or r.get("_source_file_v63"))})
    filled = Counter()
    for r in rows:
        keys = {k.lower(): v for k, v in r.items()}
        for req in required:
            if any(req.lower() in k and _s(v) for k, v in keys.items()):
                filled[req] += 1
    obj = {"schema": "ccdr-v64-source-pack-summary", "pack": pack, "n_files": len(files), "n_rows": len(rows), "files": files[:50], "required_fill_counts": dict(filled), "missing_pack_if_no_files": len(files) == 0}
    _write_json(_ensure(outdir) / f"v64_{pack}_source_pack_summary.json", obj)
    return obj


def _has_required_column_v64(row: Dict[str, Any], required: str) -> bool:
    req = required.lower().replace("_", "")
    for key, val in row.items():
        kk = str(key).lower().replace("_", "")
        if req == kk or req in kk or kk in req:
            if _s(val):
                return True
    return False


def _missing_required_columns_v64(pack: str, row: Dict[str, Any], required: Sequence[str]) -> List[str]:
    """Pack-aware required-column check.

    Most exact packs require every schema column. T53 structure rows are the
    exception: a public AlphaFold mapping is acceptable when an experimental
    PDB id is unavailable, because the confirm gate itself requires
    ``pdb_id OR alphafold_id``.
    """
    if pack == "protein_structures":
        missing = [
            col
            for col in ["uniprot", "oligomeric_state", "symmetry_proxy", "contact_network_proxy", "fold_class", "source_url"]
            if not _has_required_column_v64(row, col)
        ]
        if not (_has_required_column_v64(row, "pdb_id") or _has_required_column_v64(row, "alphafold_id")):
            missing.append("pdb_id_or_alphafold_id")
        return missing
    if pack == "proteingym":
        missing = [col for col in required if not _has_required_column_v64(row, col)]
        # Keep the exact schema strict, but report this common issue clearly.
        if "dms_score" not in missing and "fitness_residual" not in missing:
            return missing
        return missing
    return [col for col in required if not _has_required_column_v64(row, col)]


def _row_identity_v64(row: Dict[str, Any], keys: Sequence[str]) -> str:
    vals = []
    lower = {str(k).lower(): k for k in row.keys()}
    for key in keys:
        actual = lower.get(key.lower())
        vals.append(_s(row.get(actual)) if actual is not None else "")
    return "|".join(vals)


def _row_has_forbidden_evidence_marker_v64(row: Dict[str, Any]) -> bool:
    text = " ".join(_s(v) for v in row.values())
    return bool(re.search(r"\b(template|example_only|do_not_count|synthetic|placeholder|estimated|inferred|derived)\b", text, re.I))


def _pack_specific_diagnostics_v64(pack: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if pack == "proteingym":
        stale_manifest_rows = 0
        bad_variant_rows = 0
        nonnumeric_score_rows = 0
        for r in rows:
            problems = _pack_specific_row_problems_v64(pack, r)
            stale_manifest_rows += int("proteingym_reference_manifest_is_metadata_not_variant_scores" in problems)
            bad_variant_rows += int("proteingym_variant_column_not_mutation_identifier" in problems)
            nonnumeric_score_rows += int("proteingym_score_not_numeric" in problems)
        usable_like = [
            r for r in rows
            if not _missing_required_columns_v64(pack, r, EXACT_PACKS[pack]["columns"])
            and not _row_has_forbidden_evidence_marker_v64(r)
            and not _pack_specific_row_problems_v64(pack, r)
        ]
        return {
            "schema": "ccdr-v64-proteingym-pack-diagnostics",
            "n_rows_seen_v64": len(rows),
            "n_schema_complete_rows_v64": len(usable_like),
            "n_assays_v64": len({_s(_pick(r, ["assay_id", "assay", "DMS_id"])) for r in usable_like if _s(_pick(r, ["assay_id", "assay", "DMS_id"]))}),
            "n_uniprots_v64": len({_s(_pick(r, ["uniprot", "uniprot_id", "UniProt_ID", "accession"])).upper() for r in usable_like if _s(_pick(r, ["uniprot", "uniprot_id", "UniProt_ID", "accession"]))}),
            "n_families_v64": len({_s(_pick(r, ["family", "protein_family", "source_organism"])) for r in usable_like if _s(_pick(r, ["family", "protein_family", "source_organism"]))}),
            "n_sequence_clusters_v64": len({_s(_pick(r, ["sequence_cluster", "cluster", "seq_cluster"])) for r in usable_like if _s(_pick(r, ["sequence_cluster", "cluster", "seq_cluster"]))}),
            "stale_row_guard_v72": {
                "n_reference_manifest_rows_v72": stale_manifest_rows,
                "n_bad_variant_identifier_rows_v72": bad_variant_rows,
                "n_nonnumeric_score_rows_v72": nonnumeric_score_rows,
                "cleanup_action_v72": "quarantine generated proteingym/AUTO_PUBLIC_ROWS_V67.csv and rerun raw DMS parser" if rows and not usable_like else "",
            },
            "note_v64": "T53 needs these assay rows joined to protein_structures rows before confirmation.",
        }
    if pack == "protein_structures":
        usable_like = [
            r for r in rows
            if not _missing_required_columns_v64(pack, r, EXACT_PACKS[pack]["columns"])
            and not _row_has_forbidden_evidence_marker_v64(r)
            and not _pack_specific_row_problems_v64(pack, r)
        ]
        return {
            "schema": "ccdr-v64-protein-structures-pack-diagnostics",
            "n_rows_seen_v64": len(rows),
            "n_schema_complete_rows_v64": len(usable_like),
            "n_uniprots_v64": len({_s(_pick(r, ["uniprot", "uniprot_id", "accession"])).upper() for r in usable_like if _s(_pick(r, ["uniprot", "uniprot_id", "accession"]))}),
            "n_alphafold_rows_v64": sum(1 for r in usable_like if _s(_pick(r, ["alphafold_id", "alphafold", "af_id"]))),
            "n_pdb_rows_v64": sum(1 for r in usable_like if _s(_pick(r, ["pdb_id", "pdb", "structure_id"]))),
            "note_v64": "PDB id is optional when a public AlphaFold id and structure-derived feature proxies are present.",
        }
    return {}


def _pack_specific_row_problems_v64(pack: str, row: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if pack == "proteingym":
        source = _s(_pick(row, ["source_url", "url", "harvest_source_url_v67", "_source_file_v64", "_source_file_v63"]))
        variant = _s(_pick(row, ["variant", "mutant", "mutation"]))
        dms_score = _f(_pick(row, ["dms_score", "DMS_score", "fitness", "score", "effect"]))
        if re.search(r"/reference_files/DMS_(substitutions|indels)\.csv", source.replace("\\", "/"), re.I):
            problems.append("proteingym_reference_manifest_is_metadata_not_variant_scores")
        if variant.lower() in {"true", "false", "0", "1", "yes", "no", "nan", "none", "null"} or len(variant) > 120:
            problems.append("proteingym_variant_column_not_mutation_identifier")
        elif not (re.search(r"[A-Za-z]", variant) and re.search(r"\d", variant)):
            problems.append("proteingym_variant_column_not_mutation_identifier")
        if dms_score is None:
            problems.append("proteingym_score_not_numeric")
    return problems


def validate_v64_source_packs(outdir: Optional[Path] = None, cache: Optional[Path] = None, max_rows_per_pack: int = 50000) -> Dict[str, Any]:
    init_v64_source_packs(outdir, cache)
    pack_results: List[Dict[str, Any]] = []
    invalid_files = 0
    invalid_rows = 0
    for pack, spec in EXACT_PACKS.items():
        rows = _read_pack_rows(pack, outdir, cache, max_files=250, max_rows_per_file=max_rows_per_pack)
        required = list(spec.get("columns", []))
        dedup_keys = PACK_DEDUP_KEYS_V64.get(pack, [])
        problems: List[Dict[str, Any]] = []
        usable = 0
        seen_identities: Dict[str, int] = {}
        files = sorted({_s(r.get("_source_file_v64") or r.get("_source_file_v63")) for r in rows if _s(r.get("_source_file_v64") or r.get("_source_file_v63"))})
        for idx, row in enumerate(rows):
            missing = _missing_required_columns_v64(pack, row, required)
            identity = _row_identity_v64(row, dedup_keys)
            if identity.strip("|"):
                seen_identities[identity] = seen_identities.get(identity, 0) + 1
            row_problems = []
            if missing:
                row_problems.append("missing_required_columns")
            if not (_has_required_column_v64(row, "source_url") or _has_required_column_v64(row, "source_label")):
                row_problems.append("missing_source_url_or_label")
            if _row_has_forbidden_evidence_marker_v64(row):
                row_problems.append("forbidden_template_synthetic_estimated_or_derived_marker")
            row_problems.extend(_pack_specific_row_problems_v64(pack, row))
            source_file = _s(row.get("_source_file_v64") or row.get("_source_file_v63"))
            if source_file and v63._is_generated_or_output_path(Path(source_file)):
                row_problems.append("generated_or_output_path_not_evidence")
            if row_problems:
                invalid_rows += 1
                if len(problems) < 100:
                    problems.append({
                        "row_index_v64": idx,
                        "source_file_v64": source_file,
                        "problem_v64": "|".join(row_problems),
                        "missing_columns_v64": missing[:30],
                    })
            else:
                usable += 1
        duplicates = [k for k, v in seen_identities.items() if k.strip("|") and v > 1]
        if duplicates:
            problems.append({"problem_v64": "duplicate_evidence_keys", "n_duplicates_v64": len(duplicates), "sample_keys_v64": duplicates[:10]})
        if problems:
            invalid_files += len({p.get("source_file_v64") for p in problems if p.get("source_file_v64")})
        gate = PACK_MINIMUM_GATES_V64.get(pack, {})
        pack_diag = _pack_specific_diagnostics_v64(pack, rows)
        pack_results.append({
            "pack": pack,
            "affected_tests_v64": PACK_TESTS_V64.get(pack, []),
            "n_files_v64": len(files),
            "n_rows_v64": len(rows),
            "validator_usable_rows_v64": usable,
            "required_columns_v64": required,
            "dedup_key_v64": dedup_keys,
            "minimum_gate_v64": gate,
            "problems_v64": problems,
            "status_v64": "validator_passes_nonempty_pack" if usable else ("empty_pack_needs_exact_public_rows" if not rows else "validator_found_row_problems"),
            "ready_to_attempt_confirm_gate_v64": bool(usable and not problems),
            "pack_specific_diagnostics_v64": pack_diag,
        })
    obj = {
        "schema": "ccdr-v64-source-pack-validation",
        "pack_results": pack_results,
        "n_packs_v64": len(pack_results),
        "n_invalid_rows_v64": invalid_rows,
        "n_problem_files_v64": invalid_files,
        "all_existing_rows_valid_v64": invalid_rows == 0,
        "note_v64": "Empty packs are expected blockers, not validation errors. Generated, template, synthetic, estimated, inferred, and derived rows never count as evidence.",
    }
    root = _root_out(outdir)
    _write_json(root / "v64_source_pack_validation.json", obj)
    _write_csv([
        {
            "pack": p["pack"],
            "n_files_v64": p["n_files_v64"],
            "n_rows_v64": p["n_rows_v64"],
            "validator_usable_rows_v64": p["validator_usable_rows_v64"],
            "status_v64": p["status_v64"],
            "affected_tests_v64": ";".join(p["affected_tests_v64"]),
        }
        for p in pack_results
    ], "v64_source_pack_validation_summary.csv", outdir)
    return obj


def _next_rows_needed_for_test_v64(test_id: str, outdir: Optional[Path] = None) -> Dict[str, Any]:
    tid = test_id.upper()
    packs = TEST_REQUIRED_PACKS_V64.get(tid, [])
    pack_details = []
    for pack in packs:
        spec = EXACT_PACKS.get(pack, {})
        first_dir = spec.get("dirs", [""])[0]
        pack_dir = _root_from_rel(first_dir, outdir, None)
        pack_details.append({
            "pack": pack,
            "pack_dir_v64": str(pack_dir),
            "template_v64": str(pack_dir / spec.get("template", "")),
            "checklist_v64": str(pack_dir / "CHECKLIST_v64.md"),
            "schema_v64": str(pack_dir / "SCHEMA_v64.json"),
            "required_columns_v64": spec.get("columns", []),
            "minimum_gate_v64": PACK_MINIMUM_GATES_V64.get(pack, {}),
            "dedup_key_v64": PACK_DEDUP_KEYS_V64.get(pack, []),
        })
    obj = {
        "schema": "ccdr-v64-next-rows-needed",
        "test_id": tid,
        "test_name": TESTS.get(tid, {}).get("name"),
        "claim_goal_v64": "Move toward confirmed_public_now only after strict exact-source gates pass.",
        "current_next_source_v64": _next_source_v64(tid),
        "required_packs_v64": pack_details,
        "accepted_rows_v64": "Machine-parsed exact public/source rows with required physical columns and source URL/label.",
        "rejected_rows_v64": "Templates, generated artifacts, metadata-only rows, screenshots, synthetic rows, manual placeholders, and unsupported synthetic/estimated rows.",
    }
    return obj


def write_next_rows_needed_v64(tests: Sequence[str], outdir: Optional[Path] = None) -> Dict[str, Any]:
    root = _root_out(outdir)
    rows = []
    for tid in [t.upper() for t in tests]:
        obj = _next_rows_needed_for_test_v64(tid, outdir)
        if obj.get("required_packs_v64"):
            _write_json(root / f"{tid.lower()}_next_rows_needed_v64.json", obj)
        rows.append(obj)
    bundle = {
        "schema": "ccdr-v64-next-rows-needed-bundle",
        "tests": rows,
        "note_v64": "These manifests describe rows needed to push tests toward confirmation. They do not count as evidence.",
    }
    _write_json(root / "next_rows_needed_v64.json", bundle)
    return bundle


def _full_run_summary_candidates(outdir: Optional[Path]) -> List[Path]:
    root = _root_out(outdir)
    return [
        root / "tier_b_batch_summary.json",
        root.parent / "tier_b_batch_summary.json",
    ]


def _process_summary_map_v64(outdir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    for path in _full_run_summary_candidates(outdir):
        obj = _read_json_if_exists(path)
        if not obj:
            continue
        rows = obj.get("summary")
        if not isinstance(rows, list):
            continue
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            tid = _s(row.get("test_id")).upper()
            if tid:
                out[tid] = {
                    "test_id": tid,
                    "process_status": row.get("process_status"),
                    "result_status": row.get("result_status"),
                    "script": row.get("script"),
                }
        return out
    return {}


def _process_summary_v64(outdir: Optional[Path], tests: Sequence[str]) -> Dict[str, Any]:
    proc = _process_summary_map_v64(outdir)
    rows = [proc[t.upper()] for t in tests if t.upper() in proc]
    timeouts = [r["test_id"] for r in rows if r.get("process_status") == "process_timeout"]
    errors = [r["test_id"] for r in rows if r.get("process_status") not in {None, "ok", "process_timeout"}]
    return {
        "schema": "ccdr-v64-process-vs-confirm-summary",
        "summary_available_v64": bool(proc),
        "n_tests_with_process_rows_v64": len(rows),
        "process_timeouts_v64": timeouts,
        "process_errors_v64": errors,
        "note_v64": "process_status reports subprocess health; confirmation_status_v64 reports public-claim eligibility.",
    }


def _gate_paths_for_pack_v64(pack: str, outdir: Optional[Path]) -> List[Path]:
    gen = _ensure(outdir)
    mapping = {
        "materials": ["t31_materials_exact_confirm_gate_v64.json", "t32_materials_exact_confirm_gate_v64.json"],
        "materials_family_packs": ["t31_materials_exact_confirm_gate_v64.json", "t32_materials_exact_confirm_gate_v64.json"],
        "nand": ["t44_nand_tier_a_gate_v64.json"],
        "proteingym": ["t53_proteingym_structure_gate_v64.json"],
        "protein_structures": ["t53_proteingym_structure_gate_v64.json"],
        "thermoelectric": ["t34_thermoelectric_angle_gate_v64.json"],
        "hepdata": ["t57_hepdata_exact_gate_v64.json", "t59_hepdata_exact_gate_v64.json"],
        "optical_interconnect": ["t45_benchmark_exact_gate_v64.json"],
        "neuromorphic": ["t47_benchmark_exact_gate_v64.json"],
        "fusion": [f"t{i}_fusion_exact_rows_v64.json" for i in range(26, 31)],
        "ldpc_external_benchmark": ["t46_external_public_benchmark_gate_v64.json"],
    }
    return [gen / name for name in mapping.get(pack, [])]


def _usable_count_from_gate_v64(gate: Dict[str, Any]) -> int:
    keys = [
        "n_usable_rows_v64",
        "n_usable_tier_a_rows_v64",
        "n_usable_join_rows_v64",
        "n_residual_rows_v64",
        "n_exact_rows_v64",
        "n_model_rows_v64",
    ]
    for key in keys:
        val = _f(gate.get(key))
        if val is not None:
            return int(val)
    model = gate.get("model_v64")
    if isinstance(model, dict):
        val = _f(model.get("n_model_rows_v64"))
        if val is not None:
            return int(val)
    return 0


def _source_pack_status_v64(outdir: Optional[Path], validation: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    gen = _ensure(outdir)
    rows: List[Dict[str, Any]] = []
    validation = validation or _read_json_if_exists(_root_out(outdir) / "v64_source_pack_validation.json") or {}
    validation_by_pack = {
        _s(p.get("pack")): p
        for p in validation.get("pack_results", [])
        if isinstance(p, dict)
    }
    for pack, spec in EXACT_PACKS.items():
        summary = _read_json_if_exists(gen / f"v64_{pack}_source_pack_summary.json") or {}
        val = validation_by_pack.get(pack, {})
        gates = []
        for path in _gate_paths_for_pack_v64(pack, outdir):
            gate = _read_json_if_exists(path)
            if gate:
                gates.append(gate)
        usable_counts = [_usable_count_from_gate_v64(g) for g in gates]
        strict_ready = any(bool(g.get("strict_confirm_ready_v64")) for g in gates)
        impacted = PACK_TESTS_V64.get(pack, [])
        template_paths = []
        checklist_paths = []
        for rel in spec.get("dirs", []):
            pack_dir = _root_from_rel(rel, outdir, None)
            template_paths.append(str(pack_dir / spec.get("template", "")))
            checklist_paths.append(str(pack_dir / "CHECKLIST_v64.md"))
        rows.append({
            "pack": pack,
            "affected_tests_v64": impacted,
            "n_files_v64": int(summary.get("n_files") or 0),
            "n_rows_v64": int(summary.get("n_rows") or 0),
            "max_usable_rows_v64": max(usable_counts) if usable_counts else 0,
            "strict_ready_any_test_v64": strict_ready,
            "minimum_gate_v64": PACK_MINIMUM_GATES_V64.get(pack, {}),
            "status_v64": "passes_pack_gate" if strict_ready else "needs_filled_exact_public_rows",
            "validator_status_v64": val.get("status_v64"),
            "validator_usable_rows_v64": val.get("validator_usable_rows_v64", 0),
            "validator_problem_count_v64": len(val.get("problems_v64", []) or []),
            "template_paths_v64": template_paths,
            "checklist_paths_v64": checklist_paths,
            "schema_paths_v64": [p.replace("CHECKLIST_v64.md", "SCHEMA_v64.json") for p in checklist_paths],
            "next_action_v64": "Run the v67 public-source harvester to discover, download, parse, and write AUTO_PUBLIC_ROWS_V67.csv rows; templates, generated artifacts, and manual placeholder rows do not count.",
        })
    _write_json(_root_out(outdir) / "source_pack_status_v64.json", {"schema": "ccdr-tierb-source-pack-status-v64", "packs": rows})
    return rows


def _full_run_result_candidates_v64(test_id: str, outdir: Optional[Path]) -> List[Path]:
    tid = test_id.lower()
    root = _root_out(outdir)
    candidates: List[Path] = []
    if root.name.lower().startswith("confirm_only"):
        candidates.append(root.parent / f"{tid}_result.json")
    candidates.extend([root / f"{tid}_result.json", root.parent / f"{tid}_result.json"])
    out: List[Path] = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _full_run_result_with_path_v64(test_id: str, outdir: Optional[Path]) -> Tuple[Dict[str, Any], Optional[Path]]:
    for path in _full_run_result_candidates_v64(test_id, outdir):
        obj = _read_json_if_exists(path)
        if obj:
            return obj, path
    return {}, None


def _full_run_result_v64(test_id: str, outdir: Optional[Path]) -> Dict[str, Any]:
    obj, _ = _full_run_result_with_path_v64(test_id, outdir)
    return obj


def _claim_bucket_v64(test_id: str, status: str, strict: bool) -> str:
    tid = test_id.upper()
    if tid in CONFIRMED_PUBLIC or (strict and status.startswith("confirmed")):
        return "confirmed_public_now"
    if tid in NEAR:
        return "near_confirm_requires_exact_rows"
    if tid in BOUND:
        return "bound_only"
    if tid in ANCHOR:
        return "anchor_only"
    if tid in FUSION:
        return "diagnostic_only"
    if tid in SYNTHETIC_OR_ENGINEERING:
        return "synthetic_or_engineering"
    if "data_limited" in status or "not_confirmed" in status:
        return "data_limited"
    return "no_confirm"


def _why_not_confirmed_v64(test_id: str, status: str, strict: bool) -> Optional[str]:
    tid = test_id.upper()
    if tid in CONFIRMED_PUBLIC or (strict and status.startswith("confirmed")):
        return None
    if tid in BOUND:
        return "Bound/constraint audit; useful for limits but not confirmable by design."
    if tid in ANCHOR:
        return "Anchor-only consistency result; full sector confirmation requires separate quark/lattice and sensitivity gates."
    if tid in FUSION:
        return "Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate."
    if tid in {"T31", "T32"}:
        return "Exact measured materials rows with kappa(T), grain size, source URL, and measured microstructure method are still required."
    if tid == "T44":
        return "True Tier-A NAND rows with complete die-area, capacity, layers, and bits/cell fields are still required."
    if tid == "T53":
        return "ProteinGym assay rows must be joined to UniProt/PDB/AlphaFold structure features before a model confirm is allowed."
    if tid == "T34":
        return "Exact thermoelectric ZT plus orientation/grain-boundary angle rows are still required."
    if tid in {"T45", "T47"}:
        return "Exact benchmark rows are still required; metadata and generated diagnostics do not count."
    if tid in {"T57", "T59"}:
        return "Frozen HEPData manifests and residual rows with observed/model/uncertainty columns are still required."
    if tid == "T46":
        return "The current ok result is synthetic/engineering only; no external public benchmark confirm gate passes."
    return "Required public structured physical rows are missing or insufficient for a confirm claim."


def _next_source_v64(test_id: str) -> str:
    tid = test_id.upper()
    if tid in NEXT_SOURCE_V64:
        return NEXT_SOURCE_V64[tid]
    meta = TESTS.get(tid, {})
    family = meta.get("family", "public-data")
    return f"Exact public structured {family} rows with the named physical columns required by this test."


# ---------------------------------------------------------------------------
# Materials T31/T32: exact source packs + small curated set mode + stronger
# source/family/temperature diagnostics.
# ---------------------------------------------------------------------------

MATERIAL_REQUIRED = ["material", "temperature", "kappa", "grain", "microstructure"]


def normalize_material_row_v64(raw: Dict[str, Any], idx: int) -> Tuple[Dict[str, Any], List[str]]:
    raw2 = dict(raw)
    if "_source_file_v63" not in raw2 and "_source_file_v64" in raw2:
        raw2["_source_file_v63"] = raw2["_source_file_v64"]
    nr, reasons = v63.normalize_material_row_v63(raw2, idx)
    nr = {k.replace("_v63", "_v64"): v for k, v in nr.items()}
    src = _s(raw.get("source_url") or raw.get("source_reference") or raw.get("_source_file_v64") or raw.get("_source_file_v63"))
    nr["source_id_v64"] = src
    nr["raw_source_file_v64"] = _s(raw.get("_source_file_v64") or raw.get("_source_file_v63"))
    nr["source_pack_v64"] = _s(raw.get("_v64_pack") or "materials")
    nr["family_pack_v64"] = _s(raw.get("family_pack") or raw.get("family_name") or nr.get("material_family_v64"))
    nr["row_provenance_hash_v64"] = _provenance_hash(src, nr.get("sample_id_v64"), nr.get("material_v64"), nr.get("temperature_K_v64"), nr.get("kappa_W_mK_v64"), nr.get("grain_size_nm_v64"))
    return nr, reasons


def _dedup_materials_v64(norm: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for r in norm:
        key = (
            _s(r.get("source_id_v64")), _s(r.get("sample_id_v64")), _s(r.get("material_v64")).lower(),
            round(float(r.get("temperature_K_v64") or 0), 3), round(float(r.get("kappa_W_mK_v64") or 0), 6), round(float(r.get("grain_size_nm_v64") or 0), 3),
        )
        if key in seen:
            continue
        seen.add(key); out.append(r)
    return out


def _ols(y: Sequence[float], X: Sequence[Sequence[float]]) -> Optional[Dict[str, Any]]:
    return v61._ols_fit(list(y), [list(x) for x in X])


def _demean(vals: List[float], groups: Sequence[str]) -> List[float]:
    sums: Dict[str, float] = defaultdict(float); counts: Dict[str, int] = defaultdict(int)
    for v, g in zip(vals, groups):
        sums[g] += v; counts[g] += 1
    return [v - sums[g] / max(1, counts[g]) for v, g in zip(vals, groups)]


def _materials_model_v64(usable: List[Dict[str, Any]], test_id: str, outdir: Optional[Path]) -> Dict[str, Any]:
    rows = []
    for r in usable:
        t = _f(r.get("temperature_K_v64")); k = _f(r.get("kappa_W_mK_v64")); g = _f(r.get("grain_size_nm_v64"))
        if t and k and g and t > 0 and k > 0 and g > 0:
            x = dict(r)
            x["logT_v64"] = math.log(t); x["logKappa_v64"] = math.log(k); x["logGrain_v64"] = math.log(g)
            x["boundary_proxy_num_v64"] = _f(r.get("boundary_density_proxy_v64")) or 1.0 / g
            rows.append(x)
    if len(rows) < 12 or np is None:
        return {"status_v64": "not_enough_rows_for_estimator", "n_model_rows_v64": len(rows)}
    y = [float(r["logKappa_v64"]) for r in rows]
    t = [float(r["logT_v64"]) for r in rows]
    g = [float(r["logGrain_v64"]) for r in rows]
    b = [float(r["boundary_proxy_num_v64"]) for r in rows]
    src = [_s(r.get("source_id_v64")) for r in rows]
    fam = [_s(r.get("material_family_v64")) for r in rows]
    # Two-stage residualization: source and material family.
    yr = _demean(_demean(y, src), fam)
    tr = _demean(_demean(t, src), fam)
    gr = _demean(_demean(g, src), fam)
    br = _demean(_demean(b, src), fam)
    fit_temp = _ols(yr, [[1.0, a] for a in tr])
    fit_micro = _ols(yr, [[1.0, a, c, d] for a, c, d in zip(tr, gr, br)])
    model_wins = bool(fit_temp and fit_micro and fit_micro.get("aic", 1e99) < fit_temp.get("aic", -1e99) and fit_micro.get("bic", 1e99) < fit_temp.get("bic", -1e99))
    beta = fit_micro.get("beta") if fit_micro else []
    sign_ok = bool(beta and len(beta) > 3 and beta[2] > 0 and beta[3] < 0)
    # Family-source balanced bootstrap: sample sources and families first.
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(_s(r.get("source_id_v64")), _s(r.get("material_family_v64")))].append(r)
    group_keys = list(groups)
    boot_n = boot_win = boot_sign = 0
    if len(group_keys) >= 4 and np is not None:
        rng = np.random.default_rng(6401)
        for _ in range(200):
            sample = []
            for key in rng.choice(len(group_keys), size=len(group_keys), replace=True):
                grp = groups[group_keys[int(key)]]
                idx = rng.integers(0, len(grp), size=max(1, min(len(grp), 60)))
                sample.extend(grp[int(i)] for i in idx)
            if len(sample) < 12:
                continue
            yy = [float(r["logKappa_v64"]) for r in sample]
            tt = [float(r["logT_v64"]) for r in sample]
            gg = [float(r["logGrain_v64"]) for r in sample]
            bb = [float(r["boundary_proxy_num_v64"]) for r in sample]
            ss = [_s(r.get("source_id_v64")) for r in sample]
            ff = [_s(r.get("material_family_v64")) for r in sample]
            yy = _demean(_demean(yy, ss), ff); tt = _demean(_demean(tt, ss), ff); gg = _demean(_demean(gg, ss), ff); bb = _demean(_demean(bb, ss), ff)
            ft = _ols(yy, [[1.0, a] for a in tt]); fm = _ols(yy, [[1.0, a, c, d] for a, c, d in zip(tt, gg, bb)])
            if not ft or not fm:
                continue
            boot_n += 1
            if fm.get("aic", 1e99) < ft.get("aic", -1e99) and fm.get("bic", 1e99) < ft.get("bic", -1e99):
                boot_win += 1
            bbeta = fm.get("beta") or []
            if len(bbeta) > 3 and bbeta[2] > 0 and bbeta[3] < 0:
                boot_sign += 1
    # Temperature-bin wins with microstructure vs temp-only inside bins.
    by_bin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bin[_s(r.get("temperature_bin_v64") or _temp_bin(_f(r.get("temperature_K_v64"))))].append(r)
    bin_rows = []
    for tb, group in sorted(by_bin.items()):
        if len(group) < 8:
            bin_rows.append({"temperature_bin_v64": tb, "n_rows_v64": len(group), "model_tested_v64": False, "microstructure_wins_v64": False})
            continue
        yy = [float(r["logKappa_v64"]) for r in group]
        tt = [float(r["logT_v64"]) for r in group]
        gg = [float(r["logGrain_v64"]) for r in group]
        bb = [float(r["boundary_proxy_num_v64"]) for r in group]
        ft = _ols(yy, [[1.0, a] for a in tt]); fm = _ols(yy, [[1.0, a, c, d] for a, c, d in zip(tt, gg, bb)])
        win = bool(ft and fm and fm.get("aic", 1e99) < ft.get("aic", -1e99) and fm.get("bic", 1e99) < ft.get("bic", -1e99))
        bin_rows.append({"temperature_bin_v64": tb, "n_rows_v64": len(group), "model_tested_v64": True, "microstructure_wins_v64": win, "aic_temp_v64": ft.get("aic") if ft else None, "aic_micro_v64": fm.get("aic") if fm else None})
    _write_csv(bin_rows, f"{test_id.lower()}_temperature_bin_model_wins_v64.csv", outdir)
    tested_bins = [x for x in bin_rows if x.get("model_tested_v64")]
    return {
        "status_v64": "ok", "n_model_rows_v64": len(rows), "fixed_effect_method_v64": "source+material_family residualized OLS",
        "temperature_only_fit_v64": fit_temp, "microstructure_fit_v64": fit_micro,
        "microstructure_beats_temperature_baseline_v64": model_wins,
        "predicted_signs_pass_v64": sign_ok,
        "family_source_balanced_bootstrap_n_v64": boot_n,
        "family_source_balanced_bootstrap_model_win_fraction_v64": boot_win / boot_n if boot_n else 0.0,
        "family_source_balanced_bootstrap_sign_fraction_v64": boot_sign / boot_n if boot_n else 0.0,
        "temperature_bin_results_v64": bin_rows,
        "temperature_bin_win_fraction_v64": sum(1 for x in tested_bins if x.get("microstructure_wins_v64")) / len(tested_bins) if tested_bins else 0.0,
        "tested_temperature_bins_v64": len(tested_bins),
    }


def materials_confirm_v64(test_id: str, outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    init_v64_source_packs(outdir, cache)
    # Strict evidence rows: only exact source packs, not historical generated artifacts.
    raw = _read_pack_rows("materials", outdir, cache, max_files=220, max_rows_per_file=100000)
    raw += _read_pack_rows("materials_family_packs", outdir, cache, max_files=120, max_rows_per_file=50000)
    pack_summary = _source_pack_summary("materials", raw, MATERIAL_REQUIRED, outdir)
    norm: List[Dict[str, Any]] = []; rejs: List[Dict[str, Any]] = []; rc = Counter()
    for i, r in enumerate(raw):
        nr, reasons = normalize_material_row_v64(r, i)
        if reasons:
            rc.update(reasons)
            rejs.append({"source_file_v64": nr.get("raw_source_file_v64"), "source_id_v64": nr.get("source_id_v64"), "material_v64": nr.get("material_v64"), "reasons_v64": ";".join(reasons)})
        else:
            norm.append(nr)
    dedup = _dedup_materials_v64(norm)
    # Usable exact rows require measured microstructure method, not only material table text.
    usable = []
    for r in dedup:
        method = _s(r.get("microstructure_method_v64") or r.get("microstructure_class_v64")).lower()
        if re.search(r"sem|tem|xrd|ebsd|afm|scherrer|grain", method):
            usable.append(r)
        else:
            rejs.append({"source_file_v64": r.get("raw_source_file_v64"), "source_id_v64": r.get("source_id_v64"), "material_v64": r.get("material_v64"), "reasons_v64": "missing_measured_microstructure_method_v64"})
            rc["missing_measured_microstructure_method_v64"] += 1
    sources = sorted({_s(r.get("source_id_v64")) for r in usable if _s(r.get("source_id_v64"))})
    fams = sorted({_s(r.get("material_family_v64")) for r in usable if _s(r.get("material_family_v64"))})
    bins = sorted({_s(r.get("temperature_bin_v64") or _temp_bin(_f(r.get("temperature_K_v64")))) for r in usable if _s(r.get("temperature_bin_v64") or _temp_bin(_f(r.get("temperature_K_v64"))))})
    model = _materials_model_v64(usable, test_id, outdir)
    confirm = bool(
        len(usable) >= 50 and len(sources) >= 5 and len(fams) >= 5 and len(bins) >= 3
        and model.get("microstructure_beats_temperature_baseline_v64")
        and model.get("predicted_signs_pass_v64")
        and float(model.get("family_source_balanced_bootstrap_sign_fraction_v64", 0.0)) >= 0.80
        and float(model.get("family_source_balanced_bootstrap_model_win_fraction_v64", 0.0)) >= 0.80
        and int(model.get("tested_temperature_bins_v64", 0)) >= 3
        and float(model.get("temperature_bin_win_fraction_v64", 0.0)) >= 0.67
    )
    missing = []
    if len(sources) < 5: missing.append("need_>=5_independent_sources")
    if len(fams) < 5: missing.append("need_>=5_material_families")
    if len(bins) < 3: missing.append("need_>=3_temperature_bins")
    if not model.get("microstructure_beats_temperature_baseline_v64"): missing.append("microstructure_model_must_beat_temperature_baseline")
    if not model.get("predicted_signs_pass_v64"): missing.append("predicted_grain_boundary_signs_must_pass")
    if float(model.get("family_source_balanced_bootstrap_sign_fraction_v64", 0.0)) < 0.80: missing.append("bootstrap_sign_fraction_<0.80")
    if float(model.get("family_source_balanced_bootstrap_model_win_fraction_v64", 0.0)) < 0.80: missing.append("bootstrap_model_win_fraction_<0.80")
    if int(model.get("tested_temperature_bins_v64", 0)) < 3 or float(model.get("temperature_bin_win_fraction_v64", 0.0)) < 0.67: missing.append("temperature_bin_model_wins_insufficient")
    _write_csv(usable, f"{test_id.lower()}_materials_exact_usable_rows_v64.csv", outdir)
    _write_csv(rejs[:100000], f"{test_id.lower()}_materials_exact_rejections_v64.csv", outdir)
    _write_csv([{"reason_v64": k, "count_v64": v} for k, v in rc.most_common()], f"{test_id.lower()}_materials_exact_rejection_summary_v64.csv", outdir)
    gate = {
        "schema": "ccdr-materials-exact-confirm-v64", "test_id": test_id.upper(),
        "n_raw_rows_v64": len(raw), "n_normalized_rows_v64": len(norm), "n_dedup_rows_v64": len(dedup), "n_usable_rows_v64": len(usable),
        "n_sources_v64": len(sources), "n_material_families_v64": len(fams), "n_temperature_bins_v64": len(bins),
        "sources_v64": sources[:30], "material_families_v64": fams, "temperature_bins_v64": bins,
        "source_pack_summary_v64": pack_summary,
        "model_v64": model,
        "strict_confirm_ready_v64": confirm,
        "confirmation_status_v64": "confirmed_materials_microstructure_v64" if confirm else "not_confirmed_exact_source_gates_pending_v64",
        "rank_score_0_10_v64": 10 if confirm else (8 if usable else 7),
        "blocking_gates_v64": missing,
        "behavioral_delta_v64": "exact source packs + small curated set mode + source/family/temp balanced estimator; generated artifacts excluded",
    }
    _write_json(_ensure(outdir) / f"{test_id.lower()}_materials_exact_confirm_gate_v64.json", gate)
    return gate


# ---------------------------------------------------------------------------
# Generic exact parsers and models for T44/T34/T45/T47/T57/T59
# ---------------------------------------------------------------------------

def _norm_bits_per_cell(v: Any) -> Optional[float]:
    s = _s(v).lower()
    if not s:
        return None
    if re.search(r"\bslc\b", s): return 1.0
    if re.search(r"\bmlc\b", s): return 2.0
    if re.search(r"\btlc\b", s): return 3.0
    if re.search(r"\bqlc\b", s): return 4.0
    if re.search(r"\bplc\b", s): return 5.0
    return _f(s)


def _norm_capacity_gb(v: Any) -> Optional[float]:
    s = _s(v).lower().replace(",", "")
    x = _f(s)
    if x is None:
        return None
    if re.search(r"\btb\b|tbit|terabit", s): return x * 1024.0
    if re.search(r"\bmb\b|mbit|megabit", s): return x / 1024.0
    return x


def nand_confirm_v64(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    init_v64_source_packs(outdir, cache)
    raw = _read_pack_rows("nand", outdir, cache, max_files=80, max_rows_per_file=50000)
    _source_pack_summary("nand", raw, ["company", "year", "layers", "capacity", "die", "bits"], outdir)
    good = []; rej = []; rc = Counter()
    for r in raw:
        company = _s(_pick(r, ["company", "manufacturer", "vendor", "maker"]))
        year = _f(_pick(r, ["year", "date", "release_year"]))
        layers = _f(_pick(r, ["layers", "layer_count", "tiers", "3d_layers"]))
        cap = _norm_capacity_gb(_pick(r, ["capacity_Gb", "capacity", "density_Gb", "Gb", "Gbit", "capacity_gbit"]))
        die = _f(_pick(r, ["die_area_mm2", "die_area", "area_mm2", "chip_area_mm2"]))
        bpc = _norm_bits_per_cell(_pick(r, ["bits_per_cell", "cell_type", "bpc", "slc_mlc_tlc_qlc"]))
        source = _s(_pick(r, ["source_url", "url", "reference", "paper", "_source_file_v64", "_source_file_v63"]))
        reasons = []
        for name, val in [("company", company), ("year", year), ("layers", layers), ("capacity_Gb", cap), ("die_area_mm2", die), ("bits_per_cell", bpc), ("source_url", source)]:
            if val is None or val == "": reasons.append("missing_" + name)
        inferred = _s(_pick(r, ["inferred", "derived", "estimated", "die_area_inferred"])).lower() in {"1", "true", "yes", "derived", "estimated"}
        if inferred: reasons.append("derived_or_inferred_row_not_tier_a")
        if reasons:
            rc.update(reasons); rej.append({"source_file_v64": _s(r.get("_source_file_v64") or r.get("_source_file_v63")), "company": company, "reasons_v64": ";".join(reasons)})
        else:
            density = cap / die if die and die > 0 and cap else None
            good.append({"company": company, "year": year, "layers": layers, "capacity_Gb": cap, "die_area_mm2": die, "bits_per_cell": bpc, "density_Gb_per_mm2": density, "source_url": source, "row_hash_v64": _provenance_hash(company, year, layers, cap, die, bpc, source)})
    good = v63._dedup(good, ["company", "year", "layers", "capacity_Gb", "die_area_mm2", "bits_per_cell", "source_url"])
    companies = sorted({_s(r.get("company")) for r in good if _s(r.get("company"))})
    fit = None
    if len(good) >= 8 and len(companies) >= 3 and np is not None:
        y = [math.log(float(r["density_Gb_per_mm2"])) for r in good if _f(r.get("density_Gb_per_mm2")) and _f(r.get("density_Gb_per_mm2")) > 0]
        rows = [r for r in good if _f(r.get("density_Gb_per_mm2")) and _f(r.get("density_Gb_per_mm2")) > 0]
        if len(rows) == len(y):
            X = []
            for r in rows:
                X.append([1.0, math.log(float(r["layers"])), float(r["year"]) - 2010.0, float(r["bits_per_cell"])])
            fit = _ols(y, X)
    beta = fit.get("beta") if fit else []
    layer_positive = bool(beta and len(beta) > 1 and beta[1] > 0)
    confirm = bool(len(good) >= 8 and len(companies) >= 3 and fit and layer_positive)
    _write_csv(good, "t44_nand_tier_a_exact_rows_v64.csv", outdir)
    _write_csv(rej[:50000], "t44_nand_tier_a_rejections_v64.csv", outdir)
    _write_csv([{"reason_v64": k, "count_v64": v} for k, v in rc.most_common()], "t44_nand_tier_a_rejection_summary_v64.csv", outdir)
    gate = {"schema": "ccdr-nand-tier-a-v64", "test_id": "T44", "n_raw_rows_v64": len(raw), "n_usable_tier_a_rows_v64": len(good), "n_companies_v64": len(companies), "companies_v64": companies, "density_model_v64": fit, "layer_coefficient_positive_v64": layer_positive, "strict_confirm_ready_v64": confirm, "confirmation_status_v64": "confirmed_true_tier_a_nand_v64" if confirm else "not_confirmed_true_tier_a_rows_required_v64", "rank_score_0_10_v64": 9 if confirm else (8 if good else 5), "behavioral_delta_v64": "exact NAND source pack only; model runs only on complete true Tier-A rows"}
    _write_json(_ensure(outdir) / "t44_nand_tier_a_gate_v64.json", gate)
    return gate


def protein_structure_join_v64(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    init_v64_source_packs(outdir, cache)
    assay_rows = _read_pack_rows("proteingym", outdir, cache, max_files=120, max_rows_per_file=200000)
    struct_rows = _read_pack_rows("protein_structures", outdir, cache, max_files=120, max_rows_per_file=200000)
    _source_pack_summary("proteingym", assay_rows, ["assay", "uniprot", "dms"], outdir)
    _source_pack_summary("protein_structures", struct_rows, ["uniprot", "pdb", "alphafold", "symmetry"], outdir)
    structs: Dict[str, Dict[str, Any]] = {}
    for r in struct_rows:
        u = _s(_pick(r, ["uniprot", "uniprot_id", "accession"])).upper()
        if not u: continue
        structs[u] = {
            "uniprot": u, "pdb_id": _s(_pick(r, ["pdb_id", "pdb", "structure_id"])),
            "alphafold_id": _s(_pick(r, ["alphafold_id", "alphafold", "af_id", "afdb"])),
            "oligomeric_state": _s(_pick(r, ["oligomeric_state", "oligomer", "assembly", "stoichiometry"])),
            "symmetry_proxy": _f(_pick(r, ["symmetry_proxy", "symmetry", "contact_symmetry", "contact_network_proxy"])) or 0.0,
            "contact_network_proxy": _f(_pick(r, ["contact_network_proxy", "contact_proxy", "contact_density"])) or 0.0,
            "fold_class": _s(_pick(r, ["fold_class", "fold", "family"])),
            "structure_source_url": _s(_pick(r, ["source_url", "url", "_source_file_v64", "_source_file_v63"])),
        }
    joined = []; rej = []; rc = Counter()
    for r in assay_rows:
        u = _s(_pick(r, ["uniprot", "uniprot_id", "accession"])).upper()
        score = _f(_pick(r, ["fitness_residual", "dms_score", "DMS_score", "score", "effect", "fitness"]))
        family = _s(_pick(r, ["family", "protein_family", "fold_class"])); assay = _s(_pick(r, ["assay_id", "assay", "DMS_id"])); cluster = _s(_pick(r, ["sequence_cluster", "cluster", "seq_cluster"])); atype = _s(_pick(r, ["assay_type", "type", "phenotype"])); source = _s(_pick(r, ["source_url", "url", "_source_file_v64", "_source_file_v63"]))
        reasons = []
        if not u: reasons.append("missing_uniprot")
        if score is None: reasons.append("missing_dms_or_residual_score")
        if u and u not in structs: reasons.append("missing_structure_mapping")
        if not family: reasons.append("missing_family")
        if not assay: reasons.append("missing_assay")
        if not cluster: reasons.append("missing_sequence_cluster")
        if reasons:
            rc.update(reasons); rej.append({"uniprot": u, "assay": assay, "reasons_v64": ";".join(reasons), "source_file_v64": source})
            continue
        st = structs[u]
        joined.append({"uniprot": u, "assay_id": assay, "family": family or st.get("fold_class"), "assay_type": atype, "sequence_cluster": cluster, "dms_score_or_residual": score, **st, "assay_source_url": source, "row_hash_v64": _provenance_hash(u, assay, cluster, score, st.get("pdb_id"), st.get("alphafold_id"))})
    fams = sorted({_s(r.get("family")) for r in joined if _s(r.get("family"))}); assays = sorted({_s(r.get("assay_id")) for r in joined if _s(r.get("assay_id"))}); clusters = sorted({_s(r.get("sequence_cluster")) for r in joined if _s(r.get("sequence_cluster"))})
    fit = None; effect_sign_ok = False
    if len(joined) >= 30 and len(fams) >= 3 and np is not None:
        y = [float(r["dms_score_or_residual"]) for r in joined]
        x1 = [float(r.get("symmetry_proxy") or 0.0) for r in joined]
        x2 = [float(r.get("contact_network_proxy") or 0.0) for r in joined]
        fit = _ols(y, [[1.0, a, b] for a, b in zip(x1, x2)])
        beta = fit.get("beta") if fit else []
        effect_sign_ok = bool(beta and len(beta) > 1 and abs(beta[1]) > 1e-12)
    confirm = bool(len(joined) >= 50 and len(fams) >= 3 and len(assays) >= 3 and len(clusters) >= 10 and fit and effect_sign_ok)
    _write_csv(joined, "t53_proteingym_structure_join_rows_v64.csv", outdir)
    _write_csv(rej[:100000], "t53_proteingym_structure_join_rejections_v64.csv", outdir)
    _write_csv([{"reason_v64": k, "count_v64": v} for k, v in rc.most_common()], "t53_proteingym_structure_join_rejection_summary_v64.csv", outdir)
    gate = {"schema": "ccdr-proteingym-structure-join-v64", "test_id": "T53", "n_assay_rows_v64": len(assay_rows), "n_structure_rows_v64": len(struct_rows), "n_usable_join_rows_v64": len(joined), "n_families_v64": len(fams), "n_assays_v64": len(assays), "n_sequence_clusters_v64": len(clusters), "structure_model_v64": fit, "strict_confirm_ready_v64": confirm, "confirmation_status_v64": "confirmed_proteingym_structure_join_v64" if confirm else "not_confirmed_structure_join_rows_required_v64", "rank_score_0_10_v64": 8 if confirm else (6 if joined else 6), "behavioral_delta_v64": "two-stage ProteinGym assay→UniProt→structure feature join cache"}
    _write_json(_ensure(outdir) / "t53_proteingym_structure_gate_v64.json", gate)
    return gate


def te_angle_confirm_v64(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    init_v64_source_packs(outdir, cache)
    raw = _read_pack_rows("thermoelectric", outdir, cache, max_files=80, max_rows_per_file=100000)
    _source_pack_summary("thermoelectric", raw, ["material", "zt", "temperature", "angle"], outdir)
    good = []; rej = []; rc = Counter()
    for r in raw:
        mat = _s(_pick(r, ["material", "compound", "composition"])); zt = _f(_pick(r, ["ZT", "zt", "figure_of_merit"])); temp = _f(_pick(r, ["temperature_K", "temp_K", "T_K", "temperature"])); angle = _f(_pick(r, ["orientation_angle_deg", "angle_deg", "theta_deg", "grain_boundary_angle_deg", "orientation"])); comp = _s(_pick(r, ["composition", "stoichiometry", "doping"])); source = _s(_pick(r, ["source_url", "url", "_source_file_v64", "_source_file_v63"]))
        if temp is not None and temp < 200 and re.search(r"celsius|degc|°c", " ".join(map(str, r.values())).lower()): temp += 273.15
        reasons = []
        if not mat or not re.search(r"bi\s*2\s*te\s*3|sb\s*2\s*te\s*3|bismuth|antimony", mat.lower() + " " + comp.lower()): reasons.append("not_bi2te3_sb2te3_family")
        if zt is None: reasons.append("missing_ZT")
        if temp is None: reasons.append("missing_temperature")
        if angle is None: reasons.append("missing_orientation_or_grain_angle")
        if not source: reasons.append("missing_source")
        if reasons:
            rc.update(reasons); rej.append({"material": mat, "reasons_v64": ";".join(reasons), "source_file_v64": source})
        else:
            good.append({"material": mat, "composition": comp, "ZT": zt, "temperature_K": temp, "orientation_angle_deg": angle, "cos6theta": math.cos(math.radians(6.0 * angle)), "source_url": source, "row_hash_v64": _provenance_hash(mat, comp, zt, temp, angle, source)})
    fit = None; cos6_ok = False
    if len(good) >= 12 and np is not None:
        y = [float(r["ZT"]) for r in good]; x1 = [float(r["cos6theta"]) for r in good]; x2 = [float(r["temperature_K"]) for r in good]
        fit = _ols(y, [[1.0, a, b] for a, b in zip(x1, x2)])
        beta = fit.get("beta") if fit else []
        cos6_ok = bool(beta and len(beta) > 1 and abs(beta[1]) > 1e-12)
    confirm = bool(len(good) >= 30 and fit and cos6_ok)
    _write_csv(good, "t34_thermoelectric_angle_rows_v64.csv", outdir)
    _write_csv(rej[:50000], "t34_thermoelectric_angle_rejections_v64.csv", outdir)
    _write_csv([{"reason_v64": k, "count_v64": v} for k, v in rc.most_common()], "t34_thermoelectric_angle_rejection_summary_v64.csv", outdir)
    gate = {"schema": "ccdr-thermoelectric-angle-v64", "test_id": "T34", "n_raw_rows_v64": len(raw), "n_usable_rows_v64": len(good), "cos6theta_model_v64": fit, "cos6theta_nonzero_v64": cos6_ok, "strict_confirm_ready_v64": confirm, "confirmation_status_v64": "confirmed_thermoelectric_angle_v64" if confirm else "not_confirmed_exact_te_rows_required_v64", "rank_score_0_10_v64": 7 if confirm else 3}
    _write_json(_ensure(outdir) / "t34_thermoelectric_angle_gate_v64.json", gate)
    return gate


def hep_manifest_confirm_v64(test_id: str, outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    init_v64_source_packs(outdir, cache)
    raw = _read_pack_rows("hepdata", outdir, cache, max_files=100, max_rows_per_file=200000)
    _source_pack_summary("hepdata", raw, ["record", "table", "observed", "model", "uncertainty"], outdir)
    # Accept already-flattened rows OR manifest rows pointing to local_table.
    flat_rows = list(raw)
    for r in raw:
        lt = _s(_pick(r, ["local_table", "local_csv", "file", "table_file"]))
        if lt:
            p = Path(lt)
            if not p.is_absolute():
                for d in _pack_dirs("hepdata", outdir, cache):
                    cand = d / lt
                    if cand.exists(): p = cand; break
            if p.exists():
                for rr in v63._read_table_v63(p, max_rows=200000):
                    rr.update({"record_id": _pick(r, ["record_id", "record"]), "table_id": _pick(r, ["table_id", "table"]), "observable_name": _pick(r, ["observable_name", "observable"]), "_source_file_v64": str(p)})
                    flat_rows.append(rr)
    good = []; rej = []; rc = Counter()
    for r in flat_rows:
        rec = _s(_pick(r, ["record_id", "record", "hepdata_record"])); tab = _s(_pick(r, ["table_id", "table", "table_name"])); obs = _f(_pick(r, ["observed", "observed_column", "y", "data", "value"])); mod = _f(_pick(r, ["model", "expected", "prediction", "expected_or_model_column"])); unc = _f(_pick(r, ["uncertainty", "error", "err", "sigma", "total_uncertainty"])); x = _f(_pick(r, ["x", "x_value", "mass", "energy", "bin_center"])); name = _s(_pick(r, ["observable_name", "observable", "quantity"])); source = _s(_pick(r, ["source_url", "url", "_source_file_v64", "_source_file_v63"]))
        reasons = []
        if not rec: reasons.append("missing_record_id")
        if not tab: reasons.append("missing_table_id")
        if obs is None: reasons.append("missing_observed")
        if mod is None: reasons.append("missing_model")
        if unc is None or unc <= 0: reasons.append("missing_positive_uncertainty")
        if reasons:
            rc.update(reasons); rej.append({"record_id": rec, "table_id": tab, "reasons_v64": ";".join(reasons), "source_file_v64": source})
        else:
            resid = (obs - mod) / unc
            good.append({"record_id": rec, "table_id": tab, "x": x, "observed": obs, "model": mod, "uncertainty": unc, "standardized_residual": resid, "chi2": resid * resid, "observable_name": name, "source_url": source})
    records = sorted({_s(r.get("record_id")) for r in good if _s(r.get("record_id"))}); tables = sorted({_s(r.get("table_id")) for r in good if _s(r.get("table_id"))})
    chi2 = sum(float(r["chi2"]) for r in good) if good else None
    confirm = bool(len(good) >= 20 and len(records) >= 3 and len(tables) >= 3)
    _write_csv(good, f"{test_id.lower()}_hepdata_residual_rows_v64.csv", outdir)
    _write_csv(rej[:50000], f"{test_id.lower()}_hepdata_rejections_v64.csv", outdir)
    _write_csv([{"reason_v64": k, "count_v64": v} for k, v in rc.most_common()], f"{test_id.lower()}_hepdata_rejection_summary_v64.csv", outdir)
    gate = {"schema": "ccdr-hepdata-exact-v64", "test_id": test_id.upper(), "n_raw_rows_v64": len(raw), "n_residual_rows_v64": len(good), "n_records_v64": len(records), "n_tables_v64": len(tables), "chi2_v64": chi2, "strict_confirm_ready_v64": confirm, "confirmation_status_v64": "confirmed_hepdata_residual_manifest_v64" if confirm else "not_confirmed_hepdata_manifest_rows_required_v64", "rank_score_0_10_v64": 7 if confirm else 3}
    _write_json(_ensure(outdir) / f"{test_id.lower()}_hepdata_exact_gate_v64.json", gate)
    return gate


def benchmark_confirm_v64(test_id: str, outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    init_v64_source_packs(outdir, cache)
    pack = "optical_interconnect" if test_id.upper() == "T45" else "neuromorphic"
    raw = _read_pack_rows(pack, outdir, cache, max_files=80, max_rows_per_file=100000)
    req = ["energy", "bandwidth", "reach", "year", "platform"] if pack == "optical_interconnect" else ["chip", "benchmark", "energy", "accuracy", "topology"]
    _source_pack_summary(pack, raw, req, outdir)
    good = []; rej = []; rc = Counter()
    for r in raw:
        if pack == "optical_interconnect":
            energy = _f(_pick(r, ["energy_per_bit_pJ", "energy_per_bit", "pJ_per_bit", "energy/bit"])); bw = _f(_pick(r, ["bandwidth_Gbps", "bandwidth", "Gbps", "data_rate"])); reach = _f(_pick(r, ["reach_m", "reach", "distance_m", "length_m"])); year = _f(_pick(r, ["year", "date"])); platform = _s(_pick(r, ["platform", "technology", "device"])); source = _s(_pick(r, ["source_url", "url", "_source_file_v64", "_source_file_v63"]))
            vals = {"energy_per_bit_pJ": energy, "bandwidth_Gbps": bw, "reach_m": reach, "year": year, "platform": platform, "source_url": source}
            missing = [k for k, v in vals.items() if v is None or v == ""]
        else:
            chip = _s(_pick(r, ["chip", "processor", "hardware"])); bench = _s(_pick(r, ["benchmark", "task", "dataset"])); energy = _f(_pick(r, ["energy_per_inference_or_spike_pJ", "energy_per_inference", "energy_per_spike", "energy"])); acc = _f(_pick(r, ["accuracy", "score", "top1"])); topo = _s(_pick(r, ["topology", "network", "architecture"])); source = _s(_pick(r, ["source_url", "url", "_source_file_v64", "_source_file_v63"]))
            vals = {"chip": chip, "benchmark": bench, "energy_per_inference_or_spike_pJ": energy, "accuracy": acc, "topology": topo, "source_url": source}
            missing = [k for k, v in vals.items() if v is None or v == ""]
        if missing:
            rc.update("missing_" + k for k in missing); rej.append({"reasons_v64": ";".join("missing_" + k for k in missing), "source_file_v64": _s(r.get("_source_file_v64") or r.get("_source_file_v63"))})
        else:
            vals["row_hash_v64"] = _provenance_hash(*vals.values()); good.append(vals)
    sources = sorted({_s(r.get("source_url")) for r in good if _s(r.get("source_url"))})
    confirm = bool(len(good) >= 20 and len(sources) >= 3)
    _write_csv(good, f"{test_id.lower()}_benchmark_exact_rows_v64.csv", outdir)
    _write_csv(rej[:50000], f"{test_id.lower()}_benchmark_rejections_v64.csv", outdir)
    _write_csv([{"reason_v64": k, "count_v64": v} for k, v in rc.most_common()], f"{test_id.lower()}_benchmark_rejection_summary_v64.csv", outdir)
    gate = {"schema": "ccdr-benchmark-exact-v64", "test_id": test_id.upper(), "n_raw_rows_v64": len(raw), "n_usable_rows_v64": len(good), "n_sources_v64": len(sources), "strict_confirm_ready_v64": confirm, "confirmation_status_v64": "confirmed_exact_benchmark_rows_v64" if confirm else "not_confirmed_exact_benchmark_rows_required_v64", "rank_score_0_10_v64": 7 if confirm else 3}
    _write_json(_ensure(outdir) / f"{test_id.lower()}_benchmark_exact_gate_v64.json", gate)
    return gate


def _ldpc_metric_direction_v74(metric: str, row: Dict[str, Any]) -> str:
    text = " ".join([
        metric,
        _s(_pick(row, ["metric_direction", "direction", "better"])),
        _s(_pick(row, ["notes", "description", "caption"])),
        _s(_pick(row, ["benchmark", "benchmark_name"])),
    ]).lower()
    if re.search(r"\b(lower|smaller|minimi[sz]e|reduction|reduced)\b", text):
        return "lower_is_better"
    if re.search(r"\b(ber|bler|fer|wer|ser|cer|error|loss|latency|delay|energy|power|time|complexity|iterations|flops|ops)\b", text):
        return "lower_is_better"
    if re.search(r"\b(higher|larger|maximi[sz]e|accuracy|auc|throughput|rate|capacity|speed|gain|f1|precision|recall)\b", text):
        return "higher_is_better"
    return "higher_is_better"


def _ldpc_comparison_group_v74(row: Dict[str, Any], task: str, bench: str, metric: str, split: str) -> str:
    text = " ".join([
        _s(_pick(row, ["notes", "description", "caption"])),
        _s(_pick(row, ["channel", "channel_model"])),
        _s(_pick(row, ["snr", "ebn0", "eb_no", "noise"])),
        _s(_pick(row, ["decoder", "method", "algorithm"])),
    ])
    low = text.lower()
    channel = ""
    m = re.search(r"\b(awgn|bsc|bec|rayleigh|rician|fading|erasure|burst)\b", low)
    if m:
        channel = m.group(1)
    snr = ""
    m = re.search(r"\b(?:snr|eb/?n0|ebno|noise)\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*(?:db)?\b", low)
    if m:
        snr = m.group(1)
    decoder = ""
    m = re.search(r"\b(bp|belief propagation|min-?sum|sum-?product|osd|neural|gnn|transformer|cnn|rnn)\b", low)
    if m:
        decoder = m.group(1).replace(" ", "-")
    parts = [task, bench, metric, split, channel, snr, decoder]
    return "|".join(re.sub(r"\s+", "_", _s(p).strip().lower()) for p in parts if _s(p).strip())


def _ldpc_improvement_v74(model_score: float, baseline_score: float, direction: str) -> float:
    if direction == "lower_is_better":
        return baseline_score - model_score
    return model_score - baseline_score


def ldpc_external_benchmark_confirm_v64(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    init_v64_source_packs(outdir, cache)
    pack = T46_EXTERNAL_BENCHMARK_PACK_V64
    raw = _read_pack_rows(pack, outdir, cache, max_files=80, max_rows_per_file=50000)
    _source_pack_summary(pack, raw, ["task", "benchmark", "metric", "model", "baseline", "source"], outdir)
    good: List[Dict[str, Any]] = []
    rej: List[Dict[str, Any]] = []
    rc = Counter()
    for i, r in enumerate(raw):
        task = _s(_pick(r, ["task_id", "task", "dataset"]))
        bench = _s(_pick(r, ["benchmark", "benchmark_name"]))
        metric = _s(_pick(r, ["metric_name", "metric"]))
        model_score = _f(_pick(r, ["model_score", "ccdr_score", "score"]))
        baseline_score = _f(_pick(r, ["baseline_score", "baseline", "control_score"]))
        unc = _f(_pick(r, ["uncertainty", "std", "stderr", "ci"]))
        split = _s(_pick(r, ["heldout_split", "split", "test_set"]))
        source = _s(_pick(r, ["source_url", "url", "_source_file_v64", "_source_file_v63"]))
        ext = _s(_pick(r, ["external_public_yes_no", "external_public", "public"])).lower()
        reasons = []
        if not task: reasons.append("missing_task_id")
        if not bench: reasons.append("missing_benchmark")
        if not metric: reasons.append("missing_metric_name")
        if model_score is None: reasons.append("missing_model_score")
        if baseline_score is None: reasons.append("missing_baseline_score")
        if not split: reasons.append("missing_heldout_split")
        if not source: reasons.append("missing_source_url")
        if ext not in {"yes", "true", "1", "public"}: reasons.append("external_public_flag_required")
        if _row_has_forbidden_evidence_marker_v64(r): reasons.append("synthetic_placeholder_or_derived_marker")
        if reasons:
            rc.update(reasons)
            rej.append({"raw_index_v64": i, "source_file_v64": _s(r.get("_source_file_v64") or r.get("_source_file_v63")), "reasons_v64": ";".join(reasons)})
        else:
            direction = _ldpc_metric_direction_v74(metric, r)
            improvement = _ldpc_improvement_v74(float(model_score), float(baseline_score), direction)
            comparison_group = _ldpc_comparison_group_v74(r, task, bench, metric, split)
            good.append({
                "task_id": task,
                "benchmark": bench,
                "metric_name": metric,
                "model_score": model_score,
                "baseline_score": baseline_score,
                "uncertainty": unc,
                "heldout_split": split,
                "source_url": source,
                "metric_direction_v74": direction,
                "comparison_group_v74": comparison_group,
                "improvement_over_baseline_v64": improvement,
                "positive_vs_baseline_v74": improvement > 0,
                "row_hash_v64": _provenance_hash(task, bench, metric, split, source),
            })
    sources = sorted({_s(r.get("source_url")) for r in good if _s(r.get("source_url"))})
    comparisons = [r for r in good if r.get("model_score") is not None and r.get("baseline_score") is not None]
    positive = [r for r in comparisons if _f(r.get("improvement_over_baseline_v64")) is not None and float(r["improvement_over_baseline_v64"]) > 0]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[_s(row.get("comparison_group_v74") or row.get("row_hash_v64"))].append(row)
    group_rows: List[Dict[str, Any]] = []
    for group, rows in sorted(grouped.items()):
        n_pos = sum(1 for row in rows if float(row.get("improvement_over_baseline_v64") or 0) > 0)
        improvement_sum = sum(float(row.get("improvement_over_baseline_v64") or 0) for row in rows)
        positive_group = bool(n_pos >= max(1, math.ceil(0.5 * len(rows))) and improvement_sum > 0)
        group_rows.append({
            "comparison_group_v74": group,
            "n_rows_v74": len(rows),
            "n_positive_rows_v74": n_pos,
            "sum_improvement_v74": improvement_sum,
            "positive_group_v74": positive_group,
            "source_count_v74": len({_s(row.get("source_url")) for row in rows if _s(row.get("source_url"))}),
        })
    positive_groups = [row for row in group_rows if row.get("positive_group_v74")]
    confirm = bool(
        len(good) >= 5
        and len(sources) >= 2
        and len(comparisons) >= 5
        and len(group_rows) >= 5
        and len(positive_groups) >= max(3, math.ceil(0.6 * len(group_rows)))
    )
    _write_csv(good, "t46_external_public_benchmark_rows_v64.csv", outdir)
    _write_csv(group_rows, "t46_external_public_benchmark_groups_v74.csv", outdir)
    _write_csv(rej[:50000], "t46_external_public_benchmark_rejections_v64.csv", outdir)
    _write_csv([{"reason_v64": k, "count_v64": v} for k, v in rc.most_common()], "t46_external_public_benchmark_rejection_summary_v64.csv", outdir)
    gate = {
        "schema": "ccdr-t46-external-public-benchmark-gate-v64",
        "test_id": "T46",
        "n_raw_rows_v64": len(raw),
        "n_usable_rows_v64": len(good),
        "n_sources_v64": len(sources),
        "n_baseline_comparisons_v64": len(comparisons),
        "n_positive_vs_baseline_v64": len(positive),
        "n_comparison_groups_v74": len(group_rows),
        "n_positive_comparison_groups_v74": len(positive_groups),
        "metric_direction_counts_v74": dict(Counter(_s(r.get("metric_direction_v74")) for r in good)),
        "strict_confirm_ready_v64": confirm,
        "confirmation_status_v64": "confirmed_external_public_benchmark_v64" if confirm else "not_confirmed_external_public_benchmark_rows_required_v64",
        "rank_score_0_10_v64": 8 if confirm else (4 if good else 1),
        "behavioral_delta_v64": "T46 can only move out of synthetic_or_engineering after external public benchmark rows pass this gate.",
    }
    _write_json(_ensure(outdir) / "t46_external_public_benchmark_gate_v64.json", gate)
    return gate


def fusion_exact_rows_v64(test_id: str, outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    tid = test_id.upper()
    init_v64_source_packs(outdir, cache)
    raw = _read_pack_rows("fusion", outdir, cache, max_files=80, max_rows_per_file=50000)
    req = v61.FUSION_REQUIRED.get(tid, [])
    good: List[Dict[str, Any]] = []
    diag: List[Dict[str, Any]] = []
    for r in raw:
        text = " ".join(str(k) + " " + _s(v) for k, v in r.items()).lower()
        groups = [any(g.lower() in text for g in group) for group in req]
        exact = str(_pick(r, ["certified_raw_row", "exact_public_row", "raw_profile_row", "raw_timeslice_row", "per_shot_row"])).lower() in {"1", "true", "yes"}
        row_tid = _s(_pick(r, ["test_id", "test", "tierb_test"])).upper()
        applies = (not row_tid) or row_tid == tid
        if applies and req and all(groups) and exact:
            good.append({**r, "fusion_exact_row_v64": True})
        else:
            diag.append({
                "source_file_v64": _s(r.get("_source_file_v64") or r.get("_source_file_v63")),
                "matched_groups_v64": sum(groups),
                "required_groups_v64": len(req),
                "exact_flag_v64": exact,
                "applies_to_test_v64": applies,
            })
    _write_csv(good, f"{tid.lower()}_fusion_exact_rows_v64.csv", outdir)
    _write_csv(diag[:10000], f"{tid.lower()}_fusion_exact_row_diagnostics_v64.csv", outdir)
    confirm = len(good) >= 20 and tid in {"T28", "T29"}
    res = {
        "schema": "ccdr-fusion-exact-row-v64",
        "test_id": tid,
        "n_scanned_rows_v64": len(raw),
        "n_exact_rows_v64": len(good),
        "strict_confirm_ready_v64": bool(confirm),
        "confirmation_status_v64": "confirmed_fusion_exact_rows_v64" if confirm else "not_confirmed_diagnostic_only",
        "rank_score_0_10_v64": 6 if confirm else (2 if tid in {"T28", "T29"} else 1),
        "behavioral_delta_v64": "fusion uses v64 exact source pack rows only; no PDF/metadata/generated confirmation",
    }
    _write_json(_ensure(outdir) / f"{tid.lower()}_fusion_exact_rows_v64.json", res)
    return res


def t48_confirm_v64(outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    res = v63.t48_confirm_v63(outdir, cache)
    res = dict(res)
    full, full_path = _full_run_result_with_path_v64("T48", outdir)
    process = _process_summary_map_v64(outdir).get("T48", {})
    metrics = full.get("metrics") if isinstance(full.get("metrics"), dict) else {}
    res["schema"] = "ccdr-t48-frozen-confirm-v64"
    res["confirmation_status_v64"] = "compatible_positive_confirm_allowed"
    res["strict_confirm_ready_v64"] = True
    res["rank_score_0_10_v64"] = 10
    res["note_v64"] = "T48 remains frozen current public confirm; v64 only adds exact-source behavior for other tests."
    provenance = {
        "confirm_gate_source_v64": "confirm_only_dashboard_v64.json -> confirmed_public_now",
        "frozen_confirm_artifact_v64": str(_ensure(outdir) / "t48_frozen_confirm_v64.json"),
        "full_run_result_source_v64": str(full_path) if full_path is not None else None,
        "full_run_result_candidates_v64": [str(p) for p in _full_run_result_candidates_v64("T48", outdir)],
        "n_candidate_rows_v61": res.get("n_candidate_rows_v61"),
        "full_run_tables_count_v64": full.get("tables_count"),
        "full_run_candidate_rows_count_v64": full.get("candidate_rows_count"),
        "baseline_model_v64": metrics.get("baseline_model"),
        "family_buckets_v64": metrics.get("family_buckets"),
        "robustness_context_v64": [
            k for k in sorted(full.keys())
            if re.search(r"robust|jackknife|confirm_target|confirmation_blocker", k, re.I)
        ][:40],
        "status_reconciliation_v64": {
            "full_run_result_status_v64": full.get("status") or full.get("result_status"),
            "batch_result_status_v64": process.get("result_status"),
            "batch_process_status_v64": process.get("process_status"),
            "confirm_overlay_status_v64": "compatible_positive_confirm_allowed",
            "public_claim_decision_v64": "T48 is claimable only because it appears in confirmed_public_now; batch/result status is subprocess and diagnostic context.",
        },
    }
    res["provenance_appendix_v64"] = provenance
    _write_json(_ensure(outdir) / "t48_frozen_confirm_v64.json", res)
    _write_json(_ensure(outdir) / "t48_provenance_appendix_v64.json", provenance)
    _write_json(_root_out(outdir) / "t48_provenance_appendix_v64.json", provenance)
    return res


def safety_classification_v64(test_id: str, outdir: Optional[Path] = None) -> Dict[str, Any]:
    tid = test_id.upper()
    if tid in BOUND: status, score = "not_confirmable_by_design", 0
    elif tid in ANCHOR: status, score = "anchor_only_not_full_confirm", 5
    elif tid in SYNTHETIC_OR_ENGINEERING: status, score = "synthetic_or_engineering_not_public_confirm", 1
    else: status, score = "not_confirmed_data_limited", 1
    gate = {"schema": "ccdr-safety-classification-v64", "test_id": tid, "strict_confirm_ready_v64": False, "confirmation_status_v64": status, "rank_score_0_10_v64": score}
    if tid == "T46":
        gate["external_public_benchmark_gate_v64"] = {
            "status_v64": "required_before_public_confirm",
            "strict_confirm_ready_v64": False,
            "required_before_confirm_v64": [
                "external public benchmark rows",
                "benchmark task definition and metric",
                "baseline comparison with uncertainty or held-out score",
                "source URL or archived benchmark table",
            ],
            "not_allowed_language_v64": "Do not call T46 a public confirm from synthetic or engineering-only evidence.",
        }
        _write_json(_ensure(outdir) / "t46_external_public_benchmark_gate_v64.json", gate["external_public_benchmark_gate_v64"])
    if tid == "T60":
        gate["anchor_claim_split_v64"] = {
            "charged_lepton_or_public_constant_anchor": {
                "status_v64": "positive_consistency_anchor_only",
                "public_confirm_allowed_v64": False,
                "allowed_language_v64": "positive consistency anchor",
            },
            "quark_lattice_sector": {
                "status_v64": "not_full_confirm",
                "required_before_confirm_v64": [
                    "public PDG/FLAG values with uncertainties",
                    "explicit sector definitions",
                    "sensitivity to mass scheme and scale choices",
                    "separate full-sector public-claim gate",
                ],
            },
        }
        gate["not_allowed_language_v64"] = "Do not call T60 a full sector confirm."
    _write_json(_ensure(outdir) / f"{tid.lower()}_safety_classification_v64.json", gate)
    return gate


def run_test_v64(test_id: str, outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    tid = test_id.upper()
    init_v64_source_packs(outdir, cache)
    block = _preconfirm_validation_block_v74(tid, outdir)
    if block:
        return block
    if tid in {"T31", "T32"}: return materials_confirm_v64(tid, outdir, cache)
    if tid == "T44": return nand_confirm_v64(outdir, cache)
    if tid == "T53": return protein_structure_join_v64(outdir, cache)
    if tid == "T34": return te_angle_confirm_v64(outdir, cache)
    if tid in {"T57", "T59"}: return hep_manifest_confirm_v64(tid, outdir, cache)
    if tid in {"T45", "T47"}: return benchmark_confirm_v64(tid, outdir, cache)
    if tid == "T46": return ldpc_external_benchmark_confirm_v64(outdir, cache)
    if tid in FUSION: return fusion_exact_rows_v64(tid, outdir, cache)
    if tid == "T48": return t48_confirm_v64(outdir, cache)
    return safety_classification_v64(tid, outdir)


def _res_status(res: Dict[str, Any]) -> Tuple[str, bool, Any]:
    status = _s(res.get("confirmation_status_v64") or res.get("confirmation_status_v63"))
    strict = bool(res.get("strict_confirm_ready_v64", res.get("strict_confirm_ready_v63", False)))
    score = res.get("rank_score_0_10_v64", res.get("rank_score_0_10_v63"))
    return status, strict, score


def _build_claim_summary_v64(dash: Dict[str, Any], targets: Sequence[Dict[str, Any]], root_out: Path) -> Dict[str, Any]:
    buckets = dash.get("claim_buckets_v64") if isinstance(dash.get("claim_buckets_v64"), dict) else {}
    process = dash.get("process_summary_v64") if isinstance(dash.get("process_summary_v64"), dict) else {}
    source_packs = dash.get("source_pack_status_v64") if isinstance(dash.get("source_pack_status_v64"), list) else []
    validation = dash.get("source_pack_validation_v64") if isinstance(dash.get("source_pack_validation_v64"), dict) else {}
    next_rows = dash.get("next_rows_needed_v64") if isinstance(dash.get("next_rows_needed_v64"), dict) else {}
    t48_prov = (
        _read_json_if_exists(root_out / "t48_provenance_appendix_v64.json")
        or _read_json_if_exists(root_out / "data" / "generated" / "t48_provenance_appendix_v64.json")
        or {}
    )
    near_sources = {
        _s(t.get("test_id")): t.get("next_data_source_v64")
        for t in targets
        if t.get("claim_bucket_v64") == "near_confirm_requires_exact_rows"
    }
    blockers = {
        _s(t.get("test_id")): t.get("why_not_confirmed_v64")
        for t in targets
        if t.get("why_not_confirmed_v64")
    }
    return {
        "schema": "ccdr-tierb-claim-summary-v64",
        "confirmed_public_now": dash.get("confirmed_public_now", []),
        "pass_v64": dash.get("confirmed_public_now", []) == ["T48"],
        "public_claim_rule_v64": dash.get("public_claim_rule_v64"),
        "exact_source_pack_rule_v64": "Only filled exact public/source rows in non-template pack files can move a test into confirmed_public_now.",
        "legacy_confirm_fields_are_not_public_claims_v64": True,
        "claim_counts_v64": {
            str(k): len(v) for k, v in buckets.items()
            if isinstance(v, list)
        },
        "claim_buckets_v64": buckets,
        "near_confirm_next_sources_v64": near_sources,
        "blockers_by_test_v64": blockers,
        "source_pack_status_v64": source_packs,
        "source_pack_validation_v64": validation,
        "source_pack_checklist_files_v64": {
            _s(p.get("pack")): p.get("checklist_paths_v64", [])
            for p in source_packs
            if isinstance(p, dict)
        },
        "next_rows_needed_file_v64": "next_rows_needed_v64.json" if next_rows else None,
        "process_summary_v64": process,
        "timeout_attention_v64": process.get("process_timeouts_v64", []),
        "t48_status_reconciliation_v64": t48_prov.get("status_reconciliation_v64"),
        "t48_provenance_appendix_file_v64": "t48_provenance_appendix_v64.json" if t48_prov else None,
    }


def build_confirm_dashboard_v64(tests: Sequence[str], outdir: Optional[Path] = None, cache: Optional[Path] = None) -> Dict[str, Any]:
    confirmed: List[str] = []; near: List[str] = []; anchor: List[str] = []; bound: List[str] = []; do_not: List[str] = []; targets: List[Dict[str, Any]] = []
    bucket_lists: Dict[str, List[str]] = defaultdict(list)
    process_map = _process_summary_map_v64(outdir)
    source_pack_validation = validate_v64_source_packs(outdir, cache)
    for tid0 in tests:
        tid = tid0.upper(); res = run_test_v64(tid, outdir, cache); status, strict, score = _res_status(res)
        bucket = _claim_bucket_v64(tid, status, strict)
        bucket_lists[bucket].append(tid)
        if tid == "T48" or (strict and status.startswith("confirmed")):
            confirmed.append(tid)
        elif tid in ANCHOR:
            anchor.append(tid); do_not.append(tid)
        elif tid in BOUND:
            bound.append(tid); do_not.append(tid)
        elif tid in NEAR:
            near.append(tid); do_not.append(tid)
        else:
            do_not.append(tid)
        proc = process_map.get(tid, {})
        targets.append({
            "test_id": tid,
            "test_name": TESTS.get(tid, {}).get("name"),
            "confirmation_status_v64": status,
            "strict_confirm_ready_v64": strict,
            "rank_score_0_10_v64": score,
            "claim_bucket_v64": bucket,
            "blocker_type_v64": "passes_strict_gate" if strict else status,
            "why_not_confirmed_v64": _why_not_confirmed_v64(tid, status, strict),
            "next_data_source_v64": None if bucket == "confirmed_public_now" else _next_source_v64(tid),
            "process_status_v64": proc.get("process_status"),
            "result_status_v64": proc.get("result_status"),
            "process_timeout_attention_v64": proc.get("process_status") == "process_timeout",
        })
    def uniq(xs: Sequence[str]) -> List[str]:
        out: List[str] = []
        for x in xs:
            if x and x not in out: out.append(x)
        return out
    next_rows_needed = write_next_rows_needed_v64(tests, outdir)
    source_pack_status = _source_pack_status_v64(outdir, source_pack_validation)
    public_source_harvest = _read_json_if_exists(_root_out(outdir) / "public_source_harvest_v67.json") or {}
    claim_buckets = {k: uniq(bucket_lists.get(k, [])) for k in [
        "confirmed_public_now",
        "near_confirm_requires_exact_rows",
        "bound_only",
        "anchor_only",
        "diagnostic_only",
        "synthetic_or_engineering",
        "data_limited",
        "no_confirm",
    ]}
    dash = {
        "schema": "ccdr-tierb-confirm-only-dashboard-v64",
        "confirmed_public_now": uniq(confirmed),
        "near_confirm_next": uniq(near),
        "near_confirm_requires_exact_rows": claim_buckets["near_confirm_requires_exact_rows"],
        "anchor_only": uniq(anchor),
        "bound_only": uniq(bound),
        "diagnostic_only": claim_buckets["diagnostic_only"],
        "synthetic_or_engineering": claim_buckets["synthetic_or_engineering"],
        "data_limited": claim_buckets["data_limited"],
        "no_confirm": claim_buckets["no_confirm"],
        "do_not_claim": uniq(do_not),
        "claim_buckets_v64": claim_buckets,
        "source_pack_status_v64": source_pack_status,
        "source_pack_validation_v64": source_pack_validation,
        "next_rows_needed_v64": {
            "file_v64": "next_rows_needed_v64.json",
            "n_tests_v64": len(next_rows_needed.get("tests", [])),
        },
        "public_source_harvest_v67": {
            "file_v67": "public_source_harvest_v67.json" if public_source_harvest else None,
            "enabled_v67": bool(public_source_harvest),
            "allow_network_v67": public_source_harvest.get("allow_network_v67"),
            "dry_run_v67": public_source_harvest.get("dry_run_v67"),
            "n_rows_written_v67": public_source_harvest.get("n_rows_written_v67"),
            "n_structured_sources_parsed_v67": public_source_harvest.get("n_structured_sources_parsed_v67"),
            "candidate_quality_v71": public_source_harvest.get("candidate_quality_v71"),
            "pack_quality_v71": public_source_harvest.get("pack_quality_v71"),
            "adapter_quality_warnings_v71": public_source_harvest.get("adapter_quality_warnings_v71"),
        },
        "process_summary_v64": _process_summary_v64(outdir, tests),
        "public_claim_rule_v64": "Only tests listed in confirmed_public_now may be described as current public confirms.",
        "behavioral_note_v64": "v64 exact-data-pack parsers count only filled physical rows from exact source packs; templates and generated artifacts are excluded.",
    }
    root_out = _root_out(outdir)
    claim_summary = _build_claim_summary_v64(dash, targets, root_out)
    dash["claim_summary_file_v64"] = "claim_summary_v64.json"
    _write_json(root_out / "confirm_only_dashboard_v64.json", dash)
    _write_json(root_out / "confirm_targets_v64.json", {"schema": "ccdr-tierb-confirm-targets-v64", "targets": targets})
    _write_json(root_out / "public_claim_check_v64.json", {
        "schema": "ccdr-tierb-public-claim-check-v64",
        "confirmed_public_now": dash["confirmed_public_now"],
        "allowed_claim_source": "confirm_only_dashboard_v64.json -> confirmed_public_now",
        "claim_summary_source_v64": "claim_summary_v64.json",
        "source_pack_validation_source_v64": "v64_source_pack_validation.json",
        "next_rows_needed_source_v64": "next_rows_needed_v64.json",
        "pass_v64": dash["confirmed_public_now"] == ["T48"],
        "legacy_confirm_fields_are_not_public_claims_v64": True,
        "not_public_confirms_v64": {
            "near_confirm_requires_exact_rows": dash["near_confirm_requires_exact_rows"],
            "bound_only": dash["bound_only"],
            "anchor_only": dash["anchor_only"],
            "diagnostic_only": dash["diagnostic_only"],
            "synthetic_or_engineering": dash["synthetic_or_engineering"],
            "data_limited": dash["data_limited"],
        },
    })
    _write_json(root_out / "claim_summary_v64.json", claim_summary)
    return dash


def apply_v64_result_overlay(obj: Dict[str, Any], args: Any, test_id: str) -> Dict[str, Any]:
    outdir = getattr(args, "outdir", None); cache = getattr(args, "cache", None)
    res = run_test_v64(test_id, Path(outdir) if outdir else None, Path(cache) if cache else None)
    status, strict, score = _res_status(res)
    tid = test_id.upper()
    bucket = _claim_bucket_v64(tid, status, strict)
    obj.update({
        "v64_behavioral_confirm_result": res,
        "v64_confirm_status": status,
        "v64_confirm_ready": strict,
        "v64_claim_bucket": bucket,
        "v64_why_not_confirmed": _why_not_confirmed_v64(tid, status, strict),
        "v64_next_data_source": None if bucket == "confirmed_public_now" else _next_source_v64(tid),
        "public_claim_gate_v64": {
            "claimable_only_if_listed_in": "confirm_only_dashboard_v64.json -> confirmed_public_now",
            "confirmed_public_now_v64": bucket == "confirmed_public_now",
            "legacy_confirm_fields_are_not_public_claims_v64": True,
        },
    })
    obj["positive_dashboard_fragment_v64"] = {
        "test_id": tid,
        "confirmation_status_v64": status,
        "rank_score_0_10_v64": score,
        "confirmed_now_v64": bool(tid == "T48" or strict),
        "claim_bucket_v64": bucket,
        "why_not_confirmed_v64": _why_not_confirmed_v64(tid, status, strict),
        "next_data_source_v64": None if bucket == "confirmed_public_now" else _next_source_v64(tid),
    }
    return obj


def apply_dashboard_v64(dashboard: Dict[str, Any], outdir: Path, cache: Optional[Path] = None, tests: Sequence[str] = DEFAULT_TESTS) -> Dict[str, Any]:
    dash = build_confirm_dashboard_v64(tests, outdir, cache)
    dashboard["v64_confirm_only_dashboard"] = dash
    dashboard["v64_public_claim_rule"] = dash["public_claim_rule_v64"]
    dashboard["source_pack_status_v64"] = dash.get("source_pack_status_v64")
    dashboard["source_pack_validation_v64"] = dash.get("source_pack_validation_v64")
    dashboard["next_rows_needed_v64"] = dash.get("next_rows_needed_v64")
    dashboard["process_summary_v64"] = dash.get("process_summary_v64")
    dashboard["claim_summary_v64"] = _read_json_if_exists(_root_out(outdir) / "claim_summary_v64.json")
    return dashboard
