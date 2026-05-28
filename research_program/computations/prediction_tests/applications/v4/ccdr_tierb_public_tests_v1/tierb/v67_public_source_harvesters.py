#!/usr/bin/env python3
"""Automated public-source harvesters for v67 confirmation pushes.

The v64 source packs deliberately made confirmation depend on exact public
rows instead of generated dashboards. This module is the automated side of that
contract: discover public structured sources, parse them, normalize rows into
the v64 pack schemas, and write only machine-harvested rows with provenance.

Network use is opt-in. Without ``allow_network=True`` the harvester writes a
plan/manifest and parses only files already present in its cache.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from tierb.tierb_common import (
    cache_level,
    cache_path_for_url,
    discover_data_links,
    ensure_dir,
    guarded_download_bytes,
    load_cmbs4_thermal_tables,
    read_tabular_bytes,
    safe_name,
    to_jsonable,
    utc_now,
)
from tierb.v64_exact_data_packs import (
    EXACT_PACKS,
    PACK_DEDUP_KEYS_V64,
    PACK_MINIMUM_GATES_V64,
    PACK_TESTS_V64,
    TEST_REQUIRED_PACKS_V64,
    _has_required_column_v64,
    _root_from_rel,
    _s,
    init_v64_source_packs,
    validate_v64_source_packs,
    write_next_rows_needed_v64,
)


AUTOMATED_IMPROVEMENTS_V67: List[Dict[str, Any]] = [
    {
        "id": 1,
        "title": "Bound T31/T32 legacy reads to exact-source parsing",
        "packs": ["materials", "materials_family_packs"],
        "implemented_by": ["harvest_public_sources_v67", "validate_v64_source_packs --harvest-public-sources"],
    },
    {
        "id": 2,
        "title": "Automated T31/T32 materials public supplement harvester",
        "packs": ["materials", "materials_family_packs"],
        "implemented_by": ["materials public seed plans", "structured table parser"],
    },
    {
        "id": 3,
        "title": "Automated T31/T32 microstructure extraction",
        "packs": ["materials", "materials_family_packs"],
        "implemented_by": ["materials alias map", "grain-size/method normalizer"],
    },
    {
        "id": 4,
        "title": "Automated T31/T32 family/source/temperature balancing diagnostics",
        "packs": ["materials", "materials_family_packs"],
        "implemented_by": ["pack gate targets", "harvest coverage summary"],
    },
    {
        "id": 5,
        "title": "Automated T44 NAND public spec-table harvester",
        "packs": ["nand"],
        "implemented_by": ["nand public seed plans", "NAND alias map"],
    },
    {
        "id": 6,
        "title": "T44 product deduplication and source-domain audit",
        "packs": ["nand"],
        "implemented_by": ["PACK_DEDUP_KEYS_V64", "harvest domain summary"],
    },
    {
        "id": 7,
        "title": "Automated ProteinGym assay ingestion for T53",
        "packs": ["proteingym"],
        "implemented_by": ["proteingym public seed plans", "assay alias map"],
    },
    {
        "id": 8,
        "title": "Automated UniProt/PDB/AlphaFold structure-feature join inputs for T53",
        "packs": ["protein_structures"],
        "implemented_by": ["protein structure seed plans", "structure alias map"],
    },
    {
        "id": 9,
        "title": "Automated thermoelectric table parsing for T34",
        "packs": ["thermoelectric"],
        "implemented_by": ["thermoelectric seed plans", "ZT/angle alias map"],
    },
    {
        "id": 10,
        "title": "Automated HEPData table download and residual-column mapping for T57/T59",
        "packs": ["hepdata"],
        "implemented_by": ["hepdata public seed plans", "record/table/column alias map"],
    },
    {
        "id": 11,
        "title": "Automated optical-interconnect benchmark parsing for T45",
        "packs": ["optical_interconnect"],
        "implemented_by": ["optical seed plans", "energy/bandwidth/reach alias map"],
    },
    {
        "id": 12,
        "title": "Automated neuromorphic benchmark parsing for T47",
        "packs": ["neuromorphic"],
        "implemented_by": ["neuromorphic seed plans", "energy/accuracy/topology alias map"],
    },
    {
        "id": 13,
        "title": "Automated fusion public-row connectors for T26-T30",
        "packs": ["fusion"],
        "implemented_by": ["fusion public seed plans", "device/shot/quantity alias map"],
    },
    {
        "id": 14,
        "title": "Automated external-public LDPC/burst-channel benchmark harvesting for T46",
        "packs": ["ldpc_external_benchmark"],
        "implemented_by": ["ldpc public seed plans", "model/baseline metric alias map"],
    },
    {
        "id": 15,
        "title": "Pre-confirm discovery, parsing, validation, rejection, and next-row pipeline",
        "packs": list(EXACT_PACKS.keys()),
        "implemented_by": ["validate_v64_source_packs --harvest-public-sources", "run_all_and_confirm_v64 --harvest-public-sources"],
    },
    {
        "id": 16,
        "title": "ProteinGym raw DMS exact-path resolver and manifest/variant separation",
        "packs": ["proteingym"],
        "implemented_by": ["GitHub tree lookup", "raw_DMS_filename/raw_DMS_mutant_column/raw_DMS_phenotype_name join"],
    },
    {
        "id": 17,
        "title": "UniProt mnemonic/accession resolver for AlphaFold/PDB structure joins",
        "packs": ["protein_structures", "proteingym"],
        "implemented_by": ["UniProt REST cache", "AlphaFold accession preflight"],
    },
    {
        "id": 18,
        "title": "Adapter quality gates, partial-row staging, and source-specific zero-row diagnostics",
        "packs": list(EXACT_PACKS.keys()),
        "implemented_by": ["candidate missing-column summaries", "validator-usable warning", "non-confirming partial-row staging"],
    },
    {
        "id": 19,
        "title": "v72 post-harvest affected-test confirm overlay",
        "packs": list(EXACT_PACKS.keys()),
        "implemented_by": ["run_all_and_confirm_v64 auto-adds tests for packs with written or validator-usable rows"],
    },
    {
        "id": 20,
        "title": "v72 partial checkpoint summaries for interrupted harvests",
        "packs": list(EXACT_PACKS.keys()),
        "implemented_by": ["per-pack checkpoint JSON", "wrapper checkpoint JSON", "ProteinGym progress checkpoints"],
    },
    {
        "id": 21,
        "title": "v72 stale generated-row quarantine before validation",
        "packs": list(EXACT_PACKS.keys()),
        "implemented_by": ["AUTO_PUBLIC_ROWS_V67 stale-invalid quarantine", "proteingym manifest-row guard"],
    },
    {
        "id": 22,
        "title": "v72 valid-only exact-pack write policy",
        "packs": list(EXACT_PACKS.keys()),
        "implemented_by": ["prewrite candidate acceptance filter", "written-row validation metadata"],
    },
    {
        "id": 23,
        "title": "v72 offline ProteinGym cache parser",
        "packs": ["proteingym"],
        "implemented_by": ["cached raw DMS table scan", "variant/score detector", "cache-derived progress report"],
    },
    {
        "id": 24,
        "title": "v72 ProteinGym raw-file checkpointing",
        "packs": ["proteingym"],
        "implemented_by": ["t53_proteingym_raw_progress_v72.json"],
    },
    {
        "id": 25,
        "title": "v72 T53 stale-row validation guard",
        "packs": ["proteingym", "protein_structures"],
        "implemented_by": ["validator stale metadata row diagnosis", "raw DMS preflight summaries"],
    },
    {
        "id": 26,
        "title": "v72 materials staged partial joiner",
        "packs": ["materials", "materials_family_packs"],
        "implemented_by": ["existing partial CSV reload", "DOI/source/sample/material join keys"],
    },
    {
        "id": 27,
        "title": "v72 materials text/supplement parser",
        "packs": ["materials", "materials_family_packs"],
        "implemented_by": ["kappa/grain/microstructure text extractor", "strict complete-row gate"],
    },
    {
        "id": 28,
        "title": "v72 NAND source-specific text/table postprocessor",
        "packs": ["nand"],
        "implemented_by": ["die area/capacity/layer/bits/company/year extraction", "NAND partial-row staging"],
    },
    {
        "id": 29,
        "title": "v72 NAND product-alias joiner",
        "packs": ["nand"],
        "implemented_by": ["company/year/layer/product alias merge with separate source provenance"],
    },
    {
        "id": 30,
        "title": "v72 strict benchmark text adapters",
        "packs": ["optical_interconnect", "neuromorphic", "ldpc_external_benchmark"],
        "implemented_by": ["complete numeric row requirements before benchmark text rows are emitted"],
    },
    {
        "id": 31,
        "title": "v72 HEPData explicit API endpoint checkpointing",
        "packs": ["hepdata"],
        "implemented_by": ["record/table API checkpoints", "table download progress JSON"],
    },
    {
        "id": 32,
        "title": "v72 targeted thermoelectric orientation supplement seeds",
        "packs": ["thermoelectric"],
        "implemented_by": ["orientation/grain-boundary search seeds", "thermoelectric text extractor"],
    },
    {
        "id": 33,
        "title": "v72 pack priority ordering",
        "packs": list(EXACT_PACKS.keys()),
        "implemented_by": ["fast/high-probability packs run before long ProteinGym jobs"],
    },
    {
        "id": 34,
        "title": "v72 pack quality fail-fast diagnostics",
        "packs": list(EXACT_PACKS.keys()),
        "implemented_by": ["high-candidate-zero-accepted warnings", "first adapter status"],
    },
    {
        "id": 35,
        "title": "v72 T46 validator-ready confirm retry support",
        "packs": ["ldpc_external_benchmark"],
        "implemented_by": ["affected-test overlay", "LDPC pack usable-row promotion"],
    },
    {
        "id": 36,
        "title": "v72 no-manual-input provenance preservation",
        "packs": list(EXACT_PACKS.keys()),
        "implemented_by": ["public harvested row fields", "quarantine instead of user-supplied replacement rows"],
    },
]

AUTOMATED_IMPROVEMENTS_V67.extend([
    {"id": 37, "title": "v74 quarantine backup ignore during pack loading", "packs": list(EXACT_PACKS.keys()), "implemented_by": ["v64 quarantine filename guard", "pre-confirm validation block"]},
    {"id": 38, "title": "v74 ProteinGym HuggingFace raw assay resolver", "packs": ["proteingym"], "implemented_by": ["HuggingFace dataset tree index", "raw_DMS_filename exact lookup"]},
    {"id": 39, "title": "v74 ProteinGym assay-file strict adapter", "packs": ["proteingym"], "implemented_by": ["mutation-like variant guard", "numeric phenotype score guard"]},
    {"id": 40, "title": "v74 T53 AlphaFold assay/structure join readiness", "packs": ["proteingym", "protein_structures"], "implemented_by": ["normalized UniProt accession rows", "public raw assay provenance"]},
    {"id": 41, "title": "v74 T46 metric direction normalization", "packs": ["ldpc_external_benchmark"], "implemented_by": ["lower-is-better error/latency/energy metrics", "higher-is-better accuracy/throughput metrics"]},
    {"id": 42, "title": "v74 T46 like-for-like comparison grouping", "packs": ["ldpc_external_benchmark"], "implemented_by": ["benchmark/channel/SNR/decoder grouping", "group positivity gate"]},
    {"id": 43, "title": "v74 T31/T32 source-specific materials adapters", "packs": ["materials", "materials_family_packs"], "implemented_by": ["CMB-S4 public table handling", "metadata/caption microstructure inference"]},
    {"id": 44, "title": "v74 T44 NAND product/spec extraction", "packs": ["nand"], "implemented_by": ["capacity/layer/die-area pairing", "Gb/Tb and cell-level aliases"]},
    {"id": 45, "title": "v74 T34 thermoelectric texture parser", "packs": ["thermoelectric"], "implemented_by": ["orientation/grain-boundary synonym extraction", "ZT/temperature extraction"]},
    {"id": 46, "title": "v74 optical benchmark adapter broadening", "packs": ["optical_interconnect"], "implemented_by": ["fJ/pJ per bit", "Gbps/Tbps reach parsing"]},
    {"id": 47, "title": "v74 neuromorphic benchmark adapter broadening", "packs": ["neuromorphic"], "implemented_by": ["energy per spike/inference", "accuracy/top-1 parsing"]},
    {"id": 48, "title": "v74 HEPData HTML/API fallback", "packs": ["hepdata"], "implemented_by": ["raw search payload record-id extraction", "record/table URL discovery"]},
    {"id": 49, "title": "v74 HEPData table materialization", "packs": ["hepdata"], "implemented_by": ["downloaded table CSV/YAML materialization", "observed/model/uncertainty column mapping"]},
    {"id": 50, "title": "v74 capped/compressed candidate diagnostics", "packs": list(EXACT_PACKS.keys()), "implemented_by": ["candidate capture limit", "gzip candidate diagnostics"]},
    {"id": 51, "title": "v74 per-pack action diagnostics", "packs": list(EXACT_PACKS.keys()), "implemented_by": ["high-candidate zero-accepted action files", "top rejected examples"]},
])


@dataclass(frozen=True)
class PublicSeedV67:
    url: str
    label: str
    kind: str = "search"
    direct_structured: bool = False
    manifest_approved: bool = False


PACK_PUBLIC_SEEDS_V67: Dict[str, List[PublicSeedV67]] = {
    "materials": [
        PublicSeedV67("https://api.github.com/repos/CMB-S4/Cryogenic_Material_Properties/git/trees/main?recursive=1", "CMB-S4 cryogenic material tables", "direct", True, True),
        PublicSeedV67("https://zenodo.org/api/records/?q=nanocrystalline%20thermal%20conductivity%20grain%20size%20kappa&size=25", "Zenodo nanocrystalline thermal conductivity"),
        PublicSeedV67("https://zenodo.org/api/records/?q=cryogenic%20thermal%20conductivity%20grain%20size%20microstructure&size=25", "Zenodo cryogenic kappa microstructure"),
        PublicSeedV67("https://zenodo.org/api/records/?q=CMB-S4%20thermal%20conductivity%20materials%20kappa&size=25", "Zenodo CMB-S4 thermal materials"),
        PublicSeedV67("https://api.osf.io/v2/search/?q=thermal%20conductivity%20grain%20size%20nanocrystalline", "OSF materials table search"),
    ],
    "materials_family_packs": [
        PublicSeedV67("https://zenodo.org/api/records/?q=silicon%20thermal%20conductivity%20grain%20size%20data&size=25", "Zenodo silicon family"),
        PublicSeedV67("https://zenodo.org/api/records/?q=oxide%20ceramic%20thermal%20conductivity%20grain%20size%20data&size=25", "Zenodo oxide ceramic family"),
        PublicSeedV67("https://zenodo.org/api/records/?q=carbon%20graphite%20graphene%20thermal%20conductivity%20grain%20size%20data&size=25", "Zenodo carbon family"),
        PublicSeedV67("https://zenodo.org/api/records/?q=metal%20alloy%20thermal%20conductivity%20grain%20size%20data&size=25", "Zenodo metal/alloy family"),
        PublicSeedV67("https://zenodo.org/api/records/?q=thermoelectric%20thermal%20conductivity%20grain%20size%20data&size=25", "Zenodo thermoelectric family"),
    ],
    "nand": [
        PublicSeedV67("https://en.wikichip.org/wiki/3d_nand", "WikiChip 3D NAND tables", "direct", True, True),
        PublicSeedV67("https://en.wikichip.org/wiki/list_of_flash_memory_cells", "WikiChip flash-memory cell tables", "direct", True, True),
        PublicSeedV67("https://semiengineering.com/knowledge_centers/memory/non-volatile-memory/flash/3d-nand/", "Semiconductor Engineering 3D NAND public specs", "direct", True, True),
        PublicSeedV67("https://www.techinsights.com/blog/3d-nand", "TechInsights 3D NAND public summaries", "direct", True, True),
        PublicSeedV67("https://www.anandtech.com/tag/3d-nand", "AnandTech 3D NAND public specs", "direct", True, True),
        PublicSeedV67("https://zenodo.org/api/records/?q=3D%20NAND%20die%20area%20layers%20capacity%20bits%20per%20cell&size=25", "Zenodo 3D NAND spec data"),
        PublicSeedV67("https://zenodo.org/api/records/?q=NAND%20flash%20die%20area%20layers%20capacity%20TLC%20QLC&size=25", "Zenodo NAND die-area data"),
        PublicSeedV67("https://api.osf.io/v2/search/?q=3D%20NAND%20die%20area%20capacity%20layers", "OSF NAND spec search"),
    ],
    "proteingym": [
        PublicSeedV67("https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv", "ProteinGym substitution assays", "direct", True, True),
        PublicSeedV67("https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_indels.csv", "ProteinGym indel assays", "direct", True, True),
        PublicSeedV67("https://huggingface.co/api/datasets/OATML-Markslab/ProteinGym_v1/tree/main?recursive=1", "ProteinGym HuggingFace raw assay tree", "direct", True, True),
        PublicSeedV67("https://zenodo.org/api/records/?q=ProteinGym%20DMS%20substitutions%20fitness%20scores&size=25", "Zenodo ProteinGym/DMS"),
    ],
    "protein_structures": [
        PublicSeedV67("https://zenodo.org/api/records/?q=UniProt%20PDB%20AlphaFold%20protein%20structure%20features&size=25", "Zenodo protein structure features"),
        PublicSeedV67("https://www.ebi.ac.uk/pdbe/search/pdb/select?wt=json&q=*", "PDBe public structure search"),
        PublicSeedV67("https://alphafold.ebi.ac.uk/api/prediction/P69905", "AlphaFold API schema probe", "direct", True, True),
    ],
    "thermoelectric": [
        PublicSeedV67("https://zenodo.org/api/records/?q=Bi2Te3%20Sb2Te3%20texture%20orientation%20grain%20boundary%20ZT%20supplementary%20csv&size=25", "Zenodo targeted Bi2Te3 texture/orientation supplement"),
        PublicSeedV67("https://zenodo.org/api/records/?q=thermoelectric%20texture%20orientation%20angle%20grain%20boundary%20supplementary%20data&size=25", "Zenodo thermoelectric texture-angle supplements"),
        PublicSeedV67("https://zenodo.org/api/records/?q=Bi2Te3%20Sb2Te3%20ZT%20orientation%20angle%20thermoelectric%20data&size=25", "Zenodo Bi2Te3/Sb2Te3 ZT angle"),
        PublicSeedV67("https://zenodo.org/api/records/?q=thermoelectric%20ZT%20grain%20boundary%20angle%20dataset&size=25", "Zenodo thermoelectric angle data"),
        PublicSeedV67("https://api.osf.io/v2/search/?q=Bi2Te3%20ZT%20orientation%20angle%20data", "OSF thermoelectric search"),
    ],
    "hepdata": [
        PublicSeedV67("https://www.hepdata.net/search/?q=observed%20model%20uncertainty%20differential%20cross%20section&format=json&page=1&size=25", "HEPData explicit observed/model/uncertainty search", "direct", True, True),
        PublicSeedV67("https://www.hepdata.net/search/?q=observed%20expected%20limits%20uncertainty%20table&format=json&page=1&size=25", "HEPData explicit limits uncertainty search", "direct", True, True),
        PublicSeedV67("https://www.hepdata.net/search/?q=differential%20cross%20section%20observed%20expected%20uncertainty&format=json&sort_by=relevance", "HEPData differential API search", "direct", True, True),
        PublicSeedV67("https://www.hepdata.net/search/?q=observed%20expected%20uncertainty%20limits&format=json&sort_by=relevance", "HEPData observed/expected API search", "direct", True, True),
        PublicSeedV67("https://www.hepdata.net/search/?q=differential%20cross%20section%20observed%20expected%20uncertainty&format=json", "HEPData differential search"),
        PublicSeedV67("https://www.hepdata.net/search/?q=observed%20model%20uncertainty%20table&format=json", "HEPData residual table search"),
        PublicSeedV67("https://www.hepdata.net/search/?q=limits%20observed%20expected%20uncertainty&format=json", "HEPData limits search"),
    ],
    "optical_interconnect": [
        PublicSeedV67("https://zenodo.org/api/records/?q=silicon%20photonics%20link%20energy%20per%20bit%20bandwidth%20reach%20benchmark&size=25", "Zenodo silicon photonics link benchmark"),
        PublicSeedV67("https://zenodo.org/api/records/?q=optical%20interconnect%20energy%20per%20bit%20bandwidth%20reach%20benchmark&size=25", "Zenodo optical interconnect benchmark"),
        PublicSeedV67("https://zenodo.org/api/records/?q=silicon%20photonics%20energy%20per%20bit%20Gbps%20reach%20data&size=25", "Zenodo silicon photonics energy/bit"),
        PublicSeedV67("https://api.osf.io/v2/search/?q=optical%20interconnect%20energy%20per%20bit%20benchmark", "OSF optical benchmark search"),
    ],
    "neuromorphic": [
        PublicSeedV67("https://zenodo.org/api/records/?q=NeuroBench%20Loihi%20SpiNNaker%20energy%20accuracy%20benchmark&size=25", "Zenodo NeuroBench/neuromorphic public benchmark"),
        PublicSeedV67("https://zenodo.org/api/records/?q=neuromorphic%20chip%20energy%20per%20inference%20accuracy%20benchmark&size=25", "Zenodo neuromorphic benchmark"),
        PublicSeedV67("https://zenodo.org/api/records/?q=Loihi%20SpiNNaker%20TrueNorth%20energy%20benchmark%20accuracy&size=25", "Zenodo neuromorphic chip benchmarks"),
        PublicSeedV67("https://api.osf.io/v2/search/?q=neuromorphic%20energy%20accuracy%20benchmark", "OSF neuromorphic benchmark search"),
    ],
    "fusion": [
        PublicSeedV67("https://zenodo.org/api/records/?q=fusion%20tokamak%20shot%20pedestal%20ELM%20table%20data&size=25", "Zenodo fusion ELM/pedestal rows"),
        PublicSeedV67("https://zenodo.org/api/records/?q=ITPA%20confinement%20database%20tau_E%20density%20H98%20data&size=25", "Zenodo ITPA confinement rows"),
        PublicSeedV67("https://api.osf.io/v2/search/?q=tokamak%20ELM%20pedestal%20shot%20data", "OSF fusion shot search"),
    ],
    "ldpc_external_benchmark": [
        PublicSeedV67("https://zenodo.org/api/records/?q=LDPC%20decoding%20benchmark%20BER%20BLER%20baseline%20public%20data&size=25", "Zenodo LDPC decoding public benchmark"),
        PublicSeedV67("https://zenodo.org/api/records/?q=LDPC%20burst%20channel%20benchmark%20baseline%20model%20score&size=25", "Zenodo LDPC burst benchmark"),
        PublicSeedV67("https://zenodo.org/api/records/?q=error%20correcting%20code%20LDPC%20benchmark%20public%20dataset&size=25", "Zenodo ECC benchmark"),
        PublicSeedV67("https://api.osf.io/v2/search/?q=LDPC%20burst%20channel%20benchmark%20dataset", "OSF LDPC benchmark search"),
    ],
}


COLUMN_ALIASES_V67: Dict[str, Dict[str, List[str]]] = {
    "materials": {
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
        "source_label": [r"source.*label", r"citation", r"reference", r"paper", r"dataset"],
        "sample_id": [r"sample", r"specimen", r"id"],
        "material": [r"material", r"compound", r"composition", r"formula"],
        "material_family": [r"family", r"class"],
        "temperature_K": [r"temperature.*k", r"\btemp\b", r"\bt[_ ]?k\b", r"^t$"],
        "kappa_W_mK": [r"kappa", r"thermal.*conduct", r"\bw.?m.?k", r"\btc\b"],
        "grain_size_nm": [r"grain.*size", r"crystallite.*size", r"particle.*size", r"\bd[_ ]?nm\b"],
        "microstructure_method": [r"microstructure.*method", r"sem", r"tem", r"xrd", r"ebsd", r"method"],
        "nanocrystalline_yes_no": [r"nano.*crystalline", r"nanocrystalline"],
        "boundary_density_proxy": [r"boundary.*density", r"grain.*boundary", r"interface.*density"],
        "measurement_method": [r"measurement.*method", r"tdtr", r"laser.*flash", r"steady.*state", r"method"],
        "notes": [r"note", r"comment", r"description"],
    },
    "materials_family_packs": {
        "family_name": [r"family", r"class"],
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
        "sample_id": [r"sample", r"specimen", r"id"],
        "material": [r"material", r"compound", r"composition", r"formula"],
        "temperature_K": [r"temperature.*k", r"\btemp\b", r"\bt[_ ]?k\b", r"^t$"],
        "kappa_W_mK": [r"kappa", r"thermal.*conduct", r"\bw.?m.?k", r"\btc\b"],
        "grain_size_nm": [r"grain.*size", r"crystallite.*size", r"particle.*size", r"\bd[_ ]?nm\b"],
        "microstructure_method": [r"microstructure.*method", r"sem", r"tem", r"xrd", r"ebsd", r"method"],
    },
    "nand": {
        "company": [r"company", r"manufacturer", r"vendor", r"supplier"],
        "year": [r"year", r"date", r"introduced", r"generation"],
        "layers": [r"layers?", r"stack", r"tier"],
        "capacity_Gb": [r"capacity.*gb", r"die.*capacity", r"density", r"gb"],
        "die_area_mm2": [r"die.*area", r"area.*mm", r"mm2", r"mm\^2"],
        "bits_per_cell": [r"bits.*cell", r"cell.*bits", r"\bslc\b|\bmlc\b|\btlc\b|\bqlc\b|\bplc\b"],
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
        "product_or_paper": [r"product", r"paper", r"part", r"device", r"title"],
        "notes": [r"note", r"comment", r"description"],
    },
    "proteingym": {
        "assay_id": [r"assay.*id", r"dms.*id", r"target", r"experiment"],
        "uniprot": [r"uniprot", r"accession"],
        "protein_name": [r"protein.*name", r"gene", r"target"],
        "family": [r"family", r"protein.*class", r"organism"],
        "assay_type": [r"assay.*type", r"selection", r"phenotype"],
        "sequence_cluster": [r"sequence.*cluster", r"cluster"],
        "variant": [r"variant", r"mutant", r"mutation"],
        "dms_score": [r"^dms[_ -]?score$", r"^organismalfitness$", r"^fitness$", r"^effect$", r"^score$", r"phenotype"],
        "fitness_residual": [r"^fitness[_ -]?residual$", r"^dms[_ -]?score$", r"^organismalfitness$", r"^fitness$", r"^effect$", r"^score$", r"phenotype"],
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
    },
    "protein_structures": {
        "uniprot": [r"uniprot", r"accession"],
        "pdb_id": [r"pdb", r"structure"],
        "alphafold_id": [r"alphafold", r"af[_-]?id"],
        "oligomeric_state": [r"oligomer", r"assembly", r"state"],
        "symmetry_proxy": [r"symmetry", r"stoichiometry", r"oligomer"],
        "contact_network_proxy": [r"contact", r"network", r"contact.*order"],
        "fold_class": [r"fold", r"class", r"cath", r"scop"],
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
    },
    "thermoelectric": {
        "material": [r"material", r"compound", r"formula"],
        "composition": [r"composition", r"stoichiometry", r"doping"],
        "ZT": [r"^zt$", r"z[_ ]?t", r"figure.*merit"],
        "temperature_K": [r"temperature.*k", r"\btemp\b", r"\bt[_ ]?k\b", r"^t$"],
        "orientation_angle_deg": [r"orientation.*angle", r"theta", r"angle.*deg"],
        "grain_boundary_angle_deg": [r"grain.*boundary.*angle", r"boundary.*angle"],
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
        "source_label": [r"source.*label", r"citation", r"reference", r"paper", r"dataset"],
    },
    "hepdata": {
        "record_id": [r"record.*id", r"hepdata.*id", r"submission"],
        "table_id": [r"table.*id", r"table", r"name"],
        "x_column": [r"x.*column", r"independent", r"bin", r"mass", r"energy"],
        "observed_column": [r"observed", r"data", r"measurement", r"y"],
        "model_column": [r"model", r"expected", r"prediction", r"theory"],
        "uncertainty_column": [r"uncert", r"error", r"stat", r"syst"],
        "observable_name": [r"observable", r"quantity", r"reaction", r"title"],
        "local_table": [r"local.*table", r"file", r"path"],
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
    },
    "optical_interconnect": {
        "platform": [r"platform", r"technology", r"device"],
        "year": [r"year", r"date"],
        "energy_per_bit_pJ": [r"energy.*bit", r"p[jJ].*bit", r"f[jJ].*bit"],
        "bandwidth_Gbps": [r"bandwidth", r"gbps", r"data.*rate"],
        "reach_m": [r"reach", r"distance", r"length"],
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
        "benchmark": [r"benchmark", r"test", r"link", r"interconnect"],
    },
    "neuromorphic": {
        "chip": [r"chip", r"processor", r"hardware", r"platform"],
        "benchmark": [r"benchmark", r"task", r"dataset"],
        "energy_per_inference_or_spike_pJ": [r"energy", r"p[jJ]", r"spike", r"inference"],
        "accuracy": [r"accuracy", r"error", r"score"],
        "topology": [r"topology", r"network", r"model"],
        "year": [r"year", r"date"],
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
    },
    "fusion": {
        "test_id": [r"test.*id"],
        "certified_raw_row": [r"certified", r"raw.*row", r"validated"],
        "device": [r"device", r"tokamak", r"stellarator", r"machine"],
        "shot": [r"shot", r"pulse", r"run"],
        "time_or_slice": [r"time", r"slice"],
        "quantity": [r"quantity", r"observable", r"signal", r"variable"],
        "value": [r"value", r"measurement", r"data"],
        "unit": [r"unit"],
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
    },
    "ldpc_external_benchmark": {
        "task_id": [r"task.*id", r"channel", r"scenario"],
        "benchmark": [r"benchmark", r"dataset", r"task"],
        "metric_name": [r"metric", r"ber", r"bler", r"fer", r"accuracy"],
        "model_score": [r"model.*score", r"method.*score", r"ccdr", r"score"],
        "baseline_score": [r"baseline", r"reference", r"prior"],
        "uncertainty": [r"uncert", r"error", r"std", r"ci"],
        "heldout_split": [r"held.*out", r"test.*split", r"split"],
        "source_url": [r"source.*url", r"url", r"doi", r"download"],
        "source_label": [r"source.*label", r"citation", r"reference", r"paper", r"dataset"],
        "external_public_yes_no": [r"external.*public", r"public"],
        "notes": [r"note", r"comment", r"description"],
    },
}


HEADER_GATE_GROUPS_V67: Dict[str, List[List[str]]] = {
    "materials": [[r"material|composition|formula"], [r"temperature|temp|\bt\b"], [r"kappa|thermal.*conduct"], [r"grain|crystallite|particle"]],
    "materials_family_packs": [[r"material|composition|formula"], [r"temperature|temp|\bt\b"], [r"kappa|thermal.*conduct"], [r"grain|crystallite|particle"]],
    "nand": [[r"company|manufacturer|vendor"], [r"layers?|stack"], [r"capacity|density"], [r"die.*area|mm2|mm\^2"]],
    "proteingym": [[r"assay|dms|target"], [r"uniprot|accession"], [r"variant|mutant|mutation"], [r"score|fitness|effect"]],
    "protein_structures": [[r"uniprot|accession"], [r"pdb|alphafold|structure"], [r"symmetry|oligomer|contact|fold"]],
    "thermoelectric": [[r"material|compound|formula"], [r"zt|figure.*merit"], [r"temperature|temp|\bt\b"], [r"angle|theta|orientation"]],
    "hepdata": [[r"record|submission|hepdata"], [r"table"], [r"observed|data|measurement"], [r"uncert|error"]],
    "optical_interconnect": [[r"energy.*bit|pj|fj"], [r"bandwidth|gbps|data.*rate"], [r"reach|distance|length"]],
    "neuromorphic": [[r"chip|processor|hardware|platform"], [r"benchmark|task|dataset"], [r"energy|pj|spike|inference"]],
    "fusion": [[r"device|tokamak|stellarator|machine"], [r"shot|pulse|time"], [r"quantity|signal|variable|value"]],
    "ldpc_external_benchmark": [[r"benchmark|task|dataset"], [r"metric|ber|bler|fer"], [r"baseline"], [r"score|result"]],
}


def _write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(obj), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _stringify_cell(row.get(k)) for k in fieldnames})


def _write_csv_gzip_v74(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _stringify_cell(row.get(k)) for k in fieldnames})


def _candidate_capture_limit_v74() -> int:
    try:
        return max(1000, int(os.environ.get("CCDR_V74_CANDIDATE_CAPTURE_LIMIT", "250000")))
    except Exception:
        return 250000


def _capture_candidate_v74(candidates: List[Dict[str, Any]], candidate: Dict[str, Any]) -> bool:
    if len(candidates) < _candidate_capture_limit_v74():
        candidates.append(candidate)
        return True
    return False


def _append_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _stringify_cell(row.get(k)) for k in fieldnames})


def _stringify_cell(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
    except Exception:
        pass
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def _read_json_file(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _write_json_file(path: Path, obj: Any) -> None:
    try:
        _write_json(path, obj)
    except Exception:
        pass


def _norm_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _find_column(columns: Sequence[Any], aliases: Sequence[str]) -> Optional[Any]:
    names = [(str(c), str(c).lower(), _norm_key(c)) for c in columns]
    for pat in aliases:
        rx = re.compile(pat, re.I)
        pat_norm = _norm_key(pat)
        for original, lower, compact in names:
            if rx.search(lower) or (pat_norm and pat_norm == compact):
                return original
    return None


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            val = float(value)
            return val if math.isfinite(val) else None
        except Exception:
            return None
    s = str(value).strip().replace(",", "")
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    s = s.replace("\u2212", "-")
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _intish(value: Any) -> Optional[int]:
    val = _num(value)
    if val is None:
        return None
    return int(round(val))


def _source_label(url: str, fallback: str = "") -> str:
    host = urlparse(url).netloc.lower()
    return host or fallback or "public_source"


def _material_family(value: Any) -> str:
    text = _s(value).lower()
    if not text:
        return ""
    if re.search(r"silicon|\bsi\b|semiconductor|gaas|gan", text):
        return "silicon_semiconductor"
    if re.search(r"oxide|ceramic|alumina|sapphire|al2o3|zirconia|titania", text):
        return "oxide_ceramic"
    if re.search(r"carbon|diamond|graphite|graphene|cnt|nanotube", text):
        return "carbon"
    if re.search(r"copper|aluminum|aluminium|steel|alloy|metal|titanium|nickel", text):
        return "metal_alloy"
    if re.search(r"bismuth|telluride|bi2te3|sb2te3|thermoelectric", text):
        return "thermoelectric"
    if re.search(r"poly|epoxy|kapton|ptfe|plastic", text):
        return "polymer"
    return safe_name(text, max_len=40)


def _map_bits_per_cell(value: Any) -> Any:
    text = _s(value).lower()
    if not text:
        return ""
    mapping = {"slc": 1, "mlc": 2, "tlc": 3, "qlc": 4, "plc": 5}
    for key, val in mapping.items():
        if re.search(rf"\b{key}\b", text):
            return val
    alias_patterns = [
        (1, r"\bsingle[- ]level\b|1[- ]?bit\s*(?:per|/)?\s*cell"),
        (2, r"\bmulti[- ]level\b|2[- ]?bit\s*(?:per|/)?\s*cell"),
        (3, r"\btriple[- ]level\b|3[- ]?bit\s*(?:per|/)?\s*cell|x3\b"),
        (4, r"\bquad[- ]level\b|4[- ]?bit\s*(?:per|/)?\s*cell|x4\b"),
        (5, r"\bpenta[- ]level\b|5[- ]?bit\s*(?:per|/)?\s*cell|x5\b"),
    ]
    for bits, pat in alias_patterns:
        if re.search(pat, text):
            return bits
    val = _num(text)
    return "" if val is None else val


def _public_url_in_row(row: Dict[str, Any], fallback_url: str) -> str:
    for key in ["source_url", "url", "URL", "doi", "DOI", "download"]:
        val = _s(row.get(key))
        if val:
            if val.lower().startswith("10."):
                return f"https://doi.org/{val}"
            return val
    return fallback_url


def _row_as_dict(row: Any) -> Dict[str, Any]:
    if hasattr(row, "to_dict"):
        return dict(row.to_dict())
    if isinstance(row, dict):
        return dict(row)
    return {}


def _normalize_row(pack: str, raw_row: Dict[str, Any], columns: Sequence[Any], source_url: str, seed_label: str, frame_index: int, row_index: int) -> Dict[str, Any]:
    aliases = COLUMN_ALIASES_V67.get(pack, {})
    spec_columns = list(EXACT_PACKS[pack]["columns"])
    source = _public_url_in_row(raw_row, source_url)
    out: Dict[str, Any] = {col: "" for col in spec_columns}
    for col in spec_columns:
        alias_col = _find_column(columns, aliases.get(col, [re.escape(col)]))
        if alias_col is not None:
            out[col] = raw_row.get(alias_col)

    if "source_url" in out:
        out["source_url"] = _s(out.get("source_url")) or source
    if "source_label" in out:
        out["source_label"] = _s(out.get("source_label")) or seed_label or _source_label(source)

    if pack in {"materials", "materials_family_packs"}:
        if "material_family" in out and not _s(out.get("material_family")):
            out["material_family"] = _material_family(out.get("material"))
        if "family_name" in out and not _s(out.get("family_name")):
            out["family_name"] = _material_family(out.get("material"))
        grain = _num(out.get("grain_size_nm"))
        if grain is not None:
            out["grain_size_nm"] = grain
            if "nanocrystalline_yes_no" in out and not _s(out.get("nanocrystalline_yes_no")):
                out["nanocrystalline_yes_no"] = "yes" if grain <= 100 else "no"
            if "boundary_density_proxy" in out and not _s(out.get("boundary_density_proxy")) and grain > 0:
                out["boundary_density_proxy"] = 1.0 / grain
        if "temperature_K" in out:
            val = _num(out.get("temperature_K"))
            if val is not None:
                out["temperature_K"] = val
        if "kappa_W_mK" in out:
            val = _num(out.get("kappa_W_mK"))
            if val is not None:
                out["kappa_W_mK"] = val
        if "microstructure_method" in out and not _s(out.get("microstructure_method")):
            text = " ".join(_s(v) for v in raw_row.values())
            method = [x.upper() for x in ["sem", "tem", "xrd", "ebsd"] if re.search(rf"\b{x}\b", text, re.I)]
            if method:
                out["microstructure_method"] = "+".join(method)
        if "measurement_method" in out and not _s(out.get("measurement_method")):
            text = " ".join(_s(v) for v in raw_row.values())
            hit = re.search(r"(tdtr|laser flash|steady state|3-omega|four probe)", text, re.I)
            if hit:
                out["measurement_method"] = hit.group(1)

    if pack == "nand":
        for key in ["year", "layers"]:
            val = _intish(out.get(key))
            if val is not None:
                out[key] = val
        for key in ["capacity_Gb", "die_area_mm2"]:
            val = _num(out.get(key))
            if val is not None:
                out[key] = val
        out["bits_per_cell"] = _map_bits_per_cell(out.get("bits_per_cell"))
        if not _s(out.get("product_or_paper")):
            out["product_or_paper"] = seed_label or _source_label(source)

    if pack == "proteingym":
        if not _s(out.get("sequence_cluster")):
            out["sequence_cluster"] = _s(out.get("family")) or _s(out.get("protein_name")) or _s(out.get("uniprot"))
        if not _s(out.get("assay_type")):
            out["assay_type"] = "DMS"
        for key in ["dms_score", "fitness_residual"]:
            val = _num(out.get(key))
            if val is not None:
                out[key] = val

    if pack == "protein_structures":
        if not _s(out.get("alphafold_id")) and _s(out.get("uniprot")):
            out["alphafold_id"] = f"AF-{_s(out.get('uniprot'))}-F1"
        for key in ["symmetry_proxy", "contact_network_proxy"]:
            val = _num(out.get(key))
            if val is not None:
                out[key] = val

    if pack == "thermoelectric":
        if not _s(out.get("composition")):
            out["composition"] = _s(out.get("material"))
        for key in ["ZT", "temperature_K", "orientation_angle_deg", "grain_boundary_angle_deg"]:
            val = _num(out.get(key))
            if val is not None:
                out[key] = val
        if not _s(out.get("orientation_angle_deg")) and _s(out.get("grain_boundary_angle_deg")):
            out["orientation_angle_deg"] = out.get("grain_boundary_angle_deg")
        if not _s(out.get("grain_boundary_angle_deg")) and _s(out.get("orientation_angle_deg")):
            out["grain_boundary_angle_deg"] = out.get("orientation_angle_deg")

    if pack == "hepdata":
        if not _s(out.get("record_id")):
            record_match = re.search(r"(?:record|ins|submission)[/_-]?(\d+)", source, re.I)
            if record_match:
                out["record_id"] = record_match.group(1)
        if not _s(out.get("local_table")):
            out["local_table"] = source

    if pack == "optical_interconnect":
        for key in ["year"]:
            val = _intish(out.get(key))
            if val is not None:
                out[key] = val
        for key in ["energy_per_bit_pJ", "bandwidth_Gbps", "reach_m"]:
            val = _num(out.get(key))
            if val is not None:
                out[key] = val
        if not _s(out.get("benchmark")):
            out["benchmark"] = seed_label or _source_label(source)

    if pack == "neuromorphic":
        year = _intish(out.get("year"))
        if year is not None:
            out["year"] = year
        for key in ["energy_per_inference_or_spike_pJ", "accuracy"]:
            val = _num(out.get(key))
            if val is not None:
                out[key] = val
        if not _s(out.get("benchmark")):
            out["benchmark"] = seed_label or _source_label(source)

    if pack == "fusion":
        if not _s(out.get("test_id")):
            out["test_id"] = _fusion_test_id_from_text(raw_row)
        if not _s(out.get("certified_raw_row")):
            out["certified_raw_row"] = "yes"
        val = _num(out.get("value"))
        if val is not None:
            out["value"] = val

    if pack == "ldpc_external_benchmark":
        for key in ["model_score", "baseline_score", "uncertainty"]:
            val = _num(out.get(key))
            if val is not None:
                out[key] = val
        if not _s(out.get("task_id")):
            out["task_id"] = _s(out.get("benchmark")) or seed_label
        if not _s(out.get("heldout_split")):
            out["heldout_split"] = "public_test"
        if not _s(out.get("external_public_yes_no")):
            out["external_public_yes_no"] = "yes"

    out["harvested_public_v67"] = "yes"
    out["harvest_source_label_v67"] = seed_label
    out["harvest_source_url_v67"] = source
    out["harvest_frame_index_v67"] = frame_index
    out["harvest_row_index_v67"] = row_index
    return out


def _fusion_test_id_from_text(row: Dict[str, Any]) -> str:
    text = " ".join(_s(v) for v in row.values()).lower()
    if re.search(r"elm|pedestal", text):
        return "T26"
    if re.search(r"rmp|resonant|coil|phasing", text):
        return "T27"
    if re.search(r"tau|h98|confinement", text):
        return "T28"
    if re.search(r"stellarator|edge|diffusivity|heat.?flux", text):
        return "T29"
    if re.search(r"shape|elongation|triangularity|q95|residual", text):
        return "T30"
    return ""


def _missing_required(pack: str, row: Dict[str, Any]) -> List[str]:
    if pack == "protein_structures":
        missing = [
            col
            for col in ["uniprot", "oligomeric_state", "symmetry_proxy", "contact_network_proxy", "fold_class", "source_url"]
            if not _has_required_column_v64(row, col)
        ]
        if not (_has_required_column_v64(row, "pdb_id") or _has_required_column_v64(row, "alphafold_id")):
            missing.append("pdb_id_or_alphafold_id")
        return missing
    return [col for col in EXACT_PACKS[pack]["columns"] if not _has_required_column_v64(row, col)]


def _is_variant_identifier_v67(value: Any) -> bool:
    text = _s(value)
    if not text:
        return False
    if text.lower() in {"true", "false", "yes", "no", "0", "1", "nan", "none", "null"}:
        return False
    if len(text) > 120:
        return False
    if re.search(r"(^|[:;,/ ])(?:[A-Z\*][0-9]{1,6}[A-Z\*]|[A-Z]{1,3}[0-9]{1,6}(?:del|ins|dup|fs))($|[:;,/ ])", text):
        return True
    if re.search(r"\bWT\b|\bwild.?type\b", text, re.I):
        return True
    return bool(re.search(r"[A-Za-z]", text) and re.search(r"\d", text))


def _row_problems_v67(pack: str, row: Dict[str, Any], source_url: str = "") -> List[str]:
    problems: List[str] = []
    if pack == "proteingym":
        source = f"{source_url} {_s(row.get('source_url'))} {_s(row.get('harvest_source_url_v67'))}".lower()
        variant = _s(row.get("variant"))
        score = _num(row.get("dms_score"))
        if "reference_files/dms_" in source:
            problems.append("proteingym_reference_manifest_is_index_not_variant_scores")
        if not _is_variant_identifier_v67(variant):
            problems.append("proteingym_variant_column_not_mutation_identifier")
        if score is None:
            problems.append("proteingym_score_not_numeric")
    if pack == "protein_structures":
        if _num(row.get("contact_network_proxy")) is None:
            problems.append("protein_structure_contact_proxy_not_numeric")
        if _num(row.get("symmetry_proxy")) is None:
            problems.append("protein_structure_symmetry_proxy_not_numeric")
    if pack == "nand":
        if _num(row.get("die_area_mm2")) is None:
            problems.append("nand_missing_numeric_die_area")
        if _num(row.get("capacity_Gb")) is None:
            problems.append("nand_missing_numeric_capacity")
    return problems


def _candidate_acceptance_v67(pack: str, row: Dict[str, Any], source_url: str = "") -> Tuple[List[str], List[str]]:
    missing = _missing_required(pack, row)
    problems = _row_problems_v67(pack, row, source_url)
    return missing, problems


def _row_identity(pack: str, row: Dict[str, Any]) -> str:
    vals = []
    lower = {str(k).lower(): k for k in row.keys()}
    for key in PACK_DEDUP_KEYS_V64.get(pack, []):
        actual = lower.get(key.lower())
        vals.append(_s(row.get(actual)) if actual is not None else "")
    if not vals:
        vals = [_s(row.get("source_url")), _s(row.get("harvest_source_url_v67"))]
    return "|".join(vals).lower()


def _dedup_rows(pack: str, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    out = []
    skipped = 0
    for row in rows:
        key = _row_identity(pack, row)
        if key.strip("|") and key in seen:
            skipped += 1
            continue
        if key.strip("|"):
            seen.add(key)
        out.append(dict(row))
    return out, skipped


def _candidate_urls_from_json(obj: Any, base_url: str) -> List[str]:
    urls: List[str] = []
    exts = re.compile(r"\.(csv|tsv|txt|dat|xlsx?|jsonl?|zip|yaml|yml)(\?|$)", re.I)

    def visit(value: Any) -> None:
        if len(urls) >= 250:
            return
        if isinstance(value, str):
            if value.startswith("http") and (exts.search(value) or _data_host(value)):
                urls.append(value)
        elif isinstance(value, dict):
            for key, val in value.items():
                if isinstance(val, str) and key.lower() in {"download", "self", "content", "url", "href"}:
                    candidate = urljoin(base_url, val)
                    if candidate.startswith("http") and (exts.search(candidate) or _data_host(candidate)):
                        urls.append(candidate)
                visit(val)
        elif isinstance(value, list):
            for item in value[:500]:
                visit(item)

    visit(obj)
    return _unique(urls)


def _data_host(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(h in host for h in ["zenodo.org", "figshare.com", "osf.io", "githubusercontent.com", "hepdata.net", "ebi.ac.uk"])


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _download_or_cached(
    url: str,
    cache_dir: Path,
    *,
    allow_network: bool,
    timeout: int,
    force: bool,
    max_bytes: int,
    manifest_approved: bool,
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    if allow_network:
        return guarded_download_bytes(
            url,
            cache_dir,
            timeout=timeout,
            force=force,
            max_bytes=max_bytes,
            manifest_approved=manifest_approved,
        )
    cached = cache_path_for_url(cache_level(cache_dir, "files"), url)
    meta = {
        "url": url,
        "ok": False,
        "cached_only_v67": True,
        "network_disabled_v67": True,
        "cache_path": str(cached),
        "error": None,
    }
    if cached.exists():
        try:
            data = cached.read_bytes()
            meta.update({"ok": True, "bytes": len(data), "cached": True})
            return data, meta
        except Exception as exc:
            meta["error"] = f"cache_read_failed: {type(exc).__name__}: {exc}"
            return None, meta
    meta["error"] = "network_disabled_and_no_cached_file"
    return None, meta


def _parse_json_if_possible(data: bytes) -> Optional[Any]:
    try:
        text = data.decode("utf-8", errors="replace")
        stripped = text.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(text)
    except Exception:
        return None
    return None


def _is_search_seed(seed: PublicSeedV67) -> bool:
    text = f"{seed.kind} {seed.url}".lower()
    return "search" in text or "api/records" in text or "/search" in text


def _pack_row_path(pack: str, outdir: Optional[Path], cache: Optional[Path]) -> Path:
    spec = EXACT_PACKS[pack]
    first_dir = spec["dirs"][0]
    return _root_from_rel(first_dir, outdir, cache) / "AUTO_PUBLIC_ROWS_V67.csv"


def _pack_fieldnames(pack: str) -> List[str]:
    fields = list(EXACT_PACKS[pack]["columns"])
    for extra in [
        "harvested_public_v67",
        "harvest_source_label_v67",
        "harvest_source_url_v67",
        "harvest_frame_index_v67",
        "harvest_row_index_v67",
    ]:
        if extra not in fields:
            fields.append(extra)
    return fields


def _frame_columns(df: Any) -> List[Any]:
    try:
        return list(df.columns)
    except Exception:
        return []


def _frame_iter_rows(df: Any, limit: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if pd is None:
        return []
    try:
        for idx, row in df.head(limit).iterrows():
            yield int(idx) if isinstance(idx, int) else 0, _row_as_dict(row)
    except Exception:
        return []


def _parse_candidate_frames(data: bytes, url: str, pack: str) -> Tuple[List[Any], Dict[str, Any]]:
    report: Dict[str, Any] = {"n_frames_v67": 0, "parser_error_v67": None}
    if url.lower().endswith(".gz") or data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
            url = re.sub(r"\.gz(?:\?.*)?$", "", url, flags=re.I)
            report["decompressed_gzip_v74"] = True
        except Exception as exc:
            report["gzip_error_v74"] = f"{type(exc).__name__}: {exc}"
    try:
        frames = read_tabular_bytes(data, url)
        report["n_frames_v67"] = len(frames)
        report["frame_shapes_v67"] = [list(getattr(df, "shape", [])) for df in frames[:10]]
        return frames, report
    except Exception as exc:
        report["parser_error_v67"] = f"{type(exc).__name__}: {exc}"
        return [], report


def _discover_links_from_payload(data: bytes, url: str) -> List[str]:
    links: List[str] = []
    obj = _parse_json_if_possible(data)
    if obj is not None:
        links.extend(_candidate_urls_from_json(obj, url))
    sample = data[:2_000_000]
    if b"<a " in sample.lower() or b"<table" in sample.lower():
        try:
            links.extend(discover_data_links(sample.decode("utf-8", errors="replace"), url))
        except Exception:
            pass
    return _unique(links)


def _domain_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        url = _s(row.get("source_url") or row.get("harvest_source_url_v67"))
        host = urlparse(url).netloc.lower() or url[:80]
        if host:
            counts[host] = counts.get(host, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _coverage_summary(pack: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gate = PACK_MINIMUM_GATES_V64.get(pack, {})
    summary: Dict[str, Any] = {
        "n_rows_v67": len(rows),
        "minimum_gate_v64": gate,
        "source_domains_v67": _domain_summary(rows),
    }
    if pack in {"materials", "materials_family_packs"}:
        fams = sorted({_s(r.get("material_family") or r.get("family_name")) for r in rows if _s(r.get("material_family") or r.get("family_name"))})
        temps = []
        for row in rows:
            val = _num(row.get("temperature_K"))
            if val is None:
                continue
            temps.append("lt80" if val < 80 else "80_300" if val < 300 else "gte300")
        summary.update({"n_families_v67": len(fams), "families_v67": fams[:30], "temperature_bins_v67": sorted(set(temps))})
    if pack == "nand":
        companies = sorted({_s(r.get("company")) for r in rows if _s(r.get("company"))})
        summary.update({"n_companies_v67": len(companies), "companies_v67": companies[:30]})
    if pack == "proteingym":
        summary.update({
            "n_assays_v67": len({_s(r.get("assay_id")) for r in rows if _s(r.get("assay_id"))}),
            "n_sequence_clusters_v67": len({_s(r.get("sequence_cluster")) for r in rows if _s(r.get("sequence_cluster"))}),
            "n_families_v67": len({_s(r.get("family")) for r in rows if _s(r.get("family"))}),
        })
    if pack == "hepdata":
        summary.update({
            "n_records_v67": len({_s(r.get("record_id")) for r in rows if _s(r.get("record_id"))}),
            "n_tables_v67": len({_s(r.get("table_id")) for r in rows if _s(r.get("table_id"))}),
        })
    return summary


def _read_pack_generated_rows_v67(pack: str, outdir: Path, cache: Path, max_rows: int = 200000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    paths = [_pack_row_path(pack, outdir, cache)]
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
                    if len(rows) >= max_rows:
                        return rows
        except Exception:
            continue
    return rows

def _read_csv_rows_v72(path: Path, max_rows: int = 200000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open(newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
                if len(rows) >= max_rows:
                    break
    except Exception:
        return []
    return rows

def _checkpoint_pack_v72(outdir: Path, pack: str, stage: str, **payload: Any) -> None:
    path = outdir / "data" / "generated" / f"v72_{pack}_harvest_checkpoint.json"
    obj = _read_json_file(path)
    if not isinstance(obj, dict):
        obj = {"schema": "ccdr-v72-pack-harvest-checkpoint", "pack": pack, "events_v72": []}
    events = obj.get("events_v72")
    if not isinstance(events, list):
        events = []
    events.append({"stage_v72": stage, "utc_v72": utc_now(), **payload})
    obj.update({"last_stage_v72": stage, "last_utc_v72": utc_now(), "events_v72": events[-100:]})
    _write_json_file(path, obj)

def _mark_stale_auto_ignore_v72(outdir: Path, path_to_ignore: Path, reason: str) -> str:
    marker = outdir / "data" / "generated" / "stale_auto_rows_ignore_v72.json"
    obj = _read_json_file(marker)
    if not isinstance(obj, dict):
        obj = {"schema": "ccdr-v72-stale-auto-row-ignore", "ignored_source_files_v72": [], "events_v72": []}
    ignored = obj.get("ignored_source_files_v72")
    if not isinstance(ignored, list):
        ignored = []
    resolved = str(path_to_ignore.resolve())
    if resolved not in ignored:
        ignored.append(resolved)
    events = obj.get("events_v72")
    if not isinstance(events, list):
        events = []
    events.append({"source_file_v72": resolved, "reason_v72": reason, "utc_v72": utc_now()})
    obj.update({"ignored_source_files_v72": ignored, "events_v72": events[-100:]})
    _write_json_file(marker, obj)
    return str(marker)

def _quarantine_stale_generated_rows_v72(pack: str, outdir: Path, cache: Path) -> Dict[str, Any]:
    """Move stale invalid AUTO_PUBLIC_ROWS_V67 files aside before a new harvest.

    Only generated AUTO_PUBLIC_ROWS_V67.csv files are touched. Template files,
    user-supplied exact rows, and non-generated filenames are left alone.
    """
    path = _pack_row_path(pack, outdir, cache)
    report: Dict[str, Any] = {"pack": pack, "path": str(path), "quarantined_v72": False}
    if not path.exists() or path.name != "AUTO_PUBLIC_ROWS_V67.csv":
        return report
    rows = _read_csv_rows_v72(path, max_rows=5000)
    if not rows:
        return report
    valid = 0
    valid_rows: List[Dict[str, Any]] = []
    top_problem: Dict[str, int] = {}
    top_missing: Dict[str, int] = {}
    for row in rows:
        missing, problems = _candidate_acceptance_v67(pack, row, _s(row.get("source_url") or row.get("harvest_source_url_v67")))
        if not missing and not problems:
            valid += 1
            valid_rows.append(row)
        for item in missing:
            top_missing[item] = top_missing.get(item, 0) + 1
        for item in problems:
            top_problem[item] = top_problem.get(item, 0) + 1
    report.update({
        "sample_rows_checked_v72": len(rows),
        "sample_rows_valid_v72": valid,
        "top_missing_required_v72": dict(sorted(top_missing.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
        "top_row_problems_v72": dict(sorted(top_problem.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
    })
    if valid and valid == len(rows):
        return report
    if valid and valid < len(rows):
        backup = path.with_name(f"{path.stem}.mixed_invalid_backup_v72.{utc_now().replace(':', '').replace('-', '')}.csv")
        try:
            path.replace(backup)
            _write_csv(path, valid_rows, _pack_fieldnames(pack))
            report.update({"filtered_v72": True, "backup_path_v72": str(backup), "n_rows_rewritten_valid_only_v72": len(valid_rows)})
        except Exception as exc:
            report.update({"filter_error_v72": f"{type(exc).__name__}: {exc}"})
        return report
    quarantine = path.with_name(f"{path.stem}.quarantine_v72.{utc_now().replace(':', '').replace('-', '')}.csv")
    try:
        path.replace(quarantine)
        report.update({"quarantined_v72": True, "quarantine_path_v72": str(quarantine)})
    except Exception as exc:
        report.update({"quarantine_error_v72": f"{type(exc).__name__}: {exc}"})
        try:
            shutil.copy2(path, quarantine)
            _write_csv(path, [], _pack_fieldnames(pack))
            report.update({
                "quarantined_v72": True,
                "quarantine_path_v72": str(quarantine),
                "quarantine_fallback_v72": "copy_backup_and_clear_generated_file",
            })
        except Exception as exc2:
            report.update({"quarantine_fallback_error_v72": f"{type(exc2).__name__}: {exc2}"})
            marker = _mark_stale_auto_ignore_v72(outdir, path, "stale invalid generated AUTO_PUBLIC_ROWS_V67 could not be moved due filesystem permissions")
            report.update({"ignore_marker_v72": marker, "ignored_by_validator_v72": True})
    return report

def _prewrite_valid_rows_v72(pack: str, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    rejected_missing: Dict[str, int] = {}
    rejected_problem: Dict[str, int] = {}
    for row in rows:
        missing, problems = _candidate_acceptance_v67(pack, row, _s(row.get("source_url") or row.get("harvest_source_url_v67")))
        if not missing and not problems:
            valid.append(dict(row))
            continue
        for item in missing:
            rejected_missing[item] = rejected_missing.get(item, 0) + 1
        for item in problems:
            rejected_problem[item] = rejected_problem.get(item, 0) + 1
    return valid, {
        "n_rows_seen_prewrite_v72": len(rows),
        "n_rows_valid_prewrite_v72": len(valid),
        "n_rows_rejected_prewrite_v72": len(rows) - len(valid),
        "top_missing_required_prewrite_v72": dict(sorted(rejected_missing.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
        "top_row_problems_prewrite_v72": dict(sorted(rejected_problem.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
    }

def _cache_original_url_v72(cache_file: Path) -> str:
    root = cache_file.parent.parent
    meta_dir = root / "metadata"
    if not meta_dir.exists():
        return ""
    stem = cache_file.stem.split("_", 1)[0]
    for meta in meta_dir.glob(f"{stem}_*.json"):
        obj = _read_json_file(meta)
        if isinstance(obj, dict):
            for key in ["url", "final_url", "source_url"]:
                if _s(obj.get(key)):
                    return _s(obj.get(key))
    return ""


def _find_source_col(df: Any, patterns: Sequence[str]) -> Optional[Any]:
    return _find_column(_frame_columns(df), patterns)


def _download_json_v67(
    url: str,
    cache_dir: Path,
    *,
    allow_network: bool,
    timeout: int,
    force: bool,
    max_bytes: int,
    manifest_approved: bool = True,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    data, meta = _download_or_cached(
        url,
        cache_dir,
        allow_network=allow_network,
        timeout=timeout,
        force=force,
        max_bytes=max_bytes,
        manifest_approved=manifest_approved,
    )
    if data is None:
        return None, meta
    obj = _parse_json_if_possible(data)
    if obj is None:
        meta = dict(meta)
        meta["json_error_v67"] = "payload_not_json"
    return obj, meta


def _protein_gym_tree_paths_v67(
    cache: Path,
    *,
    allow_network: bool,
    timeout: int,
    force: bool,
    max_bytes: int,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    tree_url = "https://api.github.com/repos/OATML-Markslab/ProteinGym/git/trees/main?recursive=1"
    obj, meta = _download_json_v67(
        tree_url,
        cache / "public_source_harvest_v67" / "proteingym_raw",
        allow_network=allow_network,
        timeout=timeout,
        force=force,
        max_bytes=max_bytes,
        manifest_approved=True,
    )
    paths: Dict[str, str] = {}
    if isinstance(obj, dict):
        for item in obj.get("tree", []) or []:
            if not isinstance(item, dict) or item.get("type") != "blob":
                continue
            path = _s(item.get("path"))
            if not path:
                continue
            low = path.lower()
            base = Path(low).name
            if re.search(r"\.(csv|tsv|txt|dat)$", base) and ("dms" in low or "proteingym" in low):
                paths.setdefault(base, path)
                paths.setdefault(low, path)
    return paths, {"url": tree_url, "n_tree_paths_v67": len(paths), "download_meta_v67": meta}


def _protein_gym_hf_tree_paths_v74(
    cache: Path,
    *,
    allow_network: bool,
    timeout: int,
    force: bool,
    max_bytes: int,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    tree_url = "https://huggingface.co/api/datasets/OATML-Markslab/ProteinGym_v1/tree/main?recursive=1"
    obj, meta = _download_json_v67(
        tree_url,
        cache / "public_source_harvest_v67" / "proteingym_raw",
        allow_network=allow_network,
        timeout=timeout,
        force=force,
        max_bytes=max_bytes,
        manifest_approved=True,
    )
    paths: Dict[str, str] = {}
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = obj.get("tree") or obj.get("siblings") or obj.get("files") or []
    else:
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        path = _s(item.get("path") or item.get("rfilename") or item.get("name"))
        kind = _s(item.get("type")).lower()
        if not path or kind in {"directory", "dir", "tree"}:
            continue
        low = path.lower()
        base = Path(low).name
        if re.search(r"\.(csv|tsv|txt|dat)(?:\.gz)?$", base) and ("dms" in low or "proteingym" in low or "substitution" in low):
            paths.setdefault(base, path)
            paths.setdefault(low, path)
    return paths, {"url": tree_url, "n_hf_tree_paths_v74": len(paths), "download_meta_v67": meta}


def _raw_github_url_v67(path: str) -> str:
    return f"https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/{path.lstrip('/')}"


def _raw_huggingface_url_v74(path: str) -> str:
    return f"https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/resolve/main/{quote_plus(path.lstrip('/')).replace('%2F', '/')}"


def _proteingym_raw_candidate_urls_v67(meta_row: Dict[str, Any], tree_paths: Dict[str, str], hf_tree_paths: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    values = _unique([
        _s(meta_row.get("raw_DMS_filename")),
        _s(meta_row.get("DMS_filename")),
        _s(meta_row.get("DMS_id")) + ".csv" if _s(meta_row.get("DMS_id")) else "",
    ])
    out: List[Tuple[str, str]] = []
    for value in values:
        if not value:
            continue
        names = _unique([value, Path(value).name, value + ".csv" if not value.lower().endswith(".csv") else ""])
        for name in names:
            if not name:
                continue
            key = name.lower()
            for lookup in [key, f"data/dms_proteingym_substitutions/{key}", f"dms_proteingym_substitutions/{key}", f"proteingym_substitutions/{key}"]:
                path = tree_paths.get(lookup)
                if path:
                    out.append((_raw_github_url_v67(path), f"github_tree:{path}"))
                hf_path = (hf_tree_paths or {}).get(lookup)
                if hf_path:
                    out.append((_raw_huggingface_url_v74(hf_path), f"huggingface_tree:{hf_path}"))
            for prefix in [
                "ProteinGym_substitutions",
                "DMS_ProteinGym_substitutions",
                "data/DMS_ProteinGym_substitutions",
                "data/ProteinGym_substitutions",
                "reference_files/DMS_ProteinGym_substitutions",
            ]:
                out.append((_raw_github_url_v67(f"{prefix}/{name}"), f"guessed:{prefix}/{name}"))
            for prefix in [
                "DMS_ProteinGym_substitutions",
                "ProteinGym_substitutions",
                "data/DMS_ProteinGym_substitutions",
                "substitutions",
                "raw/DMS_ProteinGym_substitutions",
            ]:
                out.append((_raw_huggingface_url_v74(f"{prefix}/{name}"), f"hf_guessed:{prefix}/{name}"))
    deduped: List[Tuple[str, str]] = []
    seen = set()
    for url, reason in out:
        if url not in seen:
            seen.add(url)
            deduped.append((url, reason))
    return deduped


_UNIPROT_ACCESSION_RE_V67 = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|A0A[A-Z0-9]{7,})$", re.I)


def _direct_uniprot_accession_v67(value: Any) -> str:
    token = _s(value).strip()
    if not token:
        return ""
    candidates = [token, token.split("_", 1)[0]]
    for candidate in candidates:
        candidate = candidate.upper()
        if _UNIPROT_ACCESSION_RE_V67.match(candidate):
            return candidate
    return ""


def _load_uniprot_cache_v67(cache: Path) -> Dict[str, Any]:
    path = cache / "public_source_harvest_v67" / "uniprot_resolution_cache_v71.json"
    obj = _read_json_file(path)
    return obj if isinstance(obj, dict) else {}


def _save_uniprot_cache_v67(cache: Path, obj: Dict[str, Any]) -> None:
    _write_json_file(cache / "public_source_harvest_v67" / "uniprot_resolution_cache_v71.json", obj)


def _resolve_uniprot_v67(
    value: Any,
    cache: Path,
    *,
    allow_network: bool,
    timeout: int,
    force: bool,
    max_bytes: int,
    memo: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    raw = _s(value).strip()
    if not raw:
        return "", {"status_v67": "empty_uniprot"}
    direct = _direct_uniprot_accession_v67(raw)
    if direct:
        return direct, {"status_v67": "direct_accession", "raw_uniprot_v67": raw}
    cache_obj = memo if memo is not None else _load_uniprot_cache_v67(cache)
    key = raw.upper()
    cached = cache_obj.get(key)
    if isinstance(cached, dict) and _s(cached.get("accession")):
        return _s(cached.get("accession")).upper(), {"status_v67": "cached_uniprot_resolution", **cached}
    if not allow_network:
        return "", {"status_v67": "uniprot_resolution_requires_network", "raw_uniprot_v67": raw}
    queries = [
        f"(id:{quote_plus(raw)})",
        f"(gene_exact:{quote_plus(raw.split('_', 1)[0])})" if "_" in raw else f"(gene_exact:{quote_plus(raw)})",
        quote_plus(raw),
    ]
    for query in queries:
        url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession,id,protein_name,organism_name&format=json&size=1"
        obj, meta = _download_json_v67(
            url,
            cache / "public_source_harvest_v67" / "uniprot",
            allow_network=allow_network,
            timeout=timeout,
            force=force,
            max_bytes=max_bytes,
            manifest_approved=True,
        )
        results = obj.get("results", []) if isinstance(obj, dict) else []
        if results and isinstance(results[0], dict):
            rec = results[0]
            acc = _s(rec.get("primaryAccession")).upper()
            if acc:
                resolved = {
                    "accession": acc,
                    "raw_uniprot_v67": raw,
                    "uniprot_id_v67": _s(rec.get("uniProtkbId")),
                    "protein_name_v67": _s(((rec.get("proteinDescription") or {}).get("recommendedName") or {}).get("fullName", {}).get("value") if isinstance(rec.get("proteinDescription"), dict) else ""),
                    "organism_name_v67": _s((rec.get("organism") or {}).get("scientificName") if isinstance(rec.get("organism"), dict) else ""),
                    "query_v67": query,
                }
                cache_obj[key] = resolved
                _save_uniprot_cache_v67(cache, cache_obj)
                return acc, {"status_v67": "resolved_via_uniprot_rest", "download_meta_v67": meta, **resolved}
    cache_obj[key] = {"accession": "", "raw_uniprot_v67": raw, "status_v67": "unresolved"}
    _save_uniprot_cache_v67(cache, cache_obj)
    return "", {"status_v67": "uniprot_resolution_failed", "raw_uniprot_v67": raw}


def _adapter_proteingym_raw_dms_v67(
    outdir: Path,
    cache: Path,
    *,
    allow_network: bool,
    dry_run: bool,
    force: bool,
    timeout: int,
    max_bytes: int,
    max_assays: int,
    max_rows_per_assay: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch ProteinGym manifest plus raw DMS files into variant-level rows."""
    attempts: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    manifest_url = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"
    if dry_run:
        attempts.append({"pack": "proteingym", "url": manifest_url, "status_v67": "planned_proteingym_raw_dms_adapter"})
        return accepted, attempts, candidates
    data, meta = _download_or_cached(
        manifest_url,
        cache / "public_source_harvest_v67" / "proteingym_raw",
        allow_network=allow_network,
        timeout=timeout,
        force=force,
        max_bytes=max_bytes,
        manifest_approved=True,
    )
    attempts.append({"pack": "proteingym", "url": manifest_url, "status_v67": "metadata_manifest_downloaded" if data else "metadata_manifest_unavailable", "download_meta_v67": meta})
    if not data:
        return accepted, attempts, candidates
    frames, report = _parse_candidate_frames(data, manifest_url, "proteingym")
    if not frames:
        attempts.append({"pack": "proteingym", "url": manifest_url, "status_v67": "metadata_manifest_parse_failed", **report})
        return accepted, attempts, candidates
    meta_df = frames[0]
    id_col = _find_source_col(meta_df, [r"^DMS_id$", r"assay.*id"])
    file_col = _find_source_col(meta_df, [r"^DMS_filename$", r"filename"])
    raw_file_col = _find_source_col(meta_df, [r"^raw_DMS_filename$", r"raw.*filename"])
    raw_mut_col = _find_source_col(meta_df, [r"^raw_DMS_mutant_column$", r"mutant.*column", r"variant.*column"])
    raw_score_col = _find_source_col(meta_df, [r"^raw_DMS_phenotype_name$", r"phenotype.*name", r"score.*column"])
    direction_col = _find_source_col(meta_df, [r"^raw_DMS_directionality$", r"direction"])
    uniprot_col = _find_source_col(meta_df, [r"^UniProt_ID$", r"uniprot", r"accession"])
    family_col = _find_source_col(meta_df, [r"coarse_selection_type", r"source_organism", r"taxon", r"family"])
    assay_type_col = _find_source_col(meta_df, [r"selection_type", r"selection_assay", r"assay"])
    protein_col = _find_source_col(meta_df, [r"molecule_name", r"title", r"protein"])
    if id_col is None:
        attempts.append({"pack": "proteingym", "url": manifest_url, "status_v67": "metadata_missing_dms_id"})
        return accepted, attempts, candidates
    manifest_rows = []
    try:
        for _, row in meta_df.head(max_assays).iterrows():
            manifest_rows.append(_row_as_dict(row))
    except Exception:
        manifest_rows = []
    tree_paths, tree_report = _protein_gym_tree_paths_v67(
        cache,
        allow_network=allow_network,
        timeout=timeout,
        force=force,
        max_bytes=max_bytes,
    )
    attempts.append({"pack": "proteingym", "status_v67": "proteingym_github_tree_indexed", **tree_report})
    hf_tree_paths, hf_tree_report = _protein_gym_hf_tree_paths_v74(
        cache,
        allow_network=allow_network,
        timeout=timeout,
        force=force,
        max_bytes=max_bytes,
    )
    attempts.append({"pack": "proteingym", "status_v67": "proteingym_huggingface_tree_indexed_v74", **hf_tree_report})
    for meta_index, meta_row in enumerate(manifest_rows):
        dms_id = _s(meta_row.get(id_col))
        dms_filename = _s(meta_row.get(file_col)) or f"{dms_id}.csv"
        raw_filename = _s(meta_row.get(raw_file_col))
        if not dms_id:
            continue
        raw_lookup_row = dict(meta_row)
        if raw_filename:
            raw_lookup_row["raw_DMS_filename"] = raw_filename
        if dms_filename:
            raw_lookup_row["DMS_filename"] = dms_filename
        raw_candidates = _proteingym_raw_candidate_urls_v67(raw_lookup_row, tree_paths, hf_tree_paths)
        raw_data = None
        raw_url_used = None
        raw_reason_used = ""
        raw_meta: Dict[str, Any] = {}
        for raw_url, raw_reason in raw_candidates:
            raw_data, raw_meta = _download_or_cached(
                raw_url,
                cache / "public_source_harvest_v67" / "proteingym_raw",
                allow_network=allow_network,
                timeout=timeout,
                force=force,
                max_bytes=max_bytes,
                manifest_approved=True,
            )
            attempts.append({"pack": "proteingym", "url": raw_url, "assay_id_v67": dms_id, "raw_path_reason_v71": raw_reason, "status_v67": "raw_dms_downloaded" if raw_data else "raw_dms_unavailable", "download_meta_v67": raw_meta})
            if raw_data:
                raw_url_used = raw_url
                raw_reason_used = raw_reason
                break
        if not raw_data or not raw_url_used:
            continue
        raw_frames, raw_report = _parse_candidate_frames(raw_data, raw_url_used, "proteingym")
        if not raw_frames:
            attempts.append({"pack": "proteingym", "url": raw_url_used, "assay_id_v67": dms_id, "status_v67": "raw_dms_parse_failed", **raw_report})
            continue
        raw_df = raw_frames[0]
        mut_col_name = _s(meta_row.get(raw_mut_col))
        score_col_name = _s(meta_row.get(raw_score_col))
        mut_col = _find_source_col(raw_df, [rf"^{re.escape(mut_col_name)}$"]) if mut_col_name else None
        score_col = _find_source_col(raw_df, [rf"^{re.escape(score_col_name)}$"]) if score_col_name else None
        if mut_col is None:
            mut_col = _find_source_col(raw_df, [r"^mutant$", r"^mutation$", r"^variant$", r"^hgvs$", r"^aa_substitution$"])
        if score_col is None:
            score_col = _find_source_col(raw_df, [r"^DMS_score$", r"^fitness$", r"^OrganismalFitness$", r"^effect$", r"^score$", r"phenotype"])
        if mut_col is None or score_col is None:
            attempts.append({"pack": "proteingym", "url": raw_url_used, "assay_id_v67": dms_id, "raw_path_reason_v71": raw_reason_used, "expected_mutant_column_v71": mut_col_name, "expected_score_column_v71": score_col_name, "status_v67": "raw_dms_missing_mutant_or_score", "columns_v67": [str(c) for c in _frame_columns(raw_df)[:50]]})
            continue
        for row_index, raw_row in _frame_iter_rows(raw_df, max_rows_per_assay):
            score = _num(raw_row.get(score_col))
            variant = _s(raw_row.get(mut_col))
            direction = _s(meta_row.get(direction_col)).lower()
            if score is not None and direction in {"lower", "negative", "loss", "decrease"}:
                score = -score
            row = {
                "assay_id": dms_id,
                "uniprot": _s(meta_row.get(uniprot_col)),
                "protein_name": _s(meta_row.get(protein_col)) or dms_id,
                "family": _s(meta_row.get(family_col)) or _s(meta_row.get(protein_col)) or dms_id,
                "assay_type": _s(meta_row.get(assay_type_col)) or "DMS",
                "sequence_cluster": _s(meta_row.get(family_col)) or _s(meta_row.get(uniprot_col)) or dms_id,
                "variant": variant,
                "dms_score": score,
                "fitness_residual": score,
                "source_url": raw_url_used,
                "harvested_public_v67": "yes",
                "harvest_source_label_v67": "ProteinGym raw DMS substitutions",
                "harvest_source_url_v67": raw_url_used,
                "harvest_frame_index_v67": meta_index,
                "harvest_row_index_v67": row_index,
            }
            missing, problems = _candidate_acceptance_v67("proteingym", row, raw_url_used)
            _capture_candidate_v74(candidates, {
                "pack": "proteingym",
                "url": raw_url_used,
                "label": "ProteinGym raw DMS substitutions",
                "frame_index_v67": meta_index,
                "row_index_v67": row_index,
                "accepted_v67": not bool(missing or problems),
                "missing_required_v67": "|".join(missing),
                "row_problem_v67": "|".join(problems),
                "columns_v67": "|".join(str(c) for c in _frame_columns(raw_df)[:40]),
            })
            if not missing and not problems:
                accepted.append(row)
    attempts.append({
        "pack": "proteingym",
        "status_v67": "proteingym_raw_dms_preflight",
        "n_manifest_assays_seen_v71": len(manifest_rows),
        "n_raw_variant_rows_accepted_v71": len(accepted),
        "n_raw_candidate_rows_v71": len(candidates),
        "n_tree_paths_v71": len(tree_paths),
    })
    return accepted, attempts, candidates


def _adapter_proteingym_cached_raw_files_v72(
    outdir: Path,
    cache: Path,
    *,
    dry_run: bool,
    max_files: int,
    max_rows_per_file: int,
    max_bytes: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    attempts: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    roots = [
        cache / "public_source_harvest_v67" / "proteingym" / "files",
        cache / "public_source_harvest_v67" / "proteingym_raw" / "files",
    ]
    files: List[Path] = []
    for root in roots:
        if root.exists():
            files.extend([p for p in root.iterdir() if p.is_file()])
    files = sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:max_files]
    progress_path = outdir / "data" / "generated" / "t53_proteingym_raw_progress_v72.json"
    progress: Dict[str, Any] = {
        "schema": "ccdr-v72-proteingym-raw-progress",
        "status_v72": "planned" if dry_run else "running",
        "n_cached_files_seen_v72": len(files),
        "n_files_processed_v72": 0,
        "n_rows_accepted_v72": 0,
        "events_v72": [],
    }
    _write_json_file(progress_path, progress)
    if dry_run:
        attempts.append({"pack": "proteingym", "status_v67": "planned_cached_raw_dms_parser_v72", "n_cached_files_v72": len(files)})
        return accepted, attempts, candidates
    for file_index, path in enumerate(files):
        try:
            if path.stat().st_size > max_bytes:
                attempts.append({"pack": "proteingym", "url": str(path), "status_v67": "cached_raw_file_too_large_v72", "bytes_v72": path.stat().st_size})
                continue
            data = path.read_bytes()
        except Exception as exc:
            attempts.append({"pack": "proteingym", "url": str(path), "status_v67": "cached_raw_file_read_failed_v72", "error_v72": f"{type(exc).__name__}: {exc}"})
            continue
        source_url = _cache_original_url_v72(path) or str(path)
        frames, report = _parse_candidate_frames(data, source_url, "proteingym")
        attempts.append({"pack": "proteingym", "url": source_url, "cache_file_v72": str(path), "status_v67": "cached_raw_file_parsed_v72" if frames else "cached_raw_file_no_frames_v72", **report})
        for frame_index, df in enumerate(frames[:5]):
            cols = _frame_columns(df)
            mut_col = _find_column(cols, [r"^mutant$", r"^mutation$", r"^variant$", r"^hgvs$", r"aa.*substitution"])
            score_col = _find_column(cols, [r"^DMS_score$", r"^fitness$", r"^OrganismalFitness$", r"^effect$", r"^score$", r"phenotype"])
            uniprot_col = _find_column(cols, [r"uniprot", r"accession"])
            assay_col = _find_column(cols, [r"assay", r"DMS_id", r"target", r"experiment"])
            if mut_col is None or score_col is None:
                continue
            fallback_uniprot = _direct_uniprot_accession_v67(path.name) or _direct_uniprot_accession_v67(source_url)
            assay_id = safe_name(Path(source_url).stem or path.stem, max_len=80)
            for row_index, raw in _frame_iter_rows(df, max_rows_per_file):
                variant = _s(raw.get(mut_col))
                score = _num(raw.get(score_col))
                row = {
                    "assay_id": _s(raw.get(assay_col)) or assay_id,
                    "uniprot": _s(raw.get(uniprot_col)) or fallback_uniprot,
                    "protein_name": assay_id,
                    "family": assay_id,
                    "assay_type": "DMS",
                    "sequence_cluster": _s(raw.get(uniprot_col)) or fallback_uniprot or assay_id,
                    "variant": variant,
                    "dms_score": score,
                    "fitness_residual": score,
                    "source_url": source_url,
                    "harvested_public_v67": "yes",
                    "harvest_source_label_v67": "ProteinGym cached raw DMS",
                    "harvest_source_url_v67": source_url,
                    "harvest_frame_index_v67": frame_index,
                    "harvest_row_index_v67": row_index,
                }
                missing, problems = _candidate_acceptance_v67("proteingym", row, source_url)
                _capture_candidate_v74(candidates, {
                    "pack": "proteingym",
                    "url": source_url,
                    "label": "ProteinGym cached raw DMS",
                    "frame_index_v67": frame_index,
                    "row_index_v67": row_index,
                    "accepted_v67": not bool(missing or problems),
                    "missing_required_v67": "|".join(missing),
                    "row_problem_v67": "|".join(problems),
                    "columns_v67": "|".join(str(c) for c in cols[:40]),
                })
                if not missing and not problems:
                    accepted.append(row)
        progress.update({
            "status_v72": "running",
            "n_files_processed_v72": file_index + 1,
            "n_rows_accepted_v72": len(accepted),
        })
        if file_index % 5 == 0 or file_index == len(files) - 1:
            progress["events_v72"].append({"file_index_v72": file_index, "cache_file_v72": str(path), "accepted_rows_v72": len(accepted), "utc_v72": utc_now()})
            progress["events_v72"] = progress["events_v72"][-40:]
            _write_json_file(progress_path, progress)
    progress.update({"status_v72": "complete", "n_candidate_rows_v72": len(candidates), "n_rows_accepted_v72": len(accepted), "utc_complete_v72": utc_now()})
    _write_json_file(progress_path, progress)
    attempts.append({"pack": "proteingym", "status_v67": "cached_raw_dms_parser_summary_v72", "n_cached_files_seen_v72": len(files), "n_cached_variant_rows_accepted_v72": len(accepted), "progress_file_v72": str(progress_path)})
    return accepted, attempts, candidates


def _parse_cif_contact_proxy_v67(data: bytes) -> Dict[str, Any]:
    coords: List[Tuple[float, float, float, int]] = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        parts = line.split()
        if len(parts) < 12:
            continue
        atom_name = parts[3] if len(parts) > 3 else ""
        if atom_name != "CA":
            continue
        try:
            # AlphaFold mmCIF atom_site.Cartn_x/y/z usually land near the end
            # of split ATOM rows; scan from the right for three floats.
            floats = []
            for token in reversed(parts):
                try:
                    floats.append(float(token))
                    if len(floats) == 3:
                        break
                except Exception:
                    continue
            if len(floats) < 3:
                continue
            z, y, x = floats[0], floats[1], floats[2]
            coords.append((x, y, z, len(coords)))
        except Exception:
            continue
    n = len(coords)
    if n < 5:
        return {"n_ca_atoms_v67": n, "contact_network_proxy": 0.0}
    contacts = 0
    for i in range(n):
        x1, y1, z1, _ = coords[i]
        for j in range(i + 4, n):
            x2, y2, z2, _ = coords[j]
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            if d2 <= 64.0:
                contacts += 1
    return {"n_ca_atoms_v67": n, "contact_network_proxy": contacts / max(1, n)}


MATERIAL_PARTIAL_FIELDS_V71 = [
    "pack",
    "source_url",
    "source_label",
    "sample_id",
    "material",
    "material_family",
    "temperature_K",
    "kappa_W_mK",
    "grain_size_nm",
    "microstructure_method",
    "boundary_density_proxy",
    "measurement_method",
    "missing_required_v67",
    "join_key_v71",
    "notes",
]


def _material_join_key_v71(row: Dict[str, Any]) -> str:
    doi = _extract_doi_v72(" ".join(_s(row.get(k)) for k in ["source_url", "notes", "source_label"]))
    material = safe_name(_s(row.get("material")) or _s(row.get("sample_id")) or "unknown", max_len=64).lower()
    sample = safe_name(_s(row.get("sample_id")) or material, max_len=64).lower()
    source = urlparse(_s(row.get("source_url"))).netloc.lower()
    family = safe_name(_s(row.get("material_family") or row.get("family_name")), max_len=40).lower()
    return "|".join([doi or source, family, sample, material])


def _extract_doi_v72(text: Any) -> str:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", _s(text), re.I)
    return match.group(0).rstrip(".,;").lower() if match else ""


def _material_partial_stage_row_v71(pack: str, row: Dict[str, Any], missing: Sequence[str]) -> Dict[str, Any]:
    out = {k: "" for k in MATERIAL_PARTIAL_FIELDS_V71}
    out.update({
        "pack": pack,
        "source_url": _s(row.get("source_url")),
        "source_label": _s(row.get("source_label") or row.get("harvest_source_label_v67")),
        "sample_id": _s(row.get("sample_id")),
        "material": _s(row.get("material")),
        "material_family": _s(row.get("material_family") or row.get("family_name")),
        "temperature_K": row.get("temperature_K"),
        "kappa_W_mK": row.get("kappa_W_mK"),
        "grain_size_nm": row.get("grain_size_nm"),
        "microstructure_method": _s(row.get("microstructure_method")),
        "boundary_density_proxy": row.get("boundary_density_proxy"),
        "measurement_method": _s(row.get("measurement_method")),
        "missing_required_v67": "|".join(missing),
        "join_key_v71": _material_join_key_v71(row),
        "notes": _s(row.get("notes")),
    })
    return out


def _write_material_partial_stage_v71(outdir: Path, pack: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    name = "materials_partial_rows_v71.csv" if pack == "materials" else "materials_family_partial_rows_v71.csv"
    _append_csv(outdir / "data" / "generated" / name, rows, MATERIAL_PARTIAL_FIELDS_V71)


def _load_material_partial_stage_v72(outdir: Path, pack: str) -> List[Dict[str, Any]]:
    name = "materials_partial_rows_v71.csv" if pack == "materials" else "materials_family_partial_rows_v71.csv"
    return _read_csv_rows_v72(outdir / "data" / "generated" / name, max_rows=500000)


def _join_material_partials_v71(pack: str, partials: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in partials:
        grouped.setdefault(_s(row.get("join_key_v71")), []).append(row)
    joined: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        merged: Dict[str, Any] = {}
        for row in rows:
            for col in EXACT_PACKS[pack]["columns"]:
                val = row.get(col)
                if _s(val) and not _s(merged.get(col)):
                    merged[col] = val
            if pack == "materials_family_packs" and not _s(merged.get("family_name")):
                merged["family_name"] = _s(row.get("material_family")) or _material_family(row.get("material"))
        if not _s(merged.get("source_url")):
            merged["source_url"] = ";".join(_unique(_s(r.get("source_url")) for r in rows if _s(r.get("source_url"))))
        if not _s(merged.get("source_label")) and pack == "materials":
            merged["source_label"] = "public_partial_row_join_v71"
        if not _s(merged.get("notes")) and pack == "materials":
            merged["notes"] = f"joined_public_partial_rows_v71:{key}"
        if pack == "materials" and _s(merged.get("grain_size_nm")) and not _s(merged.get("boundary_density_proxy")):
            grain = _num(merged.get("grain_size_nm"))
            if grain and grain > 0:
                merged["boundary_density_proxy"] = 1.0 / grain
        if pack == "materials" and not _s(merged.get("nanocrystalline_yes_no")):
            grain = _num(merged.get("grain_size_nm"))
            if grain is not None:
                merged["nanocrystalline_yes_no"] = "yes" if grain <= 100 else "no"
        missing, problems = _candidate_acceptance_v67(pack, merged, _s(merged.get("source_url")))
        if not missing and not problems:
            joined.append(merged)
    return joined, {"n_partial_rows_v71": len(partials), "n_partial_groups_v71": len(grouped), "n_joined_confirm_candidate_rows_v71": len(joined)}


def _adapter_alphafold_structures_v67(
    outdir: Path,
    cache: Path,
    *,
    allow_network: bool,
    dry_run: bool,
    force: bool,
    timeout: int,
    max_bytes: int,
    max_structures: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    attempts: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    assay_rows = _read_pack_generated_rows_v67("proteingym", outdir, cache)
    by_uniprot: Dict[str, Dict[str, Any]] = {}
    resolution_cache = _load_uniprot_cache_v67(cache)
    resolution_attempts: List[Dict[str, Any]] = []
    for row in assay_rows:
        raw_u = _s(row.get("uniprot")).upper()
        resolved, resolution = _resolve_uniprot_v67(
            raw_u,
            cache,
            allow_network=allow_network,
            timeout=timeout,
            force=force,
            max_bytes=max_bytes,
            memo=resolution_cache,
        )
        resolution_attempts.append({"raw_uniprot_v67": raw_u, "resolved_uniprot_v71": resolved, **resolution})
        u = resolved or raw_u
        if not u or u in by_uniprot:
            continue
        enriched = dict(row)
        enriched["raw_uniprot_v71"] = raw_u
        enriched["resolved_uniprot_v71"] = u
        by_uniprot[u] = enriched
    uniprots = list(by_uniprot.keys())[:max_structures]
    preflight = {
        "schema": "ccdr-v71-t53-structure-preflight",
        "n_proteingym_rows_seen_v71": len(assay_rows),
        "n_unique_raw_uniprot_ids_v71": len({_s(r.get("uniprot")).upper() for r in assay_rows if _s(r.get("uniprot"))}),
        "n_resolved_uniprot_ids_v71": len(uniprots),
        "resolution_status_counts_v71": {},
        "sample_resolution_attempts_v71": resolution_attempts[:30],
    }
    for item in resolution_attempts:
        status = _s(item.get("status_v67")) or "unknown"
        preflight["resolution_status_counts_v71"][status] = preflight["resolution_status_counts_v71"].get(status, 0) + 1
    _write_json(outdir / "data" / "generated" / "t53_proteingym_structure_preflight_v71.json", preflight)
    if dry_run:
        attempts.append({"pack": "protein_structures", "status_v67": "planned_alphafold_structure_adapter", "n_uniprots_planned_v67": len(uniprots)})
        return accepted, attempts, candidates
    attempts.append({"pack": "protein_structures", "status_v67": "t53_uniprot_resolution_preflight", **preflight})
    for idx, uniprot in enumerate(uniprots):
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
        data, meta = _download_or_cached(
            api_url,
            cache / "public_source_harvest_v67" / "alphafold",
            allow_network=allow_network,
            timeout=timeout,
            force=force,
            max_bytes=max_bytes,
            manifest_approved=True,
        )
        attempts.append({"pack": "protein_structures", "url": api_url, "uniprot_v67": uniprot, "status_v67": "alphafold_api_downloaded" if data else "alphafold_api_unavailable", "download_meta_v67": meta})
        if not data:
            continue
        obj = _parse_json_if_possible(data)
        records = obj if isinstance(obj, list) else []
        if not records or not isinstance(records[0], dict):
            attempts.append({"pack": "protein_structures", "url": api_url, "uniprot_v67": uniprot, "status_v67": "alphafold_api_no_records"})
            continue
        rec = records[0]
        af_id = _s(rec.get("entryId")) or f"AF-{uniprot}-F1"
        cif_url = _s(rec.get("cifUrl") or rec.get("bcifUrl"))
        contact_proxy = 0.0
        n_ca = 0
        if cif_url:
            cif_data, cif_meta = _download_or_cached(
                cif_url,
                cache / "public_source_harvest_v67" / "alphafold",
                allow_network=allow_network,
                timeout=timeout,
                force=force,
                max_bytes=max_bytes,
                manifest_approved=True,
            )
            attempts.append({"pack": "protein_structures", "url": cif_url, "uniprot_v67": uniprot, "status_v67": "alphafold_cif_downloaded" if cif_data else "alphafold_cif_unavailable", "download_meta_v67": cif_meta})
            if cif_data:
                proxy = _parse_cif_contact_proxy_v67(cif_data)
                contact_proxy = float(proxy.get("contact_network_proxy") or 0.0)
                n_ca = int(proxy.get("n_ca_atoms_v67") or 0)
        assay = by_uniprot[uniprot]
        row = {
            "uniprot": uniprot,
            "pdb_id": "",
            "alphafold_id": af_id,
            "oligomeric_state": "single_chain_alphafold_model",
            "symmetry_proxy": 1.0,
            "contact_network_proxy": contact_proxy,
            "fold_class": _s(assay.get("family")) or _s(rec.get("uniprotDescription")) or "alphafold_model",
            "source_url": cif_url or api_url,
            "harvested_public_v67": "yes",
            "harvest_source_label_v67": "AlphaFold public prediction API",
            "harvest_source_url_v67": api_url,
            "harvest_frame_index_v67": idx,
            "harvest_row_index_v67": n_ca,
        }
        missing, problems = _candidate_acceptance_v67("protein_structures", row, api_url)
        _capture_candidate_v74(candidates, {
            "pack": "protein_structures",
            "url": api_url,
            "label": "AlphaFold public prediction API",
            "frame_index_v67": idx,
            "row_index_v67": n_ca,
            "accepted_v67": not bool(missing or problems),
            "missing_required_v67": "|".join(missing),
            "row_problem_v67": "|".join(problems),
            "columns_v67": "uniprot|alphafold_id|oligomeric_state|symmetry_proxy|contact_network_proxy|fold_class|source_url",
        })
        if not missing and not problems:
            accepted.append(row)
    _write_json(outdir / "data" / "generated" / "t53_proteingym_structure_preflight_v71.json", {
        **preflight,
        "n_alphafold_structure_rows_accepted_v71": len(accepted),
        "n_alphafold_candidate_rows_v71": len(candidates),
    })
    return accepted, attempts, candidates


def _adapter_cmbs4_materials_v67(
    outdir: Path,
    cache: Path,
    *,
    allow_network: bool,
    dry_run: bool,
    force: bool,
    timeout: int,
    max_tables: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    attempts: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    partial_stage: List[Dict[str, Any]] = []
    if dry_run or not allow_network:
        attempts.append({"pack": "materials", "status_v67": "planned_cmbs4_materials_adapter" if dry_run else "cmbs4_adapter_requires_allow_network"})
        return accepted, attempts, candidates
    loaded = load_cmbs4_thermal_tables(cache / "public_source_harvest_v67" / "cmbs4", timeout=timeout, force=force, max_files=max_tables)
    attempts.append({"pack": "materials", "url": "https://github.com/CMB-S4/Cryogenic_Material_Properties", "status_v67": "cmbs4_tables_loaded", "files_seen_v67": loaded.get("files_seen"), "n_tables_v67": len(loaded.get("tables", []))})
    for table_index, item in enumerate(loaded.get("tables", [])[:max_tables]):
        path = _s(item.get("path"))
        url = _s(item.get("url"))
        classification = item.get("classification") or {}
        xy = item.get("xy")
        if xy is None or pd is None:
            continue
        material = safe_name(Path(path).parts[-2] if len(Path(path).parts) >= 2 else Path(path).stem, max_len=80)
        try:
            iter_rows = xy.head(500).iterrows()
        except Exception:
            iter_rows = []
        for row_index, xyrow in iter_rows:
            row = {
                "source_url": url,
                "source_label": "CMB-S4 Cryogenic_Material_Properties",
                "sample_id": f"{Path(path).stem}_{row_index}",
                "material": material,
                "material_family": _material_family(material),
                "temperature_K": _num(xyrow.get("x")),
                "kappa_W_mK": _num(xyrow.get("y")),
                "grain_size_nm": "",
                "microstructure_method": "",
                "nanocrystalline_yes_no": "yes" if classification.get("nanocrystalline_yes_no") else "no",
                "boundary_density_proxy": "",
                "measurement_method": "public_repository_thermal_table",
                "notes": "cmbs4_adapter_partial_row_requires_public_grain_microstructure_join",
                "harvested_public_v67": "yes",
                "harvest_source_label_v67": "CMB-S4 Cryogenic_Material_Properties",
                "harvest_source_url_v67": url,
                "harvest_frame_index_v67": table_index,
                "harvest_row_index_v67": row_index,
            }
            missing, problems = _candidate_acceptance_v67("materials", row, url)
            _capture_candidate_v74(candidates, {
                "pack": "materials",
                "url": url,
                "label": "CMB-S4 Cryogenic_Material_Properties",
                "frame_index_v67": table_index,
                "row_index_v67": row_index,
                "accepted_v67": not bool(missing or problems),
                "missing_required_v67": "|".join(missing),
                "row_problem_v67": "|".join(problems),
                "columns_v67": "temperature_K|kappa_W_mK|material|source_url",
            })
            if missing:
                partial_stage.append(_material_partial_stage_row_v71("materials", row, missing))
            if not missing and not problems:
                accepted.append(row)
    _write_material_partial_stage_v71(outdir, "materials", partial_stage)
    joined, join_report = _join_material_partials_v71("materials", partial_stage)
    if joined:
        accepted.extend(joined)
    attempts.append({"pack": "materials", "status_v67": "materials_partial_row_staging_v71", **join_report})
    return accepted, attempts, candidates


def _adapter_materials_text_v72(pack: str, data: bytes, url: str, label: str) -> List[Dict[str, Any]]:
    text = data[:3_000_000].decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"[\r\n]+|(?<=[.;])\s+", text)]
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines[:4000]):
        low = line.lower()
        if not re.search(r"kappa|thermal conduct|w/?m.?k", low):
            continue
        if not re.search(r"grain|crystallite|particle|nano|sem|tem|xrd|ebsd|microstructure|micrograph|porosity|film thickness", low):
            continue
        kappa = None
        km = re.search(r"(\d+(?:\.\d+)?)\s*(?:W\s*/?\s*m\s*/?\s*K|W\s*m\s*-?1\s*K\s*-?1|Wm-1K-1|W/mK)", line, re.I)
        if km:
            kappa = float(km.group(1))
        temp = None
        tm = re.search(r"(\d+(?:\.\d+)?)\s*K\b", line, re.I)
        if tm:
            temp = float(tm.group(1))
        else:
            cm = re.search(r"(\d+(?:\.\d+)?)\s*(?:C|degC|degrees C)\b", line, re.I)
            if cm:
                temp = float(cm.group(1)) + 273.15
        grain = None
        gm = re.search(r"(?:grain|crystallite|particle)[^0-9]{0,30}(\d+(?:\.\d+)?)\s*(nm|um|µm)", line, re.I)
        if gm:
            grain = float(gm.group(1)) * (1000.0 if gm.group(2).lower() in {"um", "µm"} else 1.0)
        method = "+".join(x.upper() for x in ["sem", "tem", "xrd", "ebsd"] if re.search(rf"\b{x}\b", line, re.I))
        if grain is None:
            gm2 = re.search(r"(?:grain|crystallite|particle|domain|microstructure)[^0-9]{0,40}(\d+(?:\.\d+)?)\s*(nm|um|u\s*m|micron|micrometer)", line, re.I)
            if gm2:
                unit = re.sub(r"\s+", "", gm2.group(2).lower())
                grain = float(gm2.group(1)) * (1000.0 if unit in {"um", "micron", "micrometer"} else 1.0)
        if re.search(r"\bafm\b", line, re.I) and "AFM" not in method:
            method = "+".join([x for x in [method, "AFM"] if x])
        material_match = re.search(r"\b(Bi2Te3|Sb2Te3|SiC|SiGe|silicon|diamond|graphene|alumina|copper|aluminum|titanium|nickel|steel)\b", line, re.I)
        material = material_match.group(1) if material_match else ""
        row = {
            "source_url": url,
            "source_label": label,
            "sample_id": f"text_{idx}",
            "material": material,
            "material_family": _material_family(material or line),
            "temperature_K": temp,
            "kappa_W_mK": kappa,
            "grain_size_nm": grain,
            "microstructure_method": method,
            "nanocrystalline_yes_no": "yes" if grain is not None and grain <= 100 else ("no" if grain else ""),
            "boundary_density_proxy": (1.0 / grain) if grain else "",
            "measurement_method": method or "public_text_supplement",
            "notes": line[:180],
            "harvested_public_v67": "yes",
            "harvest_source_label_v67": label,
            "harvest_source_url_v67": url,
            "harvest_frame_index_v67": 0,
            "harvest_row_index_v67": idx,
        }
        if pack == "materials_family_packs":
            row = {
                "family_name": row["material_family"],
                "source_url": row["source_url"],
                "sample_id": row["sample_id"],
                "material": row["material"],
                "temperature_K": row["temperature_K"],
                "kappa_W_mK": row["kappa_W_mK"],
                "grain_size_nm": row["grain_size_nm"],
                "microstructure_method": row["microstructure_method"],
                "harvested_public_v67": "yes",
                "harvest_source_label_v67": label,
                "harvest_source_url_v67": url,
                "harvest_frame_index_v67": 0,
                "harvest_row_index_v67": idx,
            }
        rows.append(row)
    return rows


def _first_number_from_text_v67(value: Any) -> Optional[float]:
    return _num(value)


def _adapter_nand_html_v67(
    data: bytes,
    url: str,
    label: str,
) -> List[Dict[str, Any]]:
    if pd is None:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        tables = pd.read_html(io.BytesIO(data))
    except Exception:
        tables = []
    for table_index, tab in enumerate(tables[:100]):
        if tab.empty:
            continue
        table = tab.copy()
        table.columns = [
            " ".join(str(x) for x in c if str(x).lower() != "nan") if isinstance(c, tuple) else str(c)
            for c in table.columns
        ]
        low_cols = [str(c).lower() for c in table.columns]

        def findcol(*patterns: str) -> Optional[Any]:
            for i, col in enumerate(low_cols):
                if any(re.search(p, col, re.I) for p in patterns):
                    return table.columns[i]
            return None

        c_layers = findcol(r"\blayers?\b", r"layer count", r"word.?line")
        c_cap = findcol(r"die.*capacity", r"capacity", r"\bgb\b", r"gbit", r"tbit", r"\btb\b")
        c_area = findcol(r"die.*area", r"area.*mm", r"mm\s*(?:2|\^2|\u00b2)")
        c_bits = findcol(r"bits?.*cell", r"cell.*bits", r"cell type", r"tlc|qlc|mlc|slc")
        c_year = findcol(r"year", r"announced", r"introduced", r"date")
        c_company = findcol(r"manufacturer", r"company", r"vendor", r"maker", r"samsung|toshiba|kioxia|micron|intel|sk hynix|sandisk|western digital")
        c_product = findcol(r"product", r"generation", r"technology", r"node", r"name")
        if sum(bool(x) for x in [c_layers, c_cap, c_area, c_bits, c_year, c_company, c_product]) < 3:
            continue
        last_context: Dict[str, Any] = {}
        for row_index, raw in table.iterrows():
            line = " | ".join(_s(v) for v in raw.tolist() if _s(v))
            if not re.search(r"nand|v-?nand|flash|layer|tlc|qlc|mlc|slc|gb|tb|mm", line, re.I):
                continue
            layers = _first_number_from_text_v67(raw.get(c_layers)) if c_layers else None
            cap = _first_number_from_text_v67(raw.get(c_cap)) if c_cap else None
            if c_cap and re.search(r"\b(tb|tbit)\b", _s(raw.get(c_cap)), re.I) and cap is not None:
                cap *= 1000.0
            area = _first_number_from_text_v67(raw.get(c_area)) if c_area else None
            bits = _map_bits_per_cell(raw.get(c_bits)) if c_bits else ""
            year = _first_number_from_text_v67(raw.get(c_year)) if c_year else None
            company = _s(raw.get(c_company)) if c_company else _nand_company_from_text_v71(line)
            product = _s(raw.get(c_product)) if c_product else line[:180]
            if company:
                last_context["company"] = company
            if year:
                last_context["year"] = year
            if bits:
                last_context["bits_per_cell"] = bits
            if layers:
                last_context["layers"] = layers
            row = {
                "company": company or _s(last_context.get("company")),
                "year": year or last_context.get("year"),
                "layers": layers or last_context.get("layers"),
                "capacity_Gb": cap,
                "die_area_mm2": area,
                "bits_per_cell": bits or last_context.get("bits_per_cell"),
                "source_url": url,
                "product_or_paper": product,
                "notes": f"{label}; html_table_{table_index}",
                "harvested_public_v67": "yes",
                "harvest_source_label_v67": label,
                "harvest_source_url_v67": url,
                "harvest_frame_index_v67": table_index,
                "harvest_row_index_v67": int(row_index) if isinstance(row_index, int) else 0,
            }
            rows.append(row)
    return rows


def _nand_company_from_text_v71(text: str) -> str:
    patterns = [
        ("Samsung", r"\bsamsung\b"),
        ("Kioxia/Toshiba", r"\bkioxia\b|\btoshiba\b"),
        ("SK hynix", r"\bsk\s*hynix\b|\bhynix\b"),
        ("Micron", r"\bmicron\b"),
        ("Intel", r"\bintel\b"),
        ("Western Digital/SanDisk", r"\bwestern\s*digital\b|\bsandisk\b|\bwd\b"),
        ("YMTC", r"\bymtc\b|yangtze"),
    ]
    for name, pat in patterns:
        if re.search(pat, text, re.I):
            return name
    return ""


def _adapter_nand_text_v71(data: bytes, url: str, label: str) -> List[Dict[str, Any]]:
    text = data[:5_000_000].decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"[\r\n]+|(?<=[.;])\s+", text)]
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines[:5000]):
        low = line.lower()
        if not re.search(r"nand|v-?nand|flash", low):
            continue
        if not re.search(r"layer|tlc|qlc|mlc|slc|gb|tb|gbit|tbit|gbyte|tbyte|die|mm", low):
            continue
        company = _nand_company_from_text_v71(line)
        layers = None
        layer_match = re.search(r"(\d{2,3})\s*[- ]?(?:layer|layers|l\b)", line, re.I)
        if layer_match:
            layers = _intish(layer_match.group(1))
        cap = None
        cap_match = re.search(r"(\d+(?:\.\d+)?)\s*(Tb|Tbit|Tbit/s|Gb|Gbit|Gbit/s|GB|GByte|TB|TByte)\b", line, re.I)
        if cap_match:
            value = float(cap_match.group(1))
            unit_raw = cap_match.group(2)
            unit = unit_raw.lower()
            if unit in {"gb", "gbyte"} and "byte" in unit_raw.lower() or unit_raw == "GB":
                cap = value * 8.0
            elif unit in {"tb", "tbyte"} and "byte" in unit_raw.lower() or unit_raw == "TB":
                cap = value * 8000.0
            else:
                cap = value * (1000.0 if unit.startswith("t") else 1.0)
        area = None
        area_match = re.search(r"(?:die\s*(?:size|area)\s*)?(\d+(?:\.\d+)?)\s*(?:mm2|mm\^2|mm\s*\\u00b2|sq\.?\s*mm|square\s*mm)", line, re.I)
        if area_match:
            area = float(area_match.group(1))
        bits = _map_bits_per_cell(line)
        year = None
        year_match = re.search(r"\b(20[0-3]\d)\b", line)
        if year_match:
            year = int(year_match.group(1))
        row = {
            "company": company,
            "year": year,
            "layers": layers,
            "capacity_Gb": cap,
            "die_area_mm2": area,
            "bits_per_cell": bits,
            "source_url": url,
            "product_or_paper": line[:180],
            "notes": f"{label}; text_line_{idx}; source_specific_text_pdf_extraction_v71",
            "harvested_public_v67": "yes",
            "harvest_source_label_v67": label,
            "harvest_source_url_v67": url,
            "harvest_frame_index_v67": 0,
            "harvest_row_index_v67": idx,
        }
        rows.append(row)
    return rows


def _nand_alias_key_v72(row: Dict[str, Any]) -> str:
    company = safe_name(_s(row.get("company")), max_len=40).lower()
    year = _s(row.get("year"))
    layers = _s(row.get("layers"))
    bits = _s(row.get("bits_per_cell"))
    product = safe_name(re.sub(r"\b(20[0-3]\d|\d+\s*(?:gb|tb|gbit|tbit)|\d+\s*layers?)\b", " ", _s(row.get("product_or_paper")), flags=re.I), max_len=60).lower()
    return "|".join([company, year, layers, bits, product])


def _join_nand_partials_v72(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = _nand_alias_key_v72(row)
        if key.strip("|"):
            grouped.setdefault(key, []).append(row)
    joined: List[Dict[str, Any]] = []
    for key, members in grouped.items():
        merged: Dict[str, Any] = {}
        for row in members:
            for col in EXACT_PACKS["nand"]["columns"]:
                if _s(row.get(col)) and not _s(merged.get(col)):
                    merged[col] = row.get(col)
        urls = _unique(_s(r.get("source_url")) for r in members if _s(r.get("source_url")))
        if urls:
            merged["source_url"] = ";".join(urls)
        if not _s(merged.get("notes")):
            merged["notes"] = f"joined_public_nand_alias_rows_v72:{key}"
        missing, problems = _candidate_acceptance_v67("nand", merged, _s(merged.get("source_url")))
        if not missing and not problems:
            joined.append(merged)
    return joined, {"n_nand_partial_rows_v72": len(rows), "n_nand_alias_groups_v72": len(grouped), "n_joined_nand_rows_v72": len(joined)}


def _adapter_thermoelectric_text_v72(data: bytes, url: str, label: str) -> List[Dict[str, Any]]:
    text = data[:3_000_000].decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"[\r\n]+|(?<=[.;])\s+", text)]
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines[:4000]):
        low = line.lower()
        if not re.search(r"zt|figure of merit|thermoelectric|bi2te3|sb2te3", low):
            continue
        if not re.search(r"orientation|texture|textur|grain boundary|misorientation|tilt boundary|angle|theta|\bdeg\b|degree|ebsd|xrd", low):
            continue
        zt = None
        ztm = re.search(r"(?:ZT|zT|figure of merit)[^0-9]{0,20}(\d+(?:\.\d+)?)", line)
        if ztm:
            zt = float(ztm.group(1))
        temp = None
        tm = re.search(r"(\d+(?:\.\d+)?)\s*(?:K|kelvin)\b", line, re.I)
        if tm:
            temp = float(tm.group(1))
        angle = None
        am = re.search(r"(\d+(?:\.\d+)?)\s*(?:deg|degree|°)", line, re.I)
        if am:
            angle = float(am.group(1))
        if angle is None:
            am2 = re.search(r"(?:orientation|texture|misorientation|grain boundary|theta|angle)[^0-9]{0,40}(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|\\u00b0)?", line, re.I)
            if am2:
                angle = float(am2.group(1))
        material_match = re.search(r"\b(Bi2Te3|Sb2Te3|BiSbTe|bismuth telluride|antimony telluride)\b", line, re.I)
        material = material_match.group(1) if material_match else ""
        rows.append({
            "material": material,
            "composition": material or line[:80],
            "ZT": zt,
            "temperature_K": temp,
            "orientation_angle_deg": angle,
            "grain_boundary_angle_deg": angle,
            "source_url": url,
            "source_label": label,
            "harvested_public_v67": "yes",
            "harvest_source_label_v67": label,
            "harvest_source_url_v67": url,
            "harvest_frame_index_v67": 0,
            "harvest_row_index_v67": idx,
        })
    return rows


def _collect_hepdata_record_ids_v71(obj: Any) -> List[str]:
    ids: List[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, val in value.items():
                low = str(key).lower()
                if low in {"recid", "record_id", "recordid", "id"}:
                    text = _s(val)
                    if re.fullmatch(r"(?:ins)?\d{4,}", text, re.I):
                        ids.append(text)
                if isinstance(val, str):
                    for match in re.finditer(r"(?:record/)?(ins\d{4,}|\d{5,})", val, re.I):
                        ids.append(match.group(1))
                visit(val)
        elif isinstance(value, list):
            for item in value[:1000]:
                visit(item)
        elif isinstance(value, str):
            for match in re.finditer(r"(?:record/)?(ins\d{4,}|\d{5,})", value, re.I):
                ids.append(match.group(1))

    visit(obj)
    return _unique(ids)[:50]


def _hepdata_record_urls_v71(record_id: str) -> List[str]:
    rid = _s(record_id)
    if not rid:
        return []
    bare = rid[3:] if rid.lower().startswith("ins") else rid
    ins = rid if rid.lower().startswith("ins") else f"ins{bare}"
    return _unique([
        f"https://www.hepdata.net/record/{ins}?format=json",
        f"https://www.hepdata.net/record/{bare}?format=json",
        f"https://www.hepdata.net/record/{ins}",
    ])


def _collect_table_urls_from_hepdata_v71(obj: Any, base_url: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    def add(url: str, table_id: str = "") -> None:
        if not url:
            return
        full = urljoin(base_url, url)
        if "hepdata.net" in urlparse(full).netloc.lower() or re.search(r"\.(csv|yaml|yml|json)(\?|$)", full, re.I):
            out.append((full, table_id))

    def visit(value: Any, table_id: str = "") -> None:
        if isinstance(value, dict):
            tid = table_id or _s(value.get("name") or value.get("table") or value.get("table_id") or value.get("id"))
            for key, val in value.items():
                low = str(key).lower()
                if isinstance(val, str) and (low in {"data", "csv", "yaml", "yml", "json", "url", "download", "href"} or re.search(r"csv|yaml|download|data", low)):
                    add(val, tid)
                visit(val, tid)
        elif isinstance(value, list):
            for item in value[:500]:
                visit(item, table_id)
        elif isinstance(value, str):
            if re.search(r"\.(csv|yaml|yml|json)(\?|$)", value, re.I):
                add(value, table_id)

    visit(obj)
    unique = _unique([f"{u}|||{tid}" for u, tid in out])
    return [(x.split("|||", 1)[0], x.split("|||", 1)[1]) for x in unique]


def _hepdata_value_v74(cell: Any) -> Optional[float]:
    if isinstance(cell, dict):
        return _num(cell.get("value") or cell.get("low") or cell.get("high"))
    return _num(cell)


def _hepdata_uncertainty_v74(cell: Any) -> Optional[float]:
    if not isinstance(cell, dict):
        return None
    errors = cell.get("errors")
    if not isinstance(errors, list):
        return None
    vals: List[float] = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        for key in ["symerror", "error", "plus", "minus"]:
            val = _num(err.get(key))
            if val is not None:
                vals.append(abs(val))
    if not vals:
        return None
    return math.sqrt(sum(v * v for v in vals))


def _hepdata_table_object_v74(data: bytes) -> Optional[Any]:
    obj = _parse_json_if_possible(data)
    if obj is not None:
        return obj
    if yaml is None:
        return None
    try:
        return yaml.safe_load(data.decode("utf-8", errors="replace"))
    except Exception:
        return None


def _hepdata_dependent_name_v74(dep: Dict[str, Any]) -> str:
    header = dep.get("header") if isinstance(dep, dict) else {}
    if isinstance(header, dict):
        return _s(header.get("name") or header.get("title") or header.get("label"))
    return ""


def _flatten_hepdata_table_v74(data: bytes, record_id: str, table_id: str, source_url: str) -> List[Dict[str, Any]]:
    obj = _hepdata_table_object_v74(data)
    if not isinstance(obj, dict):
        return []
    indep = obj.get("independent_variables") if isinstance(obj.get("independent_variables"), list) else []
    dep = obj.get("dependent_variables") if isinstance(obj.get("dependent_variables"), list) else []
    if len(dep) < 2:
        return []
    dep_pairs: List[Tuple[str, Dict[str, Any]]] = []
    for item in dep:
        if isinstance(item, dict) and isinstance(item.get("values"), list):
            dep_pairs.append((_hepdata_dependent_name_v74(item).lower(), item))
    if len(dep_pairs) < 2:
        return []
    observed_item = next((item for name, item in dep_pairs if re.search(r"observed|measured|data", name)), dep_pairs[0][1])
    model_item = next((item for name, item in dep_pairs if re.search(r"expected|model|theory|prediction|sm", name)), None)
    if model_item is None:
        model_item = dep_pairs[1][1] if dep_pairs[0][1] is observed_item and len(dep_pairs) > 1 else dep_pairs[0][1]
    if model_item is observed_item:
        return []
    obs_values = observed_item.get("values") if isinstance(observed_item.get("values"), list) else []
    model_values = model_item.get("values") if isinstance(model_item.get("values"), list) else []
    x_values = []
    if indep and isinstance(indep[0], dict) and isinstance(indep[0].get("values"), list):
        x_values = indep[0].get("values") or []
    n = min(len(obs_values), len(model_values))
    rows: List[Dict[str, Any]] = []
    for idx in range(n):
        obs = _hepdata_value_v74(obs_values[idx])
        mod = _hepdata_value_v74(model_values[idx])
        unc = _hepdata_uncertainty_v74(obs_values[idx]) or _hepdata_uncertainty_v74(model_values[idx])
        if obs is None or mod is None or unc is None or unc <= 0:
            continue
        x = _hepdata_value_v74(x_values[idx]) if idx < len(x_values) else idx
        rows.append({
            "record_id": record_id,
            "table_id": table_id,
            "x": x,
            "observed": obs,
            "model": mod,
            "uncertainty": unc,
            "observable_name": _hepdata_dependent_name_v74(observed_item) or table_id,
            "source_url": source_url,
        })
    return rows


def _adapter_hepdata_api_v71(
    outdir: Path,
    cache: Path,
    *,
    allow_network: bool,
    dry_run: bool,
    force: bool,
    timeout: int,
    max_bytes: int,
    max_sources: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    attempts: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    if dry_run:
        attempts.append({"pack": "hepdata", "status_v67": "planned_hepdata_api_adapter_v71"})
        return accepted, attempts, candidates
    progress_path = outdir / "data" / "generated" / "hepdata_api_progress_v72.json"
    progress: Dict[str, Any] = {"schema": "ccdr-v72-hepdata-api-progress", "status_v72": "running", "events_v72": []}
    _write_json_file(progress_path, progress)
    search_urls = [s.url for s in PACK_PUBLIC_SEEDS_V67.get("hepdata", [])[:max_sources]]
    record_ids: List[str] = []
    for search_url in search_urls:
        data, meta = _download_or_cached(
            search_url,
            cache / "public_source_harvest_v67" / "hepdata_api",
            allow_network=allow_network,
            timeout=timeout,
            force=force,
            max_bytes=max_bytes,
            manifest_approved=True,
        )
        obj = _parse_json_if_possible(data) if data else None
        found = _collect_hepdata_record_ids_v71(obj) if obj is not None else []
        if data and not found:
            found = _collect_hepdata_record_ids_v71(data[:2_000_000].decode("utf-8", errors="ignore"))
        direct_record = re.search(r"/record/(ins\d{4,}|\d{5,})", search_url, re.I)
        if direct_record:
            found.append(direct_record.group(1))
        record_ids.extend(found)
        attempts.append({"pack": "hepdata", "url": search_url, "status_v67": "hepdata_search_downloaded_v74" if data else "hepdata_search_unavailable", "json_parsed_v74": obj is not None, "n_record_ids_v71": len(found), "download_meta_v67": meta})
        progress["events_v72"].append({"stage_v72": "search", "url": search_url, "n_record_ids_v72": len(found), "utc_v72": utc_now()})
        _write_json_file(progress_path, progress)
    table_urls: List[Tuple[str, str, str]] = []
    for record_id in _unique(record_ids)[:max_sources]:
        for record_url in _hepdata_record_urls_v71(record_id):
            data, meta = _download_or_cached(
                record_url,
                cache / "public_source_harvest_v67" / "hepdata_api",
                allow_network=allow_network,
                timeout=timeout,
                force=force,
                max_bytes=max_bytes,
                manifest_approved=True,
            )
            obj = _parse_json_if_possible(data) if data else None
            attempts.append({"pack": "hepdata", "url": record_url, "record_id_v71": record_id, "status_v67": "hepdata_record_downloaded_v74" if data else "hepdata_record_unavailable", "json_parsed_v74": obj is not None, "download_meta_v67": meta})
            progress["events_v72"].append({"stage_v72": "record", "record_id_v72": record_id, "url": record_url, "ok_v72": bool(data), "utc_v72": utc_now()})
            progress["events_v72"] = progress["events_v72"][-80:]
            _write_json_file(progress_path, progress)
            if not data:
                continue
            discovered_tables = _collect_table_urls_from_hepdata_v71(obj, record_url) if obj is not None else []
            if not discovered_tables:
                for link in _discover_links_from_payload(data, record_url):
                    if re.search(r"hepdata\.net/.*/(?:table|download|record)|\.(csv|yaml|yml|json)(?:\?|$)", link, re.I):
                        discovered_tables.append((link, safe_name(link, max_len=60)))
            for table_url, table_id in discovered_tables:
                table_urls.append((record_id, table_id or safe_name(table_url, max_len=60), table_url))
            break
    table_dir = outdir / "data" / "generated" / "hepdata_tables_v71"
    for record_id, table_id, table_url in table_urls[:max_sources * 5]:
        data, meta = _download_or_cached(
            table_url,
            cache / "public_source_harvest_v67" / "hepdata_api",
            allow_network=allow_network,
            timeout=timeout,
            force=force,
            max_bytes=max_bytes,
            manifest_approved=True,
        )
        attempts.append({"pack": "hepdata", "url": table_url, "record_id_v71": record_id, "table_id_v71": table_id, "status_v67": "hepdata_table_downloaded" if data else "hepdata_table_unavailable", "download_meta_v67": meta})
        progress["events_v72"].append({"stage_v72": "table", "record_id_v72": record_id, "table_id_v72": table_id, "url": table_url, "ok_v72": bool(data), "accepted_so_far_v72": len(accepted), "utc_v72": utc_now()})
        progress["events_v72"] = progress["events_v72"][-80:]
        _write_json_file(progress_path, progress)
        if not data:
            continue
        flat_rows = _flatten_hepdata_table_v74(data, record_id, table_id, table_url)
        if flat_rows:
            local_path = table_dir / f"{safe_name(record_id)}_{safe_name(table_id)}_flattened_v74.csv"
            _write_csv(local_path, flat_rows, ["x", "observed", "model", "uncertainty", "record_id", "table_id", "observable_name", "source_url"])
            row = {
                "record_id": record_id,
                "table_id": table_id,
                "x_column": "x",
                "observed_column": "observed",
                "model_column": "model",
                "uncertainty_column": "uncertainty",
                "observable_name": table_id or _s(table_url),
                "local_table": str(local_path),
                "source_url": table_url,
                "harvested_public_v67": "yes",
                "harvest_source_label_v67": "HEPData public API flattened table v74",
                "harvest_source_url_v67": table_url,
                "harvest_frame_index_v67": 0,
                "harvest_row_index_v67": len(flat_rows),
            }
            missing, problems = _candidate_acceptance_v67("hepdata", row, table_url)
            _capture_candidate_v74(candidates, {"pack": "hepdata", "url": table_url, "label": "HEPData public API flattened table v74", "frame_index_v67": 0, "row_index_v67": len(flat_rows), "accepted_v67": not bool(missing or problems), "missing_required_v67": "|".join(missing), "row_problem_v67": "|".join(problems), "columns_v67": "x|observed|model|uncertainty"})
            if not missing and not problems:
                accepted.append(row)
                continue
        frames, report = _parse_candidate_frames(data, table_url, "hepdata")
        for frame_index, df in enumerate(frames[:10]):
            cols = _frame_columns(df)
            observed = _find_column(cols, COLUMN_ALIASES_V67["hepdata"]["observed_column"])
            model = _find_column(cols, COLUMN_ALIASES_V67["hepdata"]["model_column"])
            uncert = _find_column(cols, COLUMN_ALIASES_V67["hepdata"]["uncertainty_column"])
            xcol = _find_column(cols, COLUMN_ALIASES_V67["hepdata"]["x_column"])
            local_path = table_dir / f"{safe_name(record_id)}_{safe_name(table_id)}_{frame_index}.csv"
            try:
                ensure_dir(local_path.parent)
                df.to_csv(local_path, index=False)
            except Exception:
                pass
            row = {
                "record_id": record_id,
                "table_id": table_id or f"table_{frame_index}",
                "x_column": _s(xcol),
                "observed_column": _s(observed),
                "model_column": _s(model),
                "uncertainty_column": _s(uncert),
                "observable_name": table_id or _s(table_url),
                "local_table": str(local_path),
                "source_url": table_url,
                "harvested_public_v67": "yes",
                "harvest_source_label_v67": "HEPData public API",
                "harvest_source_url_v67": table_url,
                "harvest_frame_index_v67": frame_index,
                "harvest_row_index_v67": len(df),
            }
            missing, problems = _candidate_acceptance_v67("hepdata", row, table_url)
            _capture_candidate_v74(candidates, {"pack": "hepdata", "url": table_url, "label": "HEPData public API", "frame_index_v67": frame_index, "row_index_v67": len(df), "accepted_v67": not bool(missing or problems), "missing_required_v67": "|".join(missing), "row_problem_v67": "|".join(problems), "columns_v67": "|".join(str(c) for c in cols[:40])})
            if not missing and not problems:
                accepted.append(row)
    progress.update({"status_v72": "complete", "n_records_seen_v72": len(_unique(record_ids)), "n_table_urls_seen_v72": len(table_urls), "n_manifest_rows_accepted_v72": len(accepted), "utc_complete_v72": utc_now()})
    _write_json_file(progress_path, progress)
    attempts.append({"pack": "hepdata", "status_v67": "hepdata_api_adapter_summary_v71", "n_records_seen_v71": len(_unique(record_ids)), "n_table_urls_seen_v71": len(table_urls), "n_manifest_rows_accepted_v71": len(accepted), "progress_file_v72": str(progress_path)})
    return accepted, attempts, candidates


def _adapter_benchmark_text_v71(pack: str, data: bytes, url: str, label: str) -> List[Dict[str, Any]]:
    text = data[:3_000_000].decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"[\r\n]+|(?<=[.;])\s+", text)]
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines[:3000]):
        low = line.lower()
        if pack == "optical_interconnect":
            if not re.search(r"optical|photonic|interconnect|link", low):
                continue
            energy = None
            em = re.search(r"(\d+(?:\.\d+)?)\s*(fJ|pJ)\s*/?\s*(?:bit|b)", line, re.I)
            if em:
                energy = float(em.group(1)) / 1000.0 if em.group(2).lower() == "fj" else float(em.group(1))
            bw = None
            bm = re.search(r"(\d+(?:\.\d+)?)\s*(Tb/s|Tbit/s|Gb/s|Gbit/s|Gbps)", line, re.I)
            if bm:
                bw = float(bm.group(1)) * (1000.0 if bm.group(2).lower().startswith("t") else 1.0)
            reach = None
            rm = re.search(r"(\d+(?:\.\d+)?)\s*(km|m|cm|mm)\b", line, re.I)
            if rm:
                factor = {"km": 1000.0, "m": 1.0, "cm": 0.01, "mm": 0.001}[rm.group(2).lower()]
                reach = float(rm.group(1)) * factor
            year = _intish(re.search(r"\b(20[0-3]\d)\b", line).group(1)) if re.search(r"\b(20[0-3]\d)\b", line) else None
            if energy is None or bw is None or reach is None or year is None:
                continue
            rows.append({"platform": line[:80], "year": year, "energy_per_bit_pJ": energy, "bandwidth_Gbps": bw, "reach_m": reach, "source_url": url, "benchmark": label, "harvested_public_v67": "yes", "harvest_source_label_v67": label, "harvest_source_url_v67": url, "harvest_frame_index_v67": 0, "harvest_row_index_v67": idx})
        elif pack == "neuromorphic":
            if not re.search(r"loihi|spinnaker|truenorth|neuromorphic|neurobench", low):
                continue
            energy = None
            em = re.search(r"(\d+(?:\.\d+)?)\s*(nJ|pJ|uJ)\b", line, re.I)
            if em:
                unit = em.group(2).lower()
                energy = float(em.group(1)) * (1000.0 if unit == "nj" else 1_000_000.0 if unit == "uj" else 1.0)
            accuracy = None
            am = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
            if am:
                accuracy = float(am.group(1)) / 100.0
            year = _intish(re.search(r"\b(20[0-3]\d)\b", line).group(1)) if re.search(r"\b(20[0-3]\d)\b", line) else None
            chip = re.search(r"\b(Loihi\s*\d*|SpiNNaker\s*\d*|TrueNorth|DYNAP|BrainScaleS|Neurogrid)\b", line, re.I)
            if energy is None or accuracy is None or year is None or not chip:
                continue
            rows.append({"chip": chip.group(1) if chip else line[:60], "benchmark": label, "energy_per_inference_or_spike_pJ": energy, "accuracy": accuracy, "topology": line[:120], "year": year, "source_url": url, "harvested_public_v67": "yes", "harvest_source_label_v67": label, "harvest_source_url_v67": url, "harvest_frame_index_v67": 0, "harvest_row_index_v67": idx})
        elif pack == "ldpc_external_benchmark":
            if not re.search(r"ldpc|burst|bler|ber|fer|code", low):
                continue
            nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", line, re.I)[:4]]
            if len(nums) < 3:
                continue
            metric = "BER" if re.search(r"\bber\b", line, re.I) else "BLER" if re.search(r"\bbler\b", line, re.I) else "FER" if re.search(r"\bfer\b", line, re.I) else "public_text_metric"
            rows.append({"task_id": safe_name(line[:50]), "benchmark": label, "metric_name": "public_text_metric", "model_score": nums[0] if nums else "", "baseline_score": nums[1] if len(nums) > 1 else "", "uncertainty": nums[2] if len(nums) > 2 else "", "heldout_split": "public_test", "source_url": url, "source_label": label, "external_public_yes_no": "yes", "notes": line[:180], "harvested_public_v67": "yes", "harvest_source_label_v67": label, "harvest_source_url_v67": url, "harvest_frame_index_v67": 0, "harvest_row_index_v67": idx})
            rows[-1]["metric_name"] = metric
    return rows


def _source_specific_adapter_rows_v67(
    pack: str,
    outdir: Path,
    cache: Path,
    *,
    allow_network: bool,
    dry_run: bool,
    force: bool,
    timeout: int,
    max_bytes: int,
    max_sources_per_pack: int,
    max_rows_per_source: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if pack == "proteingym":
        cached_rows, cached_attempts, cached_candidates = _adapter_proteingym_cached_raw_files_v72(
            outdir,
            cache,
            dry_run=dry_run,
            max_files=max(10, max_sources_per_pack * 10),
            max_rows_per_file=max_rows_per_source,
            max_bytes=max_bytes,
        )
        raw_rows, raw_attempts, raw_candidates = _adapter_proteingym_raw_dms_v67(
            outdir,
            cache,
            allow_network=allow_network,
            dry_run=dry_run,
            force=force,
            timeout=timeout,
            max_bytes=max_bytes,
            max_assays=max(1, min(max_sources_per_pack, 50)),
            max_rows_per_assay=max_rows_per_source,
        )
        return cached_rows + raw_rows, cached_attempts + raw_attempts, cached_candidates + raw_candidates
    if pack == "protein_structures":
        return _adapter_alphafold_structures_v67(
            outdir,
            cache,
            allow_network=allow_network,
            dry_run=dry_run,
            force=force,
            timeout=timeout,
            max_bytes=max_bytes,
            max_structures=max(1, max_sources_per_pack * 8),
        )
    if pack == "materials":
        return _adapter_cmbs4_materials_v67(
            outdir,
            cache,
            allow_network=allow_network,
            dry_run=dry_run,
            force=force,
            timeout=timeout,
            max_tables=max(1, max_sources_per_pack * 10),
        )
    if pack == "hepdata":
        return _adapter_hepdata_api_v71(
            outdir,
            cache,
            allow_network=allow_network,
            dry_run=dry_run,
            force=force,
            timeout=timeout,
            max_bytes=max_bytes,
            max_sources=max(1, max_sources_per_pack),
        )
    return [], [], []


def _packs_for_tests(tests: Optional[Sequence[str]]) -> List[str]:
    if not tests:
        return list(EXACT_PACKS.keys())
    packs: List[str] = []
    for test in tests:
        for pack in TEST_REQUIRED_PACKS_V64.get(test.upper(), []):
            if pack not in packs:
                packs.append(pack)
    return packs or list(EXACT_PACKS.keys())


PACK_PRIORITY_V72 = [
    "ldpc_external_benchmark",
    "nand",
    "materials",
    "materials_family_packs",
    "hepdata",
    "optical_interconnect",
    "neuromorphic",
    "thermoelectric",
    "fusion",
    "protein_structures",
    "proteingym",
]


def _order_packs_v72(packs: Sequence[str]) -> List[str]:
    priority = {pack: idx for idx, pack in enumerate(PACK_PRIORITY_V72)}
    return sorted(list(packs), key=lambda p: (priority.get(p, 999), p))


def _candidate_quality_summary_v71(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(candidates)
    accepted = sum(1 for c in candidates if str(c.get("accepted_v67")).lower() == "true")
    missing_counts: Dict[str, int] = {}
    problem_counts: Dict[str, int] = {}
    for c in candidates:
        for miss in _s(c.get("missing_required_v67")).split("|"):
            if miss:
                missing_counts[miss] = missing_counts.get(miss, 0) + 1
        for problem in _s(c.get("row_problem_v67")).split("|"):
            if problem:
                problem_counts[problem] = problem_counts.get(problem, 0) + 1
    return {
        "n_candidate_rows_v71": total,
        "n_candidate_rows_accepted_v71": accepted,
        "n_candidate_rows_rejected_v71": total - accepted,
        "top_missing_required_v71": dict(sorted(missing_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
        "top_row_problems_v71": dict(sorted(problem_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
    }


def _write_pack_action_file_v74(
    outdir: Path,
    pack: str,
    candidates: Sequence[Dict[str, Any]],
    attempts: Sequence[Dict[str, Any]],
    prewrite_report: Dict[str, Any],
    rows_written: int,
) -> Optional[str]:
    quality = _candidate_quality_summary_v71(candidates)
    candidate_rows = int(quality.get("n_candidate_rows_v71") or 0)
    accepted_candidates = int(quality.get("n_candidate_rows_accepted_v71") or 0)
    rejected_prewrite = int(prewrite_report.get("n_rows_rejected_prewrite_v72") or 0)
    if candidate_rows < 1000 and accepted_candidates > 0 and rows_written > 0 and rejected_prewrite == 0:
        return None
    rejected_examples = [
        c for c in candidates
        if not bool(c.get("accepted_v67")) and (_s(c.get("missing_required_v67")) or _s(c.get("row_problem_v67")))
    ][:25]
    action = {
        "schema": "ccdr-v74-pack-action-diagnostics",
        "pack": pack,
        "generated_utc": utc_now(),
        "rows_written_v67": rows_written,
        "candidate_quality_v71": quality,
        "prewrite_valid_only_report_v72": prewrite_report,
        "top_rejected_candidate_examples_v74": rejected_examples,
        "recent_adapter_attempts_v74": list(attempts)[-20:],
        "next_adapter_action_v74": "Tighten or add a source-specific public parser for these missing fields/problems; do not request manual rows.",
    }
    path = outdir / "data" / "generated" / f"v74_{safe_name(pack)}_action_diagnostics.json"
    _write_json(path, action)
    return str(path)


def _validation_usable_by_pack_v71(validation: Optional[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(validation, dict):
        return out
    for pack in validation.get("pack_results", []) or []:
        if isinstance(pack, dict):
            out[_s(pack.get("pack"))] = int(pack.get("validator_usable_rows_v64") or 0)
    return out


def _harvest_pack(
    pack: str,
    outdir: Path,
    cache: Path,
    *,
    allow_network: bool,
    dry_run: bool,
    force: bool,
    timeout: int,
    max_bytes: int,
    max_sources_per_pack: int,
    max_rows_per_source: int,
    write_rows: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    seeds = PACK_PUBLIC_SEEDS_V67.get(pack, [])
    pack_cache = cache / "public_source_harvest_v67" / pack
    attempts: List[Dict[str, Any]] = []
    candidates_csv: List[Dict[str, Any]] = []
    accepted_rows: List[Dict[str, Any]] = []
    partial_stage_rows: List[Dict[str, Any]] = []
    source_specific_partial_rows: List[Dict[str, Any]] = []
    quarantine_report = _quarantine_stale_generated_rows_v72(pack, outdir, cache)
    if quarantine_report.get("quarantined_v72"):
        attempts.append({"pack": pack, "status_v67": "stale_generated_rows_quarantined_v72", **quarantine_report})
    _checkpoint_pack_v72(outdir, pack, "start", allow_network_v67=allow_network, dry_run_v67=dry_run, quarantine_report_v72=quarantine_report)
    adapter_rows, adapter_attempts, adapter_candidates = _source_specific_adapter_rows_v67(
        pack,
        outdir,
        cache,
        allow_network=allow_network,
        dry_run=dry_run,
        force=force,
        timeout=timeout,
        max_bytes=max_bytes,
        max_sources_per_pack=max_sources_per_pack,
        max_rows_per_source=max_rows_per_source,
    )
    accepted_rows.extend(adapter_rows)
    attempts.extend(adapter_attempts)
    candidates_csv.extend(adapter_candidates)
    _checkpoint_pack_v72(outdir, pack, "after_source_specific_adapters", n_adapter_rows_v72=len(adapter_rows), n_adapter_candidates_v72=len(adapter_candidates), n_adapter_attempts_v72=len(adapter_attempts))
    source_queue: List[Tuple[str, str, bool, bool]] = [
        (seed.url, seed.label, seed.direct_structured, seed.manifest_approved)
        for seed in seeds
    ]
    seen_urls = set()
    n_downloaded = 0
    n_parsed_sources = 0

    for seed in seeds:
        if len(seen_urls) >= max_sources_per_pack:
            break
        source_queue = [(seed.url, seed.label, seed.direct_structured, seed.manifest_approved)]
        queue_index = 0
        while queue_index < len(source_queue) and len(seen_urls) < max_sources_per_pack:
            url, label, direct_structured, manifest_approved = source_queue[queue_index]
            queue_index += 1
            if url in seen_urls:
                continue
            seen_urls.add(url)
            attempt: Dict[str, Any] = {
                "pack": pack,
                "url": url,
                "label": label,
                "direct_structured_v67": direct_structured,
                "manifest_approved_v67": manifest_approved,
                "dry_run_v67": dry_run,
                "allow_network_v67": allow_network,
                "n_candidate_links_v67": 0,
                "n_frames_v67": 0,
                "n_rows_accepted_v67": 0,
                "n_rows_rejected_v67": 0,
            }
            if dry_run:
                attempt["status_v67"] = "planned_only_dry_run"
                attempts.append(attempt)
                continue
            data, meta = _download_or_cached(
                url,
                pack_cache,
                allow_network=allow_network,
                timeout=timeout,
                force=force,
                max_bytes=max_bytes,
                manifest_approved=manifest_approved,
            )
            attempt["download_meta_v67"] = meta
            if not data:
                attempt["status_v67"] = "download_or_cache_unavailable"
                attempts.append(attempt)
                continue
            n_downloaded += 1

            if pack in {"materials", "materials_family_packs"}:
                material_text_rows = _adapter_materials_text_v72(pack, data, url, label)
                attempt["n_source_specific_material_text_rows_v72"] = len(material_text_rows)
                for row_index, norm in enumerate(material_text_rows[:max_rows_per_source]):
                    missing, problems = _candidate_acceptance_v67(pack, norm, url)
                    _capture_candidate_v74(candidates_csv, {
                        "pack": pack,
                        "url": url,
                        "label": label,
                        "frame_index_v67": norm.get("harvest_frame_index_v67", 0),
                        "row_index_v67": row_index,
                        "accepted_v67": not bool(missing or problems),
                        "missing_required_v67": "|".join(missing),
                        "row_problem_v67": "|".join(problems),
                        "columns_v67": "source_specific_materials_text_adapter_v72",
                    })
                    if missing:
                        partial_stage_rows.append(_material_partial_stage_row_v71(pack, norm, missing))
                        attempt["n_rows_rejected_v67"] += 1
                    elif problems:
                        attempt["n_rows_rejected_v67"] += 1
                    else:
                        accepted_rows.append(norm)
                        attempt["n_rows_accepted_v67"] += 1

            if pack == "nand":
                nand_rows = _adapter_nand_html_v67(data, url, label)
                nand_text_rows = _adapter_nand_text_v71(data, url, label)
                if nand_text_rows:
                    nand_rows.extend(nand_text_rows)
                attempt["n_source_specific_rows_v67"] = len(nand_rows)
                for row_index, norm in enumerate(nand_rows[:max_rows_per_source]):
                    source_specific_partial_rows.append(norm)
                    missing, problems = _candidate_acceptance_v67(pack, norm, url)
                    _capture_candidate_v74(candidates_csv, {
                        "pack": pack,
                        "url": url,
                        "label": label,
                        "frame_index_v67": norm.get("harvest_frame_index_v67", 0),
                        "row_index_v67": row_index,
                        "accepted_v67": not bool(missing or problems),
                        "missing_required_v67": "|".join(missing),
                        "row_problem_v67": "|".join(problems),
                        "columns_v67": "source_specific_nand_html_adapter",
                    })
                    if missing or problems:
                        attempt["n_rows_rejected_v67"] += 1
                    else:
                        accepted_rows.append(norm)
                        attempt["n_rows_accepted_v67"] += 1
            if pack == "thermoelectric":
                te_rows = _adapter_thermoelectric_text_v72(data, url, label)
                attempt["n_source_specific_thermoelectric_text_rows_v72"] = len(te_rows)
                for row_index, norm in enumerate(te_rows[:max_rows_per_source]):
                    missing, problems = _candidate_acceptance_v67(pack, norm, url)
                    _capture_candidate_v74(candidates_csv, {
                        "pack": pack,
                        "url": url,
                        "label": label,
                        "frame_index_v67": norm.get("harvest_frame_index_v67", 0),
                        "row_index_v67": row_index,
                        "accepted_v67": not bool(missing or problems),
                        "missing_required_v67": "|".join(missing),
                        "row_problem_v67": "|".join(problems),
                        "columns_v67": "source_specific_thermoelectric_text_adapter_v72",
                    })
                    if missing or problems:
                        attempt["n_rows_rejected_v67"] += 1
                    else:
                        accepted_rows.append(norm)
                        attempt["n_rows_accepted_v67"] += 1
            if pack in {"optical_interconnect", "neuromorphic", "ldpc_external_benchmark"}:
                text_rows = _adapter_benchmark_text_v71(pack, data, url, label)
                attempt["n_source_specific_benchmark_rows_v71"] = len(text_rows)
                for row_index, norm in enumerate(text_rows[:max_rows_per_source]):
                    missing, problems = _candidate_acceptance_v67(pack, norm, url)
                    _capture_candidate_v74(candidates_csv, {
                        "pack": pack,
                        "url": url,
                        "label": label,
                        "frame_index_v67": norm.get("harvest_frame_index_v67", 0),
                        "row_index_v67": row_index,
                        "accepted_v67": not bool(missing or problems),
                        "missing_required_v67": "|".join(missing),
                        "row_problem_v67": "|".join(problems),
                        "columns_v67": "source_specific_benchmark_text_adapter_v71",
                    })
                    if missing or problems:
                        attempt["n_rows_rejected_v67"] += 1
                    else:
                        accepted_rows.append(norm)
                        attempt["n_rows_accepted_v67"] += 1

            links = _discover_links_from_payload(data, url)
            attempt["n_candidate_links_v67"] = len(links)
            if _is_search_seed(PublicSeedV67(url, label)) or not direct_structured:
                for link in links:
                    if len(source_queue) >= max_sources_per_pack:
                        break
                    source_queue.append((link, label, True, False))
                if not direct_structured:
                    attempt["status_v67"] = "discovery_seed_links_queued"
                    attempts.append(attempt)
                    continue

            frames, parse_report = _parse_candidate_frames(data, url, pack)
            attempt.update(parse_report)
            n_parsed_sources += int(bool(frames))
            for frame_index, df in enumerate(frames[:20]):
                columns = _frame_columns(df)
                for row_index, raw in _frame_iter_rows(df, max_rows_per_source):
                    norm = _normalize_row(pack, raw, columns, url, label, frame_index, row_index)
                    missing, problems = _candidate_acceptance_v67(pack, norm, url)
                    candidate = {
                        "pack": pack,
                        "url": url,
                        "label": label,
                        "frame_index_v67": frame_index,
                        "row_index_v67": row_index,
                        "accepted_v67": not bool(missing or problems),
                        "missing_required_v67": "|".join(missing),
                        "row_problem_v67": "|".join(problems),
                        "columns_v67": "|".join(str(c) for c in columns[:40]),
                    }
                    _capture_candidate_v74(candidates_csv, candidate)
                    if missing or problems:
                        attempt["n_rows_rejected_v67"] += 1
                        if pack in {"materials", "materials_family_packs"} and any(_s(norm.get(k)) for k in ["temperature_K", "kappa_W_mK", "grain_size_nm", "microstructure_method", "material"]):
                            partial_stage_rows.append(_material_partial_stage_row_v71(pack, norm, missing))
                    else:
                        accepted_rows.append(norm)
                        attempt["n_rows_accepted_v67"] += 1
            attempt["status_v67"] = "parsed_structured_source" if attempt["n_frames_v67"] else "no_structured_frames"
            attempts.append(attempt)
        _checkpoint_pack_v72(outdir, pack, "seed_complete", seed_url_v72=seed.url, n_seen_urls_v72=len(seen_urls), n_accepted_rows_so_far_v72=len(accepted_rows), n_candidate_rows_so_far_v72=len(candidates_csv))

    if pack in {"materials", "materials_family_packs"}:
        _write_material_partial_stage_v71(outdir, pack, partial_stage_rows)
        all_partials = _load_material_partial_stage_v72(outdir, pack)
        joined_rows, join_report = _join_material_partials_v71(pack, all_partials or partial_stage_rows)
        if joined_rows:
            accepted_rows.extend(joined_rows)
        attempts.append({"pack": pack, "status_v67": "generic_material_partial_join_v72", **join_report})

    if pack == "nand":
        joined_nand_rows, nand_join_report = _join_nand_partials_v72(source_specific_partial_rows)
        if joined_nand_rows:
            accepted_rows.extend(joined_nand_rows)
        attempts.append({"pack": pack, "status_v67": "nand_product_alias_join_v72", **nand_join_report})

    deduped, duplicate_skips = _dedup_rows(pack, accepted_rows)
    deduped, prewrite_report = _prewrite_valid_rows_v72(pack, deduped)
    row_path = _pack_row_path(pack, outdir, cache)
    wrote_rows = bool(write_rows and not dry_run and deduped)
    if wrote_rows:
        _write_csv(row_path, deduped, _pack_fieldnames(pack))
    action_file_v74 = _write_pack_action_file_v74(
        outdir,
        pack,
        candidates_csv,
        attempts,
        prewrite_report,
        len(deduped) if wrote_rows else 0,
    )
    _checkpoint_pack_v72(outdir, pack, "before_manifest", n_accepted_rows_v72=len(accepted_rows), n_deduped_valid_rows_v72=len(deduped), wrote_rows_v72=wrote_rows, prewrite_report_v72=prewrite_report)

    manifest = {
        "schema": "ccdr-v67-public-source-pack-harvest",
        "pack": pack,
        "affected_tests_v64": PACK_TESTS_V64.get(pack, []),
        "improvement_ids_v67": [x["id"] for x in AUTOMATED_IMPROVEMENTS_V67 if pack in x.get("packs", [])],
        "generated_utc": utc_now(),
        "allow_network_v67": allow_network,
        "dry_run_v67": dry_run,
        "network_policy_v67": "Downloads occur only with --allow-network; otherwise cached files and manifests only.",
        "no_manual_input_policy_v67": "Rows are written only by this harvester from downloaded or cached public-source payloads.",
        "seed_count_v67": len(seeds),
        "seeds_v67": [seed.__dict__ for seed in seeds],
        "n_downloaded_or_cached_sources_v67": n_downloaded,
        "n_parsed_structured_sources_v67": n_parsed_sources,
        "n_accepted_rows_before_dedup_v67": len(accepted_rows),
        "n_duplicate_rows_removed_v67": duplicate_skips,
        "n_accepted_rows_written_v67": len(deduped) if wrote_rows else 0,
        "prewrite_valid_only_report_v72": prewrite_report,
        "stale_generated_rows_quarantine_v72": quarantine_report,
        "row_output_path_v67": str(row_path) if wrote_rows else None,
        "coverage_v67": _coverage_summary(pack, deduped),
        "candidate_quality_v71": _candidate_quality_summary_v71(candidates_csv),
        "candidate_capture_limit_v74": _candidate_capture_limit_v74(),
        "candidate_rows_captured_v74": len(candidates_csv),
        "action_diagnostics_file_v74": action_file_v74,
        "attempts_v67": attempts,
    }
    _write_json(outdir / "data" / "generated" / f"v67_{pack}_public_harvest_manifest.json", manifest)
    _checkpoint_pack_v72(outdir, pack, "complete", manifest_file_v72=str(outdir / "data" / "generated" / f"v67_{pack}_public_harvest_manifest.json"), rows_written_v72=manifest["n_accepted_rows_written_v67"])
    return manifest, candidates_csv


def harvest_public_sources_v67(
    outdir: Optional[Path] = None,
    cache: Optional[Path] = None,
    *,
    only_tests: Optional[Sequence[str]] = None,
    only_packs: Optional[Sequence[str]] = None,
    allow_network: bool = False,
    dry_run: bool = False,
    force: bool = False,
    timeout: int = 45,
    max_bytes: int = 50_000_000,
    max_sources_per_pack: int = 12,
    max_rows_per_source: int = 5000,
    write_rows: bool = True,
    run_validation: bool = True,
) -> Dict[str, Any]:
    root = Path(outdir or "tierb_out_v67_public_source_harvest")
    cache_root = Path(cache or "tierb_cache")
    root.mkdir(parents=True, exist_ok=True)
    init_v64_source_packs(root, cache_root)
    packs = list(only_packs or _packs_for_tests(only_tests))
    packs = [pack for pack in packs if pack in EXACT_PACKS]
    packs = _order_packs_v72(packs)

    manifests: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for pack in packs:
        manifest, pack_candidates = _harvest_pack(
            pack,
            root,
            cache_root,
            allow_network=allow_network,
            dry_run=dry_run,
            force=force,
            timeout=timeout,
            max_bytes=max_bytes,
            max_sources_per_pack=max_sources_per_pack,
            max_rows_per_source=max_rows_per_source,
            write_rows=write_rows,
        )
        manifests.append(manifest)
        for candidate in pack_candidates:
            _capture_candidate_v74(candidates, candidate)

    candidate_fields = [
        "pack",
        "url",
        "label",
        "frame_index_v67",
        "row_index_v67",
        "accepted_v67",
        "missing_required_v67",
        "row_problem_v67",
        "columns_v67",
    ]
    candidate_csv_path = root / "data" / "generated" / "public_source_harvest_candidates_v67.csv"
    candidate_gzip_path = root / "data" / "generated" / "public_source_harvest_candidates_v74.csv.gz"
    _write_csv(candidate_csv_path, candidates, candidate_fields)
    _write_csv_gzip_v74(candidate_gzip_path, candidates, candidate_fields)
    next_rows = write_next_rows_needed_v64(only_tests or [t for tests in PACK_TESTS_V64.values() for t in tests], root)
    validation = validate_v64_source_packs(root, cache_root) if run_validation else None
    usable_by_pack = _validation_usable_by_pack_v71(validation)
    pack_quality: Dict[str, Any] = {}
    quality_warnings: List[Dict[str, Any]] = []
    for manifest in manifests:
        pack = _s(manifest.get("pack"))
        written = int(manifest.get("n_accepted_rows_written_v67") or 0)
        usable = usable_by_pack.get(pack, 0)
        candidate_quality = manifest.get("candidate_quality_v71") or {}
        candidate_rows = int(candidate_quality.get("n_candidate_rows_v71") or 0)
        accepted_candidates = int(candidate_quality.get("n_candidate_rows_accepted_v71") or 0)
        attempts = manifest.get("attempts_v67") or []
        first_adapter_status = None
        if isinstance(attempts, list):
            for attempt in attempts:
                if isinstance(attempt, dict) and ("adapter" in _s(attempt.get("status_v67")).lower() or "source_specific" in json.dumps(to_jsonable(attempt)).lower()):
                    first_adapter_status = attempt
                    break
        quality = {
            "rows_written_v67": written,
            "validator_usable_rows_v64": usable,
            "candidate_quality_v71": candidate_quality,
            "coverage_v67": manifest.get("coverage_v67") or {},
            "first_adapter_status_v72": first_adapter_status,
        }
        pack_quality[pack] = quality
        if candidate_rows > 1000 and accepted_candidates == 0:
            quality_warnings.append({
                "pack": pack,
                "warning_v72": "high_candidate_count_zero_accepted_fail_fast",
                "n_candidate_rows_v72": candidate_rows,
                "top_missing_required_v71": candidate_quality.get("top_missing_required_v71", {}),
                "top_row_problems_v71": candidate_quality.get("top_row_problems_v71", {}),
                "first_adapter_status_v72": first_adapter_status,
            })
        if written > 0 and usable == 0:
            quality_warnings.append({
                "pack": pack,
                "warning_v71": "rows_written_but_zero_validator_usable_rows",
                "rows_written_v67": written,
                "validator_usable_rows_v64": usable,
                "top_row_problems_v71": (manifest.get("candidate_quality_v71") or {}).get("top_row_problems_v71", {}),
            })
        if written == 0:
            quality_warnings.append({
                "pack": pack,
                "warning_v71": "source_specific_adapter_zero_rows_or_generic_mapping_incomplete",
                "top_missing_required_v71": (manifest.get("candidate_quality_v71") or {}).get("top_missing_required_v71", {}),
            })

    summary = {
        "schema": "ccdr-v67-public-source-harvest-summary",
        "generated_utc": utc_now(),
        "outdir": str(root),
        "cache": str(cache_root),
        "allow_network_v67": allow_network,
        "dry_run_v67": dry_run,
        "no_manual_input_policy_v67": "No rows are requested from the user. Countable rows must be parsed from public downloaded/cached sources and pass v64 validation.",
        "implemented_improvements_v67": AUTOMATED_IMPROVEMENTS_V67,
        "packs_attempted_v67": packs,
        "pack_priority_order_v72": PACK_PRIORITY_V72,
        "n_packs_attempted_v67": len(packs),
        "n_rows_written_v67": int(sum(m.get("n_accepted_rows_written_v67") or 0 for m in manifests)),
        "n_sources_downloaded_or_cached_v67": int(sum(m.get("n_downloaded_or_cached_sources_v67") or 0 for m in manifests)),
        "n_structured_sources_parsed_v67": int(sum(m.get("n_parsed_structured_sources_v67") or 0 for m in manifests)),
        "candidate_quality_v71": _candidate_quality_summary_v71(candidates),
        "pack_quality_v71": pack_quality,
        "adapter_quality_warnings_v71": quality_warnings,
        "manifest_files_v67": [str(root / "data" / "generated" / f"v67_{m['pack']}_public_harvest_manifest.json") for m in manifests],
        "candidate_capture_limit_v74": _candidate_capture_limit_v74(),
        "candidate_rows_file_v67": str(candidate_csv_path),
        "candidate_rows_gzip_file_v74": str(candidate_gzip_path),
        "next_rows_needed_file_v64": str(root / "next_rows_needed_v64.json"),
        "validation_file_v64": str(root / "v64_source_pack_validation.json") if validation else None,
        "validation_summary_v64": {
            "all_existing_rows_valid_v64": validation.get("all_existing_rows_valid_v64") if isinstance(validation, dict) else None,
            "n_invalid_rows_v64": validation.get("n_invalid_rows_v64") if isinstance(validation, dict) else None,
            "n_problem_files_v64": validation.get("n_problem_files_v64") if isinstance(validation, dict) else None,
        },
        "next_rows_needed_v64": next_rows,
    }
    _write_json(root / "public_source_harvest_v67.json", summary)
    _write_json(root / "data" / "generated" / "public_source_harvest_v67.json", summary)
    return summary
