# CCDR Tier-B All Tests Confirm Report v67

Latest complete run analyzed: `tierb_out_v67_all`

Run timestamp from output directory: 2026-05-18 02:42 Europe/Kiev

## Executive Confirm Result

- Public confirms now: `T48` only.
- Public-claim check: `pass_v64 = true`.
- Claim source: `tierb_out_v67_all/confirm_only_dashboard_v64.json -> confirmed_public_now`.
- Claim summary: `tierb_out_v67_all/claim_summary_v64.json`.
- Validation artifact: `tierb_out_v67_all/v64_source_pack_validation.json`.
- Next-row manifest: `tierb_out_v67_all/next_rows_needed_v64.json`.
- Legacy confirm-like fields are not public claims.

Important nuance: `T48` full-run `result_status` is `data_limited`, but the confirm overlay allows the current public claim because `T48` is listed in `confirmed_public_now`.

## Process Health

- Total tests: 35.
- Subprocess status: 34 `ok`, 1 `process_timeout`.
- Timeout: `T32`.
- `T32` result was repaired by the v57 fallback as `data_limited_runtime_output_repaired_v57`.
- T32 stderr tail shows repeated pandas `DtypeWarning` messages from legacy generated-materials CSV reads in `tierb_runner.py:19939`.
- T44 stayed operationally healthy through the fast exact-source-required path.

## Claim Bucket Counts

| Bucket | Count | Tests |
|---|---:|---|
| confirmed_public_now | 1 | T48 |
| near_confirm_requires_exact_rows | 9 | T31, T32, T34, T44, T45, T47, T53, T57, T59 |
| diagnostic_only | 5 | T26, T27, T28, T29, T30 |
| bound_only | 3 | T50, T51, T52 |
| anchor_only | 1 | T60 |
| synthetic_or_engineering | 1 | T46 |
| data_limited | 15 | T33, T35, T36, T37, T38, T39, T40, T41, T42, T43, T49, T54, T55, T56, T58 |
| no_confirm | 0 | none |

## Source-Pack Validation

All exact source packs are still empty in this run. Validation found no invalid existing rows, but no pack is ready to attempt a confirm gate.

| Pack | Files | Rows | Validator usable rows | Status |
|---|---:|---:|---:|---|
| materials | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| materials_family_packs | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| nand | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| proteingym | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| protein_structures | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| thermoelectric | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| hepdata | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| optical_interconnect | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| neuromorphic | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| fusion | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| ldpc_external_benchmark | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |

## Per-Test Confirm Table

| Test | Process | Result | Claim bucket | Confirm status | Confirm / blocker |
|---|---|---|---|---|---|
| T26 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. |
| T27 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. |
| T28 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. |
| T29 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. |
| T30 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. |
| T31 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_source_gates_pending_v64 | Exact measured materials rows with kappa(T), grain size, source URL, and measured microstructure method are still required. |
| T32 | process_timeout | data_limited_runtime_output_repaired_v57 | near_confirm_requires_exact_rows | not_confirmed_exact_source_gates_pending_v64 | Exact measured materials rows with kappa(T), grain size, source URL, and measured microstructure method are still required. |
| T33 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T34 | ok | data_limited_no_cached_t34_orientation_rows | near_confirm_requires_exact_rows | not_confirmed_exact_te_rows_required_v64 | Exact thermoelectric ZT plus orientation/grain-boundary angle rows are still required. |
| T35 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T36 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T37 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T38 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T39 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T40 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T41 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T42 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T43 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T44 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_true_tier_a_rows_required_v64 | True Tier-A NAND rows with complete die-area, capacity, layers, and bits/cell fields are still required. |
| T45 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_benchmark_rows_required_v64 | Exact benchmark rows are still required; metadata and generated diagnostics do not count. |
| T46 | ok | ok | synthetic_or_engineering | not_confirmed_external_public_benchmark_rows_required_v64 | The current ok result is synthetic/engineering only; no external public benchmark confirm gate passes. |
| T47 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_benchmark_rows_required_v64 | Exact benchmark rows are still required; metadata and generated diagnostics do not count. |
| T48 | ok | data_limited | confirmed_public_now | compatible_positive_confirm_allowed | Confirmed public now by the v64 confirm gate. |
| T49 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T50 | ok | data_limited | bound_only | not_confirmable_by_design | Bound/constraint audit; useful for limits but not confirmable by design. |
| T51 | ok | data_limited | bound_only | not_confirmable_by_design | Bound/constraint audit; useful for limits but not confirmable by design. |
| T52 | ok | data_limited | bound_only | not_confirmable_by_design | Bound/constraint audit; useful for limits but not confirmable by design. |
| T53 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_structure_join_rows_required_v64 | ProteinGym assay rows must be joined to UniProt/PDB/AlphaFold structure features before a model confirm is allowed. |
| T54 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T55 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T56 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T57 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_hepdata_manifest_rows_required_v64 | Frozen HEPData manifests and residual rows with observed/model/uncertainty columns are still required. |
| T58 | ok | data_limited | data_limited | not_confirmed_data_limited | Required public structured physical rows are missing or insufficient for a confirm claim. |
| T59 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_hepdata_manifest_rows_required_v64 | Frozen HEPData manifests and residual rows with observed/model/uncertainty columns are still required. |
| T60 | ok | data_limited | anchor_only | anchor_only_not_full_confirm | Anchor-only consistency result; full sector confirmation requires separate quark/lattice and sensitivity gates. |

## Automated Public-Source Improvements Needed To Push More Confirms

Constraint: no manual input rows. Every row must be automatically discovered, downloaded, parsed, normalized, validated, and provenance-linked from public sources. Local exact-source packs may be used as machine caches/artifacts, but they must be produced by code, not hand-filled.

1. Fix the T32 timeout path by replacing legacy generated-CSV scans around `tierb_runner.py:19939` with the bounded v64 exact-pack reader and by disabling historical generated materials artifacts as evidence. This restores operational reliability before any T32 confirmation attempt.
2. Build an automated materials-public-source harvester for T31/T32. It should query public publisher supplements, NIST/Materials Project-like public tables where allowed, Zenodo/Figshare/OSF records, and open repository CSV/XLSX files; download only machine-readable tables; and write parsed rows to `data/exact_sources/materials` as generated cache artifacts with source URLs.
3. Add an automated materials microstructure parser for T31/T32. It should extract measured grain size and microstructure method from public tables or supplements, reject prose-only PDFs unless table extraction produces physical columns, and classify families/temperature bins automatically.
4. Add automated per-family balancing for T31/T32. The harvester should keep searching until it has public rows across semiconductor/silicon, oxide/ceramic, carbon, metal/alloy, and thermoelectric families, then emit family-balanced source packs for jackknife gates.
5. Build an automated T44 NAND spec-table harvester. It should parse public WikiChip/vendor/ISSCC/VLSI/TechInsights-style structured pages or attached tables for company, year, layers, capacity, die_area_mm2, bits_per_cell, product key, and source URL, rejecting inferred die area.
6. Add T44 product deduplication and source-domain audit. Automatically derive stable product keys and domain labels so repeated mirrors/spec copies cannot inflate evidence, then run the true Tier-A NAND gate only on deduplicated public rows.
7. Build an automated ProteinGym ingestion path for T53. Download public ProteinGym/DMS assay tables, normalize assay_id, UniProt, family, assay type, sequence cluster, variant, DMS score, and source URL, then cache only rows with machine-readable provenance.
8. Build an automated UniProt/PDB/AlphaFold structure joiner for T53. For each public ProteinGym UniProt ID, fetch public structure metadata/features, derive symmetry/contact proxies, and join them to assay rows before the model gate runs.
9. Build an automated thermoelectric source parser for T34. Search public Starrydata/teMatDb/open supplements and parse Bi2Te3/Sb2Te3 rows with ZT, temperature, composition, orientation_angle_deg or grain_boundary_angle_deg, and source URL.
10. Build an automated HEPData downloader for T57/T59. Use public HEPData APIs to discover records/tables, download YAML/CSV/JSON tables, map observed/model/uncertainty columns, and generate residual-row tables without manual manifests.
11. Build an automated optical-interconnect benchmark parser for T45. Discover public benchmark tables from papers, vendor datasets, conference supplements, and open repositories, then parse energy/bit, bandwidth, reach, platform, benchmark, year, and source URL.
12. Build an automated neuromorphic benchmark parser for T47. Discover public Loihi/SpiNNaker/TrueNorth/benchmark tables, parse chip, task, energy, accuracy, topology, year, and source URL, and reject benchmark prose without numeric table rows.
13. Build automated fusion public-table connectors for T26-T30. Use public OSF/Zenodo/Figshare/repository APIs to download machine-readable per-shot or per-timeslice tables; reject paper summaries unless exact physical rows are extracted with device, shot/time, quantity, value, unit, and source URL.
14. Build an automated external-benchmark harvester for T46. Discover public LDPC/burst-channel benchmark tables with task, metric, model score, baseline score, held-out split, uncertainty, and source URL; reject synthetic-only engineering runs.
15. Extend `validate_v64_source_packs.py` into an automated pre-confirm pipeline: discover public sources, parse candidate rows, validate required columns/provenance, reject templates/generated dashboards/derived rows, emit `next_rows_needed_v64.json`, and fail any new confirm attempt unless `confirmed_public_now` changes through strict gates only.

## Implementation Update

The 15 public-source-only improvements are now implemented as a v67 harvester/pre-confirm pipeline:

- `tierb/v67_public_source_harvesters.py`
- `harvest_v67_public_sources.py`
- `validate_v64_source_packs.py --harvest-public-sources`
- `run_confirm_only_v64.py --harvest-public-sources`
- `run_all_and_confirm_v64.py --harvest-public-sources`

Network downloads are explicit with `--allow-network`. Dry runs emit harvest plans/manifests but write no countable rows. Non-network verification still leaves public confirms at `T48` only, which is the correct guardrail because no public downloads were performed.

## Bottom Line

The latest v67 run preserves the same scientific confirm state: only `T48` is claimable. The automated harvester/validator path is now present and working, but the dry-run checks intentionally did not download public rows, so no additional tests can move to confirmation yet. The next step is a network-enabled harvest run, followed by strict v64 gate checks.
