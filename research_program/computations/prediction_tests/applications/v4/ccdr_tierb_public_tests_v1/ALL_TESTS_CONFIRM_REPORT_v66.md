# CCDR Tier-B All Tests Confirm Report v66

Latest complete run analyzed: `tierb_out_v66_all`

Run date observed from output directory: 2026-05-18

## Executive Confirm Result

- Public confirms now: `T48` only.
- Public-claim check: `pass_v64 = true`.
- Claim source: `tierb_out_v66_all/confirm_only_dashboard_v64.json -> confirmed_public_now`.
- Claim summary: `tierb_out_v66_all/claim_summary_v64.json`.
- Legacy confirm-like fields are not public claims.

Important nuance: `T48` full-run `result_status` is still `data_limited`, but the confirm overlay allows the current public claim because `T48` is listed in `confirmed_public_now`. The reconciliation is recorded in `tierb_out_v66_all/t48_provenance_appendix_v64.json`.

## Process Health

- Total tests: 35.
- Subprocess status: 35 `ok`, 0 timeout, 0 process error.
- This is an operational improvement over the earlier v65 run: the full suite no longer shows process timeouts.
- T44 used the v65 fast exact-source-required path and finished as `data_limited` rather than re-reading historical generated NAND artifacts.

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

## Exact Source Pack Status

All exact source packs are still empty in this run. No non-template rows were counted.

| Pack | Files | Rows | Max usable rows | Status |
|---|---:|---:|---:|---|
| materials | 0 | 0 | 0 | needs_filled_exact_public_rows |
| materials_family_packs | 0 | 0 | 0 | needs_filled_exact_public_rows |
| nand | 0 | 0 | 0 | needs_filled_exact_public_rows |
| proteingym | 0 | 0 | 0 | needs_filled_exact_public_rows |
| protein_structures | 0 | 0 | 0 | needs_filled_exact_public_rows |
| thermoelectric | 0 | 0 | 0 | needs_filled_exact_public_rows |
| hepdata | 0 | 0 | 0 | needs_filled_exact_public_rows |
| optical_interconnect | 0 | 0 | 0 | needs_filled_exact_public_rows |
| neuromorphic | 0 | 0 | 0 | needs_filled_exact_public_rows |
| fusion | 0 | 0 | 0 | needs_filled_exact_public_rows |

## Per-Test Confirm Table

| Test | Process | Result | Claim bucket | Confirm status | Confirm / blocker |
|---|---|---|---|---|---|
| T26 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. |
| T27 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. |
| T28 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. |
| T29 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. |
| T30 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. |
| T31 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_source_gates_pending_v64 | Exact measured materials rows with kappa(T), grain size, source URL, and measured microstructure method are still required. |
| T32 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_source_gates_pending_v64 | Exact measured materials rows with kappa(T), grain size, source URL, and measured microstructure method are still required. |
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
| T45 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_benchmark_rows_required_v64 | Exact optical-interconnect benchmark rows are still required; metadata and generated diagnostics do not count. |
| T46 | ok | ok | synthetic_or_engineering | synthetic_or_engineering_not_public_confirm | The current ok result is synthetic/engineering only; no external public benchmark confirm gate passes. |
| T47 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_benchmark_rows_required_v64 | Exact neuromorphic benchmark rows are still required; metadata and generated diagnostics do not count. |
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

## Confirmation-Push Improvements

These are the highest-value improvements that can move tests toward confirmation. None of them should fabricate rows or count generated artifacts as evidence.

1. Fill the materials exact pack for T31/T32. Add measured `kappa(T)`, `grain_size_nm`, `material_family`, `temperature_K`, source URL, and measured microstructure method rows into `data/exact_sources/materials`. Gate target: at least 50 usable rows, 5 sources, 5 families, and 3 temperature bins.
2. Add per-family materials packs for T31/T32. Use `data/exact_sources/materials/families` to balance silicon/semiconductor, oxide/ceramic, carbon, metal/alloy, and thermoelectric families. This directly supports the family/source/temperature jackknife gates.
3. Build a T44 true Tier-A NAND pack. Add rows to `data/exact_sources/nand` with company, year, layers, capacity_Gb, die_area_mm2, bits_per_cell, and source_url. Gate target: at least 8 usable true Tier-A rows and 3 companies for the v64 gate, then expand toward the stricter older 20-row robustness target.
4. Add a T44 source-domain audit column and dedup key. Include product_or_paper/source_label plus a stable product key so repeated vendor/spec rows do not inflate evidence. This helps T44 become confirmable without re-opening the expensive legacy generated-artifact path.
5. Create the ProteinGym assay pack for T53. Fill `data/exact_sources/proteingym` with assay_id, UniProt, family, assay type, sequence cluster, variant, DMS score, and source URL. Gate target: at least 50 usable joined rows across 3+ families, 3+ assays, and 10+ sequence clusters.
6. Create the T53 protein structure feature pack. Fill `data/exact_sources/protein_structures` with UniProt to PDB/AlphaFold IDs plus oligomeric state, symmetry/contact proxy, fold class, and source URL. This is required before T53 can test the structure-DMS model gate.
7. Fill the T34 thermoelectric angle pack. Add exact Bi2Te3/Sb2Te3 rows with composition, ZT, temperature_K, orientation_angle_deg or grain_boundary_angle_deg, and source URL. Gate target: at least 30 usable rows before the cos(6theta) model can matter.
8. Build frozen HEPData manifests for T57/T59. Add record_id, table_id, x column, observed column, model column, uncertainty column, observable name, local table path, and source URL into `data/exact_sources/hepdata`. Gate target: at least 3 records, 3 tables, and 20 residual rows.
9. Attach local HEPData tables for T57/T59. The manifest is not enough by itself; the listed local CSV/YAML tables must exist and expose observed/model/uncertainty columns so residual rows can be computed.
10. Fill the T45 optical interconnect benchmark pack. Add platform, year, energy_per_bit_pJ, bandwidth_Gbps, reach_m, benchmark, and source URL into `data/exact_sources/optical_interconnect`. Gate target: at least 20 usable rows and 3 sources.
11. Fill the T47 neuromorphic benchmark pack. Add chip, benchmark/task, energy_per_inference_or_spike_pJ, accuracy, topology, year, and source URL into `data/exact_sources/neuromorphic`. Gate target: at least 20 usable rows and 3 sources.
12. Add certified fusion exact-row attachments for T26-T30. Fill `data/exact_sources/fusion` with per-shot or per-timeslice physical rows, device, shot, quantity, value, unit, source URL, and test_id. This can move fusion tests out of diagnostic-only once exact physical rows pass the gate.
13. Add an external public benchmark gate for T46. Current T46 is `ok` but synthetic/engineering-only. To push it toward confirmation, add public benchmark rows with task definition, metric, baseline, uncertainty or held-out score, and source URL.
14. Add a source-pack validator command. It should fail if a pack has only templates, generated outputs, missing required columns, missing source URLs, duplicate evidence keys, or rows marked derived/inferred where exact rows are required.
15. Add per-test "next row needed" manifests. For every near-confirm test, emit a short JSON listing the minimum missing row groups and exact pack path. This makes it easier to fill the right rows without accidentally adding non-counting metadata.

## Implementation Status For These Improvements

Implemented in the codebase after this report:

- Source-pack schemas and disabled accepted/rejected row examples are generated for every exact-source pack.
- Per-family materials templates are generated under `data/exact_sources/materials/families`.
- A new T46 external public benchmark pack and gate were added at `data/exact_sources/ldpc_external_benchmark`.
- `validate_v64_source_packs.py` emits `v64_source_pack_validation.json`.
- Confirm-only runs now emit `next_rows_needed_v64.json` and per-test `*_next_rows_needed_v64.json` files.
- `run_all_and_confirm_v64.py` copies `v64_source_pack_validation.json` and `next_rows_needed_v64.json` to the top-level output.

Validation after implementation preserved the public-claim set as `T48` only. The new artifacts make more tests easier to push toward confirmation, but real public rows are still required.

## Bottom Line

The latest complete run is operationally healthy: all tests ran successfully, and no process timeouts occurred. Scientifically, the public-confirm state remains unchanged: only `T48` is claimable now. Every other positive-looking route is either exact-row-blocked, diagnostic-only, bound-only, anchor-only, synthetic/engineering-only, or broadly data-limited.
