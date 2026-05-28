# Tier-B All-Tests Confirmation Report, v65 Output

Analysis date: 2026-05-17

Scope: current Tier-B public-data outputs in `tierb_out_v65_all`, covering tests T26-T60. The active public-claim gate is still the v64 exact-source confirm overlay copied into the v65 output directory.

Primary artifacts reviewed:

- `tierb_out_v65_all/tier_b_batch_summary.json`
- `tierb_out_v65_all/v64_one_command_summary.json`
- `tierb_out_v65_all/confirm_only_dashboard_v64.json`
- `tierb_out_v65_all/confirm_targets_v64.json`
- `tierb_out_v65_all/public_claim_check_v64.json`
- `tierb_out_v65_all/source_pack_status_v64.json`
- `tierb_out_v65_all/confirm_only_v64/*_result.json`

## Executive Summary

Current public confirms: **T48 only**.

`public_claim_check_v64.json` reports `pass_v64: true` and `confirmed_public_now: ["T48"]`. This is the only claim surface that should be used for public confirmation language.

Current bucket counts:

- Confirmed public now: 1 test, `T48`
- Near-confirm requiring exact rows: 9 tests, `T31`, `T32`, `T34`, `T44`, `T45`, `T47`, `T53`, `T57`, `T59`
- Bound-only: 3 tests, `T50`, `T51`, `T52`
- Anchor-only: 1 test, `T60`
- Diagnostic-only: 5 fusion tests, `T26`-`T30`
- Synthetic/engineering only: 1 test, `T46`
- Data-limited/no confirm: 15 tests

The full v65 batch contains all 35 tests. Four full-run subprocesses timed out: `T31`, `T32`, `T44`, and `T53`. The confirm overlay still generated per-test claim statuses for all 35 tests.

## Confirm Language

Allowed:

- "The current public-confirm list contains exactly one test: T48."
- "T31/T32, T53, and T44 are the strongest next confirm candidates, but exact public rows are still missing."
- "T50-T52 are useful bound/constraint audits, not confirmations."
- "T60 is an anchor-only consistency result, not a full sector confirm."

Not allowed:

- Calling near-confirm tests confirmed before exact-source gates pass.
- Treating `ok` execution as confirmation.
- Treating generated templates, generated diagnostics, broad-discovery metadata, or PDF/table summaries as evidence rows.

## All-Test Matrix

| Test | Bucket | Confirm status | Full-run process/result | Rank | What blocks confirm | Next data source |
|---|---|---|---|---:|---|---|
| T26 | diagnostic_only | not_confirmed_diagnostic_only | ok/data_limited | 1 | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. | Certified fusion per-shot ELM energy rows with pedestal pressure/volume/drop columns. |
| T27 | diagnostic_only | not_confirmed_diagnostic_only | ok/data_limited | 1 | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. | Certified fusion RMP/helicity rows with ELM frequency and coil/phasing columns. |
| T28 | diagnostic_only | not_confirmed_diagnostic_only | ok/data_limited | 2 | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. | Exact ITPA/H-mode rows with tau_E, density, and transport columns. |
| T29 | diagnostic_only | not_confirmed_diagnostic_only | ok/data_limited | 2 | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. | Exact stellarator/tokamak edge transport rows with device and diffusivity/heat-flux columns. |
| T30 | diagnostic_only | not_confirmed_diagnostic_only | ok/data_limited | 1 | Fusion route is diagnostic until certified exact public per-shot/per-timeslice physical rows pass the gate. | Exact confinement residual rows with density plus shaping/curvature columns. |
| T31 | near_confirm_requires_exact_rows | not_confirmed_exact_source_gates_pending_v64 | process_timeout/data_limited_runtime_output_repaired_v57 | 7 | Exact measured materials rows with kappa(T), grain size, source URL, and measured microstructure method are still required. | Filled exact materials packs with measured kappa(T), grain size, source URL, and SEM/TEM/XRD/EBSD microstructure method. |
| T32 | near_confirm_requires_exact_rows | not_confirmed_exact_source_gates_pending_v64 | process_timeout/data_limited_runtime_output_repaired_v57 | 7 | Exact measured materials rows with kappa(T), grain size, source URL, and measured microstructure method are still required. | Filled exact materials packs with measured low-temperature kappa(T), grain size, source URL, and microstructure method. |
| T33 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured materials rows with the named physical columns required by this test. |
| T34 | near_confirm_requires_exact_rows | not_confirmed_exact_te_rows_required_v64 | ok/data_limited_no_cached_t34_orientation_rows | 3 | Exact thermoelectric ZT plus orientation/grain-boundary angle rows are still required. | Exact Bi2Te3/Sb2Te3 ZT plus orientation/grain-boundary angle rows. |
| T35 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured materials rows with the named physical columns required by this test. |
| T36 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured materials rows with the named physical columns required by this test. |
| T37 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured materials rows with the named physical columns required by this test. |
| T38 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured materials rows with the named physical columns required by this test. |
| T39 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured materials_quantum rows with the named physical columns required by this test. |
| T40 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured quantum rows with the named physical columns required by this test. |
| T41 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured quantum rows with the named physical columns required by this test. |
| T42 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured quantum rows with the named physical columns required by this test. |
| T43 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured quantum rows with the named physical columns required by this test. |
| T44 | near_confirm_requires_exact_rows | not_confirmed_true_tier_a_rows_required_v64 | process_timeout/data_limited_runtime_output_repaired_v57 | 5 | True Tier-A NAND rows with complete die-area, capacity, layers, and bits/cell fields are still required. | True Tier-A NAND rows: company, year, layers, capacity, die area, bits/cell, source URL. |
| T45 | near_confirm_requires_exact_rows | not_confirmed_exact_benchmark_rows_required_v64 | ok/data_limited | 3 | Exact benchmark rows are still required; metadata and generated diagnostics do not count. | Exact optical-interconnect benchmark rows with energy/bit, bandwidth, reach, platform, year, and source URL. |
| T46 | synthetic_or_engineering | not_confirmed_data_limited | ok/ok | 1 | The current ok result is synthetic/engineering only; no external public benchmark confirm gate passes. | Exact public structured electronics rows with the named physical columns required by this test. |
| T47 | near_confirm_requires_exact_rows | not_confirmed_exact_benchmark_rows_required_v64 | ok/data_limited | 3 | Exact benchmark rows are still required; metadata and generated diagnostics do not count. | Exact neuromorphic benchmark rows with chip, task, energy, accuracy, topology, year. |
| T48 | confirmed_public_now | compatible_positive_confirm_allowed | ok/data_limited | 10 | None. This is the only current public confirm. | Robustness/provenance audit only; no extra source rows required for current claim. |
| T49 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured energy-materials rows with the named physical columns required by this test. |
| T50 | bound_only | not_confirmable_by_design | ok/data_limited | 0 | Bound/constraint audit; useful for limits but not confirmable by design. | Bound-table evidence only; this test is not confirmable by design. |
| T51 | bound_only | not_confirmable_by_design | ok/data_limited | 0 | Bound/constraint audit; useful for limits but not confirmable by design. | Bound-table evidence only; this test is not confirmable by design. |
| T52 | bound_only | not_confirmable_by_design | ok/data_limited | 0 | Bound/constraint audit; useful for limits but not confirmable by design. | Bound-table evidence only; this test is not confirmable by design. |
| T53 | near_confirm_requires_exact_rows | not_confirmed_structure_join_rows_required_v64 | process_timeout/data_limited_runtime_output_repaired_v57 | 6 | ProteinGym assay rows must be joined to UniProt/PDB/AlphaFold structure features before a model confirm is allowed. | ProteinGym assay rows joined to UniProt/PDB/AlphaFold structure-feature rows. |
| T54 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured biotech rows with the named physical columns required by this test. |
| T55 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured aerospace rows with the named physical columns required by this test. |
| T56 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured aerospace rows with the named physical columns required by this test. |
| T57 | near_confirm_requires_exact_rows | not_confirmed_hepdata_manifest_rows_required_v64 | ok/data_limited | 3 | Frozen HEPData manifests and residual rows with observed/model/uncertainty columns are still required. | Frozen HEPData manifest plus local tables with observed/model/uncertainty residual columns. |
| T58 | data_limited | not_confirmed_data_limited | ok/data_limited | 1 | Required public structured physical rows are missing or insufficient for a confirm claim. | Exact public structured aerospace rows with the named physical columns required by this test. |
| T59 | near_confirm_requires_exact_rows | not_confirmed_hepdata_manifest_rows_required_v64 | ok/data_limited | 3 | Frozen HEPData manifests and residual rows with observed/model/uncertainty columns are still required. | Frozen HEPData manifest plus local tables with observed/model/uncertainty residual columns. |
| T60 | anchor_only | anchor_only_not_full_confirm | ok/data_limited | 5 | Anchor-only consistency result; full sector confirmation requires separate quark/lattice and sensitivity gates. | Separate charged-lepton/public-constant anchor from any quark/lattice sector claim. |

## Exact Source Pack Status

All exact source packs are still empty in the current v65 output: every pack has `n_files_v64: 0`, `n_rows_v64: 0`, and `max_usable_rows_v64: 0`.

| Pack | Affected tests | Status |
|---|---|---|
| materials | T31, T32 | needs filled exact public rows |
| materials_family_packs | T31, T32 | needs filled exact public rows |
| nand | T44 | needs filled exact public rows |
| proteingym | T53 | needs filled exact public rows |
| protein_structures | T53 | needs filled exact public rows |
| thermoelectric | T34 | needs filled exact public rows |
| hepdata | T57, T59 | needs filled exact public rows |
| optical_interconnect | T45 | needs filled exact public rows |
| neuromorphic | T47 | needs filled exact public rows |
| fusion | T26-T30 | needs filled exact public rows |

## Current Confirm Interpretation

### T48 is the only public confirm

`T48` has confirm status `compatible_positive_confirm_allowed`, rank 10, and bucket `confirmed_public_now`. The confirm overlay reports `n_candidate_rows_v61: 63074`. The full-run batch row for T48 is `ok/data_limited`, which is confusing but not decisive for claim language: the public claim gate is `public_claim_check_v64.json`, and it still permits only T48.

Suggested improvement: reconcile the top-level T48 full-run `result_status` with the confirm overlay, or add an explicit top-level note saying that full-run legacy status and confirm-overlay status are separate.

### Near-confirms are blocked by evidence, not by dashboard logic

The strongest next candidates are `T31`, `T32`, `T53`, and `T44`. They have useful model routes and ranks, but the exact packs currently have zero rows. They should not be called confirms until exact public rows are added and the gates pass.

### Bound-only and anchor-only results are useful but not confirms

`T50`, `T51`, and `T52` are constraint/bound checks. `T60` is a positive consistency anchor, and its own split says the quark/lattice sector still requires public PDG/FLAG values, explicit sector definitions, mass-scheme sensitivity, and a separate full-sector gate.

## Priority Improvements For More Confirms

1. Fill exact source packs, starting with `T31/T32`.

   Add real public rows under the relevant `data/exact_sources/` pack, not generated artifacts. For `T31/T32`, the minimum useful rows need material, temperature, kappa, grain size, measured microstructure method, source URL, and enough source/family/temperature diversity.

2. Fix or isolate long-running tests.

   `T31`, `T32`, `T44`, and `T53` timed out in the full-run layer. Keep the confirm-only overlay fast, but either raise `--script-timeout` for full runs or split broad discovery from exact-pack gate checks.

3. Add a top-level "claim summary" to every run.

   Put `confirmed_public_now`, `near_confirm_requires_exact_rows`, `bound_only`, `anchor_only`, `diagnostic_only`, and `synthetic_or_engineering` in one short JSON object so a reviewer cannot accidentally read legacy `status: ok` as confirmation.

4. Create a T48 provenance appendix artifact.

   T48 is the sole public confirm, so it should have a compact audit appendix: frozen artifact path, source/candidate row count, model formula when available, family buckets, and robustness keys. The current overlay has partial provenance, but some fields are null because it points at the confirm-only result rather than the full top-level result.

5. Promote exact-pack row templates into contributor-facing checklists.

   For each near-confirm source pack, add a README checklist with required columns, accepted units, minimum row counts, and examples of rows that will be rejected. This will make it much easier to add evidence without accidentally counting templates or diagnostics.

6. Keep T46 out of public confirms until it has an external benchmark gate.

   T46 is `ok/ok` in the full batch, but the confirm overlay correctly buckets it as synthetic/engineering only. A future confirm path should require public benchmark data, matched baselines, seeds, code/config provenance, and an external reproducibility record.

## Bottom Line

Use this strict claim:

> The current v65 output has exactly one public confirm: T48. The best next confirm candidates are T31/T32, T53, and T44, but all near-confirms still require filled exact public source packs before they can be claimed.

