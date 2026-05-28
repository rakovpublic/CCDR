# Tier-B All-Tests Confirmation Report, v64

Analysis date: 2026-05-16

Scope: CCDR Tier-B public-data tests T26-T60 in this workspace. This report analyzes the latest existing v64 artifacts under `tierb_out_v64_all`; it does not claim a fresh network rerun.

Primary artifacts reviewed:

- `tierb_out_v64_all/tier_b_batch_summary.json`
- `tierb_out_v64_all/v64_one_command_summary.json`
- `tierb_out_v64_all/confirm_only_dashboard_v64.json`
- `tierb_out_v64_all/confirm_targets_v64.json`
- `tierb_out_v64_all/public_claim_check_v64.json`
- `tierb_out_v64_all/confirm_only_v64/data/generated/*_gate_v64.json`
- `data/exact_sources/**`

## Executive Summary

Current public-confirm set: **T48 only**.

The v64 public-claim gate is explicit: `confirm_only_dashboard_v64.json -> confirmed_public_now` contains only `T48`. The matching `public_claim_check_v64.json` also lists only `T48`.

The full suite has 35 tests. The latest batch summary reports all 35 selected tests as represented in output JSON, with the one-command v64 wrapper returning `full_run_returncode: 0` and `confirm_only_returncode: 0`. Five full-run subprocesses timed out (`T31`, `T32`, `T44`, `T51`, `T52`), but repaired or bound-only result JSONs and the confirm-only overlay still exist.

Confirmation buckets:

- **Confirmed now:** `T48`
- **Near-confirm next:** `T31`, `T32`, `T44`, `T53`, `T34`, `T57`, `T59`, `T45`, `T47`
- **Bound-only, not confirmable by design:** `T50`, `T51`, `T52`
- **Anchor-only, not full confirm:** `T60`
- **Diagnostic/no-confirm/data-limited:** all remaining tests

The main blocker is not code coverage of the test list. The blocker is evidence quality: v64 exact-source-pack summaries report zero files and zero rows for the source packs that would promote near-confirm tests.

## Confirm Claim Language

Allowed:

- "In the current v64 run, T48 is the only public confirm."
- "T31 and T32 are high-priority near-confirms, but exact measured materials rows are still missing."
- "T60 is a positive consistency anchor, not a full confirm."
- "T50-T52 are bounds/constraint checks and should not be advertised as confirmations."

Avoid:

- Calling `T31`, `T32`, `T44`, `T53`, `T34`, `T45`, `T47`, `T57`, or `T59` confirmed before the v64 exact-source gates pass.
- Treating generated templates, dashboards, PDF summaries, or metadata-only rows as evidence.
- Treating `ok` execution status as scientific confirmation.

## All-Test Matrix

| Test | Family | Test | Full-run result | Confirm bucket | Rank | Main blocker |
|---|---|---|---|---|---:|---|
| T26 | fusion | fusion ELM energy scaling | ok/data_limited | NO CONFIRM | 1 | not_confirmed_diagnostic_only |
| T27 | fusion | ELM helicity proxy | ok/data_limited | NO CONFIRM | 1 | not_confirmed_diagnostic_only |
| T28 | fusion | global H-mode confinement/KSS margin | ok/data_limited | NO CONFIRM | 2 | not_confirmed_diagnostic_only |
| T29 | fusion | stellarator vs tokamak edge transport | ok/data_limited | NO CONFIRM | 2 | not_confirmed_diagnostic_only |
| T30 | fusion | fusion residual curvature coupling | ok/data_limited | NO CONFIRM | 1 | not_confirmed_diagnostic_only |
| T31 | materials | cryogenic kappa CCDR vs Casimir | process_timeout/data_limited_runtime_output_repaired_v57 | NEAR | 7 | not_confirmed_exact_source_gates_pending_v64 |
| T32 | materials | low-T kappa exponent | process_timeout/data_limited_runtime_output_repaired_v57 | NEAR | 7 | not_confirmed_exact_source_gates_pending_v64 |
| T33 | materials | diamond/hBN thermal ceiling audit | ok/data_limited | NO CONFIRM |  |  |
| T34 | materials | Bi2Te3 ZT angle meta-analysis | ok/data_limited_no_cached_t34_orientation_rows | NEAR | 3 | not_confirmed_exact_te_rows_required_v64 |
| T35 | materials | Kibble-Zurek grain-size exponent | ok/data_limited | NO CONFIRM |  |  |
| T36 | materials | density-stratified grain scattering | ok/data_limited | NO CONFIRM |  |  |
| T37 | materials | auxetic phonon transport | ok/data_limited | NO CONFIRM |  |  |
| T38 | materials | skyrmion lifetime literature audit | ok/data_limited | NO CONFIRM |  |  |
| T39 | materials_quantum | moire phononic bandgap twist scaling | ok/data_limited | NO CONFIRM |  |  |
| T40 | quantum | transmon phononic-substrate T1 audit | ok/data_limited | NO CONFIRM |  |  |
| T41 | quantum | qubit T2 plateau meta-analysis | ok/data_limited | NO CONFIRM |  |  |
| T42 | quantum | spin-qubit T2 in isotopically pure Si | ok/data_limited | NO CONFIRM |  |  |
| T43 | quantum | DTC qubit error-per-cycle trend | ok/data_limited | NO CONFIRM |  |  |
| T44 | electronics | 3D NAND area vs volume scaling | process_timeout/data_limited_runtime_output_repaired_v57 | NEAR | 5 | not_confirmed_true_tier_a_rows_required_v64 |
| T45 | electronics | optical interconnect trend | ok/data_limited | NEAR | 3 | not_confirmed_exact_benchmark_rows_required_v64 |
| T46 | electronics | LDPC burst-channel benchmark | ok/ok | NO CONFIRM |  |  |
| T47 | electronics | neuromorphic graph energy audit | ok/data_limited | NEAR | 3 | not_confirmed_exact_benchmark_rows_required_v64 |
| T48 | energy | photovoltaic acoustic-optical proxy | ok/ok | CONFIRM | 10 | passes_strict_gate |
| T49 | energy_materials | battery/thermoelectric materials symmetry audit | ok/data_limited | NO CONFIRM |  |  |
| T50 | sensors | Casimir residual public-table audit | ok/data_limited | BOUND | 0 | not_confirmable_by_design |
| T51 | sensors | optical-clock drift literature bound | process_timeout/bound_only | BOUND | 0 | not_confirmable_by_design |
| T52 | sensors | atom-interferometer noise-floor audit | process_timeout/bound_only | BOUND | 0 | not_confirmable_by_design |
| T53 | biotech | biological symmetry/protein-folding proxy | ok/ok | NEAR | 6 | not_confirmed_structure_join_rows_required_v64 |
| T54 | biotech | photosynthetic coherence meta-analysis | ok/data_limited | NO CONFIRM |  |  |
| T55 | aerospace | radiation/spacecraft anomaly audit | ok/data_limited | NO CONFIRM |  |  |
| T56 | aerospace | solar-sail residual proxy | ok/data_limited | NO CONFIRM |  |  |
| T57 | aerospace | cosmic-ray cross-section enhancement | ok/data_limited | NEAR | 3 | not_confirmed_hepdata_manifest_rows_required_v64 |
| T58 | aerospace | exoplanet/stellar chronometry null | ok/data_limited | NO CONFIRM |  |  |
| T59 | hep | public HEP anomaly ledger | ok/data_limited | NEAR | 3 | not_confirmed_hepdata_manifest_rows_required_v64 |
| T60 | particle | Koide sector-distance audit | ok/ok | ANCHOR | 5 | anchor_only_not_full_confirm |

## What The Confirms Actually Prove

### T48: current public confirm

`T48` is the only test passing the current v64 strict public-confirm gate. The full run reports `ok`, parsed 314 tables, and the confirm-only overlay preserves the frozen public-confirm result. The v64 frozen confirm artifact says:

- `confirmation_status_v64: compatible_positive_confirm_allowed`
- `strict_confirm_ready_v64: true`
- `rank_score_0_10_v64: 10`
- `note_v64: T48 remains frozen current public confirm; v64 only adds exact-source behavior for other tests.`

This is the claimable confirm for the present bundle.

### Near-confirms: high-value but blocked

The near-confirm group is useful, but none are claimable yet.

| Tests | Why they matter | Current v64 gate state | Best next action |
|---|---|---|---|
| T31, T32 | Materials kappa and low-temperature exponent are the highest-ranked next candidates. | 0 raw rows, 0 usable rows, 0 sources, 0 material families, model not enough rows. | Fill `data/exact_sources/materials/*.csv` and family packs with exact measured kappa(T), grain size, microstructure method, source URL. |
| T44 | 3D NAND area/volume scaling could become a strong electronics confirm. | 0 raw rows, 0 usable Tier-A rows, 0 companies. | Fill `data/exact_sources/nand/*.csv` with true Tier-A public rows: company, year, layers, capacity, die area, bits/cell, source URL. |
| T53 | Biology route has real model potential, but needs a proper DMS-to-structure join. | 0 assay rows, 0 structure rows, 0 usable join rows. | Fill ProteinGym assay rows plus UniProt/PDB/AlphaFold structure-feature rows, then require family/assay/sequence jackknife. |
| T34 | Thermoelectric angle path. | 0 raw rows, 0 usable rows. | Fill exact Bi2Te3/Sb2Te3 ZT plus angle rows and run the cos(6 theta) model. |
| T45 | Optical interconnect benchmark path. | 0 raw rows, 0 usable rows, 0 sources. | Fill exact benchmark rows with energy/bit, bandwidth, reach, platform, year, source URL. |
| T47 | Neuromorphic benchmark path. | 0 raw rows, 0 usable rows, 0 sources. | Fill exact neuromorphic rows with energy, accuracy, topology, chip, benchmark, year. |
| T57, T59 | HEPData residual/anomaly ledgers. | 0 raw rows, 0 residual rows, 0 records, 0 tables. | Fill frozen HEPData manifests and local tables with observed/model/uncertainty columns. |

### Bound-only and anchor-only

`T50`, `T51`, and `T52` are classified as `not_confirmable_by_design`. They can constrain or bound a claim, but should not be counted as confirms.

`T60` is `anchor_only_not_full_confirm`. Its result is valuable as a consistency anchor, but the report should not call it a sector-wide public confirm.

### T46 synthetic result

`T46` has `ok/ok`, but it is not in the current public-confirm set. Treat it as an engineering/synthetic benchmark result unless an external, public benchmark gate is added and passes.

## Exact Source Pack Status

v64 created exact-source-pack templates, but the generated summaries show zero evidence rows:

| Pack | Files | Rows | Affected tests |
|---|---:|---:|---|
| materials | 0 | 0 | T31, T32 |
| nand | 0 | 0 | T44 |
| proteingym | 0 | 0 | T53 |
| protein_structures | 0 | 0 | T53 |
| thermoelectric | 0 | 0 | T34 |
| hepdata | 0 | 0 | T57, T59 |
| optical_interconnect | 0 | 0 | T45 |
| neuromorphic | 0 | 0 | T47 |

Template locations already exist under `data/exact_sources/`. They are schemas only. They are not evidence and should stay excluded from confirm counts until filled with real public-source rows.

## Priority Improvements

1. Fill the exact source packs before changing confirm language.

   The fastest path to more confirms is not broad discovery. It is filling exact source rows for the v64 gates. Start with `T31/T32`, then `T53`, then `T44`, because those have the highest current ranks.

2. Add source-pack minimum gates to the report surface.

   The report should show, for each pack: file count, row count, usable row count, rejected row count, source count, and the exact failed gate. Right now those details exist in generated JSONs but are not prominent enough for claim review.

3. Make timeout behavior less ambiguous.

   `T31`, `T32`, `T44`, `T51`, and `T52` timed out in the full-run subprocess layer. Even when repaired outputs exist, the report should show both process status and scientific status. For future runs, use a larger `--script-timeout`, or split long broad-discovery tests from confirm-only exact-pack checks.

4. Preserve the public-claim gate as the single source of truth.

   Keep using `public_claim_check_v64.json` and `confirm_only_dashboard_v64.json -> confirmed_public_now` as the claim surface. This prevents accidental promotion from legacy `ok`, `support_like`, or older positive-dashboard fields.

5. Separate discovery, diagnostics, bounds, anchors, and confirms in every dashboard.

   The suite is doing the right conservative thing, but the naming is still easy to misread. Recommended buckets: `confirmed_public_now`, `near_confirm_requires_exact_rows`, `bound_only`, `anchor_only`, `diagnostic_only`, `data_limited`.

6. Add a "why not confirmed" one-liner for all 35 tests.

   The v64 confirm targets cover the current confirm-candidate subset. Extend that style to every test so non-candidate tests also expose a structured blocker and next data source.

7. For `T48`, add a compact provenance appendix.

   Since `T48` is the sole confirm, give it extra audit visibility: exact frozen artifact path, candidate row count used by the frozen gate, model formula, source families, and robustness checks. This will make the confirm easier to defend.

8. For `T60`, split anchor claims from full-sector claims.

   Keep charged-lepton/public-constant consistency separate from any quark/lattice-sector claim. Require public PDG/FLAG values, uncertainties, sector definitions, and sensitivity tests before considering anything beyond anchor language.

## Recommended Next Commands

Refresh confirm-only gates after adding exact source rows:

```powershell
python run_confirm_only_v64.py --outdir tierb_out_v64_all/confirm_only_v64 --cache tierb_cache_v64_all
```

Refresh the one-command summary without rerunning all broad tests:

```powershell
python run_all_and_confirm_v64.py --skip-full-run --outdir tierb_out_v64_all --cache tierb_cache_v64_all
```

Full rerun when source packs or runner behavior changed:

```powershell
python run_all_and_confirm_v64.py --cache tierb_cache_v64_all --outdir tierb_out_v64_all --timeout 240 --max-tables 300 --script-timeout 1800 --force
```

## Bottom Line

For current v64 claims, use this strict statement:

> The current public-confirm list contains exactly one test: T48, photovoltaic acoustic-optical proxy. The strongest next candidates are T31/T32, T53, and T44, but all require filled exact source packs before they can become confirms.

