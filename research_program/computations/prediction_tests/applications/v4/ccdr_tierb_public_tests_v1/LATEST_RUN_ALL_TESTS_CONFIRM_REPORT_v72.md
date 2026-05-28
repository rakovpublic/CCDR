# Latest Run All-Tests Confirm Report v72

Run analyzed: `tierb_out_v72_all`

Supplemental validation run: `tierb_out_v72_analysis_validation`

Generated: 2026-05-18 23:45 +03:00

## Executive Summary

- Full batch artifacts are present for all `35/35` tests: `T26` through `T60`.
- Process health: `35/35` tests have `process_status=ok`.
- Current public confirms from `confirm_only_dashboard_v59.json`: `T48` only.
- Public claim check: pass, because `public_claim_check_v59.json` allows only `T48`.
- Near-confirm queue remains: `T31`, `T32`, `T44`, `T53`, `T34`, `T57`, `T59`, `T45`, `T47`.
- Result-status counts: `28` data-limited, `4` ok, `2` partial, `1` data-limited no cached T34 orientation rows.
- Confirmation labels: `1` compatible positive (`T48`), `9` near/audit next gates, `5` diagnostic-only, `3` not-confirmable-by-design, `1` anchor-only, and the rest data-limited.
- The v72 wrapper/harvest phase did not finish cleanly: there is no `v64_one_command_summary.json`, no top-level `public_source_harvest_v67.json`, and no v72 top-level source-pack validation file in `tierb_out_v72_all`.
- The last v72 cache writes were ProteinGym files at `23:22`; no completed v72 ProteinGym harvest manifest was emitted.
- I ran local validation without network after the landed artifacts. It found `T46`'s LDPC pack is now validator-ready: `250` usable public rows.
- A quick confirm-only retry for `T46`/`T48` could not complete because Python/oneMKL failed to load under Windows paging pressure. That is a local resource blocker, not a scientific gate result.

Bottom line: the latest full-test dashboard still has only `T48` as claimable. The most important new development is that v72 produced a real, validator-usable LDPC external benchmark pack for `T46`. `T46` should be the next fast confirm retry once memory/pagefile pressure is cleared.

## Confirm Buckets

- Confirmed public now: `T48`
- Ready to recheck after v72 harvest: `T46`
- Near confirm, exact rows or next gates required: `T31`, `T32`, `T34`, `T44`, `T45`, `T47`, `T53`, `T57`, `T59`
- Diagnostic only: `T26`, `T27`, `T28`, `T29`, `T30`
- Bound only by design: `T50`, `T51`, `T52`
- Anchor only: `T60`
- Data-limited/open exact-source tests: `T33`, `T35`, `T36`, `T37`, `T38`, `T39`, `T40`, `T41`, `T42`, `T43`, `T49`, `T54`, `T55`, `T56`, `T58`

Claim counts:

| Bucket | Count |
|---|---:|
| confirmed_public_now | 1 |
| ready_to_recheck_after_harvest | 1 |
| near_confirm_requires_exact_rows_or_gate | 9 |
| diagnostic_only | 5 |
| data_limited | 15 |
| bound_only | 3 |
| anchor_only | 1 |

## Harvest And Validation

The v72 run emitted per-pack harvest manifests for `8` packs, not the full set. Those completed manifests report:

- `2,004` source attempts
- `1,158` downloaded or cached sources
- `708` structured sources parsed
- `210,353` candidate rows inspected
- `283` accepted before dedup
- `250` rows written after dedup
- All written rows came from `ldpc_external_benchmark`

Local post-run validation result:

- `all_existing_rows_valid_v64=false`
- `n_empty_required_packs=10`
- `n_problem_packs=1`
- The problem pack is stale `proteingym/AUTO_PUBLIC_ROWS_V67.csv` from the earlier manifest-row failure.
- `ldpc_external_benchmark` passes with `250` validator-usable rows and is ready for a confirm gate attempt.

| Pack | Tests | v72 candidates | Rows written | Validator usable | Status |
|---|---|---:|---:|---:|---|
| fusion | T26-T30 | 8,987 | 0 | 0 | empty exact rows |
| materials | T31/T32 | 3,009 | 0 | 0 | empty exact rows |
| materials_family_packs | T31/T32 | 192,446 | 0 | 0 | empty exact rows |
| thermoelectric | T34 | 2,392 | 0 | 0 | empty exact rows |
| nand | T44 | 481 | 0 | 0 | empty exact rows |
| optical_interconnect | T45 | 1,046 | 0 | 0 | empty exact rows |
| ldpc_external_benchmark | T46 | 1,051 | 250 | 250 | validator-ready |
| neuromorphic | T47 | 941 | 0 | 0 | empty exact rows |
| proteingym | T53 | no completed v72 manifest | 0 new | 0 | stale invalid rows remain |
| protein_structures | T53 | no completed v72 manifest | 0 | 0 | empty exact rows |
| hepdata | T57/T59 | no completed v72 manifest | 0 | 0 | empty exact rows |

## Candidate-Ledger Blockers

| Pack | Main blocker |
|---|---|
| fusion | Generic/public tables still do not expose certified per-shot/per-timeslice `device`, `shot`, `time_or_slice`, `quantity`, and value rows. |
| materials | Kappa-like rows exist, but validator rows still miss `grain_size_nm`, `microstructure_method`, and `boundary_density_proxy`. |
| materials_family_packs | Many staged rows, but no complete family/sample/material/temperature/kappa/grain/microstructure tuples. |
| thermoelectric | Current sources miss the exact `ZT`, composition, orientation angle, and grain-boundary angle schema. |
| nand | All candidate rows still miss numeric `die_area_mm2`; most also miss capacity, company, year, layers, or bits per cell. |
| optical_interconnect | Rows still miss `energy_per_bit_pJ`, bandwidth, reach, platform, or year. |
| ldpc_external_benchmark | Now passes validation with 250 usable rows; needs confirm-only rerun. |
| neuromorphic | Rows still miss energy per inference/spike and many miss accuracy/chip/year/topology. |
| proteingym | Raw files were cached, but no v72 manifest completed; old manifest metadata rows still poison validation. |
| protein_structures | No usable UniProt/PDB/AlphaFold rows yet. |
| hepdata | v72 did not emit a HEPData manifest; T57/T59 still need API table rows with observed/model/uncertainty fields. |

## All Tests

| Test | Process | Result status | Public-confirm status | Main blocker or next action |
|---|---|---|---|---|
| T26 | ok | data_limited | diagnostic_only | Exact fusion ELM per-shot/per-timeslice public rows required. |
| T27 | ok | data_limited | diagnostic_only | Exact fusion RMP/helicity public rows required. |
| T28 | ok | data_limited | diagnostic_only | Exact ITPA/H-mode row table required. |
| T29 | ok | data_limited | diagnostic_only | Exact stellarator/tokamak edge-transport row table required. |
| T30 | ok | data_limited | diagnostic_only | Exact confinement residual rows required. |
| T31 | ok | partial | near_confirm_next | Measured kappa(T), grain size, microstructure, source URL, and grouped robustness gates required. |
| T32 | ok | partial | near_confirm_next | Same measured microstructure gate as T31, especially low-temperature kappa rows. |
| T33 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T34 | ok | data_limited_no_cached_t34_orientation_rows | near_confirm_next | Exact thermoelectric orientation/grain-boundary rows required. |
| T35 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T36 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T37 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T38 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T39 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T40 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T41 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T42 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T43 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T44 | ok | data_limited | audit_repair_route | True Tier-A NAND rows required: company, year, layers, capacity, die area, bits/cell, source URL. |
| T45 | ok | data_limited | near_confirm_next | Exact optical benchmark rows required. |
| T46 | ok | ok | ready_to_recheck_after_harvest | LDPC source pack now validates with 250 usable rows; rerun confirm-only when resources allow. |
| T47 | ok | data_limited | near_confirm_next | Exact neuromorphic benchmark rows required. |
| T48 | ok | ok | confirmed_public_now | Still the only current public confirm. |
| T49 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T50 | ok | data_limited | bound_only | Constraint/upper-limit route only; not a positive confirm path. |
| T51 | ok | data_limited | bound_only | Constraint/upper-limit route only; not a positive confirm path. |
| T52 | ok | data_limited | bound_only | Constraint/upper-limit route only; not a positive confirm path. |
| T53 | ok | ok | near_confirm_next | Needs valid ProteinGym variant rows joined to UniProt/PDB/AlphaFold features. |
| T54 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T55 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T56 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T57 | ok | data_limited | near_confirm_next | HEPData observed/model/uncertainty residual rows required. |
| T58 | ok | data_limited | data_limited | Test-specific exact structured public rows missing. |
| T59 | ok | data_limited | near_confirm_next | HEPData observed/model/uncertainty residual rows required. |
| T60 | ok | ok | anchor_only | Positive consistency anchor only; full confirm requires T60b/T60c/T60d gates. |

## Confirm-Focused Improvements

All of these preserve the rule that rows must be parsed from public sources, not manually supplied.

1. Run the local confirm-only retry for `T46` after memory/pagefile pressure clears: `python .\run_confirm_only_v64.py --outdir .\tierb_out_v72_confirm_only_after_ldpc --cache .\tierb_cache_v72_all --only T46 T48`.
2. Add a post-harvest affected-test confirm overlay to `run_all_and_confirm_v64.py`. If a harvest writes validator-usable rows, immediately rerun confirm-only for those affected tests so cases like `T46` are not missed.
3. Add write-on-failure checkpoint summaries. v72 lacks `v64_one_command_summary.json` and the top-level harvest/validation summaries, so the wrapper should emit partial status after every pack.
4. Quarantine or delete stale invalid auto rows before validation. The old `proteingym/AUTO_PUBLIC_ROWS_V67.csv` still contains reference-manifest metadata rows and keeps `T53` validation dirty.
5. Add a valid-only output rule per exact-source pack: write `AUTO_PUBLIC_ROWS_V67.csv` only after candidate rows pass the pack validator, otherwise keep them in candidate ledgers/staging.
6. Turn the cached ProteinGym raw downloads into an offline parser pass. v72 cached large raw ProteinGym files but did not emit a completed manifest, so use the cache to build variant-level rows without another network pass.
7. Add a ProteinGym raw-file checkpoint every N files. That prevents a long ProteinGym run from ending with cache files but no summary or usable rows.
8. Complete T53 by joining valid ProteinGym variants to UniProt accessions, then to AlphaFold/PDB structure features, then running the family/assay/sequence jackknife gate.
9. Add a T53 stale-row guard to validation: if every ProteinGym row has `proteingym_reference_manifest_is_metadata_not_variant_scores`, report that as a cleanup action before model work.
10. Build an automated materials partial-row joiner from `materials_partial_rows_v71.csv` and `materials_family_partial_rows_v71.csv`, keyed by DOI/source/sample/material/temperature.
11. Add source-specific materials supplement parsers for nanocrystalline grain-size and microstructure tables, then join them to public kappa(T) tables.
12. For T44, add a NAND source-specific text/table postprocessor that extracts die area in `mm2`, capacity in `Gb/Tb`, layer count, bits per cell, company, product family, and year from TechInsights/AnandTech/Semiconductor Engineering-style pages.
13. For T44, add product-alias joins across public source pages so die area from one source can join capacity/layers/bits-per-cell from another source with separate provenance URLs.
14. For T45/T47, add strict benchmark text adapters that only write rows when the energy, performance, platform/chip, year, and benchmark fields are all numeric/complete.
15. For T57/T59, seed explicit HEPData API record/table endpoints and checkpoint each table download; v72 did not emit a HEPData manifest.
16. For T34, target public thermoelectric supplement files with orientation/grain-boundary fields instead of generic thermoelectric pages.
17. Add pack ordering and time-budget controls. Run fast/high-probability packs (`ldpc_external_benchmark`, `nand`, `materials`, `hepdata`) before long ProteinGym raw downloads, or split ProteinGym into its own resumable job.
18. Add a pack-quality fail-fast rule: if `candidate_rows > 1000` and `accepted_rows == 0`, write top missing columns, top row problems, and the first source-specific adapter that should be improved.

## Highest-Probability Confirm Path

`T46` is now the fastest likely new confirm because its LDPC exact-source pack has `250` validator-usable public rows. The only blocker I hit was local resource pressure during the confirm-only retry.

After that, the best confirm candidates are:

1. `T53`, if stale invalid ProteinGym rows are quarantined and cached raw DMS files are parsed into variant-level rows.
2. `T44`, if the NAND adapter starts joining die area/capacity/layers/bits-per-cell across source-specific public pages.
3. `T31`/`T32`, if staged materials rows are joined into complete kappa/grain/microstructure tuples.
4. `T57`/`T59`, if HEPData API table endpoints are harvested before the long ProteinGym stage.
