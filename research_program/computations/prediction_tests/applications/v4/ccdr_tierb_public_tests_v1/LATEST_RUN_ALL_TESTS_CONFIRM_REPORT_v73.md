# Latest Run All-Tests Confirm Report v73

Run analyzed: `tierb_out_v73_all`

Generated: 2026-05-19 08:36 +03:00

## Executive Summary

- `tierb_out_v73_all` is the latest complete all-tests run. It contains all `35/35` result files from `T26` through `T60`.
- Process health is good: `35/35` tests have `process_status=ok`, and the wrapper finished with `full_run_returncode=0` and `confirm_only_returncode=0`.
- Public claim check passes, but the only current public confirm is still `T48`.
- Confirm buckets from `confirm_only_dashboard_v64.json`:
  - `confirmed_public_now`: `T48`
  - `near_confirm_requires_exact_rows`: `T31`, `T32`, `T34`, `T44`, `T45`, `T47`, `T53`, `T57`, `T59`
  - `synthetic_or_engineering`: `T46`
  - `diagnostic_only`: `T26`, `T27`, `T28`, `T29`, `T30`
  - `bound_only`: `T50`, `T51`, `T52`
  - `anchor_only`: `T60`
  - `data_limited`: `T33`, `T35`, `T36`, `T37`, `T38`, `T39`, `T40`, `T41`, `T42`, `T43`, `T49`, `T54`, `T55`, `T56`, `T58`
- The v73 network harvest ran across all 11 source packs and parsed `946` structured sources.
- It wrote `413` exact-source rows total: `261` LDPC external benchmark rows and `152` protein-structure rows.
- Validation is not clean: `all_existing_rows_valid_v64=false`, with `283` invalid ProteinGym rows in one quarantined CSV still being counted by the validator.
- `T46` improved materially but is still not confirmed: it has `261` usable external public LDPC rows from `92` sources, but only `117/261` are positive versus baseline.
- `T53` has structure rows now, but zero usable assay join rows. Its blocker is no longer only structure data; it is invalid/stale ProteinGym assay rows plus missing raw DMS assay extraction.

Bottom line: v73 is a healthier full run than v72, but it did not add a new confirm. The highest-probability paths to additional confirms are `T46`, `T53`, `T31/T32`, and `T44`.

## Public Claim Rule

Only tests listed in `confirmed_public_now` may be described as current public confirms. Legacy `result_status=ok`, `support_like`, or compatibility fields are useful diagnostics, but they do not create a public confirm.

Current public confirms:

| Test | Claim status | Note |
|---|---|---|
| `T48` | `confirmed_public_now` | `compatible_positive_confirm_allowed`; strict public claim allowed |

## Batch And Wrapper Health

| Item | Value |
|---|---:|
| Result files present | 35 |
| Process OK | 35 |
| Process errors | 0 |
| Process timeouts | 0 |
| Full run return code | 0 |
| Confirm-only return code | 0 |
| Script timeout per test | 1800 s |
| Confirm tests executed | 35 |

The wrapper also wrote `v72_run_checkpoint.json`, and confirm-only wrote `confirm_only_run_summary_v72.json`. That means the run is reproducible and resumable enough for targeted retry work.

## Harvest And Validation

Network harvest summary:

| Pack | Candidate rows | Accepted rows | Rows written | Validator usable | Status |
|---|---:|---:|---:|---:|---|
| `fusion` | 8,998 | 0 | 0 | 0 | exact per-shot rows missing |
| `materials` | 3,061 | 0 | 0 | 0 | high candidates, zero accepted |
| `materials_family_packs` | 200,201 | 0 | 0 | 0 | high candidates, zero accepted |
| `nand` | 533 | 0 | 0 | 0 | no complete Tier-A rows |
| `thermoelectric` | 8,393 | 0 | 0 | 0 | orientation/ZT fields incomplete |
| `hepdata` | 0 | 0 | 0 | 0 | HEPData search found zero records |
| `optical_interconnect` | 172 | 0 | 0 | 0 | benchmark fields incomplete |
| `neuromorphic` | 179 | 0 | 0 | 0 | benchmark fields incomplete |
| `ldpc_external_benchmark` | 470 | 293 before dedup | 261 | 261 | validator-ready, but gate not positive enough |
| `protein_structures` | 1,067,614 | 152 | 152 | 152 | structure pack now usable |
| `proteingym` | 1,157,257 | 0 | 0 | 0 | raw assay parser did not extract valid DMS rows |

Validation result:

| Metric | Value |
|---|---:|
| `all_existing_rows_valid_v64` | false |
| `n_invalid_rows_v64` | 283 |
| `n_problem_files_v64` | 1 |
| Problem pack | `proteingym` |
| Problem file | `AUTO_PUBLIC_ROWS_V67.quarantine_v72.20260519T013633Z.csv` |

ProteinGym diagnostics:

- `283` rows are stale metadata/reference-manifest rows, not variant-score assay rows.
- `283` rows have bad variant identifiers.
- `258` rows have nonnumeric score values.
- The raw DMS parser progress says `n_cached_files_seen_v72=0`, `n_rows_accepted_v72=0`.

## Current Near-Confirm Gates

| Test | Gate result | Current numbers | Main blocker |
|---|---|---:|---|
| `T31` | not confirmed | `0` usable materials rows | Need measured kappa(T), grain size, source URL, microstructure method |
| `T32` | not confirmed | `0` usable materials rows | Same materials pack blocker as `T31` |
| `T34` | not confirmed | `0` usable thermoelectric rows | Need ZT plus orientation/grain-boundary angle rows |
| `T44` | not confirmed | `0` usable Tier-A NAND rows | Need company/year/layers/capacity/die area/bits per cell |
| `T45` | not confirmed | `0` usable optical rows | Need complete energy/bit, bandwidth, reach, platform, year |
| `T46` | not confirmed | `261` usable rows, `117` positive | External LDPC rows exist, but positive gate fails |
| `T47` | not confirmed | `0` usable neuromorphic rows | Need chip/task/energy/accuracy/topology/year |
| `T53` | not confirmed | `152` structure rows, `0` usable assay joins | Need valid ProteinGym variant-score assay rows joined to structure rows |
| `T57` | not confirmed | `0` HEPData rows | Need records/tables with observed/model/uncertainty columns |
| `T59` | not confirmed | `0` HEPData rows | Same HEPData exact-manifest blocker as `T57` |

## All Tests

| Test | Name | Process | Result | Claim bucket | Confirmation status | Rank | Next needed source |
|---|---|---|---|---|---|---:|---|
| `T26` | fusion ELM energy scaling | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | 1 | Certified fusion per-shot ELM energy rows with pedestal pressure/volume/drop columns |
| `T27` | ELM helicity proxy | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | 1 | Certified fusion RMP/helicity rows with ELM frequency and coil/phasing columns |
| `T28` | global H-mode confinement/KSS margin | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | 2 | Exact ITPA/H-mode rows with tau_E, density, and transport columns |
| `T29` | stellarator vs tokamak edge transport | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | 2 | Exact stellarator/tokamak edge transport rows with device and diffusivity/heat-flux columns |
| `T30` | fusion residual curvature coupling | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | 1 | Exact confinement residual rows with density plus shaping/curvature columns |
| `T31` | cryogenic kappa CCDR vs Casimir | ok | partial | near_confirm_requires_exact_rows | not_confirmed_exact_source_gates_pending_v64 | 7 | Filled exact materials packs with measured kappa(T), grain size, source URL, and microstructure method |
| `T32` | low-T kappa exponent | ok | partial | near_confirm_requires_exact_rows | not_confirmed_exact_source_gates_pending_v64 | 7 | Filled exact materials packs with measured low-temperature kappa(T), grain size, source URL, and microstructure method |
| `T33` | diamond/hBN thermal ceiling audit | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured materials rows |
| `T34` | Bi2Te3 ZT angle meta-analysis | ok | data_limited_no_cached_t34_orientation_rows | near_confirm_requires_exact_rows | not_confirmed_exact_te_rows_required_v64 | 3 | Exact Bi2Te3/Sb2Te3 ZT plus orientation/grain-boundary angle rows |
| `T35` | Kibble-Zurek grain-size exponent | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured materials rows |
| `T36` | density-stratified grain scattering | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured materials rows |
| `T37` | auxetic phonon transport | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured materials rows |
| `T38` | skyrmion lifetime literature audit | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured materials rows |
| `T39` | moire phononic bandgap twist scaling | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured materials_quantum rows |
| `T40` | transmon phononic-substrate T1 audit | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured quantum rows |
| `T41` | qubit T2 plateau meta-analysis | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured quantum rows |
| `T42` | spin-qubit T2 in isotopically pure Si | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured quantum rows |
| `T43` | DTC qubit error-per-cycle trend | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured quantum rows |
| `T44` | 3D NAND area vs volume scaling | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_true_tier_a_rows_required_v64 | 5 | True Tier-A NAND rows: company, year, layers, capacity, die area, bits/cell, source URL |
| `T45` | optical interconnect trend | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_benchmark_rows_required_v64 | 3 | Exact optical-interconnect benchmark rows |
| `T46` | LDPC burst-channel benchmark | ok | ok | synthetic_or_engineering | not_confirmed_external_public_benchmark_rows_required_v64 | 4 | External public LDPC/burst-channel benchmark rows with model and baseline scores |
| `T47` | neuromorphic graph energy audit | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_benchmark_rows_required_v64 | 3 | Exact neuromorphic benchmark rows |
| `T48` | photovoltaic acoustic-optical proxy | ok | ok | confirmed_public_now | compatible_positive_confirm_allowed | 10 | Already public-confirmed |
| `T49` | battery/thermoelectric materials symmetry audit | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured energy_materials rows |
| `T50` | Casimir residual public-table audit | ok | data_limited | bound_only | not_confirmable_by_design | 0 | Bound-table evidence only |
| `T51` | optical-clock drift literature bound | ok | data_limited | bound_only | not_confirmable_by_design | 0 | Bound-table evidence only |
| `T52` | atom-interferometer noise-floor audit | ok | data_limited | bound_only | not_confirmable_by_design | 0 | Bound-table evidence only |
| `T53` | biological symmetry/protein-folding proxy | ok | ok | near_confirm_requires_exact_rows | not_confirmed_structure_join_rows_required_v64 | 6 | ProteinGym assay rows joined to UniProt/PDB/AlphaFold structure-feature rows |
| `T54` | photosynthetic coherence meta-analysis | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured biotech rows |
| `T55` | radiation/spacecraft anomaly audit | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured aerospace rows |
| `T56` | solar-sail residual proxy | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured aerospace rows |
| `T57` | cosmic-ray cross-section enhancement | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_hepdata_manifest_rows_required_v64 | 3 | Frozen HEPData manifest plus observed/model/uncertainty residual tables |
| `T58` | exoplanet/stellar chronometry null | ok | data_limited | data_limited | not_confirmed_data_limited | 1 | Exact public structured aerospace rows |
| `T59` | public HEP anomaly ledger | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_hepdata_manifest_rows_required_v64 | 3 | Frozen HEPData manifest plus observed/model/uncertainty residual tables |
| `T60` | Koide sector-distance audit | ok | ok | anchor_only | anchor_only_not_full_confirm | 5 | Separate charged-lepton anchor from quark/lattice sector claim |

## Suggested Improvements To Push Toward Confirms

All improvements below are automated public-source parsing changes. No manual evidence rows should be entered by the user.

1. Ignore quarantined exact-source CSVs in validation and pack loading. Files matching `*.quarantine_v72.*.csv` should never count as active evidence; the current ProteinGym quarantine is still poisoning validation.
2. Add a pre-confirm validation stop: if `all_existing_rows_valid_v64=false`, write diagnostics and skip confirm gates for the invalid pack instead of letting stale rows appear as assay counts.
3. Fix ProteinGym raw DMS retrieval by resolving assay file names from the reference manifest/tree into actual raw CSV/blob URLs. The v73 raw parser saw `0` cached files, so it never reached real variant-score tables.
4. Add a ProteinGym assay-file adapter that requires mutation-like `variant` values and numeric DMS/fitness columns before candidate rows can enter the pack.
5. Join ProteinGym assay rows to the existing `152` AlphaFold/structure rows by normalized UniProt ID, then enforce the `50` usable join rows and `10` sequence-cluster gate.
6. For `T46`, add metric direction normalization. BER/FER/BLER/error-rate metrics are lower-is-better, while accuracy/throughput are higher-is-better; the current `117/261` positive count may be undercounting valid improvements if all metrics use one sign convention.
7. For `T46`, group LDPC comparisons by benchmark/channel/SNR/decoder/heldout split before counting positives, so only like-for-like public comparisons drive the gate.
8. For `T31/T32`, add source-specific materials adapters for the CMB-S4 cryogenic material tables and Zenodo supplemental files; the harvester saw `203,262` materials-family/materials candidates but accepted `0`.
9. For `T31/T32`, infer material family and microstructure only from public table/file metadata, captions, or source text, and write rows only when kappa, temperature, grain size, method, sample id, and URL are all present.
10. For `T44`, add a NAND product-table adapter that pairs company/year/product with layers, capacity, die area, and bits/cell across table rows and nearby text. Current NAND harvest found `533` candidates but `0` complete Tier-A rows.
11. For `T44`, normalize units and aliases aggressively: Gb vs Tb, Gbit vs GB, TLC/QLC/SLC to bits-per-cell, die size vs die area, and product aliases across source titles.
12. For `T34`, add a thermoelectric supplemental-table parser for Bi2Te3/Sb2Te3 papers that recognizes texture/orientation/grain-boundary synonyms and extracts ZT plus temperature.
13. For `T45`, add optical-interconnect benchmark adapters for silicon-photonics tables with energy/bit, bandwidth, reach, platform, and year. Current optical pack has `172` rejected candidates and no usable rows.
14. For `T47`, add neuromorphic benchmark adapters for Loihi/TrueNorth/SpiNNaker/Dynap-SE style tables with chip, task, energy, accuracy, topology, year, and source URL.
15. Fix HEPData search/API handling. The v73 HEPData progress file shows `0` record ids for all search URLs, so either the endpoint is wrong for JSON search or the parser is reading the wrong response shape.
16. Add HEPData table materialization: after record discovery, download YAML/CSV tables and map observed/model/uncertainty columns into local frozen manifest rows for `T57/T59`.
17. Add pack-level streaming and compressed candidate diagnostics. The v73 candidate CSV is over 1 GB; streaming top-missing summaries would reduce memory pressure and make `T44/T53` retries safer.
18. Promote high-candidate/zero-accepted diagnostics into per-pack action files. For each pack, save the top rejected row examples and missing fields so the next adapter patch targets the exact parser failure rather than re-running broad harvests blindly.

## Recommended Next Run

Fast confirm-focused retry after fixes:

```powershell
python .\run_all_and_confirm_v64.py --skip-full-run --cache tierb_cache_v74_confirm --outdir tierb_out_v74_confirm --harvest-public-sources --allow-network --confirm-only T31 T32 T34 T44 T45 T46 T47 T48 T53 T57 T59 --max-sources-per-pack 250 --max-rows-per-source 50000 --timeout 24000
```

For lower RAM pressure, split the retry:

```powershell
python .\run_all_and_confirm_v64.py --skip-full-run --cache tierb_cache_v74_fast --outdir tierb_out_v74_fast --harvest-public-sources --allow-network --confirm-only T31 T32 T34 T44 T45 T46 T47 T48 T57 T59 --max-sources-per-pack 250 --max-rows-per-source 50000 --timeout 24000
```

```powershell
python .\run_all_and_confirm_v64.py --skip-full-run --cache tierb_cache_v74_proteingym --outdir tierb_out_v74_proteingym --harvest-public-sources --allow-network --confirm-only T53 T48 --max-sources-per-pack 250 --max-rows-per-source 50000 --timeout 24000
```

Expected confirm order after improvements:

1. `T46`, if metric direction and like-for-like grouping turn the LDPC external rows positive enough.
2. `T53`, if ProteinGym raw DMS rows can be extracted and joined to the existing structure rows.
3. `T31/T32`, if materials adapters convert the large candidate pool into at least `50` usable exact rows.
4. `T44`, if source-specific NAND table parsing produces at least `8` complete true Tier-A rows across `3` companies.
5. `T57/T59`, if HEPData search and table materialization start producing exact residual rows.
