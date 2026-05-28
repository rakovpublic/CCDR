# Latest Run All-Tests Confirm Report v71

Run analyzed: `tierb_out_v71_all`

Generated: 2026-05-18

## Executive Summary

- Full run return code: `0`
- Confirm-only return code: `0`
- Process health: `35/35` tests finished with `process_status=ok`
- Process timeouts: none
- Current public confirms: `T48` only
- Public claim check: pass, because the only claimable current public confirm remains `T48`
- Network harvest: enabled
- Public-source harvest: `11` packs attempted, `1,487` sources downloaded or cached, `1,021` structured sources parsed
- Candidate ledger: `2,443,110` candidate rows inspected
- Rows written by harvester: `283`
- Validator-usable harvested rows: `0`
- Strict source-pack validation: not clean, because the `proteingym` output contains `283` invalid metadata rows

The run is operationally healthy. It is not scientifically confirm-rich yet. The important change from the previous run is that the CSV reader problem is gone, so validation can now see the harvested rows. The stricter validator correctly rejects the `proteingym` rows because they are ProteinGym reference-manifest rows, not variant-level DMS assay rows. The new T31/T32 and T44 source-specific adapters ran, but they still produced `0` accepted rows for the confirm gates.

## Confirm Buckets

- Confirmed public now: `T48`
- Near confirm, exact rows required: `T31`, `T32`, `T34`, `T44`, `T45`, `T47`, `T53`, `T57`, `T59`
- Diagnostic only: `T26`, `T27`, `T28`, `T29`, `T30`
- Synthetic or engineering only: `T46`
- Bound only: `T50`, `T51`, `T52`
- Anchor only: `T60`
- Data limited: `T33`, `T35`, `T36`, `T37`, `T38`, `T39`, `T40`, `T41`, `T42`, `T43`, `T49`, `T54`, `T55`, `T56`, `T58`

Claim counts:

| Bucket | Count |
|---|---:|
| confirmed_public_now | 1 |
| near_confirm_requires_exact_rows | 9 |
| diagnostic_only | 5 |
| data_limited | 15 |
| bound_only | 3 |
| synthetic_or_engineering | 1 |
| anchor_only | 1 |
| no_confirm | 0 |

## Harvest And Validation

| Pack | Tests | Candidate rows | Accepted before dedup | Rows written | Validator usable rows | Validator status |
|---|---|---:|---:|---:|---:|---|
| fusion | T26-T30 | 8,997 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| materials | T31/T32 | 3,026 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| materials_family_packs | T31/T32 | 200,897 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| thermoelectric | T34 | 2,400 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| nand | T44 | 2,459 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| optical_interconnect | T45 | 197 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| ldpc_external_benchmark | T46 | 202 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| neuromorphic | T47 | 213 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| proteingym | T53 | 1,157,257 | 1,132 | 283 | 0 | validator_found_row_problems |
| protein_structures | T53 | 1,067,462 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| hepdata | T57/T59 | 0 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |

Validation details:

- `proteingym` has `1` output file and `283` rows, but all are rejected as `proteingym_reference_manifest_is_metadata_not_variant_scores|proteingym_variant_column_not_mutation_identifier`.
- `proteingym` diagnostics after validation: `n_assays=0`, `n_families=0`, `n_sequence_clusters=0`, `n_uniprots=0`, `n_schema_complete_rows=0`.
- `protein_structures` has `0` rows. AlphaFold fetch attempts used ProteinGym-style identifiers such as `A0A140D2T1_ZIKV`, which need UniProt accession resolution before AlphaFold can return usable models.
- `materials`, `materials_family_packs`, and `nand` remain empty even after source-specific adapter attempts.

## Candidate-Ledger Blockers

The candidate ledger is useful: it shows public sources were reached, but the column mapping is not yet specific enough to satisfy strict gates.

| Pack | Main missing pattern |
|---|---|
| fusion | `test_id|device|shot|time_or_slice|quantity|value|unit` |
| materials | `grain_size_nm|microstructure_method|boundary_density_proxy` |
| materials_family_packs | `family_name|sample_id|material|temperature_K|kappa_W_mK|grain_size_nm|microstructure_method` |
| thermoelectric | `material|composition|ZT|temperature_K|orientation_angle_deg|grain_boundary_angle_deg` |
| nand | `company|year|layers|capacity_Gb|die_area_mm2|bits_per_cell|notes` |
| optical_interconnect | `platform|year|energy_per_bit_pJ|bandwidth_Gbps|reach_m` |
| ldpc_external_benchmark | `benchmark|metric_name|model_score|baseline_score|uncertainty|notes` |
| neuromorphic | `chip|energy_per_inference_or_spike_pJ|accuracy|topology|year` |
| proteingym | `assay_id|uniprot|protein_name|family|sequence_cluster|variant|dms_score|fitness_residual` |
| protein_structures | `uniprot|oligomeric_state|symmetry_proxy|contact_network_proxy|fold_class|pdb_id_or_alphafold_id` |

## All Tests

| Test | Process | Result status | Public-confirm bucket | Main blocker |
|---|---|---|---|---|
| T26 | ok | data_limited | diagnostic_only | Fusion exact per-shot/per-timeslice rows required |
| T27 | ok | data_limited | diagnostic_only | Fusion exact RMP/helicity rows required |
| T28 | ok | data_limited | diagnostic_only | Exact ITPA/H-mode rows required |
| T29 | ok | data_limited | diagnostic_only | Exact stellarator/tokamak edge transport rows required |
| T30 | ok | data_limited | diagnostic_only | Exact confinement residual rows required |
| T31 | ok | partial | near_confirm_requires_exact_rows | Exact measured kappa(T), grain size, source URL, and microstructure method rows required |
| T32 | ok | partial | near_confirm_requires_exact_rows | Exact low-temperature kappa(T), grain size, source URL, and microstructure method rows required |
| T33 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T34 | ok | data_limited_no_cached_t34_orientation_rows | near_confirm_requires_exact_rows | Exact Bi2Te3/Sb2Te3 ZT plus orientation/grain-boundary angle rows required |
| T35 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T36 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T37 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T38 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T39 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T40 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T41 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T42 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T43 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T44 | ok | data_limited | near_confirm_requires_exact_rows | True Tier-A NAND rows with company, year, layers, capacity, die area, bits/cell, and source URL required |
| T45 | ok | data_limited | near_confirm_requires_exact_rows | Exact optical benchmark rows required |
| T46 | ok | ok | synthetic_or_engineering | External public benchmark rows required before public confirm |
| T47 | ok | data_limited | near_confirm_requires_exact_rows | Exact neuromorphic benchmark rows required |
| T48 | ok | ok | confirmed_public_now | Confirmed public now |
| T49 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T50 | ok | data_limited | bound_only | Bound/constraint audit; useful for limits, not confirmable by design |
| T51 | ok | data_limited | bound_only | Bound/constraint audit; useful for limits, not confirmable by design |
| T52 | ok | data_limited | bound_only | Bound/constraint audit; useful for limits, not confirmable by design |
| T53 | ok | ok | near_confirm_requires_exact_rows | ProteinGym assay rows must be variant-level and joined to UniProt/PDB/AlphaFold structure features |
| T54 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T55 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T56 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T57 | ok | data_limited | near_confirm_requires_exact_rows | Frozen HEPData manifests and residual rows with observed/model/uncertainty columns required |
| T58 | ok | data_limited | data_limited | Required public structured physical rows are missing or insufficient |
| T59 | ok | data_limited | near_confirm_requires_exact_rows | Frozen HEPData manifests and residual rows with observed/model/uncertainty columns required |
| T60 | ok | ok | anchor_only | Anchor-only consistency result; full confirmation requires separate quark/lattice and null gates |

## Confirm-Focused Improvements

1. Fix ProteinGym raw DMS path resolution. The adapter tried `ProteinGym_substitutions/...`, `data/DMS_ProteinGym_substitutions/...`, and `DMS_ProteinGym_substitutions/...`, but every raw attempt reported `raw_dms_unavailable`. Add GitHub-tree lookup for `raw_DMS_filename` and fetch the exact blob URL rather than guessing paths.
2. Stop writing ProteinGym reference-manifest rows to `data/exact_sources/proteingym/AUTO_PUBLIC_ROWS_V67.csv`. Keep the manifest as an index only; write rows only when `variant` is a mutation identifier and `dms_score` is a real numeric phenotype/fitness value.
3. Add a ProteinGym metadata-to-raw join using `raw_DMS_filename`, `raw_DMS_mutant_column`, `raw_DMS_phenotype_name`, and `raw_DMS_directionality`. This should produce variant-level rows and is the highest-probability path for T53.
4. Resolve ProteinGym UniProt mnemonic IDs before AlphaFold fetch. Values like `A4_HUMAN` and `ACE2_HUMAN` need mapping to UniProt accessions such as `P05067`-style IDs before `alphafold.ebi.ac.uk/api/prediction/...` can work.
5. Add a local UniProt mapping cache for T53. Query UniProt by mnemonic/accession once, cache accession, organism, reviewed status, and canonical sequence, then feed the resolved accession into AlphaFold/PDB fetchers.
6. Add an AlphaFold fallback that reads model metadata directly from the AlphaFold API and computes structure features from CIF/PDB only after a valid model URL is returned. Current `alphafold_api_unavailable` attempts indicate the identifier layer is failing before structure parsing begins.
7. Add a T53 join preflight report: count raw DMS rows, unique assays, unique resolved UniProt accessions, AlphaFold/PDB hits, and final joined rows before running the model gate. This makes T53 blockers visible without reading the full candidate ledger.
8. Improve the T31/T32 CMB-S4 adapter to write partial public rows into a non-confirming staging file with clear missing fields. The run did load CMB-S4 tables, but accepted `0` rows because grain size, microstructure method, and boundary proxy are absent.
9. Add materials supplement adapters for nanocrystalline/grain-size papers that already contain grain size and microstructure. The candidate ledger shows many rows with kappa-like data but missing `grain_size_nm` and `microstructure_method`; T31/T32 need a controlled join between kappa tables and microstructure tables.
10. Implement materials partial-row joining by DOI/sample/material/temperature. Allow kappa(T) rows from one public supplement to join grain-size/microstructure rows from the same sample in another public supplement, while keeping provenance fields for both URLs.
11. Add source-specific NAND adapters for public vendor/ISSCC/VLSI spec tables, not only WikiChip and generic Zenodo. The WikiChip URLs were unavailable in this run, and generic Zenodo rows missed the complete `company/year/layers/capacity/die_area/bits_per_cell` tuple required by T44.
12. Add NAND text/PDF table extraction for die photos and technology tables. T44 likely needs rows from vendor presentation PDFs where die area and layer count are in text or tabular figure captions rather than CSV-like web tables.
13. Add T44 unit normalization and product alias joining. Public sources often split `capacity_Gb`, `bits_per_cell`, and `die_area_mm2` across product pages, papers, and tables; joining by company, year, generation, and product family should produce validator-ready rows.
14. Fix HEPData access by using the HEPData API record endpoint and table download endpoints directly. This run downloaded `0` HEPData sources, so T57/T59 cannot move until search URLs become API/record/table URLs.
15. Add an adapter-quality gate: if a source-specific adapter runs but writes `0` rows, include per-source status and top missing columns in `public_source_harvest_v67.json`. The candidate ledger already has this data, but putting it in the small dashboard will make confirm blockers obvious.
16. Add direct benchmark adapters for T45 and T47. Generic discovery found rows, but accepted `0`; optical and neuromorphic benchmarks need dedicated parsers for energy, bandwidth/reach, topology, year, accuracy, and platform/chip fields.
17. Add external public benchmark rows for T46. The full test result is `ok`, but the confirm bucket remains synthetic/engineering until public model-vs-baseline benchmark rows satisfy the LDPC gate.
18. Add a run-level warning when `n_rows_written_v67 > 0` but `validator_usable_rows_v64 == 0`. This is the exact failure mode in v71 and should be flagged as a harvest-regression condition.

## Highest-Probability Confirm Path

T53 remains the best confirm candidate, but only if the pipeline switches from ProteinGym manifest rows to raw variant-level DMS rows and then resolves UniProt IDs for the structure join.

The fastest useful sequence is:

1. GitHub-tree lookup for exact ProteinGym raw DMS files.
2. Write only variant-level ProteinGym assay rows.
3. Resolve UniProt mnemonic/accession IDs.
4. Fetch AlphaFold/PDB structure rows.
5. Re-run `validate_v64_source_packs.py --only T53` and then `run_confirm_only_v64.py --only T53 T48`.

T31/T32 and T44 are still worthwhile, but their adapters currently produce zero accepted rows. They need source-specific row construction and public-source joins before they can realistically add confirms.
