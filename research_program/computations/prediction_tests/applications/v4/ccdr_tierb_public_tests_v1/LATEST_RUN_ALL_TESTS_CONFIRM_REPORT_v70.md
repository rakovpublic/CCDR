# Latest Run All-Tests Confirm Report v70

Run analyzed: `tierb_out_v70_all`

Generated: 2026-05-18

## Executive Summary

- Full run return code: `0`
- Confirm-only return code: `0`
- Process health: `35/35` tests finished with `process_status=ok`
- Process timeouts: none
- Current public confirms: `T48` only
- Public claim check: pass, because the only claimable current public confirm remains `T48`
- Network harvest: enabled
- Public-source rows written by harvester: `283`
- Structured sources parsed: `79`
- Downloaded or cached sources: `93`
- Strict source-pack validation: not clean yet; `proteingym` has one validator read error row

The run is operationally healthy. Scientifically, no new test is claimable yet. The most important near-term blocker is not lack of execution: it is source-pack acceptance. The harvester found and wrote ProteinGym rows, but the validator currently rejects the pack because the v63 CSV reader raises `ValueError: The 'low_memory' option is not supported with the 'python' engine`.

## Confirm Buckets

- Confirmed public now: `T48`
- Near confirm, exact rows required: `T31`, `T32`, `T34`, `T44`, `T45`, `T47`, `T53`, `T57`, `T59`
- Diagnostic only: `T26`, `T27`, `T28`, `T29`, `T30`
- Synthetic or engineering only: `T46`
- Bound only: `T50`, `T51`, `T52`
- Anchor only: `T60`
- Data limited: `T33`, `T35`, `T36`, `T37`, `T38`, `T39`, `T40`, `T41`, `T42`, `T43`, `T49`, `T54`, `T55`, `T56`, `T58`

## Harvest And Validation

| Pack | Tests | Harvest accepted rows | Rows written | Validator usable rows | Validator status |
|---|---|---:|---:|---:|---|
| fusion | T26-T30 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| materials | T31/T32 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| materials_family_packs | T31/T32 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| thermoelectric | T34 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| nand | T44 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| optical_interconnect | T45 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| ldpc_external_benchmark | T46 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| neuromorphic | T47 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| proteingym | T53 | 566 before dedup | 283 | 0 | validator_found_row_problems |
| protein_structures | T53 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |
| hepdata | T57/T59 | 0 | 0 | 0 | empty_pack_needs_exact_public_rows |

Important: `proteingym` is the only pack with actual harvested rows. They are present in `data/exact_sources/proteingym/AUTO_PUBLIC_ROWS_V67.csv`, but validation reads the file as one error row because of a CSV-reader option bug. Fixing that is the highest-value immediate repair.

## All Tests

| Test | Process | Result status | Bucket | Confirm status | Main blocker |
|---|---|---|---|---|---|
| T26 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion exact per-shot/per-timeslice rows required |
| T27 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Fusion exact RMP/helicity rows required |
| T28 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Exact ITPA/H-mode rows required |
| T29 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Exact stellarator/tokamak edge transport rows required |
| T30 | ok | data_limited | diagnostic_only | not_confirmed_diagnostic_only | Exact confinement residual rows required |
| T31 | ok | partial | near_confirm_requires_exact_rows | not_confirmed_exact_source_gates_pending_v64 | Exact measured kappa, grain size, source, and microstructure rows required |
| T32 | ok | partial | near_confirm_requires_exact_rows | not_confirmed_exact_source_gates_pending_v64 | Exact low-temperature kappa, grain size, source, and microstructure rows required |
| T33 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public materials rows insufficient |
| T34 | ok | data_limited_no_cached_t34_orientation_rows | near_confirm_requires_exact_rows | not_confirmed_exact_te_rows_required_v64 | Exact Bi2Te3/Sb2Te3 ZT plus angle rows required |
| T35 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public materials rows insufficient |
| T36 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public materials rows insufficient |
| T37 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public materials rows insufficient |
| T38 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public materials rows insufficient |
| T39 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public materials/quantum rows insufficient |
| T40 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public quantum rows insufficient |
| T41 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public quantum rows insufficient |
| T42 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public quantum rows insufficient |
| T43 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public quantum rows insufficient |
| T44 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_true_tier_a_rows_required_v64 | True Tier-A NAND rows with die area/capacity/layers/bits per cell required |
| T45 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_benchmark_rows_required_v64 | Exact optical benchmark rows required |
| T46 | ok | ok | synthetic_or_engineering | not_confirmed_external_public_benchmark_rows_required_v64 | External public benchmark rows required |
| T47 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_exact_benchmark_rows_required_v64 | Exact neuromorphic benchmark rows required |
| T48 | ok | ok | confirmed_public_now | compatible_positive_confirm_allowed | Confirmed public now |
| T49 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public energy/materials rows insufficient |
| T50 | ok | data_limited | bound_only | not_confirmable_by_design | Bound-only audit |
| T51 | ok | data_limited | bound_only | not_confirmable_by_design | Bound-only audit |
| T52 | ok | data_limited | bound_only | not_confirmable_by_design | Bound-only audit |
| T53 | ok | ok | near_confirm_requires_exact_rows | not_confirmed_structure_join_rows_required_v64 | ProteinGym rows must join to UniProt/PDB/AlphaFold structure features |
| T54 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public biotech rows insufficient |
| T55 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public aerospace rows insufficient |
| T56 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public aerospace rows insufficient |
| T57 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_hepdata_manifest_rows_required_v64 | HEPData observed/model/uncertainty residual rows required |
| T58 | ok | data_limited | data_limited | not_confirmed_data_limited | Structured public aerospace rows insufficient |
| T59 | ok | data_limited | near_confirm_requires_exact_rows | not_confirmed_hepdata_manifest_rows_required_v64 | HEPData observed/model/uncertainty residual rows required |
| T60 | ok | ok | anchor_only | anchor_only_not_full_confirm | Anchor only; not full confirm |

## Confirm-Focused Improvements

1. Fix the v63/v64 source-pack CSV reader by removing `low_memory=False` when `engine="python"` is used. This should let the validator read the 283 harvested ProteinGym rows instead of seeing one `_read_error_v63` row.
2. Re-run `validate_v64_source_packs.py` and `run_confirm_only_v64.py` immediately after the reader fix. This is the fastest way to see whether T53 progresses from zero assay rows to a real assay count.
3. Build the T53 structure join from harvested ProteinGym UniProt IDs. The assay rows exist, but `protein_structures` is still empty; fetch AlphaFold/PDB/UniProt features for the 59 harvested sequence clusters.
4. Add a T53 pack-specific validator that reports assay count, family count, sequence-cluster count, and joinable UniProt IDs before the model gate. Right now the generic validator hides useful near-confirm detail behind the CSV read failure.
5. Improve ProteinGym row writing so sequence-level metadata and variant-level DMS rows are separated. The current rows appear assay-level; the model may still need variant-level rows to pass the join gate robustly.
6. Fix HEPData discovery URLs and recursive record/table downloads. Current HEPData harvest has `0` downloaded sources, so T57/T59 cannot move.
7. Implement HEPData table-column inference for observed/model/uncertainty columns and write local CSV/YAML residual manifests automatically.
8. Replace generic NAND search harvesting with known public structured NAND sources and HTML table adapters. Current T44 harvest parses sources but accepts `0` rows because die area, capacity, layers, and bits/cell are not all found together.
9. Add source-specific materials adapters for public supplements instead of relying on generic Zenodo/OSF search result tables. T31/T32 currently parse structured metadata but not measured kappa/grain/microstructure rows.
10. Add materials partial-row join logic: source A may have kappa/temperature and source B may have grain/microstructure for the same sample. Without a controlled join, T31/T32 will keep rejecting near-useful rows.
11. Add thermoelectric-specific public table sources such as Starrydata/teMatDb-style exports or direct open supplement file links. Generic search produced no T34 rows.
12. Add optical-interconnect benchmark-specific source adapters for energy-per-bit, bandwidth, reach, platform, and year. Current generic harvest parsed tables but all lacked required fields.
13. Add neuromorphic benchmark-specific adapters for Loihi/SpiNNaker/TrueNorth/NeuroBench-style tables. Current parsed tables miss chip/energy/accuracy/topology/year completeness.
14. Add LDPC/burst-channel direct benchmark source seeds instead of generic repository search. T46 remains synthetic/engineering until external public model-vs-baseline benchmark rows are accepted.
15. Add a harvest quality gate that fails the run when rows are written but validation cannot read them. This would have caught the ProteinGym CSV-reader bug as a pipeline failure rather than leaving it as a confirm blocker.

## Highest-Probability Confirm Path

T53 is the closest path because the network harvester already wrote `283` ProteinGym rows. The next sequence is:

1. Fix the CSV reader bug.
2. Validate the harvested ProteinGym rows.
3. Fetch/join public structure features for the harvested UniProt IDs.
4. Re-run confirm-only.

T31/T32 and T44 remain important, but their latest harvest accepted no rows, so they need source-specific adapters before they can realistically confirm.

