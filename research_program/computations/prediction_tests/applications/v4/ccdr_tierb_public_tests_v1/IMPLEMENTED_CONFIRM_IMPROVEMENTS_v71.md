# Implemented Confirm Improvements v71

Generated: 2026-05-18

## Scope

Implemented the 18 confirm-focused improvements from `LATEST_RUN_ALL_TESTS_CONFIRM_REPORT_v71.md`. The changes keep the no-manual-input rule: countable rows must come from public downloaded/cached sources and pass the v64 source-pack validator.

## Implemented Changes

1. Added ProteinGym GitHub-tree lookup for exact raw DMS file paths instead of relying only on guessed paths.
2. Stopped ProteinGym reference-manifest rows from being accepted as source-pack evidence.
3. Added ProteinGym raw metadata joins using `raw_DMS_filename`, `raw_DMS_mutant_column`, `raw_DMS_phenotype_name`, and directionality.
4. Added UniProt accession extraction for accession-like IDs embedded in ProteinGym mnemonic fields.
5. Added a cached UniProt REST resolver for mnemonic IDs before AlphaFold lookup.
6. Reworked AlphaFold structure fetching to use resolved UniProt accessions and write a T53 preflight report.
7. Added `t53_proteingym_structure_preflight_v71.json` with raw rows, unique IDs, resolution status, and structure-row counts.
8. Added non-confirming partial-row staging for T31/T32 materials rows.
9. Added generic materials partial-row capture for kappa/grain/microstructure fragments.
10. Added materials partial-row join logic by source/sample/material join key, with rows counted only if the strict schema is complete after joining.
11. Added more direct T44 NAND public-source seeds for vendor/technology public summaries.
12. Added NAND text-line extraction for public HTML/text/PDF-like payloads.
13. Added NAND context carry-forward and company/product alias extraction to join split table fields.
14. Added a HEPData API adapter that collects record IDs, follows record/table URLs, writes local table CSVs, and maps observed/model/uncertainty columns.
15. Added adapter-quality warnings when a pack writes zero rows or writes rows that validation cannot use.
16. Added direct text adapters for T45 optical interconnect benchmark rows.
17. Added direct text adapters for T47 neuromorphic and T46 LDPC/public benchmark rows.
18. Added run/dashboard summary fields for candidate quality, pack quality, and adapter warnings.

## Files Changed

- `tierb/v67_public_source_harvesters.py`
- `tierb/v64_exact_data_packs.py`
- `run_all_and_confirm_v64.py`
- `validate_v64_source_packs.py`

## Verification

Commands run:

```powershell
python -m py_compile .\tierb\v67_public_source_harvesters.py .\tierb\v64_exact_data_packs.py .\validate_v64_source_packs.py .\run_all_and_confirm_v64.py .\run_confirm_only_v64.py
python .\harvest_v67_public_sources.py --outdir .\tierb_out_v71_improvements_dryrun --cache .\tierb_cache_v71_all --only T31 T32 T44 T45 T46 T47 T53 T57 T59 --dry-run --max-sources-per-pack 2 --max-rows-per-source 10 --no-validation
python .\harvest_v67_public_sources.py --outdir .\tierb_out_v71_improvements_cachecheck --cache .\tierb_cache_v71_all --only-pack proteingym protein_structures materials nand hepdata optical_interconnect neuromorphic ldpc_external_benchmark --max-sources-per-pack 1 --max-rows-per-source 5 --no-write-rows --no-validation
```

Results:

- Compile passed.
- Dry-run manifest generation passed.
- Cached no-write adapter check passed.
- ProteinGym reference-manifest rows are now rejected in the candidate ledger with `proteingym_reference_manifest_is_index_not_variant_scores` and are not accepted as evidence.
- New summary fields are present in `public_source_harvest_v67.json`: `candidate_quality_v71`, `pack_quality_v71`, and `adapter_quality_warnings_v71`.
