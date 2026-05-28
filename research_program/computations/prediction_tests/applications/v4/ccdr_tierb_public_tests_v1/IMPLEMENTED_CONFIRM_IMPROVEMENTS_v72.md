# Implemented Confirm Improvements v72

Generated: 2026-05-19

## Scope

Implemented the 18 improvements from `LATEST_RUN_ALL_TESTS_CONFIRM_REPORT_v72.md`, preserving the no-manual-input rule: countable rows must be parsed from public downloaded/cached sources, not supplied by hand.

## Implemented Changes

1. Added wrapper checkpoints in `run_all_and_confirm_v64.py` via `v72_run_checkpoint.json`.
2. Added confirm-only checkpoints in `run_confirm_only_v64.py` via `confirm_only_run_summary_v72.json`.
3. Added post-harvest affected-test promotion: packs with written or validator-usable rows auto-add their affected tests to confirm-only.
4. Added stale generated-row quarantine/ignore handling for invalid `AUTO_PUBLIC_ROWS_V67.csv`.
5. Added valid-only prewrite filtering so new auto rows are written only after required-column and row-problem checks.
6. Added fallback stale-row ignore markers when Windows permissions prevent moving a bad generated file.
7. Added offline ProteinGym cached raw-file parsing and `t53_proteingym_raw_progress_v72.json`.
8. Added ProteinGym raw progress checkpoints every few files.
9. Added stronger ProteinGym validator guards for manifest metadata rows, non-mutation variants, and nonnumeric scores.
10. Added materials partial-row reload/join from existing `materials*_partial_rows_v71.csv` staging files.
11. Added materials text/supplement extraction for kappa, grain size, and microstructure methods.
12. Added NAND partial row staging from source-specific text/table rows.
13. Added NAND product/company/year/layer alias joining.
14. Tightened benchmark text adapters so optical, neuromorphic, and LDPC rows are emitted only when required numeric fields are present.
15. Added explicit HEPData API search seeds plus `hepdata_api_progress_v72.json` checkpoints.
16. Added targeted thermoelectric orientation/grain-boundary supplement seeds and text extraction.
17. Added pack priority ordering so fast/high-probability packs run before long ProteinGym jobs.
18. Added high-candidate/zero-accepted fail-fast warnings with first-adapter context.

## Files Changed

- `run_all_and_confirm_v64.py`
- `run_confirm_only_v64.py`
- `validate_v64_source_packs.py`
- `tierb/v67_public_source_harvesters.py`
- `tierb/v64_exact_data_packs.py`

## Verification

Passed:

```powershell
python -m py_compile .\tierb\v67_public_source_harvesters.py .\tierb\v64_exact_data_packs.py .\validate_v64_source_packs.py .\run_confirm_only_v64.py .\run_all_and_confirm_v64.py .\harvest_v67_public_sources.py
```

Passed dry-run harvest:

```powershell
python .\harvest_v67_public_sources.py --outdir .\tierb_out_v72_impl_dryrun --cache .\tierb_cache_v72_all --only T31 T32 T44 T45 T46 T47 T53 T57 T59 --dry-run --max-sources-per-pack 2 --max-rows-per-source 10 --no-validation
```

Passed cached smoke harvest:

```powershell
python .\harvest_v67_public_sources.py --outdir .\tierb_out_v72_impl_cache_smoke --cache .\tierb_cache_v72_all --only-pack proteingym nand materials thermoelectric ldpc_external_benchmark --max-sources-per-pack 1 --max-rows-per-source 5 --no-write-rows --no-validation
```

Passed stale-row ignore validation:

```powershell
python .\validate_v64_source_packs.py --outdir .\tierb_out_v72_impl_validate_after_ignore --cache .\tierb_cache_v72_all --only T46
```

Result: validation no longer has problem packs after stale ProteinGym generated rows are ignored; empty packs remain expected blockers.

Passed targeted confirm-only:

```powershell
$env:MKL_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_THREADING_LAYER='SEQUENTIAL'
python .\run_confirm_only_v64.py --outdir .\tierb_out_v72_impl_confirm_t46_after_ignore --cache .\tierb_cache_v72_all --only T46 T48
```

Result: still `T48` only. `T46` now has 250 usable public rows, but the gate reports `112/250` positive-vs-baseline comparisons, below the current confirmation threshold.

Passed wrapper check:

```powershell
python .\run_all_and_confirm_v64.py --skip-full-run --outdir .\tierb_out_v72_impl_wrapper_check --cache .\tierb_cache_v72_all --confirm-only T46 T48
```

Result: wrapper checkpoint and v72 summary fields were emitted; confirmed public remains `T48`.

## Current Confirm State

- Current confirmed public: `T48`
- `T46` is now validator-clean and runnable, but not confirmed because the public benchmark rows do not pass the positive-vs-baseline gate.
- `T53` stale manifest metadata rows are now detected and ignored/quarantined where filesystem permissions allow. Cached raw ProteinGym files are parsed offline and checkpointed, but the sampled cached rows did not yet produce accepted variant-level rows.
