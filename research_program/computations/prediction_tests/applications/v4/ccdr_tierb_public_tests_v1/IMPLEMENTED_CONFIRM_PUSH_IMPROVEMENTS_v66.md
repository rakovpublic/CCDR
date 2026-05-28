# Implemented Confirm-Push Improvements v66

This implements the 15 confirmation-push improvements from `ALL_TESTS_CONFIRM_REPORT_v66.md` as source-pack infrastructure, validators, and gates. It does not fabricate evidence rows.

## Implemented Items

1. T31/T32 materials pack schemas, checklists, accepted/rejected disabled examples, and validation.
2. Per-family materials templates for silicon/semiconductor, oxide/ceramic, carbon, metal/alloy, and thermoelectric packs.
3. T44 true Tier-A NAND schema/checklist/examples with duplicate-key guidance and validation.
4. T44 dedup/source-domain support through pack schema `dedup_key_v64`.
5. T53 ProteinGym assay pack schema/checklist/examples and validation.
6. T53 protein-structure feature pack schema/checklist/examples and validation.
7. T34 thermoelectric angle pack schema/checklist/examples and validation.
8. T57/T59 HEPData manifest schema/checklist/examples and validation.
9. T57/T59 local residual-table disabled template for observed/model/uncertainty columns.
10. T45 optical interconnect benchmark schema/checklist/examples and validation.
11. T47 neuromorphic benchmark schema/checklist/examples and validation.
12. T26-T30 fusion exact-row schema/checklist/examples and validation.
13. T46 external public benchmark source pack and confirm gate.
14. `validate_v64_source_packs.py` validator command plus `v64_source_pack_validation.json` artifacts.
15. Per-test `*_next_rows_needed_v64.json` manifests plus `next_rows_needed_v64.json`.

## Validation

- `python -m py_compile .\tierb\v64_exact_data_packs.py .\run_confirm_only_v64.py .\run_all_and_confirm_v64.py .\validate_v64_source_packs.py`
- `python .\validate_v64_source_packs.py --outdir .\tierb_out_v66_improvements_check --cache .\tierb_cache`
- `python .\run_confirm_only_v64.py --outdir .\tierb_out_v66_improvements_check\confirm_only_v64 --cache .\tierb_cache`
- `python .\run_all_and_confirm_v64.py --skip-full-run --cache .\tierb_cache --outdir .\tierb_out_v66_improvements_wrapper_check`

## Current Confirm State

- Public confirms remain exactly `T48`.
- Validation packs: 11.
- Invalid existing rows: 0.
- Empty required packs: 11.
- T46 is now explicitly gateable through `data/exact_sources/ldpc_external_benchmark`, but remains synthetic/engineering-only until external public benchmark rows are added.
