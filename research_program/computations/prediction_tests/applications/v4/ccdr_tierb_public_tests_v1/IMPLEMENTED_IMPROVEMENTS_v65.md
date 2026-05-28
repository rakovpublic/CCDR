# Implemented Improvements v65

This note records the implementation pass after `ALL_TESTS_CONFIRM_REPORT_v65.md`.

## Confirmed Public Claims

- Current public confirms remain exactly: `T48`.
- Validation output: `tierb_out_v65_improvements_check/claim_summary_v64.json`.
- `pass_v64` is `true`.
- Exact source packs still report zero filled rows; no evidence rows were fabricated.

## Implemented

- Added contributor-facing `CHECKLIST_v64.md` files for all exact source packs, including required columns, minimum gates, accepted evidence, and rejected rows.
- Added `claim_summary_v64.json` as a top-level artifact containing claim buckets, counts, blockers, source-pack status, process status, timeout attention, and the public-claim rule.
- Added checklist paths into `source_pack_status_v64.json` so near-confirm blockers point directly to source-pack preparation work.
- Fixed T48 provenance selection so confirm-only runs prefer a parent/top-level `t48_result.json` when present.
- Added `t48_provenance_appendix_v64.json` with status reconciliation between full-run status and public-claim gate status.
- Added an explicit T46 external public benchmark gate so synthetic/engineering evidence cannot be promoted to a public confirm.
- Updated the one-command wrapper to copy `claim_summary_v64.json` and T48 provenance into the top-level output, and to default full-run subprocess timeout to 1800 seconds.
- Added a v65 fast default T44 path so ordinary manifest-only runs do not execute the legacy NAND robustness stack that re-read historical generated CSVs.
- Bounded v57/v58/v59 CSV readers with file, row, and byte caps and `dtype=str` to avoid repeated mixed-type scans and high-memory pandas inference.

## Validation

- `python -m py_compile .\tierb\v64_exact_data_packs.py .\run_confirm_only_v64.py .\run_all_and_confirm_v64.py`
- `python .\run_all_and_confirm_v64.py --skip-full-run --cache .\tierb_cache --outdir .\tierb_out_v65_improvements_check`
- `python .\run_confirm_only_v64.py --outdir .\tierb_out_v65_all\confirm_only_v64_impl_check --cache .\tierb_cache --only T48`
- `python .\run_all_tier_b.py --cache .\tierb_cache --outdir .\tierb_out_t44_ram_check --only T44 --timeout 5 --max-tables 30 --script-timeout 120 --force --continue-on-error`

The focused T48 validation resolved provenance to `tierb_out_v65_all\t48_result.json` and preserved the reconciliation:

- batch process status: `ok`
- batch result status: `data_limited`
- confirm overlay status: `compatible_positive_confirm_allowed`
- public claim decision: T48 is claimable only because it appears in `confirmed_public_now`.

The focused T44 validation completed in about 3 seconds through `run_all_tier_b.py`, emitted `ccdr-tierb-result-v65-fast-t44`, and kept T44 as `data_limited` / `not_confirmed_audit_repair_required`.
