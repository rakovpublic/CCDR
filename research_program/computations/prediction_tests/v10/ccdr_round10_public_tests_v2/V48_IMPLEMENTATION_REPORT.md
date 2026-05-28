# v48 Implementation Report

The requested 10 improvements have been implemented in `ccdr_r10_common.py` as a final v48/v48.1 override layer.

## Key behavioral change

The suite is now stricter: it prioritizes honest confirmation gates over positive-looking labels. Several routes intentionally downgrade to diagnostic/data-limited/required states until the exact statistic and control gates exist.

Most importantly, P36 high-z no longer promotes from stale local guard JSON or standalone summaries. In this patched bundle T13/T14 correctly return `highz_object_catalogue_data_limited_v48` because no trusted high-z object rows with source-file hashes are present.

## Files changed or added

- `ccdr_r10_common.py` — v48/v48.1 wrapper patch appended.
- `ROUND10_V48_PATCH_NOTES.md` — detailed patch notes.
- `V48_IMPLEMENTATION_REPORT.md` — this report.
- `reference_summaries/round10_summary_74_external_reference.json` — copied reference summary used by dashboard conflict detector.
- `outputs/p36_t13_executable_guard_v48.json` and `outputs/p36_t14_executable_guard_v48.json` — current v48 high-z guards.
- `outputs/smd_ccdr_predictions_template_v48.json` — derivation-gate template.
- selected `outputs/test*.json` refreshed in quick/offline mode.
- `outputs/round10_summary.json` regenerated from current patched output files.

## What this means for confirms

Current confirm counting should now use:

- Bucket A: non-SM rows ending in `*_confirm_like` and not source-conflicted.
- Bucket B: SM-D `smd_constant_consistency_confirm_like`, counted only as constant-level consistency.
- Bucket C: ready/compatible/bound statuses, not counted as confirms.
- Bucket D: data-limited/gate-failed/required statuses.

The dashboard reports source conflicts when a previous standalone summary disagrees with current test outputs. This directly catches the P36 high-z issue from the uploaded artifacts.
