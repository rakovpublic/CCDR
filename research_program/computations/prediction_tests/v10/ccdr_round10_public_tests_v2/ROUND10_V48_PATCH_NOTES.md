# Round-10 v48 Confirm-Hardening Patch

This patch implements the 10 improvement items requested after the Round-10 all-test report. It is intentionally a wrapper layer in `ccdr_r10_common.py`, so existing test files and manifest `kind` names continue to work.

## Implemented improvements

1. **P36 high-z provenance/stale-output hard gate**
   - Adds `run_highz_unit_field_table_v48` overriding `highz_unit_field_table_v22`.
   - Ignores older executable guard outputs, test JSONs, dashboards, and summaries as row sources.
   - Requires source-file SHA256 hashes, row hashes, current-run audit timestamp, no self-ingestion, >=30 rows, >=2 source groups, visible units, source leave-one stability, and source-bootstrap CI16 > local a0.
   - Writes `outputs/p36_t13_executable_guard_v48.json` and `outputs/p36_t14_executable_guard_v48.json`.

2. **P30 mask/curl/patch publication gate**
   - Adds `p30_publication_confirm_gate_v48`.
   - Blocks confirm-like unless official mask, same mask for science/curl/nulls, >=2 science variants, weak/null curl, clean patch protocol, and leave-one-patch-out stability are all present.
   - Quick mode now returns `density_kappa_diagnostic_only` instead of waiting on heavy ACT/Euclid products.

3. **P33 density-split BAO measured gate**
   - Adds `p33_density_bao_confirm_gate_v48`.
   - Requires measured high/low-density BAO alpha, DESI randoms, covariance-aware fit, sky shuffle, density-label shuffle, and redshift jackknife.
   - Default status is now `p33_density_bao_measurement_ready_not_confirmed` unless those gates exist.

4. **P8/P8c PTA kappa/residual statistic gate**
   - Adds `p8_pta_confirm_gate_v48` and `cl2_pta_density_confirm_gate_v48`.
   - Coordinates alone are not counted as evidence. Confirm-like requires residual/TOA weights, kappa samples at pulsars, weighted statistic, and sky-shuffle p <= 0.05.

5. **P32 ringdown strain-level gate**
   - Adds `p32_ringdown_confirm_gate_v48`.
   - Requires strain download, H1/L1 availability, PSD estimation, GR ringdown fit, CCDR residual-template fit, improvement over GR, injection nulls, detector split, and leave-one-event-out.

6. **P40 BB likelihood gate**
   - Adds `p40_bmode_confirm_gate_v48`.
   - File-role discovery and inventory are no longer promotable. Confirm-like requires BB bandpowers, covariance, foreground controls, low-ell template amplitude fit, and Planck/BK18 cross-check.
   - Quick mode returns `p40_bb_likelihood_required`.

7. **P41 q²/Wilson/CP-null likelihood gate**
   - Adds `p41_likelihood_confirm_gate_v48`.
   - Requires q²/value/error rows, sign basis, SM-vs-Wilson Δχ², CP-asymmetry null, and observable-bin jackknife.

8. **Direct-detection coverage-vs-detection separation**
   - Adds `direct_detection_claim_gate_v48`.
   - Public limit-curve overlap can become coverage-confirmed only after unit/window gates pass.
   - Detection claims remain disabled without event-level likelihoods; P37 drift additionally needs time-tagged event likelihoods.

9. **SMD derivation JSON gate**
   - Adds `smd_derivation_confirm_gate_v48` and writes `outputs/smd_ccdr_predictions_template_v48.json` when no preregistered prediction file exists.
   - Existing SM-D constant hits are now labelled `smd_constant_consistency_confirm_like` unless a preregistered independent derivation with uncertainty, residual sigma, no target fit, cross-prediction group, and timestamp is supplied.

10. **Dashboard source-conflict/stale-output detector**
    - Adds `dashboard_v48` and overrides `dashboard_v22`/`round10_dashboard`.
    - Compares current `outputs/test*.json`, `outputs/round10_summary.json`, and optional `reference_summaries/*.json`.
    - Any status disagreement for a test is reported under `source_conflicts` and changes dashboard status to `dashboard_positive_with_source_conflicts`.
    - The uploaded standalone `round10_summary(74).json` has been copied into `reference_summaries/round10_summary_74_external_reference.json` so the previous P36 high-z conflict is visible.

## Validation performed in this patched bundle

Targeted quick/offline validations were run for the changed routes. Important observed statuses:

- T04 P30: `density_kappa_diagnostic_only`
- T07 P33: `p33_density_bao_measurement_ready_not_confirmed`
- T13/T14 P36 high-z: `highz_object_catalogue_data_limited_v48`
- T16/T17 P8/P8c: statistic still required / data availability blocked
- T19 P32: `ringdown_ready_positive_bound`
- T21/T22 P40: `p40_bb_likelihood_required`
- T25/T28/T29 direct detection: `direct_detection_coverage_ready`
- T31 P41: `p41_q2_likelihood_gate_ready`
- SMD01/SMD05: `smd_constant_consistency_confirm_like`
- Dashboard: `dashboard_positive_with_source_conflicts`

The validation was quick/offline and not a full fresh online all-test rerun. `outputs/round10_summary.json` in this zip aggregates current patched outputs and is marked with an `aggregation_note`.
