# Round-10 v22 confirm-oriented hardening patch

Implemented the 10 priority improvements requested after the round-10 report.  The patch is append-only: v21 behavior remains available, while selected tests now route through stricter v22 wrappers.

## Implemented improvements

1. **P36 high-z a0 unit/table/field gate**
   - Added `highz_unit_field_table_v22`.
   - Promotion to `highz_a0_vrot_confirm_like` now requires unit evidence, bootstrap fraction above SPARC >= 0.95, and independent table/field/leave-one-table split readiness.
   - Added None-safe hotfix for the v21 bootstrap gate.

2. **P30 ACT/Euclid density-kappa freeze policy**
   - Added `p30_maskrandom_freeze_v22`.
   - Keeps P30 frozen unless explicit ACT mask/weight or random-catalogue/mask-normalized density, HEALPix64/128/KNN agreement, >=2 science variants, and curl-null gates pass.

3. **P38 void morphology hardening**
   - Added `p38_catalogue_nulls_v22`.
   - Adds independent-family gate and publication hardening note for radius-preserving angular shuffles.

4. **P8/P8c NANOGrav cache-integrity fix**
   - Added `nanograv_retry_cache_v22`.
   - Corrupt/truncated gzip/tar cache no longer crashes the suite; it invalidates cache and returns JSON `data_limited` if retry fails.

5. **P33 density-BAO measured-confirm scaffold**
   - Added `p33_density_bao_measured_scaffold_v22`.
   - Separates BAO inventory readiness from actual density-split BAO confirmation.

6. **P39 likelihood-confirm gate**
   - Added `p39_likelihood_gate_v22`.
   - Blocks confirmation unless full covariance, systematics splits, model-penalty evidence, and adequate Δχ² are present.

7. **P32 ringdown strain-level plan**
   - Added `ringdown_strain_plan_v22`.
   - Adds gates for strain residual measurement, injection nulls, detector split, and leave-one-event-out stability.

8. **P41 structured q²/value/error + CP-null gate**
   - Added `p41_structured_cp_v22`.
   - Pattern hits no longer count as confirm-like; structured rows, sign basis, and CP asymmetry null are required.

9. **Direct-detection unit gate**
   - Added `direct_detection_units_v22`.
   - Allows only coverage confirmation when mass and limit units are explicit; detection claims remain disabled without event-level likelihoods.

10. **HEPData schema hardening**
    - Added `hepdata_schema_hardened_v22`.
    - Rejects HTML previews as physics evidence; requires non-HTML table endpoint, units/schema, and numeric rows.

## Operational fixes

- Added `--script-timeout` to `run_all.py` to keep a single slow endpoint from hanging the whole suite.
- Added quick-mode guards for P30 ACT, P30 Planck, P36 high-z, P39, P3, P41, and HEPData so `--quick` does not trigger heavy map/archive downloads.

## Updated test routing

Updated manifest/test `kind` fields for:

- R10-T01, R10-T02, R10-T03 -> `p39_likelihood_gate_v22`
- R10-T04 -> `p30_maskrandom_freeze_v22`
- R10-T07 -> `p33_density_bao_measured_scaffold_v22`
- R10-T10 -> `p38_catalogue_nulls_v22`
- R10-T13, R10-T14 -> `highz_unit_field_table_v22`
- R10-T16 -> `nanograv_retry_cache_v22`
- R10-T19 -> `ringdown_strain_plan_v22`
- R10-T25, R10-T26, R10-T27 -> `direct_detection_units_v22`
- R10-T30, R10-T33 -> `hepdata_schema_hardened_v22`
- R10-T31, R10-T32 -> `p41_structured_cp_v22`
- R10-DASH -> `dashboard_v22`

## Validation performed

- `python -m py_compile ccdr_r10_common.py run_all.py tests/*.py`
- Individual quick-mode JSON checks passed for representative v22 routes: P30 ACT, P30 Planck, P33, P36 high-z, P8 NANOGrav, P32 ringdown, direct detection, HEPData, and P41.

## Important claim policy

Only `*_confirm_like` statuses should be treated as claim-grade.  `*_positive_ready` means the route and parser are ready; `*_positive_compatible` means consistency/supporting evidence but not a full confirmation.
