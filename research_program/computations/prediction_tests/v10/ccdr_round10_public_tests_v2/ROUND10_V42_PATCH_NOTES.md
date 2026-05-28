# Round-10 v42 patch notes

v42 implements the ten improvement priorities from the v41 report.

1. P36 high-z audit package with accepted rows, source medians, units, and leave-one-source checks.
2. P36 high-z source robustness gate with source-bootstrap ratio-to-local-a0 CI.
3. P30 explicit patch-row repair; residual curl-null requires `n_patch_rows >= 6`.
4. P30 residual curl-null is patch-only; object-level p-values are diagnostic.
5. P41 numerical likelihood implementation with Wilson/SM delta-chi2 and fallback one-parameter Wilson-shift fit.
6. P41 CP split / CP-control null accounting.
7. P33 strengthened alpha_high/alpha_low/delta_alpha consumer with covariance/null gates.
8. P32 GW150914 one-event strain product hardening and reproducibility manifest.
9. P3 endpoint hard-skip is preserved.
10. P30 empirical ACT finite-support mask product is persisted and kept diagnostic-equivalent.

New outputs:

- `outputs/p36_highz_objectlevel_audit_v42.json`
- `outputs/p30_explicit_patch_rows_v42.json`
- `outputs/p30_empirical_act_mask_product_v42.json`
- `outputs/p41_wilson_sm_likelihood_v42.json`
- `outputs/p33_density_split_alpha_measurement_v42.json`
- `outputs/p32_gw150914_one_event_strain_result_v42.json`

