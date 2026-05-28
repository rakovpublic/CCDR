# Round-10 v26 confirm-execution hardening

v26 implements the ten latest confirmation upgrades without loosening the claim policy.

## Implemented

1. **P30 official/equivalent ACT mask gate**
   - Adds `p30_act_mask_or_equivalent_v26`.
   - Separates official masks from empirical finite-support masks.
   - Empirical masks remain diagnostic-only unless promoted by documented route + random-density normalization.

2. **P30 random-density normalization contract**
   - Adds `p30_random_density_normalization_v26`.
   - Searches cache for random/selection/mask-normalization files.
   - Requires `delta = N_data/N_random - 1` for confirmation.

3. **P30 science-variant matrix**
   - Adds `p30_variant_route_matrix_v26`.
   - Requires same density bins, same mask, same random-normalized density, >=2 science variants, and curl weaker/null.

4. **P30 route separation**
   - Adds `p30_independent_route_claims_v26`.
   - SDSS and Euclid are treated as separate claims; merging is disabled until both pass the same gates.

5. **P36 high-z source-specific object scan**
   - Adds `highz_source_specific_object_scan_v26`.
   - Scans FITS/CSV/TSV/TXT cache entries for strict Vrot/R/z object rows.
   - Groups KMOS3D/KGES, KROSS, MOSDEF, SAMI, SINFONI.

6. **P36 high-z image-FITS rejection**
   - Adds `highz_image_fits_rejection_v26`.
   - Image HDUs / velocity maps do not count as object-level Vrot/R/z rows.

7. **P33 measured density-split BAO alpha contract**
   - Adds `p33_density_split_bao_alpha_measurement_v26` and `p33_publication_confirm_gate_v26`.
   - Requires LSS catalogues, randoms, covariance, measured alpha high/low, nulls, and redshift jackknife.

8. **P41 CP-control extraction**
   - Adds `p41_cp_control_extraction_v26`.
   - Extracts CP-like, non-CP, and uncertainty-like q² rows from downloaded text/table sources.

9. **P41 Wilson/SM major-claim gate**
   - Adds `p41_wilson_sm_fit_gate_v26`.
   - Major P41 claims require q² numeric rows, CP controls, uncertainty rows, and Wilson/SM comparison fit.

10. **P32 strain + SM-D derivation gates**
   - Adds `ringdown_executable_plan_v26` and `ringdown_strain_confirm_gate_v26`.
   - Adds `smd_derivation_gate_v26` for SM-D constants and Koide.

## Claim policy

v26 still does not promote readiness/proxy/coverage results into publication-grade confirmations.
The current hard-publication route remains:

- P38 void morphology
- P36 local a0

P30, P36 high-z, P33, P41, P32, and SM-D derivation branches now expose stricter operational gates for the next run.
