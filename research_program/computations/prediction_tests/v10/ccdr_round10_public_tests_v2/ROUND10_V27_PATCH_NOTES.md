# Round-10 v27 confirm-execution patch

This patch implements the 10 requested improvements from the v26 report while keeping claim promotion strict.

## Implemented changes

1. **P30 empirical ACT mask product**: persists `act_dr6_empirical_finite_support_mask_v27.json` in the cache. It is explicitly labelled diagnostic unless an official ACT mask is found.
2. **P30 random catalogues**: creates deterministic route random catalogues where cached RA/Dec coordinates exist and records the required random-normalized density definition.
3. **P30 same-split science-variant matrix**: records whether baseline/f090/f150/tonly/cibdeproj/curl use compatible high/low counts and whether >=2 science variants and curl gates pass.
4. **P30 route-specific claims**: reports SDSS and Euclid separately and prevents a mixed Euclid/SDSS global claim.
5. **P30 v27 publication gate**: only promotes after official/equivalent mask, random-density rerun, science variants, curl, and separated route gates pass.
6. **P36 high-z object acceleration table**: writes `outputs/p36_highz_object_acceleration_rows_v27.json` with strict object-level Vrot/R/z/a schema when rows are available.
7. **P36 source-specific parser contract**: defines accepted aliases and rejects SIGMA, VEL_PA, inclination-only, and image-HDU products as object-rotation evidence.
8. **P33 measured-alpha schema**: writes `outputs/p33_density_split_alpha_result_v27.json` and adds a publication gate for actual alpha_high/alpha_low measurement with nulls.
9. **P41 q²/CP rows and Wilson/SM gate**: writes `outputs/p41_q2_cp_rows_v27.json` and blocks major claim until CP controls and numerical Wilson/SM likelihood exist.
10. **P32 strain manifest + SM-D derivation schema**: writes a strain execution manifest and adds derivation-output schemas for SM-D constants/Koide.

## Claim policy

v27 creates operational artifacts and rerun contracts. It does **not** promote P30/P36-high-z/P41/P33/P32 unless the strict gates pass on actual measured data.
