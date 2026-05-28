# Round-10 v29 confirm-execution patch

v29 implements the next 10 confirm-focused improvements:

1. P30-SDSS same-split ACT variant rerun machinery: baseline/f090/f150/tonly/cibdeproj/curl are evaluated against the same SDSS random-normalized high/low-density split when `--allow-large` is enabled.
2. P30 curl-control gate: curl must be null/weaker under the same split before P30-SDSS can promote.
3. P30 route-specific confirm gate: P30-SDSS can promote separately; global P30 still needs a second independent repaired route.
4. P30-Euclid repair policy remains separate and explicit: photo-z/depth/quality cuts + field randoms + same mask/random definition.
5. P36 high-z source-specific object parsers: adds wider, source-specific Vrot/R/z aliases and FITS binary-table scanning.
6. P36 object acceleration table v29: writes strict object rows to `outputs/p36_highz_object_acceleration_rows_v29.json`.
7. P41 Wilson/SM likelihood consumer v29: consumes q2 likelihood JSONs and blocks major claims until numerical SM/Wilson and CP-control gates pass.
8. P41 CP-control/null gate remains required; pattern counts alone are not enough.
9. P33 measured alpha consumer v29: consumes `alpha_high_density`/`alpha_low_density` result files and enforces covariance/null/jackknife gates.
10. P32 strain execution manifest v29 and SM-D derivation gates remain strict: no strain or constant-derivation claim without product JSONs.

Claim policy: v29 improves measured execution paths, but it intentionally does not promote proxy/readiness outputs unless gates pass.
