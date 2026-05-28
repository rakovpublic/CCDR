# Round-10 v55 confirm-recovery patch

Built on v54.

## Purpose

v54 successfully made P38 artifact-backed, but the remaining confirm opportunities still needed filled, non-template measurement artifacts. v55 adds strict builders/gates for those paths while preventing post-hoc promotion.

## Implemented improvements

1. P36 high-z raw-row normalizer scans `inputs/`, `measurements/`, and cache for KGES/KROSS/KMOS3D/SAMI-like object rows.
2. P36 high-z large-radius gate requires raw/non-output rows, `R_kpc >= 0.5`, >=30 rows, >=2 sources, >=20 rows per two sources, tiny-radius fraction <=20%, and median acceleration above local a0.
3. P30 requires an active predeclared patch protocol file filled before the run; auto/next-run files do not count.
4. P30 same-mask route gate requires same science/curl/null mask and same density labels for variants.
5. P30 route proof can use an explicit same-mask or independent Planck/Euclid artifact, but it must be non-template and positive.
6. P33 alpha-measurement gate now accepts legacy measured alpha artifacts only if `alpha_high`, `alpha_low`, and `delta_alpha` are numeric and nulls/covariance are present.
7. PTA/CL2 gate requires a real weighted kappa-residual artifact with coordinate hashes, residual/TOA weights, kappa samples, sky-shuffle p <= 0.05, and top-weight stability.
8. P32, P40, and P41 likelihood gates require non-template likelihood artifacts with required boolean/numeric fields.
9. SMD derivation gate requires non-template preregistered prediction metadata with `no_target_fit` and independent derivation.
10. Dashboard v55 adds strict buckets, v55 artifact indexing, failed-gate reporting, and confirm-recovery priorities.

## Expected behavior

v55 should not inflate confirmations. It promotes only when strict non-template artifacts exist and pass required fields. Otherwise it produces clearer missing-field reports and fillable v55 contracts.
