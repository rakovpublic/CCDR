# Round-10 v44 confirm-target hardening

v44 implements the next ten improvements after v43:

1. P36 high-z second-source robustness: independent source-group audit, source-level bootstrap, leave-one-source stability, and compact CSV audit output.
2. P36 audit table: writes `outputs/p36_highz_source_audit_v44.csv` and `outputs/p36_highz_second_source_robustness_v44.json`.
3. P30 predeclared patch protocol: fixed quality thresholds and patch-stability rule before any claim-level interpretation.
4. P30 bad-patch investigation: diagnostic classification for bad patches, especially low-count, curl-dominated, negative-science, or large-curl patches.
5. P30 mask-equivalence appendix: publication-level promotion requires official ACT mask or accepted `p30_mask_equivalence_appendix_accepted.json`.
6. P41 Wilson coefficient fit v44: stronger row discovery from local/downloaded q2/value/error/SM/Wilson tables.
7. P33 alpha measurement v44: consumes alpha rows and can produce a diagnostic xi-peak alpha estimate from supplied correlation rows.
8. P32 GWOSC endpoint resolver: queries GWOSC event API candidates and resolves H1/L1 strain URLs before product gating.
9. P3 endpoint hard skip preserved.
10. Dashboard v44 strict separation: no promotion from candidate, readiness, or diagnostic-only gates.

Promotion remains strict:

- P36 high-z requires >=30 object rows from >=2 independent source groups, source-bootstrap CI16 > 1, and leave-one-source stability.
- P30 route-specific confirmation requires predeclared patch stability, residual curl patch null, and official/equivalent mask.
- P41 requires q2/value/error/SM/Wilson likelihood rows, delta chi2 >= 9, and CP controls null/weak.
- P33 requires measured alpha_high/alpha_low with covariance and nulls.
- P32 requires measured strain, PSD, GR ringdown fit, CCDR residual fit, residual improvement, injection nulls, detector split, and leave-one-event/single-event robustness.
