# Round-10 v15: confirmation squeeze / process-more-data patch

Generated: 2026-05-05T20:11:59Z

Implemented:
1. Fixed CL2 ACT-helper blocker with local ACT baseline extraction and full NANOGrav coordinate parsing.
2. Added robust ACT ALM reader: row-order full ALM, zero/one-based sparse index modes, then healpy fallback.
3. Added ACT finite-map acceptance guard; all-nonfinite maps are rejected before science statistics.
4. P30 now processes Euclid mer_catalogue and SDSS fallback, then picks only real catalogue-footprint statistics for confirmation.
5. P30 confirm_like requires delta>0, sky p<=0.05, density-shuffle p<=0.05, and jackknife same sign.
6. P38 confirm-like squeeze: jackknife all-positive + lognormal-null p<=0.01.
7. High-z a0 Vrot bootstrap: robust-suggestive only if >=95% bootstrap medians exceed SPARC local a0.
8. P41 supplementary ZIP/table parser for CDS attachments and LHCb supplementary archive candidates.
9. Direct-detection schema parser: detects mass/limit/cross-section headers and separates coverage readiness from detection.
10. Dashboard v15 counts confirm-like, guarded-ready, and blockers explicitly.
