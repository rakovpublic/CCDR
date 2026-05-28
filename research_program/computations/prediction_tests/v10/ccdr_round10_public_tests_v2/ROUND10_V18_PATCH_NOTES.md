# Round-10 v18: 10 near-confirm improvements

Generated: 2026-05-06T20:49:11Z

Implemented:
1. P30 primary 20k statistic path fixed: mer['sample'] is directly passed into the density-kappa statistic.
2. P30 spatial jackknife added using HEALPix/sky regions instead of gaia_id as field.
3. P30 variant/curl formal summary added: science variant pass count and science/curl delta ratio.
4. P30 near-confirm status added when primary nulls plus jackknife or replication pass.
5. CL2 residual-weighted scaffold: unweighted result is retained; residual/red-noise weighting path is explicit.
6. P3 endpoint null plan: endpoint shuffle status plus redshift/density shuffle readiness.
7. P41 structured-row guard: structured rows + sign basis required; CP-asymmetry null requirement explicit.
8. High-z a0 FITS TUNIT/unit metadata probe for Vrot/R columns.
9. Direct-detection units-guarded coverage confirmation; detection claim remains forbidden.
10. Dashboard near-confirm bucket: confirm_like, near_confirm_positive_compatible, coverage_confirmed, guarded.
