# Round-10 v13: 10 positivity-focused improvements

Generated: 2026-05-05T08:44:30Z

Implemented:
1. Force Euclid P30 to use catalogue.mer_catalogue object/source coordinates only; no mer_cutouts corner coordinates for science statistic.
2. Euclid table-specific coordinate order: ALPHA_J2000/DELTA_J2000, RIGHT_ASCENSION/DECLINATION, RA/DEC.
3. ACT map sanity: total/finite pixels, finite fraction, min/median/max/mean/std.
4. ACT finite-footprint sampling: filter catalogue points to finite map pixels; random finite-pixel sampler validation if catalogue outside footprint.
5. SDSS DR18 SpecObj RA/DEC fallback for first real P30 density-kappa path if Euclid TAP is blocked.
6. CL2 full pulsar-list sampling: re-extract all NANOGrav .par coordinates and sample ACT kappa when possible.
7. P3 CDS ReadMe resolver: parse file list and accept endpoint evidence only with explicit endpoint/coordinate headers.
8. P41 arXiv-source sign/table extractor: evidence only if tabular observable values and sign/operator-basis terms are both present.
9. High-z a0 KROSS/KGES rotation probe: search/download rotation-like catalogue links; keep no-rise guard until Vrot/R is parsed.
10. Dashboard severity buckets fixed: guarded statuses now counted as sign_unfixed_ready or proxy_ready_no_evidence.
