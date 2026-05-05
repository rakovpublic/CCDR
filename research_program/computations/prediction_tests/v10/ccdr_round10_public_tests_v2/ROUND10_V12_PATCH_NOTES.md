# Round-10 v12: all 10 v11 report improvements

Generated: 2026-05-04T18:42:14Z

Implemented:
1. Euclid TAP sample query: prioritises catalogue.mer_catalogue and exact ALPHA_J2000/DELTA_J2000-style columns.
2. ACT ALM FITS inspection: astropy HDU/column summaries.
3. ACT ALM reader fallbacks: healpy.read_alm variants plus astropy REAL/IMAG fallback.
4. P30 ACT x Euclid statistic path: density bins, sky shuffle, density shuffle, RA-field jackknife.
5. CL2 strict status logic: positive-compatible only if n_pulsars_sampled > 20 and sky-shuffle p is finite.
6. CL2 reuses/extracts ACT ALM map when available.
7. P3 endpoint parser tightened: explicit endpoint columns required for evidence; otherwise proxy-ready/no-evidence.
8. P41 table/sign convention guard: numeric table rows + sign convention required before evidence.
9. High-z a0 rotation-source probes: KROSS/KMOS endpoints checked; SIG^2/R remains proxy-only unless Vrot/dynamical columns exist.
10. Dashboard severity split: hard_confirm, compatible_positive, positive_ready, proxy_ready_no_evidence, sign_unfixed_ready, tension/null, hard_blocker.
