# Round-10 v11: real-statistic and guardrail patch

Generated: 2026-05-04T15:09:53Z

Implemented:
1. Fixed read_text_any() MemoryError by streaming only max_bytes.
2. P30 ACT extracts baseline kappa ALM FITS from ACT DR6 tarball.
3. P30 ACT adds healpy read_alm + alm2map path.
4. P30 Euclid tries real TAP table/column discovery and ADQL RA/DEC/photo-z query.
5. P30 statistic implements high-density vs low-density kappa, sky shuffle, density-bin shuffle, field-split jackknife, and mask-aware placeholder.
6. CL2 samples ACT kappa at NANOGrav pulsar coordinates when ACT map exists, with sky-shuffled pulsar positions.
7. P3 streaming CDS/VizieR endpoint parser avoids huge in-memory reads and computes endpoint-orientation proxy when possible.
8. High-z a0 guard prevents high-z-rise claims from SIG^2/R without rotation/dynamical-mass columns.
9. P41 sign-convention guard prevents counting pattern hooks as evidence until sign convention is fixed.
10. Dashboard reports hard_blockers explicitly.
