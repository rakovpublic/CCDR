# Round-10 v59 patch notes

Goal: no manual filling anywhere in active tests.

Implemented:

1. Quarantine active TEMPLATE/FILL/manual-fill artifacts from `inputs/` and `measurements/` into `docs/examples/legacy_manual_fill_artifacts_quarantined_v59/`.
2. P36 high-z public/cache parser expansion for CSV/TSV/TXT/DAT/JSON/HTML/FITS tables with KROSS/KGES/KMOS3D/SAMI/MOSDEF/PHIBSS-like source hints.
3. P36 source-specific alias handling for redshift, velocity, radius, IDs and unit heuristics.
4. P36 strict claim gate remains unchanged: large-radius, multi-source, row-count and acceleration conditions required.
5. P30 no-manual same-mask route diagnostic: uses same-run ACT/SDSS/curl/variant statistics, no protocol fill file.
6. P30 fail-fast curl/variant tension classifier.
7. P33 no-manual alpha artifact normalizer; refuses to invent density-split BAO alpha from compressed non-density BAO.
8. PTA/P32/P40/P41 no-manual public/cache likelihood artifact normalizers.
9. SMD derivation gate refuses consistency→derivation promotion without no-target-fit auto/public metadata.
10. Dashboard v59 reports no-manual policy, quarantined artifacts, failed gates and confirm-recovery priorities.

Expected behavior: confirms do not increase unless public/cached data really support them.
