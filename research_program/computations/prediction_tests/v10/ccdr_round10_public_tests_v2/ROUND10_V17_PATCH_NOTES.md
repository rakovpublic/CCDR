# Round-10 v17: all 10 positive-ready to confirmation improvements

Generated: 2026-05-06T18:18:54Z

Implemented:
1. P30 true field/region jackknife using Euclid field-like columns when available, otherwise RA/Dec regions.
2. P30 ACT finite-footprint/mask-aware statistic retained; explicit ACT mask remains optional/heavy.
3. P30 Euclid sample increased to TOP 20000 by default.
4. P30 ACT variant replication: f090, f150, tonly, cibdeproj, and curl control when --allow-large.
5. CL2 NANOGrav cached/static fallback classification; metadata failure becomes data_availability_blocker, not execution failure.
6. P3 limited exact endpoint-table query and endpoint-shuffle orientation proxy.
7. P41 supplementary structured table guard; rows + sign basis required before promotion.
8. High-z a0 quality-gated Vrot result: v/r/z column checks and robust-suggestive only when quality criteria exist.
9. Direct-detection HEPData CSV mass-window coverage confirmation parser.
10. Dashboard blocker classes: execution_blocker vs data_availability_blocker; suite_status_v17.
