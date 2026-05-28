# v38 final confirm-gate hardening

Implements the nine requested improvements after the v37 report:

1. T48b final gate consistency and promote-if-pass logic.
2. T44 NAND normalized-row expansion and stricter layer-vs-baseline model.
3. T53 symmetry/contact proxy model with family/assay jackknife.
4. T31/T32 microstructure-to-kappa join artifact and exponent gate.
5. T34 Starrydata/teMatDb-style header recovery and orientation/ZT parser.
6. Fusion metadata pre-filter: OSF/Zenodo wrappers are rejected before physical scoring.
7. T57/T59 exact HEPData registry resolver with official YAML/JSON/table fallbacks.
8. T45/T47 exact benchmark narrowing.
9. T60b/T60c/T60d null-suite scaffolding while keeping T60a as anchor only.

No manual CSV filling is required. Generated CSVs are cache/audit artifacts only.
