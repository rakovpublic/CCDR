# v45 confirm robustness + next target hardening

Implemented the nine next improvements from the v44 report:

1. Preserve T48b and T44 frozen positives; add robustness-only v45 dashboards.
2. T44 Tier-A recovery candidate artifact and leave-one-manufacturer audit.
3. T53 final DMS/PDB/AlphaFold proxy model with bootstrap coefficient CI scaffold.
4. T31/T32 stricter temperature/kappa/grain-size unit normalization and fixed-exponent model.
5. T34 exact thermoelectric mapping with orientation/grain-boundary angle and cos(6θ) scaffold.
6. T57/T59 exact HEPData registry with official fetch-order endpoint candidates.
7. T45/T47 exact benchmark row parsers with unit conversions.
8. Fusion T26-T30 exact-attachment-only low-priority policy artifacts.
9. T50-T52 bound-only preservation and T60 full-null-suite readiness audit.

Generated artifacts are written under `data/generated/*_v45.csv`. The dashboard emits
`ccdr-tierb-positive-dashboard-v45` and `v45_confirm_status`.
