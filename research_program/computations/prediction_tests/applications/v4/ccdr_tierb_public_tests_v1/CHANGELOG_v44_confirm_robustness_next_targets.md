# v44 confirm robustness and next-target hardening

Implements the next 9 improvements requested after the v43 analysis:

1. Preserves T48b/T44 as frozen strict positives and adds robustness-only dashboards.
2. Adds T48b absorber-family, certification-source, year-block, and descriptor-only audit artifacts.
3. Adds T44 Tier A/B/C row-quality audit, source-domain audit, and leave-one-manufacturer checks.
4. Hardens T53 DMS/PDB/AlphaFold structure-contact model with bootstrap coefficient audit.
5. Hardens T31/T32 strict grain/nano κ(T)+microstructure normalization and fixed-exponent model.
6. Hardens T34 Bi2Te3/Sb2Te3 thermoelectric orientation/ZT mapping and cos(6θ) model scaffold.
7. Hardens T57/T59 HEPData exact record/table/column registry artifacts.
8. Hardens T45/T47 exact public benchmark row parsers.
9. Keeps fusion exact-attachment-only and T60 anchor-only/null-suite policies.

Generated artifacts are written under `data/generated/*_v44.csv`. T48b/T44 gates are preserved rather than moved.
