# v40 confirm-preserve and next-target hardening

Implemented the 9 requested improvements from the v39 report:

1. Preserves T48b as the frozen compatible-positive and adds robustness-only artifacts.
2. Adds T44 Tier A/B/C NAND row-quality filtering and tiered strict model gate.
3. Adds T31/T32 temperature_K/kappa_W_mK/grain-size normalization for the narrow grain/nano branch.
4. Adds T53 real-structure proxy fields for symmetry/contact/oligomeric-state hardening.
5. Adds T34 thermoelectric header/row mapping into Bi2Te3/Sb2Te3 orientation/ZT rows.
6. Keeps fusion exact-file-only: metadata/PDF wrappers remain non-evidence.
7. Adds per-test exact HEPData registry artifacts for T57/T59.
8. Adds exact benchmark parsers for T45 optical and T47 neuromorphic rows.
9. Keeps T60 anchor-only while emitting a full-null-suite status artifact.

Dashboard schema is upgraded to `ccdr-tierb-positive-dashboard-v40` with `v40_confirm_status`.
