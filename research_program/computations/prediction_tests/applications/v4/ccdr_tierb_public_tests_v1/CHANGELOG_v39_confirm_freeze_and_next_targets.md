# v39 Confirm Freeze and Next-Target Hardening

Implements the nine requested improvements from the v38 analysis:

1. Freezes T48b as `compatible_positive` only if the v38 final gate remains passed; adds a frozen audit artifact and robustness-only next checks.
2. Expands T44 NAND normalization from exact public tables and requires >=20 rows plus manufacturer jackknife before strict confirmation.
3. Normalizes T31/T32 microstructure-to-kappa rows with explicit `temperature_K` and `kappa_W_mK` fields before any exponent-model confirmation.
4. Adds a T53 PDB/RCSB/structure-aware symmetry/contact proxy layer and writes v39 model rows.
5. Improves T34 header recovery into explicit Bi2Te3/Sb2Te3 thermoelectric orientation/ZT mapping rows.
6. Keeps fusion lower-priority and rejects OSF/Zenodo metadata wrappers as physical evidence.
7. Adds exact HEPData registry scaffold for T57/T59 and preserves official YAML/JSON fallback paths.
8. Narrows T45/T47 to exact benchmark tables with required units/fields.
9. Extends T60 full-null-suite scaffolding while preserving T60a as anchor-only.

The v39 dashboard separates `confirmed_now`, `near_confirm_next`, `positive_anchor_only`, and `bound_only`.
