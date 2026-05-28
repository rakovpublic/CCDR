# v46 confirm robustness + next-target hardening

This patch is additive over v45. It preserves the frozen strict positives and adds stronger artifacts for the next confirmation targets.

## Implemented

1. T48b frozen-confirm robustness dashboard: absorber-family, certification-source, year-block, and descriptor-only robustness artifacts.
2. T44 Tier-A/source audit: stricter Tier A/B/C row classification, Tier-A recovery candidates, source-domain audit, leave-one-manufacturer artifact.
3. T53 final DMS/PDB/AlphaFold model scaffold: structure/contact proxy, bootstrap CI, family/assay/sequence jackknife artifact.
4. T31/T32 strict κ(T)+microstructure parser: improved unit parsing for temperature, kappa, grain size, material/source references; narrow grain/nano branch only.
5. T34 exact thermoelectric export mapping: Bi2Te3/Sb2Te3 + ZT + temperature + angle rows and cos(6θ) model gate.
6. T57/T59 exact HEPData registry hardening: v46 complete-row checks and official endpoint order.
7. T45/T47 exact benchmark parsers: optical energy/bit + bandwidth + reach; neuromorphic chip + energy/inference + accuracy.
8. Fusion exact-attachment-only policy for T26–T30: metadata/search/schema/PDF wrappers remain rejected.
9. T50–T52 bound-only preserved; T60a anchor-only with T60b/T60c/T60d readiness audit.

## Output schema

The dashboard now emits `ccdr-tierb-positive-dashboard-v46` and `v46_confirm_status`.

Frozen confirms remain T48b and T44 when their previous gates are present. No v46 code moves their gates; robustness-only artifacts are appended.
