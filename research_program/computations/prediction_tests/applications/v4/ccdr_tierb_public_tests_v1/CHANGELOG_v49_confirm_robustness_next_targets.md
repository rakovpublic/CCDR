# v49 confirm robustness and next-target hardening

Implements the next 10 improvements over v48:

1. T31 strict measured grain/nano kappa(T)+microstructure model comparison.
2. T32 strict measured grain/nano kappa(T)+microstructure model comparison.
3. T48b frozen-compatible-positive robustness audit only.
4. T44 frozen strict-confirmed Tier-A/source-domain audit.
5. T53 final ProteinGym DMS + UniProt/PDB/AlphaFold proxy gate scaffold.
6. T34 exact Bi2Te3/Sb2Te3 thermoelectric angle/ZT mapping and cos(6theta) model artifact.
7. T57/T59 exact HEPData registry completeness checks.
8. T45/T47 exact benchmark parser artifacts.
9. Fusion T26-T30 exact-attachment-only policy with timeout/fallback surfacing.
10. T50-T52 bound-only preservation and T60 full-null-suite audit.

The v49 dashboard preserves frozen positives T48b and T44. It adds `third_confirm_priority`, currently intended for T31/T32 when their strict model/jackknife gates are closest to a third confirmation.
