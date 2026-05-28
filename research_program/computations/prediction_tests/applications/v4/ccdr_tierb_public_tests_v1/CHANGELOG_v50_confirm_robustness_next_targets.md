# v50 confirm robustness and next-target hardening

Implements the 10 requested improvements after the v49 analysis:

1. T31 adaptive strict grain/nano kappa(T)+microstructure model using both grain-size and boundary-density parameterizations.
2. T32 same adaptive strict model; broad T^0.5 remains pressure/control.
3. T48b frozen compatible-positive robustness dashboard only; no gate changes.
4. T44 Tier-A/source audit with recoverable Tier-A candidates and source-domain/leave-one-manufacturer artifacts.
5. T53 final DMS/PDB/AlphaFold structure-contact model scaffolding and gate manifest.
6. T34 exact Bi2Te3/Sb2Te3 thermoelectric parser with cos(6theta) model artifact.
7. T57/T59 exact HEPData registry hardening.
8. T45/T47 exact benchmark parser hardening.
9. Fusion exact-attachment-only policy and timeout/missing-output surfacing.
10. T50-T52 bound-only preservation and T60 full null-suite readiness audit.

The v50 dashboard preserves frozen positives T48b and T44 and uses `third_confirm_priority` for T31/T32 when their strict adaptive model is closest to a third confirmation.
